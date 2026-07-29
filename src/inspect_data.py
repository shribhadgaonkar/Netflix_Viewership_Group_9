from __future__ import annotations

import argparse
import heapq
import re
import shutil
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


SUPPORTED_SUFFIXES = {
    ".csv",
    ".tsv",
    ".tsv.gz",
    ".parquet",
    ".xls",
    ".xlsx",
    ".xlsm",
}
DELIMITED_SUFFIXES = {".csv", ".tsv", ".tsv.gz"}
EXCEL_SUFFIXES = {".xls", ".xlsx", ".xlsm"}
IDENTIFIER_NAME_RE = re.compile(
    r"(^id$|^id_|id$|_id$|identifier|uuid|guid|code$|_code$|key$|_key$|const$)",
    re.IGNORECASE,
)
SMALL_FILE_BYTES = 100 * 1024 * 1024
CHUNK_SIZE = 200_000
HASH_BUFFER_SIZE = 5_000_000
NA_VALUES = ["\\N"]


@dataclass
class DatasetReport:
    file_name: str
    relative_path: str
    file_size_bytes: int
    rows: int
    columns: int
    column_names: list[str]
    dtypes: dict[str, str]
    missing_pct: dict[str, float]
    duplicate_rows: int
    candidate_identifiers: list[str]
    sheet_name: str | None = None
    notes: list[str] = field(default_factory=list)


class UInt64RunAggregator:
    def __init__(self, temp_dir: Path, stem: str, buffer_size: int = HASH_BUFFER_SIZE) -> None:
        self.temp_dir = temp_dir
        self.stem = stem
        self.buffer_size = buffer_size
        self.buffers: list[np.ndarray] = []
        self.buffer_count = 0
        self.total_count = 0
        self.run_files: list[Path] = []

    def add(self, values: np.ndarray) -> None:
        if values.size == 0:
            return
        array = np.asarray(values, dtype=np.uint64)
        self.buffers.append(array.copy())
        self.buffer_count += int(array.size)
        self.total_count += int(array.size)
        if self.buffer_count >= self.buffer_size:
            self.flush()

    def flush(self) -> None:
        if self.buffer_count == 0:
            return
        merged = np.concatenate(self.buffers)
        merged.sort()
        run_path = self.temp_dir / f"{self.stem}_{len(self.run_files):04d}.bin"
        with run_path.open("wb") as handle:
            merged.tofile(handle)
        self.run_files.append(run_path)
        self.buffers = []
        self.buffer_count = 0

    def unique_count(self) -> int:
        self.flush()
        if self.total_count == 0:
            return 0
        if len(self.run_files) == 1:
            return self._unique_count_single_run(self.run_files[0])
        return self._unique_count_multi_run()

    def duplicates_count(self) -> int:
        return self.total_count - self.unique_count()

    def cleanup(self) -> None:
        for run_file in self.run_files:
            if run_file.exists():
                run_file.unlink()
        self.run_files = []

    @staticmethod
    def _unique_count_single_run(path: Path) -> int:
        values = np.memmap(path, dtype=np.uint64, mode="r")
        if values.size == 0:
            del values
            return 0
        unique_values = int(np.count_nonzero(values[1:] != values[:-1]) + 1)
        del values
        return unique_values

    def _unique_count_multi_run(self) -> int:
        arrays = [np.memmap(path, dtype=np.uint64, mode="r") for path in self.run_files]
        indices = [0] * len(arrays)
        heap: list[tuple[int, int]] = []

        for idx, array in enumerate(arrays):
            if array.size:
                heapq.heappush(heap, (int(array[0]), idx))

        unique_values = 0
        previous: int | None = None

        while heap:
            value, source_idx = heapq.heappop(heap)
            if previous != value:
                unique_values += 1
                previous = value

            indices[source_idx] += 1
            next_idx = indices[source_idx]
            if next_idx < arrays[source_idx].size:
                heapq.heappush(heap, (int(arrays[source_idx][next_idx]), source_idx))

        del arrays
        return unique_values


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def supported_files(raw_dir: Path) -> list[Path]:
    files: list[Path] = []
    for path in raw_dir.rglob("*"):
        if not path.is_file() or path.name == ".gitkeep":
            continue
        if file_suffix(path) in SUPPORTED_SUFFIXES:
            files.append(path)
    return sorted(files)


def file_suffix(path: Path) -> str:
    name = path.name.lower()
    if name.endswith(".tsv.gz"):
        return ".tsv.gz"
    return path.suffix.lower()


def human_size(size_bytes: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    size = float(size_bytes)
    for unit in units:
        if size < 1024 or unit == units[-1]:
            return f"{size:.2f} {unit}"
        size /= 1024
    return f"{size_bytes} B"


def dtype_label(series: pd.Series) -> str:
    from pandas.api.types import (
        is_bool_dtype,
        is_datetime64_any_dtype,
        is_float_dtype,
        is_integer_dtype,
        is_timedelta64_dtype,
        infer_dtype,
    )

    if is_bool_dtype(series.dtype):
        return "boolean"
    if is_integer_dtype(series.dtype):
        return "integer"
    if is_float_dtype(series.dtype):
        return "float"
    if is_datetime64_any_dtype(series.dtype):
        return "datetime"
    if is_timedelta64_dtype(series.dtype):
        return "timedelta"

    inferred = infer_dtype(series.dropna(), skipna=True)
    if inferred in {"string", "unicode", "bytes", "categorical"}:
        return "string"
    if inferred in {"mixed-integer", "mixed-integer-float", "floating", "decimal"}:
        return "mixed numeric"
    if inferred.startswith("datetime"):
        return "datetime"
    if inferred == "boolean":
        return "boolean"
    if inferred == "integer":
        return "integer"
    if inferred == "empty":
        return "all-null"
    return "mixed/object"


def resolve_dtype_labels(labels: set[str]) -> str:
    clean = {label for label in labels if label != "all-null"}
    if not clean:
        return "all-null"
    if "mixed/object" in clean:
        return "mixed/object"
    if "string" in clean and len(clean) > 1:
        return "mixed/object"
    if "mixed numeric" in clean:
        return "mixed numeric"
    if "float" in clean and "integer" in clean:
        return "float"
    if len(clean) == 1:
        return next(iter(clean))
    return "mixed/object"


def row_hashes(frame: pd.DataFrame) -> np.ndarray:
    hashed = pd.util.hash_pandas_object(frame, index=False)
    return hashed.to_numpy(dtype=np.uint64, copy=False)


def series_hashes(series: pd.Series) -> np.ndarray:
    hashed = pd.util.hash_pandas_object(series, index=False)
    return hashed.to_numpy(dtype=np.uint64, copy=False)


def choose_identifier_columns(path: Path, columns: list[str]) -> list[str]:
    strong_name_matches = identifier_name_matches(columns)
    if strong_name_matches:
        return strong_name_matches
    if path.stat().st_size <= SMALL_FILE_BYTES and len(columns) <= 25:
        return columns
    return []


def identifier_name_matches(columns: list[str]) -> list[str]:
    return [column for column in columns if IDENTIFIER_NAME_RE.search(column)]


def inspect_delimited(path: Path, relative_path: str, temp_dir: Path) -> list[DatasetReport]:
    separator = "\t" if file_suffix(path) in {".tsv", ".tsv.gz"} else ","
    header = pd.read_csv(path, sep=separator, nrows=0, na_values=NA_VALUES, keep_default_na=True)
    columns = header.columns.tolist()
    missing_counts = pd.Series(0, index=columns, dtype="int64")
    dtype_observations: dict[str, set[str]] = defaultdict(set)
    named_id_columns = identifier_name_matches(columns)
    id_columns = choose_identifier_columns(path, columns)

    duplicate_tracker = UInt64RunAggregator(temp_dir, stem=f"{path.stem}_rows")
    identifier_trackers = {
        column: UInt64RunAggregator(temp_dir, stem=f"{path.stem}_{sanitize_name(column)}")
        for column in id_columns
    }
    identifier_non_null_counts = {column: 0 for column in id_columns}

    row_count = 0

    try:
        reader = pd.read_csv(
            path,
            sep=separator,
            chunksize=CHUNK_SIZE,
            na_values=NA_VALUES,
            keep_default_na=True,
            low_memory=False,
        )

        for chunk in reader:
            row_count += len(chunk)
            missing_counts = missing_counts.add(chunk.isna().sum().astype("int64"), fill_value=0)

            for column in columns:
                dtype_observations[column].add(dtype_label(chunk[column]))

            duplicate_tracker.add(row_hashes(chunk))

            for column, tracker in identifier_trackers.items():
                non_null = chunk[column].dropna()
                identifier_non_null_counts[column] += len(non_null)
                tracker.add(series_hashes(non_null))

        duplicate_rows = duplicate_tracker.duplicates_count()
        candidate_identifiers = []

        for column, tracker in identifier_trackers.items():
            unique_values = tracker.unique_count()
            non_null_values = identifier_non_null_counts[column]
            if non_null_values == row_count and unique_values == non_null_values:
                candidate_identifiers.append(column)

        report = DatasetReport(
            file_name=path.name,
            relative_path=relative_path,
            file_size_bytes=path.stat().st_size,
            rows=row_count,
            columns=len(columns),
            column_names=columns,
            dtypes={column: resolve_dtype_labels(dtype_observations[column]) for column in columns},
            missing_pct={
                column: (float(missing_counts[column]) / row_count * 100.0) if row_count else 0.0
                for column in columns
            },
            duplicate_rows=duplicate_rows,
            candidate_identifiers=candidate_identifiers,
        )

        if not named_id_columns:
            report.notes.append("No obvious identifier-like column names were detected.")
        elif not candidate_identifiers:
            report.notes.append(
                "Identifier-like columns were present, but none were fully non-null and unique."
            )

        return [report]
    finally:
        duplicate_tracker.cleanup()
        for tracker in identifier_trackers.values():
            tracker.cleanup()


def inspect_parquet(path: Path, relative_path: str, temp_dir: Path) -> list[DatasetReport]:
    parquet_file = pq.ParquetFile(path)
    columns = parquet_file.schema.names
    missing_counts = pd.Series(0, index=columns, dtype="int64")
    dtype_observations: dict[str, set[str]] = defaultdict(set)
    named_id_columns = identifier_name_matches(columns)
    id_columns = choose_identifier_columns(path, columns)

    duplicate_tracker = UInt64RunAggregator(temp_dir, stem=f"{path.stem}_rows")
    identifier_trackers = {
        column: UInt64RunAggregator(temp_dir, stem=f"{path.stem}_{sanitize_name(column)}")
        for column in id_columns
    }
    identifier_non_null_counts = {column: 0 for column in id_columns}

    row_count = 0

    try:
        for batch in parquet_file.iter_batches(batch_size=CHUNK_SIZE):
            chunk = batch.to_pandas()
            row_count += len(chunk)
            missing_counts = missing_counts.add(chunk.isna().sum().astype("int64"), fill_value=0)

            for column in columns:
                dtype_observations[column].add(dtype_label(chunk[column]))

            duplicate_tracker.add(row_hashes(chunk))

            for column, tracker in identifier_trackers.items():
                non_null = chunk[column].dropna()
                identifier_non_null_counts[column] += len(non_null)
                tracker.add(series_hashes(non_null))

        duplicate_rows = duplicate_tracker.duplicates_count()
        candidate_identifiers = []

        for column, tracker in identifier_trackers.items():
            unique_values = tracker.unique_count()
            non_null_values = identifier_non_null_counts[column]
            if non_null_values == row_count and unique_values == non_null_values:
                candidate_identifiers.append(column)

        report = DatasetReport(
            file_name=path.name,
            relative_path=relative_path,
            file_size_bytes=path.stat().st_size,
            rows=row_count,
            columns=len(columns),
            column_names=columns,
            dtypes={column: resolve_dtype_labels(dtype_observations[column]) for column in columns},
            missing_pct={
                column: (float(missing_counts[column]) / row_count * 100.0) if row_count else 0.0
                for column in columns
            },
            duplicate_rows=duplicate_rows,
            candidate_identifiers=candidate_identifiers,
        )

        if not named_id_columns:
            report.notes.append("No obvious identifier-like column names were detected.")
        elif not candidate_identifiers:
            report.notes.append(
                "Identifier-like columns were present, but none were fully non-null and unique."
            )

        return [report]
    finally:
        duplicate_tracker.cleanup()
        for tracker in identifier_trackers.values():
            tracker.cleanup()


def inspect_excel(path: Path, relative_path: str) -> list[DatasetReport]:
    workbook = pd.ExcelFile(path)
    reports: list[DatasetReport] = []

    for sheet_name in workbook.sheet_names:
        frame = workbook.parse(sheet_name=sheet_name)
        columns = frame.columns.tolist()
        candidate_identifiers = []

        for column in choose_identifier_columns(path, columns):
            series = frame[column].dropna()
            if len(series) == len(frame) and series.nunique(dropna=True) == len(series):
                candidate_identifiers.append(column)

        report = DatasetReport(
            file_name=path.name,
            relative_path=relative_path,
            file_size_bytes=path.stat().st_size,
            rows=len(frame),
            columns=len(columns),
            column_names=columns,
            dtypes={column: dtype_label(frame[column]) for column in columns},
            missing_pct={
                column: (float(frame[column].isna().sum()) / len(frame) * 100.0) if len(frame) else 0.0
                for column in columns
            },
            duplicate_rows=int(frame.duplicated().sum()),
            candidate_identifiers=candidate_identifiers,
            sheet_name=sheet_name,
        )

        if not columns:
            report.notes.append("Sheet has no columns.")
        elif not candidate_identifiers:
            report.notes.append("No fully unique, non-null identifier candidates were detected.")

        reports.append(report)

    return reports


def inspect_file(path: Path, raw_dir: Path, temp_dir: Path) -> list[DatasetReport]:
    relative_path = path.relative_to(repo_root()).as_posix()
    suffix = file_suffix(path)

    if suffix in DELIMITED_SUFFIXES:
        return inspect_delimited(path, relative_path, temp_dir)
    if suffix == ".parquet":
        return inspect_parquet(path, relative_path, temp_dir)
    if suffix in EXCEL_SUFFIXES:
        return inspect_excel(path, relative_path)

    return []


def sanitize_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", value).strip("_").lower() or "column"


def render_markdown(reports: Iterable[DatasetReport], raw_dir: Path) -> str:
    report_list = list(reports)
    lines = [
        "# Generated Data Inventory",
        "",
        f"Scanned raw data directory: `{raw_dir.relative_to(repo_root()).as_posix()}`",
        "",
        f"Supported file types: {', '.join(sorted(SUPPORTED_SUFFIXES))}",
        "",
        f"Datasets inspected: {len(report_list)}",
        "",
    ]

    for report in report_list:
        header = f"## `{report.relative_path}`"
        if report.sheet_name:
            header += f" (sheet: `{report.sheet_name}`)"
        lines.extend(
            [
                header,
                "",
                f"- File name: `{report.file_name}`",
                f"- Relative path: `{report.relative_path}`",
                f"- File size: {human_size(report.file_size_bytes)}",
                f"- Rows: {report.rows:,}",
                f"- Columns: {report.columns:,}",
                f"- Duplicate rows: {report.duplicate_rows:,}",
                (
                    "- Candidate identifier columns: "
                    + (
                        ", ".join(f"`{column}`" for column in report.candidate_identifiers)
                        if report.candidate_identifiers
                        else "None"
                    )
                ),
                "",
                "### Columns",
                "",
                "| Column | Dtype | Missing % |",
                "| --- | --- | ---: |",
            ]
        )

        for column in report.column_names:
            lines.append(
                f"| `{escape_pipes(column)}` | `{report.dtypes.get(column, 'unknown')}` | {report.missing_pct.get(column, 0.0):.2f}% |"
            )

        if report.notes:
            lines.extend(["", "### Notes", ""])
            for note in report.notes:
                lines.append(f"- {note}")

        lines.append("")

    return "\n".join(lines).strip() + "\n"


def escape_pipes(value: str) -> str:
    return value.replace("|", "\\|")


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect raw tabular data files and generate a markdown inventory.")
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=repo_root() / "data" / "raw",
        help="Directory containing raw source files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=repo_root() / "docs" / "data_inventory_generated.md",
        help="Markdown output path.",
    )
    args = parser.parse_args()

    raw_dir = args.raw_dir.resolve()
    output_path = args.output.resolve()
    files = supported_files(raw_dir)

    reports: list[DatasetReport] = []
    temp_dir = repo_root() / ".codex_tmp_inspect_data"
    if temp_dir.exists():
        shutil.rmtree(temp_dir, ignore_errors=True)
    temp_dir.mkdir(parents=True, exist_ok=True)

    try:
        for file_path in files:
            reports.extend(inspect_file(file_path, raw_dir, temp_dir))
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

    output_path.write_text(render_markdown(reports, raw_dir), encoding="utf-8")


if __name__ == "__main__":
    main()
