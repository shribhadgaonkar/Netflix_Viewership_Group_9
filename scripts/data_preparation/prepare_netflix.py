from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.pipeline_utils import (  # noqa: E402
    coerce_nullable_int,
    detect_netflix_raw_file,
    ensure_parent,
    infer_netflix_format,
    interim_dir,
    log,
    parse_netflix_title,
    parse_numeric_series,
    parse_runtime_minutes,
    pick_first_existing_column,
    raw_dir,
    standardize_columns,
    to_snake_case,
)


OUTPUT_PATH = interim_dir() / "netflix_cleaned.csv"


def read_netflix_file(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path, low_memory=False)
    if suffix in {".xlsx", ".xls", ".xlsm"}:
        return pd.read_excel(path)
    raise ValueError(f"Unsupported Netflix input type: {path.name}")


def add_numeric_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    for column in frame.columns:
        lowered = column.lower()
        if any(token in lowered for token in ("hours", "views", "rank")):
            frame[column] = parse_numeric_series(frame[column])

    frame["netflix_runtime"] = frame["netflix_runtime_raw"].map(parse_runtime_minutes)

    canonical_hours = pick_first_existing_column(
        frame,
        ["total_hours_viewed", "hours_viewed", "weekly_hours_viewed"],
    )
    canonical_views = pick_first_existing_column(
        frame,
        ["total_views", "views", "weekly_views"],
    )

    frame["netflix_hours_viewed"] = frame[canonical_hours] if canonical_hours else pd.NA
    frame["netflix_views"] = frame[canonical_views] if canonical_views else pd.NA
    return frame


def add_release_fields(frame: pd.DataFrame) -> pd.DataFrame:
    release_date_column = pick_first_existing_column(
        frame,
        ["release_date", "premiere_date", "date_added", "availability_date"],
    )

    if release_date_column:
        release_dates = pd.to_datetime(frame[release_date_column], errors="coerce")
        frame["netflix_release_date"] = release_dates.dt.strftime("%Y-%m-%d")
        frame["netflix_release_year"] = release_dates.dt.year.astype("Int64")
    else:
        frame["netflix_release_date"] = pd.NA
        frame["netflix_release_year"] = pd.Series(pd.NA, index=frame.index, dtype="Int64")

    available_globally_column = pick_first_existing_column(
        frame,
        ["available_globally", "globally_available", "global", "worldwide"],
    )
    frame["netflix_available_globally"] = (
        frame[available_globally_column] if available_globally_column else pd.NA
    )
    return frame


def prepare_netflix() -> pd.DataFrame:
    netflix_raw_dir = raw_dir() / "netflix"
    input_path = detect_netflix_raw_file(netflix_raw_dir)
    log(f"Reading Netflix raw file: {input_path.relative_to(REPO_ROOT).as_posix()}")

    frame = read_netflix_file(input_path)
    log(f"Netflix rows read: {len(frame):,}")
    log(f"Netflix columns read: {len(frame.columns):,}")
    log(f"Netflix source columns: {', '.join(frame.columns)}")

    raw_columns = list(frame.columns)
    frame.columns = standardize_columns(raw_columns)

    title_column = pick_first_existing_column(frame, ["title_name", "title"])
    type_column = pick_first_existing_column(frame, ["type", "format"])
    runtime_column = pick_first_existing_column(frame, ["runtime", "duration"])

    if title_column is None:
        raise KeyError("Netflix raw file is missing a title column such as 'Title Name'.")

    frame.insert(0, "netflix_row_id", range(1, len(frame) + 1))
    frame["netflix_title_raw"] = frame[title_column].astype("string")
    frame["netflix_runtime_raw"] = frame[runtime_column].astype("string") if runtime_column else pd.NA
    frame["source_netflix_file"] = input_path.relative_to(REPO_ROOT).as_posix()

    parsed = frame["netflix_title_raw"].map(parse_netflix_title).apply(pd.Series)
    frame["netflix_series_title"] = parsed["series_title"]
    frame["netflix_normalized_title"] = parsed["normalized_title"]
    frame["netflix_season_number"] = coerce_nullable_int(parsed["season_number"])
    frame["netflix_season_label"] = parsed["season_label"].astype("string")
    frame["netflix_title_year_hint"] = coerce_nullable_int(parsed["title_year_hint"])

    raw_type = frame[type_column] if type_column else pd.Series(pd.NA, index=frame.index)
    frame["netflix_format"] = [
        infer_netflix_format(type_value, season_label)
        for type_value, season_label in zip(raw_type, frame["netflix_season_label"], strict=False)
    ]

    frame = add_release_fields(frame)
    frame = add_numeric_metrics(frame)

    if type_column and type_column != "netflix_format":
        frame.rename(columns={type_column: "raw_netflix_type"}, inplace=True)
    if runtime_column and runtime_column != "netflix_runtime_raw":
        frame.rename(columns={runtime_column: "runtime_raw_source"}, inplace=True)

    ensure_parent(OUTPUT_PATH)
    frame.to_csv(OUTPUT_PATH, index=False)

    series_rows = int((frame["netflix_format"] == "series").sum())
    season_rows = int(frame["netflix_season_number"].notna().sum())
    log(f"Netflix series-like rows: {series_rows:,}")
    log(f"Netflix rows with extracted seasons: {season_rows:,}")
    log(f"Saved cleaned Netflix data: {OUTPUT_PATH.relative_to(REPO_ROOT).as_posix()}")
    return frame


def main() -> None:
    prepare_netflix()


if __name__ == "__main__":
    main()
