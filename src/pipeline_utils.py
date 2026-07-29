from __future__ import annotations

import re
import unicodedata
from pathlib import Path
from typing import Any

import pandas as pd


NETFLIX_SUFFIXES = {".csv", ".xlsx", ".xls", ".xlsm"}
IMDB_REQUIRED_FILES = (
    "title.basics.tsv.gz",
    "title.episode.tsv.gz",
    "title.ratings.tsv.gz",
)
SERIES_TITLE_TYPES = {"tvSeries", "tvMiniSeries"}
SEASON_PATTERNS = [
    ("limited_series", re.compile(r"^(?P<base>.+?)\s*:\s*limited series\s*$", re.IGNORECASE)),
    ("season", re.compile(r"^(?P<base>.+?)\s*:\s*season\s*(?P<number>\d+)\s*$", re.IGNORECASE)),
    ("series", re.compile(r"^(?P<base>.+?)\s*:\s*series\s*(?P<number>\d+)\s*$", re.IGNORECASE)),
    ("part", re.compile(r"^(?P<base>.+?)\s*:\s*part\s*(?P<number>\d+)\s*$", re.IGNORECASE)),
    ("volume", re.compile(r"^(?P<base>.+?)\s*:\s*volume\s*(?P<number>\d+)\s*$", re.IGNORECASE)),
    ("chapter", re.compile(r"^(?P<base>.+?)\s*:\s*chapter\s*(?P<number>\d+)\s*$", re.IGNORECASE)),
    ("book", re.compile(r"^(?P<base>.+?)\s*:\s*book\s*(?P<number>\d+)\s*$", re.IGNORECASE)),
    ("class", re.compile(r"^(?P<base>.+?)\s*:\s*class\s*(?P<number>\d+)\s*$", re.IGNORECASE)),
    ("collection", re.compile(r"^(?P<base>.+?)\s*:\s*collection\s*(?P<number>\d+)\s*$", re.IGNORECASE)),
    (
        "temporada",
        re.compile(r"^(?P<base>.+?)\s*:\s*(?P<number>\d+)\s*[ªa]?\s*temporada\s*$", re.IGNORECASE),
    ),
]
YEAR_HINT_PATTERN = re.compile(r"\((?P<year>(?:19|20)\d{2})\)\s*$")


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def data_dir() -> Path:
    return repo_root() / "data"


def raw_dir() -> Path:
    return data_dir() / "raw"


def interim_dir() -> Path:
    return data_dir() / "interim"


def processed_dir() -> Path:
    return data_dir() / "processed"


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def log(message: str) -> None:
    print(f"[pipeline] {message}")


def to_snake_case(value: str) -> str:
    text = value.strip()
    text = re.sub(r"[%/]", " ", text)
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"[-\s]+", "_", text)
    text = re.sub(r"_+", "_", text)
    return text.strip("_").lower()


def standardize_columns(columns: list[str]) -> list[str]:
    seen: dict[str, int] = {}
    standardized: list[str] = []

    for column in columns:
        base = to_snake_case(column) or "column"
        if base not in seen:
            seen[base] = 0
            standardized.append(base)
            continue

        seen[base] += 1
        standardized.append(f"{base}_{seen[base]}")

    return standardized


def detect_netflix_raw_file(netflix_raw_dir: Path) -> Path:
    candidates = [
        path
        for path in netflix_raw_dir.iterdir()
        if path.is_file() and path.suffix.lower() in NETFLIX_SUFFIXES
    ]

    if not candidates:
        raise FileNotFoundError(
            f"No supported Netflix raw file found in {netflix_raw_dir}. "
            "Expected CSV or Excel input."
        )

    candidates.sort(key=lambda path: (path.stat().st_size, path.name.lower()), reverse=True)
    chosen = candidates[0]

    if len(candidates) > 1:
        log(
            "Multiple Netflix raw files found; using the largest file: "
            f"{chosen.name}. Candidates: {', '.join(path.name for path in candidates)}"
        )

    return chosen


def require_imdb_inputs(imdb_raw_dir: Path) -> dict[str, Path]:
    resolved: dict[str, Path] = {}
    missing: list[str] = []

    for file_name in IMDB_REQUIRED_FILES:
        path = imdb_raw_dir / file_name
        if path.exists():
            resolved[file_name] = path
        else:
            missing.append(file_name)

    if missing:
        raise FileNotFoundError(
            "Missing required IMDb raw files in "
            f"{imdb_raw_dir}: {', '.join(missing)}"
        )

    return resolved


def parse_numeric_series(series: pd.Series) -> pd.Series:
    text = series.astype("string").str.strip()
    cleaned = (
        text.str.replace(",", "", regex=False)
        .str.replace("%", "", regex=False)
        .str.replace(r"^\-$", "", regex=True)
        .str.replace(r"^$", "", regex=True)
    )
    return pd.to_numeric(cleaned, errors="coerce")


def parse_runtime_minutes(value: Any) -> float | None:
    if value is None or pd.isna(value):
        return None

    text = str(value).strip()
    if not text or text == "-":
        return None

    if ":" in text:
        parts = text.split(":")
        if len(parts) == 2 and all(part.isdigit() for part in parts):
            hours, minutes = parts
            return float(int(hours) * 60 + int(minutes))
        return None

    numeric = re.sub(r"[^\d.]", "", text)
    if not numeric:
        return None

    try:
        return float(numeric)
    except ValueError:
        return None


def normalize_title(text: Any) -> str | None:
    if text is None or pd.isna(text):
        return None

    value = str(text).strip()
    if not value:
        return None

    value = value.replace("’", "'").replace("‘", "'").replace("`", "'")
    value = value.replace("–", "-").replace("—", "-").replace("‐", "-")
    value = YEAR_HINT_PATTERN.sub("", value).strip()
    value = unicodedata.normalize("NFKD", value)
    value = "".join(character for character in value if not unicodedata.combining(character))
    value = value.lower()
    value = value.replace("&", " and ")
    value = re.sub(r"[^\w\s]", " ", value)
    value = re.sub(r"_", " ", value)
    value = re.sub(r"\s+", " ", value).strip()
    return value or None


def extract_year_hint(text: Any) -> int | None:
    if text is None or pd.isna(text):
        return None

    match = YEAR_HINT_PATTERN.search(str(text).strip())
    if not match:
        return None
    return int(match.group("year"))


def parse_netflix_title(title: Any) -> dict[str, Any]:
    if title is None or pd.isna(title):
        return {
            "series_title": None,
            "normalized_title": None,
            "season_number": pd.NA,
            "season_label": None,
            "title_year_hint": pd.NA,
        }

    raw_title = str(title).strip()
    parsed_title = raw_title
    season_number: int | None = None
    season_label: str | None = None

    for label, pattern in SEASON_PATTERNS:
        match = pattern.match(raw_title)
        if not match:
            continue

        parsed_title = match.group("base").strip()
        season_label = label
        if label == "limited_series":
            season_number = 1
        else:
            season_number = int(match.group("number"))
        break

    title_year_hint = extract_year_hint(parsed_title)
    normalized_title = normalize_title(parsed_title)

    return {
        "series_title": parsed_title,
        "normalized_title": normalized_title,
        "season_number": season_number if season_number is not None else pd.NA,
        "season_label": season_label,
        "title_year_hint": title_year_hint if title_year_hint is not None else pd.NA,
    }


def infer_netflix_format(raw_type: Any, season_label: Any) -> str:
    text = "" if raw_type is None or pd.isna(raw_type) else str(raw_type).strip().lower()

    if text == "movie":
        return "movie"
    if text in {"tv", "series", "show"}:
        return "series"
    if season_label:
        return "series"
    return "unknown"


def pick_first_existing_column(frame: pd.DataFrame, candidates: list[str]) -> str | None:
    for candidate in candidates:
        if candidate in frame.columns:
            return candidate
    return None


def coerce_nullable_int(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").round().astype("Int64")


def year_consistency(reference_year: Any, start_year: Any, end_year: Any) -> bool | pd.NA:
    if reference_year is None or pd.isna(reference_year):
        return pd.NA

    ref = int(reference_year)
    start = None if start_year is None or pd.isna(start_year) else int(start_year)
    end = None if end_year is None or pd.isna(end_year) else int(end_year)

    if start is None and end is None:
        return pd.NA
    if start is not None and ref < start:
        return False
    if end is not None and ref > end:
        return False
    return True
