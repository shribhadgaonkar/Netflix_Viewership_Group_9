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
IMDB_OPTIONAL_FILES = ("title.akas.tsv.gz",)
SERIES_TITLE_TYPES = {"tvSeries", "tvMiniSeries"}
MATCH_SOURCE_PRIORITY = {"primary": 1, "original": 2, "aka": 3}
ROMAN_NUMERAL_PATTERN = re.compile(r"^[IVXLCDM]+$", re.IGNORECASE)
YEAR_HINT_PATTERN = re.compile(r"\((?P<year>(?:19|20)\d{2})\)\s*$")
COUNTRY_HINT_PATTERN = re.compile(r"\((?P<label>u\.?s\.?|us|u\.?k\.?|uk)\)\s*$", re.IGNORECASE)
TITLE_PARSE_HINT_PATTERN = re.compile(
    r"(?i)\b(season|series|part|volume|chapter|book|collection|limited series|temporada)\b"
)
LIMITED_SERIES_PATTERN = re.compile(
    r"^(?P<base>.+?)(?:\s*[:\-|]\s*|\s+)limited series\s*$",
    re.IGNORECASE,
)
TEMPORADA_PATTERN = re.compile(
    r"^(?P<base>.+?)(?:\s*[:\-|]\s*|\s+)(?P<number>\d+|[ivxlcdm]+)\s*(?:a|o|ra|da|[a-z]{0,4})?\s*temporada\s*$",
    re.IGNORECASE,
)
GENERIC_SEASON_PATTERNS: list[tuple[str, re.Pattern[str], float]] = [
    (
        "season",
        re.compile(
            r"^(?P<base>.+?)(?:\s*[:\-|]\s*|\s+)season\s*(?P<number>\d+|[ivxlcdm]+)\s*$",
            re.IGNORECASE,
        ),
        1.0,
    ),
    (
        "series",
        re.compile(
            r"^(?P<base>.+?)(?:\s*[:\-|]\s*|\s+)series\s*(?P<number>\d+|[ivxlcdm]+)\s*$",
            re.IGNORECASE,
        ),
        0.98,
    ),
    (
        "s_shorthand",
        re.compile(
            r"^(?P<base>.+?)(?:\s*[:\-|]\s*|\s+)s(?P<number>\d+|[ivxlcdm]+)\s*$",
            re.IGNORECASE,
        ),
        0.96,
    ),
    (
        "part",
        re.compile(
            r"^(?P<base>.+?)(?:\s*[:\-|]\s*|\s+)part\s*(?P<number>\d+|[ivxlcdm]+)\s*$",
            re.IGNORECASE,
        ),
        0.93,
    ),
    (
        "volume",
        re.compile(
            r"^(?P<base>.+?)(?:\s*[:\-|]\s*|\s+)volume\s*(?P<number>\d+|[ivxlcdm]+)\s*$",
            re.IGNORECASE,
        ),
        0.91,
    ),
    (
        "chapter",
        re.compile(
            r"^(?P<base>.+?)(?:\s*[:\-|]\s*|\s+)chapter\s*(?P<number>\d+|[ivxlcdm]+)\s*$",
            re.IGNORECASE,
        ),
        0.91,
    ),
    (
        "book",
        re.compile(
            r"^(?P<base>.+?)(?:\s*[:\-|]\s*|\s+)book\s*(?P<number>\d+|[ivxlcdm]+)\s*$",
            re.IGNORECASE,
        ),
        0.9,
    ),
    (
        "collection",
        re.compile(
            r"^(?P<base>.+?)(?:\s*[:\-|]\s*|\s+)collection\s*(?P<number>\d+|[ivxlcdm]+)\s*$",
            re.IGNORECASE,
        ),
        0.88,
    ),
    (
        "class",
        re.compile(
            r"^(?P<base>.+?)(?:\s*[:\-|]\s*|\s+)class\s*(?P<number>\d+|[ivxlcdm]+)\s*$",
            re.IGNORECASE,
        ),
        0.87,
    ),
]
SECONDARY_TRAILING_SEGMENT_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    (
        "part_parenthetical",
        re.compile(
            r"^(?P<base>.+?)\s*\((?P<label>part)\s*(?P<number>\d+|[ivxlcdm]+)\)\s*$",
            re.IGNORECASE,
        ),
    ),
    (
        "part_suffix",
        re.compile(
            r"^(?P<base>.+?)(?:\s*[:\-|]\s*|\s+)(?P<label>part)\s*(?P<number>\d+|[ivxlcdm]+)\s*$",
            re.IGNORECASE,
        ),
    ),
    (
        "volume_parenthetical",
        re.compile(
            r"^(?P<base>.+?)\s*\((?P<label>volume)\s*(?P<number>\d+|[ivxlcdm]+)\)\s*$",
            re.IGNORECASE,
        ),
    ),
    (
        "chapter_parenthetical",
        re.compile(
            r"^(?P<base>.+?)\s*\((?P<label>chapter)\s*(?P<number>\d+|[ivxlcdm]+)\)\s*$",
            re.IGNORECASE,
        ),
    ),
]
TV_ONLY_BARE_NUMBER_PATTERN = re.compile(r"^(?P<base>.+?)\s+(?P<number>\d{1,2})\s*$")
TV_ONLY_ROMAN_SUFFIX_PATTERN = re.compile(r"^(?P<base>.+?)\s+(?P<number>[IVXLCDM]+)\s*$", re.IGNORECASE)
TV_ONLY_YEARLESS_SEASON_BLACKLIST = re.compile(
    r"(?i)\b(movie|film|holiday|special|bonus|videos?|lyric|concert|live|story|case|vs\.?|versus)\b"
)
EPISODIC_ONE_OFF_HINTS = re.compile(
    r"(?i)\b(movie|holiday|special|bonus|videos?|lyric|concert|live|story|case|vs\.?|versus|adventure)\b"
)


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


def optional_imdb_inputs(imdb_raw_dir: Path) -> dict[str, Path]:
    resolved: dict[str, Path] = {}
    for file_name in IMDB_OPTIONAL_FILES:
        path = imdb_raw_dir / file_name
        if path.exists():
            resolved[file_name] = path
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


def normalize_unicode_punctuation(text: str) -> str:
    value = text.replace("’", "'").replace("‘", "'").replace("`", "'")
    value = value.replace("–", "-").replace("—", "-").replace("‐", "-")
    value = value.replace("：", ":").replace("｜", "|")
    value = value.replace("ª", "a")
    value = value.replace("º", "o")
    value = value.replace("\u00A0", " ")
    return value


def collapse_dotted_initialisms(text: str) -> str:
    return re.sub(
        r"\b((?:[A-Za-z]\.){2,})(?=\s|$)",
        lambda match: match.group(1).replace(".", ""),
        text,
    )


def normalize_title(text: Any) -> str | None:
    if text is None or pd.isna(text):
        return None

    value = str(text).strip()
    if not value:
        return None

    value = normalize_unicode_punctuation(value)
    value = collapse_dotted_initialisms(value)
    value = YEAR_HINT_PATTERN.sub("", value).strip()
    value = unicodedata.normalize("NFKD", value)
    value = "".join(character for character in value if not unicodedata.combining(character))
    value = value.lower()
    value = value.replace("&", " and ")
    value = re.sub(r"[^\w\s]", " ", value)
    value = re.sub(r"_", " ", value)
    value = re.sub(r"\s+", " ", value).strip()
    return value or None


def canonicalize_title(text: Any) -> str | None:
    normalized = normalize_title(text)
    if normalized is None:
        return None

    canonical = normalized
    canonical = re.sub(r"\bthe series\b$", "", canonical).strip()
    canonical = re.sub(r"\b(us|uk)\b$", "", canonical).strip()
    canonical = re.sub(r"\s+", " ", canonical).strip()
    return canonical or None


def compact_title_key(text: Any) -> str | None:
    canonical = canonicalize_title(text)
    if canonical is None:
        return None
    compact = canonical.replace(" ", "")
    return compact or None


def extract_year_hint(text: Any) -> int | None:
    if text is None or pd.isna(text):
        return None

    match = YEAR_HINT_PATTERN.search(str(text).strip())
    if not match:
        return None
    return int(match.group("year"))


def parse_roman_numeral(value: str) -> int | None:
    if not value or not ROMAN_NUMERAL_PATTERN.match(value):
        return None

    values = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}
    total = 0
    previous = 0
    for character in value.upper()[::-1]:
        current = values[character]
        if current < previous:
            total -= current
        else:
            total += current
            previous = current
    return total


def parse_season_token(value: str | None) -> int | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    if text.isdigit():
        return int(text)
    return parse_roman_numeral(text)


def parse_notes_text(notes: list[str]) -> str | None:
    if not notes:
        return None
    return "; ".join(dict.fromkeys(note for note in notes if note))


def is_tv_type(raw_type: Any) -> bool:
    if raw_type is None or pd.isna(raw_type):
        return False
    return str(raw_type).strip().lower() in {"tv", "series", "show"}


def strip_secondary_segments(title: str) -> tuple[str, list[str]]:
    working = title.strip()
    notes: list[str] = []

    while True:
        changed = False
        for label, pattern in SECONDARY_TRAILING_SEGMENT_PATTERNS:
            match = pattern.match(working)
            if not match:
                continue

            number = parse_season_token(match.group("number"))
            working = match.group("base").strip(" -:|")
            notes.append(f"removed trailing {match.group('label').lower()} {number} before season parsing")
            changed = True
            break

        if not changed:
            break

    return working, notes


def should_parse_trailing_number_as_season(base_title: str, number: int, raw_type: Any) -> bool:
    if not is_tv_type(raw_type):
        return False
    if number < 2 or number > 20:
        return False
    if TV_ONLY_YEARLESS_SEASON_BLACKLIST.search(base_title):
        return False
    alpha_tokens = re.findall(r"[A-Za-z]+", base_title)
    return len(alpha_tokens) >= 2


def should_parse_trailing_roman_as_season(base_title: str, number: int, raw_type: Any) -> bool:
    if not is_tv_type(raw_type):
        return False
    if number < 2 or number > 20:
        return False
    if TV_ONLY_YEARLESS_SEASON_BLACKLIST.search(base_title):
        return False
    alpha_tokens = re.findall(r"[A-Za-z]+", base_title)
    return len(alpha_tokens) >= 2


def parse_netflix_title(title: Any, raw_type: Any = None) -> dict[str, Any]:
    if title is None or pd.isna(title):
        return {
            "series_title": None,
            "normalized_title": None,
            "canonical_title": None,
            "compact_title": None,
            "raw_normalized_title": None,
            "raw_canonical_title": None,
            "raw_compact_title": None,
            "season_number": pd.NA,
            "season_label": None,
            "title_year_hint": pd.NA,
            "season_parse_method": "unparsed",
            "season_parse_confidence": 0.0,
            "title_parse_notes": None,
        }

    raw_title = normalize_unicode_punctuation(str(title).strip())
    raw_title = collapse_dotted_initialisms(raw_title)
    raw_normalized_title = normalize_title(raw_title)
    raw_canonical_title = canonicalize_title(raw_title)
    raw_compact_title = compact_title_key(raw_title)
    parse_notes: list[str] = []
    working_title, stripped_notes = strip_secondary_segments(raw_title)
    parse_notes.extend(stripped_notes)

    parsed_title = working_title
    season_number: int | None = None
    season_label: str | None = None
    season_parse_method = "unparsed"
    season_parse_confidence = 0.0

    limited_match = LIMITED_SERIES_PATTERN.match(working_title)
    if limited_match:
        parsed_title = limited_match.group("base").strip()
        season_number = 1
        season_label = "limited_series"
        season_parse_method = "limited_series"
        season_parse_confidence = 0.95
        parse_notes.append("interpreted limited-series title as season 1")
    else:
        temporada_match = TEMPORADA_PATTERN.match(working_title)
        if temporada_match:
            parsed_title = temporada_match.group("base").strip()
            season_number = parse_season_token(temporada_match.group("number"))
            season_label = "temporada"
            season_parse_method = "temporada"
            season_parse_confidence = 0.94
        else:
            for label, pattern, confidence in GENERIC_SEASON_PATTERNS:
                match = pattern.match(working_title)
                if not match:
                    continue

                parsed_title = match.group("base").strip()
                season_number = parse_season_token(match.group("number"))
                season_label = label
                season_parse_method = label
                season_parse_confidence = confidence
                break

    if season_number is None:
        bare_number_match = TV_ONLY_BARE_NUMBER_PATTERN.match(working_title)
        if bare_number_match:
            guessed_number = int(bare_number_match.group("number"))
            base_title = bare_number_match.group("base").strip()
            if should_parse_trailing_number_as_season(base_title, guessed_number, raw_type):
                parsed_title = base_title
                season_number = guessed_number
                season_label = "bare_number"
                season_parse_method = "trailing_number_tv_heuristic"
                season_parse_confidence = 0.8
                parse_notes.append(
                    "inferred season from trailing number because raw type is TV and title does not look movie-like"
                )

    if season_number is None:
        roman_match = TV_ONLY_ROMAN_SUFFIX_PATTERN.match(working_title)
        if roman_match:
            guessed_number = parse_roman_numeral(roman_match.group("number"))
            base_title = roman_match.group("base").strip()
            if (
                guessed_number is not None
                and should_parse_trailing_roman_as_season(base_title, guessed_number, raw_type)
            ):
                parsed_title = base_title
                season_number = guessed_number
                season_label = "roman_suffix"
                season_parse_method = "trailing_roman_tv_heuristic"
                season_parse_confidence = 0.76
                parse_notes.append(
                    "inferred season from trailing Roman numeral because raw type is TV and title structure is season-like"
                )

    if season_number is not None and season_parse_method in {
        "part",
        "volume",
        "chapter",
        "book",
        "collection",
        "class",
    }:
        parse_notes.append(
            "season number comes from a release-style label and should be reviewed if IMDb title includes that label as part of the name"
        )

    title_year_hint = extract_year_hint(parsed_title)
    normalized_title = normalize_title(parsed_title)
    canonical_title = canonicalize_title(parsed_title)
    compact_title = compact_title_key(parsed_title)

    return {
        "series_title": parsed_title,
        "normalized_title": normalized_title,
        "canonical_title": canonical_title,
        "compact_title": compact_title,
        "raw_normalized_title": raw_normalized_title,
        "raw_canonical_title": raw_canonical_title,
        "raw_compact_title": raw_compact_title,
        "season_number": season_number if season_number is not None else pd.NA,
        "season_label": season_label,
        "title_year_hint": title_year_hint if title_year_hint is not None else pd.NA,
        "season_parse_method": season_parse_method,
        "season_parse_confidence": season_parse_confidence,
        "title_parse_notes": parse_notes_text(parse_notes),
    }


def infer_netflix_format(raw_type: Any, parse_result: dict[str, Any]) -> str:
    text = "" if raw_type is None or pd.isna(raw_type) else str(raw_type).strip().lower()
    title_raw = parse_result.get("raw_normalized_title") or ""

    if text == "movie":
        return "movie"
    if text in {"tv", "series", "show"}:
        return "series"
    if pd.notna(parse_result.get("season_number")):
        return "series"
    if parse_result.get("season_label") == "limited_series":
        return "series"
    if TITLE_PARSE_HINT_PATTERN.search(title_raw):
        return "series"
    return "unknown"


def infer_implied_season_one(title_raw: Any, raw_type: Any) -> tuple[pd._libs.missing.NAType | int, str | None]:
    if not is_tv_type(raw_type):
        return pd.NA, None
    if title_raw is None or pd.isna(title_raw):
        return pd.NA, None

    text = str(title_raw).strip()
    if EPISODIC_ONE_OFF_HINTS.search(text):
        return pd.NA, None
    if TITLE_PARSE_HINT_PATTERN.search(text):
        return pd.NA, None

    return 1, "tv_title_default_single_season_candidate"


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


def year_distance(reference_year: Any, start_year: Any) -> float | pd.NA:
    if reference_year is None or pd.isna(reference_year):
        return pd.NA
    if start_year is None or pd.isna(start_year):
        return pd.NA
    return float(abs(int(reference_year) - int(start_year)))
