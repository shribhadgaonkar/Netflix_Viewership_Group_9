from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import re

import pandas as pd
from rapidfuzz import fuzz, process


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.pipeline_utils import (  # noqa: E402
    MATCH_SOURCE_PRIORITY,
    canonicalize_title,
    coerce_nullable_int,
    compact_title_key,
    ensure_parent,
    interim_dir,
    log,
    normalize_title,
    optional_imdb_inputs,
    parse_numeric_series,
    raw_dir,
    require_imdb_inputs,
    year_consistency,
    year_distance,
)


MASTER_INPUT = (REPO_ROOT / "data" / "processed" / "netflix_imdb_master.parquet").resolve()
MASTER_INPUT_CSV = (REPO_ROOT / "data" / "processed" / "netflix_imdb_master.csv").resolve()
SERIES_SEASONS_INPUT = (interim_dir() / "imdb_series_seasons.csv").resolve()
SERIES_TITLE_KEYS_INPUT = (interim_dir() / "imdb_title_keys.csv").resolve()

IMDB_MOVIES_OUTPUT = (interim_dir() / "imdb_movies.csv").resolve()
IMDB_MOVIE_TITLE_KEYS_OUTPUT = (interim_dir() / "imdb_movie_title_keys.csv").resolve()
IMDB_SERIES_PARENTS_OUTPUT = (interim_dir() / "imdb_series_parents.csv").resolve()
IMDB_SERIES_PARENT_TITLE_KEYS_OUTPUT = (interim_dir() / "imdb_series_parent_title_keys.csv").resolve()

MASTER_V3_OUTPUT = (REPO_ROOT / "data" / "processed" / "netflix_imdb_master_v3.parquet").resolve()
MASTER_V3_CSV_OUTPUT = (REPO_ROOT / "data" / "processed" / "netflix_imdb_master_v3.csv").resolve()
MOVIE_REVIEW_OUTPUT = (REPO_ROOT / "data" / "processed" / "third_pass_movie_matches_review.csv").resolve()
SERIES_PARENT_REVIEW_OUTPUT = (
    REPO_ROOT / "data" / "processed" / "third_pass_series_overall_matches_review.csv"
).resolve()
STILL_UNMATCHED_OUTPUT = (REPO_ROOT / "data" / "processed" / "third_pass_still_unmatched.csv").resolve()
AMBIGUOUS_OUTPUT = (REPO_ROOT / "data" / "processed" / "third_pass_ambiguous_candidates.csv").resolve()
DELTA_SUMMARY_OUTPUT = (REPO_ROOT / "data" / "processed" / "third_pass_delta_summary.csv").resolve()

MANUAL_MOVIE_OVERRIDE_INPUT = (REPO_ROOT / "config" / "manual_movie_match_overrides.csv").resolve()
MANUAL_SERIES_PARENT_OVERRIDE_INPUT = (
    REPO_ROOT / "config" / "manual_series_parent_overrides.csv"
).resolve()

NA_VALUES = ["\\N"]
CHUNK_SIZE = 500_000
MOVIE_TITLE_TYPES = {"movie", "tvMovie"}
EXACT_SOURCES_BY_STAGE = {
    "movie_primary_exact": ["primary"],
    "movie_original_exact": ["original"],
    "movie_aka_exact": ["aka"],
    "series_parent_primary_exact": ["primary"],
    "series_parent_original_exact": ["original"],
    "series_parent_aka_exact": ["aka"],
}
MATCH_CONFIDENCE_BY_STAGE = {
    "movie_primary_exact": 0.99,
    "movie_original_exact": 0.97,
    "movie_aka_exact": 0.96,
    "movie_canonical_exact": 0.95,
    "movie_compact_exact": 0.93,
    "movie_fuzzy": 0.88,
    "series_parent_primary_exact": 0.98,
    "series_parent_original_exact": 0.96,
    "series_parent_aka_exact": 0.95,
    "series_parent_canonical_exact": 0.94,
    "series_parent_compact_exact": 0.92,
    "series_parent_fuzzy": 0.87,
    "manual_movie_override": 1.0,
    "manual_series_parent_override": 1.0,
}
THIRD_PASS_METHOD_BY_STAGE = {
    "movie_primary_exact": "movie_exact_primary_title",
    "movie_original_exact": "movie_exact_original_title",
    "movie_aka_exact": "movie_exact_aka_title",
    "movie_canonical_exact": "movie_exact_canonical_title",
    "movie_compact_exact": "movie_exact_compact_title",
    "movie_fuzzy": "movie_fuzzy_title",
    "series_parent_primary_exact": "series_parent_exact_primary_title",
    "series_parent_original_exact": "series_parent_exact_original_title",
    "series_parent_aka_exact": "series_parent_exact_aka_title",
    "series_parent_canonical_exact": "series_parent_exact_canonical_title",
    "series_parent_compact_exact": "series_parent_exact_compact_title",
    "series_parent_fuzzy": "series_parent_fuzzy_title",
    "manual_movie_override": "manual_movie_override",
    "manual_series_parent_override": "manual_series_parent_override",
}
REVIEW_BASE_COLUMNS = [
    "netflix_row_id",
    "netflix_title_raw",
    "netflix_series_title",
    "netflix_content_grain",
    "netflix_season_number",
    "prior_match_method",
    "prior_match_confidence",
    "third_pass_match_method",
    "third_pass_match_stage",
    "third_pass_match_confidence",
    "third_pass_match_notes",
    "candidate_imdb_count",
    "candidate_imdb_parent_tconsts",
    "candidate_imdb_primary_titles",
    "candidate_match_source",
    "netflix_match_key_used",
    "imdb_match_key_used",
    "imdb_match_entity_type",
    "imdb_resolved_tconst",
    "imdb_primary_title",
    "imdb_original_title",
    "imdb_title_type",
    "imdb_start_year",
    "imdb_end_year",
    "imdb_runtime_minutes",
    "imdb_average_rating",
    "imdb_num_votes",
    "title_similarity_score",
    "year_distance",
]

FUZZY_THRESHOLD = {
    "movie": 98.0,
    "series_parent": 97.0,
}
FUZZY_MIN_GAP = {
    "movie": 2.0,
    "series_parent": 2.0,
}


@dataclass(frozen=True)
class CandidateIndex:
    entities: pd.DataFrame
    title_keys: pd.DataFrame
    normalized_lookup: dict[tuple[str, str], list[int]]
    canonical_lookup: dict[str, list[int]]
    compact_lookup: dict[str, list[int]]


MOVIE_SPECIAL_HINT_PATTERN = re.compile(r"(?i)\b(?:movie|film)\b")
SERIES_SPECIAL_UNKNOWN_PATTERN = re.compile(
    r"(?i)\b(?:holiday|special|bonus|lyric|concert|live|vs\.?|versus)\b"
)
LIMITED_SERIES_HINT_PATTERN = r"(?i)\blimited series\b"


def load_master() -> pd.DataFrame:
    if MASTER_INPUT.exists():
        log(f"Loading baseline master parquet: {MASTER_INPUT.relative_to(REPO_ROOT).as_posix()}")
        return pd.read_parquet(MASTER_INPUT)
    if MASTER_INPUT_CSV.exists():
        log(f"Loading baseline master csv: {MASTER_INPUT_CSV.relative_to(REPO_ROOT).as_posix()}")
        return pd.read_csv(MASTER_INPUT_CSV, low_memory=False)
    raise FileNotFoundError("Baseline master dataset not found. Expected parquet or CSV output.")


def normalize_master_types(master: pd.DataFrame) -> pd.DataFrame:
    frame = master.copy()
    for column in [
        "netflix_row_id",
        "netflix_season_number",
        "netflix_release_year",
        "netflix_title_year_hint",
        "imdb_start_year",
        "imdb_end_year",
        "imdb_runtime_minutes",
        "imdb_num_votes",
        "imdb_parent_season_count",
        "imdb_season_number",
        "imdb_season_episode_count",
        "candidate_imdb_count",
        "candidate_rank",
    ]:
        if column in frame.columns:
            frame[column] = coerce_nullable_int(frame[column])

    for column in [
        "match_confidence",
        "year_distance",
        "title_similarity_score",
        "netflix_runtime",
        "imdb_average_rating",
    ]:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")

    return frame


def mtime(path: Path) -> float:
    return path.stat().st_mtime if path.exists() else 0.0


def outputs_fresh(outputs: list[Path], inputs: list[Path]) -> bool:
    if any(not output.exists() for output in outputs):
        return False
    oldest_output = min(mtime(output) for output in outputs)
    newest_input = max(mtime(path) for path in inputs if path.exists())
    return oldest_output >= newest_input


def value_options(row: pd.Series, columns: list[str]) -> list[str]:
    values: list[str] = []
    for column in columns:
        value = row.get(column)
        if value is None or pd.isna(value):
            continue
        text = str(value).strip()
        if text and text not in values:
            values.append(text)
    return values


def compact_candidate_values(series: pd.Series) -> str | None:
    values = [str(value) for value in pd.unique(series.dropna()) if str(value).strip()]
    return " | ".join(values) if values else None


def get_reference_year(row: pd.Series) -> int | None:
    for column in ["netflix_release_year", "netflix_title_year_hint"]:
        value = row.get(column)
        if value is not None and not pd.isna(value):
            return int(value)
    return None


def classify_content_grain(row: pd.Series) -> str:
    title_raw = "" if pd.isna(row.get("netflix_title_raw")) else str(row.get("netflix_title_raw"))
    raw_type = "" if pd.isna(row.get("raw_netflix_type")) else str(row.get("raw_netflix_type")).strip().lower()
    season_number = row.get("netflix_season_number")
    netflix_format = "" if pd.isna(row.get("netflix_format")) else str(row.get("netflix_format")).strip().lower()
    runtime_value = pd.to_numeric(row.get("netflix_runtime"), errors="coerce")

    if raw_type == "movie":
        return "movie"
    if pd.isna(season_number) and runtime_value is not None and not pd.isna(runtime_value):
        if runtime_value >= 40 and pd.notna(row.get("netflix_title_raw")) and MOVIE_SPECIAL_HINT_PATTERN.search(
            title_raw
        ):
            return "movie"
    if netflix_format == "movie":
        return "movie"
    if netflix_format == "series" and pd.notna(season_number):
        return "series_season"
    if row.get("match_status") == "matched" and pd.notna(row.get("imdb_season_number")):
        return "series_season"
    if netflix_format == "series":
        if SERIES_SPECIAL_UNKNOWN_PATTERN.search(title_raw):
            return "unknown"
        return "series_overall"
    return "unknown"


def read_ratings_lookup(ratings_path: Path, candidate_ids: set[str]) -> pd.DataFrame:
    collected: list[pd.DataFrame] = []
    for chunk in pd.read_csv(
        ratings_path,
        sep="\t",
        compression="gzip",
        na_values=NA_VALUES,
        keep_default_na=True,
        usecols=["tconst", "averageRating", "numVotes"],
        chunksize=CHUNK_SIZE,
        low_memory=False,
    ):
        filtered = chunk[chunk["tconst"].isin(candidate_ids)].copy()
        if filtered.empty:
            continue
        filtered.rename(
            columns={
                "tconst": "imdb_tconst",
                "averageRating": "imdb_average_rating",
                "numVotes": "imdb_num_votes",
            },
            inplace=True,
        )
        filtered["imdb_num_votes"] = coerce_nullable_int(filtered["imdb_num_votes"])
        collected.append(filtered)
    if not collected:
        return pd.DataFrame(columns=["imdb_tconst", "imdb_average_rating", "imdb_num_votes"])
    return pd.concat(collected, ignore_index=True).drop_duplicates(subset=["imdb_tconst"])


def build_target_key_sets(unresolved: pd.DataFrame, grain: str) -> dict[str, set[str]]:
    key_sets = {"normalized": set(), "canonical": set(), "compact": set()}
    key_columns = {
        "normalized": ["netflix_normalized_title", "netflix_raw_normalized_title"],
        "canonical": ["netflix_canonical_title", "netflix_raw_canonical_title"],
        "compact": ["netflix_compact_title", "netflix_raw_compact_title"],
    }
    for _, row in unresolved[unresolved["netflix_content_grain"] == grain].iterrows():
        for key_type, columns in key_columns.items():
            for column in columns:
                value = row.get(column)
                if value is None or pd.isna(value):
                    continue
                text = str(value).strip()
                if text:
                    key_sets[key_type].add(text)
    return key_sets


def build_movie_index(unresolved: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    imdb_raw_dir = raw_dir() / "imdb"
    required_inputs = require_imdb_inputs(imdb_raw_dir)
    optional_inputs = optional_imdb_inputs(imdb_raw_dir)
    basics_path = required_inputs["title.basics.tsv.gz"]
    ratings_path = required_inputs["title.ratings.tsv.gz"]
    akas_path = optional_inputs.get("title.akas.tsv.gz")
    target_keys = build_target_key_sets(unresolved, "movie")

    inputs = [MASTER_INPUT if MASTER_INPUT.exists() else MASTER_INPUT_CSV, basics_path, ratings_path, Path(__file__)]
    if akas_path is not None:
        inputs.append(akas_path)

    if outputs_fresh([IMDB_MOVIES_OUTPUT, IMDB_MOVIE_TITLE_KEYS_OUTPUT], inputs):
        log("Reusing cached incremental IMDb movie index.")
        return (
            pd.read_csv(IMDB_MOVIES_OUTPUT, low_memory=False),
            pd.read_csv(IMDB_MOVIE_TITLE_KEYS_OUTPUT, low_memory=False),
        )

    if not any(target_keys.values()):
        movies = pd.DataFrame(
            columns=[
                "imdb_tconst",
                "imdb_primary_title",
                "imdb_original_title",
                "imdb_title_type",
                "imdb_start_year",
                "imdb_end_year",
                "imdb_runtime_minutes",
                "imdb_genres",
                "imdb_average_rating",
                "imdb_num_votes",
                "imdb_primary_normalized_title",
                "imdb_original_normalized_title",
                "imdb_primary_canonical_title",
                "imdb_original_canonical_title",
                "imdb_primary_compact_title",
                "imdb_original_compact_title",
            ]
        )
        title_keys = pd.DataFrame(
            columns=[
                "imdb_tconst",
                "imdb_candidate_display_title",
                "imdb_match_key_used",
                "imdb_match_key_canonical",
                "imdb_match_key_compact",
                "candidate_match_source",
            ]
        )
        ensure_parent(IMDB_MOVIES_OUTPUT)
        movies.to_csv(IMDB_MOVIES_OUTPUT, index=False)
        title_keys.to_csv(IMDB_MOVIE_TITLE_KEYS_OUTPUT, index=False)
        return movies, title_keys

    matched_basic_chunks: list[pd.DataFrame] = []
    candidate_ids: set[str] = set()
    basics_usecols = [
        "tconst",
        "titleType",
        "primaryTitle",
        "originalTitle",
        "startYear",
        "endYear",
        "runtimeMinutes",
        "genres",
    ]

    for chunk in pd.read_csv(
        basics_path,
        sep="\t",
        compression="gzip",
        na_values=NA_VALUES,
        keep_default_na=True,
        usecols=basics_usecols,
        chunksize=CHUNK_SIZE,
        low_memory=False,
    ):
        filtered = chunk[chunk["titleType"].isin(MOVIE_TITLE_TYPES)].copy()
        if filtered.empty:
            continue
        filtered["imdb_primary_normalized_title"] = filtered["primaryTitle"].map(normalize_title)
        filtered["imdb_original_normalized_title"] = filtered["originalTitle"].map(normalize_title)
        filtered["imdb_primary_canonical_title"] = filtered["primaryTitle"].map(canonicalize_title)
        filtered["imdb_original_canonical_title"] = filtered["originalTitle"].map(canonicalize_title)
        filtered["imdb_primary_compact_title"] = filtered["primaryTitle"].map(compact_title_key)
        filtered["imdb_original_compact_title"] = filtered["originalTitle"].map(compact_title_key)
        mask = (
            filtered["imdb_primary_normalized_title"].isin(target_keys["normalized"])
            | filtered["imdb_original_normalized_title"].isin(target_keys["normalized"])
            | filtered["imdb_primary_canonical_title"].isin(target_keys["canonical"])
            | filtered["imdb_original_canonical_title"].isin(target_keys["canonical"])
            | filtered["imdb_primary_compact_title"].isin(target_keys["compact"])
            | filtered["imdb_original_compact_title"].isin(target_keys["compact"])
        )
        matched = filtered[mask].copy()
        if matched.empty:
            continue
        candidate_ids.update(matched["tconst"].astype(str).tolist())
        matched_basic_chunks.append(matched)

    aka_matches: list[pd.DataFrame] = []
    aka_candidate_ids: set[str] = set()
    if akas_path is not None:
        for chunk in pd.read_csv(
            akas_path,
            sep="\t",
            compression="gzip",
            na_values=NA_VALUES,
            keep_default_na=True,
            usecols=["titleId", "title", "region", "language", "types", "isOriginalTitle"],
            chunksize=CHUNK_SIZE,
            low_memory=False,
        ):
            chunk["imdb_aka_normalized_title"] = chunk["title"].map(normalize_title)
            chunk["imdb_aka_canonical_title"] = chunk["title"].map(canonicalize_title)
            chunk["imdb_aka_compact_title"] = chunk["title"].map(compact_title_key)
            mask = (
                chunk["imdb_aka_normalized_title"].isin(target_keys["normalized"])
                | chunk["imdb_aka_canonical_title"].isin(target_keys["canonical"])
                | chunk["imdb_aka_compact_title"].isin(target_keys["compact"])
            )
            filtered = chunk[mask].copy()
            if filtered.empty:
                continue
            filtered.rename(columns={"titleId": "imdb_tconst", "title": "imdb_aka_title"}, inplace=True)
            aka_candidate_ids.update(filtered["imdb_tconst"].astype(str).tolist())
            aka_matches.append(filtered)

    missing_ids = aka_candidate_ids.difference(candidate_ids)
    if missing_ids:
        for chunk in pd.read_csv(
            basics_path,
            sep="\t",
            compression="gzip",
            na_values=NA_VALUES,
            keep_default_na=True,
            usecols=basics_usecols,
            chunksize=CHUNK_SIZE,
            low_memory=False,
        ):
            filtered = chunk[
                chunk["tconst"].isin(missing_ids) & chunk["titleType"].isin(MOVIE_TITLE_TYPES)
            ].copy()
            if filtered.empty:
                continue
            filtered["imdb_primary_normalized_title"] = filtered["primaryTitle"].map(normalize_title)
            filtered["imdb_original_normalized_title"] = filtered["originalTitle"].map(normalize_title)
            filtered["imdb_primary_canonical_title"] = filtered["primaryTitle"].map(canonicalize_title)
            filtered["imdb_original_canonical_title"] = filtered["originalTitle"].map(canonicalize_title)
            filtered["imdb_primary_compact_title"] = filtered["primaryTitle"].map(compact_title_key)
            filtered["imdb_original_compact_title"] = filtered["originalTitle"].map(compact_title_key)
            matched_basic_chunks.append(filtered)
            candidate_ids.update(filtered["tconst"].astype(str).tolist())

    if matched_basic_chunks:
        movies = pd.concat(matched_basic_chunks, ignore_index=True)
    else:
        movies = pd.DataFrame(columns=basics_usecols)
    if not movies.empty:
        movies = movies[movies["titleType"].isin(MOVIE_TITLE_TYPES)].copy()
    movies.drop_duplicates(subset=["tconst"], inplace=True)

    if not movies.empty:
        movies.rename(
            columns={
                "tconst": "imdb_tconst",
                "titleType": "imdb_title_type",
                "primaryTitle": "imdb_primary_title",
                "originalTitle": "imdb_original_title",
                "startYear": "imdb_start_year",
                "endYear": "imdb_end_year",
                "runtimeMinutes": "imdb_runtime_minutes",
                "genres": "imdb_genres",
            },
            inplace=True,
        )
        movies["imdb_start_year"] = coerce_nullable_int(movies["imdb_start_year"])
        movies["imdb_end_year"] = coerce_nullable_int(movies["imdb_end_year"])
        movies["imdb_runtime_minutes"] = parse_numeric_series(movies["imdb_runtime_minutes"]).round().astype("Int64")

    ratings = read_ratings_lookup(ratings_path, candidate_ids) if candidate_ids else pd.DataFrame()
    if not ratings.empty:
        movies = movies.merge(ratings, how="left", on="imdb_tconst")
    else:
        movies["imdb_average_rating"] = pd.NA
        movies["imdb_num_votes"] = pd.Series(pd.NA, index=movies.index, dtype="Int64")

    key_frames: list[pd.DataFrame] = []
    if not movies.empty:
        primary = movies[
            [
                "imdb_tconst",
                "imdb_primary_title",
                "imdb_primary_normalized_title",
                "imdb_primary_canonical_title",
                "imdb_primary_compact_title",
            ]
        ].copy()
        primary["candidate_match_source"] = "primary"
        primary.rename(
            columns={
                "imdb_primary_title": "imdb_candidate_display_title",
                "imdb_primary_normalized_title": "imdb_match_key_used",
                "imdb_primary_canonical_title": "imdb_match_key_canonical",
                "imdb_primary_compact_title": "imdb_match_key_compact",
            },
            inplace=True,
        )
        key_frames.append(primary)

        original = movies[
            [
                "imdb_tconst",
                "imdb_original_title",
                "imdb_original_normalized_title",
                "imdb_original_canonical_title",
                "imdb_original_compact_title",
            ]
        ].copy()
        original["candidate_match_source"] = "original"
        original.rename(
            columns={
                "imdb_original_title": "imdb_candidate_display_title",
                "imdb_original_normalized_title": "imdb_match_key_used",
                "imdb_original_canonical_title": "imdb_match_key_canonical",
                "imdb_original_compact_title": "imdb_match_key_compact",
            },
            inplace=True,
        )
        key_frames.append(original)

    if aka_matches:
        akas = pd.concat(aka_matches, ignore_index=True)
        if not movies.empty:
            akas = akas[akas["imdb_tconst"].isin(movies["imdb_tconst"].astype("string"))].copy()
        aka_keys = akas[
            [
                "imdb_tconst",
                "imdb_aka_title",
                "imdb_aka_normalized_title",
                "imdb_aka_canonical_title",
                "imdb_aka_compact_title",
            ]
        ].copy()
        aka_keys["candidate_match_source"] = "aka"
        aka_keys.rename(
            columns={
                "imdb_aka_title": "imdb_candidate_display_title",
                "imdb_aka_normalized_title": "imdb_match_key_used",
                "imdb_aka_canonical_title": "imdb_match_key_canonical",
                "imdb_aka_compact_title": "imdb_match_key_compact",
            },
            inplace=True,
        )
        key_frames.append(aka_keys)

    title_keys = pd.concat(key_frames, ignore_index=True) if key_frames else pd.DataFrame()
    if not title_keys.empty:
        title_keys = title_keys[
            title_keys["imdb_match_key_used"].notna()
            & title_keys["imdb_match_key_used"].astype("string").str.strip().ne("")
        ].copy()
        title_keys.drop_duplicates(
            subset=["imdb_tconst", "candidate_match_source", "imdb_match_key_used"],
            inplace=True,
        )

    ensure_parent(IMDB_MOVIES_OUTPUT)
    movies.to_csv(IMDB_MOVIES_OUTPUT, index=False)
    title_keys.to_csv(IMDB_MOVIE_TITLE_KEYS_OUTPUT, index=False)
    log(f"Saved incremental IMDb movie index: {len(movies):,} rows")
    log(f"Saved incremental IMDb movie title keys: {len(title_keys):,} rows")
    return movies, title_keys


def build_parent_series_index(unresolved: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    inputs = [
        MASTER_INPUT if MASTER_INPUT.exists() else MASTER_INPUT_CSV,
        SERIES_SEASONS_INPUT,
        SERIES_TITLE_KEYS_INPUT,
        Path(__file__),
    ]
    if outputs_fresh([IMDB_SERIES_PARENTS_OUTPUT, IMDB_SERIES_PARENT_TITLE_KEYS_OUTPUT], inputs):
        log("Reusing cached incremental IMDb parent-series index.")
        return (
            pd.read_csv(IMDB_SERIES_PARENTS_OUTPUT, low_memory=False),
            pd.read_csv(IMDB_SERIES_PARENT_TITLE_KEYS_OUTPUT, low_memory=False),
        )

    parents_source = pd.read_csv(SERIES_SEASONS_INPUT, low_memory=False)
    title_keys_source = pd.read_csv(SERIES_TITLE_KEYS_INPUT, low_memory=False)
    target_keys = build_target_key_sets(unresolved, "series_overall")

    parents = (
        parents_source[
            [
                "imdb_parent_tconst",
                "imdb_primary_title",
                "imdb_original_title",
                "imdb_normalized_title",
                "imdb_primary_normalized_title",
                "imdb_original_normalized_title",
                "imdb_primary_canonical_title",
                "imdb_original_canonical_title",
                "imdb_title_type",
                "imdb_start_year",
                "imdb_end_year",
                "imdb_genres",
                "imdb_runtime_minutes",
                "imdb_average_rating",
                "imdb_num_votes",
                "imdb_parent_season_count",
                "imdb_aka_normalized_titles",
                "imdb_aka_canonical_titles",
                "imdb_aka_title_count",
            ]
        ]
        .drop_duplicates(subset=["imdb_parent_tconst"])
        .copy()
    )

    key_mask = (
        title_keys_source["imdb_match_key_used"].isin(target_keys["normalized"])
        | title_keys_source["imdb_match_key_canonical"].isin(target_keys["canonical"])
        | title_keys_source["imdb_match_key_compact"].isin(target_keys["compact"])
    )
    parent_title_keys = title_keys_source[key_mask].copy()
    candidate_ids = set(parent_title_keys["imdb_parent_tconst"].astype(str).tolist())

    if candidate_ids:
        parents = parents[parents["imdb_parent_tconst"].isin(candidate_ids)].copy()
    else:
        parents = parents.iloc[0:0].copy()

    ensure_parent(IMDB_SERIES_PARENTS_OUTPUT)
    parents.to_csv(IMDB_SERIES_PARENTS_OUTPUT, index=False)
    parent_title_keys.to_csv(IMDB_SERIES_PARENT_TITLE_KEYS_OUTPUT, index=False)
    log(f"Saved incremental IMDb parent-series index: {len(parents):,} rows")
    log(f"Saved incremental IMDb parent-series title keys: {len(parent_title_keys):,} rows")
    return parents, parent_title_keys


def build_candidate_index(
    entities: pd.DataFrame,
    title_keys: pd.DataFrame,
    id_column: str,
    entity_type: str,
) -> CandidateIndex:
    entity_frame = entities.copy()
    entity_frame[id_column] = entity_frame[id_column].astype("string")
    entity_frame["imdb_match_entity_type"] = entity_type
    key_frame = title_keys.copy()
    key_frame[id_column] = key_frame[id_column].astype("string")
    candidate_pool = entity_frame.merge(key_frame, how="inner", on=id_column)
    candidate_pool["candidate_match_source"] = candidate_pool["candidate_match_source"].astype("string")
    candidate_pool["source_priority"] = candidate_pool["candidate_match_source"].map(MATCH_SOURCE_PRIORITY).fillna(9)

    normalized_lookup: dict[tuple[str, str], list[int]] = {}
    normalized_groups = candidate_pool.groupby(["candidate_match_source", "imdb_match_key_used"], sort=False).groups
    for (source, key_value), indices in normalized_groups.items():
        if pd.isna(source) or pd.isna(key_value):
            continue
        normalized_lookup[(str(source), str(key_value))] = list(indices)

    canonical_lookup: dict[str, list[int]] = {}
    canonical_groups = candidate_pool.groupby("imdb_match_key_canonical", sort=False).groups
    for key_value, indices in canonical_groups.items():
        if pd.isna(key_value):
            continue
        canonical_lookup[str(key_value)] = list(indices)

    compact_lookup: dict[str, list[int]] = {}
    compact_groups = candidate_pool.groupby("imdb_match_key_compact", sort=False).groups
    for key_value, indices in compact_groups.items():
        if pd.isna(key_value):
            continue
        compact_lookup[str(key_value)] = list(indices)

    return CandidateIndex(
        entities=entity_frame,
        title_keys=candidate_pool,
        normalized_lookup=normalized_lookup,
        canonical_lookup=canonical_lookup,
        compact_lookup=compact_lookup,
    )


def title_type_priority_for_parent(row: pd.Series, title_type: Any) -> int:
    season_label = "" if pd.isna(row.get("netflix_season_label")) else str(row.get("netflix_season_label"))
    text = "" if pd.isna(title_type) else str(title_type)
    if pd.Series([season_label]).str.contains(LIMITED_SERIES_HINT_PATTERN, regex=True).iloc[0]:
        if text == "tvMiniSeries":
            return 0
        if text == "tvSeries":
            return 1
        return 2
    if text == "tvSeries":
        return 0
    if text == "tvMiniSeries":
        return 1
    return 2


def deduplicate_candidates(candidates: pd.DataFrame, id_column: str) -> pd.DataFrame:
    ordered = candidates.sort_values(
        ["source_priority", "candidate_match_source", "imdb_match_key_used", id_column]
    )
    return ordered.drop_duplicates(subset=[id_column], keep="first").copy()


def resolve_match_confidence(stage_name: str, resolution_method: str, candidate_count: int) -> float:
    confidence = MATCH_CONFIDENCE_BY_STAGE[stage_name]
    if resolution_method == "single_candidate":
        confidence += 0.01
    elif resolution_method in {
        "source_priority",
        "year_consistency",
        "exact_start_year",
        "closest_start_year",
        "runtime_proximity",
    }:
        confidence += 0.005
    elif resolution_method in {"higher_num_votes", "title_similarity", "title_type_preference"}:
        confidence -= 0.02
    elif "fuzzy" in stage_name:
        confidence -= 0.01 * max(candidate_count - 1, 0)
    return float(max(0.0, min(1.0, confidence)))


def build_unresolved_snapshot(
    row: pd.Series,
    candidate_frame: pd.DataFrame,
    grain: str,
    stage_name: str,
    netflix_key_used: str,
    note: str,
) -> dict[str, object]:
    return {
        "match_status": "unmatched",
        "match_method": row.get("match_method"),
        "match_stage": row.get("match_stage"),
        "match_confidence": row.get("match_confidence", 0.0),
        "match_notes": row.get("match_notes"),
        "third_pass_match_status": "unmatched",
        "third_pass_match_method": f"{grain}_unmatched_ambiguous_candidates",
        "third_pass_match_stage": stage_name,
        "third_pass_match_confidence": 0.0,
        "third_pass_match_notes": note,
        "candidate_imdb_count": int(len(candidate_frame)),
        "candidate_imdb_parent_tconsts": compact_candidate_values(
            candidate_frame["imdb_entity_id"] if "imdb_entity_id" in candidate_frame.columns else candidate_frame.iloc[:, 0]
        ),
        "candidate_imdb_primary_titles": compact_candidate_values(candidate_frame["imdb_primary_title"]),
        "candidate_match_source": compact_candidate_values(candidate_frame["candidate_match_source"]),
        "netflix_match_key_used": netflix_key_used,
        "imdb_match_key_used": pd.NA,
        "candidate_rank": pd.NA,
        "ambiguity_resolution_method": pd.NA,
        "year_distance": pd.NA,
        "title_similarity_score": pd.to_numeric(candidate_frame["title_similarity_score"], errors="coerce").max(),
        "imdb_match_entity_type": grain,
    }


def resolve_candidate_group(
    row: pd.Series,
    candidates: pd.DataFrame,
    grain: str,
    entity_type: str,
    id_column: str,
    stage_name: str,
    netflix_key_used: str,
    title_similarity_metric: str,
) -> tuple[dict[str, object] | None, dict[str, object], pd.DataFrame]:
    candidate_frame = deduplicate_candidates(candidates, id_column)
    candidate_frame["imdb_entity_id"] = candidate_frame[id_column].astype("string")
    reference_year = get_reference_year(row)
    runtime_reference = pd.to_numeric(row.get("netflix_runtime"), errors="coerce")
    candidate_frame["year_consistency_flag"] = [
        year_consistency(reference_year, start_year, end_year)
        for start_year, end_year in zip(
            candidate_frame["imdb_start_year"], candidate_frame.get("imdb_end_year", pd.Series([pd.NA] * len(candidate_frame))), strict=False
        )
    ]
    candidate_frame["year_distance"] = [
        year_distance(reference_year, start_year)
        for start_year in candidate_frame["imdb_start_year"]
    ]
    candidate_frame["exact_start_year_match"] = [
        bool(reference_year == int(start_year))
        if reference_year is not None and start_year is not None and not pd.isna(start_year)
        else False
        for start_year in candidate_frame["imdb_start_year"]
    ]
    candidate_frame["title_similarity_score"] = pd.to_numeric(
        candidate_frame.get("title_similarity_score", 100.0), errors="coerce"
    ).fillna(100.0)
    candidate_frame["title_similarity_rank"] = -candidate_frame["title_similarity_score"]
    candidate_frame["source_priority"] = candidate_frame["candidate_match_source"].map(MATCH_SOURCE_PRIORITY).fillna(9)
    candidate_frame["votes_rank"] = -pd.to_numeric(candidate_frame["imdb_num_votes"], errors="coerce").fillna(-1)
    candidate_frame["year_consistency_rank"] = candidate_frame["year_consistency_flag"].map({True: 0, False: 2}).fillna(1)
    candidate_frame["exact_start_year_rank"] = candidate_frame["exact_start_year_match"].map({True: 0, False: 1})
    candidate_frame["year_distance_rank"] = pd.to_numeric(candidate_frame["year_distance"], errors="coerce").fillna(9999.0)
    if grain == "movie":
        candidate_frame["runtime_distance"] = [
            float(abs(float(runtime_reference) - float(runtime_value)))
            if runtime_reference is not None
            and not pd.isna(runtime_reference)
            and runtime_value is not None
            and not pd.isna(runtime_value)
            else pd.NA
            for runtime_value in candidate_frame["imdb_runtime_minutes"]
        ]
        candidate_frame["runtime_distance_rank"] = pd.to_numeric(
            candidate_frame["runtime_distance"], errors="coerce"
        ).fillna(9999.0)
        candidate_frame["title_type_priority"] = 0
    else:
        candidate_frame["runtime_distance"] = pd.NA
        candidate_frame["runtime_distance_rank"] = 9999.0
        candidate_frame["title_type_priority"] = [
            title_type_priority_for_parent(row, value)
            for value in candidate_frame["imdb_title_type"]
        ]

    candidate_frame.sort_values(
        [
            "source_priority",
            "year_consistency_rank",
            "exact_start_year_rank",
            "year_distance_rank",
            "runtime_distance_rank",
            "title_type_priority",
            "title_similarity_rank",
            "votes_rank",
            "imdb_entity_id",
        ],
        inplace=True,
    )

    note = "Multiple IMDb candidates remained after third-pass tie-breaks."
    unresolved_snapshot = build_unresolved_snapshot(
        row=row,
        candidate_frame=candidate_frame,
        grain=entity_type,
        stage_name=stage_name,
        netflix_key_used=netflix_key_used,
        note=note,
    )

    if reference_year is not None and (candidate_frame["year_consistency_flag"] == False).all():
        unresolved_snapshot["third_pass_match_method"] = f"{entity_type}_unmatched_year_conflict"
        unresolved_snapshot["third_pass_match_notes"] = (
            f"All third-pass candidates conflicted with reference year {reference_year}."
        )
        return None, unresolved_snapshot, candidate_frame

    top = candidate_frame.iloc[0]
    if len(candidate_frame) == 1:
        resolution_method = "single_candidate"
    else:
        second = candidate_frame.iloc[1]
        comparison_columns = [
            "source_priority",
            "year_consistency_rank",
            "exact_start_year_rank",
            "year_distance_rank",
            "runtime_distance_rank",
            "title_type_priority",
            "title_similarity_rank",
            "votes_rank",
        ]
        top_tuple = tuple(top[column] for column in comparison_columns)
        second_tuple = tuple(second[column] for column in comparison_columns)
        if top_tuple == second_tuple:
            return None, unresolved_snapshot, candidate_frame

        if top["source_priority"] < second["source_priority"]:
            resolution_method = "source_priority"
        elif top["year_consistency_rank"] < second["year_consistency_rank"]:
            resolution_method = "year_consistency"
        elif top["exact_start_year_rank"] < second["exact_start_year_rank"]:
            resolution_method = "exact_start_year"
        elif top["year_distance_rank"] < second["year_distance_rank"]:
            resolution_method = "closest_start_year"
        elif top["runtime_distance_rank"] < second["runtime_distance_rank"]:
            resolution_method = "runtime_proximity"
        elif top["title_type_priority"] < second["title_type_priority"]:
            resolution_method = "title_type_preference"
        elif top["title_similarity_rank"] < second["title_similarity_rank"]:
            resolution_method = "title_similarity"
        elif top["votes_rank"] < second["votes_rank"]:
            resolution_method = "higher_num_votes"
        else:
            return None, unresolved_snapshot, candidate_frame

    if grain == "movie" and pd.notna(top["runtime_distance"]) and float(top["runtime_distance"]) > 45.0:
        unresolved_snapshot["third_pass_match_method"] = "movie_unmatched_runtime_conflict"
        unresolved_snapshot["third_pass_match_notes"] = (
            f"Best movie candidate differed by {float(top['runtime_distance']):.0f} runtime minutes."
        )
        return None, unresolved_snapshot, candidate_frame

    result = {
        "match_status": "matched",
        "match_method": THIRD_PASS_METHOD_BY_STAGE[stage_name],
        "match_stage": stage_name,
        "match_confidence": resolve_match_confidence(stage_name, resolution_method, len(candidate_frame)),
        "match_notes": pd.NA
        if resolution_method == "single_candidate"
        else f"Third-pass resolved {len(candidate_frame)} candidates using {resolution_method}.",
        "third_pass_match_status": "matched",
        "third_pass_match_method": THIRD_PASS_METHOD_BY_STAGE[stage_name],
        "third_pass_match_stage": stage_name,
        "third_pass_match_confidence": resolve_match_confidence(
            stage_name, resolution_method, len(candidate_frame)
        ),
        "third_pass_match_notes": pd.NA
        if resolution_method == "single_candidate"
        else f"Third-pass resolved {len(candidate_frame)} candidates using {resolution_method}.",
        "candidate_imdb_count": int(len(candidate_frame)),
        "candidate_imdb_parent_tconsts": compact_candidate_values(candidate_frame["imdb_entity_id"]),
        "candidate_imdb_primary_titles": compact_candidate_values(candidate_frame["imdb_primary_title"]),
        "candidate_match_source": top["candidate_match_source"],
        "netflix_match_key_used": netflix_key_used,
        "imdb_match_key_used": top["imdb_match_key_used"],
        "candidate_rank": 1,
        "ambiguity_resolution_method": resolution_method,
        "year_consistency_flag": top["year_consistency_flag"],
        "year_distance": top["year_distance"],
        "title_similarity_score": top["title_similarity_score"],
        "title_similarity_metric": title_similarity_metric,
        "imdb_match_entity_type": entity_type,
        "movie_runtime_distance": top["runtime_distance"] if grain == "movie" else pd.NA,
        "movie_year_distance": top["year_distance"] if grain == "movie" else pd.NA,
        "series_parent_year_distance": top["year_distance"] if grain == "series_parent" else pd.NA,
        "movie_title_similarity_score": top["title_similarity_score"] if grain == "movie" else pd.NA,
        "series_parent_title_similarity_score": top["title_similarity_score"] if grain == "series_parent" else pd.NA,
    }
    metadata_columns = [
        "imdb_primary_title",
        "imdb_original_title",
        "imdb_title_type",
        "imdb_start_year",
        "imdb_end_year",
        "imdb_runtime_minutes",
        "imdb_genres",
        "imdb_average_rating",
        "imdb_num_votes",
        "imdb_parent_season_count",
        "imdb_normalized_title",
        "imdb_primary_normalized_title",
        "imdb_original_normalized_title",
        "imdb_primary_canonical_title",
        "imdb_original_canonical_title",
    ]
    for column in metadata_columns:
        if column in top.index:
            result[column] = top[column]

    if grain == "movie":
        result["imdb_parent_tconst"] = pd.NA
        result["imdb_season_number"] = pd.NA
        result["imdb_season_episode_count"] = pd.NA
        result["imdb_resolved_tconst"] = top[id_column]
    else:
        result["imdb_parent_tconst"] = top[id_column]
        result["imdb_resolved_tconst"] = top[id_column]
        result["imdb_season_number"] = pd.NA
        result["imdb_season_episode_count"] = pd.NA

    return result, unresolved_snapshot, candidate_frame


def perform_fuzzy_match(
    row: pd.Series,
    index: CandidateIndex,
    grain: str,
    entity_type: str,
    id_column: str,
) -> tuple[dict[str, object] | None, dict[str, object] | None, pd.DataFrame | None]:
    canonical_values = value_options(row, ["netflix_canonical_title", "netflix_raw_canonical_title"])
    if not canonical_values:
        return None, None, None

    candidate_pool = index.title_keys[index.title_keys["imdb_match_key_canonical"].notna()].copy()
    if grain == "series_parent":
        candidate_pool = candidate_pool[candidate_pool["candidate_match_source"].isin(["primary", "original", "aka"])]
    else:
        candidate_pool = candidate_pool[candidate_pool["candidate_match_source"].isin(["primary", "original", "aka"])]

    key_choices = candidate_pool["imdb_match_key_canonical"].dropna().astype(str).unique().tolist()
    if not key_choices:
        return None, None, None

    for netflix_key in canonical_values:
        matches = process.extract(str(netflix_key), key_choices, scorer=fuzz.ratio, limit=5)
        if not matches:
            continue
        top_key, top_score, _ = matches[0]
        second_score = matches[1][1] if len(matches) > 1 else 0.0
        if top_score < FUZZY_THRESHOLD[grain] or (top_score - second_score) < FUZZY_MIN_GAP[grain]:
            continue
        fuzzy_candidates = candidate_pool[candidate_pool["imdb_match_key_canonical"] == top_key].copy()
        fuzzy_candidates["title_similarity_score"] = float(top_score)
        stage_name = "movie_fuzzy" if grain == "movie" else "series_parent_fuzzy"
        match_result, unresolved_snapshot, candidate_frame = resolve_candidate_group(
            row=row,
            candidates=fuzzy_candidates,
            grain=grain,
            entity_type=entity_type,
            id_column=id_column,
            stage_name=stage_name,
            netflix_key_used=str(netflix_key),
            title_similarity_metric="fuzz_ratio",
        )
        return match_result, unresolved_snapshot, candidate_frame
    return None, None, None


def exact_stage_sequence(grain: str) -> list[tuple[str, list[str], str]]:
    if grain == "movie":
        return [
            ("movie_primary_exact", ["netflix_normalized_title", "netflix_raw_normalized_title"], "normalized"),
            ("movie_original_exact", ["netflix_normalized_title", "netflix_raw_normalized_title"], "normalized"),
            ("movie_aka_exact", ["netflix_normalized_title", "netflix_raw_normalized_title"], "normalized"),
            ("movie_canonical_exact", ["netflix_canonical_title", "netflix_raw_canonical_title"], "canonical"),
            ("movie_compact_exact", ["netflix_compact_title", "netflix_raw_compact_title"], "compact"),
        ]
    return [
        ("series_parent_primary_exact", ["netflix_normalized_title", "netflix_raw_normalized_title"], "normalized"),
        ("series_parent_original_exact", ["netflix_normalized_title", "netflix_raw_normalized_title"], "normalized"),
        ("series_parent_aka_exact", ["netflix_normalized_title", "netflix_raw_normalized_title"], "normalized"),
        ("series_parent_canonical_exact", ["netflix_canonical_title", "netflix_raw_canonical_title"], "canonical"),
        ("series_parent_compact_exact", ["netflix_compact_title", "netflix_raw_compact_title"], "compact"),
    ]


def select_candidates_for_stage(
    row: pd.Series,
    index: CandidateIndex,
    grain: str,
    stage_name: str,
    row_columns: list[str],
    key_type: str,
) -> tuple[pd.DataFrame | None, str | None]:
    row_values = value_options(row, row_columns)
    if not row_values:
        return None, None

    if key_type == "normalized":
        sources = EXACT_SOURCES_BY_STAGE[stage_name]
        for source in sources:
            for value in row_values:
                indices = index.normalized_lookup.get((source, value))
                if indices:
                    return index.title_keys.loc[indices].copy(), value
        return None, None

    if key_type == "canonical":
        for value in row_values:
            indices = index.canonical_lookup.get(value)
            if indices:
                return index.title_keys.loc[indices].copy(), value
        return None, None

    for value in row_values:
        indices = index.compact_lookup.get(value)
        if indices:
            return index.title_keys.loc[indices].copy(), value
    return None, None


def match_unresolved_row(
    row: pd.Series,
    movie_index: CandidateIndex,
    parent_index: CandidateIndex,
) -> tuple[dict[str, object], list[dict[str, object]], str | None]:
    grain = row["netflix_content_grain"]
    if grain == "movie":
        index = movie_index
        entity_type = "movie"
        id_column = "imdb_tconst"
    elif grain == "series_overall":
        index = parent_index
        entity_type = "series_parent"
        id_column = "imdb_parent_tconst"
    else:
        return {
            "match_status": row.get("match_status"),
            "match_method": row.get("match_method"),
            "match_stage": row.get("match_stage"),
            "match_confidence": row.get("match_confidence"),
            "match_notes": row.get("match_notes"),
            "third_pass_match_status": row.get("match_status"),
            "third_pass_match_method": row.get("match_method"),
            "third_pass_match_stage": row.get("match_stage"),
            "third_pass_match_confidence": row.get("match_confidence"),
            "third_pass_match_notes": "Third pass did not apply to this grain.",
            "imdb_match_entity_type": row.get("imdb_match_entity_type", pd.NA),
        }, [], None

    if index.title_keys.empty:
        return {
            "match_status": row.get("match_status"),
            "match_method": row.get("match_method"),
            "match_stage": row.get("match_stage"),
            "match_confidence": row.get("match_confidence"),
            "match_notes": row.get("match_notes"),
            "third_pass_match_status": "unmatched",
            "third_pass_match_method": f"{entity_type}_no_candidates_indexed",
            "third_pass_match_stage": "index_lookup",
            "third_pass_match_confidence": 0.0,
            "third_pass_match_notes": "No incremental IMDb candidates were indexed for this grain.",
            "imdb_match_entity_type": entity_type,
        }, [], None

    review_rows: list[dict[str, object]] = []
    for stage_name, row_columns, key_type in exact_stage_sequence(grain):
        candidates, netflix_key = select_candidates_for_stage(row, index, grain, stage_name, row_columns, key_type)
        if candidates is None or candidates.empty or netflix_key is None:
            continue
        candidates["title_similarity_score"] = 100.0
        match_result, unresolved_snapshot, candidate_frame = resolve_candidate_group(
            row=row,
            candidates=candidates,
            grain=grain,
            entity_type=entity_type,
            id_column=id_column,
            stage_name=stage_name,
            netflix_key_used=netflix_key,
            title_similarity_metric="exact",
        )
        if match_result is not None:
            return match_result, review_rows, stage_name
        if candidate_frame is not None and len(candidate_frame) > 1:
            review_rows.extend(
                build_review_rows(row, candidate_frame, unresolved_snapshot, entity_type, id_column)
            )

    fuzzy_result, unresolved_snapshot, candidate_frame = perform_fuzzy_match(
        row=row,
        index=index,
        grain=grain if grain == "movie" else "series_parent",
        entity_type=entity_type,
        id_column=id_column,
    )
    if fuzzy_result is not None:
        return fuzzy_result, review_rows, fuzzy_result["third_pass_match_stage"]
    if candidate_frame is not None and unresolved_snapshot is not None:
        review_rows.extend(build_review_rows(row, candidate_frame, unresolved_snapshot, entity_type, id_column))

    result = {
        "match_status": row.get("match_status"),
        "match_method": row.get("match_method"),
        "match_stage": row.get("match_stage"),
        "match_confidence": row.get("match_confidence"),
        "match_notes": row.get("match_notes"),
        "third_pass_match_status": "unmatched",
        "third_pass_match_method": f"{entity_type}_no_exact_match",
        "third_pass_match_stage": "unresolved",
        "third_pass_match_confidence": 0.0,
        "third_pass_match_notes": "Third pass found no acceptable exact or fuzzy candidate.",
        "imdb_match_entity_type": entity_type,
    }
    return result, review_rows, None


def build_review_rows(
    row: pd.Series,
    candidate_frame: pd.DataFrame,
    unresolved_snapshot: dict[str, object],
    entity_type: str,
    id_column: str,
) -> list[dict[str, object]]:
    review_rows: list[dict[str, object]] = []
    for rank, (_, candidate) in enumerate(candidate_frame.iterrows(), start=1):
        review_rows.append(
            {
                "netflix_row_id": row["netflix_row_id"],
                "netflix_title_raw": row.get("netflix_title_raw"),
                "netflix_series_title": row.get("netflix_series_title"),
                "netflix_content_grain": row.get("netflix_content_grain"),
                "netflix_season_number": row.get("netflix_season_number"),
                "prior_match_method": row.get("prior_match_method"),
                "third_pass_match_method": unresolved_snapshot.get("third_pass_match_method"),
                "third_pass_match_stage": unresolved_snapshot.get("third_pass_match_stage"),
                "third_pass_match_notes": unresolved_snapshot.get("third_pass_match_notes"),
                "candidate_rank": rank,
                "candidate_match_source": candidate.get("candidate_match_source"),
                "netflix_match_key_used": unresolved_snapshot.get("netflix_match_key_used"),
                "imdb_match_key_used": candidate.get("imdb_match_key_used"),
                "imdb_match_entity_type": entity_type,
                "imdb_candidate_tconst": candidate.get(id_column),
                "imdb_primary_title": candidate.get("imdb_primary_title"),
                "imdb_original_title": candidate.get("imdb_original_title"),
                "imdb_title_type": candidate.get("imdb_title_type"),
                "imdb_start_year": candidate.get("imdb_start_year"),
                "imdb_end_year": candidate.get("imdb_end_year"),
                "imdb_runtime_minutes": candidate.get("imdb_runtime_minutes"),
                "imdb_num_votes": candidate.get("imdb_num_votes"),
                "year_consistency_flag": candidate.get("year_consistency_flag"),
                "year_distance": candidate.get("year_distance"),
                "runtime_distance": candidate.get("runtime_distance"),
                "title_similarity_score": candidate.get("title_similarity_score"),
            }
        )
    return review_rows


def load_optional_override(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, low_memory=False)


def apply_override_row(
    row_mask: pd.Series,
    master: pd.DataFrame,
    entities: pd.DataFrame,
    entity_type: str,
    id_column: str,
    override_value: str,
    override_reason: str | None,
) -> None:
    entity_lookup = entities.set_index(id_column)
    if override_value not in entity_lookup.index:
        return

    entity = entity_lookup.loc[override_value]
    if isinstance(entity, pd.DataFrame):
        entity = entity.iloc[0]

    stage_name = "manual_movie_override" if entity_type == "movie" else "manual_series_parent_override"
    master.loc[row_mask, "match_status"] = "matched"
    master.loc[row_mask, "match_method"] = THIRD_PASS_METHOD_BY_STAGE[stage_name]
    master.loc[row_mask, "match_stage"] = stage_name
    master.loc[row_mask, "match_confidence"] = MATCH_CONFIDENCE_BY_STAGE[stage_name]
    master.loc[row_mask, "match_notes"] = override_reason or "Manual third-pass override applied."
    master.loc[row_mask, "third_pass_match_status"] = "matched"
    master.loc[row_mask, "third_pass_match_method"] = THIRD_PASS_METHOD_BY_STAGE[stage_name]
    master.loc[row_mask, "third_pass_match_stage"] = stage_name
    master.loc[row_mask, "third_pass_match_confidence"] = MATCH_CONFIDENCE_BY_STAGE[stage_name]
    master.loc[row_mask, "third_pass_match_notes"] = override_reason or "Manual third-pass override applied."
    master.loc[row_mask, "ambiguity_resolution_method"] = "manual_override"
    master.loc[row_mask, "candidate_rank"] = 1
    master.loc[row_mask, "candidate_match_source"] = "manual_override"
    master.loc[row_mask, "imdb_match_entity_type"] = entity_type if entity_type == "movie" else "series_parent"
    master.loc[row_mask, "imdb_resolved_tconst"] = override_value
    for column in [
        "imdb_primary_title",
        "imdb_original_title",
        "imdb_title_type",
        "imdb_start_year",
        "imdb_end_year",
        "imdb_runtime_minutes",
        "imdb_genres",
        "imdb_average_rating",
        "imdb_num_votes",
        "imdb_parent_season_count",
        "imdb_normalized_title",
        "imdb_primary_normalized_title",
        "imdb_original_normalized_title",
        "imdb_primary_canonical_title",
        "imdb_original_canonical_title",
    ]:
        if column in entity.index and column in master.columns:
            master.loc[row_mask, column] = entity[column]
    if entity_type == "movie":
        master.loc[row_mask, "imdb_parent_tconst"] = pd.NA
        master.loc[row_mask, "imdb_season_number"] = pd.NA
        master.loc[row_mask, "imdb_season_episode_count"] = pd.NA
    else:
        master.loc[row_mask, "imdb_parent_tconst"] = override_value
        master.loc[row_mask, "imdb_season_number"] = pd.NA
        master.loc[row_mask, "imdb_season_episode_count"] = pd.NA


def apply_manual_overrides(
    master: pd.DataFrame,
    movie_entities: pd.DataFrame,
    parent_entities: pd.DataFrame,
) -> pd.DataFrame:
    updated = master.copy()
    movie_overrides = load_optional_override(MANUAL_MOVIE_OVERRIDE_INPUT)
    parent_overrides = load_optional_override(MANUAL_SERIES_PARENT_OVERRIDE_INPUT)

    if not movie_overrides.empty:
        movie_entities = movie_entities.copy()
        movie_entities["imdb_tconst"] = movie_entities["imdb_tconst"].astype("string")
        for _, override in movie_overrides.iterrows():
            row_mask = updated["netflix_row_id"] == int(override["netflix_row_id"])
            if not row_mask.any():
                continue
            apply_override_row(
                row_mask=row_mask,
                master=updated,
                entities=movie_entities,
                entity_type="movie",
                id_column="imdb_tconst",
                override_value=str(override["override_imdb_tconst"]),
                override_reason=override.get("override_reason"),
            )

    if not parent_overrides.empty:
        parent_entities = parent_entities.copy()
        parent_entities["imdb_parent_tconst"] = parent_entities["imdb_parent_tconst"].astype("string")
        for _, override in parent_overrides.iterrows():
            row_mask = updated["netflix_row_id"] == int(override["netflix_row_id"])
            if not row_mask.any():
                continue
            apply_override_row(
                row_mask=row_mask,
                master=updated,
                entities=parent_entities,
                entity_type="series_parent",
                id_column="imdb_parent_tconst",
                override_value=str(override["override_imdb_parent_tconst"]),
                override_reason=override.get("override_reason"),
            )

    return updated


def initialize_audit_fields(master: pd.DataFrame) -> pd.DataFrame:
    frame = master.copy()
    frame["prior_match_status"] = frame["match_status"]
    frame["prior_match_method"] = frame["match_method"]
    frame["prior_match_stage"] = frame["match_stage"]
    frame["prior_match_confidence"] = frame["match_confidence"]
    frame["netflix_content_grain"] = frame.apply(classify_content_grain, axis=1)
    frame["third_pass_applied"] = frame["match_status"] != "matched"
    frame["third_pass_candidate_grain"] = frame["netflix_content_grain"]
    frame["third_pass_match_status"] = frame["match_status"]
    frame["third_pass_match_method"] = frame["match_method"]
    frame["third_pass_match_stage"] = frame["match_stage"]
    frame["third_pass_match_confidence"] = frame["match_confidence"]
    frame["third_pass_match_notes"] = frame["match_notes"]
    frame["imdb_match_entity_type"] = pd.NA
    season_mask = frame["match_status"].eq("matched") & frame["imdb_season_number"].notna()
    frame.loc[season_mask, "imdb_match_entity_type"] = "series_season"
    frame["movie_runtime_distance"] = pd.NA
    frame["movie_year_distance"] = pd.NA
    frame["movie_title_similarity_score"] = pd.NA
    frame["series_parent_year_distance"] = pd.NA
    frame["series_parent_title_similarity_score"] = pd.NA
    frame["imdb_resolved_tconst"] = frame.get("imdb_parent_tconst", pd.Series(pd.NA, index=frame.index))
    return frame


def update_row_from_result(master: pd.DataFrame, index: int, result: dict[str, object]) -> None:
    for column, value in result.items():
        if column not in master.columns:
            master[column] = pd.NA
        master.at[index, column] = value


def review_frame(master: pd.DataFrame, mask: pd.Series) -> pd.DataFrame:
    available = [column for column in REVIEW_BASE_COLUMNS if column in master.columns]
    aligned_mask = mask.reindex(master.index, fill_value=False)
    return master.loc[aligned_mask, available].copy()


def create_delta_summary(master: pd.DataFrame, baseline: pd.DataFrame) -> pd.DataFrame:
    baseline_matched = int((baseline["match_status"] == "matched").sum())
    final_matched = int((master["match_status"] == "matched").sum())
    baseline_unmatched = baseline[baseline["match_status"] != "matched"].copy()
    final_unmatched = master[master["match_status"] != "matched"].copy()

    newly_matched = master[
        (master["prior_match_status"] != "matched") & (master["match_status"] == "matched")
    ].copy()
    movie_new_matches = newly_matched[newly_matched["imdb_match_entity_type"] == "movie"]
    parent_new_matches = newly_matched[newly_matched["imdb_match_entity_type"] == "series_parent"]

    rows: list[dict[str, object]] = [
        {"metric": "baseline_matched_rows", "value": baseline_matched},
        {"metric": "third_pass_matched_rows", "value": final_matched},
        {"metric": "newly_matched_rows", "value": final_matched - baseline_matched},
        {"metric": "newly_matched_movie_rows", "value": int(len(movie_new_matches))},
        {"metric": "newly_matched_series_overall_rows", "value": int(len(parent_new_matches))},
        {
            "metric": "still_unmatched_movie_rows",
            "value": int(len(final_unmatched[final_unmatched["netflix_content_grain"] == "movie"])),
        },
        {
            "metric": "still_unmatched_series_overall_rows",
            "value": int(
                len(final_unmatched[final_unmatched["netflix_content_grain"] == "series_overall"])
            ),
        },
    ]

    for grain, count in master["netflix_content_grain"].value_counts(dropna=False).items():
        rows.append({"metric": f"grain_count::{grain}", "value": int(count)})
    for entity_type, count in (
        master["imdb_match_entity_type"].fillna("unresolved").value_counts(dropna=False).items()
    ):
        rows.append({"metric": f"imdb_match_entity_type::{entity_type}", "value": int(count)})
    for method, count in master["third_pass_match_method"].fillna("NA").value_counts(dropna=False).items():
        rows.append({"metric": f"third_pass_match_method::{method}", "value": int(count)})

    return pd.DataFrame(rows)


def validate_outputs(master: pd.DataFrame, baseline: pd.DataFrame) -> None:
    if master["netflix_row_id"].duplicated().any():
        raise ValueError("Duplicate Netflix row ids were created during the third pass.")

    existing_matched = baseline["match_status"] == "matched"
    downgraded = master.loc[existing_matched, "match_status"].ne("matched")
    if downgraded.any():
        raise ValueError("Some previously matched rows were downgraded during the third pass.")

    bad_movie = master[
        (master["imdb_match_entity_type"] == "movie")
        & master["imdb_title_type"].notna()
        & (~master["imdb_title_type"].isin(list(MOVIE_TITLE_TYPES)))
    ]
    if not bad_movie.empty:
        raise ValueError("A movie-grain match pointed to a non-movie IMDb entity.")

    bad_parent = master[
        (master["imdb_match_entity_type"] == "series_parent")
        & master["imdb_title_type"].notna()
        & (master["imdb_title_type"].isin(list(MOVIE_TITLE_TYPES)))
    ]
    if not bad_parent.empty:
        raise ValueError("A parent-series match pointed to a movie IMDb entity.")


def run_third_pass() -> pd.DataFrame:
    baseline = normalize_master_types(load_master())
    master = initialize_audit_fields(baseline)

    unresolved = master[master["match_status"] != "matched"].copy()
    log(f"Third-pass unresolved baseline rows: {len(unresolved):,}")
    log(
        "Third-pass unresolved rows by grain: "
        + ", ".join(
            f"{grain}={count:,}"
            for grain, count in unresolved["netflix_content_grain"].value_counts(dropna=False).items()
        )
    )

    movie_entities, movie_title_keys = build_movie_index(master[master["match_status"] != "matched"].copy())
    parent_entities, parent_title_keys = build_parent_series_index(
        master[master["match_status"] != "matched"].copy()
    )
    movie_index = build_candidate_index(movie_entities, movie_title_keys, "imdb_tconst", "movie")
    parent_index = build_candidate_index(
        parent_entities, parent_title_keys, "imdb_parent_tconst", "series_parent"
    )

    ambiguous_review_rows: list[dict[str, object]] = []
    for index, row in master[master["match_status"] != "matched"].iterrows():
        result, review_rows, _ = match_unresolved_row(row, movie_index, parent_index)
        update_row_from_result(master, index, result)
        if review_rows:
            ambiguous_review_rows.extend(review_rows)

    master = apply_manual_overrides(master, movie_entities, parent_entities)
    validate_outputs(master, baseline)

    newly_matched = master[
        (master["prior_match_status"] != "matched") & (master["match_status"] == "matched")
    ].copy()
    movie_review = review_frame(master, newly_matched["imdb_match_entity_type"] == "movie")
    series_parent_review = review_frame(
        master, newly_matched["imdb_match_entity_type"] == "series_parent"
    )
    still_unmatched = master[master["match_status"] != "matched"].copy()
    ambiguous_review = pd.DataFrame(ambiguous_review_rows)
    delta_summary = create_delta_summary(master, baseline)

    ensure_parent(MASTER_V3_OUTPUT)
    master.to_parquet(MASTER_V3_OUTPUT, index=False)
    master.to_csv(MASTER_V3_CSV_OUTPUT, index=False)
    movie_review.to_csv(MOVIE_REVIEW_OUTPUT, index=False)
    series_parent_review.to_csv(SERIES_PARENT_REVIEW_OUTPUT, index=False)
    still_unmatched.to_csv(STILL_UNMATCHED_OUTPUT, index=False)
    ambiguous_review.to_csv(AMBIGUOUS_OUTPUT, index=False)
    delta_summary.to_csv(DELTA_SUMMARY_OUTPUT, index=False)

    log(f"Saved third-pass master parquet: {MASTER_V3_OUTPUT.relative_to(REPO_ROOT).as_posix()}")
    log(f"Saved third-pass master csv: {MASTER_V3_CSV_OUTPUT.relative_to(REPO_ROOT).as_posix()}")
    log(f"Saved movie review rows: {len(movie_review):,}")
    log(f"Saved series-overall review rows: {len(series_parent_review):,}")
    log(f"Saved still-unmatched rows: {len(still_unmatched):,}")
    log(f"Saved ambiguous candidate review rows: {len(ambiguous_review):,}")
    return master


def main() -> None:
    run_third_pass()


if __name__ == "__main__":
    main()
