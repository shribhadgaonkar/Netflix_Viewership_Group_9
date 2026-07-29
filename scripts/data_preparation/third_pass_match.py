from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pandas as pd
from rapidfuzz import fuzz, process


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.pipeline_utils import (  # noqa: E402
    MATCH_SOURCE_PRIORITY,
    SERIES_TITLE_TYPES,
    canonicalize_title,
    coerce_nullable_int,
    compact_title_key,
    ensure_parent,
    interim_dir,
    log,
    normalize_title,
    optional_imdb_inputs,
    parse_numeric_series,
    processed_dir,
    raw_dir,
    require_imdb_inputs,
    year_consistency,
    year_distance,
)


MASTER_PARQUET = processed_dir() / "netflix_imdb_master.parquet"
MASTER_CSV = processed_dir() / "netflix_imdb_master.csv"
UNMATCHED_SERIES_INPUT = processed_dir() / "unmatched_series_rows.csv"
MOVIE_INDEX_OUTPUT = interim_dir() / "imdb_movies.csv"
MOVIE_KEYS_OUTPUT = interim_dir() / "imdb_movie_title_keys.csv"
SERIES_PARENT_OUTPUT = interim_dir() / "imdb_series_parents.csv"
SERIES_PARENT_KEYS_OUTPUT = interim_dir() / "imdb_series_parent_title_keys.csv"
V3_PARQUET_OUTPUT = processed_dir() / "netflix_imdb_master_v3.parquet"
V3_CSV_OUTPUT = processed_dir() / "netflix_imdb_master_v3.csv"
MOVIE_REVIEW_OUTPUT = processed_dir() / "third_pass_movie_matches_review.csv"
SERIES_PARENT_REVIEW_OUTPUT = processed_dir() / "third_pass_series_overall_matches_review.csv"
STILL_UNMATCHED_OUTPUT = processed_dir() / "third_pass_still_unmatched.csv"
AMBIGUOUS_REVIEW_OUTPUT = processed_dir() / "third_pass_ambiguous_candidates.csv"
DELTA_SUMMARY_OUTPUT = processed_dir() / "third_pass_delta_summary.csv"
MANUAL_MOVIE_OVERRIDE_INPUT = REPO_ROOT / "config" / "manual_movie_match_overrides.csv"
MANUAL_PARENT_OVERRIDE_INPUT = REPO_ROOT / "config" / "manual_series_parent_overrides.csv"

MOVIE_TITLE_TYPES = {"movie", "tvMovie"}
FUZZY_SCORE_THRESHOLD = 97.0
FUZZY_MIN_SCORE_GAP = 2.0
MOVIE_RUNTIME_CONFLICT_THRESHOLD = 50.0
SERIES_OVERALL_SKIP_PATTERN = (
    r"(?i)\b(holiday|special|bonus|lyric|lyrics|video|videos|concert|live|interactive|halloween|christmas|movie)\b"
)
PARENT_TTITLE_PREFERENCE_PATTERN = r"(?i)\blimited series\b"
NA_VALUES = ["\\N"]
CHUNK_SIZE = 500_000


def load_master() -> tuple[pd.DataFrame, Path]:
    if MASTER_PARQUET.exists():
        return pd.read_parquet(MASTER_PARQUET), MASTER_PARQUET
    if MASTER_CSV.exists():
        return pd.read_csv(MASTER_CSV, low_memory=False), MASTER_CSV
    raise FileNotFoundError(
        "Could not find the current master dataset. Expected either "
        f"{MASTER_PARQUET} or {MASTER_CSV}."
    )


def file_is_fresh(output_paths: list[Path], source_paths: list[Path]) -> bool:
    if not all(path.exists() for path in output_paths):
        return False
    newest_source = max(path.stat().st_mtime for path in source_paths if path.exists())
    oldest_output = min(path.stat().st_mtime for path in output_paths)
    return oldest_output >= newest_source


def build_target_key_sets(frame: pd.DataFrame) -> dict[str, set[str]]:
    return {
        "normalized": set(
            pd.concat(
                [
                    frame.get("netflix_normalized_title", pd.Series(dtype="string")).astype("string"),
                    frame.get("netflix_raw_normalized_title", pd.Series(dtype="string")).astype("string"),
                ]
            ).dropna().astype(str)
        ),
        "canonical": set(
            pd.concat(
                [
                    frame.get("netflix_canonical_title", pd.Series(dtype="string")).astype("string"),
                    frame.get("netflix_raw_canonical_title", pd.Series(dtype="string")).astype("string"),
                ]
            ).dropna().astype(str)
        ),
        "compact": set(
            pd.concat(
                [
                    frame.get("netflix_compact_title", pd.Series(dtype="string")).astype("string"),
                    frame.get("netflix_raw_compact_title", pd.Series(dtype="string")).astype("string"),
                ]
            ).dropna().astype(str)
        ),
    }


def normalize_movie_basics(chunk: pd.DataFrame) -> pd.DataFrame:
    chunk = chunk.copy()
    chunk["imdb_primary_normalized_title"] = chunk["primaryTitle"].map(normalize_title)
    chunk["imdb_original_normalized_title"] = chunk["originalTitle"].map(normalize_title)
    chunk["imdb_primary_canonical_title"] = chunk["primaryTitle"].map(canonicalize_title)
    chunk["imdb_original_canonical_title"] = chunk["originalTitle"].map(canonicalize_title)
    chunk["imdb_primary_compact_title"] = chunk["primaryTitle"].map(compact_title_key)
    chunk["imdb_original_compact_title"] = chunk["originalTitle"].map(compact_title_key)
    return chunk


def matches_target_sets(frame: pd.DataFrame, target_sets: dict[str, set[str]]) -> pd.Series:
    mask = pd.Series(False, index=frame.index)
    for column in [
        "imdb_primary_normalized_title",
        "imdb_original_normalized_title",
    ]:
        mask = mask | frame[column].astype("string").isin(target_sets["normalized"])
    for column in [
        "imdb_primary_canonical_title",
        "imdb_original_canonical_title",
    ]:
        mask = mask | frame[column].astype("string").isin(target_sets["canonical"])
    for column in [
        "imdb_primary_compact_title",
        "imdb_original_compact_title",
    ]:
        mask = mask | frame[column].astype("string").isin(target_sets["compact"])
    return mask


def build_incremental_movie_index(unresolved_movies: pd.DataFrame, master_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    required_inputs = require_imdb_inputs(raw_dir() / "imdb")
    optional_inputs = optional_imdb_inputs(raw_dir() / "imdb")
    outputs = [MOVIE_INDEX_OUTPUT, MOVIE_KEYS_OUTPUT]
    source_paths = [
        master_path,
        required_inputs["title.basics.tsv.gz"],
        required_inputs["title.ratings.tsv.gz"],
    ]
    if "title.akas.tsv.gz" in optional_inputs:
        source_paths.append(optional_inputs["title.akas.tsv.gz"])

    if file_is_fresh(outputs, source_paths):
        log("Reusing cached incremental IMDb movie index.")
        return (
            pd.read_csv(MOVIE_INDEX_OUTPUT, low_memory=False),
            pd.read_csv(MOVIE_KEYS_OUTPUT, low_memory=False),
        )

    log("Building incremental IMDb movie index for currently unresolved movie rows.")
    target_sets = build_target_key_sets(unresolved_movies)
    basics_path = required_inputs["title.basics.tsv.gz"]
    ratings_path = required_inputs["title.ratings.tsv.gz"]

    collected_basics: list[pd.DataFrame] = []
    direct_movie_ids: set[str] = set()

    usecols = [
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
        usecols=usecols,
        na_values=NA_VALUES,
        keep_default_na=True,
        chunksize=CHUNK_SIZE,
        low_memory=False,
    ):
        chunk = chunk[chunk["titleType"].isin(MOVIE_TITLE_TYPES)].copy()
        if chunk.empty:
            continue
        chunk = normalize_movie_basics(chunk)
        matched = chunk[matches_target_sets(chunk, target_sets)].copy()
        if matched.empty:
            continue
        matched["startYear"] = coerce_nullable_int(matched["startYear"])
        matched["endYear"] = coerce_nullable_int(matched["endYear"])
        matched["runtimeMinutes"] = parse_numeric_series(matched["runtimeMinutes"]).round().astype("Int64")
        collected_basics.append(matched)
        direct_movie_ids.update(matched["tconst"].astype(str))

    aka_rows = pd.DataFrame()
    aka_movie_ids: set[str] = set()
    if "title.akas.tsv.gz" in optional_inputs:
        aka_chunks: list[pd.DataFrame] = []
        for chunk in pd.read_csv(
            optional_inputs["title.akas.tsv.gz"],
            sep="\t",
            compression="gzip",
            na_values=NA_VALUES,
            keep_default_na=True,
            usecols=["titleId", "title", "region", "language", "types", "isOriginalTitle"],
            chunksize=CHUNK_SIZE,
            low_memory=False,
        ):
            chunk = chunk.copy()
            chunk["imdb_aka_normalized_title"] = chunk["title"].map(normalize_title)
            chunk["imdb_aka_canonical_title"] = chunk["title"].map(canonicalize_title)
            chunk["imdb_aka_compact_title"] = chunk["title"].map(compact_title_key)
            mask = (
                chunk["imdb_aka_normalized_title"].astype("string").isin(target_sets["normalized"])
                | chunk["imdb_aka_canonical_title"].astype("string").isin(target_sets["canonical"])
                | chunk["imdb_aka_compact_title"].astype("string").isin(target_sets["compact"])
            )
            matched = chunk[mask].copy()
            if matched.empty:
                continue
            aka_chunks.append(matched)
            aka_movie_ids.update(matched["titleId"].astype(str))

        if aka_chunks:
            aka_rows = pd.concat(aka_chunks, ignore_index=True).drop_duplicates(
                subset=["titleId", "title", "imdb_aka_normalized_title", "imdb_aka_canonical_title", "imdb_aka_compact_title"]
            )

    missing_basics_ids = aka_movie_ids - direct_movie_ids
    if missing_basics_ids:
        for chunk in pd.read_csv(
            basics_path,
            sep="\t",
            compression="gzip",
            usecols=usecols,
            na_values=NA_VALUES,
            keep_default_na=True,
            chunksize=CHUNK_SIZE,
            low_memory=False,
        ):
            chunk = chunk[chunk["titleType"].isin(MOVIE_TITLE_TYPES) & chunk["tconst"].astype(str).isin(missing_basics_ids)].copy()
            if chunk.empty:
                continue
            chunk = normalize_movie_basics(chunk)
            chunk["startYear"] = coerce_nullable_int(chunk["startYear"])
            chunk["endYear"] = coerce_nullable_int(chunk["endYear"])
            chunk["runtimeMinutes"] = parse_numeric_series(chunk["runtimeMinutes"]).round().astype("Int64")
            collected_basics.append(chunk)

    if collected_basics:
        movies = pd.concat(collected_basics, ignore_index=True).drop_duplicates(subset=["tconst"])
    else:
        movies = pd.DataFrame(columns=usecols)

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

    ratings = pd.read_csv(
        ratings_path,
        sep="\t",
        compression="gzip",
        na_values=NA_VALUES,
        keep_default_na=True,
        low_memory=False,
    ).rename(
        columns={
            "tconst": "imdb_tconst",
            "averageRating": "imdb_average_rating",
            "numVotes": "imdb_num_votes",
        }
    )
    ratings["imdb_num_votes"] = coerce_nullable_int(ratings["imdb_num_votes"])
    movies = movies.merge(ratings, how="left", on="imdb_tconst")

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

    if not aka_rows.empty and not movies.empty:
        aka_rows = aka_rows.rename(columns={"titleId": "imdb_tconst", "title": "imdb_candidate_display_title"})
        aka_rows = aka_rows[aka_rows["imdb_tconst"].isin(set(movies["imdb_tconst"].astype(str)))].copy()
        aka_rows["candidate_match_source"] = "aka"
        aka_rows.rename(
            columns={
                "imdb_aka_normalized_title": "imdb_match_key_used",
                "imdb_aka_canonical_title": "imdb_match_key_canonical",
                "imdb_aka_compact_title": "imdb_match_key_compact",
            },
            inplace=True,
        )
        key_frames.append(
            aka_rows[
                [
                    "imdb_tconst",
                    "imdb_candidate_display_title",
                    "imdb_match_key_used",
                    "imdb_match_key_canonical",
                    "imdb_match_key_compact",
                    "candidate_match_source",
                ]
            ]
        )

    movie_title_keys = pd.concat(key_frames, ignore_index=True) if key_frames else pd.DataFrame()
    if not movie_title_keys.empty:
        movie_title_keys = movie_title_keys[
            movie_title_keys["imdb_match_key_used"].notna()
            & movie_title_keys["imdb_match_key_used"].astype("string").str.strip().ne("")
        ].drop_duplicates(
            subset=["imdb_tconst", "candidate_match_source", "imdb_match_key_used"]
        )

    ensure_parent(MOVIE_INDEX_OUTPUT)
    movies.to_csv(MOVIE_INDEX_OUTPUT, index=False)
    ensure_parent(MOVIE_KEYS_OUTPUT)
    movie_title_keys.to_csv(MOVIE_KEYS_OUTPUT, index=False)
    log(f"Saved incremental IMDb movie index: {MOVIE_INDEX_OUTPUT.relative_to(REPO_ROOT).as_posix()}")
    log(f"Saved incremental IMDb movie title keys: {MOVIE_KEYS_OUTPUT.relative_to(REPO_ROOT).as_posix()}")
    return movies, movie_title_keys


def build_incremental_series_parent_index(unresolved_series_overall: pd.DataFrame, master_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    series_seasons_path = interim_dir() / "imdb_series_seasons.csv"
    title_keys_path = interim_dir() / "imdb_title_keys.csv"
    outputs = [SERIES_PARENT_OUTPUT, SERIES_PARENT_KEYS_OUTPUT]
    source_paths = [master_path, series_seasons_path, title_keys_path]

    if file_is_fresh(outputs, source_paths):
        log("Reusing cached incremental IMDb parent-series index.")
        return (
            pd.read_csv(SERIES_PARENT_OUTPUT, low_memory=False),
            pd.read_csv(SERIES_PARENT_KEYS_OUTPUT, low_memory=False),
        )

    log("Building incremental IMDb parent-series index for currently unresolved series-overall rows.")
    target_sets = build_target_key_sets(unresolved_series_overall)
    title_keys = pd.read_csv(title_keys_path, low_memory=False)
    matching_keys = title_keys[
        title_keys["imdb_match_key_used"].astype("string").isin(target_sets["normalized"])
        | title_keys["imdb_match_key_canonical"].astype("string").isin(target_sets["canonical"])
        | title_keys["imdb_match_key_compact"].astype("string").isin(target_sets["compact"])
    ].copy()
    candidate_parent_ids = set(matching_keys["imdb_parent_tconst"].astype(str))

    series_seasons = pd.read_csv(series_seasons_path, low_memory=False)
    parent_columns = [
        "imdb_parent_tconst",
        "imdb_primary_title",
        "imdb_original_title",
        "imdb_normalized_title",
        "imdb_primary_normalized_title",
        "imdb_original_normalized_title",
        "imdb_primary_canonical_title",
        "imdb_original_canonical_title",
        "imdb_aka_normalized_titles",
        "imdb_aka_canonical_titles",
        "imdb_aka_title_count",
        "imdb_title_type",
        "imdb_start_year",
        "imdb_end_year",
        "imdb_genres",
        "imdb_runtime_minutes",
        "imdb_average_rating",
        "imdb_num_votes",
        "imdb_parent_season_count",
    ]
    series_parents = (
        series_seasons[parent_columns]
        .drop_duplicates(subset=["imdb_parent_tconst"])
        .copy()
    )
    series_parents = series_parents[
        series_parents["imdb_parent_tconst"].astype(str).isin(candidate_parent_ids)
    ].copy()
    parent_title_keys = matching_keys[
        matching_keys["imdb_parent_tconst"].astype(str).isin(set(series_parents["imdb_parent_tconst"].astype(str)))
    ].copy()

    ensure_parent(SERIES_PARENT_OUTPUT)
    series_parents.to_csv(SERIES_PARENT_OUTPUT, index=False)
    ensure_parent(SERIES_PARENT_KEYS_OUTPUT)
    parent_title_keys.to_csv(SERIES_PARENT_KEYS_OUTPUT, index=False)
    log(f"Saved incremental IMDb parent-series index: {SERIES_PARENT_OUTPUT.relative_to(REPO_ROOT).as_posix()}")
    log(f"Saved incremental IMDb parent-series title keys: {SERIES_PARENT_KEYS_OUTPUT.relative_to(REPO_ROOT).as_posix()}")
    return series_parents, parent_title_keys


def likely_series_overall(title: Any) -> bool:
    if title is None or pd.isna(title):
        return False
    return pd.isna(pd.Series([title]).astype("string")).iloc[0] is False and not bool(
        pd.Series([str(title)]).str.contains(SERIES_OVERALL_SKIP_PATTERN, regex=True, na=False).iloc[0]
    )


def classify_content_grain(row: pd.Series) -> str:
    if row.get("netflix_format") == "movie":
        return "movie"
    if row.get("netflix_format") == "series" and not pd.isna(row.get("netflix_season_number")):
        return "series_season"
    if row.get("netflix_format") == "series" and likely_series_overall(row.get("netflix_title_raw")):
        return "series_overall"
    if row.get("netflix_format") == "series":
        return "unknown"
    return "unknown"


def seed_resolved_imdb_columns(master: pd.DataFrame) -> pd.DataFrame:
    frame = master.copy()
    frame["imdb_resolved_tconst"] = frame["imdb_parent_tconst"]
    frame["imdb_resolved_primary_title"] = frame["imdb_primary_title"]
    frame["imdb_resolved_original_title"] = frame["imdb_original_title"]
    frame["imdb_resolved_title_type"] = frame["imdb_title_type"]
    frame["imdb_resolved_start_year"] = frame["imdb_start_year"]
    frame["imdb_resolved_end_year"] = frame["imdb_end_year"]
    frame["imdb_resolved_runtime_minutes"] = frame["imdb_runtime_minutes"]
    frame["imdb_resolved_genres"] = frame["imdb_genres"]
    frame["imdb_resolved_average_rating"] = frame["imdb_average_rating"]
    frame["imdb_resolved_num_votes"] = frame["imdb_num_votes"]
    frame["imdb_resolved_parent_season_count"] = frame["imdb_parent_season_count"]
    frame["imdb_resolved_season_number"] = frame["imdb_season_number"]
    frame["imdb_resolved_season_episode_count"] = frame["imdb_season_episode_count"]
    return frame


def build_entity_pool(entity_df: pd.DataFrame, title_keys_df: pd.DataFrame, entity_key: str) -> pd.DataFrame:
    pool = entity_df.merge(title_keys_df, how="left", on=entity_key)
    pool = pool[pool["imdb_match_key_used"].notna()].copy()
    pool["candidate_match_source"] = pool["candidate_match_source"].astype("string")
    pool["source_priority"] = pool["candidate_match_source"].map(MATCH_SOURCE_PRIORITY).fillna(9)
    return pool


def build_lookup(pool: pd.DataFrame, key_col: str, entity_key: str, sources: list[str] | None = None) -> dict[str, list[Any]]:
    subset = pool
    if sources is not None:
        subset = subset[subset["candidate_match_source"].isin(sources)]
    subset = subset[subset[key_col].notna()].copy()
    grouped = subset.groupby(key_col, sort=False)[entity_key].unique()
    return {str(key): list(values) for key, values in grouped.items() if pd.notna(key)}


def get_reference_year(row: pd.Series) -> int | None:
    for column in ["netflix_release_year", "netflix_title_year_hint"]:
        value = row.get(column)
        if value is not None and not pd.isna(value):
            return int(value)
    return None


def get_runtime_distance(row: pd.Series, imdb_runtime: Any) -> float | pd.NA:
    runtime = row.get("netflix_runtime")
    if runtime is None or pd.isna(runtime) or imdb_runtime is None or pd.isna(imdb_runtime):
        return pd.NA
    return float(abs(float(runtime) - float(imdb_runtime)))


def series_parent_type_priority(row: pd.Series, imdb_title_type: Any) -> int:
    title_type = "" if imdb_title_type is None or pd.isna(imdb_title_type) else str(imdb_title_type)
    title_notes = "" if pd.isna(row.get("netflix_title_parse_notes")) else str(row.get("netflix_title_parse_notes"))
    if pd.notna(row.get("netflix_season_label")) and str(row.get("netflix_season_label")) == "limited_series":
        if title_type == "tvMiniSeries":
            return 0
        if title_type == "tvSeries":
            return 1
    elif "limited series" in title_notes.lower():
        if title_type == "tvMiniSeries":
            return 0
    if title_type == "tvSeries":
        return 0
    if title_type == "tvMiniSeries":
        return 1
    return 2


def dedupe_candidates(candidates: pd.DataFrame, entity_key: str) -> pd.DataFrame:
    ordered = candidates.sort_values(["source_priority", "candidate_match_source", entity_key])
    return ordered.drop_duplicates(subset=[entity_key], keep="first").copy()


def compact_values(series: pd.Series) -> str | None:
    values = [str(value) for value in pd.unique(series.dropna()) if str(value).strip()]
    return " | ".join(values) if values else None


def resolve_candidates(
    row: pd.Series,
    candidates: pd.DataFrame,
    entity_kind: str,
    entity_key: str,
    stage_name: str,
    match_method: str,
    netflix_key_used: str,
    title_similarity_metric: str,
) -> tuple[dict[str, object] | None, dict[str, object], pd.DataFrame]:
    candidate_frame = dedupe_candidates(candidates, entity_key)
    reference_year = get_reference_year(row)
    candidate_frame["year_consistency_flag"] = [
        year_consistency(reference_year, start_year, end_year)
        for start_year, end_year in zip(
            candidate_frame["imdb_start_year"], candidate_frame["imdb_end_year"], strict=False
        )
    ]
    candidate_frame["year_distance"] = [
        year_distance(reference_year, start_year) for start_year in candidate_frame["imdb_start_year"]
    ]
    candidate_frame["exact_start_year_match"] = [
        bool(reference_year == int(start_year))
        if reference_year is not None and start_year is not None and not pd.isna(start_year)
        else False
        for start_year in candidate_frame["imdb_start_year"]
    ]
    candidate_frame["runtime_distance"] = [
        get_runtime_distance(row, runtime) for runtime in candidate_frame["imdb_runtime_minutes"]
    ]
    candidate_frame["title_similarity_score"] = pd.to_numeric(
        candidate_frame.get("title_similarity_score", pd.Series(100.0, index=candidate_frame.index)),
        errors="coerce",
    ).fillna(100.0)
    candidate_frame["year_consistency_rank"] = candidate_frame["year_consistency_flag"].map({True: 0, False: 2}).fillna(1)
    candidate_frame["exact_start_year_rank"] = candidate_frame["exact_start_year_match"].map({True: 0, False: 1})
    candidate_frame["year_distance_rank"] = candidate_frame["year_distance"].fillna(9999.0)
    candidate_frame["title_similarity_rank"] = -candidate_frame["title_similarity_score"]
    candidate_frame["votes_rank"] = -candidate_frame["imdb_num_votes"].fillna(-1)
    if entity_kind == "movie":
        candidate_frame["runtime_distance_rank"] = candidate_frame["runtime_distance"].fillna(9999.0)
        sort_columns = [
            "source_priority",
            "year_consistency_rank",
            "exact_start_year_rank",
            "year_distance_rank",
            "runtime_distance_rank",
            "title_similarity_rank",
            "votes_rank",
            entity_key,
        ]
    else:
        candidate_frame["series_type_rank"] = [
            series_parent_type_priority(row, title_type)
            for title_type in candidate_frame["imdb_title_type"]
        ]
        sort_columns = [
            "source_priority",
            "year_consistency_rank",
            "exact_start_year_rank",
            "year_distance_rank",
            "series_type_rank",
            "title_similarity_rank",
            "votes_rank",
            entity_key,
        ]
    candidate_frame.sort_values(sort_columns, inplace=True)

    unresolved = {
        "third_pass_applied": True,
        "third_pass_match_status": "unmatched",
        "third_pass_match_stage": stage_name,
        "third_pass_match_confidence": 0.0,
        "third_pass_match_notes": pd.NA,
        "third_pass_candidate_count": int(len(candidate_frame)),
        "third_pass_candidate_tconsts": compact_values(candidate_frame[entity_key]),
        "third_pass_candidate_titles": compact_values(candidate_frame["imdb_primary_title"]),
        "third_pass_candidate_match_source": compact_values(candidate_frame["candidate_match_source"]),
        "third_pass_netflix_match_key_used": netflix_key_used,
        "third_pass_imdb_match_key_used": pd.NA,
        "third_pass_title_similarity_score": float(candidate_frame["title_similarity_score"].max()),
        "third_pass_year_distance": pd.NA,
        "third_pass_runtime_distance": pd.NA,
        "third_pass_ambiguity_resolution_method": pd.NA,
        "imdb_match_entity_type": pd.NA,
    }

    if candidate_frame.empty:
        unresolved["third_pass_match_method"] = (
            "movie_no_exact_title_match" if entity_kind == "movie" else "series_parent_no_exact_title_match"
        )
        unresolved["third_pass_match_notes"] = (
            "No exact title-key match was found in the incremental movie index."
            if entity_kind == "movie"
            else "No exact title-key match was found in the incremental parent-series index."
        )
        return None, unresolved, candidate_frame

    if reference_year is not None and (candidate_frame["year_consistency_flag"] == False).all():
        unresolved["third_pass_match_method"] = (
            "movie_year_conflict" if entity_kind == "movie" else "series_parent_year_conflict"
        )
        unresolved["third_pass_match_notes"] = (
            f"All candidate years conflicted with Netflix reference year {reference_year}."
        )
        unresolved["third_pass_year_distance"] = float(candidate_frame["year_distance_rank"].min())
        return None, unresolved, candidate_frame

    top = candidate_frame.iloc[0]
    if len(candidate_frame) > 1:
        second = candidate_frame.iloc[1]
        if entity_kind == "movie":
            compare_columns = [
                "source_priority",
                "year_consistency_rank",
                "exact_start_year_rank",
                "year_distance_rank",
                "runtime_distance_rank",
                "title_similarity_rank",
                "votes_rank",
            ]
        else:
            compare_columns = [
                "source_priority",
                "year_consistency_rank",
                "exact_start_year_rank",
                "year_distance_rank",
                "series_type_rank",
                "title_similarity_rank",
                "votes_rank",
            ]
        if tuple(top[column] for column in compare_columns) == tuple(second[column] for column in compare_columns):
            unresolved["third_pass_match_method"] = (
                "movie_ambiguous_exact_match" if entity_kind == "movie" else "series_parent_ambiguous_exact_match"
            )
            unresolved["third_pass_match_notes"] = "Multiple candidates remained tied after deterministic tie-breaks."
            unresolved["third_pass_year_distance"] = top["year_distance"]
            unresolved["third_pass_runtime_distance"] = top["runtime_distance"]
            return None, unresolved, candidate_frame

    if entity_kind == "movie" and pd.notna(top["runtime_distance"]) and float(top["runtime_distance"]) > MOVIE_RUNTIME_CONFLICT_THRESHOLD:
        unresolved["third_pass_match_method"] = "movie_runtime_conflict"
        unresolved["third_pass_match_notes"] = (
            f"Best movie candidate had runtime distance {float(top['runtime_distance']):.1f} minutes, which exceeded the conservative threshold."
        )
        unresolved["third_pass_year_distance"] = top["year_distance"]
        unresolved["third_pass_runtime_distance"] = top["runtime_distance"]
        return None, unresolved, candidate_frame

    result = {
        "third_pass_applied": True,
        "third_pass_match_status": "matched",
        "third_pass_match_method": match_method,
        "third_pass_match_stage": stage_name,
        "third_pass_match_confidence": float(0.99 if title_similarity_metric == "exact" else 0.88),
        "third_pass_match_notes": pd.NA,
        "third_pass_candidate_count": int(len(candidate_frame)),
        "third_pass_candidate_tconsts": compact_values(candidate_frame[entity_key]),
        "third_pass_candidate_titles": compact_values(candidate_frame["imdb_primary_title"]),
        "third_pass_candidate_match_source": top["candidate_match_source"],
        "third_pass_netflix_match_key_used": netflix_key_used,
        "third_pass_imdb_match_key_used": top["imdb_match_key_used"],
        "third_pass_title_similarity_score": top["title_similarity_score"],
        "third_pass_year_distance": top["year_distance"],
        "third_pass_runtime_distance": top["runtime_distance"],
        "third_pass_ambiguity_resolution_method": (
            "single_candidate"
            if len(candidate_frame) == 1
            else "deterministic_tie_break"
        ),
        "imdb_match_entity_type": entity_kind,
        "imdb_candidate_row": top.to_dict(),
    }
    if len(candidate_frame) > 1:
        result["third_pass_match_notes"] = f"Resolved {len(candidate_frame)} candidates using deterministic tie-breaks."
        result["third_pass_match_confidence"] = 0.95 if title_similarity_metric == "exact" else 0.86
    return result, unresolved, candidate_frame


def collect_ambiguous_review_rows(
    row: pd.Series,
    candidate_frame: pd.DataFrame,
    entity_key: str,
    unresolved: dict[str, object],
) -> list[dict[str, object]]:
    review_rows: list[dict[str, object]] = []
    for rank, (_, candidate) in enumerate(candidate_frame.iterrows(), start=1):
        review_rows.append(
            {
                "netflix_row_id": row["netflix_row_id"],
                "netflix_title_raw": row.get("netflix_title_raw"),
                "netflix_content_grain": row.get("netflix_content_grain"),
                "third_pass_match_method": unresolved.get("third_pass_match_method"),
                "third_pass_match_stage": unresolved.get("third_pass_match_stage"),
                "third_pass_match_notes": unresolved.get("third_pass_match_notes"),
                "candidate_rank": rank,
                "candidate_match_source": candidate.get("candidate_match_source"),
                "netflix_match_key_used": unresolved.get("third_pass_netflix_match_key_used"),
                "imdb_match_key_used": candidate.get("imdb_match_key_used"),
                "imdb_candidate_tconst": candidate.get(entity_key),
                "imdb_primary_title": candidate.get("imdb_primary_title"),
                "imdb_original_title": candidate.get("imdb_original_title"),
                "imdb_title_type": candidate.get("imdb_title_type"),
                "imdb_start_year": candidate.get("imdb_start_year"),
                "imdb_end_year": candidate.get("imdb_end_year"),
                "imdb_runtime_minutes": candidate.get("imdb_runtime_minutes"),
                "imdb_num_votes": candidate.get("imdb_num_votes"),
                "title_similarity_score": candidate.get("title_similarity_score"),
                "year_distance": candidate.get("year_distance"),
                "runtime_distance": candidate.get("runtime_distance"),
            }
        )
    return review_rows


def run_fuzzy_match(
    row: pd.Series,
    pool: pd.DataFrame,
    entity_kind: str,
    entity_key: str,
) -> tuple[dict[str, object] | None, dict[str, object] | None, pd.DataFrame]:
    netflix_key = row.get("netflix_canonical_title")
    if netflix_key is None or pd.isna(netflix_key) or not str(netflix_key).strip():
        return None, None, pd.DataFrame()

    key_text = str(netflix_key)
    prefix = key_text[:3]
    subset = pool[pool["imdb_match_key_canonical"].notna()].copy()
    subset = subset[
        subset["imdb_match_key_canonical"].astype("string").str.startswith(prefix, na=False)
        | subset["imdb_match_key_canonical"].astype("string").str[:1].eq(key_text[:1])
    ].copy()
    if subset.empty:
        return None, None, pd.DataFrame()

    subset = dedupe_candidates(subset, entity_key)
    choice_map = subset[["imdb_match_key_canonical"]].drop_duplicates()["imdb_match_key_canonical"].astype(str).tolist()
    matches = process.extract(key_text, choice_map, scorer=fuzz.ratio, limit=5)
    if not matches:
        return None, None, pd.DataFrame()

    top_key, top_score, _ = matches[0]
    second_score = matches[1][1] if len(matches) > 1 else 0.0
    if top_score < FUZZY_SCORE_THRESHOLD or (top_score - second_score) < FUZZY_MIN_SCORE_GAP:
        return None, None, pd.DataFrame()

    fuzzy_candidates = subset[subset["imdb_match_key_canonical"] == top_key].copy()
    fuzzy_candidates["title_similarity_score"] = float(top_score)
    stage = f"third_pass_{entity_kind}_fuzzy"
    method = "movie_fuzzy_title" if entity_kind == "movie" else "series_parent_fuzzy_title"
    return resolve_candidates(
        row=row,
        candidates=fuzzy_candidates,
        entity_kind=entity_kind,
        entity_key=entity_key,
        stage_name=stage,
        match_method=method,
        netflix_key_used=key_text,
        title_similarity_metric="fuzz_ratio",
    )


def try_entity_match(
    row: pd.Series,
    pool: pd.DataFrame,
    lookups: dict[str, dict[str, list[Any]]],
    entity_kind: str,
    entity_key: str,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    review_rows: list[dict[str, object]] = []
    stage_defs = [
        (
            "normalized_primary",
            "netflix_normalized_title",
            f"third_pass_{entity_kind}_primary_exact",
            f"{entity_kind}_exact_primary_title",
        ),
        (
            "normalized_original",
            "netflix_normalized_title",
            f"third_pass_{entity_kind}_original_exact",
            f"{entity_kind}_exact_original_title",
        ),
        (
            "normalized_aka",
            "netflix_normalized_title",
            f"third_pass_{entity_kind}_aka_exact",
            f"{entity_kind}_exact_aka_title",
        ),
        (
            "canonical_all",
            "netflix_canonical_title",
            f"third_pass_{entity_kind}_canonical_exact",
            f"{entity_kind}_exact_canonical_title",
        ),
        (
            "compact_all",
            "netflix_compact_title",
            f"third_pass_{entity_kind}_compact_exact",
            f"{entity_kind}_exact_compact_title",
        ),
    ]

    best_unresolved: dict[str, object] | None = None
    for lookup_name, netflix_key_col, stage_name, match_method in stage_defs:
        netflix_key = row.get(netflix_key_col)
        if netflix_key is None or pd.isna(netflix_key) or not str(netflix_key).strip():
            continue
        ids = lookups[lookup_name].get(str(netflix_key), [])
        if not ids:
            continue
        candidates = pool[pool[entity_key].isin(ids)].copy()
        candidates["title_similarity_score"] = 100.0
        match_result, unresolved, candidate_frame = resolve_candidates(
            row=row,
            candidates=candidates,
            entity_kind=entity_kind,
            entity_key=entity_key,
            stage_name=stage_name,
            match_method=match_method,
            netflix_key_used=str(netflix_key),
            title_similarity_metric="exact",
        )
        if match_result is not None:
            return match_result, review_rows
        best_unresolved = unresolved
        review_rows.extend(collect_ambiguous_review_rows(row, candidate_frame, entity_key, unresolved))
        break

    fuzzy_result, fuzzy_unresolved, fuzzy_frame = run_fuzzy_match(row, pool, entity_kind, entity_key)
    if fuzzy_result is not None:
        return fuzzy_result, review_rows
    if fuzzy_unresolved is not None:
        review_rows.extend(collect_ambiguous_review_rows(row, fuzzy_frame, entity_key, fuzzy_unresolved))
        best_unresolved = fuzzy_unresolved

    if best_unresolved is not None:
        return best_unresolved, review_rows

    return (
        {
            "third_pass_applied": True,
            "third_pass_match_status": "unmatched",
            "third_pass_match_method": (
                "movie_no_exact_title_match" if entity_kind == "movie" else "series_parent_no_exact_title_match"
            ),
            "third_pass_match_stage": f"third_pass_{entity_kind}_unresolved",
            "third_pass_match_confidence": 0.0,
            "third_pass_match_notes": (
                "No candidate title key matched in the incremental movie index."
                if entity_kind == "movie"
                else "No candidate title key matched in the incremental parent-series index."
            ),
            "third_pass_candidate_count": 0,
            "third_pass_candidate_tconsts": pd.NA,
            "third_pass_candidate_titles": pd.NA,
            "third_pass_candidate_match_source": pd.NA,
            "third_pass_netflix_match_key_used": pd.NA,
            "third_pass_imdb_match_key_used": pd.NA,
            "third_pass_title_similarity_score": pd.NA,
            "third_pass_year_distance": pd.NA,
            "third_pass_runtime_distance": pd.NA,
            "third_pass_ambiguity_resolution_method": pd.NA,
            "imdb_match_entity_type": pd.NA,
        },
        review_rows,
    )


def load_movie_override_frame() -> pd.DataFrame:
    if not MANUAL_MOVIE_OVERRIDE_INPUT.exists():
        return pd.DataFrame()
    return pd.read_csv(MANUAL_MOVIE_OVERRIDE_INPUT, low_memory=False)


def load_parent_override_frame() -> pd.DataFrame:
    if not MANUAL_PARENT_OVERRIDE_INPUT.exists():
        return pd.DataFrame()
    return pd.read_csv(MANUAL_PARENT_OVERRIDE_INPUT, low_memory=False)


def apply_override_match(
    frame: pd.DataFrame,
    row_index: Any,
    entity_row: pd.Series,
    entity_kind: str,
    override_reason: str,
) -> None:
    frame.at[row_index, "third_pass_applied"] = True
    frame.at[row_index, "third_pass_match_status"] = "matched"
    frame.at[row_index, "third_pass_match_method"] = f"{entity_kind}_manual_override"
    frame.at[row_index, "third_pass_match_stage"] = "third_pass_manual_override"
    frame.at[row_index, "third_pass_match_confidence"] = 1.0
    frame.at[row_index, "third_pass_match_notes"] = override_reason
    frame.at[row_index, "third_pass_candidate_count"] = 1
    frame.at[row_index, "third_pass_candidate_tconsts"] = str(entity_row["imdb_tconst"] if entity_kind == "movie" else entity_row["imdb_parent_tconst"])
    frame.at[row_index, "third_pass_candidate_titles"] = entity_row["imdb_primary_title"]
    frame.at[row_index, "third_pass_candidate_match_source"] = "manual_override"
    frame.at[row_index, "third_pass_ambiguity_resolution_method"] = "manual_override"
    frame.at[row_index, "imdb_match_entity_type"] = "movie" if entity_kind == "movie" else "series_parent"


def apply_movie_match(frame: pd.DataFrame, row_index: Any, match_result: dict[str, object]) -> None:
    candidate = match_result["imdb_candidate_row"]
    frame.at[row_index, "match_status"] = "matched"
    frame.at[row_index, "match_method"] = match_result["third_pass_match_method"]
    frame.at[row_index, "match_stage"] = match_result["third_pass_match_stage"]
    frame.at[row_index, "match_confidence"] = match_result["third_pass_match_confidence"]
    frame.at[row_index, "match_notes"] = match_result["third_pass_match_notes"]
    frame.at[row_index, "candidate_match_source"] = match_result["third_pass_candidate_match_source"]
    frame.at[row_index, "netflix_match_key_used"] = match_result["third_pass_netflix_match_key_used"]
    frame.at[row_index, "imdb_match_key_used"] = match_result["third_pass_imdb_match_key_used"]
    frame.at[row_index, "title_similarity_score"] = match_result["third_pass_title_similarity_score"]
    frame.at[row_index, "year_distance"] = match_result["third_pass_year_distance"]
    frame.at[row_index, "candidate_rank"] = 1
    frame.at[row_index, "ambiguity_resolution_method"] = match_result["third_pass_ambiguity_resolution_method"]
    frame.at[row_index, "candidate_imdb_count"] = match_result["third_pass_candidate_count"]
    frame.at[row_index, "candidate_imdb_parent_tconsts"] = match_result["third_pass_candidate_tconsts"]
    frame.at[row_index, "candidate_imdb_primary_titles"] = match_result["third_pass_candidate_titles"]
    frame.at[row_index, "imdb_match_entity_type"] = "movie"
    frame.at[row_index, "imdb_primary_title"] = candidate["imdb_primary_title"]
    frame.at[row_index, "imdb_original_title"] = candidate["imdb_original_title"]
    frame.at[row_index, "imdb_title_type"] = candidate["imdb_title_type"]
    frame.at[row_index, "imdb_start_year"] = candidate["imdb_start_year"]
    frame.at[row_index, "imdb_end_year"] = candidate["imdb_end_year"]
    frame.at[row_index, "imdb_genres"] = candidate["imdb_genres"]
    frame.at[row_index, "imdb_runtime_minutes"] = candidate["imdb_runtime_minutes"]
    frame.at[row_index, "imdb_average_rating"] = candidate["imdb_average_rating"]
    frame.at[row_index, "imdb_num_votes"] = candidate["imdb_num_votes"]
    frame.at[row_index, "imdb_resolved_tconst"] = candidate["imdb_tconst"]
    frame.at[row_index, "imdb_resolved_primary_title"] = candidate["imdb_primary_title"]
    frame.at[row_index, "imdb_resolved_original_title"] = candidate["imdb_original_title"]
    frame.at[row_index, "imdb_resolved_title_type"] = candidate["imdb_title_type"]
    frame.at[row_index, "imdb_resolved_start_year"] = candidate["imdb_start_year"]
    frame.at[row_index, "imdb_resolved_end_year"] = candidate["imdb_end_year"]
    frame.at[row_index, "imdb_resolved_runtime_minutes"] = candidate["imdb_runtime_minutes"]
    frame.at[row_index, "imdb_resolved_genres"] = candidate["imdb_genres"]
    frame.at[row_index, "imdb_resolved_average_rating"] = candidate["imdb_average_rating"]
    frame.at[row_index, "imdb_resolved_num_votes"] = candidate["imdb_num_votes"]


def apply_series_parent_match(frame: pd.DataFrame, row_index: Any, match_result: dict[str, object]) -> None:
    candidate = match_result["imdb_candidate_row"]
    frame.at[row_index, "match_status"] = "matched"
    frame.at[row_index, "match_method"] = match_result["third_pass_match_method"]
    frame.at[row_index, "match_stage"] = match_result["third_pass_match_stage"]
    frame.at[row_index, "match_confidence"] = match_result["third_pass_match_confidence"]
    frame.at[row_index, "match_notes"] = match_result["third_pass_match_notes"]
    frame.at[row_index, "candidate_match_source"] = match_result["third_pass_candidate_match_source"]
    frame.at[row_index, "netflix_match_key_used"] = match_result["third_pass_netflix_match_key_used"]
    frame.at[row_index, "imdb_match_key_used"] = match_result["third_pass_imdb_match_key_used"]
    frame.at[row_index, "title_similarity_score"] = match_result["third_pass_title_similarity_score"]
    frame.at[row_index, "year_distance"] = match_result["third_pass_year_distance"]
    frame.at[row_index, "candidate_rank"] = 1
    frame.at[row_index, "ambiguity_resolution_method"] = match_result["third_pass_ambiguity_resolution_method"]
    frame.at[row_index, "candidate_imdb_count"] = match_result["third_pass_candidate_count"]
    frame.at[row_index, "candidate_imdb_parent_tconsts"] = match_result["third_pass_candidate_tconsts"]
    frame.at[row_index, "candidate_imdb_primary_titles"] = match_result["third_pass_candidate_titles"]
    frame.at[row_index, "imdb_match_entity_type"] = "series_parent"
    frame.at[row_index, "imdb_parent_tconst"] = candidate["imdb_parent_tconst"]
    frame.at[row_index, "imdb_primary_title"] = candidate["imdb_primary_title"]
    frame.at[row_index, "imdb_original_title"] = candidate["imdb_original_title"]
    frame.at[row_index, "imdb_normalized_title"] = candidate["imdb_normalized_title"]
    frame.at[row_index, "imdb_primary_normalized_title"] = candidate["imdb_primary_normalized_title"]
    frame.at[row_index, "imdb_original_normalized_title"] = candidate["imdb_original_normalized_title"]
    frame.at[row_index, "imdb_primary_canonical_title"] = candidate["imdb_primary_canonical_title"]
    frame.at[row_index, "imdb_original_canonical_title"] = candidate["imdb_original_canonical_title"]
    frame.at[row_index, "imdb_aka_normalized_titles"] = candidate["imdb_aka_normalized_titles"]
    frame.at[row_index, "imdb_aka_canonical_titles"] = candidate["imdb_aka_canonical_titles"]
    frame.at[row_index, "imdb_aka_title_count"] = candidate["imdb_aka_title_count"]
    frame.at[row_index, "imdb_title_type"] = candidate["imdb_title_type"]
    frame.at[row_index, "imdb_start_year"] = candidate["imdb_start_year"]
    frame.at[row_index, "imdb_end_year"] = candidate["imdb_end_year"]
    frame.at[row_index, "imdb_genres"] = candidate["imdb_genres"]
    frame.at[row_index, "imdb_runtime_minutes"] = candidate["imdb_runtime_minutes"]
    frame.at[row_index, "imdb_average_rating"] = candidate["imdb_average_rating"]
    frame.at[row_index, "imdb_num_votes"] = candidate["imdb_num_votes"]
    frame.at[row_index, "imdb_parent_season_count"] = candidate["imdb_parent_season_count"]
    frame.at[row_index, "imdb_resolved_tconst"] = candidate["imdb_parent_tconst"]
    frame.at[row_index, "imdb_resolved_primary_title"] = candidate["imdb_primary_title"]
    frame.at[row_index, "imdb_resolved_original_title"] = candidate["imdb_original_title"]
    frame.at[row_index, "imdb_resolved_title_type"] = candidate["imdb_title_type"]
    frame.at[row_index, "imdb_resolved_start_year"] = candidate["imdb_start_year"]
    frame.at[row_index, "imdb_resolved_end_year"] = candidate["imdb_end_year"]
    frame.at[row_index, "imdb_resolved_runtime_minutes"] = candidate["imdb_runtime_minutes"]
    frame.at[row_index, "imdb_resolved_genres"] = candidate["imdb_genres"]
    frame.at[row_index, "imdb_resolved_average_rating"] = candidate["imdb_average_rating"]
    frame.at[row_index, "imdb_resolved_num_votes"] = candidate["imdb_num_votes"]
    frame.at[row_index, "imdb_resolved_parent_season_count"] = candidate["imdb_parent_season_count"]


def initialize_third_pass_columns(master: pd.DataFrame) -> pd.DataFrame:
    frame = master.copy()
    frame["prior_match_status"] = frame["match_status"]
    frame["prior_match_method"] = frame["match_method"]
    frame["prior_match_stage"] = frame["match_stage"]
    frame["prior_match_confidence"] = frame["match_confidence"]
    frame["netflix_content_grain"] = frame.apply(classify_content_grain, axis=1)
    frame = seed_resolved_imdb_columns(frame)
    frame["imdb_match_entity_type"] = pd.Series(pd.NA, index=frame.index, dtype="string")
    frame.loc[frame["match_status"] == "matched", "imdb_match_entity_type"] = "series_season"
    frame["third_pass_applied"] = False
    frame["third_pass_candidate_grain"] = frame["netflix_content_grain"]
    frame["third_pass_match_status"] = frame["match_status"]
    frame["third_pass_match_method"] = "preserved_existing_match"
    frame.loc[frame["match_status"] != "matched", "third_pass_match_method"] = "pending_third_pass"
    frame["third_pass_match_stage"] = "not_applied"
    frame["third_pass_match_confidence"] = frame["match_confidence"]
    frame["third_pass_match_notes"] = pd.Series(pd.NA, index=frame.index, dtype="string")
    frame.loc[frame["match_status"] == "matched", "third_pass_match_notes"] = (
        "Third pass not applied because the row was already matched."
    )
    frame["third_pass_candidate_count"] = frame.get("candidate_imdb_count", pd.Series(pd.NA, index=frame.index))
    frame["third_pass_candidate_tconsts"] = frame.get("candidate_imdb_parent_tconsts", pd.Series(pd.NA, index=frame.index))
    frame["third_pass_candidate_titles"] = frame.get("candidate_imdb_primary_titles", pd.Series(pd.NA, index=frame.index))
    frame["third_pass_candidate_match_source"] = frame.get("candidate_match_source", pd.Series(pd.NA, index=frame.index))
    frame["third_pass_netflix_match_key_used"] = frame.get("netflix_match_key_used", pd.Series(pd.NA, index=frame.index))
    frame["third_pass_imdb_match_key_used"] = frame.get("imdb_match_key_used", pd.Series(pd.NA, index=frame.index))
    frame["third_pass_title_similarity_score"] = frame.get("title_similarity_score", pd.Series(pd.NA, index=frame.index))
    frame["third_pass_year_distance"] = frame.get("year_distance", pd.Series(pd.NA, index=frame.index))
    frame["third_pass_runtime_distance"] = pd.Series(pd.NA, index=frame.index)
    frame["third_pass_ambiguity_resolution_method"] = frame.get(
        "ambiguity_resolution_method", pd.Series(pd.NA, index=frame.index)
    )
    return frame


def apply_unresolved_defaults(frame: pd.DataFrame, row_index: Any, grain: str) -> None:
    frame.at[row_index, "third_pass_applied"] = True
    frame.at[row_index, "third_pass_match_status"] = frame.at[row_index, "match_status"]
    frame.at[row_index, "third_pass_match_stage"] = "third_pass_skipped"
    if grain == "series_season":
        frame.at[row_index, "third_pass_match_method"] = "preserved_series_season_outcome"
        frame.at[row_index, "third_pass_match_notes"] = (
            "Third pass preserved the second-pass series-season outcome without re-running season-level matching."
        )
    else:
        frame.at[row_index, "third_pass_match_method"] = "third_pass_skipped_unknown_grain"
        frame.at[row_index, "third_pass_match_notes"] = (
            "Third pass did not apply movie or parent-series matching because the content grain remained uncertain."
        )


def create_delta_summary(
    baseline: pd.DataFrame,
    final_frame: pd.DataFrame,
) -> pd.DataFrame:
    baseline_matched = int((baseline["match_status"] == "matched").sum())
    final_matched = int((final_frame["match_status"] == "matched").sum())
    newly_matched = final_frame[
        (final_frame["prior_match_status"] != "matched") & (final_frame["match_status"] == "matched")
    ].copy()
    rows = [
        {"section": "headline", "metric": "baseline_matched_rows", "value": baseline_matched},
        {"section": "headline", "metric": "third_pass_matched_rows", "value": final_matched},
        {"section": "headline", "metric": "newly_matched_rows", "value": int(len(newly_matched))},
        {
            "section": "headline",
            "metric": "newly_matched_movie_rows",
            "value": int((newly_matched["netflix_content_grain"] == "movie").sum()),
        },
        {
            "section": "headline",
            "metric": "newly_matched_series_overall_rows",
            "value": int((newly_matched["netflix_content_grain"] == "series_overall").sum()),
        },
        {
            "section": "headline",
            "metric": "still_unmatched_movie_rows",
            "value": int(
                (
                    (final_frame["netflix_content_grain"] == "movie")
                    & (final_frame["match_status"] != "matched")
                ).sum()
            ),
        },
        {
            "section": "headline",
            "metric": "still_unmatched_series_overall_rows",
            "value": int(
                (
                    (final_frame["netflix_content_grain"] == "series_overall")
                    & (final_frame["match_status"] != "matched")
                ).sum()
            ),
        },
        {
            "section": "validation",
            "metric": "prior_matched_rows_downgraded",
            "value": int(
                (
                    (final_frame["prior_match_status"] == "matched")
                    & (final_frame["match_status"] != "matched")
                ).sum()
            ),
        },
        {
            "section": "validation",
            "metric": "duplicate_netflix_row_ids",
            "value": int(final_frame["netflix_row_id"].duplicated().sum()),
        },
        {
            "section": "validation",
            "metric": "matched_movie_rows_with_non_movie_imdb_type",
            "value": int(
                (
                    (final_frame["imdb_match_entity_type"] == "movie")
                    & ~final_frame["imdb_resolved_title_type"].astype("string").isin(list(MOVIE_TITLE_TYPES))
                ).sum()
            ),
        },
        {
            "section": "validation",
            "metric": "matched_series_parent_rows_with_movie_imdb_type",
            "value": int(
                (
                    (final_frame["imdb_match_entity_type"] == "series_parent")
                    & final_frame["imdb_resolved_title_type"].astype("string").isin(list(MOVIE_TITLE_TYPES))
                ).sum()
            ),
        },
    ]

    for grain, count in final_frame["netflix_content_grain"].value_counts(dropna=False).items():
        rows.append({"section": "grain_counts", "metric": str(grain), "value": int(count)})
    for entity_type, count in final_frame["imdb_match_entity_type"].fillna("<NA>").value_counts().items():
        rows.append({"section": "entity_type_counts", "metric": str(entity_type), "value": int(count)})
    applied = final_frame[final_frame["third_pass_applied"] == True].copy()
    for method, count in applied["third_pass_match_method"].value_counts(dropna=False).items():
        rows.append({"section": "third_pass_method_counts", "metric": str(method), "value": int(count)})

    return pd.DataFrame(rows)


def merge_third_pass() -> pd.DataFrame:
    baseline_master, master_path = load_master()
    working = initialize_third_pass_columns(baseline_master)

    unresolved = working[working["prior_match_status"] != "matched"].copy()
    unresolved_movies = unresolved[unresolved["netflix_content_grain"] == "movie"].copy()
    unresolved_series_overall = unresolved[unresolved["netflix_content_grain"] == "series_overall"].copy()

    movie_index, movie_keys = build_incremental_movie_index(unresolved_movies, master_path)
    series_parents, series_parent_keys = build_incremental_series_parent_index(
        unresolved_series_overall, master_path
    )

    movie_pool = build_entity_pool(movie_index, movie_keys, "imdb_tconst")
    movie_lookups = {
        "normalized_primary": build_lookup(movie_pool, "imdb_match_key_used", "imdb_tconst", ["primary"]),
        "normalized_original": build_lookup(movie_pool, "imdb_match_key_used", "imdb_tconst", ["original"]),
        "normalized_aka": build_lookup(movie_pool, "imdb_match_key_used", "imdb_tconst", ["aka"]),
        "canonical_all": build_lookup(movie_pool, "imdb_match_key_canonical", "imdb_tconst"),
        "compact_all": build_lookup(movie_pool, "imdb_match_key_compact", "imdb_tconst"),
    }

    series_parent_pool = build_entity_pool(series_parents, series_parent_keys, "imdb_parent_tconst")
    series_parent_lookups = {
        "normalized_primary": build_lookup(
            series_parent_pool, "imdb_match_key_used", "imdb_parent_tconst", ["primary"]
        ),
        "normalized_original": build_lookup(
            series_parent_pool, "imdb_match_key_used", "imdb_parent_tconst", ["original"]
        ),
        "normalized_aka": build_lookup(
            series_parent_pool, "imdb_match_key_used", "imdb_parent_tconst", ["aka"]
        ),
        "canonical_all": build_lookup(series_parent_pool, "imdb_match_key_canonical", "imdb_parent_tconst"),
        "compact_all": build_lookup(series_parent_pool, "imdb_match_key_compact", "imdb_parent_tconst"),
    }

    movie_override_frame = load_movie_override_frame()
    parent_override_frame = load_parent_override_frame()
    movie_lookup_by_tconst = movie_index.set_index("imdb_tconst") if not movie_index.empty else pd.DataFrame()
    parent_lookup_by_tconst = series_parents.set_index("imdb_parent_tconst") if not series_parents.empty else pd.DataFrame()

    movie_review_rows: list[dict[str, object]] = []
    parent_review_rows: list[dict[str, object]] = []
    ambiguous_review_rows: list[dict[str, object]] = []

    for row_index, row in unresolved.iterrows():
        grain = row["netflix_content_grain"]
        working.at[row_index, "third_pass_applied"] = True
        working.at[row_index, "third_pass_candidate_grain"] = grain

        if grain == "movie":
            override_applied = False
            if not movie_override_frame.empty and "netflix_row_id" in movie_override_frame.columns:
                override_rows = movie_override_frame[movie_override_frame["netflix_row_id"] == row["netflix_row_id"]]
                if not override_rows.empty:
                    override = override_rows.iloc[0]
                    imdb_tconst = str(override["override_imdb_tconst"])
                    if imdb_tconst in movie_lookup_by_tconst.index:
                        entity_row = movie_lookup_by_tconst.loc[imdb_tconst]
                        if isinstance(entity_row, pd.DataFrame):
                            entity_row = entity_row.iloc[0]
                        apply_override_match(
                            working,
                            row_index,
                            entity_row,
                            entity_kind="movie",
                            override_reason=str(override.get("override_reason", "Manual movie override applied.")),
                        )
                        override_applied = True
                        working.at[row_index, "match_status"] = "matched"
                        working.at[row_index, "match_method"] = "movie_manual_override"
                        working.at[row_index, "match_stage"] = "third_pass_manual_override"
                        working.at[row_index, "match_confidence"] = 1.0
                        working.at[row_index, "match_notes"] = working.at[row_index, "third_pass_match_notes"]
                        working.at[row_index, "imdb_resolved_tconst"] = imdb_tconst
                        movie_review_rows.append(working.loc[row_index].to_dict())
            if override_applied:
                continue

            match_result, review_rows = try_entity_match(
                row,
                movie_pool,
                movie_lookups,
                entity_kind="movie",
                entity_key="imdb_tconst",
            )
            for key, value in match_result.items():
                if key == "imdb_candidate_row":
                    continue
                working.at[row_index, key] = value
            if match_result["third_pass_match_status"] == "matched":
                apply_movie_match(working, row_index, match_result)
                movie_review_rows.append(working.loc[row_index].to_dict())
            else:
                ambiguous_review_rows.extend(review_rows)
            continue

        if grain == "series_overall":
            override_applied = False
            if not parent_override_frame.empty and "netflix_row_id" in parent_override_frame.columns:
                override_rows = parent_override_frame[
                    parent_override_frame["netflix_row_id"] == row["netflix_row_id"]
                ]
                if not override_rows.empty:
                    override = override_rows.iloc[0]
                    imdb_parent_tconst = str(override["override_imdb_parent_tconst"])
                    if imdb_parent_tconst in parent_lookup_by_tconst.index:
                        entity_row = parent_lookup_by_tconst.loc[imdb_parent_tconst]
                        if isinstance(entity_row, pd.DataFrame):
                            entity_row = entity_row.iloc[0]
                        apply_override_match(
                            working,
                            row_index,
                            entity_row,
                            entity_kind="series_parent",
                            override_reason=str(override.get("override_reason", "Manual parent-series override applied.")),
                        )
                        override_applied = True
                        working.at[row_index, "match_status"] = "matched"
                        working.at[row_index, "match_method"] = "series_parent_manual_override"
                        working.at[row_index, "match_stage"] = "third_pass_manual_override"
                        working.at[row_index, "match_confidence"] = 1.0
                        working.at[row_index, "match_notes"] = working.at[row_index, "third_pass_match_notes"]
                        apply_series_parent_match(
                            working,
                            row_index,
                            {
                                "third_pass_match_method": "series_parent_manual_override",
                                "third_pass_match_stage": "third_pass_manual_override",
                                "third_pass_match_confidence": 1.0,
                                "third_pass_match_notes": working.at[row_index, "third_pass_match_notes"],
                                "third_pass_candidate_count": 1,
                                "third_pass_candidate_tconsts": imdb_parent_tconst,
                                "third_pass_candidate_titles": entity_row["imdb_primary_title"],
                                "third_pass_candidate_match_source": "manual_override",
                                "third_pass_netflix_match_key_used": pd.NA,
                                "third_pass_imdb_match_key_used": pd.NA,
                                "third_pass_title_similarity_score": 100.0,
                                "third_pass_year_distance": pd.NA,
                                "third_pass_runtime_distance": pd.NA,
                                "third_pass_ambiguity_resolution_method": "manual_override",
                                "imdb_candidate_row": entity_row.to_dict(),
                            },
                        )
                        parent_review_rows.append(working.loc[row_index].to_dict())
            if override_applied:
                continue

            match_result, review_rows = try_entity_match(
                row,
                series_parent_pool,
                series_parent_lookups,
                entity_kind="series_parent",
                entity_key="imdb_parent_tconst",
            )
            for key, value in match_result.items():
                if key == "imdb_candidate_row":
                    continue
                working.at[row_index, key] = value
            if match_result["third_pass_match_status"] == "matched":
                apply_series_parent_match(working, row_index, match_result)
                parent_review_rows.append(working.loc[row_index].to_dict())
            else:
                ambiguous_review_rows.extend(review_rows)
            continue

        apply_unresolved_defaults(working, row_index, grain)

    delta_summary = create_delta_summary(baseline_master, working)
    still_unmatched = working[working["match_status"] != "matched"].copy()
    movie_review = pd.DataFrame(movie_review_rows)
    parent_review = pd.DataFrame(parent_review_rows)
    ambiguous_review = pd.DataFrame(ambiguous_review_rows)

    ensure_parent(V3_PARQUET_OUTPUT)
    working.to_parquet(V3_PARQUET_OUTPUT, index=False)
    working.to_csv(V3_CSV_OUTPUT, index=False)
    movie_review.to_csv(MOVIE_REVIEW_OUTPUT, index=False)
    parent_review.to_csv(SERIES_PARENT_REVIEW_OUTPUT, index=False)
    still_unmatched.to_csv(STILL_UNMATCHED_OUTPUT, index=False)
    ambiguous_review.to_csv(AMBIGUOUS_REVIEW_OUTPUT, index=False)
    delta_summary.to_csv(DELTA_SUMMARY_OUTPUT, index=False)

    baseline_matched = int((baseline_master["match_status"] == "matched").sum())
    final_matched = int((working["match_status"] == "matched").sum())
    newly_matched = working[
        (working["prior_match_status"] != "matched") & (working["match_status"] == "matched")
    ].copy()
    log(f"Baseline matched rows: {baseline_matched:,}")
    log(f"Third-pass matched rows: {final_matched:,}")
    log(f"Newly matched by third pass: {len(newly_matched):,}")
    log(
        "Newly matched movie rows: "
        f"{int((newly_matched['netflix_content_grain'] == 'movie').sum()):,}"
    )
    log(
        "Newly matched series-overall rows: "
        f"{int((newly_matched['netflix_content_grain'] == 'series_overall').sum()):,}"
    )
    log(
        "Still-unmatched movie rows: "
        f"{int(((working['netflix_content_grain'] == 'movie') & (working['match_status'] != 'matched')).sum()):,}"
    )
    log(
        "Still-unmatched series-overall rows: "
        f"{int(((working['netflix_content_grain'] == 'series_overall') & (working['match_status'] != 'matched')).sum()):,}"
    )
    log(
        "Counts by grain: "
        + ", ".join(f"{grain}={count}" for grain, count in working["netflix_content_grain"].value_counts().items())
    )
    log(
        "Counts by IMDb entity type: "
        + ", ".join(
            f"{entity}={count}"
            for entity, count in working["imdb_match_entity_type"].fillna("<NA>").value_counts().items()
        )
    )
    log(
        "Counts by third-pass method: "
        + ", ".join(
            f"{method}={count}"
            for method, count in working[working["third_pass_applied"] == True]["third_pass_match_method"]
            .value_counts()
            .items()
        )
    )
    log(
        "Remaining hard unresolved rows: "
        + ", ".join(
            f"{title} ({count})"
            for title, count in still_unmatched["netflix_title_raw"].fillna("<missing>").value_counts().head(10).items()
        )
    )
    log(f"Saved V3 master parquet: {V3_PARQUET_OUTPUT.relative_to(REPO_ROOT).as_posix()}")
    log(f"Saved V3 master CSV: {V3_CSV_OUTPUT.relative_to(REPO_ROOT).as_posix()}")
    return working


def main() -> None:
    merge_third_pass()


if __name__ == "__main__":
    main()
