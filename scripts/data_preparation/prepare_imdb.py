from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.pipeline_utils import (  # noqa: E402
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
    raw_dir,
    require_imdb_inputs,
)


OUTPUT_PATH = interim_dir() / "imdb_series_seasons.csv"
TITLE_KEYS_OUTPUT = interim_dir() / "imdb_title_keys.csv"
NA_VALUES = ["\\N"]
CHUNK_SIZE = 500_000


def load_parent_series(basics_path: Path) -> pd.DataFrame:
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
    collected: list[pd.DataFrame] = []

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
        filtered = chunk[chunk["titleType"].isin(SERIES_TITLE_TYPES)].copy()
        if filtered.empty:
            continue

        filtered["startYear"] = coerce_nullable_int(filtered["startYear"])
        filtered["endYear"] = coerce_nullable_int(filtered["endYear"])
        filtered["runtimeMinutes"] = (
            parse_numeric_series(filtered["runtimeMinutes"]).round().astype("Int64")
        )
        collected.append(filtered)

    parents = pd.concat(collected, ignore_index=True)
    parents.rename(
        columns={
            "tconst": "imdb_parent_tconst",
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
    parents["imdb_normalized_title"] = parents["imdb_primary_title"].map(normalize_title)
    parents["imdb_primary_normalized_title"] = parents["imdb_primary_title"].map(normalize_title)
    parents["imdb_original_normalized_title"] = parents["imdb_original_title"].map(normalize_title)
    parents["imdb_primary_canonical_title"] = parents["imdb_primary_title"].map(canonicalize_title)
    parents["imdb_original_canonical_title"] = parents["imdb_original_title"].map(canonicalize_title)
    parents["imdb_primary_compact_title"] = parents["imdb_primary_title"].map(compact_title_key)
    parents["imdb_original_compact_title"] = parents["imdb_original_title"].map(compact_title_key)
    return parents


def load_series_ratings(ratings_path: Path) -> pd.DataFrame:
    ratings = pd.read_csv(
        ratings_path,
        sep="\t",
        compression="gzip",
        na_values=NA_VALUES,
        keep_default_na=True,
        low_memory=False,
    )
    ratings.rename(
        columns={
            "tconst": "imdb_parent_tconst",
            "averageRating": "imdb_average_rating",
            "numVotes": "imdb_num_votes",
        },
        inplace=True,
    )
    ratings["imdb_num_votes"] = coerce_nullable_int(ratings["imdb_num_votes"])
    return ratings


def build_season_episode_counts(episodes_path: Path) -> pd.DataFrame:
    grouped_chunks: list[pd.DataFrame] = []
    missing_season_rows = 0
    special_season_rows = 0

    for chunk in pd.read_csv(
        episodes_path,
        sep="\t",
        compression="gzip",
        na_values=NA_VALUES,
        keep_default_na=True,
        chunksize=CHUNK_SIZE,
        low_memory=False,
    ):
        chunk["seasonNumber"] = parse_numeric_series(chunk["seasonNumber"])
        missing_season_rows += int(chunk["seasonNumber"].isna().sum())

        valid = chunk.dropna(subset=["parentTconst", "seasonNumber"]).copy()
        special_season_rows += int((valid["seasonNumber"] < 1).sum())
        valid = valid[valid["seasonNumber"] >= 1]

        if valid.empty:
            continue

        valid["seasonNumber"] = valid["seasonNumber"].round().astype("Int64")
        grouped = (
            valid.groupby(["parentTconst", "seasonNumber"], as_index=False)
            .size()
            .rename(
                columns={
                    "parentTconst": "imdb_parent_tconst",
                    "seasonNumber": "imdb_season_number",
                    "size": "imdb_season_episode_count",
                }
            )
        )
        grouped_chunks.append(grouped)

    season_counts = (
        pd.concat(grouped_chunks, ignore_index=True)
        .groupby(["imdb_parent_tconst", "imdb_season_number"], as_index=False)[
            "imdb_season_episode_count"
        ]
        .sum()
    )

    log(f"IMDb episode rows missing season numbers: {missing_season_rows:,}")
    log(f"IMDb episode rows dropped as season 0/specials: {special_season_rows:,}")
    return season_counts


def load_aka_titles(akas_path: Path, valid_parent_ids: set[str]) -> pd.DataFrame:
    collected: list[pd.DataFrame] = []

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
        filtered = chunk[chunk["titleId"].isin(valid_parent_ids)].copy()
        if filtered.empty:
            continue

        filtered.rename(columns={"titleId": "imdb_parent_tconst", "title": "imdb_aka_title"}, inplace=True)
        filtered["imdb_aka_normalized_title"] = filtered["imdb_aka_title"].map(normalize_title)
        filtered["imdb_aka_canonical_title"] = filtered["imdb_aka_title"].map(canonicalize_title)
        filtered["imdb_aka_compact_title"] = filtered["imdb_aka_title"].map(compact_title_key)
        filtered = filtered[
            filtered["imdb_aka_normalized_title"].notna()
            & filtered["imdb_aka_normalized_title"].astype("string").str.strip().ne("")
        ]
        collected.append(filtered)

    if not collected:
        return pd.DataFrame(
            columns=[
                "imdb_parent_tconst",
                "imdb_aka_title",
                "region",
                "language",
                "types",
                "isOriginalTitle",
                "imdb_aka_normalized_title",
                "imdb_aka_canonical_title",
                "imdb_aka_compact_title",
            ]
        )

    akas = pd.concat(collected, ignore_index=True).drop_duplicates(
        subset=["imdb_parent_tconst", "imdb_aka_title", "imdb_aka_normalized_title"]
    )
    return akas


def build_title_key_table(parents: pd.DataFrame, akas: pd.DataFrame) -> pd.DataFrame:
    key_frames: list[pd.DataFrame] = []

    primary = parents[
        [
            "imdb_parent_tconst",
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

    original = parents[
        [
            "imdb_parent_tconst",
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

    if not akas.empty:
        aka_keys = akas[
            [
                "imdb_parent_tconst",
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

    title_keys = pd.concat(key_frames, ignore_index=True)
    title_keys = title_keys[
        title_keys["imdb_match_key_used"].notna()
        & title_keys["imdb_match_key_used"].astype("string").str.strip().ne("")
    ].copy()
    title_keys.drop_duplicates(
        subset=["imdb_parent_tconst", "candidate_match_source", "imdb_match_key_used"],
        inplace=True,
    )
    return title_keys


def aggregate_aka_columns(akas: pd.DataFrame) -> pd.DataFrame:
    if akas.empty:
        return pd.DataFrame(
            columns=[
                "imdb_parent_tconst",
                "imdb_aka_normalized_titles",
                "imdb_aka_canonical_titles",
                "imdb_aka_title_count",
            ]
        )

    aggregated = (
        akas.groupby("imdb_parent_tconst", as_index=False)
        .agg(
            imdb_aka_normalized_titles=(
                "imdb_aka_normalized_title",
                lambda values: " | ".join(sorted(pd.unique(values.dropna()))),
            ),
            imdb_aka_canonical_titles=(
                "imdb_aka_canonical_title",
                lambda values: " | ".join(sorted(pd.unique(values.dropna()))),
            ),
            imdb_aka_title_count=("imdb_aka_title", "nunique"),
        )
    )
    return aggregated


def prepare_imdb() -> tuple[pd.DataFrame, pd.DataFrame]:
    imdb_raw_dir = raw_dir() / "imdb"
    required_inputs = require_imdb_inputs(imdb_raw_dir)
    optional_inputs = optional_imdb_inputs(imdb_raw_dir)
    log("Reading IMDb raw files directly from compressed TSV.GZ inputs.")

    parents = load_parent_series(required_inputs["title.basics.tsv.gz"])
    ratings = load_series_ratings(required_inputs["title.ratings.tsv.gz"])
    season_counts = build_season_episode_counts(required_inputs["title.episode.tsv.gz"])

    akas = pd.DataFrame()
    if "title.akas.tsv.gz" in optional_inputs:
        log("IMDb title.akas.tsv.gz found; building alternate title keys.")
        akas = load_aka_titles(optional_inputs["title.akas.tsv.gz"], set(parents["imdb_parent_tconst"]))
        log(f"IMDb alternate titles retained: {len(akas):,}")
    else:
        log("IMDb title.akas.tsv.gz not found; skipping alternate title matching keys.")

    aka_aggregates = aggregate_aka_columns(akas)
    parent_season_counts = (
        season_counts.groupby("imdb_parent_tconst", as_index=False)
        .size()
        .rename(columns={"size": "imdb_parent_season_count"})
    )

    imdb = season_counts.merge(parents, how="left", on="imdb_parent_tconst")
    imdb = imdb.merge(ratings, how="left", on="imdb_parent_tconst")
    imdb = imdb.merge(parent_season_counts, how="left", on="imdb_parent_tconst")
    imdb = imdb.merge(aka_aggregates, how="left", on="imdb_parent_tconst")

    imdb = imdb[
        [
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
            "imdb_season_number",
            "imdb_season_episode_count",
        ]
    ].sort_values(["imdb_normalized_title", "imdb_season_number", "imdb_parent_tconst"])

    title_keys = build_title_key_table(parents, akas)

    ensure_parent(OUTPUT_PATH)
    imdb.to_csv(OUTPUT_PATH, index=False)
    ensure_parent(TITLE_KEYS_OUTPUT)
    title_keys.to_csv(TITLE_KEYS_OUTPUT, index=False)

    log(f"IMDb parent series retained: {len(parents):,}")
    log(f"IMDb season rows created: {len(imdb):,}")
    log(
        "IMDb duplicate primary exact keys (normalized title + season): "
        f"{int(imdb.duplicated(['imdb_normalized_title', 'imdb_season_number'], keep=False).sum()):,}"
    )
    log(f"IMDb title keys saved: {len(title_keys):,}")
    log(f"Saved IMDb series-season data: {OUTPUT_PATH.relative_to(REPO_ROOT).as_posix()}")
    log(f"Saved IMDb title-key data: {TITLE_KEYS_OUTPUT.relative_to(REPO_ROOT).as_posix()}")
    return imdb, title_keys


def main() -> None:
    prepare_imdb()


if __name__ == "__main__":
    main()
