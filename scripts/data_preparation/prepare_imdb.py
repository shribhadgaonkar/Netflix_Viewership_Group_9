from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.pipeline_utils import (  # noqa: E402
    SERIES_TITLE_TYPES,
    coerce_nullable_int,
    ensure_parent,
    interim_dir,
    log,
    normalize_title,
    parse_numeric_series,
    raw_dir,
    require_imdb_inputs,
)


OUTPUT_PATH = interim_dir() / "imdb_series_seasons.csv"
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
        filtered["runtimeMinutes"] = parse_numeric_series(filtered["runtimeMinutes"]).round().astype("Int64")
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
    parents["imdb_original_normalized_title"] = parents["imdb_original_title"].map(normalize_title)
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
            .rename(columns={"parentTconst": "imdb_parent_tconst", "seasonNumber": "imdb_season_number", "size": "imdb_season_episode_count"})
        )
        grouped_chunks.append(grouped)

    season_counts = (
        pd.concat(grouped_chunks, ignore_index=True)
        .groupby(["imdb_parent_tconst", "imdb_season_number"], as_index=False)["imdb_season_episode_count"]
        .sum()
    )

    log(f"IMDb episode rows missing season numbers: {missing_season_rows:,}")
    log(f"IMDb episode rows dropped as season 0/specials: {special_season_rows:,}")
    return season_counts


def prepare_imdb() -> pd.DataFrame:
    imdb_raw_dir = raw_dir() / "imdb"
    inputs = require_imdb_inputs(imdb_raw_dir)
    log("Reading IMDb raw files directly from compressed TSV.GZ inputs.")

    parents = load_parent_series(inputs["title.basics.tsv.gz"])
    ratings = load_series_ratings(inputs["title.ratings.tsv.gz"])
    season_counts = build_season_episode_counts(inputs["title.episode.tsv.gz"])

    imdb = season_counts.merge(parents, how="left", on="imdb_parent_tconst")
    imdb = imdb.merge(ratings, how="left", on="imdb_parent_tconst")

    imdb = imdb[
        [
            "imdb_parent_tconst",
            "imdb_primary_title",
            "imdb_original_title",
            "imdb_normalized_title",
            "imdb_original_normalized_title",
            "imdb_title_type",
            "imdb_start_year",
            "imdb_end_year",
            "imdb_genres",
            "imdb_runtime_minutes",
            "imdb_average_rating",
            "imdb_num_votes",
            "imdb_season_number",
            "imdb_season_episode_count",
        ]
    ].sort_values(["imdb_normalized_title", "imdb_season_number", "imdb_parent_tconst"])

    ensure_parent(OUTPUT_PATH)
    imdb.to_csv(OUTPUT_PATH, index=False)

    log(f"IMDb parent series retained: {len(parents):,}")
    log(f"IMDb season rows created: {len(imdb):,}")
    log(
        "IMDb duplicate exact keys (normalized title + season): "
        f"{int(imdb.duplicated(['imdb_normalized_title', 'imdb_season_number'], keep=False).sum()):,}"
    )
    log(f"Saved IMDb series-season data: {OUTPUT_PATH.relative_to(REPO_ROOT).as_posix()}")
    return imdb


def main() -> None:
    prepare_imdb()


if __name__ == "__main__":
    main()
