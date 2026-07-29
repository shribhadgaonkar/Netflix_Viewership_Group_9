from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.pipeline_utils import (  # noqa: E402
    coerce_nullable_int,
    ensure_parent,
    interim_dir,
    log,
    parse_numeric_series,
    raw_dir,
    require_imdb_inputs,
)


MATCHED_ONLY_PARQUET = REPO_ROOT / "data" / "processed" / "netflix_imdb_master_v3_matched_only.parquet"
MATCHED_ONLY_CSV = REPO_ROOT / "data" / "processed" / "netflix_imdb_master_v3_matched_only.csv"

ENTITY_FEATURES_OUTPUT = interim_dir() / "imdb_entity_features.csv"
EPISODE_AGG_OUTPUT = interim_dir() / "imdb_episode_aggregates.csv"
AKA_AGG_OUTPUT = interim_dir() / "imdb_aka_aggregates.csv"
CREW_AGG_OUTPUT = interim_dir() / "imdb_crew_aggregates.csv"
PRINCIPAL_AGG_OUTPUT = interim_dir() / "imdb_principal_aggregates.csv"

NA_VALUES = ["\\N"]
CHUNK_SIZE = 500_000
CURRENT_YEAR = 2026
MAJOR_GENRES = [
    ("Drama", "genre_drama"),
    ("Comedy", "genre_comedy"),
    ("Action", "genre_action"),
    ("Thriller", "genre_thriller"),
    ("Crime", "genre_crime"),
    ("Romance", "genre_romance"),
    ("Animation", "genre_animation"),
    ("Documentary", "genre_documentary"),
    ("Fantasy", "genre_fantasy"),
    ("Horror", "genre_horror"),
    ("Sci-Fi", "genre_sci_fi"),
    ("Family", "genre_family"),
]
SERIES_TYPES = {"tvSeries", "tvMiniSeries"}


def script_input_paths() -> list[Path]:
    imdb_raw_dir = raw_dir() / "imdb"
    required = require_imdb_inputs(imdb_raw_dir)
    return [
        MATCHED_ONLY_PARQUET if MATCHED_ONLY_PARQUET.exists() else MATCHED_ONLY_CSV,
        required["title.basics.tsv.gz"],
        required["title.ratings.tsv.gz"],
        imdb_raw_dir / "title.episode.tsv.gz",
        imdb_raw_dir / "title.akas.tsv.gz",
        imdb_raw_dir / "title.crew.tsv.gz",
        imdb_raw_dir / "title.principals.tsv.gz",
        imdb_raw_dir / "name.basics.tsv.gz",
        Path(__file__),
    ]


def outputs_fresh(outputs: list[Path], inputs: list[Path], refresh: bool = False) -> bool:
    if refresh or any(not output.exists() for output in outputs):
        return False
    oldest_output = min(output.stat().st_mtime for output in outputs)
    newest_input = max(path.stat().st_mtime for path in inputs if path.exists())
    return oldest_output >= newest_input


def load_matched_only_baseline() -> pd.DataFrame:
    if MATCHED_ONLY_PARQUET.exists():
        return pd.read_parquet(MATCHED_ONLY_PARQUET)
    if MATCHED_ONLY_CSV.exists():
        return pd.read_csv(MATCHED_ONLY_CSV, low_memory=False)
    raise FileNotFoundError(
        "Matched-only baseline not found. Run scripts/data_enrichment/enrich_matched_master_with_imdb.py first."
    )


def derive_enrichment_keys(frame: pd.DataFrame) -> pd.DataFrame:
    matched = frame.copy()
    entity_type = matched["imdb_match_entity_type"].astype("string")
    resolved = matched["imdb_resolved_tconst"].astype("string")
    parent = matched["imdb_parent_tconst"].astype("string")
    enrichment_id = resolved.where(entity_type == "movie", parent.fillna(resolved))
    enrichment_parent = parent.where(entity_type.isin(["series_parent", "series_season"]), pd.NA)

    return pd.DataFrame(
        {
            "netflix_row_id": coerce_nullable_int(matched["netflix_row_id"]),
            "imdb_enrichment_entity_id": enrichment_id,
            "imdb_enrichment_entity_type": entity_type,
            "imdb_enrichment_parent_tconst": enrichment_parent,
            "imdb_season_number": coerce_nullable_int(matched["imdb_season_number"]),
        }
    )


def safe_split(value: Any, separator: str = ",") -> list[str]:
    if value is None or pd.isna(value):
        return []
    parts = [part.strip() for part in str(value).split(separator)]
    return [part for part in parts if part and part != "\\N"]


def safe_join(values: list[str]) -> str | pd._libs.missing.NAType:
    unique_values = []
    for value in values:
        if value and value not in unique_values:
            unique_values.append(value)
    return " | ".join(unique_values) if unique_values else pd.NA


def load_entity_id_sets(matched_only: pd.DataFrame) -> tuple[set[str], set[str]]:
    key_frame = derive_enrichment_keys(matched_only)
    entity_ids = set(
        key_frame["imdb_enrichment_entity_id"].dropna().astype("string").str.strip().tolist()
    )
    series_parent_ids = set(
        key_frame["imdb_enrichment_parent_tconst"].dropna().astype("string").str.strip().tolist()
    )
    return entity_ids, series_parent_ids


def derive_genre_columns(frame: pd.DataFrame) -> pd.DataFrame:
    genres_series = frame["imdb_genres"].astype("string")
    genre_lists = genres_series.fillna("").map(lambda value: safe_split(value, separator=","))
    frame["genre_count"] = genre_lists.map(len).astype("Int64")

    for source_genre, target_column in MAJOR_GENRES:
        frame[target_column] = genre_lists.map(
            lambda genres: int(source_genre in genres)
        ).astype("Int64")

    frame["is_animation"] = frame["genre_animation"].fillna(0).astype("Int64")
    frame["is_documentary"] = frame["genre_documentary"].fillna(0).astype("Int64")
    frame["is_kids_family_like"] = (
        (
            frame["genre_animation"].fillna(0).astype("Int64")
            + frame["genre_family"].fillna(0).astype("Int64")
        )
        > 0
    ).astype("Int64")
    return frame


def build_entity_features(
    entity_ids: set[str],
    basics_path: Path,
    ratings_path: Path,
    refresh: bool = False,
) -> pd.DataFrame:
    if outputs_fresh([ENTITY_FEATURES_OUTPUT], script_input_paths(), refresh=refresh):
        log("Reusing cached IMDb entity feature table.")
        return pd.read_csv(ENTITY_FEATURES_OUTPUT, low_memory=False)

    usecols = [
        "tconst",
        "titleType",
        "primaryTitle",
        "originalTitle",
        "isAdult",
        "startYear",
        "endYear",
        "runtimeMinutes",
        "genres",
    ]
    basics_chunks: list[pd.DataFrame] = []
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
        filtered = chunk[chunk["tconst"].isin(entity_ids)].copy()
        if filtered.empty:
            continue
        basics_chunks.append(filtered)

    if basics_chunks:
        entity_features = pd.concat(basics_chunks, ignore_index=True)
    else:
        entity_features = pd.DataFrame(columns=usecols)

    entity_features.rename(
        columns={
            "tconst": "imdb_enrichment_entity_id",
            "titleType": "imdb_title_type_entity",
            "primaryTitle": "imdb_primary_title_entity",
            "originalTitle": "imdb_original_title_entity",
            "isAdult": "imdb_is_adult",
            "startYear": "imdb_start_year_entity",
            "endYear": "imdb_end_year_entity",
            "runtimeMinutes": "imdb_runtime_minutes_entity",
            "genres": "imdb_genres_entity",
        },
        inplace=True,
    )
    if not entity_features.empty:
        entity_features["imdb_is_adult"] = coerce_nullable_int(entity_features["imdb_is_adult"])
        entity_features["imdb_start_year_entity"] = coerce_nullable_int(
            entity_features["imdb_start_year_entity"]
        )
        entity_features["imdb_end_year_entity"] = coerce_nullable_int(
            entity_features["imdb_end_year_entity"]
        )
        entity_features["imdb_runtime_minutes_entity"] = (
            parse_numeric_series(entity_features["imdb_runtime_minutes_entity"]).round().astype("Int64")
        )
        entity_features["imdb_enrichment_entity_id"] = entity_features[
            "imdb_enrichment_entity_id"
        ].astype("string")

    ratings_chunks: list[pd.DataFrame] = []
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
        filtered = chunk[chunk["tconst"].isin(entity_ids)].copy()
        if filtered.empty:
            continue
        ratings_chunks.append(filtered)

    ratings = pd.concat(ratings_chunks, ignore_index=True) if ratings_chunks else pd.DataFrame()
    if not ratings.empty:
        ratings.rename(
            columns={
                "tconst": "imdb_enrichment_entity_id",
                "averageRating": "imdb_average_rating_entity",
                "numVotes": "imdb_num_votes_entity",
            },
            inplace=True,
        )
        ratings["imdb_enrichment_entity_id"] = ratings["imdb_enrichment_entity_id"].astype("string")
        ratings["imdb_num_votes_entity"] = coerce_nullable_int(ratings["imdb_num_votes_entity"])
        entity_features = entity_features.merge(ratings, how="left", on="imdb_enrichment_entity_id")
    else:
        entity_features["imdb_average_rating_entity"] = pd.NA
        entity_features["imdb_num_votes_entity"] = pd.Series(pd.NA, index=entity_features.index, dtype="Int64")

    if not entity_features.empty:
        entity_features = derive_genre_columns(
            entity_features.rename(columns={"imdb_genres_entity": "imdb_genres"})
        )
        entity_features.rename(columns={"imdb_genres": "imdb_genres_entity"}, inplace=True)

    ensure_parent(ENTITY_FEATURES_OUTPUT)
    entity_features.to_csv(ENTITY_FEATURES_OUTPUT, index=False)
    log(f"Saved IMDb entity feature table: {len(entity_features):,} rows")
    return entity_features


def build_episode_aggregates(
    series_parent_ids: set[str],
    episode_path: Path,
    refresh: bool = False,
) -> pd.DataFrame:
    if outputs_fresh([EPISODE_AGG_OUTPUT], script_input_paths(), refresh=refresh):
        log("Reusing cached IMDb episode aggregate table.")
        return pd.read_csv(EPISODE_AGG_OUTPUT, low_memory=False)

    if not series_parent_ids:
        empty = pd.DataFrame(
            columns=[
                "imdb_parent_tconst",
                "imdb_season_number",
                "imdb_season_episode_count_feature",
                "imdb_total_episode_count",
                "imdb_parent_season_count_feature",
                "imdb_max_season_number",
                "imdb_avg_episodes_per_season",
                "imdb_min_episodes_per_season",
                "imdb_max_episodes_per_season",
                "imdb_single_season_flag",
                "imdb_multi_season_flag",
            ]
        )
        ensure_parent(EPISODE_AGG_OUTPUT)
        empty.to_csv(EPISODE_AGG_OUTPUT, index=False)
        return empty

    grouped_chunks: list[pd.DataFrame] = []
    for chunk in pd.read_csv(
        episode_path,
        sep="\t",
        compression="gzip",
        na_values=NA_VALUES,
        keep_default_na=True,
        usecols=["parentTconst", "seasonNumber"],
        chunksize=CHUNK_SIZE,
        low_memory=False,
    ):
        filtered = chunk[chunk["parentTconst"].isin(series_parent_ids)].copy()
        if filtered.empty:
            continue
        filtered["seasonNumber"] = parse_numeric_series(filtered["seasonNumber"])
        filtered = filtered.dropna(subset=["seasonNumber"])
        filtered = filtered[filtered["seasonNumber"] >= 1]
        if filtered.empty:
            continue
        filtered["seasonNumber"] = filtered["seasonNumber"].round().astype("Int64")
        grouped = (
            filtered.groupby(["parentTconst", "seasonNumber"], as_index=False)
            .size()
            .rename(
                columns={
                    "parentTconst": "imdb_parent_tconst",
                    "seasonNumber": "imdb_season_number",
                    "size": "imdb_season_episode_count_feature",
                }
            )
        )
        grouped_chunks.append(grouped)

    season_counts = (
        pd.concat(grouped_chunks, ignore_index=True)
        .groupby(["imdb_parent_tconst", "imdb_season_number"], as_index=False)[
            "imdb_season_episode_count_feature"
        ]
        .sum()
        if grouped_chunks
        else pd.DataFrame(
            columns=["imdb_parent_tconst", "imdb_season_number", "imdb_season_episode_count_feature"]
        )
    )

    if season_counts.empty:
        episode_aggs = pd.DataFrame(
            columns=[
                "imdb_parent_tconst",
                "imdb_season_number",
                "imdb_season_episode_count_feature",
                "imdb_total_episode_count",
                "imdb_parent_season_count_feature",
                "imdb_max_season_number",
                "imdb_avg_episodes_per_season",
                "imdb_min_episodes_per_season",
                "imdb_max_episodes_per_season",
                "imdb_single_season_flag",
                "imdb_multi_season_flag",
            ]
        )
    else:
        parent_agg = (
            season_counts.groupby("imdb_parent_tconst", as_index=False)
            .agg(
                imdb_total_episode_count=("imdb_season_episode_count_feature", "sum"),
                imdb_parent_season_count_feature=("imdb_season_number", "nunique"),
                imdb_max_season_number=("imdb_season_number", "max"),
                imdb_avg_episodes_per_season=("imdb_season_episode_count_feature", "mean"),
                imdb_min_episodes_per_season=("imdb_season_episode_count_feature", "min"),
                imdb_max_episodes_per_season=("imdb_season_episode_count_feature", "max"),
            )
        )
        parent_agg["imdb_single_season_flag"] = (
            parent_agg["imdb_parent_season_count_feature"] == 1
        ).astype("Int64")
        parent_agg["imdb_multi_season_flag"] = (
            parent_agg["imdb_parent_season_count_feature"] > 1
        ).astype("Int64")
        episode_aggs = season_counts.merge(parent_agg, how="left", on="imdb_parent_tconst")

    ensure_parent(EPISODE_AGG_OUTPUT)
    episode_aggs.to_csv(EPISODE_AGG_OUTPUT, index=False)
    log(f"Saved IMDb episode aggregate table: {len(episode_aggs):,} rows")
    return episode_aggs


def build_aka_aggregates(entity_ids: set[str], akas_path: Path, refresh: bool = False) -> pd.DataFrame:
    if outputs_fresh([AKA_AGG_OUTPUT], script_input_paths(), refresh=refresh):
        log("Reusing cached IMDb aka aggregate table.")
        return pd.read_csv(AKA_AGG_OUTPUT, low_memory=False)

    if not entity_ids:
        empty = pd.DataFrame(
            columns=[
                "imdb_enrichment_entity_id",
                "imdb_aka_title_count_feature",
                "imdb_aka_region_count",
                "imdb_aka_language_count",
                "imdb_has_us_title",
                "imdb_has_international_title_variants",
            ]
        )
        ensure_parent(AKA_AGG_OUTPUT)
        empty.to_csv(AKA_AGG_OUTPUT, index=False)
        return empty

    aka_chunks: list[pd.DataFrame] = []
    for chunk in pd.read_csv(
        akas_path,
        sep="\t",
        compression="gzip",
        na_values=NA_VALUES,
        keep_default_na=True,
        usecols=["titleId", "title", "region", "language"],
        chunksize=CHUNK_SIZE,
        low_memory=False,
    ):
        filtered = chunk[chunk["titleId"].isin(entity_ids)].copy()
        if filtered.empty:
            continue
        filtered.rename(columns={"titleId": "imdb_enrichment_entity_id"}, inplace=True)
        aka_chunks.append(filtered)

    if aka_chunks:
        akas = pd.concat(aka_chunks, ignore_index=True)
        aka_aggs = (
            akas.groupby("imdb_enrichment_entity_id", as_index=False)
            .agg(
                imdb_aka_title_count_feature=("title", "nunique"),
                imdb_aka_region_count=("region", lambda values: values.dropna().astype(str).nunique()),
                imdb_aka_language_count=("language", lambda values: values.dropna().astype(str).nunique()),
                imdb_has_us_title=("region", lambda values: int(values.astype("string").eq("US").any())),
                imdb_has_international_title_variants=(
                    "region",
                    lambda values: int(values.dropna().astype(str).nunique() > 1),
                ),
            )
        )
        aka_aggs["imdb_aka_title_count_feature"] = coerce_nullable_int(
            aka_aggs["imdb_aka_title_count_feature"]
        )
        aka_aggs["imdb_aka_region_count"] = coerce_nullable_int(aka_aggs["imdb_aka_region_count"])
        aka_aggs["imdb_aka_language_count"] = coerce_nullable_int(aka_aggs["imdb_aka_language_count"])
        aka_aggs["imdb_has_us_title"] = coerce_nullable_int(aka_aggs["imdb_has_us_title"])
        aka_aggs["imdb_has_international_title_variants"] = coerce_nullable_int(
            aka_aggs["imdb_has_international_title_variants"]
        )
    else:
        aka_aggs = pd.DataFrame(
            columns=[
                "imdb_enrichment_entity_id",
                "imdb_aka_title_count_feature",
                "imdb_aka_region_count",
                "imdb_aka_language_count",
                "imdb_has_us_title",
                "imdb_has_international_title_variants",
            ]
        )

    ensure_parent(AKA_AGG_OUTPUT)
    aka_aggs.to_csv(AKA_AGG_OUTPUT, index=False)
    log(f"Saved IMDb aka aggregate table: {len(aka_aggs):,} rows")
    return aka_aggs


def load_person_lookup(person_ids: set[str], names_path: Path) -> dict[str, dict[str, Any]]:
    if not person_ids:
        return {}

    person_lookup: dict[str, dict[str, Any]] = {}
    for chunk in pd.read_csv(
        names_path,
        sep="\t",
        compression="gzip",
        na_values=NA_VALUES,
        keep_default_na=True,
        usecols=["nconst", "primaryName", "birthYear", "primaryProfession", "knownForTitles"],
        chunksize=CHUNK_SIZE,
        low_memory=False,
    ):
        filtered = chunk[chunk["nconst"].isin(person_ids)].copy()
        if filtered.empty:
            continue
        for _, row in filtered.iterrows():
            professions = safe_split(row.get("primaryProfession"), separator=",")
            person_lookup[str(row["nconst"])] = {
                "name": row.get("primaryName"),
                "birth_year": pd.to_numeric(row.get("birthYear"), errors="coerce"),
                "profession_list": professions,
                "known_for_count": len(safe_split(row.get("knownForTitles"), separator=",")),
            }
    return person_lookup


def mean_birth_year(person_ids: list[str], person_lookup: dict[str, dict[str, Any]]) -> float | pd._libs.missing.NAType:
    values = []
    for person_id in person_ids:
        birth_year = person_lookup.get(person_id, {}).get("birth_year")
        if birth_year is not None and not pd.isna(birth_year):
            values.append(float(birth_year))
    return float(sum(values) / len(values)) if values else pd.NA


def build_crew_and_principal_aggregates(
    entity_ids: set[str],
    crew_path: Path,
    principals_path: Path,
    names_path: Path,
    refresh: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if outputs_fresh([CREW_AGG_OUTPUT, PRINCIPAL_AGG_OUTPUT], script_input_paths(), refresh=refresh):
        log("Reusing cached IMDb crew and principal aggregate tables.")
        return (
            pd.read_csv(CREW_AGG_OUTPUT, low_memory=False),
            pd.read_csv(PRINCIPAL_AGG_OUTPUT, low_memory=False),
        )

    crew_records: dict[str, dict[str, list[str]]] = {}
    for chunk in pd.read_csv(
        crew_path,
        sep="\t",
        compression="gzip",
        na_values=NA_VALUES,
        keep_default_na=True,
        usecols=["tconst", "directors", "writers"],
        chunksize=CHUNK_SIZE,
        low_memory=False,
    ):
        filtered = chunk[chunk["tconst"].isin(entity_ids)].copy()
        if filtered.empty:
            continue
        for _, row in filtered.iterrows():
            title_id = str(row["tconst"])
            crew_records[title_id] = {
                "directors": safe_split(row.get("directors"), separator=","),
                "writers": safe_split(row.get("writers"), separator=","),
            }

    principal_records: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "imdb_principal_count": 0,
            "imdb_actor_count": 0,
            "imdb_actress_count": 0,
            "imdb_self_count": 0,
            "imdb_producer_count": 0,
            "imdb_writer_credit_count": 0,
            "imdb_director_credit_count": 0,
            "top_cast_ordering": [],
        }
    )

    for chunk in pd.read_csv(
        principals_path,
        sep="\t",
        compression="gzip",
        na_values=NA_VALUES,
        keep_default_na=True,
        usecols=["tconst", "ordering", "nconst", "category"],
        chunksize=CHUNK_SIZE,
        low_memory=False,
    ):
        filtered = chunk[chunk["tconst"].isin(entity_ids)].copy()
        if filtered.empty:
            continue
        for _, row in filtered.iterrows():
            title_id = str(row["tconst"])
            category = "" if pd.isna(row.get("category")) else str(row.get("category"))
            principal_records[title_id]["imdb_principal_count"] += 1
            if category == "actor":
                principal_records[title_id]["imdb_actor_count"] += 1
            elif category == "actress":
                principal_records[title_id]["imdb_actress_count"] += 1
            elif category == "self":
                principal_records[title_id]["imdb_self_count"] += 1
            elif category == "producer":
                principal_records[title_id]["imdb_producer_count"] += 1
            elif category == "writer":
                principal_records[title_id]["imdb_writer_credit_count"] += 1
            elif category == "director":
                principal_records[title_id]["imdb_director_credit_count"] += 1

            if category in {"actor", "actress", "self"} and pd.notna(row.get("nconst")):
                ordering = pd.to_numeric(row.get("ordering"), errors="coerce")
                if pd.isna(ordering):
                    ordering = 9999
                principal_records[title_id]["top_cast_ordering"].append((int(ordering), str(row["nconst"])))

    person_ids: set[str] = set()
    for record in crew_records.values():
        person_ids.update(record["directors"])
        person_ids.update(record["writers"])
    for record in principal_records.values():
        person_ids.update([nconst for _, nconst in record["top_cast_ordering"][:5]])

    person_lookup = load_person_lookup(person_ids, names_path)

    crew_rows: list[dict[str, Any]] = []
    for title_id in sorted(entity_ids):
        directors = crew_records.get(title_id, {}).get("directors", [])
        writers = crew_records.get(title_id, {}).get("writers", [])
        director_names = [person_lookup.get(person_id, {}).get("name") for person_id in directors]
        writer_names = [person_lookup.get(person_id, {}).get("name") for person_id in writers]
        crew_rows.append(
            {
                "imdb_enrichment_entity_id": title_id,
                "imdb_director_count": len(directors),
                "imdb_writer_count": len(writers),
                "imdb_director_nconsts": safe_join(directors),
                "imdb_writer_nconsts": safe_join(writers),
                "imdb_director_names": safe_join([name for name in director_names if name]),
                "imdb_writer_names": safe_join([name for name in writer_names if name]),
                "imdb_director_mean_birth_year": mean_birth_year(directors, person_lookup),
                "imdb_writer_mean_birth_year": mean_birth_year(writers, person_lookup),
            }
        )

    principal_rows: list[dict[str, Any]] = []
    for title_id in sorted(entity_ids):
        record = principal_records.get(title_id, None)
        if record is None:
            principal_rows.append(
                {
                    "imdb_enrichment_entity_id": title_id,
                    "imdb_principal_count": 0,
                    "imdb_actor_count": 0,
                    "imdb_actress_count": 0,
                    "imdb_self_count": 0,
                    "imdb_producer_count": 0,
                    "imdb_writer_credit_count": 0,
                    "imdb_director_credit_count": 0,
                    "imdb_top_cast_nconsts": pd.NA,
                    "imdb_top_cast_names": pd.NA,
                    "imdb_top_cast_count_used": 0,
                    "imdb_top_cast_mean_birth_year": pd.NA,
                    "imdb_top_cast_known_for_count_proxy": pd.NA,
                    "imdb_top_cast_profession_mix": pd.NA,
                }
            )
            continue

        top_cast_pairs = sorted(record["top_cast_ordering"], key=lambda item: (item[0], item[1]))
        top_cast_ids: list[str] = []
        for _, person_id in top_cast_pairs:
            if person_id not in top_cast_ids:
                top_cast_ids.append(person_id)
            if len(top_cast_ids) >= 5:
                break
        top_cast_names = [person_lookup.get(person_id, {}).get("name") for person_id in top_cast_ids]
        professions = []
        known_for_counts = []
        for person_id in top_cast_ids:
            person = person_lookup.get(person_id, {})
            professions.extend(person.get("profession_list", []))
            known_for_count = person.get("known_for_count")
            if known_for_count is not None:
                known_for_counts.append(float(known_for_count))
        principal_rows.append(
            {
                "imdb_enrichment_entity_id": title_id,
                "imdb_principal_count": record["imdb_principal_count"],
                "imdb_actor_count": record["imdb_actor_count"],
                "imdb_actress_count": record["imdb_actress_count"],
                "imdb_self_count": record["imdb_self_count"],
                "imdb_producer_count": record["imdb_producer_count"],
                "imdb_writer_credit_count": record["imdb_writer_credit_count"],
                "imdb_director_credit_count": record["imdb_director_credit_count"],
                "imdb_top_cast_nconsts": safe_join(top_cast_ids),
                "imdb_top_cast_names": safe_join([name for name in top_cast_names if name]),
                "imdb_top_cast_count_used": len(top_cast_ids),
                "imdb_top_cast_mean_birth_year": mean_birth_year(top_cast_ids, person_lookup),
                "imdb_top_cast_known_for_count_proxy": (
                    float(sum(known_for_counts) / len(known_for_counts)) if known_for_counts else pd.NA
                ),
                "imdb_top_cast_profession_mix": safe_join(sorted(set(professions))),
            }
        )

    crew_aggs = pd.DataFrame(crew_rows)
    principal_aggs = pd.DataFrame(principal_rows)
    count_columns = [
        "imdb_director_count",
        "imdb_writer_count",
        "imdb_principal_count",
        "imdb_actor_count",
        "imdb_actress_count",
        "imdb_self_count",
        "imdb_producer_count",
        "imdb_writer_credit_count",
        "imdb_director_credit_count",
        "imdb_top_cast_count_used",
    ]
    for column in count_columns:
        if column in crew_aggs.columns:
            crew_aggs[column] = coerce_nullable_int(crew_aggs[column])
        if column in principal_aggs.columns:
            principal_aggs[column] = coerce_nullable_int(principal_aggs[column])

    ensure_parent(CREW_AGG_OUTPUT)
    crew_aggs.to_csv(CREW_AGG_OUTPUT, index=False)
    principal_aggs.to_csv(PRINCIPAL_AGG_OUTPUT, index=False)
    log(f"Saved IMDb crew aggregate table: {len(crew_aggs):,} rows")
    log(f"Saved IMDb principal aggregate table: {len(principal_aggs):,} rows")
    return crew_aggs, principal_aggs


def prepare_all_feature_tables(matched_only: pd.DataFrame | None = None, refresh: bool = False) -> dict[str, pd.DataFrame]:
    baseline = load_matched_only_baseline() if matched_only is None else matched_only.copy()
    entity_ids, series_parent_ids = load_entity_id_sets(baseline)

    imdb_raw_dir = raw_dir() / "imdb"
    basics_path = imdb_raw_dir / "title.basics.tsv.gz"
    ratings_path = imdb_raw_dir / "title.ratings.tsv.gz"
    episode_path = imdb_raw_dir / "title.episode.tsv.gz"
    akas_path = imdb_raw_dir / "title.akas.tsv.gz"
    crew_path = imdb_raw_dir / "title.crew.tsv.gz"
    principals_path = imdb_raw_dir / "title.principals.tsv.gz"
    names_path = imdb_raw_dir / "name.basics.tsv.gz"

    entity_features = build_entity_features(entity_ids, basics_path, ratings_path, refresh=refresh)
    episode_aggs = build_episode_aggregates(series_parent_ids, episode_path, refresh=refresh)
    aka_aggs = build_aka_aggregates(entity_ids, akas_path, refresh=refresh)
    crew_aggs, principal_aggs = build_crew_and_principal_aggregates(
        entity_ids, crew_path, principals_path, names_path, refresh=refresh
    )

    return {
        "entity_features": entity_features,
        "episode_aggs": episode_aggs,
        "aka_aggs": aka_aggs,
        "crew_aggs": crew_aggs,
        "principal_aggs": principal_aggs,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare cached IMDb feature tables for matched-only enrichment.")
    parser.add_argument("--refresh", action="store_true", help="Rebuild cached feature tables even if current outputs exist.")
    args = parser.parse_args()

    matched_only = load_matched_only_baseline()
    prepare_all_feature_tables(matched_only, refresh=args.refresh)


if __name__ == "__main__":
    main()
