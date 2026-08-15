from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.data_enrichment.prepare_imdb_entity_features import (  # noqa: E402
    MATCHED_ONLY_CSV,
    MATCHED_ONLY_PARQUET,
    CURRENT_YEAR,
    derive_enrichment_keys,
    prepare_all_feature_tables,
)
from src.pipeline_utils import ensure_parent, log, normalize_title, parse_numeric_series  # noqa: E402


MASTER_V3_PARQUET = REPO_ROOT / "data" / "processed" / "netflix_imdb_master_v3.parquet"
MASTER_V3_CSV = REPO_ROOT / "data" / "processed" / "netflix_imdb_master_v3.csv"
ENRICHED_PARQUET = REPO_ROOT / "data" / "processed" / "netflix_imdb_master_matched_enriched.parquet"
ENRICHED_CSV = REPO_ROOT / "data" / "processed" / "netflix_imdb_master_matched_enriched.csv"
HALFYEAR_VIEWS_INPUT = REPO_ROOT / "data" / "raw" / "netflix" / "netflixlist9-HalfYear Views.csv"
HALFYEAR_HOURS_INPUT = REPO_ROOT / "data" / "raw" / "netflix" / "netflixlist10-Half year hours.csv"
HALFYEAR_PERIOD_PATTERN = re.compile(r"^(?P<year>\d{4})\s+(?P<half>H[12])\s+(?:Views|Hours)$", re.IGNORECASE)
HALFYEAR_FEATURE_COLUMNS = [
    "first_observed_halfyear_period",
    "first_observed_halfyear_views",
    "first_observed_halfyear_hours",
    "first_halfyear_hours_per_view",
]


def load_master_v3() -> pd.DataFrame:
    if MASTER_V3_PARQUET.exists():
        return pd.read_parquet(MASTER_V3_PARQUET)
    if MASTER_V3_CSV.exists():
        return pd.read_csv(MASTER_V3_CSV, low_memory=False)
    raise FileNotFoundError("Third-pass master dataset not found.")


def clean_halfyear_numeric_values(series: pd.Series) -> pd.Series:
    return parse_numeric_series(series)


def extract_available_halfyear_periods(
    views_columns: list[str], hours_columns: list[str]
) -> list[str]:
    view_periods: dict[str, tuple[int, int]] = {}
    hour_periods: dict[str, tuple[int, int]] = {}

    for column in views_columns:
        match = HALFYEAR_PERIOD_PATTERN.match(column.strip())
        if not match:
            continue
        year = int(match.group("year"))
        half = 1 if match.group("half").upper() == "H1" else 2
        view_periods[f"{year} H{half}"] = (year, half)

    for column in hours_columns:
        match = HALFYEAR_PERIOD_PATTERN.match(column.strip())
        if not match:
            continue
        year = int(match.group("year"))
        half = 1 if match.group("half").upper() == "H1" else 2
        hour_periods[f"{year} H{half}"] = (year, half)

    shared = sorted(
        set(view_periods).intersection(hour_periods),
        key=lambda period: view_periods[period],
    )
    return shared


def identify_first_valid_observed_halfyear(
    frame: pd.DataFrame, periods: list[str]
) -> pd.DataFrame:
    result = pd.DataFrame(index=frame.index)
    result["first_observed_halfyear_period"] = pd.Series(pd.NA, index=frame.index, dtype="string")
    result["first_observed_halfyear_views"] = np.nan
    result["first_observed_halfyear_hours"] = np.nan

    for period in periods:
        views_column = f"{period} Views"
        hours_column = f"{period} Hours"
        if views_column not in frame.columns or hours_column not in frame.columns:
            continue
        valid = (
            result["first_observed_halfyear_period"].isna()
            & frame[views_column].notna()
            & frame[hours_column].notna()
        )
        if not valid.any():
            continue
        result.loc[valid, "first_observed_halfyear_period"] = period
        result.loc[valid, "first_observed_halfyear_views"] = frame.loc[valid, views_column]
        result.loc[valid, "first_observed_halfyear_hours"] = frame.loc[valid, hours_column]

    views_denominator = pd.to_numeric(result["first_observed_halfyear_views"], errors="coerce")
    hours_numerator = pd.to_numeric(result["first_observed_halfyear_hours"], errors="coerce")
    result["first_halfyear_hours_per_view"] = np.where(
        views_denominator.notna() & (views_denominator != 0),
        hours_numerator / views_denominator,
        np.nan,
    )
    return result


def build_halfyear_feature_table(enriched: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object]]:
    if not HALFYEAR_VIEWS_INPUT.exists() or not HALFYEAR_HOURS_INPUT.exists():
        raise FileNotFoundError(
            "Missing Netflix half-year source files. Expected both "
            f"{HALFYEAR_VIEWS_INPUT.name} and {HALFYEAR_HOURS_INPUT.name}."
        )

    views = pd.read_csv(HALFYEAR_VIEWS_INPUT, low_memory=False)
    hours = pd.read_csv(HALFYEAR_HOURS_INPUT, low_memory=False)
    views_title_column = "Title Name"
    hours_title_column = "Title Name"
    if views_title_column not in views.columns or hours_title_column not in hours.columns:
        raise KeyError("Expected `Title Name` in both half-year Netflix files.")

    views["halfyear_title_key"] = views[views_title_column].astype("string").str.strip().map(normalize_title)
    hours["halfyear_title_key"] = hours[hours_title_column].astype("string").str.strip().map(normalize_title)

    period_list = extract_available_halfyear_periods(
        [column for column in views.columns if column.endswith("Views")],
        [column for column in hours.columns if column.endswith("Hours")],
    )
    if not period_list:
        raise ValueError("No shared half-year periods were found between the views and hours files.")

    for period in period_list:
        views_column = f"{period} Views"
        hours_column = f"{period} Hours"
        views[views_column] = clean_halfyear_numeric_values(views[views_column])
        hours[hours_column] = clean_halfyear_numeric_values(hours[hours_column])

    merged = views[[views_title_column, "halfyear_title_key"] + [f"{period} Views" for period in period_list]].merge(
        hours[[hours_title_column, "halfyear_title_key"] + [f"{period} Hours" for period in period_list]],
        how="inner",
        on="halfyear_title_key",
        suffixes=("_views_source", "_hours_source"),
    )
    merged = merged.drop_duplicates(subset=["halfyear_title_key"]).copy()
    feature_values = identify_first_valid_observed_halfyear(merged, period_list)
    feature_table = pd.concat([merged[["halfyear_title_key"]].reset_index(drop=True), feature_values.reset_index(drop=True)], axis=1)

    enriched_keys = enriched["netflix_title_raw"].astype("string").str.strip().map(normalize_title)
    matched_titles = int(enriched_keys.isin(set(feature_table["halfyear_title_key"].dropna())).sum())
    unmatched_titles = int(len(enriched) - matched_titles)

    stats = {
        "periods": period_list,
        "matched_titles": matched_titles,
        "unmatched_titles": unmatched_titles,
    }
    return feature_table, stats


def create_matched_only_baseline(master_v3: pd.DataFrame) -> pd.DataFrame:
    matched_only = master_v3[master_v3["match_status"] == "matched"].copy()
    ensure_parent(MATCHED_ONLY_PARQUET)
    matched_only.to_parquet(MATCHED_ONLY_PARQUET, index=False)
    matched_only.to_csv(MATCHED_ONLY_CSV, index=False)
    return matched_only


def coalesce_columns(frame: pd.DataFrame, base_column: str, feature_column: str) -> pd.DataFrame:
    if feature_column not in frame.columns:
        return frame
    if base_column in frame.columns:
        frame[base_column] = frame[base_column].combine_first(frame[feature_column])
    else:
        frame[base_column] = frame[feature_column]
    frame.drop(columns=[feature_column], inplace=True)
    return frame


def enrichment_sources_for_row(entity_type: object) -> str:
    text = "" if entity_type is None or pd.isna(entity_type) else str(entity_type)
    sources = [
        "title.basics",
        "title.ratings",
        "title.akas",
        "title.crew",
        "title.principals",
        "name.basics",
    ]
    if text in {"series_parent", "series_season"}:
        sources.insert(2, "title.episode")
    return "|".join(sources)


def enrichment_notes_for_row(entity_type: object) -> str:
    text = "" if entity_type is None or pd.isna(entity_type) else str(entity_type)
    if text == "movie":
        return "Movie-level IMDb enrichment applied using the matched movie entity."
    if text == "series_parent":
        return "Parent-series IMDb enrichment applied without forcing a season-level join."
    if text == "series_season":
        return "Season row enriched with parent-series features plus season-level episode counts."
    return "IMDb enrichment applied."


def merge_feature_tables(matched_only: pd.DataFrame, features: dict[str, pd.DataFrame]) -> pd.DataFrame:
    enriched = matched_only.copy()
    keys = derive_enrichment_keys(enriched)
    enriched = enriched.merge(keys, how="left", on=["netflix_row_id", "imdb_season_number"])

    entity_features = features["entity_features"].copy()
    episode_aggs = features["episode_aggs"].copy()
    aka_aggs = features["aka_aggs"].copy()
    crew_aggs = features["crew_aggs"].copy()
    principal_aggs = features["principal_aggs"].copy()
    halfyear_features = features["halfyear_features"].copy()

    enriched = enriched.merge(entity_features, how="left", on="imdb_enrichment_entity_id")
    enriched = enriched.merge(aka_aggs, how="left", on="imdb_enrichment_entity_id")
    enriched = enriched.merge(crew_aggs, how="left", on="imdb_enrichment_entity_id")
    enriched = enriched.merge(principal_aggs, how="left", on="imdb_enrichment_entity_id")
    enriched["halfyear_title_key"] = enriched["netflix_title_raw"].astype("string").str.strip().map(normalize_title)
    enriched = enriched.merge(halfyear_features, how="left", on="halfyear_title_key")
    enriched.drop(columns=["halfyear_title_key"], inplace=True)

    if not episode_aggs.empty:
        parent_episode_cols = [
            "imdb_parent_tconst",
            "imdb_total_episode_count",
            "imdb_parent_season_count_feature",
            "imdb_max_season_number",
            "imdb_avg_episodes_per_season",
            "imdb_min_episodes_per_season",
            "imdb_max_episodes_per_season",
            "imdb_single_season_flag",
            "imdb_multi_season_flag",
        ]
        parent_episode = episode_aggs[parent_episode_cols].drop_duplicates(subset=["imdb_parent_tconst"])
        season_episode = episode_aggs[
            ["imdb_parent_tconst", "imdb_season_number", "imdb_season_episode_count_feature"]
        ].drop_duplicates(subset=["imdb_parent_tconst", "imdb_season_number"])
        enriched = enriched.merge(
            parent_episode,
            how="left",
            left_on="imdb_enrichment_entity_id",
            right_on="imdb_parent_tconst",
            suffixes=("", "_parent_episode"),
        )
        enriched.drop(columns=["imdb_parent_tconst_parent_episode"], errors="ignore", inplace=True)
        enriched = enriched.merge(
            season_episode,
            how="left",
            left_on=["imdb_enrichment_entity_id", "imdb_season_number"],
            right_on=["imdb_parent_tconst", "imdb_season_number"],
            suffixes=("", "_season_episode"),
        )
        enriched.drop(columns=["imdb_parent_tconst_season_episode"], errors="ignore", inplace=True)

    coalesce_map = {
        "imdb_title_type": "imdb_title_type_entity",
        "imdb_primary_title": "imdb_primary_title_entity",
        "imdb_original_title": "imdb_original_title_entity",
        "imdb_start_year": "imdb_start_year_entity",
        "imdb_end_year": "imdb_end_year_entity",
        "imdb_runtime_minutes": "imdb_runtime_minutes_entity",
        "imdb_genres": "imdb_genres_entity",
        "imdb_average_rating": "imdb_average_rating_entity",
        "imdb_num_votes": "imdb_num_votes_entity",
        "imdb_aka_title_count": "imdb_aka_title_count_feature",
        "imdb_parent_season_count": "imdb_parent_season_count_feature",
        "imdb_season_episode_count": "imdb_season_episode_count_feature",
    }
    for base_column, feature_column in coalesce_map.items():
        enriched = coalesce_columns(enriched, base_column, feature_column)

    reference_year = enriched["netflix_release_year"].combine_first(enriched["netflix_title_year_hint"])
    enriched["imdb_log_num_votes"] = np.log1p(pd.to_numeric(enriched["imdb_num_votes"], errors="coerce"))
    enriched["imdb_rating_votes_interaction"] = (
        pd.to_numeric(enriched["imdb_average_rating"], errors="coerce")
        * enriched["imdb_log_num_votes"]
    )
    enriched["imdb_movie_flag"] = enriched["imdb_title_type"].isin(["movie", "tvMovie"]).astype("Int64")
    enriched["imdb_series_flag"] = enriched["imdb_title_type"].isin(["tvSeries", "tvMiniSeries"]).astype("Int64")
    enriched["imdb_miniseries_flag"] = (enriched["imdb_title_type"] == "tvMiniSeries").astype("Int64")
    enriched["imdb_series_ended_flag"] = (
        enriched["imdb_title_type"].isin(["tvSeries", "tvMiniSeries"])
        & enriched["imdb_end_year"].notna()
        & (pd.to_numeric(enriched["imdb_end_year"], errors="coerce") < CURRENT_YEAR)
    ).astype("Int64")
    enriched["imdb_series_ongoing_flag"] = (
        enriched["imdb_title_type"].isin(["tvSeries", "tvMiniSeries"])
        & (
            enriched["imdb_end_year"].isna()
            | (pd.to_numeric(enriched["imdb_end_year"], errors="coerce") >= CURRENT_YEAR)
        )
    ).astype("Int64")
    enriched["imdb_title_age_years"] = (
        CURRENT_YEAR - pd.to_numeric(enriched["imdb_start_year"], errors="coerce")
    )
    active_end = pd.to_numeric(enriched["imdb_end_year"], errors="coerce").fillna(CURRENT_YEAR)
    enriched["imdb_years_active"] = active_end - pd.to_numeric(enriched["imdb_start_year"], errors="coerce") + 1
    enriched["netflix_imdb_year_gap"] = reference_year - pd.to_numeric(
        enriched["imdb_start_year"], errors="coerce"
    )
    enriched["netflix_imdb_runtime_gap"] = pd.to_numeric(
        enriched["netflix_runtime"], errors="coerce"
    ) - pd.to_numeric(enriched["imdb_runtime_minutes"], errors="coerce")
    enriched["imdb_enrichment_applied"] = True
    enriched["imdb_enrichment_entity_type"] = enriched["imdb_match_entity_type"]
    enriched["imdb_enrichment_sources_used"] = enriched["imdb_enrichment_entity_type"].map(
        enrichment_sources_for_row
    )
    enriched["imdb_enrichment_notes"] = enriched["imdb_enrichment_entity_type"].map(
        enrichment_notes_for_row
    )

    return enriched


def validate_outputs(master_v3: pd.DataFrame, matched_only: pd.DataFrame, enriched: pd.DataFrame, mtime_before: float | None) -> dict[str, float]:
    total_rows = len(master_v3)
    matched_rows = len(matched_only)
    unmatched_rows = total_rows - matched_rows
    matched_percent = float(matched_rows / total_rows * 100) if total_rows else 0.0

    if set(matched_only.columns) != set(master_v3.columns):
        raise ValueError("Matched-only baseline does not preserve the v3 column set.")
    if len(enriched) != len(matched_only):
        raise ValueError("Enrichment changed the matched-only row count.")
    if enriched["netflix_row_id"].duplicated().any():
        raise ValueError("Duplicate Netflix row ids were created during enrichment.")
    if not matched_only["match_status"].eq("matched").all():
        raise ValueError("Matched-only baseline contains unmatched rows.")

    if mtime_before is not None and MASTER_V3_PARQUET.exists():
        if MASTER_V3_PARQUET.stat().st_mtime != mtime_before:
            raise ValueError("The frozen v3 parquet appears to have been modified during enrichment.")

    return {
        "total_rows": total_rows,
        "matched_rows": matched_rows,
        "unmatched_rows": unmatched_rows,
        "matched_percent": matched_percent,
    }


def major_null_rates(enriched: pd.DataFrame) -> dict[str, float]:
    groups = {
        "entity_core": ["imdb_title_type", "imdb_start_year", "imdb_genres"],
        "ratings": ["imdb_average_rating", "imdb_num_votes", "imdb_log_num_votes"],
        "episodes": [
            "imdb_total_episode_count",
            "imdb_avg_episodes_per_season",
            "imdb_season_episode_count",
        ],
        "akas": ["imdb_aka_title_count", "imdb_aka_region_count", "imdb_aka_language_count"],
        "crew": ["imdb_director_count", "imdb_writer_count"],
        "principals": ["imdb_principal_count", "imdb_top_cast_names"],
    }
    summary: dict[str, float] = {}
    for group_name, columns in groups.items():
        available = [column for column in columns if column in enriched.columns]
        if not available:
            summary[group_name] = float("nan")
            continue
        null_rate = float(enriched[available].isna().mean().mean())
        summary[group_name] = null_rate
    return summary


def sample_titles(enriched: pd.DataFrame, entity_type: str) -> str:
    sample = enriched[enriched["imdb_match_entity_type"] == entity_type]["netflix_title_raw"].head(3).tolist()
    return " | ".join(sample) if sample else "none"


def log_halfyear_validation(
    matched_only: pd.DataFrame,
    enriched: pd.DataFrame,
    halfyear_stats: dict[str, object],
) -> None:
    log(
        "Half-year periods detected: " + ", ".join(str(period) for period in halfyear_stats["periods"])
    )
    log(
        f"Half-year title matches: matched={halfyear_stats['matched_titles']:,}, "
        f"unmatched={halfyear_stats['unmatched_titles']:,}"
    )
    log(
        f"Enriched master rows before/after half-year enrichment: "
        f"{len(matched_only):,} -> {len(enriched):,}"
    )
    log(f"Duplicate netflix_row_id count in enriched master: {int(enriched['netflix_row_id'].duplicated().sum()):,}")
    for column in HALFYEAR_FEATURE_COLUMNS:
        log(f"Missing count for {column}: {int(enriched[column].isna().sum()):,}")
    example_columns = [
        "title_name",
        "netflix_row_id",
        "first_observed_halfyear_period",
        "first_observed_halfyear_views",
        "first_observed_halfyear_hours",
        "first_halfyear_hours_per_view",
    ]
    available = [column for column in example_columns if column in enriched.columns]
    examples = enriched[available].head(8)
    log("Half-year feature examples:\n" + examples.to_string(index=False))


def run_enrichment(refresh: bool = False) -> pd.DataFrame:
    master_v3 = load_master_v3()
    mtime_before = MASTER_V3_PARQUET.stat().st_mtime if MASTER_V3_PARQUET.exists() else None
    matched_only = create_matched_only_baseline(master_v3)
    log(
        f"Created matched-only baseline with {len(matched_only):,} rows from {len(master_v3):,} total v3 rows."
    )

    features = prepare_all_feature_tables(matched_only, refresh=refresh)
    halfyear_features, halfyear_stats = build_halfyear_feature_table(matched_only)
    features["halfyear_features"] = halfyear_features
    enriched = merge_feature_tables(matched_only, features)

    ensure_parent(ENRICHED_PARQUET)
    enriched.to_parquet(ENRICHED_PARQUET, index=False)
    enriched.to_csv(ENRICHED_CSV, index=False)

    metrics = validate_outputs(master_v3, matched_only, enriched, mtime_before)
    null_rates = major_null_rates(enriched)
    entity_counts = enriched["imdb_match_entity_type"].fillna("unknown").value_counts().to_dict()

    log(
        "Matched-only baseline metrics: "
        f"total_v3_rows={metrics['total_rows']:,}, matched_rows={metrics['matched_rows']:,}, "
        f"unmatched_rows={metrics['unmatched_rows']:,}, matched_percent={metrics['matched_percent']:.2f}%"
    )
    log(
        "Enriched rows by entity type: "
        + ", ".join(f"{key}={value:,}" for key, value in entity_counts.items())
    )
    log(
        "Major feature-group null rates: "
        + ", ".join(f"{key}={value:.3f}" for key, value in null_rates.items())
    )
    log(f"Example enriched movie rows: {sample_titles(enriched, 'movie')}")
    log(f"Example enriched series parent rows: {sample_titles(enriched, 'series_parent')}")
    log(f"Example enriched season rows: {sample_titles(enriched, 'series_season')}")
    log_halfyear_validation(matched_only, enriched, halfyear_stats)
    log(f"Saved matched-only baseline parquet: {MATCHED_ONLY_PARQUET.relative_to(REPO_ROOT).as_posix()}")
    log(f"Saved enriched matched dataset parquet: {ENRICHED_PARQUET.relative_to(REPO_ROOT).as_posix()}")
    return enriched


def main() -> None:
    parser = argparse.ArgumentParser(description="Create matched-only baseline and enrich it with IMDb features.")
    parser.add_argument("--refresh", action="store_true", help="Refresh cached IMDb aggregate tables before enriching.")
    args = parser.parse_args()
    run_enrichment(refresh=args.refresh)


if __name__ == "__main__":
    main()
