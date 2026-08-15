from __future__ import annotations
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.pipeline_utils import ensure_parent, log  # noqa: E402


ENRICHED_PARQUET = REPO_ROOT / "data" / "processed" / "netflix_imdb_master_matched_enriched.parquet"
ENRICHED_CSV = REPO_ROOT / "data" / "processed" / "netflix_imdb_master_matched_enriched.csv"
MODELING_SOURCE_COPY_PARQUET = (
    REPO_ROOT / "data" / "processed" / "netflix_imdb_modeling_source_copy.parquet"
)
MODELING_SOURCE_COPY_CSV = REPO_ROOT / "data" / "processed" / "netflix_imdb_modeling_source_copy.csv"
MODELING_PARQUET = REPO_ROOT / "data" / "processed" / "netflix_imdb_modeling.parquet"
MODELING_CSV = REPO_ROOT / "data" / "processed" / "netflix_imdb_modeling.csv"
FEATURE_MANIFEST_CSV = (
    REPO_ROOT / "data" / "processed" / "netflix_imdb_modeling_feature_manifest.csv"
)
QUALITY_REPORT_CSV = (
    REPO_ROOT / "data" / "processed" / "netflix_imdb_modeling_quality_report.csv"
)
MODELING_DICT_DOC = REPO_ROOT / "docs" / "modeling_data_dictionary.md"
MODELING_NOTES_DOC = REPO_ROOT / "docs" / "modeling_feature_notes.md"
HALFYEAR_FEATURE_COLUMNS = [
    "first_observed_halfyear_period",
    "first_observed_halfyear_views",
    "first_observed_halfyear_hours",
    "first_halfyear_hours_per_view",
]

TARGET_COLUMNS = [
    "target_next_season_views",
    "target_next_season_hours",
    "target_view_change_absolute",
    "target_view_change_percent",
    "target_hours_change_absolute",
    "target_hours_change_percent",
    "target_is_viewership_increase",
    "target_is_hours_increase",
]
IDENTIFIER_COLUMNS = [
    "netflix_row_id",
    "series_group_key",
    "imdb_parent_tconst",
    "imdb_enrichment_entity_id",
    "imdb_resolved_tconst",
    "netflix_series_title",
    "netflix_title_raw",
    "imdb_primary_title",
    "imdb_match_entity_type",
]
SAFE_PREDICTOR_COLUMNS = [
    "netflix_views",
    "netflix_hours_viewed",
    "netflix_runtime",
    "netflix_log_views",
    "netflix_log_hours",
    "netflix_hours_per_view",
    "first_observed_halfyear_period",
    "first_observed_halfyear_views",
    "first_observed_halfyear_hours",
    "first_halfyear_hours_per_view",
    "netflix_season_number",
    "season_order",
    "season_is_first",
    "season_is_later",
    "imdb_average_rating",
    "imdb_num_votes",
    "imdb_log_num_votes",
    "imdb_rating_votes_interaction",
    "genre_count",
    "genre_drama",
    "genre_comedy",
    "genre_action",
    "genre_thriller",
    "genre_crime",
    "genre_romance",
    "genre_animation",
    "genre_documentary",
    "genre_fantasy",
    "genre_horror",
    "genre_sci_fi",
    "genre_family",
    "is_animation",
    "is_documentary",
    "is_kids_family_like",
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
    "imdb_top_cast_known_for_count_proxy",
    "imdb_top_cast_names",
    "imdb_top_cast_nconsts",
    "imdb_aka_title_count",
    "imdb_aka_region_count",
    "imdb_aka_language_count",
    "imdb_has_us_title",
    "imdb_has_international_title_variants",
    "imdb_title_type",
    "imdb_runtime_minutes",
    "imdb_is_adult",
    "imdb_movie_flag",
    "imdb_series_flag",
    "imdb_miniseries_flag",
    "netflix_imdb_year_gap",
    "netflix_imdb_runtime_gap",
    "netflix_reference_year",
    "imdb_age_at_netflix_year",
    "imdb_started_before_netflix_flag",
    "imdb_same_year_as_netflix_flag",
    "prev_season_views",
    "prev_season_hours",
    "prev_season_rating",
    "prev_view_change_absolute",
    "prev_view_change_percent",
    "prev_hours_change_absolute",
    "prev_hours_change_percent",
    "has_prev_season_observation",
    "missing_imdb_average_rating",
    "missing_imdb_num_votes",
    "missing_netflix_runtime",
    "missing_netflix_imdb_runtime_gap",
    "missing_cast_features",
    "missing_release_year",
]
LEAKAGE_COLUMNS = {
    "imdb_parent_season_count",
    "imdb_total_episode_count",
    "imdb_max_season_number",
    "imdb_avg_episodes_per_season",
    "imdb_min_episodes_per_season",
    "imdb_max_episodes_per_season",
    "imdb_single_season_flag",
    "imdb_multi_season_flag",
    "imdb_series_ended_flag",
    "imdb_series_ongoing_flag",
    "imdb_years_active",
}
AUDIT_EXACT_COLUMNS = {
    "match_status",
    "match_method",
    "match_stage",
    "match_confidence",
    "match_notes",
    "candidate_imdb_count",
    "candidate_imdb_parent_tconsts",
    "candidate_imdb_primary_titles",
    "candidate_match_source",
    "netflix_match_key_used",
    "imdb_match_key_used",
    "candidate_rank",
    "ambiguity_resolution_method",
    "year_consistency_flag",
    "year_distance",
    "title_similarity_score",
    "title_similarity_metric",
    "imdb_candidate_display_title",
    "imdb_match_key_canonical",
    "imdb_match_key_compact",
    "prior_match_status",
    "prior_match_method",
    "prior_match_stage",
    "prior_match_confidence",
    "third_pass_applied",
    "third_pass_candidate_grain",
    "third_pass_match_status",
    "third_pass_match_method",
    "third_pass_match_stage",
    "third_pass_match_confidence",
    "third_pass_match_notes",
    "movie_runtime_distance",
    "movie_year_distance",
    "movie_title_similarity_score",
    "series_parent_year_distance",
    "series_parent_title_similarity_score",
    "imdb_enrichment_applied",
    "imdb_enrichment_sources_used",
    "imdb_enrichment_notes",
}
VERBOSE_TEXT_COLUMNS = {
    "runtime_raw_source",
    "raw_netflix_type",
    "source_netflix_file",
    "netflix_runtime_raw",
    "netflix_normalized_title",
    "netflix_canonical_title",
    "netflix_compact_title",
    "netflix_raw_normalized_title",
    "netflix_raw_canonical_title",
    "netflix_raw_compact_title",
    "netflix_season_label",
    "netflix_season_parse_method",
    "netflix_title_parse_notes",
    "imdb_original_title",
    "imdb_normalized_title",
    "imdb_primary_normalized_title",
    "imdb_original_normalized_title",
    "imdb_primary_canonical_title",
    "imdb_original_canonical_title",
    "imdb_aka_normalized_titles",
    "imdb_aka_canonical_titles",
    "imdb_director_nconsts",
    "imdb_writer_nconsts",
    "imdb_director_names",
    "imdb_writer_names",
    "imdb_top_cast_profession_mix",
}
def load_enriched() -> pd.DataFrame:
    if ENRICHED_PARQUET.exists():
        return pd.read_parquet(ENRICHED_PARQUET)
    if ENRICHED_CSV.exists():
        return pd.read_csv(ENRICHED_CSV, low_memory=False)
    raise FileNotFoundError("Enriched matched-only dataset not found. Run the enrichment script first.")


def create_source_copy(enriched: pd.DataFrame) -> pd.DataFrame:
    source_copy = enriched.copy()
    ensure_parent(MODELING_SOURCE_COPY_PARQUET)
    source_copy.to_parquet(MODELING_SOURCE_COPY_PARQUET, index=False)
    source_copy.to_csv(MODELING_SOURCE_COPY_CSV, index=False)
    return source_copy


def load_source_copy() -> pd.DataFrame:
    if MODELING_SOURCE_COPY_PARQUET.exists():
        return pd.read_parquet(MODELING_SOURCE_COPY_PARQUET)
    if MODELING_SOURCE_COPY_CSV.exists():
        return pd.read_csv(MODELING_SOURCE_COPY_CSV, low_memory=False)
    raise FileNotFoundError("Modeling source copy not found.")


def numeric_series(frame: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_numeric(frame[column], errors="coerce") if column in frame.columns else pd.Series(np.nan, index=frame.index)


def build_series_group_key(frame: pd.DataFrame) -> pd.Series:
    primary = frame["imdb_parent_tconst"].astype("string")
    fallback = frame["imdb_enrichment_entity_id"].astype("string")
    second_fallback = frame["imdb_resolved_tconst"].astype("string")
    return primary.fillna(fallback).fillna(second_fallback)


def add_sort_and_time_features(frame: pd.DataFrame) -> pd.DataFrame:
    working = frame.copy()
    working["series_group_key"] = build_series_group_key(working)
    working["season_order"] = numeric_series(working, "netflix_season_number").round().astype("Int64")
    release_year = numeric_series(working, "netflix_release_year")
    title_year_hint = numeric_series(working, "netflix_title_year_hint")
    working["netflix_reference_year"] = release_year.combine_first(title_year_hint).astype("Int64")
    working["netflix_log_views"] = np.log1p(numeric_series(working, "netflix_views"))
    working["netflix_log_hours"] = np.log1p(numeric_series(working, "netflix_hours_viewed"))
    view_denominator = numeric_series(working, "netflix_views").replace(0, np.nan)
    working["netflix_hours_per_view"] = numeric_series(working, "netflix_hours_viewed") / view_denominator
    working["season_is_first"] = (working["season_order"] == 1).astype("Int64")
    working["season_is_later"] = (working["season_order"] > 1).astype("Int64")

    imdb_start_year = numeric_series(working, "imdb_start_year")
    reference_year_numeric = numeric_series(working, "netflix_reference_year")
    working["imdb_age_at_netflix_year"] = reference_year_numeric - imdb_start_year
    working["imdb_started_before_netflix_flag"] = (
        reference_year_numeric.notna() & imdb_start_year.notna() & (imdb_start_year < reference_year_numeric)
    ).astype("Int64")
    working["imdb_same_year_as_netflix_flag"] = (
        reference_year_numeric.notna() & imdb_start_year.notna() & (imdb_start_year == reference_year_numeric)
    ).astype("Int64")
    working["missing_imdb_average_rating"] = numeric_series(working, "imdb_average_rating").isna().astype("Int64")
    working["missing_imdb_num_votes"] = numeric_series(working, "imdb_num_votes").isna().astype("Int64")
    working["missing_netflix_runtime"] = numeric_series(working, "netflix_runtime").isna().astype("Int64")
    working["missing_netflix_imdb_runtime_gap"] = numeric_series(
        working, "netflix_imdb_runtime_gap"
    ).isna().astype("Int64")
    working["missing_cast_features"] = (
        numeric_series(working, "imdb_top_cast_count_used").isna()
        | numeric_series(working, "imdb_principal_count").isna()
    ).astype("Int64")
    working["missing_release_year"] = numeric_series(working, "netflix_reference_year").isna().astype("Int64")
    return working


def filter_series_season_scope(source_copy: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    working = source_copy.copy()
    metrics: dict[str, int] = {}
    metrics["copied_baseline_row_count"] = len(working)
    metrics["matched_row_count"] = int(working["match_status"].eq("matched").sum()) if "match_status" in working.columns else len(working)

    working["series_group_key"] = build_series_group_key(working)
    working["season_order"] = numeric_series(working, "netflix_season_number").round().astype("Int64")
    working["drop_missing_series_group_key"] = working["series_group_key"].isna()
    working["drop_missing_season_number"] = working["season_order"].isna() | (working["season_order"] < 1)

    eligible = working[
        working["match_status"].eq("matched")
        & working["imdb_match_entity_type"].eq("series_season")
        & ~working["drop_missing_series_group_key"]
        & ~working["drop_missing_season_number"]
    ].copy()
    metrics["series_season_eligible_row_count"] = len(eligible)
    metrics["rows_dropped_missing_group_or_season"] = int(len(working) - len(eligible) - (len(working) - metrics["matched_row_count"]))

    duplicate_mask = eligible.groupby(["series_group_key", "season_order"])["netflix_row_id"].transform("size") > 1
    eligible["drop_invalid_season_structure"] = duplicate_mask
    metrics["rows_dropped_invalid_season_structure"] = int(duplicate_mask.sum())

    strict = eligible[~eligible["drop_invalid_season_structure"]].copy()
    strict = add_sort_and_time_features(strict)
    sort_year = numeric_series(strict, "netflix_reference_year").fillna(9999)
    strict["sort_reference_year"] = sort_year
    strict.sort_values(
        ["series_group_key", "season_order", "sort_reference_year", "netflix_row_id"],
        inplace=True,
    )
    strict.drop(columns=["sort_reference_year"], inplace=True)
    return strict, metrics


def add_targets_and_lags(strict: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    modeling = strict.copy()
    grouping = modeling.groupby("series_group_key", sort=False)

    current_views = numeric_series(modeling, "netflix_views")
    current_hours = numeric_series(modeling, "netflix_hours_viewed")

    modeling["next_season_number"] = grouping["season_order"].shift(-1)
    modeling["target_next_season_views"] = grouping["netflix_views"].shift(-1)
    modeling["target_next_season_hours"] = grouping["netflix_hours_viewed"].shift(-1)
    modeling["prev_season_views"] = grouping["netflix_views"].shift(1)
    modeling["prev_season_hours"] = grouping["netflix_hours_viewed"].shift(1)
    modeling["prev_season_rating"] = grouping["imdb_average_rating"].shift(1)
    modeling["prev_prev_season_views"] = grouping["netflix_views"].shift(2)
    modeling["prev_prev_season_hours"] = grouping["netflix_hours_viewed"].shift(2)
    modeling["has_prev_season_observation"] = modeling["prev_season_views"].notna().astype("Int64")

    prev_views = pd.to_numeric(modeling["prev_season_views"], errors="coerce")
    prev_hours = pd.to_numeric(modeling["prev_season_hours"], errors="coerce")
    prev_prev_views = pd.to_numeric(modeling["prev_prev_season_views"], errors="coerce")
    prev_prev_hours = pd.to_numeric(modeling["prev_prev_season_hours"], errors="coerce")
    modeling["prev_view_change_absolute"] = prev_views - prev_prev_views
    modeling["prev_hours_change_absolute"] = prev_hours - prev_prev_hours
    modeling["prev_view_change_percent"] = modeling["prev_view_change_absolute"] / prev_prev_views.replace(0, np.nan)
    modeling["prev_hours_change_percent"] = modeling["prev_hours_change_absolute"] / prev_prev_hours.replace(0, np.nan)

    modeling["target_view_change_absolute"] = pd.to_numeric(
        modeling["target_next_season_views"], errors="coerce"
    ) - current_views
    modeling["target_hours_change_absolute"] = pd.to_numeric(
        modeling["target_next_season_hours"], errors="coerce"
    ) - current_hours
    modeling["target_view_change_percent"] = modeling["target_view_change_absolute"] / current_views.replace(0, np.nan)
    modeling["target_hours_change_percent"] = modeling["target_hours_change_absolute"] / current_hours.replace(0, np.nan)
    modeling["target_is_viewership_increase"] = (
        pd.to_numeric(modeling["target_next_season_views"], errors="coerce") > current_views
    ).astype("Int64")
    modeling["target_is_hours_increase"] = (
        pd.to_numeric(modeling["target_next_season_hours"], errors="coerce") > current_hours
    ).astype("Int64")

    valid_next_season = (
        modeling["next_season_number"].notna()
        & ((modeling["next_season_number"] - modeling["season_order"]) == 1)
    )
    valid_current_metrics = current_views.notna() & current_hours.notna()
    valid_next_metrics = (
        pd.to_numeric(modeling["target_next_season_views"], errors="coerce").notna()
        & pd.to_numeric(modeling["target_next_season_hours"], errors="coerce").notna()
    )
    modeling["drop_missing_current_metrics"] = ~valid_current_metrics
    modeling["drop_missing_next_target"] = ~(valid_next_season & valid_next_metrics)

    metrics = {
        "rows_dropped_missing_current_metrics": int(modeling["drop_missing_current_metrics"].sum()),
        "rows_dropped_missing_next_target": int(
            (~modeling["drop_missing_current_metrics"] & modeling["drop_missing_next_target"]).sum()
        ),
    }

    final_rows = modeling[
        ~modeling["drop_missing_current_metrics"] & ~modeling["drop_missing_next_target"]
    ].copy()
    final_rows.drop(columns=["prev_prev_season_views", "prev_prev_season_hours", "next_season_number"], inplace=True)
    return final_rows, metrics


def feature_group_for_column(column: str) -> str:
    if column in IDENTIFIER_COLUMNS or column in {"season_order", "series_group_key"}:
        return "identifier"
    if column in TARGET_COLUMNS:
        return "target"
    if column.startswith("missing_"):
        return "missingness"
    if column.startswith("prev_") or column == "has_prev_season_observation":
        return "lag"
    if column.startswith("genre_") or column in {"is_animation", "is_documentary", "is_kids_family_like"}:
        return "genre"
    if column.startswith("imdb_aka_") or column.startswith("imdb_has_"):
        return "internationalization"
    if column.startswith("imdb_director_") or column.startswith("imdb_writer_") or column.startswith("imdb_principal_") or column.startswith("imdb_actor_") or column.startswith("imdb_actress_") or column.startswith("imdb_self_") or column.startswith("imdb_producer_") or column.startswith("imdb_top_cast_"):
        return "cast_crew"
    if column.startswith("imdb_") and ("rating" in column or "votes" in column):
        return "imdb_quality"
    if column.startswith("netflix_") and any(token in column for token in ["views", "hours", "runtime", "log"]):
        return "netflix_current"
    if column.startswith("first_observed_halfyear_") or column == "first_halfyear_hours_per_view":
        return "netflix_halfyear_observed"
    if column in {"netflix_season_number", "season_is_first", "season_is_later"}:
        return "season_structure"
    if column in {"netflix_imdb_year_gap", "netflix_imdb_runtime_gap", "imdb_age_at_netflix_year", "imdb_started_before_netflix_flag", "imdb_same_year_as_netflix_flag"}:
        return "timing"
    if column.startswith("imdb_"):
        return "imdb_static"
    if column.startswith("target_"):
        return "target"
    return "other"


def drop_reason_for_column(column: str) -> tuple[str, bool]:
    if column in LEAKAGE_COLUMNS:
        return "leakage_prone_future_summary", True
    if column in AUDIT_EXACT_COLUMNS or column.startswith("prior_match_") or column.startswith("third_pass_"):
        return "matching_audit_or_debug", False
    if column in VERBOSE_TEXT_COLUMNS:
        return "verbose_text_or_key_blob", False
    if column.startswith("target_"):
        return "", False
    if column.startswith("drop_"):
        return "row_filter_debug_flag", False
    if column in {"movie_runtime_distance", "movie_year_distance", "movie_title_similarity_score", "series_parent_year_distance", "series_parent_title_similarity_score"}:
        return "out_of_scope_non_series_season_debug", False
    if column in {"imdb_movie_flag"}:
        return "", False
    if column in {"match_status", "imdb_match_entity_type", "netflix_content_grain"}:
        return "", False
    if column.startswith("2023_") or column.startswith("2024_") or column.startswith("2025_") or column.startswith("2026_") or column in {"total_hours_viewed", "total_views"}:
        return "raw_multi_period_snapshot_not_used", False
    if column in {"netflix_release_date", "netflix_available_globally", "title_name"}:
        return "unused_or_sparse_metadata", False
    return "", False


def create_feature_manifest(modeling_base: pd.DataFrame, final_columns: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    final_column_set = set(final_columns)

    for column in modeling_base.columns:
        drop_reason, leakage_flag = drop_reason_for_column(column)
        if column in IDENTIFIER_COLUMNS or column in {"season_order"}:
            role = "identifier"
            included = column in final_column_set
        elif column in TARGET_COLUMNS:
            role = "target"
            included = column in final_column_set
        elif column in SAFE_PREDICTOR_COLUMNS:
            role = "predictor"
            included = column in final_column_set
        else:
            role = "dropped"
            included = False
            if not drop_reason:
                drop_reason = "not_in_strict_model_feature_set"

        rows.append(
            {
                "column_name": column,
                "role": role,
                "included_in_final_modeling_dataset": bool(included),
                "drop_reason": drop_reason if not included else "",
                "leakage_flag": bool(leakage_flag),
                "missingness_rate": float(modeling_base[column].isna().mean()),
                "feature_group": feature_group_for_column(column),
                "notes": feature_note_for_column(column, included, leakage_flag),
            }
        )

    manifest = pd.DataFrame(rows).sort_values(["role", "column_name"]).reset_index(drop=True)
    return manifest


def feature_note_for_column(column: str, included: bool, leakage_flag: bool) -> str:
    if column in TARGET_COLUMNS:
        return "Next-season supervised target derived from the subsequent observed season within the same series."
    if column in IDENTIFIER_COLUMNS or column == "season_order":
        return "Retained for grouping, ordering, or traceability."
    if leakage_flag:
        return "Excluded from the strict modeling matrix because it uses full-series future information."
    if column.startswith("prev_") or column == "has_prev_season_observation":
        return "Lag feature computed only from prior observed seasons within the same series."
    if column.startswith("missing_"):
        return "Missingness indicator retained because absence may be informative."
    if included:
        return "Included as a leakage-aware predictor for the strict season-to-season model."
    return "Excluded from the strict modeling matrix."


def final_column_order(modeling_base: pd.DataFrame) -> list[str]:
    included = [column for column in IDENTIFIER_COLUMNS if column in modeling_base.columns]
    included += [column for column in ["netflix_season_number", "season_order"] if column in modeling_base.columns and column not in included]
    included += [column for column in SAFE_PREDICTOR_COLUMNS if column in modeling_base.columns and column not in included]
    included += [column for column in TARGET_COLUMNS if column in modeling_base.columns and column not in included]
    return included


def create_quality_report(
    enriched: pd.DataFrame,
    source_copy: pd.DataFrame,
    exact_row_order_copy: bool,
    final_modeling: pd.DataFrame,
    metrics: dict[str, Any],
    manifest: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = [
        {"metric": "original_enriched_input_row_count", "value": int(len(enriched)), "notes": "Frozen enriched matched input."},
        {"metric": "copied_baseline_row_count", "value": int(len(source_copy)), "notes": "Exact modeling source snapshot."},
        {"metric": "copied_baseline_exact_row_copy", "value": int(len(enriched) == len(source_copy)), "notes": "Source copy row count matches enriched input."},
        {"metric": "copied_baseline_exact_row_order", "value": int(exact_row_order_copy), "notes": "Source copy preserves the enriched input row order."},
        {"metric": "matched_row_count_in_copy", "value": int(source_copy["match_status"].eq("matched").sum()), "notes": "Copy is already matched-only in the current repo state."},
        {"metric": "series_season_eligible_row_count", "value": int(metrics["series_season_eligible_row_count"]), "notes": "Matched rows with a stable series key and valid season number."},
        {"metric": "rows_dropped_invalid_season_structure", "value": int(metrics["rows_dropped_invalid_season_structure"]), "notes": "Dropped because the same series had duplicate season numbers."},
        {"metric": "rows_dropped_missing_current_metrics", "value": int(metrics["rows_dropped_missing_current_metrics"]), "notes": "Dropped because current-season views or hours were missing."},
        {"metric": "rows_dropped_missing_target", "value": int(metrics["rows_dropped_missing_next_target"]), "notes": "Dropped because no consecutive next season with usable targets was available."},
        {"metric": "final_modeling_row_count", "value": int(len(final_modeling)), "notes": "Strict season-to-season supervised rows."},
        {"metric": "final_modeling_unique_series", "value": int(final_modeling["series_group_key"].nunique()), "notes": "Unique series grouping keys in the strict modeling dataset."},
        {"metric": "predictor_column_count", "value": int((manifest["role"] == "predictor").sum()), "notes": "Predictor columns retained in the strict modeling dataset."},
        {"metric": "target_column_count", "value": int((manifest["role"] == "target").sum()), "notes": "Supervised target columns in the strict modeling dataset."},
        {"metric": "identifier_column_count", "value": int((manifest["role"] == "identifier").sum()), "notes": "Identifier columns retained for traceability."},
        {"metric": "dropped_column_count", "value": int((manifest["role"] == "dropped").sum()), "notes": "Columns excluded from the strict modeling matrix."},
    ]

    for column in [
        "netflix_views",
        "netflix_hours_viewed",
        "imdb_average_rating",
        "imdb_num_votes",
        "netflix_runtime",
        "prev_season_views",
        "prev_season_hours",
        "imdb_top_cast_known_for_count_proxy",
        "first_observed_halfyear_period",
        "first_observed_halfyear_views",
        "first_observed_halfyear_hours",
        "first_halfyear_hours_per_view",
    ]:
        if column in final_modeling.columns:
            rows.append(
                {
                    "metric": f"null_rate::{column}",
                    "value": float(final_modeling[column].isna().mean()),
                    "notes": "Predictor null rate within the final modeling dataset.",
                }
            )

    for target in ["target_is_viewership_increase", "target_is_hours_increase"]:
        if target in final_modeling.columns:
            positive_rate = float(pd.to_numeric(final_modeling[target], errors="coerce").mean())
            rows.append(
                {
                    "metric": f"class_balance::{target}",
                    "value": positive_rate,
                    "notes": "Mean of the binary target within the final modeling dataset.",
                }
            )

    return pd.DataFrame(rows)


def log_halfyear_modeling_validation(
    source_copy: pd.DataFrame,
    final_modeling: pd.DataFrame,
    prior_modeling_rows: int,
) -> None:
    log(
        f"Modeling rows before/after half-year propagation: "
        f"{prior_modeling_rows:,} -> {len(final_modeling):,}"
    )
    log(
        f"Duplicate netflix_row_id count in modeling dataset: "
        f"{int(final_modeling['netflix_row_id'].duplicated().sum()):,}"
    )
    for column in HALFYEAR_FEATURE_COLUMNS:
        if column in source_copy.columns:
            log(f"Modeling source missing count for {column}: {int(source_copy[column].isna().sum()):,}")
        if column in final_modeling.columns:
            log(f"Modeling final missing count for {column}: {int(final_modeling[column].isna().sum()):,}")
    example_title_column = "title_name" if "title_name" in source_copy.columns else "netflix_title_raw"
    available = [
        column
        for column in [
            example_title_column,
            "netflix_row_id",
            "first_observed_halfyear_period",
            "first_observed_halfyear_views",
            "first_observed_halfyear_hours",
            "first_halfyear_hours_per_view",
        ]
        if column in source_copy.columns
    ]
    examples = source_copy[available].drop_duplicates(subset=["netflix_row_id"]).head(8)
    log("Modeling half-year feature examples:\n" + examples.to_string(index=False))


def validate_pipeline(
    enriched: pd.DataFrame,
    source_copy: pd.DataFrame,
    final_modeling: pd.DataFrame,
    enriched_mtime_before: float | None,
) -> bool:
    if len(source_copy) != len(enriched):
        raise ValueError("Modeling source copy is not a row-for-row copy of the enriched input.")
    exact_row_order_copy = source_copy["netflix_row_id"].reset_index(drop=True).equals(
        enriched["netflix_row_id"].reset_index(drop=True)
    )
    if not exact_row_order_copy:
        raise ValueError("Modeling source copy does not preserve the enriched input row order.")
    if final_modeling["netflix_row_id"].duplicated().any():
        raise ValueError("Duplicate Netflix row ids were introduced in the final modeling dataset.")
    if final_modeling.empty:
        raise ValueError("Final modeling dataset is empty.")
    if final_modeling[TARGET_COLUMNS].isna().any().any():
        raise ValueError("Final modeling rows are missing required next-season targets.")
    if not final_modeling["imdb_match_entity_type"].eq("series_season").all():
        raise ValueError("Final modeling dataset contains non-series-season rows.")
    if set(LEAKAGE_COLUMNS).intersection(final_modeling.columns):
        raise ValueError("Leakage-prone full-series summary fields were retained in the strict modeling dataset.")
    if enriched_mtime_before is not None and ENRICHED_PARQUET.exists():
        if ENRICHED_PARQUET.stat().st_mtime != enriched_mtime_before:
            raise ValueError("The enriched input parquet was modified during modeling.")
    return exact_row_order_copy


def write_modeling_docs(manifest: pd.DataFrame) -> None:
    included_predictors = manifest[
        (manifest["role"] == "predictor") & (manifest["included_in_final_modeling_dataset"])
    ]["column_name"].tolist()
    leakage_drops = manifest[manifest["leakage_flag"]]["column_name"].tolist()

    ensure_parent(MODELING_DICT_DOC)
    MODELING_DICT_DOC.write_text(
        "# Modeling Data Dictionary\n\n"
        "## Scope\n\n"
        "The strict modeling dataset is built from the frozen enriched matched dataset copy "
        "`data/processed/netflix_imdb_modeling_source_copy.*` and only keeps matched `series_season` rows "
        "that have a stable series key, a valid season number, usable current-season Netflix metrics, "
        "and an observed consecutive next season.\n\n"
        "## Core Identifiers\n\n"
        "- `netflix_row_id`: Stable Netflix row id.\n"
        "- `series_group_key`: Stable within-series grouping key used for target and lag construction.\n"
        "- `imdb_parent_tconst`: Preferred stable parent-series identifier.\n"
        "- `imdb_enrichment_entity_id`: Parent-series enrichment join key for the modeling dataset.\n\n"
        "## Target Definitions\n\n"
        "- `target_next_season_views`: Next observed season's `netflix_views`.\n"
        "- `target_next_season_hours`: Next observed season's `netflix_hours_viewed`.\n"
        "- `target_view_change_absolute`: `target_next_season_views - netflix_views`.\n"
        "- `target_view_change_percent`: `target_view_change_absolute / netflix_views`.\n"
        "- `target_hours_change_absolute`: `target_next_season_hours - netflix_hours_viewed`.\n"
        "- `target_hours_change_percent`: `target_hours_change_absolute / netflix_hours_viewed`.\n"
        "- `target_is_viewership_increase`: `1` when next-season views are higher than current-season views.\n"
        "- `target_is_hours_increase`: `1` when next-season hours are higher than current-season hours.\n\n"
        "## Included Predictor Count\n\n"
        f"- Predictors retained in the strict modeling dataset: `{len(included_predictors)}`\n\n"
        "## Leakage Policy\n\n"
        "The strict modeling dataset excludes full-series future-summary fields such as total observed seasons, "
        "full-run episode totals, and lifecycle fields that depend on knowledge beyond the current season row.\n",
        encoding="utf-8",
    )

    MODELING_NOTES_DOC.write_text(
        "# Modeling Feature Notes\n\n"
        "## Copy-First Workflow\n\n"
        "1. Load the frozen enriched matched dataset.\n"
        "2. Write an exact row-for-row source snapshot to `data/processed/netflix_imdb_modeling_source_copy.*`.\n"
        "3. Build the strict modeling dataset only from that copied snapshot.\n\n"
        "## Leakage-Prone Columns Excluded\n\n"
        + "\n".join(f"- `{column}`" for column in leakage_drops)
        + "\n\n## Lag Features\n\n"
        "Lag features use only prior observed seasons within the same `series_group_key`. "
        "Rows without a consecutive next season are removed from the final supervised dataset.\n\n"
        "## Feature Manifest\n\n"
        "See `data/processed/netflix_imdb_modeling_feature_manifest.csv` for per-column inclusion, "
        "missingness, drop reasons, and leakage flags.\n",
        encoding="utf-8",
    )


def build_modeling_dataset() -> pd.DataFrame:
    enriched_mtime_before = ENRICHED_PARQUET.stat().st_mtime if ENRICHED_PARQUET.exists() else None
    enriched = load_enriched()
    prior_modeling_rows = 0
    if MODELING_PARQUET.exists():
        prior_modeling_rows = len(pd.read_parquet(MODELING_PARQUET))
    elif MODELING_CSV.exists():
        prior_modeling_rows = len(pd.read_csv(MODELING_CSV, low_memory=False))
    source_copy = create_source_copy(enriched)
    source_copy_loaded = load_source_copy()

    strict_scope, metrics = filter_series_season_scope(source_copy_loaded)
    modeling_base, target_metrics = add_targets_and_lags(strict_scope)
    metrics.update(target_metrics)

    final_columns = final_column_order(modeling_base)
    final_modeling = modeling_base[final_columns].copy()
    manifest = create_feature_manifest(modeling_base, final_columns)
    exact_row_order_copy = validate_pipeline(
        enriched, source_copy_loaded, final_modeling, enriched_mtime_before
    )
    quality_report = create_quality_report(
        enriched=enriched,
        source_copy=source_copy_loaded,
        exact_row_order_copy=exact_row_order_copy,
        final_modeling=final_modeling,
        metrics=metrics,
        manifest=manifest,
    )

    ensure_parent(MODELING_PARQUET)
    final_modeling.to_parquet(MODELING_PARQUET, index=False)
    final_modeling.to_csv(MODELING_CSV, index=False)
    manifest.to_csv(FEATURE_MANIFEST_CSV, index=False)
    quality_report.to_csv(QUALITY_REPORT_CSV, index=False)
    write_modeling_docs(manifest)

    predictor_count = int((manifest["role"] == "predictor").sum())
    target_count = int((manifest["role"] == "target").sum())
    log(
        f"Saved modeling source copy with {len(source_copy_loaded):,} rows; strict modeling dataset has "
        f"{len(final_modeling):,} rows across {final_modeling['series_group_key'].nunique():,} series."
    )
    log(
        f"Retained {predictor_count:,} predictor columns and created {target_count:,} target columns."
    )
    log_halfyear_modeling_validation(source_copy_loaded, final_modeling, prior_modeling_rows)
    return final_modeling


def main() -> None:
    build_modeling_dataset()


if __name__ == "__main__":
    main()
