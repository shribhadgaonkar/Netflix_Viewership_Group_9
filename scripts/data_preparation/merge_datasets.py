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
    coerce_nullable_int,
    ensure_parent,
    interim_dir,
    log,
    processed_dir,
    year_consistency,
    year_distance,
)


NETFLIX_INPUT = interim_dir() / "netflix_cleaned.csv"
IMDB_INPUT = interim_dir() / "imdb_series_seasons.csv"
IMDB_TITLE_KEYS_INPUT = interim_dir() / "imdb_title_keys.csv"
MANUAL_OVERRIDE_INPUT = REPO_ROOT / "config" / "manual_match_overrides.csv"
MASTER_OUTPUT = processed_dir() / "netflix_imdb_master.parquet"
MASTER_CSV_OUTPUT = processed_dir() / "netflix_imdb_master.csv"
SAMPLE_OUTPUT = processed_dir() / "netflix_imdb_master_sample.csv"
UNMATCHED_SERIES_OUTPUT = processed_dir() / "unmatched_series_rows.csv"
AMBIGUOUS_REVIEW_OUTPUT = processed_dir() / "ambiguous_match_candidates.csv"
FUZZY_REVIEW_OUTPUT = processed_dir() / "matched_by_fuzzy_review.csv"
ALTERNATE_REVIEW_OUTPUT = processed_dir() / "matched_by_alternate_title_review.csv"
UNMATCHED_METHOD_SUMMARY_OUTPUT = processed_dir() / "unmatched_match_method_summary.csv"
MATCH_SUMMARY_OUTPUT = processed_dir() / "match_improvement_summary.csv"

ENABLE_FUZZY_MATCHING = True
FUZZY_SCORE_THRESHOLD = 97.0
FUZZY_MIN_SCORE_GAP = 2.0

MATCH_CONFIDENCE_BY_STAGE = {
    "stage_a_primary_exact": 0.99,
    "stage_b_original_exact": 0.97,
    "stage_c_aka_exact": 0.96,
    "stage_d_raw_title_variant_exact": 0.95,
    "stage_d_canonical_exact": 0.94,
    "stage_d_raw_canonical_exact": 0.93,
    "stage_d_compact_exact": 0.92,
    "stage_d_raw_compact_exact": 0.91,
    "stage_single_season_inference": 0.9,
    "stage_e_fuzzy": 0.88,
    "manual_override": 1.0,
}
MATCH_METHOD_BY_STAGE = {
    "stage_a_primary_exact": "exact_primary_title_season",
    "stage_b_original_exact": "exact_original_title_season",
    "stage_c_aka_exact": "exact_aka_title_season",
    "stage_d_raw_title_variant_exact": "exact_raw_title_variant_season",
    "stage_d_canonical_exact": "exact_canonical_title_season",
    "stage_d_raw_canonical_exact": "exact_raw_canonical_title_season",
    "stage_d_compact_exact": "exact_compact_title_season",
    "stage_d_raw_compact_exact": "exact_raw_compact_title_season",
    "stage_single_season_inference": "exact_single_season_inference",
    "stage_e_fuzzy": "fuzzy_title_season",
    "manual_override": "manual_override",
}
UNMATCHED_MEANINGS = {
    "unmatched_not_series_like": "Netflix row was classified as a movie or otherwise not a season-level series row, so the series-season join was not attempted.",
    "unmatched_missing_season_number": "The row appears series-related, but the pipeline could not extract a season number from the Netflix title text.",
    "unmatched_no_exact_title_season_match": "A normalized title and season number were available, but no IMDb row matched that exact title-plus-season key.",
    "unmatched_ambiguous_exact_match": "Multiple IMDb rows matched the same exact normalized title and season number, so the pipeline left the row unmatched rather than guessing.",
    "unmatched_ambiguous_year_conflict": "Multiple exact IMDb candidates existed, but the Netflix title year hint conflicted with all of them, so none could be selected.",
    "unmatched_year_conflict": "A single exact IMDb candidate existed, but its series years conflicted with the Netflix year hint.",
}


def load_baseline_master() -> pd.DataFrame | None:
    if MASTER_OUTPUT.exists():
        return pd.read_parquet(MASTER_OUTPUT)
    if MASTER_CSV_OUTPUT.exists():
        return pd.read_csv(MASTER_CSV_OUTPUT, low_memory=False)
    return None


def load_existing_summary_metrics() -> dict[str, float]:
    if not MATCH_SUMMARY_OUTPUT.exists():
        return {}

    summary = pd.read_csv(MATCH_SUMMARY_OUTPUT)
    if not {"metric", "value"}.issubset(summary.columns):
        return {}

    metrics: dict[str, float] = {}
    for _, row in summary.iterrows():
        metrics[str(row["metric"])] = row["value"]
    return metrics


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    missing = [path for path in [NETFLIX_INPUT, IMDB_INPUT, IMDB_TITLE_KEYS_INPUT] if not path.exists()]
    if missing:
        missing_display = ", ".join(str(path) for path in missing)
        raise FileNotFoundError(
            f"Missing required prepared inputs: {missing_display}. "
            "Run the Netflix and IMDb preparation scripts first."
        )

    netflix = pd.read_csv(NETFLIX_INPUT, low_memory=False)
    imdb = pd.read_csv(IMDB_INPUT, low_memory=False)
    imdb_title_keys = pd.read_csv(IMDB_TITLE_KEYS_INPUT, low_memory=False)

    netflix["netflix_row_id"] = coerce_nullable_int(netflix["netflix_row_id"])
    netflix["netflix_season_number"] = coerce_nullable_int(netflix["netflix_season_number"])
    netflix["netflix_release_year"] = coerce_nullable_int(netflix["netflix_release_year"])
    netflix["netflix_title_year_hint"] = coerce_nullable_int(netflix["netflix_title_year_hint"])
    if "netflix_season_parse_confidence" in netflix.columns:
        netflix["netflix_season_parse_confidence"] = pd.to_numeric(
            netflix["netflix_season_parse_confidence"], errors="coerce"
        )

    imdb["imdb_season_number"] = coerce_nullable_int(imdb["imdb_season_number"])
    imdb["imdb_start_year"] = coerce_nullable_int(imdb["imdb_start_year"])
    imdb["imdb_end_year"] = coerce_nullable_int(imdb["imdb_end_year"])
    imdb["imdb_num_votes"] = coerce_nullable_int(imdb["imdb_num_votes"])
    imdb["imdb_season_episode_count"] = coerce_nullable_int(imdb["imdb_season_episode_count"])
    imdb["imdb_parent_season_count"] = coerce_nullable_int(imdb["imdb_parent_season_count"])

    imdb_title_keys = imdb_title_keys.copy()
    for column in [
        "imdb_match_key_used",
        "imdb_match_key_canonical",
        "imdb_match_key_compact",
        "candidate_match_source",
    ]:
        imdb_title_keys[column] = imdb_title_keys[column].astype("string")

    return netflix, imdb, imdb_title_keys


def build_candidate_pool(imdb: pd.DataFrame, imdb_title_keys: pd.DataFrame) -> pd.DataFrame:
    candidate_pool = imdb.merge(imdb_title_keys, how="left", on="imdb_parent_tconst")
    candidate_pool = candidate_pool[
        candidate_pool["imdb_match_key_used"].notna() & candidate_pool["imdb_season_number"].notna()
    ].copy()
    candidate_pool["candidate_match_source"] = candidate_pool["candidate_match_source"].astype("string")
    candidate_pool["source_priority"] = candidate_pool["candidate_match_source"].map(MATCH_SOURCE_PRIORITY)
    candidate_pool.drop_duplicates(
        subset=[
            "imdb_parent_tconst",
            "imdb_season_number",
            "candidate_match_source",
            "imdb_match_key_used",
        ],
        inplace=True,
    )
    candidate_pool.reset_index(drop=True, inplace=True)
    return candidate_pool


def build_lookup(candidate_pool: pd.DataFrame, key_col: str, sources: list[str] | None = None) -> dict[tuple[str, int], list[int]]:
    subset = candidate_pool
    if sources is not None:
        subset = subset[subset["candidate_match_source"].isin(sources)]
    subset = subset[subset[key_col].notna() & subset["imdb_season_number"].notna()].copy()

    lookup: dict[tuple[str, int], list[int]] = {}
    grouped = subset.groupby([key_col, "imdb_season_number"], sort=False).groups
    for (key_value, season_number), indices in grouped.items():
        if pd.isna(key_value) or pd.isna(season_number):
            continue
        lookup[(str(key_value), int(season_number))] = list(indices)
    return lookup


def build_single_season_lookup(candidate_pool: pd.DataFrame, key_col: str, sources: list[str] | None = None) -> dict[str, list[int]]:
    single_season_mask = candidate_pool["imdb_parent_season_count"].fillna(0).astype("Int64") == 1
    miniseries_mask = candidate_pool["imdb_title_type"].astype("string") == "tvMiniSeries"
    subset = candidate_pool[
        (candidate_pool["imdb_season_number"] == 1)
        & (single_season_mask | miniseries_mask)
    ].copy()
    if sources is not None:
        subset = subset[subset["candidate_match_source"].isin(sources)]
    subset = subset[subset[key_col].notna()].copy()

    lookup: dict[str, list[int]] = {}
    grouped = subset.groupby(key_col, sort=False).groups
    for key_value, indices in grouped.items():
        if pd.isna(key_value):
            continue
        lookup[str(key_value)] = list(indices)
    return lookup


def compact_candidate_values(series: pd.Series) -> str | None:
    values = [str(value) for value in pd.unique(series.dropna()) if str(value).strip()]
    return " | ".join(values) if values else None


def get_reference_year(row: pd.Series) -> int | None:
    for column in ["netflix_release_year", "netflix_title_year_hint"]:
        value = row.get(column)
        if value is not None and not pd.isna(value):
            return int(value)
    return None


def title_type_priority(row: pd.Series, imdb_title_type: Any) -> int:
    title_type = "" if imdb_title_type is None or pd.isna(imdb_title_type) else str(imdb_title_type)
    season_label = "" if pd.isna(row.get("netflix_season_label")) else str(row.get("netflix_season_label"))
    if season_label == "limited_series":
        if title_type == "tvMiniSeries":
            return 0
        if title_type == "tvSeries":
            return 1
        return 2
    if title_type == "tvSeries":
        return 0
    if title_type == "tvMiniSeries":
        return 1
    return 2


def deduplicate_candidates_by_parent(candidates: pd.DataFrame) -> pd.DataFrame:
    ordered = candidates.sort_values(
        ["source_priority", "candidate_match_source", "imdb_match_key_used", "imdb_parent_tconst"]
    )
    return ordered.drop_duplicates(subset=["imdb_parent_tconst", "imdb_season_number"], keep="first").copy()


def build_default_result(imdb_columns: list[str]) -> dict[str, object]:
    result: dict[str, object] = {column: pd.NA for column in imdb_columns}
    result.update(
        {
            "match_status": "unmatched",
            "match_method": "unmatched_no_exact_title_season_match",
            "match_stage": "unresolved",
            "match_confidence": 0.0,
            "match_notes": pd.NA,
            "candidate_imdb_count": 0,
            "candidate_imdb_parent_tconsts": pd.NA,
            "candidate_imdb_primary_titles": pd.NA,
            "candidate_match_source": pd.NA,
            "netflix_match_key_used": pd.NA,
            "imdb_match_key_used": pd.NA,
            "candidate_rank": pd.NA,
            "ambiguity_resolution_method": pd.NA,
            "year_consistency_flag": pd.NA,
            "year_distance": pd.NA,
            "title_similarity_score": pd.NA,
            "title_similarity_metric": pd.NA,
        }
    )
    return result


def resolve_method(method: str, stage_name: str, base_confidence: float, candidate_count: int) -> float:
    confidence = base_confidence
    if method == "single_candidate":
        confidence += 0.01
    elif method in {"source_priority", "year_consistency", "exact_start_year", "closest_start_year"}:
        confidence += 0.005
    elif method in {"title_type_preference", "higher_num_votes"}:
        confidence -= 0.02
    elif stage_name == "stage_e_fuzzy":
        confidence -= 0.01 * max(candidate_count - 1, 0)
    return float(max(0.0, min(1.0, confidence)))


def resolve_candidate_group(
    row: pd.Series,
    candidates: pd.DataFrame,
    stage_name: str,
    match_method: str,
    netflix_key_used: str,
    title_similarity_metric: str,
) -> tuple[dict[str, object] | None, dict[str, object], pd.DataFrame]:
    candidate_frame = deduplicate_candidates_by_parent(candidates)
    reference_year = get_reference_year(row)
    candidate_frame["year_consistency_flag"] = [
        year_consistency(reference_year, start_year, end_year)
        for start_year, end_year in zip(
            candidate_frame["imdb_start_year"], candidate_frame["imdb_end_year"], strict=False
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
    candidate_frame["title_type_priority"] = [
        title_type_priority(row, title_type) for title_type in candidate_frame["imdb_title_type"]
    ]
    candidate_frame["year_consistency_rank"] = candidate_frame["year_consistency_flag"].map(
        {True: 0, False: 2}
    )
    candidate_frame["year_consistency_rank"] = candidate_frame["year_consistency_rank"].fillna(1)
    candidate_frame["exact_start_year_rank"] = candidate_frame["exact_start_year_match"].map(
        {True: 0, False: 1}
    )
    candidate_frame["year_distance_rank"] = candidate_frame["year_distance"].fillna(9999.0)
    candidate_frame["title_similarity_score"] = pd.to_numeric(
        candidate_frame["title_similarity_score"], errors="coerce"
    ).fillna(100.0)
    candidate_frame["title_similarity_rank"] = -candidate_frame["title_similarity_score"]
    candidate_frame["votes_rank"] = -candidate_frame["imdb_num_votes"].fillna(-1)
    candidate_frame["source_priority"] = candidate_frame["candidate_match_source"].map(
        MATCH_SOURCE_PRIORITY
    ).fillna(9)
    candidate_frame.sort_values(
        [
            "source_priority",
            "year_consistency_rank",
            "exact_start_year_rank",
            "year_distance_rank",
            "title_type_priority",
            "title_similarity_rank",
            "votes_rank",
            "imdb_parent_tconst",
        ],
        inplace=True,
    )

    unresolved_snapshot = {
        "match_stage": stage_name,
        "match_method": "unmatched_ambiguous_exact_match",
        "match_notes": "Multiple IMDb rows shared the same normalized title + season key.",
        "candidate_imdb_count": int(len(candidate_frame)),
        "candidate_imdb_parent_tconsts": compact_candidate_values(candidate_frame["imdb_parent_tconst"]),
        "candidate_imdb_primary_titles": compact_candidate_values(candidate_frame["imdb_primary_title"]),
        "candidate_match_source": compact_candidate_values(candidate_frame["candidate_match_source"]),
        "netflix_match_key_used": netflix_key_used,
        "year_consistency_flag": pd.NA,
        "year_distance": pd.NA,
        "title_similarity_score": float(candidate_frame["title_similarity_score"].max()),
        "title_similarity_metric": title_similarity_metric,
    }

    if reference_year is not None and (candidate_frame["year_consistency_flag"] == False).all():
        if len(candidate_frame) == 1:
            unresolved_snapshot["match_method"] = "unmatched_year_conflict"
            unresolved_snapshot["match_notes"] = (
                f"Exact title candidate conflicted with reference year {reference_year}."
            )
        else:
            unresolved_snapshot["match_method"] = "unmatched_ambiguous_year_conflict"
            unresolved_snapshot["match_notes"] = (
                f"Multiple exact title candidates existed, and none matched reference year {reference_year}."
            )
        unresolved_snapshot["year_consistency_flag"] = False
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
            "title_type_priority",
            "title_similarity_rank",
            "votes_rank",
        ]
        top_tuple = tuple(top[column] for column in comparison_columns)
        second_tuple = tuple(second[column] for column in comparison_columns)
        if top_tuple == second_tuple:
            if reference_year is not None and (candidate_frame["year_consistency_flag"] == True).sum() > 1:
                unresolved_snapshot["match_notes"] = (
                    "Multiple IMDb rows remained even after applying the year consistency check."
                )
            return None, unresolved_snapshot, candidate_frame

        if top["source_priority"] < second["source_priority"]:
            resolution_method = "source_priority"
        elif top["year_consistency_rank"] < second["year_consistency_rank"]:
            resolution_method = "year_consistency"
        elif top["exact_start_year_rank"] < second["exact_start_year_rank"]:
            resolution_method = "exact_start_year"
        elif top["year_distance_rank"] < second["year_distance_rank"]:
            resolution_method = "closest_start_year"
        elif top["title_type_priority"] < second["title_type_priority"]:
            resolution_method = "title_type_preference"
        elif top["title_similarity_rank"] < second["title_similarity_rank"]:
            resolution_method = "title_similarity"
        elif top["votes_rank"] < second["votes_rank"]:
            resolution_method = "higher_num_votes"
        else:
            return None, unresolved_snapshot, candidate_frame

    result: dict[str, object] = {column: top[column] for column in candidates.columns if column.startswith("imdb_")}
    result.update(
        {
            "match_status": "matched",
            "match_method": match_method,
            "match_stage": stage_name,
            "match_confidence": resolve_method(
                resolution_method,
                stage_name,
                MATCH_CONFIDENCE_BY_STAGE[stage_name],
                len(candidate_frame),
            ),
            "match_notes": pd.NA
            if resolution_method == "single_candidate"
            else f"Resolved {len(candidate_frame)} candidates using {resolution_method}.",
            "candidate_imdb_count": int(len(candidate_frame)),
            "candidate_imdb_parent_tconsts": compact_candidate_values(candidate_frame["imdb_parent_tconst"]),
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
        }
    )
    return result, unresolved_snapshot, candidate_frame


def collect_review_rows(
    row: pd.Series,
    candidate_frame: pd.DataFrame,
    unresolved_snapshot: dict[str, object],
) -> list[dict[str, object]]:
    review_rows: list[dict[str, object]] = []
    for rank, (_, candidate) in enumerate(candidate_frame.iterrows(), start=1):
        review_rows.append(
            {
                "netflix_row_id": row["netflix_row_id"],
                "netflix_title_raw": row.get("netflix_title_raw"),
                "netflix_series_title": row.get("netflix_series_title"),
                "netflix_season_number": row.get("netflix_season_number"),
                "netflix_season_parse_method": row.get("netflix_season_parse_method"),
                "match_method": unresolved_snapshot["match_method"],
                "match_notes": unresolved_snapshot["match_notes"],
                "match_stage": unresolved_snapshot["match_stage"],
                "candidate_rank": rank,
                "candidate_match_source": candidate.get("candidate_match_source"),
                "netflix_match_key_used": unresolved_snapshot["netflix_match_key_used"],
                "imdb_match_key_used": candidate.get("imdb_match_key_used"),
                "imdb_parent_tconst": candidate.get("imdb_parent_tconst"),
                "imdb_primary_title": candidate.get("imdb_primary_title"),
                "imdb_original_title": candidate.get("imdb_original_title"),
                "imdb_start_year": candidate.get("imdb_start_year"),
                "imdb_end_year": candidate.get("imdb_end_year"),
                "imdb_num_votes": candidate.get("imdb_num_votes"),
                "imdb_title_type": candidate.get("imdb_title_type"),
                "year_consistency_flag": candidate.get("year_consistency_flag"),
                "year_distance": candidate.get("year_distance"),
                "title_similarity_score": candidate.get("title_similarity_score"),
            }
        )
    return review_rows


def perform_fuzzy_match(
    row: pd.Series,
    candidate_pool: pd.DataFrame,
    exact_lookups_available: bool,
) -> tuple[dict[str, object] | None, dict[str, object] | None, pd.DataFrame | None]:
    if not ENABLE_FUZZY_MATCHING or exact_lookups_available:
        return None, None, None

    season_number = row.get("netflix_season_number")
    if pd.isna(season_number):
        return None, None, None

    candidate_subset = candidate_pool[
        (candidate_pool["imdb_season_number"] == int(season_number))
        & candidate_pool["candidate_match_source"].isin(["primary", "original", "aka"])
        & candidate_pool["imdb_match_key_canonical"].notna()
    ].copy()
    if candidate_subset.empty:
        return None, None, None

    candidate_subset = deduplicate_candidates_by_parent(candidate_subset)
    candidate_subset.drop_duplicates(subset=["imdb_match_key_canonical", "imdb_parent_tconst"], inplace=True)

    netflix_key = row.get("netflix_canonical_title")
    if netflix_key is None or pd.isna(netflix_key) or not str(netflix_key).strip():
        return None, None, None

    key_choices = candidate_subset["imdb_match_key_canonical"].dropna().astype(str).unique().tolist()
    matches = process.extract(
        str(netflix_key),
        key_choices,
        scorer=fuzz.ratio,
        limit=5,
    )
    if not matches:
        return None, None, None

    top_key, top_score, _ = matches[0]
    second_score = matches[1][1] if len(matches) > 1 else 0.0
    if top_score < FUZZY_SCORE_THRESHOLD or (top_score - second_score) < FUZZY_MIN_SCORE_GAP:
        return None, None, None

    fuzzy_candidates = candidate_subset[candidate_subset["imdb_match_key_canonical"] == top_key].copy()
    fuzzy_candidates["title_similarity_score"] = float(top_score)
    fuzzy_candidates["candidate_match_source"] = fuzzy_candidates["candidate_match_source"].astype("string")

    match_result, unresolved_snapshot, candidate_frame = resolve_candidate_group(
        row=row,
        candidates=fuzzy_candidates,
        stage_name="stage_e_fuzzy",
        match_method=MATCH_METHOD_BY_STAGE["stage_e_fuzzy"],
        netflix_key_used=str(netflix_key),
        title_similarity_metric="fuzz_ratio",
    )
    return match_result, unresolved_snapshot, candidate_frame


def load_manual_overrides() -> pd.DataFrame:
    if not MANUAL_OVERRIDE_INPUT.exists():
        log("Manual override file not found; skipping override step.")
        return pd.DataFrame()
    overrides = pd.read_csv(MANUAL_OVERRIDE_INPUT, low_memory=False)
    log(f"Loaded manual overrides: {len(overrides):,}")
    return overrides


def apply_manual_overrides(
    master: pd.DataFrame,
    imdb: pd.DataFrame,
    overrides: pd.DataFrame,
) -> pd.DataFrame:
    if overrides.empty:
        return master

    imdb_lookup = imdb.set_index(["imdb_parent_tconst", "imdb_season_number"])
    updated = master.copy()

    for _, override in overrides.iterrows():
        row_mask = pd.Series(False, index=updated.index)
        if "netflix_row_id" in overrides.columns and not pd.isna(override.get("netflix_row_id")):
            row_mask = updated["netflix_row_id"] == int(override["netflix_row_id"])
        elif {"netflix_title_raw", "netflix_season_number"}.issubset(overrides.columns):
            row_mask = (
                updated["netflix_title_raw"].astype("string") == str(override["netflix_title_raw"])
            ) & (
                updated["netflix_season_number"] == pd.to_numeric(
                    override["netflix_season_number"], errors="coerce"
                )
            )

        if not row_mask.any():
            continue

        season_number = updated.loc[row_mask, "netflix_season_number"].iloc[0]
        if pd.isna(season_number):
            season_number = pd.to_numeric(override.get("netflix_season_number"), errors="coerce")
        if pd.isna(season_number):
            continue

        key = (str(override["override_imdb_parent_tconst"]), int(season_number))
        if key not in imdb_lookup.index:
            continue

        imdb_row = imdb_lookup.loc[key]
        if isinstance(imdb_row, pd.DataFrame):
            imdb_row = imdb_row.iloc[0]

        for column in imdb.columns:
            updated.loc[row_mask, column] = imdb_row[column]

        updated.loc[row_mask, "match_status"] = "matched"
        updated.loc[row_mask, "match_method"] = MATCH_METHOD_BY_STAGE["manual_override"]
        updated.loc[row_mask, "match_stage"] = "manual_override"
        updated.loc[row_mask, "match_confidence"] = MATCH_CONFIDENCE_BY_STAGE["manual_override"]
        updated.loc[row_mask, "match_notes"] = override.get("override_reason", "Manual override applied.")
        updated.loc[row_mask, "candidate_match_source"] = "manual_override"
        updated.loc[row_mask, "ambiguity_resolution_method"] = "manual_override"
        updated.loc[row_mask, "candidate_rank"] = 1

    return updated


def create_unmatched_method_summary(master: pd.DataFrame) -> pd.DataFrame:
    unmatched = master[master["match_status"] == "unmatched"].copy()
    rows: list[dict[str, object]] = []
    for method, group in unmatched.groupby("match_method", dropna=False, sort=True):
        sample = group.iloc[0]
        rows.append(
            {
                "match_status": "unmatched",
                "match_method": method,
                "meaning": UNMATCHED_MEANINGS.get(method, "No description available."),
                "count": int(len(group)),
                "example_title": sample.get("netflix_title_raw"),
                "match_confidence": sample.get("match_confidence"),
                "match_notes": sample.get("match_notes"),
            }
        )
    return pd.DataFrame(rows).sort_values(["count", "match_method"], ascending=[False, True])


def create_match_summary(
    master: pd.DataFrame,
    baseline_master: pd.DataFrame | None,
    preserved_baseline_metrics: dict[str, float] | None = None,
) -> pd.DataFrame:
    preserved_baseline_metrics = preserved_baseline_metrics or {}
    total_rows = len(master)
    matched_rows = int((master["match_status"] == "matched").sum())
    series_rows = master[master["netflix_format"] == "series"].copy()
    series_with_seasons = series_rows[series_rows["netflix_season_number"].notna()].copy()
    series_match_rate = (
        float((series_with_seasons["match_status"] == "matched").mean())
        if not series_with_seasons.empty
        else 0.0
    )

    baseline_matched = preserved_baseline_metrics.get("baseline_matched_rows", pd.NA)
    baseline_series_match_rate = preserved_baseline_metrics.get("baseline_series_match_rate", pd.NA)
    if baseline_master is not None and not baseline_master.empty:
        if pd.isna(baseline_matched):
            baseline_matched = int((baseline_master["match_status"] == "matched").sum())
        if pd.isna(baseline_series_match_rate):
            baseline_series_rows = baseline_master[baseline_master["netflix_format"] == "series"].copy()
            baseline_series_with_seasons = baseline_series_rows[
                baseline_series_rows["netflix_season_number"].notna()
            ].copy()
            baseline_series_match_rate = (
                float((baseline_series_with_seasons["match_status"] == "matched").mean())
                if not baseline_series_with_seasons.empty
                else 0.0
            )

    newly_matched_from_old_unmatched = (
        matched_rows - int(baseline_matched) if baseline_matched is not pd.NA and not pd.isna(baseline_matched) else pd.NA
    )

    rows = [
        {"metric": "total_rows", "value": total_rows},
        {"metric": "matched_rows", "value": matched_rows},
        {"metric": "unmatched_rows", "value": int((master["match_status"] != "matched").sum())},
        {"metric": "series_rows", "value": int(len(series_rows))},
        {"metric": "series_rows_with_seasons", "value": int(len(series_with_seasons))},
        {"metric": "series_match_rate", "value": round(series_match_rate, 6)},
        {"metric": "baseline_matched_rows", "value": baseline_matched},
        {"metric": "baseline_series_match_rate", "value": baseline_series_match_rate},
        {"metric": "newly_matched_from_old_unmatched", "value": newly_matched_from_old_unmatched},
        {
            "metric": "matched_using_improved_season_parsing",
            "value": int(
                (
                    (master["match_status"] == "matched")
                    & master["netflix_season_parse_method"].astype("string").isin(
                        [
                            "trailing_number_tv_heuristic",
                            "trailing_roman_tv_heuristic",
                            "s_shorthand",
                        ]
                    )
                ).sum()
            ),
        },
        {
            "metric": "matched_using_original_title_exact",
            "value": int((master["candidate_match_source"] == "original").sum()),
        },
        {
            "metric": "matched_using_alternate_title_exact",
            "value": int((master["candidate_match_source"] == "aka").sum()),
        },
        {
            "metric": "matched_using_fuzzy",
            "value": int((master["match_stage"] == "stage_e_fuzzy").sum()),
        },
        {
            "metric": "matched_using_ambiguity_tie_break",
            "value": int(
                (
                    (master["match_status"] == "matched")
                    & master["ambiguity_resolution_method"].astype("string").isin(
                        [
                            "source_priority",
                            "year_consistency",
                            "exact_start_year",
                            "closest_start_year",
                            "title_type_preference",
                            "higher_num_votes",
                            "title_similarity",
                        ]
                    )
                ).sum()
            ),
        },
        {
            "metric": "still_unresolved_series_rows",
            "value": int(
                ((master["netflix_format"] == "series") & (master["match_status"] != "matched")).sum()
            ),
        },
    ]
    return pd.DataFrame(rows)


def resolve_row(
    row: pd.Series,
    imdb_columns: list[str],
    candidate_pool: pd.DataFrame,
    lookups: dict[str, Any],
) -> tuple[dict[str, object], list[dict[str, object]]]:
    result = build_default_result(imdb_columns)
    review_rows: list[dict[str, object]] = []

    if row.get("netflix_format") != "series":
        result["match_method"] = "unmatched_not_series_like"
        result["match_notes"] = "Netflix record is not marked as a series entry."
        return result, review_rows

    season_number = row.get("netflix_season_number")
    any_exact_candidates_found = False
    best_unresolved_snapshot: dict[str, object] | None = None

    if season_number is not None and not pd.isna(season_number):
        stage_definitions = [
            {
                "stage_name": "stage_a_primary_exact",
                "match_method": MATCH_METHOD_BY_STAGE["stage_a_primary_exact"],
                "netflix_key_col": "netflix_normalized_title",
                "lookup_name": "primary_normalized",
                "title_similarity_metric": "exact",
            },
            {
                "stage_name": "stage_b_original_exact",
                "match_method": MATCH_METHOD_BY_STAGE["stage_b_original_exact"],
                "netflix_key_col": "netflix_normalized_title",
                "lookup_name": "original_normalized",
                "title_similarity_metric": "exact",
            },
            {
                "stage_name": "stage_c_aka_exact",
                "match_method": MATCH_METHOD_BY_STAGE["stage_c_aka_exact"],
                "netflix_key_col": "netflix_normalized_title",
                "lookup_name": "aka_normalized",
                "title_similarity_metric": "exact",
            },
            {
                "stage_name": "stage_d_raw_title_variant_exact",
                "match_method": MATCH_METHOD_BY_STAGE["stage_d_raw_title_variant_exact"],
                "netflix_key_col": "netflix_raw_normalized_title",
                "lookup_name": "all_normalized",
                "title_similarity_metric": "exact",
            },
            {
                "stage_name": "stage_d_canonical_exact",
                "match_method": MATCH_METHOD_BY_STAGE["stage_d_canonical_exact"],
                "netflix_key_col": "netflix_canonical_title",
                "lookup_name": "all_canonical",
                "title_similarity_metric": "exact",
            },
            {
                "stage_name": "stage_d_raw_canonical_exact",
                "match_method": MATCH_METHOD_BY_STAGE["stage_d_raw_canonical_exact"],
                "netflix_key_col": "netflix_raw_canonical_title",
                "lookup_name": "all_canonical",
                "title_similarity_metric": "exact",
            },
            {
                "stage_name": "stage_d_compact_exact",
                "match_method": MATCH_METHOD_BY_STAGE["stage_d_compact_exact"],
                "netflix_key_col": "netflix_compact_title",
                "lookup_name": "all_compact",
                "title_similarity_metric": "exact",
            },
            {
                "stage_name": "stage_d_raw_compact_exact",
                "match_method": MATCH_METHOD_BY_STAGE["stage_d_raw_compact_exact"],
                "netflix_key_col": "netflix_raw_compact_title",
                "lookup_name": "all_compact",
                "title_similarity_metric": "exact",
            },
        ]

        for stage in stage_definitions:
            netflix_key = row.get(stage["netflix_key_col"])
            if netflix_key is None or pd.isna(netflix_key) or not str(netflix_key).strip():
                continue

            lookup_key = (str(netflix_key), int(season_number))
            indices = lookups[stage["lookup_name"]].get(lookup_key, [])
            if not indices:
                continue

            any_exact_candidates_found = True
            candidates = candidate_pool.loc[indices].copy()
            candidates["title_similarity_score"] = 100.0
            match_result, unresolved_snapshot, candidate_frame = resolve_candidate_group(
                row=row,
                candidates=candidates,
                stage_name=stage["stage_name"],
                match_method=stage["match_method"],
                netflix_key_used=str(netflix_key),
                title_similarity_metric=stage["title_similarity_metric"],
            )
            if match_result is not None:
                return match_result, review_rows

            if best_unresolved_snapshot is None:
                best_unresolved_snapshot = unresolved_snapshot
                review_rows.extend(collect_review_rows(row, candidate_frame, unresolved_snapshot))

        fuzzy_result, fuzzy_unresolved_snapshot, fuzzy_candidate_frame = perform_fuzzy_match(
            row=row,
            candidate_pool=candidate_pool,
            exact_lookups_available=any_exact_candidates_found,
        )
        if fuzzy_result is not None:
            return fuzzy_result, review_rows
        if fuzzy_unresolved_snapshot is not None and best_unresolved_snapshot is None:
            best_unresolved_snapshot = fuzzy_unresolved_snapshot
            review_rows.extend(collect_review_rows(row, fuzzy_candidate_frame, fuzzy_unresolved_snapshot))

        if best_unresolved_snapshot is not None:
            result.update(best_unresolved_snapshot)
            return result, review_rows

        result["match_method"] = "unmatched_no_exact_title_season_match"
        result["match_notes"] = "No exact normalized title + season match was found in IMDb."
        return result, review_rows

    inference_stages = [
        {
            "stage_name": "stage_single_season_inference",
            "match_method": MATCH_METHOD_BY_STAGE["stage_single_season_inference"],
            "netflix_key_col": "netflix_normalized_title",
            "lookup_name": "single_season_primary",
        },
        {
            "stage_name": "stage_single_season_inference",
            "match_method": MATCH_METHOD_BY_STAGE["stage_single_season_inference"],
            "netflix_key_col": "netflix_raw_normalized_title",
            "lookup_name": "single_season_all_normalized",
        },
        {
            "stage_name": "stage_single_season_inference",
            "match_method": MATCH_METHOD_BY_STAGE["stage_single_season_inference"],
            "netflix_key_col": "netflix_canonical_title",
            "lookup_name": "single_season_all_canonical",
        },
    ]

    for stage in inference_stages:
        netflix_key = row.get(stage["netflix_key_col"])
        if netflix_key is None or pd.isna(netflix_key) or not str(netflix_key).strip():
            continue

        indices = lookups[stage["lookup_name"]].get(str(netflix_key), [])
        if not indices:
            continue

        candidates = candidate_pool.loc[indices].copy()
        candidates["title_similarity_score"] = 100.0
        match_result, unresolved_snapshot, candidate_frame = resolve_candidate_group(
            row=row,
            candidates=candidates,
            stage_name=stage["stage_name"],
            match_method=stage["match_method"],
            netflix_key_used=str(netflix_key),
            title_similarity_metric="exact",
        )
        if match_result is not None:
            match_result["match_notes"] = (
                "Implied season 1 because Netflix row is series-like and IMDb candidate is uniquely single-season."
            )
            return match_result, review_rows
        if best_unresolved_snapshot is None:
            best_unresolved_snapshot = unresolved_snapshot
            review_rows.extend(collect_review_rows(row, candidate_frame, unresolved_snapshot))

    if best_unresolved_snapshot is not None:
        result.update(best_unresolved_snapshot)
        return result, review_rows

    result["match_method"] = "unmatched_missing_season_number"
    result["match_notes"] = "Netflix season number could not be extracted from the title."
    return result, review_rows


def merge_datasets() -> pd.DataFrame:
    preserved_baseline_metrics = load_existing_summary_metrics()
    baseline_master = load_baseline_master()
    netflix, imdb, imdb_title_keys = load_inputs()
    candidate_pool = build_candidate_pool(imdb, imdb_title_keys)
    imdb_columns = list(imdb.columns)

    lookups = {
        "primary_normalized": build_lookup(candidate_pool, "imdb_match_key_used", ["primary"]),
        "original_normalized": build_lookup(candidate_pool, "imdb_match_key_used", ["original"]),
        "aka_normalized": build_lookup(candidate_pool, "imdb_match_key_used", ["aka"]),
        "all_normalized": build_lookup(candidate_pool, "imdb_match_key_used"),
        "all_canonical": build_lookup(candidate_pool, "imdb_match_key_canonical"),
        "all_compact": build_lookup(candidate_pool, "imdb_match_key_compact"),
        "single_season_primary": build_single_season_lookup(
            candidate_pool, "imdb_match_key_used", ["primary"]
        ),
        "single_season_all_normalized": build_single_season_lookup(
            candidate_pool, "imdb_match_key_used"
        ),
        "single_season_all_canonical": build_single_season_lookup(
            candidate_pool, "imdb_match_key_canonical"
        ),
    }

    resolved_rows: list[dict[str, object]] = []
    ambiguous_review_rows: list[dict[str, object]] = []
    for _, row in netflix.iterrows():
        resolved, review_rows = resolve_row(row, imdb_columns, candidate_pool, lookups)
        resolved_rows.append(resolved)
        ambiguous_review_rows.extend(review_rows)

    resolution_frame = pd.DataFrame(resolved_rows, index=netflix["netflix_row_id"].astype(int))
    master = netflix.merge(resolution_frame, how="left", left_on="netflix_row_id", right_index=True)
    master = apply_manual_overrides(master, imdb, load_manual_overrides())

    ensure_parent(MASTER_OUTPUT)
    master.to_parquet(MASTER_OUTPUT, index=False)
    master.to_csv(MASTER_CSV_OUTPUT, index=False)
    master.head(250).to_csv(SAMPLE_OUTPUT, index=False)

    unmatched_series = master[
        (master["netflix_format"] == "series") & (master["match_status"] != "matched")
    ].copy()
    unmatched_series.to_csv(UNMATCHED_SERIES_OUTPUT, index=False)

    ambiguous_review = pd.DataFrame(ambiguous_review_rows)
    if ambiguous_review.empty:
        ambiguous_review = pd.DataFrame(
            columns=[
                "netflix_row_id",
                "netflix_title_raw",
                "netflix_series_title",
                "netflix_season_number",
                "match_method",
                "match_notes",
                "imdb_parent_tconst",
                "imdb_primary_title",
            ]
        )
    ambiguous_review.to_csv(AMBIGUOUS_REVIEW_OUTPUT, index=False)

    matched_by_fuzzy = master[master["match_stage"] == "stage_e_fuzzy"].copy()
    matched_by_fuzzy.to_csv(FUZZY_REVIEW_OUTPUT, index=False)

    matched_by_alternate = master[master["candidate_match_source"].isin(["original", "aka"])].copy()
    matched_by_alternate.to_csv(ALTERNATE_REVIEW_OUTPUT, index=False)

    unmatched_summary = create_unmatched_method_summary(master)
    unmatched_summary.to_csv(UNMATCHED_METHOD_SUMMARY_OUTPUT, index=False)

    match_summary = create_match_summary(master, baseline_master, preserved_baseline_metrics)
    match_summary.to_csv(MATCH_SUMMARY_OUTPUT, index=False)

    matched_rows = int((master["match_status"] == "matched").sum())
    unmatched_rows = int((master["match_status"] != "matched").sum())
    total_rows = len(master)
    series_rows = master[master["netflix_format"] == "series"].copy()
    series_with_seasons = series_rows[series_rows["netflix_season_number"].notna()].copy()
    series_match_rate = (
        float((series_with_seasons["match_status"] == "matched").mean())
        if not series_with_seasons.empty
        else 0.0
    )

    baseline_matched = preserved_baseline_metrics.get("baseline_matched_rows")
    baseline_series_match_rate = preserved_baseline_metrics.get("baseline_series_match_rate")
    if baseline_matched is None and baseline_master is not None:
        baseline_matched = int((baseline_master["match_status"] == "matched").sum())
    if baseline_series_match_rate is None and baseline_master is not None:
        baseline_series_rows = baseline_master[baseline_master["netflix_format"] == "series"].copy()
        baseline_series_with_seasons = baseline_series_rows[
            baseline_series_rows["netflix_season_number"].notna()
        ].copy()
        baseline_series_match_rate = (
            float((baseline_series_with_seasons["match_status"] == "matched").mean())
            if not baseline_series_with_seasons.empty
            else 0.0
        )
    if baseline_matched is None:
        baseline_matched = 0
    if baseline_series_match_rate is None:
        baseline_series_match_rate = 0.0

    log(f"Netflix rows read: {len(netflix):,}")
    log(f"IMDb series-season rows loaded: {len(imdb):,}")
    log(f"IMDb title keys loaded: {len(imdb_title_keys):,}")
    log(f"Final merged rows: {total_rows:,}")
    log(f"Matched rows: {matched_rows:,} ({matched_rows / total_rows:.2%})")
    log(f"Unmatched rows: {unmatched_rows:,} ({unmatched_rows / total_rows:.2%})")
    log(f"Baseline matched rows: {baseline_matched:,}")
    log(f"Series-row match rate before: {baseline_series_match_rate:.2%}")
    log(f"Series-row match rate after: {series_match_rate:.2%}")
    log(
        "Newly matched using improved season parsing: "
        f"{int(match_summary.loc[match_summary['metric'] == 'matched_using_improved_season_parsing', 'value'].iloc[0]):,}"
    )
    log(
        "Newly matched using original title exact matches: "
        f"{int(match_summary.loc[match_summary['metric'] == 'matched_using_original_title_exact', 'value'].iloc[0]):,}"
    )
    log(
        "Newly matched using alternate title exact matches: "
        f"{int(match_summary.loc[match_summary['metric'] == 'matched_using_alternate_title_exact', 'value'].iloc[0]):,}"
    )
    log(
        "Rows matched using fuzzy stage: "
        f"{int(match_summary.loc[match_summary['metric'] == 'matched_using_fuzzy', 'value'].iloc[0]):,}"
    )
    log(
        "Rows resolved by tie-break logic: "
        f"{int(match_summary.loc[match_summary['metric'] == 'matched_using_ambiguity_tie_break', 'value'].iloc[0]):,}"
    )
    log(
        "Counts by match stage: "
        + ", ".join(f"{stage}={count}" for stage, count in master["match_stage"].value_counts().items())
    )
    log(
        "Counts by match method: "
        + ", ".join(f"{method}={count}" for method, count in master["match_method"].value_counts().items())
    )
    log(
        "Top remaining hard cases: "
        + ", ".join(
            f"{title} ({count})"
            for title, count in unmatched_series["netflix_series_title"]
            .fillna("<missing>")
            .value_counts()
            .head(10)
            .items()
        )
    )
    log(f"Saved final master parquet: {MASTER_OUTPUT.relative_to(REPO_ROOT).as_posix()}")
    log(f"Saved final master CSV: {MASTER_CSV_OUTPUT.relative_to(REPO_ROOT).as_posix()}")
    log(f"Saved unmatched review: {UNMATCHED_SERIES_OUTPUT.relative_to(REPO_ROOT).as_posix()}")
    log(f"Saved ambiguous review: {AMBIGUOUS_REVIEW_OUTPUT.relative_to(REPO_ROOT).as_posix()}")
    return master


def main() -> None:
    merge_datasets()


if __name__ == "__main__":
    main()
