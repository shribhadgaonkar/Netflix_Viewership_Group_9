from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.pipeline_utils import (  # noqa: E402
    coerce_nullable_int,
    ensure_parent,
    interim_dir,
    log,
    processed_dir,
    year_consistency,
)


NETFLIX_INPUT = interim_dir() / "netflix_cleaned.csv"
IMDB_INPUT = interim_dir() / "imdb_series_seasons.csv"
MASTER_OUTPUT = processed_dir() / "netflix_imdb_master.parquet"
SAMPLE_OUTPUT = processed_dir() / "netflix_imdb_master_sample.csv"


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    if not NETFLIX_INPUT.exists():
        raise FileNotFoundError(
            f"Missing {NETFLIX_INPUT}. Run scripts/data_preparation/prepare_netflix.py first."
        )
    if not IMDB_INPUT.exists():
        raise FileNotFoundError(
            f"Missing {IMDB_INPUT}. Run scripts/data_preparation/prepare_imdb.py first."
        )

    netflix = pd.read_csv(NETFLIX_INPUT, low_memory=False)
    imdb = pd.read_csv(IMDB_INPUT, low_memory=False)

    netflix["netflix_row_id"] = coerce_nullable_int(netflix["netflix_row_id"])
    netflix["netflix_season_number"] = coerce_nullable_int(netflix["netflix_season_number"])
    netflix["netflix_release_year"] = coerce_nullable_int(netflix["netflix_release_year"])
    netflix["netflix_title_year_hint"] = coerce_nullable_int(netflix["netflix_title_year_hint"])

    imdb["imdb_season_number"] = coerce_nullable_int(imdb["imdb_season_number"])
    imdb["imdb_start_year"] = coerce_nullable_int(imdb["imdb_start_year"])
    imdb["imdb_end_year"] = coerce_nullable_int(imdb["imdb_end_year"])
    imdb["imdb_num_votes"] = coerce_nullable_int(imdb["imdb_num_votes"])
    imdb["imdb_season_episode_count"] = coerce_nullable_int(imdb["imdb_season_episode_count"])
    return netflix, imdb


def build_candidate_table(netflix: pd.DataFrame, imdb: pd.DataFrame) -> pd.DataFrame:
    join_columns = ["netflix_normalized_title", "netflix_season_number"]
    right_columns = ["imdb_normalized_title", "imdb_season_number"]
    return netflix.merge(imdb, how="left", left_on=join_columns, right_on=right_columns)


def compact_candidate_values(series: pd.Series) -> str | None:
    values = [str(value) for value in pd.unique(series.dropna()) if str(value).strip()]
    return " | ".join(values) if values else None


def resolve_group(group: pd.DataFrame, imdb_columns: list[str]) -> dict[str, object]:
    result: dict[str, object] = {column: pd.NA for column in imdb_columns}
    result.update(
        {
            "match_status": "unmatched",
            "match_method": "unmatched_no_exact_title_season_match",
            "match_confidence": 0.0,
            "match_notes": pd.NA,
            "candidate_imdb_count": 0,
            "candidate_imdb_parent_tconsts": pd.NA,
            "candidate_imdb_primary_titles": pd.NA,
            "year_consistency_flag": pd.NA,
        }
    )

    base_row = group.iloc[0]
    candidates = group[group["imdb_parent_tconst"].notna()].copy()
    result["candidate_imdb_count"] = int(len(candidates))
    result["candidate_imdb_parent_tconsts"] = compact_candidate_values(candidates["imdb_parent_tconst"])
    result["candidate_imdb_primary_titles"] = compact_candidate_values(candidates["imdb_primary_title"])

    if base_row.get("netflix_format") != "series":
        result["match_method"] = "unmatched_not_series_like"
        result["match_notes"] = "Netflix record is not marked as a series entry."
        return result

    if pd.isna(base_row.get("netflix_season_number")):
        result["match_method"] = "unmatched_missing_season_number"
        result["match_notes"] = "Netflix season number could not be extracted from the title."
        return result

    if candidates.empty:
        result["match_notes"] = "No exact normalized title + season match was found in IMDb."
        return result

    reference_year = base_row.get("netflix_release_year")
    if pd.isna(reference_year):
        reference_year = base_row.get("netflix_title_year_hint")

    candidates["year_consistency_flag"] = [
        year_consistency(reference_year, start_year, end_year)
        for start_year, end_year in zip(
            candidates["imdb_start_year"], candidates["imdb_end_year"], strict=False
        )
    ]

    exact_matches = candidates
    if len(exact_matches) == 1:
        only = exact_matches.iloc[0]
        consistent = only["year_consistency_flag"]
        if reference_year is not pd.NA and not pd.isna(reference_year) and consistent is False:
            result["match_method"] = "unmatched_year_conflict"
            result["match_notes"] = (
                f"Exact title + season match conflicted with reference year {int(reference_year)}."
            )
            result["year_consistency_flag"] = False
            return result

        for column in imdb_columns:
            result[column] = only[column]
        result["match_status"] = "matched"
        result["match_method"] = (
            "exact_title_season_year"
            if reference_year is not pd.NA and not pd.isna(reference_year) and consistent is True
            else "exact_title_season"
        )
        result["match_confidence"] = 1.0 if result["match_method"] == "exact_title_season_year" else 0.95
        result["match_notes"] = pd.NA
        result["year_consistency_flag"] = consistent
        return result

    year_consistent = candidates[candidates["year_consistency_flag"] == True]
    if reference_year is not pd.NA and not pd.isna(reference_year) and len(year_consistent) == 1:
        chosen = year_consistent.iloc[0]
        for column in imdb_columns:
            result[column] = chosen[column]
        result["match_status"] = "matched"
        result["match_method"] = "exact_title_season_year"
        result["match_confidence"] = 0.98
        result["match_notes"] = "Resolved multiple exact-key candidates using year consistency."
        result["year_consistency_flag"] = True
        return result

    if reference_year is not pd.NA and not pd.isna(reference_year) and len(year_consistent) == 0:
        result["match_method"] = "unmatched_ambiguous_year_conflict"
        result["match_notes"] = (
            f"Multiple exact title + season candidates existed, and none matched reference year {int(reference_year)}."
        )
        result["year_consistency_flag"] = False
        return result

    result["match_method"] = "unmatched_ambiguous_exact_match"
    result["match_notes"] = "Multiple IMDb rows shared the same normalized title + season key."
    if reference_year is not pd.NA and not pd.isna(reference_year) and len(year_consistent) > 1:
        result["match_notes"] = (
            "Multiple IMDb rows remained even after applying the year consistency check."
        )
    return result


def merge_datasets() -> pd.DataFrame:
    netflix, imdb = load_inputs()
    candidates = build_candidate_table(netflix, imdb)
    imdb_columns = list(imdb.columns)

    resolved_rows: list[dict[str, object]] = []
    resolved_index: list[int] = []
    for netflix_row_id, group in candidates.groupby("netflix_row_id", sort=False):
        resolution = resolve_group(group, imdb_columns)
        resolved_rows.append(resolution)
        resolved_index.append(int(netflix_row_id))

    resolution_frame = pd.DataFrame(resolved_rows, index=resolved_index)
    master = netflix.merge(resolution_frame, how="left", left_on="netflix_row_id", right_index=True)

    ensure_parent(MASTER_OUTPUT)
    master.to_parquet(MASTER_OUTPUT, index=False)
    preview = master.head(250)
    preview.to_csv(SAMPLE_OUTPUT, index=False)

    netflix_key_duplicates = int(
        netflix[
            netflix["netflix_normalized_title"].notna() & netflix["netflix_season_number"].notna()
        ].duplicated(["netflix_normalized_title", "netflix_season_number"], keep=False).sum()
    )
    imdb_key_duplicates = int(
        imdb[
            imdb["imdb_normalized_title"].notna() & imdb["imdb_season_number"].notna()
        ].duplicated(["imdb_normalized_title", "imdb_season_number"], keep=False).sum()
    )
    matched_rows = int((master["match_status"] == "matched").sum())
    unmatched_rows = int((master["match_status"] != "matched").sum())
    total_rows = len(master)

    log(f"Netflix rows read: {len(netflix):,}")
    log(f"IMDb series-season rows loaded: {len(imdb):,}")
    log(f"Final merged rows: {total_rows:,}")
    log(f"Matched rows: {matched_rows:,} ({matched_rows / total_rows:.2%})")
    log(f"Unmatched rows: {unmatched_rows:,} ({unmatched_rows / total_rows:.2%})")
    log(f"Netflix duplicate join-key rows: {netflix_key_duplicates:,}")
    log(f"IMDb duplicate join-key rows: {imdb_key_duplicates:,}")
    log(
        "Netflix null join fields - normalized title: "
        f"{int(master['netflix_normalized_title'].isna().sum()):,}, "
        f"season number: {int(master['netflix_season_number'].isna().sum()):,}"
    )
    log(
        "Top unmatched title patterns: "
        + ", ".join(
            f"{title} ({count})"
            for title, count in master.loc[master["match_status"] != "matched", "netflix_series_title"]
            .fillna("<missing>")
            .value_counts()
            .head(10)
            .items()
        )
    )
    log(f"Saved final master parquet: {MASTER_OUTPUT.relative_to(REPO_ROOT).as_posix()}")
    log(f"Saved preview CSV: {SAMPLE_OUTPUT.relative_to(REPO_ROOT).as_posix()}")
    return master


def main() -> None:
    merge_datasets()


if __name__ == "__main__":
    main()
