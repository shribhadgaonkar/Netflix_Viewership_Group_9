# Modeling Data Dictionary

## Scope

The strict modeling dataset is built from the frozen enriched matched dataset copy `data/processed/netflix_imdb_modeling_source_copy.*` and only keeps matched `series_season` rows that have a stable series key, a valid season number, usable current-season Netflix metrics, and an observed consecutive next season.

## Core Identifiers

- `netflix_row_id`: Stable Netflix row id.
- `series_group_key`: Stable within-series grouping key used for target and lag construction.
- `imdb_parent_tconst`: Preferred stable parent-series identifier.
- `imdb_enrichment_entity_id`: Parent-series enrichment join key for the modeling dataset.

## Target Definitions

- `target_next_season_views`: Next observed season's `netflix_views`.
- `target_next_season_hours`: Next observed season's `netflix_hours_viewed`.
- `target_view_change_absolute`: `target_next_season_views - netflix_views`.
- `target_view_change_percent`: `target_view_change_absolute / netflix_views`.
- `target_hours_change_absolute`: `target_next_season_hours - netflix_hours_viewed`.
- `target_hours_change_percent`: `target_hours_change_absolute / netflix_hours_viewed`.
- `target_is_viewership_increase`: `1` when next-season views are higher than current-season views.
- `target_is_hours_increase`: `1` when next-season hours are higher than current-season hours.

## Included Predictor Count

- Predictors retained in the strict modeling dataset: `77`

## Current-Season Netflix Observation Features

- `first_observed_halfyear_period`: Earliest half-year Netflix reporting period in the available half-year source files where both views and hours are observed for the current row's title.
- `first_observed_halfyear_views`: Views recorded in that first observed half-year period.
- `first_observed_halfyear_hours`: Hours recorded in that first observed half-year period.
- `first_halfyear_hours_per_view`: `first_observed_halfyear_hours / first_observed_halfyear_views`, with missing output when views are missing or zero.

## Leakage Policy

The strict modeling dataset excludes full-series future-summary fields such as total observed seasons, full-run episode totals, and lifecycle fields that depend on knowledge beyond the current season row.
