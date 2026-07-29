# Modeling Data Dictionary

## Files

- Final modeling dataset:
  - `data/processed/netflix_imdb_modeling.parquet`
  - `data/processed/netflix_imdb_modeling.csv`
- Frozen modeling source copy:
  - `data/processed/netflix_imdb_modeling_source_copy.parquet`
  - `data/processed/netflix_imdb_modeling_source_copy.csv`
- Supporting transparency files:
  - `data/processed/netflix_imdb_modeling_feature_manifest.csv`
  - `data/processed/netflix_imdb_modeling_quality_report.csv`

## Scope

The strict modeling dataset is built from the frozen enriched matched dataset copy `data/processed/netflix_imdb_modeling_source_copy.*` and only keeps matched `series_season` rows that have a stable series key, a valid season number, usable current-season Netflix metrics, and an observed consecutive next season.

## Row Grain

Each row in the final modeling dataset represents one Netflix season observation for a matched IMDb parent series, where that season has a valid consecutive next-season observation used to create the supervised target.

The final modeling dataset excludes:

- unmatched rows
- movie rows
- `series_overall` rows
- season rows without a stable series key
- season rows without a valid season number
- rows without usable current-season Netflix metrics
- rows without an observed next season

## Core Identifiers

- `netflix_row_id`: Stable Netflix row id.
- `series_group_key`: Stable within-series grouping key used for target and lag construction.
- `imdb_parent_tconst`: Preferred stable parent-series identifier.
- `imdb_enrichment_entity_id`: Parent-series enrichment join key for the modeling dataset.
- `imdb_resolved_tconst`: Stable matched IMDb identifier retained for traceability.
- `netflix_series_title`: Parsed Netflix-side base series title retained for inspection.
- `netflix_title_raw`: Original Netflix title text retained for traceability.

## Target Definitions

- `target_next_season_views`: Next observed season's `netflix_views`.
- `target_next_season_hours`: Next observed season's `netflix_hours_viewed`.
- `target_view_change_absolute`: `target_next_season_views - netflix_views`.
- `target_view_change_percent`: `target_view_change_absolute / netflix_views`.
- `target_hours_change_absolute`: `target_next_season_hours - netflix_hours_viewed`.
- `target_hours_change_percent`: `target_hours_change_absolute / netflix_hours_viewed`.
- `target_is_viewership_increase`: `1` when next-season views are higher than current-season views.
- `target_is_hours_increase`: `1` when next-season hours are higher than current-season hours.

Rows without all required target fields are excluded from the final modeling dataset.

## Major Predictor Groups

- Current Netflix performance:
  - `netflix_views`
  - `netflix_hours_viewed`
  - `netflix_runtime`
  - `netflix_log_views`
  - `netflix_log_hours`
  - `netflix_hours_per_view`
- Season structure:
  - `netflix_season_number`
  - `season_order`
  - `season_is_first`
  - `season_is_later`
- IMDb quality and popularity:
  - `imdb_average_rating`
  - `imdb_num_votes`
  - `imdb_log_num_votes`
  - `imdb_rating_votes_interaction`
- Genre indicators:
  - `genre_count`
  - `genre_drama`
  - `genre_comedy`
  - `genre_action`
  - `genre_thriller`
  - `genre_crime`
  - `genre_romance`
  - `genre_animation`
  - `genre_documentary`
  - `genre_fantasy`
  - `genre_horror`
  - `genre_sci_fi`
  - `genre_family`
  - `is_animation`
  - `is_documentary`
  - `is_kids_family_like`
- Cast and crew:
  - `imdb_director_count`
  - `imdb_writer_count`
  - `imdb_principal_count`
  - `imdb_actor_count`
  - `imdb_actress_count`
  - `imdb_self_count`
  - `imdb_producer_count`
  - `imdb_writer_credit_count`
  - `imdb_director_credit_count`
  - `imdb_top_cast_count_used`
  - `imdb_top_cast_known_for_count_proxy`
  - `imdb_top_cast_names`
  - `imdb_top_cast_nconsts`
- Internationalization:
  - `imdb_aka_title_count`
  - `imdb_aka_region_count`
  - `imdb_aka_language_count`
  - `imdb_has_us_title`
  - `imdb_has_international_title_variants`
- Time-aware alignment:
  - `netflix_reference_year`
  - `imdb_age_at_netflix_year`
  - `imdb_started_before_netflix_flag`
  - `imdb_same_year_as_netflix_flag`
  - `netflix_imdb_year_gap`
  - `netflix_imdb_runtime_gap`
- Lag features:
  - `prev_season_views`
  - `prev_season_hours`
  - `prev_season_rating`
  - `prev_view_change_absolute`
  - `prev_view_change_percent`
  - `prev_hours_change_absolute`
  - `prev_hours_change_percent`
  - `has_prev_season_observation`
- Missingness indicators:
  - `missing_imdb_average_rating`
  - `missing_imdb_num_votes`
  - `missing_netflix_runtime`
  - `missing_netflix_imdb_runtime_gap`
  - `missing_cast_features`
  - `missing_release_year`

## Included Predictor Count

- Predictors retained in the strict modeling dataset: `73`

## Leakage Policy

The strict modeling dataset excludes full-series future-summary fields such as total observed seasons, full-run episode totals, and lifecycle fields that depend on knowledge beyond the current season row.

Examples of excluded leakage-prone fields:

- `imdb_parent_season_count`
- `imdb_total_episode_count`
- `imdb_max_season_number`
- `imdb_avg_episodes_per_season`
- `imdb_min_episodes_per_season`
- `imdb_max_episodes_per_season`
- `imdb_single_season_flag`
- `imdb_multi_season_flag`
- `imdb_series_ended_flag`
- `imdb_series_ongoing_flag`
- `imdb_years_active`

## Supporting Files

- `netflix_imdb_modeling_feature_manifest.csv` gives the per-column role, inclusion status, drop reason, leakage flag, and missingness rate.
- `netflix_imdb_modeling_quality_report.csv` gives row-count checks, target coverage, predictor null rates, and target class balance.
