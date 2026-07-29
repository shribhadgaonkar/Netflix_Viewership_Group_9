# Modeling Feature Notes

## Copy-First Workflow

1. Load the frozen enriched matched dataset.
2. Write an exact row-for-row source snapshot to `data/processed/netflix_imdb_modeling_source_copy.*`.
3. Build the strict modeling dataset only from that copied snapshot.

## Leakage-Prone Columns Excluded

- `imdb_avg_episodes_per_season`
- `imdb_max_episodes_per_season`
- `imdb_max_season_number`
- `imdb_min_episodes_per_season`
- `imdb_multi_season_flag`
- `imdb_parent_season_count`
- `imdb_series_ended_flag`
- `imdb_series_ongoing_flag`
- `imdb_single_season_flag`
- `imdb_total_episode_count`
- `imdb_years_active`

## Lag Features

Lag features use only prior observed seasons within the same `series_group_key`. Rows without a consecutive next season are removed from the final supervised dataset.

## Feature Manifest

See `data/processed/netflix_imdb_modeling_feature_manifest.csv` for per-column inclusion, missingness, drop reasons, and leakage flags.
