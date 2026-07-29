# Data Inventory

## Raw Inputs

| File | Location | Notes |
| --- | --- | --- |
| Netflix export | `data/raw/netflix/netflixlist7-export.csv` | Main Netflix source file with title, type, runtime, yearly hours/views, and totals. |
| IMDb title basics | `data/raw/imdb/title.basics.tsv.gz` | Compressed IMDb title metadata used to identify parent series. |
| IMDb episode links | `data/raw/imdb/title.episode.tsv.gz` | Compressed IMDb episode-to-parent mapping with season numbers. |
| IMDb ratings | `data/raw/imdb/title.ratings.tsv.gz` | Compressed IMDb average rating and vote counts. |
| IMDb alternate titles | `data/raw/imdb/title.akas.tsv.gz` | Compressed alternate-title metadata used for matching and internationalization features. |
| IMDb crew links | `data/raw/imdb/title.crew.tsv.gz` | Compressed director and writer identifiers for entity-level enrichment. |
| IMDb principals | `data/raw/imdb/title.principals.tsv.gz` | Compressed principal cast and credit structure data for entity-level enrichment. |
| IMDb people basics | `data/raw/imdb/name.basics.tsv.gz` | Compressed person metadata used to translate cast, director, and writer ids into compact summaries. |

## Intermediate Outputs

| File | Location | Created by | Purpose |
| --- | --- | --- | --- |
| Cleaned Netflix table | `data/interim/netflix_cleaned.csv` | `scripts/data_preparation/prepare_netflix.py` | Standardized Netflix titles, seasons, formats, and canonical metrics. |
| IMDb series-season table | `data/interim/imdb_series_seasons.csv` | `scripts/data_preparation/prepare_imdb.py` | Parent-series metadata aggregated to the season level. |
| IMDb title-key table | `data/interim/imdb_title_keys.csv` | `scripts/data_preparation/prepare_imdb.py` | Match-key variants built from IMDb primary, original, and alternate titles. |
| Incremental IMDb movie index | `data/interim/imdb_movies.csv` | `scripts/data_preparation/third_pass_multi_grain.py` | Cached movie-only IMDb entities used by the third-pass movie matcher. |
| Incremental IMDb movie keys | `data/interim/imdb_movie_title_keys.csv` | `scripts/data_preparation/third_pass_multi_grain.py` | Cached movie title-key variants used by the third-pass movie matcher. |
| Incremental IMDb series parents | `data/interim/imdb_series_parents.csv` | `scripts/data_preparation/third_pass_multi_grain.py` | Cached parent-series entities used by the third-pass parent-series matcher. |
| Incremental IMDb series-parent keys | `data/interim/imdb_series_parent_title_keys.csv` | `scripts/data_preparation/third_pass_multi_grain.py` | Cached parent-series key variants used by the third-pass parent-series matcher. |
| IMDb entity features | `data/interim/imdb_entity_features.csv` | `scripts/data_enrichment/prepare_imdb_entity_features.py` | Entity-level title, lifecycle, genre, and rating features for matched IMDb ids only. |
| IMDb episode aggregates | `data/interim/imdb_episode_aggregates.csv` | `scripts/data_enrichment/prepare_imdb_entity_features.py` | Parent-series and season-level episode counts derived only for matched series entities. |
| IMDb aka aggregates | `data/interim/imdb_aka_aggregates.csv` | `scripts/data_enrichment/prepare_imdb_entity_features.py` | Internationalization and title-distribution counts for matched IMDb ids only. |
| IMDb crew aggregates | `data/interim/imdb_crew_aggregates.csv` | `scripts/data_enrichment/prepare_imdb_entity_features.py` | Aggregated director and writer counts, ids, names, and birth-year summaries. |
| IMDb principal aggregates | `data/interim/imdb_principal_aggregates.csv` | `scripts/data_enrichment/prepare_imdb_entity_features.py` | Aggregated cast and credit-structure features, plus top-cast summaries. |

## Final Output

| File | Location | Format | Purpose |
| --- | --- | --- | --- |
| Master merged dataset | `data/processed/netflix_imdb_master.parquet` | Parquet | Final left-joined Netflix-to-IMDb series-season dataset. |
| Master merged dataset | `data/processed/netflix_imdb_master.csv` | CSV | Full CSV export of the final merged dataset for teammates who prefer plain-text tabular files. |
| Third-pass master dataset | `data/processed/netflix_imdb_master_v3.parquet` | Parquet | Frozen third-pass multi-grain master used as the read-only enrichment baseline. |
| Third-pass master dataset | `data/processed/netflix_imdb_master_v3.csv` | CSV | CSV export of the frozen third-pass multi-grain master. |
| Matched-only baseline | `data/processed/netflix_imdb_master_v3_matched_only.parquet` | Parquet | Exact matched-row subset of the frozen third-pass master, used as the enrichment input. |
| Matched-only baseline | `data/processed/netflix_imdb_master_v3_matched_only.csv` | CSV | CSV export of the matched-only baseline subset. |
| Enriched matched dataset | `data/processed/netflix_imdb_master_matched_enriched.parquet` | Parquet | Matched-only dataset enriched with IMDb entity, episode, aka, crew, and principal features. |
| Enriched matched dataset | `data/processed/netflix_imdb_master_matched_enriched.csv` | CSV | CSV export of the enriched matched-only dataset. |
| Modeling source copy | `data/processed/netflix_imdb_modeling_source_copy.parquet` | Parquet | Exact frozen copy of the enriched matched dataset used as the modeling pipeline baseline. |
| Modeling source copy | `data/processed/netflix_imdb_modeling_source_copy.csv` | CSV | CSV export of the frozen modeling source snapshot. |
| Modeling dataset | `data/processed/netflix_imdb_modeling.parquet` | Parquet | Season-focused modeling cut derived from the enriched matched-only dataset. |
| Modeling dataset | `data/processed/netflix_imdb_modeling.csv` | CSV | CSV export of the season-focused modeling cut. |
| Modeling feature manifest | `data/processed/netflix_imdb_modeling_feature_manifest.csv` | CSV | Column-level manifest listing identifiers, predictors, targets, dropped fields, leakage flags, and missingness rates. |
| Modeling quality report | `data/processed/netflix_imdb_modeling_quality_report.csv` | CSV | Quality checks and row-count diagnostics for the strict season-to-season modeling dataset. |
| Preview sample | `data/processed/netflix_imdb_master_sample.csv` | CSV | Small inspection-friendly preview of the master dataset. |
| Unmatched series review | `data/processed/unmatched_series_rows.csv` | CSV | Remaining unresolved series-like Netflix rows after second-pass matching. |
| Ambiguous candidate review | `data/processed/ambiguous_match_candidates.csv` | CSV | Candidate-level review file for rows that remained ambiguous or conflicted. |
| Fuzzy-match review | `data/processed/matched_by_fuzzy_review.csv` | CSV | Rows resolved by the conservative fuzzy stage for manual QA. |
| Alternate-title review | `data/processed/matched_by_alternate_title_review.csv` | CSV | Rows matched through IMDb original-title or `title.akas` keys. |
| Unmatched method summary | `data/processed/unmatched_match_method_summary.csv` | CSV | Compact summary of remaining unmatched categories, meanings, and example titles. |
| Match improvement summary | `data/processed/match_improvement_summary.csv` | CSV | Before/after validation metrics for the second-pass merge. |
| Third-pass movie review | `data/processed/third_pass_movie_matches_review.csv` | CSV | Review file for movie rows newly matched by the third-pass incremental matcher. |
| Third-pass series-overall review | `data/processed/third_pass_series_overall_matches_review.csv` | CSV | Review file for parent-series rows newly matched by the third-pass incremental matcher. |
| Third-pass unresolved review | `data/processed/third_pass_still_unmatched.csv` | CSV | Remaining unresolved rows after the third-pass incremental matcher. |
| Third-pass ambiguous review | `data/processed/third_pass_ambiguous_candidates.csv` | CSV | Candidate-level review rows for ambiguous third-pass cases. |
| Third-pass delta summary | `data/processed/third_pass_delta_summary.csv` | CSV | Before/after metrics for the incremental third-pass multi-grain matcher. |

## Pipeline Notes

- Raw files are treated as read-only inputs.
- The merge starts with exact normalized title plus season number, then expands to original titles, `title.akas`, canonicalized keys, and a conservative fuzzy stage.
- Year consistency, start-year distance, title-type preference, and vote counts are used only as deterministic tie-breakers and are recorded in the final dataset.
- The enrichment workflow is separate from matching and starts from the frozen third-pass master dataset.
- Enrichment first creates a matched-only baseline copy, then builds cached IMDb feature tables only for matched IMDb ids.
- Unmatched Netflix rows are intentionally excluded from the enrichment join process.
- The modeling workflow is separate from enrichment and begins by creating a frozen copy of the enriched matched dataset.
- The final strict modeling dataset keeps only matched `series_season` rows with stable series ids, valid season numbers, usable current-season Netflix metrics, and observed consecutive next-season targets.
- Leakage-prone full-series summary fields stay in the copied baseline and enriched dataset but are excluded from the final strict modeling matrix.
