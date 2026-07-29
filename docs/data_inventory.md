# Data Inventory

## Raw Inputs

| File | Location | Notes |
| --- | --- | --- |
| Netflix export | `data/raw/netflix/netflixlist7-export.csv` | Main Netflix source file with title, type, runtime, yearly hours/views, and totals. |
| IMDb title basics | `data/raw/imdb/title.basics.tsv.gz` | Compressed IMDb title metadata used to identify parent series. |
| IMDb episode links | `data/raw/imdb/title.episode.tsv.gz` | Compressed IMDb episode-to-parent mapping with season numbers. |
| IMDb ratings | `data/raw/imdb/title.ratings.tsv.gz` | Compressed IMDb average rating and vote counts. |

## Intermediate Outputs

| File | Location | Created by | Purpose |
| --- | --- | --- | --- |
| Cleaned Netflix table | `data/interim/netflix_cleaned.csv` | `scripts/data_preparation/prepare_netflix.py` | Standardized Netflix titles, seasons, formats, and canonical metrics. |
| IMDb series-season table | `data/interim/imdb_series_seasons.csv` | `scripts/data_preparation/prepare_imdb.py` | Parent-series metadata aggregated to the season level. |
| IMDb title-key table | `data/interim/imdb_title_keys.csv` | `scripts/data_preparation/prepare_imdb.py` | Match-key variants built from IMDb primary, original, and alternate titles. |

## Final Output

| File | Location | Format | Purpose |
| --- | --- | --- | --- |
| Master merged dataset | `data/processed/netflix_imdb_master.parquet` | Parquet | Final left-joined Netflix-to-IMDb series-season dataset. |
| Master merged dataset | `data/processed/netflix_imdb_master.csv` | CSV | Full CSV export of the final merged dataset for teammates who prefer plain-text tabular files. |
| Preview sample | `data/processed/netflix_imdb_master_sample.csv` | CSV | Small inspection-friendly preview of the master dataset. |
| Unmatched series review | `data/processed/unmatched_series_rows.csv` | CSV | Remaining unresolved series-like Netflix rows after second-pass matching. |
| Ambiguous candidate review | `data/processed/ambiguous_match_candidates.csv` | CSV | Candidate-level review file for rows that remained ambiguous or conflicted. |
| Fuzzy-match review | `data/processed/matched_by_fuzzy_review.csv` | CSV | Rows resolved by the conservative fuzzy stage for manual QA. |
| Alternate-title review | `data/processed/matched_by_alternate_title_review.csv` | CSV | Rows matched through IMDb original-title or `title.akas` keys. |
| Unmatched method summary | `data/processed/unmatched_match_method_summary.csv` | CSV | Compact summary of remaining unmatched categories, meanings, and example titles. |
| Match improvement summary | `data/processed/match_improvement_summary.csv` | CSV | Before/after validation metrics for the second-pass merge. |

## Pipeline Notes

- Raw files are treated as read-only inputs.
- The merge starts with exact normalized title plus season number, then expands to original titles, `title.akas`, canonicalized keys, and a conservative fuzzy stage.
- Year consistency, start-year distance, title-type preference, and vote counts are used only as deterministic tie-breakers and are recorded in the final dataset.
