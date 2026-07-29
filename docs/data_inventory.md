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

## Final Output

| File | Location | Format | Purpose |
| --- | --- | --- | --- |
| Master merged dataset | `data/processed/netflix_imdb_master.parquet` | Parquet | Final left-joined Netflix-to-IMDb series-season dataset. |
| Preview sample | `data/processed/netflix_imdb_master_sample.csv` | CSV | Small inspection-friendly preview of the master dataset. |

## Pipeline Notes

- Raw files are treated as read-only inputs.
- The merge starts with exact normalized title plus season number.
- Year consistency is used to resolve ambiguity or flag conflicts, not to force uncertain matches.
