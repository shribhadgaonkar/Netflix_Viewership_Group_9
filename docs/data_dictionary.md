# Data Dictionary

## Netflix Cleaned Fields

| Field | Description |
| --- | --- |
| `netflix_row_id` | Stable row identifier assigned during cleaning. |
| `netflix_title_raw` | Original title text from the Netflix source file. |
| `netflix_series_title` | Base title after removing detected season-style suffixes when present. |
| `netflix_normalized_title` | Lowercased and punctuation-normalized title used for exact matching. |
| `netflix_season_number` | Extracted season-like number when the Netflix title explicitly contains one. |
| `netflix_season_label` | Source pattern used to derive the season number, such as `season`, `series`, or `limited_series`. |
| `netflix_title_year_hint` | Year parsed from a trailing title disambiguator like `(2011)` when available. |
| `netflix_format` | Conservative classification of the Netflix row as `series`, `movie`, or `unknown`. |
| `netflix_hours_viewed` | Canonical hours-viewed metric, currently populated from total hours when available. |
| `netflix_views` | Canonical views metric, currently populated from total views when available. |
| `netflix_runtime` | Runtime converted to minutes when parsable. |
| `source_netflix_file` | Relative path of the raw Netflix input file used for the row. |

## IMDb Series-Season Fields

| Field | Description |
| --- | --- |
| `imdb_parent_tconst` | Parent IMDb title identifier for the series. |
| `imdb_primary_title` | IMDb primary series title. |
| `imdb_original_title` | IMDb original series title. |
| `imdb_normalized_title` | Normalized primary title used for exact matching. |
| `imdb_original_normalized_title` | Normalized original title retained for inspection and future fallback logic. |
| `imdb_start_year` | Series start year from IMDb. |
| `imdb_end_year` | Series end year from IMDb when present. |
| `imdb_genres` | IMDb genre list for the parent series. |
| `imdb_average_rating` | IMDb average rating for the parent series. |
| `imdb_num_votes` | IMDb vote count for the parent series. |
| `imdb_season_number` | Season number derived from episode records. |
| `imdb_season_episode_count` | Number of episodes linked to the parent series and season. |

## Merge Metadata Fields

| Field | Description |
| --- | --- |
| `match_status` | `matched` or `unmatched`. |
| `match_method` | How the row was resolved, such as `exact_title_season`, `exact_title_season_year`, or an explicit unmatched reason. |
| `match_confidence` | Numeric confidence score for the chosen match decision. |
| `match_notes` | Human-readable explanation for ambiguous or unresolved cases. |
| `candidate_imdb_count` | Number of IMDb candidates found on the exact normalized title plus season key. |
| `candidate_imdb_parent_tconsts` | Pipe-delimited IMDb parent identifiers considered for the row. |
| `candidate_imdb_primary_titles` | Pipe-delimited IMDb titles considered for the row. |
| `year_consistency_flag` | Whether the selected match agreed with an available Netflix year reference. |
