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
| `netflix_canonical_title` | Canonicalized parsed title used in the second-pass exact-matching stages. |
| `netflix_raw_normalized_title` | Normalized version of the full raw Netflix title before season suffix removal. |
| `netflix_raw_canonical_title` | Canonicalized version of the full raw Netflix title before season suffix removal. |
| `netflix_compact_title` | Space-free canonical title used for compact exact matching. |
| `netflix_season_parse_method` | Parsing rule that produced the season number, if any. |
| `netflix_season_parse_confidence` | Confidence score attached to the season parser output. |
| `netflix_title_parse_notes` | Notes describing heuristics or cleanup steps used during title parsing. |
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
| `imdb_primary_normalized_title` | Explicit normalized primary title key. |
| `imdb_original_normalized_title` | Normalized original title retained for inspection and future fallback logic. |
| `imdb_primary_canonical_title` | Canonicalized primary title key used in second-pass exact matching. |
| `imdb_original_canonical_title` | Canonicalized original title key used in second-pass exact matching. |
| `imdb_aka_normalized_titles` | Pipe-delimited normalized alternate-title keys aggregated from `title.akas.tsv.gz`. |
| `imdb_aka_canonical_titles` | Pipe-delimited canonical alternate-title keys aggregated from `title.akas.tsv.gz`. |
| `imdb_aka_title_count` | Number of unique alternate titles retained for the parent series. |
| `imdb_start_year` | Series start year from IMDb. |
| `imdb_end_year` | Series end year from IMDb when present. |
| `imdb_genres` | IMDb genre list for the parent series. |
| `imdb_average_rating` | IMDb average rating for the parent series. |
| `imdb_num_votes` | IMDb vote count for the parent series. |
| `imdb_parent_season_count` | Number of seasons retained for the IMDb parent series in the prepared season table. |
| `imdb_season_number` | Season number derived from episode records. |
| `imdb_season_episode_count` | Number of episodes linked to the parent series and season. |

## Merge Metadata Fields

| Field | Description |
| --- | --- |
| `match_status` | `matched` or `unmatched`. |
| `match_method` | How the row was resolved, such as `exact_title_season`, `exact_title_season_year`, or an explicit unmatched reason. |
| `match_stage` | Matching stage that produced the final decision, such as primary exact, aka exact, canonical exact, or fuzzy. |
| `match_confidence` | Numeric confidence score for the chosen match decision. |
| `match_notes` | Human-readable explanation for ambiguous or unresolved cases. |
| `candidate_imdb_count` | Number of IMDb candidates found on the exact normalized title plus season key. |
| `candidate_imdb_parent_tconsts` | Pipe-delimited IMDb parent identifiers considered for the row. |
| `candidate_imdb_primary_titles` | Pipe-delimited IMDb titles considered for the row. |
| `candidate_match_source` | Which IMDb title source produced the winning match or unresolved candidate set: `primary`, `original`, `aka`, `fuzzy`, or `manual_override`. |
| `netflix_match_key_used` | Netflix-side title key used in the successful or final attempted match stage. |
| `imdb_match_key_used` | IMDb-side title key used for the successful match. |
| `candidate_rank` | Rank of the selected IMDb candidate after deterministic tie-breaking. |
| `ambiguity_resolution_method` | Final tie-break rule that resolved a multi-candidate match when applicable. |
| `year_consistency_flag` | Whether the selected match agreed with an available Netflix year reference. |
| `year_distance` | Absolute distance between the Netflix reference year and the IMDb start year when both exist. |
| `title_similarity_score` | Similarity score used by the fuzzy stage or `100` for exact-key matches. |
