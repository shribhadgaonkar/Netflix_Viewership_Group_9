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
| `first_observed_halfyear_period` | Earliest half-year Netflix reporting period in the local half-year files where both views and hours are present for the normalized Netflix title. |
| `first_observed_halfyear_views` | Views reported in the earliest valid observed half-year period. |
| `first_observed_halfyear_hours` | Hours reported in the earliest valid observed half-year period. |
| `first_halfyear_hours_per_view` | `first_observed_halfyear_hours / first_observed_halfyear_views`, with missing output when views are missing or zero. |
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

## Third-Pass Grain Fields

| Field | Description |
| --- | --- |
| `prior_match_status` | Match status inherited from the second-pass master before the third-pass multi-grain rescue step. |
| `prior_match_method` | Match method inherited from the second-pass master before the third-pass rescue step. |
| `netflix_content_grain` | Conservative content-grain classification used by the third pass, such as `movie`, `series_parent`, or `series_season`. |
| `third_pass_match_method` | Final method recorded by the third-pass multi-grain rescue logic. |
| `third_pass_match_stage` | Third-pass matching stage that produced the final decision for the row. |
| `third_pass_match_confidence` | Confidence score attached to the third-pass decision. |
| `imdb_match_entity_type` | Matched IMDb grain used by the final dataset, such as `movie`, `series_parent`, or `series_season`. |
| `imdb_resolved_tconst` | Stable IMDb identifier carried forward as the resolved matched entity id. |

## Enrichment Baseline Fields

| Field | Description |
| --- | --- |
| `imdb_enrichment_entity_id` | Stable IMDb entity id used as the enrichment join key. Movie rows use the matched movie id; series rows use the matched parent-series id. |
| `imdb_enrichment_entity_type` | Enrichment grain used during feature joins. |
| `imdb_enrichment_parent_tconst` | Parent-series identifier used when enriching series-parent or series-season rows. |
| `imdb_enrichment_applied` | Flag showing that the matched-only row passed through the enrichment workflow. |
| `imdb_enrichment_sources_used` | Pipe-delimited list of IMDb raw files that contributed enrichment features to the row. |
| `imdb_enrichment_notes` | Human-readable explanation of how enrichment was applied to the row’s content grain. |

## Enrichment Feature Groups

| Field | Description |
| --- | --- |
| `imdb_is_adult` | IMDb adult-content flag from `title.basics.tsv.gz`. |
| `genre_count` | Number of IMDb genres listed for the matched entity. |
| `genre_drama` to `genre_sci_fi` | Major-genre indicator flags derived from IMDb genres. |
| `is_animation` | Convenience flag derived from IMDb genre indicators. |
| `is_documentary` | Convenience flag derived from IMDb genre indicators. |
| `is_kids_family_like` | Conservative family-oriented flag derived from `Animation` and `Family` genres. |
| `imdb_log_num_votes` | `log1p` transformation of `imdb_num_votes`. |
| `imdb_rating_votes_interaction` | Interaction term between IMDb rating and log vote count. |
| `imdb_total_episode_count` | Total episodes observed across all seasons for the matched parent series. |
| `imdb_max_season_number` | Maximum season number observed for the matched parent series. |
| `imdb_avg_episodes_per_season` | Mean number of episodes per season for the matched parent series. |
| `imdb_min_episodes_per_season` | Minimum number of episodes observed in a season for the matched parent series. |
| `imdb_max_episodes_per_season` | Maximum number of episodes observed in a season for the matched parent series. |
| `imdb_single_season_flag` | Indicates that the matched parent series has only one observed season. |
| `imdb_multi_season_flag` | Indicates that the matched parent series has more than one observed season. |
| `imdb_aka_region_count` | Number of unique IMDb alternate-title regions recorded for the matched entity. |
| `imdb_aka_language_count` | Number of unique IMDb alternate-title languages recorded for the matched entity. |
| `imdb_has_us_title` | Flag indicating the presence of a U.S. alternate title in IMDb. |
| `imdb_has_international_title_variants` | Flag indicating that the matched entity has alternate titles across multiple regions. |
| `imdb_director_count` | Number of directors linked to the matched entity through `title.crew.tsv.gz`. |
| `imdb_writer_count` | Number of writers linked to the matched entity through `title.crew.tsv.gz`. |
| `imdb_director_names` | Pipe-delimited director names for the matched entity. |
| `imdb_writer_names` | Pipe-delimited writer names for the matched entity. |
| `imdb_director_mean_birth_year` | Mean birth year of linked directors when available in `name.basics.tsv.gz`. |
| `imdb_writer_mean_birth_year` | Mean birth year of linked writers when available in `name.basics.tsv.gz`. |
| `imdb_principal_count` | Number of principal credits linked to the matched entity through `title.principals.tsv.gz`. |
| `imdb_actor_count` | Number of `actor` credits linked to the matched entity. |
| `imdb_actress_count` | Number of `actress` credits linked to the matched entity. |
| `imdb_self_count` | Number of `self` credits linked to the matched entity. |
| `imdb_producer_count` | Number of `producer` credits linked to the matched entity. |
| `imdb_writer_credit_count` | Number of `writer` principal credits linked to the matched entity. |
| `imdb_director_credit_count` | Number of `director` principal credits linked to the matched entity. |
| `imdb_top_cast_nconsts` | Pipe-delimited top cast person ids selected from principal credits. |
| `imdb_top_cast_names` | Pipe-delimited top cast names selected from principal credits. |
| `imdb_top_cast_count_used` | Number of top-cast entries retained in the compact cast summary. |
| `imdb_top_cast_mean_birth_year` | Mean birth year of the retained top-cast entries when available. |
| `imdb_top_cast_known_for_count_proxy` | Mean count of `knownForTitles` entries across the retained top-cast entries. |
| `imdb_top_cast_profession_mix` | Pipe-delimited summary of professions observed across the retained top-cast entries. |

## Derived Lifecycle Fields

| Field | Description |
| --- | --- |
| `imdb_movie_flag` | Indicates that the enriched IMDb entity is movie-like (`movie` or `tvMovie`). |
| `imdb_series_flag` | Indicates that the enriched IMDb entity is series-like (`tvSeries` or `tvMiniSeries`). |
| `imdb_miniseries_flag` | Indicates that the enriched IMDb entity is a `tvMiniSeries`. |
| `imdb_series_ended_flag` | Indicates that the enriched series ended before the current project year. |
| `imdb_series_ongoing_flag` | Indicates that the enriched series has no end year or is still active in the current project year. |
| `imdb_title_age_years` | Difference between the current project year and the IMDb start year. |
| `imdb_years_active` | Inclusive duration between the IMDb start year and end year, or the current project year when still ongoing. |
| `netflix_imdb_year_gap` | Signed difference between the Netflix-side reference year and the IMDb start year. |
| `netflix_imdb_runtime_gap` | Signed difference between the Netflix runtime and the IMDb runtime. |

## Modeling Fields

| Field | Description |
| --- | --- |
| `series_group_key` | Stable within-series grouping key used to sort seasons, create targets, and create lag features. |
| `netflix_reference_year` | Row-level Netflix reference year used for time-aware feature engineering, built from `netflix_release_year` and then `netflix_title_year_hint` as a fallback. |
| `season_order` | Modeling-oriented season order derived from `netflix_season_number`. |

## Strict Modeling Fields

| Field | Description |
| --- | --- |
| `target_next_season_views` | Next observed season's Netflix views within the same `series_group_key`. |
| `target_next_season_hours` | Next observed season's Netflix hours viewed within the same `series_group_key`. |
| `target_view_change_absolute` | Next-season views minus current-season views. |
| `target_view_change_percent` | `target_view_change_absolute / netflix_views` for the current season row. |
| `target_hours_change_absolute` | Next-season hours minus current-season hours. |
| `target_hours_change_percent` | `target_hours_change_absolute / netflix_hours_viewed` for the current season row. |
| `target_is_viewership_increase` | Binary target equal to `1` when next-season views exceed current-season views. |
| `target_is_hours_increase` | Binary target equal to `1` when next-season hours exceed current-season hours. |
| `prev_season_views` | Previous observed season's Netflix views within the same `series_group_key`. |
| `prev_season_hours` | Previous observed season's Netflix hours viewed within the same `series_group_key`. |
| `prev_season_rating` | Previous observed season's IMDb rating field, retained as a lag-style series context feature. |
| `prev_view_change_absolute` | Prior season's observed absolute views change, computed from earlier seasons only. |
| `prev_view_change_percent` | Prior season's observed percent views change, computed from earlier seasons only. |
| `prev_hours_change_absolute` | Prior season's observed absolute hours change, computed from earlier seasons only. |
| `prev_hours_change_percent` | Prior season's observed percent hours change, computed from earlier seasons only. |
| `has_prev_season_observation` | Binary flag indicating whether the current row has a prior observed season in the dataset. |
| `imdb_age_at_netflix_year` | Time-aware IMDb title age using the row-level Netflix reference year instead of the global current year. |
| `imdb_started_before_netflix_flag` | Binary flag indicating that the IMDb title started before the row-level Netflix reference year. |
| `imdb_same_year_as_netflix_flag` | Binary flag indicating that the IMDb title started in the same year as the row-level Netflix reference year. |
| `missing_imdb_average_rating` | Missingness indicator for `imdb_average_rating`. |
| `missing_imdb_num_votes` | Missingness indicator for `imdb_num_votes`. |
| `missing_netflix_runtime` | Missingness indicator for `netflix_runtime`. |
| `missing_netflix_imdb_runtime_gap` | Missingness indicator for `netflix_imdb_runtime_gap`. |
| `missing_cast_features` | Missingness indicator for cast-summary features. |
| `missing_release_year` | Missingness indicator for the row-level Netflix reference year. |
