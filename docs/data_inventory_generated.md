# Generated Data Inventory

Scanned raw data directory: `data/raw`

Supported file types: .csv, .parquet, .tsv, .tsv.gz, .xls, .xlsm, .xlsx

Datasets inspected: 8

## `data/raw/imdb/name.basics.tsv.gz`

- File name: `name.basics.tsv.gz`
- Relative path: `data/raw/imdb/name.basics.tsv.gz`
- File size: 293.02 MB
- Rows: 15,526,439
- Columns: 6
- Duplicate rows: 0
- Candidate identifier columns: `nconst`

### Columns

| Column | Dtype | Missing % |
| --- | --- | ---: |
| `nconst` | `string` | 0.00% |
| `primaryName` | `string` | 0.00% |
| `birthYear` | `float` | 95.63% |
| `deathYear` | `float` | 98.32% |
| `primaryProfession` | `string` | 20.23% |
| `knownForTitles` | `string` | 12.05% |

## `data/raw/imdb/title.akas.tsv.gz`

- File name: `title.akas.tsv.gz`
- Relative path: `data/raw/imdb/title.akas.tsv.gz`
- File size: 484.68 MB
- Rows: 58,622,642
- Columns: 8
- Duplicate rows: 0
- Candidate identifier columns: None

### Columns

| Column | Dtype | Missing % |
| --- | --- | ---: |
| `titleId` | `string` | 0.00% |
| `ordering` | `integer` | 0.00% |
| `title` | `string` | 0.00% |
| `region` | `string` | 21.68% |
| `language` | `string` | 32.69% |
| `types` | `string` | 67.07% |
| `attributes` | `string` | 99.47% |
| `isOriginalTitle` | `integer` | 0.00% |

### Notes

- Identifier-like columns were present, but none were fully non-null and unique.

## `data/raw/imdb/title.basics.tsv.gz`

- File name: `title.basics.tsv.gz`
- Relative path: `data/raw/imdb/title.basics.tsv.gz`
- File size: 214.32 MB
- Rows: 12,675,173
- Columns: 9
- Duplicate rows: 0
- Candidate identifier columns: `tconst`

### Columns

| Column | Dtype | Missing % |
| --- | --- | ---: |
| `tconst` | `string` | 0.00% |
| `titleType` | `string` | 0.00% |
| `primaryTitle` | `string` | 0.00% |
| `originalTitle` | `string` | 0.00% |
| `isAdult` | `integer` | 0.00% |
| `startYear` | `float` | 11.66% |
| `endYear` | `float` | 98.74% |
| `runtimeMinutes` | `mixed/object` | 64.13% |
| `genres` | `string` | 4.26% |

## `data/raw/imdb/title.crew.tsv.gz`

- File name: `title.crew.tsv.gz`
- Relative path: `data/raw/imdb/title.crew.tsv.gz`
- File size: 78.63 MB
- Rows: 12,675,796
- Columns: 3
- Duplicate rows: 0
- Candidate identifier columns: `tconst`

### Columns

| Column | Dtype | Missing % |
| --- | --- | ---: |
| `tconst` | `string` | 0.00% |
| `directors` | `string` | 44.33% |
| `writers` | `string` | 49.30% |

## `data/raw/imdb/title.episode.tsv.gz`

- File name: `title.episode.tsv.gz`
- Relative path: `data/raw/imdb/title.episode.tsv.gz`
- File size: 51.64 MB
- Rows: 9,796,655
- Columns: 4
- Duplicate rows: 0
- Candidate identifier columns: `tconst`

### Columns

| Column | Dtype | Missing % |
| --- | --- | ---: |
| `tconst` | `string` | 0.00% |
| `parentTconst` | `string` | 0.00% |
| `seasonNumber` | `float` | 20.88% |
| `episodeNumber` | `float` | 20.88% |

## `data/raw/imdb/title.principals.tsv.gz`

- File name: `title.principals.tsv.gz`
- Relative path: `data/raw/imdb/title.principals.tsv.gz`
- File size: 739.65 MB
- Rows: 100,782,146
- Columns: 6
- Duplicate rows: 0
- Candidate identifier columns: None

### Columns

| Column | Dtype | Missing % |
| --- | --- | ---: |
| `tconst` | `string` | 0.00% |
| `ordering` | `integer` | 0.00% |
| `nconst` | `string` | 0.00% |
| `category` | `string` | 0.00% |
| `job` | `string` | 80.72% |
| `characters` | `string` | 51.20% |

### Notes

- Identifier-like columns were present, but none were fully non-null and unique.

## `data/raw/imdb/title.ratings.tsv.gz`

- File name: `title.ratings.tsv.gz`
- Relative path: `data/raw/imdb/title.ratings.tsv.gz`
- File size: 8.19 MB
- Rows: 1,699,786
- Columns: 3
- Duplicate rows: 0
- Candidate identifier columns: `tconst`

### Columns

| Column | Dtype | Missing % |
| --- | --- | ---: |
| `tconst` | `string` | 0.00% |
| `averageRating` | `float` | 0.00% |
| `numVotes` | `integer` | 0.00% |

## `data/raw/netflix/netflixlist7-export.csv`

- File name: `netflixlist7-export.csv`
- Relative path: `data/raw/netflix/netflixlist7-export.csv`
- File size: 3.52 MB
- Rows: 32,576
- Columns: 13
- Duplicate rows: 0
- Candidate identifier columns: None

### Columns

| Column | Dtype | Missing % |
| --- | --- | ---: |
| `Title Name` | `string` | 0.00% |
| `Runtime` | `string` | 11.59% |
| `Type` | `string` | 0.00% |
| `2023 Hours` | `string` | 0.00% |
| `2023 Views` | `string` | 0.00% |
| `2024 Hours` | `string` | 0.00% |
| `2024 Views` | `string` | 0.00% |
| `2025 Hours` | `string` | 0.00% |
| `2025 Views` | `string` | 0.00% |
| `2026 H1 Hours` | `string` | 0.00% |
| `2026 H1 Views` | `string` | 0.00% |
| `Total Hours Viewed` | `string` | 0.00% |
| `Total Views` | `string` | 0.00% |

### Notes

- No obvious identifier-like column names were detected.
