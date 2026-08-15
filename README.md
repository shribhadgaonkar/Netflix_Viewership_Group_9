# Netflix Viewership Analysis — GRAD 504 Group 9

## Project Purpose

The purpose of this repository is to analyze **season-to-season shifts in Netflix viewership** and understand the factors associated with an increase or decrease in viewership across consecutive seasons of a series.

The project combines Netflix viewing engagement information with IMDb metadata to support data exploration, modeling, and evaluation.

---
## Team Members Group#9
 - Aatir Cheema
 - Shrirang Bhadgaonkar
 - Temitope Oluyomi

## Data Sources

This project uses two primary publicly available data sources.

### 1. Netflix Movie & TV Engagement Reports (2023–2026)

The Netflix engagement dataset contains approximately **32,576 entries** covering global viewing engagement across four years.

The dataset includes information such as:

* Movie or TV title
* Content type
* Runtime
* Hours viewed
* Views per reporting period
* Total views

**Source:** What's on Netflix
[Search Netflix's Movie & TV Engagement Reports (2023–2026)](https://www.whats-on-netflix.com/most-popular/netflix-engagement-report-search/)

---

### 2. IMDb Non-Commercial Datasets

IMDb Non-Commercial Datasets are used to enrich the Netflix engagement data with additional information about titles, seasons, ratings, genres, cast, crew, and related metadata.

The IMDb datasets used in this project include:

* `title.basics` — runtime, genre, start year, and end year
* `title.episode` — season and episode structure
* `title.ratings` — IMDb ratings and number of votes
* `title.crew` — directors and writers
* `title.principals` — principal cast and crew information
* `title.akas` — alternate titles, languages, and regions

**Source:** IMDb
[IMDb Data Files Download](https://datasets.imdbws.com/)

Both data sources are publicly available for non-commercial and academic use, subject to their respective usage terms.

---

## Exploratory Data Analysis

Exploratory Data Analysis (EDA) is performed to understand the dataset, identify important relationships between variables, and investigate potential predictors of season-to-season viewership changes.

---

## Modeling and Evaluation

After data preparation and exploratory analysis, the project performs modeling and evaluation using two techniques:

1. **Multiple Linear Regression**
2. **Decision Tree**

The models are used to investigate and evaluate the relationship between the selected predictors and season-to-season Netflix viewership changes.

Detailed methodology, model configuration, results, comparison, and evaluation are documented separately in the **project report**.

---

## Repository Flow

The overall project flow is:

```text
Netflix Engagement Data
          +
IMDb Non-Commercial Data
          |
          v
Data Preparation and Integration
          |
          v
Exploratory Data Analysis
          |
          v
Feature / Predictor Selection
          |
          v
Modeling
   ├── Multiple Linear Regression
   └── Decision Tree
          |
          v
Model Evaluation
          |
          v
Project Findings and Report
```

---

## Project Overview

This project investigates the factors associated with changes in Netflix
series viewership between seasons.

The initial objective is to predict whether the viewership of a subsequent
season will increase or decrease compared with the previous season.

## Repository Structure

```text
Netflix_Viewership_Group_9/
├── config/             # Project configuration
├── data/
│   ├── raw/            # Original downloaded data, Currently not added here for the size restriction
│   ├── interim/        # Cleaned and partially transformed data
│   ├── processed/      # Modeling-ready datasets
│   └── samples/        # Small sample datasets allowed in Git
├── docs/               # Data and methodology documentation
├── notebooks/          # Exploration and analysis notebooks
├── reports/
│   └── figures/        # Generated charts
├── src/                # Reusable Python scripts
├── tests/              # Automated tests
├── requirements.txt
└── README.md
```
