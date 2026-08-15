from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = ROOT / "notebooks" / "03_modeling_eda.ipynb"


def md(text: str):
    return nbf.v4.new_markdown_cell(dedent(text).strip() + "\n")


def code(text: str):
    return nbf.v4.new_code_cell(dedent(text).strip() + "\n")


cells = [
    md(
        """
        # Modeling Dataset EDA

        This notebook performs exploratory data analysis for the final machine-learning modeling dataset used in the GRAD 504 project.

        **Expected Python environment / kernel:** `Python (.venv) - Netflix_Viewership_Group_9`

        The analysis is designed to run from the repository's existing `.venv` and uses repository-relative paths only.
        """
    ),
    md("## 1. Imports and Configuration"),
    code(
        """
        from pathlib import Path
        from itertools import combinations

        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt

        from IPython.display import Markdown, display
        from scipy import stats
        from sklearn.feature_selection import mutual_info_classif
        from statsmodels.stats.outliers_influence import variance_inflation_factor

        pd.set_option("display.max_columns", 200)
        pd.set_option("display.max_rows", 200)
        pd.set_option("display.width", 160)
        pd.set_option("display.float_format", lambda x: f"{x:,.4f}")

        def find_repo_root(start: Path | None = None) -> Path:
            start = Path.cwd() if start is None else Path(start)
            for candidate in [start, *start.parents]:
                if (candidate / "data" / "processed" / "netflix_imdb_modeling.csv").exists():
                    return candidate
            raise FileNotFoundError("Could not locate the repository root from the current working directory.")

        ROOT = find_repo_root()
        DATA_DIR = ROOT / "data" / "processed"
        MODELING_PATH = DATA_DIR / "netflix_imdb_modeling.csv"
        ENRICHED_PATH = DATA_DIR / "netflix_imdb_master_matched_enriched.csv"
        MANIFEST_PATH = DATA_DIR / "netflix_imdb_modeling_feature_manifest.csv"

        modeling_df = pd.read_csv(MODELING_PATH)
        enriched_df = pd.read_csv(ENRICHED_PATH) if ENRICHED_PATH.exists() else None
        manifest_df = pd.read_csv(MANIFEST_PATH) if MANIFEST_PATH.exists() else None

        TARGET_COLUMN = "target_is_viewership_increase"
        TARGET_COLUMNS_KNOWN = [
            "target_next_season_views",
            "target_view_change_absolute",
            "target_view_change_percent",
            "target_is_viewership_increase",
            "target_next_season_hours",
            "target_hours_change_absolute",
            "target_hours_change_percent",
            "target_is_hours_increase",
        ]
        IDENTIFIER_PRIORITY = [
            "netflix_row_id",
            "series_group_key",
            "netflix_series_title",
            "season_order",
            "netflix_title_raw",
        ]

        if manifest_df is not None:
            manifest_present = manifest_df[manifest_df["column_name"].isin(modeling_df.columns)].copy()
            identifier_cols = manifest_present.loc[manifest_present["role"] == "identifier", "column_name"].tolist()
            predictor_cols = manifest_present.loc[manifest_present["role"] == "predictor", "column_name"].tolist()
            target_cols = manifest_present.loc[manifest_present["role"] == "target", "column_name"].tolist()
            dropped_manifest_cols = manifest_df.loc[manifest_df["role"] == "dropped", "column_name"].tolist()
        else:
            manifest_present = None
            target_cols = [col for col in TARGET_COLUMNS_KNOWN if col in modeling_df.columns]
            identifier_cols = [col for col in IDENTIFIER_PRIORITY if col in modeling_df.columns]
            predictor_cols = [col for col in modeling_df.columns if col not in set(identifier_cols + target_cols)]
            dropped_manifest_cols = []

        predictor_cols = [col for col in predictor_cols if col in modeling_df.columns]
        identifier_cols = [col for col in identifier_cols if col in modeling_df.columns]
        target_cols = [col for col in target_cols if col in modeling_df.columns]

        predictor_df = modeling_df[predictor_cols].copy()
        numeric_predictors = predictor_df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_predictors = [col for col in predictor_cols if col not in numeric_predictors]

        duplicate_row_count = int(modeling_df.duplicated().sum())
        duplicate_netflix_row_id_count = int(modeling_df["netflix_row_id"].duplicated().sum()) if "netflix_row_id" in modeling_df.columns else np.nan

        def outcome_label(value):
            return "Increase" if value == 1 else "No Increase"

        def draw_labeled_bar(ax, series, color="#4C78A8", horizontal=False, percent_denominator=None):
            if horizontal:
                ax.barh(series.index.astype(str), series.values, color=color)
                for idx, value in enumerate(series.values):
                    ax.text(value, idx, f" {value:,.0f}", va="center")
            else:
                ax.bar(series.index.astype(str), series.values, color=color)
                ymax = max(series.values) if len(series.values) else 0
                offset = ymax * 0.02 if ymax else 0.1
                for idx, value in enumerate(series.values):
                    label = f"{value:,.0f}"
                    if percent_denominator:
                        label += f"\\n({value / percent_denominator:.1%})"
                    ax.text(idx, value + offset, label, ha="center", va="bottom", fontsize=10)

        def make_corr_matrix_plot(df, title):
            corr = df.corr()
            fig, ax = plt.subplots(figsize=(10, 8))
            im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
            ax.set_xticks(range(len(corr.columns)))
            ax.set_yticks(range(len(corr.columns)))
            ax.set_xticklabels(corr.columns, rotation=45, ha="right")
            ax.set_yticklabels(corr.columns)
            ax.set_title(title)
            for i in range(len(corr.index)):
                for j in range(len(corr.columns)):
                    ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=8)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            plt.tight_layout()
            return corr

        def cramers_v_from_table(table: pd.DataFrame) -> float:
            chi2, _, _, _ = stats.chi2_contingency(table)
            n = table.to_numpy().sum()
            if n == 0:
                return np.nan
            r, k = table.shape
            phi2 = chi2 / n
            phi2corr = max(0, phi2 - ((k - 1) * (r - 1)) / max(n - 1, 1))
            rcorr = r - ((r - 1) ** 2) / max(n - 1, 1)
            kcorr = k - ((k - 1) ** 2) / max(n - 1, 1)
            denom = min(kcorr - 1, rcorr - 1)
            if denom <= 0:
                return np.nan
            return float(np.sqrt(phi2corr / denom))

        display(Markdown(
            f\"\"\"\
        Repository root: `{ROOT}`

        Modeling dataset: `{MODELING_PATH.relative_to(ROOT)}`

        Enriched reference dataset loaded: `{ENRICHED_PATH.exists()}`

        Feature manifest loaded: `{MANIFEST_PATH.exists()}`
        \"\"\"
        ))
        """
    ),
    md("## 2. Dataset Overview"),
    code(
        """
        print(f"Modeling dataset shape: {modeling_df.shape}")
        print(f"Rows: {len(modeling_df):,}")
        print(f"Columns: {modeling_df.shape[1]:,}")
        """
    ),
    code(
        """
        display(pd.DataFrame({"column_name": modeling_df.columns}))
        """
    ),
    code(
        """
        dtype_summary = pd.DataFrame({
            "column_name": modeling_df.columns,
            "dtype": modeling_df.dtypes.astype(str).values,
        })
        display(dtype_summary)

        print(f"Numeric predictor count: {len(numeric_predictors)}")
        print(f"Categorical predictor count: {len(categorical_predictors)}")
        print(f"Duplicate row count: {duplicate_row_count:,}")
        print(f"Duplicate netflix_row_id count: {duplicate_netflix_row_id_count:,}")
        """
    ),
    code(
        """
        display(modeling_df.describe(include="all").transpose())
        """
    ),
    code(
        """
        if manifest_df is not None:
            manifest_summary = pd.DataFrame({
                "group": ["identifier", "predictor", "target", "excluded feature"],
                "count": [
                    int((manifest_df["role"] == "identifier").sum()),
                    int((manifest_df["role"] == "predictor").sum()),
                    int((manifest_df["role"] == "target").sum()),
                    int((manifest_df["role"] == "dropped").sum()),
                ],
            })
            display(manifest_summary)

        classification_summary = pd.DataFrame({
            "classification": ["Identifiers", "Predictors", "Targets"],
            "count": [len(identifier_cols), len(predictor_cols), len(target_cols)],
            "columns": [
                ", ".join(identifier_cols),
                ", ".join(predictor_cols[:20]) + (" ..." if len(predictor_cols) > 20 else ""),
                ", ".join(target_cols),
            ],
        })
        display(classification_summary)
        """
    ),
    md("## 3. Identifier Review"),
    code(
        """
        identifier_use_map = {
            "netflix_row_id": "Row-level join key for debugging, traceability, and QA only.",
            "series_group_key": "Use for group-aware train/test splitting so seasons from the same series stay together.",
            "netflix_series_title": "Human-readable series label; not suitable as a baseline ML predictor.",
            "season_order": "Season ordering metadata; can help interpretation but should be reviewed carefully before use.",
            "netflix_title_raw": "Raw title text; not a baseline tabular predictor in the current modeling approach.",
        }

        identifier_review_rows = []
        for col in identifier_cols:
            identifier_review_rows.append({
                "identifier": col,
                "present_in_priority_list": col in IDENTIFIER_PRIORITY,
                "intended_use": identifier_use_map.get(col, "Identifier / join / audit field, not a baseline predictor."),
            })

        display(pd.DataFrame(identifier_review_rows))

        display(Markdown(
            \"\"\"\
        Identifier columns are reviewed separately because they should not be used as baseline model predictors.

        `netflix_row_id` is primarily useful for joining and debugging. `series_group_key` is especially important for future group-aware train/test splitting to avoid leakage across seasons from the same series. Title-text columns are useful for interpretation and audits, but they should not be fed directly into the current baseline tabular models.
        \"\"\"
        ))
        """
    ),
    md("## 4. Target-Variable Exploration"),
    code(
        """
        target_counts = modeling_df[TARGET_COLUMN].value_counts(dropna=False).sort_index()
        target_percents = modeling_df[TARGET_COLUMN].value_counts(normalize=True, dropna=False).sort_index() * 100

        target_summary = pd.DataFrame({
            "class_value": target_counts.index,
            "class_label": [outcome_label(v) if pd.notna(v) else "Missing" for v in target_counts.index],
            "count": target_counts.values,
            "percent": target_percents.values,
        })
        display(target_summary)

        positive_pct = float(target_percents.get(1, 0.0))
        negative_pct = float(target_percents.get(0, 0.0))
        print(f"Positive-class percentage (Increase): {positive_pct:.2f}%")
        print(f"Negative-class percentage (No Increase): {negative_pct:.2f}%")

        imbalance_gap = abs(positive_pct - negative_pct)
        target_interpretation = (
            "The target is fairly balanced, so accuracy will be easier to interpret alongside precision, recall, and ROC/PR metrics."
            if imbalance_gap < 15
            else "The target shows noticeable imbalance, so accuracy alone could be misleading during later model evaluation."
        )
        print(target_interpretation)
        """
    ),
    code(
        """
        plot_counts = pd.Series(
            [int(target_counts.get(0, 0)), int(target_counts.get(1, 0))],
            index=["No Increase", "Increase"],
        )

        fig, ax = plt.subplots(figsize=(8, 5))
        draw_labeled_bar(ax, plot_counts, percent_denominator=len(modeling_df))
        ax.set_title("Distribution of Next-Season Viewership Outcome")
        ax.set_xlabel("")
        ax.set_ylabel("Number of Season Observations")
        plt.tight_layout()
        plt.show()
        """
    ),
    md("## 5. Missing-Data Analysis"),
    code(
        """
        predictor_missing_df = pd.DataFrame({
            "feature": predictor_cols,
            "missing_count": predictor_df[predictor_cols].isna().sum().values,
        })
        predictor_missing_df["non_missing_count"] = len(modeling_df) - predictor_missing_df["missing_count"]
        predictor_missing_df["missing_percent"] = predictor_missing_df["missing_count"] / len(modeling_df) * 100
        predictor_missing_df = predictor_missing_df.sort_values(["missing_percent", "missing_count"], ascending=False).reset_index(drop=True)
        display(predictor_missing_df)

        high_missingness_summary = pd.DataFrame({
            "threshold": ["> 30%", "> 50%", "> 80%"],
            "feature_count": [
                int((predictor_missing_df["missing_percent"] > 30).sum()),
                int((predictor_missing_df["missing_percent"] > 50).sum()),
                int((predictor_missing_df["missing_percent"] > 80).sum()),
            ],
        })
        display(high_missingness_summary)
        """
    ),
    code(
        """
        missing_plot_df = predictor_missing_df.loc[predictor_missing_df["missing_count"] > 0].copy()

        if missing_plot_df.empty:
            print("No predictor columns have missing values.")
        else:
            fig, ax = plt.subplots(figsize=(10, max(4, len(missing_plot_df) * 0.25)))
            ax.barh(missing_plot_df["feature"], missing_plot_df["missing_percent"], color="#E45756")
            ax.set_title("Missingness Across Modeling Predictors")
            ax.set_xlabel("Missing Percentage")
            ax.set_ylabel("Predictor")
            ax.invert_yaxis()
            plt.tight_layout()
            plt.show()
        """
    ),
    md(
        """
        Features with elevated missingness will likely require different treatments depending on the modeling approach. Low to moderate missingness may be suitable for imputation, while very high missingness can justify exclusion or careful sensitivity analysis. Missingness flags can also be informative if the absence itself carries signal.
        """
    ),
    md("## 6. Constant and Near-Constant Predictors"),
    code(
        """
        constant_features_df = pd.DataFrame({
            "feature": predictor_cols,
            "nunique_including_nan": [predictor_df[col].nunique(dropna=False) for col in predictor_cols],
            "nunique_excluding_nan": [predictor_df[col].nunique(dropna=True) for col in predictor_cols],
        })
        constant_features_df = constant_features_df.loc[constant_features_df["nunique_excluding_nan"] <= 1].sort_values("feature").reset_index(drop=True)
        display(constant_features_df)

        near_constant_rows = []
        for col in predictor_cols:
            value_distribution = predictor_df[col].value_counts(normalize=True, dropna=False)
            top_share = float(value_distribution.iloc[0]) if not value_distribution.empty else np.nan
            if predictor_df[col].nunique(dropna=True) <= 5 and pd.notna(top_share) and top_share >= 0.95:
                near_constant_rows.append({
                    "feature": col,
                    "nunique_excluding_nan": predictor_df[col].nunique(dropna=True),
                    "most_common_share": top_share,
                })

        near_constant_df = pd.DataFrame(near_constant_rows)
        if near_constant_df.empty:
            near_constant_df = pd.DataFrame(columns=["feature", "nunique_excluding_nan", "most_common_share"])
        else:
            near_constant_df = near_constant_df.sort_values(["most_common_share", "feature"], ascending=[False, True]).reset_index(drop=True)
        display(near_constant_df)

        print("Recommendation: exclude constant predictors from model training because they contain no predictive information.")
        """
    ),
    md("## 7. Predictor Distribution and Skewness"),
    code(
        """
        numeric_skew_df = pd.DataFrame({
            "feature": numeric_predictors,
            "skewness": [predictor_df[col].dropna().skew() for col in numeric_predictors],
        })
        numeric_skew_df["abs_skewness"] = numeric_skew_df["skewness"].abs()
        numeric_skew_df["interpretation"] = pd.cut(
            numeric_skew_df["abs_skewness"],
            bins=[-np.inf, 0.5, 1.0, np.inf],
            labels=["fairly symmetric", "moderately skewed", "strongly skewed"],
        )
        numeric_skew_df = numeric_skew_df.sort_values("abs_skewness", ascending=False).reset_index(drop=True)
        display(numeric_skew_df.head(20))

        spotlight_features = [
            "netflix_views",
            "netflix_hours_viewed",
            "netflix_log_views",
            "netflix_log_hours",
            "imdb_num_votes",
            "imdb_log_num_votes",
        ]
        display(numeric_skew_df[numeric_skew_df["feature"].isin([c for c in spotlight_features if c in numeric_skew_df["feature"].values])])
        """
    ),
    md("## 8. Raw vs Log Viewership Distributions"),
    code(
        """
        if "netflix_views" in modeling_df.columns:
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.hist(modeling_df["netflix_views"].dropna(), bins=30, color="#4C78A8", edgecolor="white")
            ax.set_title("Distribution of Netflix Views")
            ax.set_xlabel("netflix_views")
            ax.set_ylabel("Frequency")
            plt.tight_layout()
            plt.show()
        else:
            print("Column netflix_views is not present.")
        """
    ),
    code(
        """
        if "netflix_log_views" in modeling_df.columns:
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.hist(modeling_df["netflix_log_views"].dropna(), bins=30, color="#72B7B2", edgecolor="white")
            ax.set_title("Distribution of Log-Transformed Netflix Views")
            ax.set_xlabel("netflix_log_views")
            ax.set_ylabel("Frequency")
            plt.tight_layout()
            plt.show()
        else:
            print("Column netflix_log_views is not present.")
        """
    ),
    code(
        """
        if "netflix_hours_viewed" in modeling_df.columns:
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.hist(modeling_df["netflix_hours_viewed"].dropna(), bins=30, color="#F58518", edgecolor="white")
            ax.set_title("Distribution of Netflix Hours Viewed")
            ax.set_xlabel("netflix_hours_viewed")
            ax.set_ylabel("Frequency")
            plt.tight_layout()
            plt.show()
        else:
            print("Column netflix_hours_viewed is not present.")
        """
    ),
    code(
        """
        if "netflix_log_hours" in modeling_df.columns:
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.hist(modeling_df["netflix_log_hours"].dropna(), bins=30, color="#54A24B", edgecolor="white")
            ax.set_title("Distribution of Log-Transformed Netflix Hours Viewed")
            ax.set_xlabel("netflix_log_hours")
            ax.set_ylabel("Frequency")
            plt.tight_layout()
            plt.show()
        else:
            print("Column netflix_log_hours is not present.")
        """
    ),
    md(
        """
        Raw viewership measures often exhibit strong right-skew because a small number of seasons receive exceptionally large audiences. The log-transformed versions should be compared directly to assess whether the transformation reduces skewness and produces a distribution that is easier for linear models to handle.
        """
    ),
    md("## 9. IMDb Vote Distribution"),
    code(
        """
        if "imdb_num_votes" in modeling_df.columns:
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.hist(modeling_df["imdb_num_votes"].dropna(), bins=30, color="#B279A2", edgecolor="white")
            ax.set_title("Distribution of IMDb Vote Count")
            ax.set_xlabel("imdb_num_votes")
            ax.set_ylabel("Frequency")
            plt.tight_layout()
            plt.show()
        else:
            print("Column imdb_num_votes is not present.")
        """
    ),
    code(
        """
        if "imdb_log_num_votes" in modeling_df.columns:
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.hist(modeling_df["imdb_log_num_votes"].dropna(), bins=30, color="#FF9DA6", edgecolor="white")
            ax.set_title("Distribution of Log-Transformed IMDb Vote Count")
            ax.set_xlabel("imdb_log_num_votes")
            ax.set_ylabel("Frequency")
            plt.tight_layout()
            plt.show()
        else:
            print("Column imdb_log_num_votes is not present.")
        """
    ),
    md("## 10. Numeric Predictor-to-Target Correlation"),
    code(
        """
        numeric_target_corr_rows = []
        target_numeric = modeling_df[TARGET_COLUMN]

        for col in numeric_predictors:
            subset = modeling_df[[col, TARGET_COLUMN]].dropna()
            if subset.empty or subset[col].nunique() <= 1:
                pearson_corr = np.nan
                spearman_corr = np.nan
            else:
                pearson_corr = subset[col].corr(subset[TARGET_COLUMN], method="pearson")
                spearman_corr = subset[col].corr(subset[TARGET_COLUMN], method="spearman")
            numeric_target_corr_rows.append({
                "feature": col,
                "pearson_correlation": pearson_corr,
                "spearman_correlation": spearman_corr,
            })

        numeric_target_corr_df = pd.DataFrame(numeric_target_corr_rows)
        numeric_target_corr_df["abs_pearson"] = numeric_target_corr_df["pearson_correlation"].abs()
        numeric_target_corr_df["abs_spearman"] = numeric_target_corr_df["spearman_correlation"].abs()
        numeric_target_corr_df["max_abs_correlation"] = numeric_target_corr_df[["abs_pearson", "abs_spearman"]].max(axis=1)
        numeric_target_corr_df = numeric_target_corr_df.sort_values(["max_abs_correlation", "feature"], ascending=[False, True]).reset_index(drop=True)
        display(numeric_target_corr_df.head(20))

        display(Markdown(
            \"\"\"\
        Because the target is binary (`0`/`1`), Pearson correlation with the target can be interpreted as a point-biserial relationship. These correlations are useful for screening linear association, but they do **not** imply causation and should not replace cross-validated model evaluation.
        \"\"\"
        ))
        """
    ),
    md("## 11. Predictor-Target Correlation Bar Chart"),
    code(
        """
        corr_plot_df = numeric_target_corr_df.head(15).sort_values("pearson_correlation")

        fig, ax = plt.subplots(figsize=(10, 6))
        colors = ["#E45756" if x < 0 else "#4C78A8" for x in corr_plot_df["pearson_correlation"]]
        ax.barh(corr_plot_df["feature"], corr_plot_df["pearson_correlation"], color=colors)
        ax.axvline(0, color="black", linewidth=1)
        ax.set_title("Top Numeric Predictors Associated with Next-Season Viewership Outcome")
        ax.set_xlabel("Pearson Correlation with target_is_viewership_increase")
        ax.set_ylabel("Predictor")
        plt.tight_layout()
        plt.show()
        """
    ),
    md("## 12. Correlation Heatmap"),
    code(
        """
        heatmap_features = [TARGET_COLUMN] + numeric_target_corr_df["feature"].head(15).tolist()
        heatmap_features = [col for col in heatmap_features if col in modeling_df.columns]
        heatmap_df = modeling_df[heatmap_features].dropna()

        if heatmap_df.empty:
            print("Not enough complete data to build the key-feature correlation matrix.")
        else:
            _ = make_corr_matrix_plot(heatmap_df, "Correlation Matrix of Key Modeling Features")
            plt.show()
        """
    ),
    md("## 13. Highly Correlated Predictor Pairs"),
    code(
        """
        numeric_corr_matrix = predictor_df[numeric_predictors].corr().abs()
        high_corr_pairs = []

        for i, col_1 in enumerate(numeric_corr_matrix.columns):
            for col_2 in numeric_corr_matrix.columns[i + 1:]:
                corr_value = predictor_df[[col_1, col_2]].corr().iloc[0, 1]
                if pd.notna(corr_value) and abs(corr_value) >= 0.90:
                    high_corr_pairs.append({
                        "feature_1": col_1,
                        "feature_2": col_2,
                        "correlation": corr_value,
                    })

        high_corr_pairs_df = pd.DataFrame(high_corr_pairs)
        if high_corr_pairs_df.empty:
            high_corr_pairs_df = pd.DataFrame(columns=["feature_1", "feature_2", "correlation"])
        else:
            high_corr_pairs_df = high_corr_pairs_df.sort_values("correlation", key=lambda s: s.abs(), ascending=False).reset_index(drop=True)
        display(high_corr_pairs_df)

        display(Markdown(
            \"\"\"\
        Highly correlated predictor pairs can create redundancy and unstable coefficient estimates, especially for logistic regression and other linear models. Raw versus transformed versions of the same variable, mirrored binary flags, and engineered derivatives are common sources of multicollinearity.
        \"\"\"
        ))
        """
    ),
    md("## 14. Runtime vs Hours-per-View Analysis"),
    code(
        """
        if {"netflix_runtime", "netflix_hours_per_view"}.issubset(modeling_df.columns):
            runtime_hours_df = modeling_df[["netflix_runtime", "netflix_hours_per_view"]].dropna()
            pearson_runtime_hours = runtime_hours_df["netflix_runtime"].corr(runtime_hours_df["netflix_hours_per_view"], method="pearson")
            spearman_runtime_hours = runtime_hours_df["netflix_runtime"].corr(runtime_hours_df["netflix_hours_per_view"], method="spearman")
            print(f"Pearson correlation: {pearson_runtime_hours:.4f}")
            print(f"Spearman correlation: {spearman_runtime_hours:.4f}")
        else:
            print("Required columns for runtime vs hours-per-view are not present.")
        """
    ),
    code(
        """
        if {"netflix_runtime", "netflix_hours_per_view"}.issubset(modeling_df.columns):
            plot_df = modeling_df[["netflix_runtime", "netflix_hours_per_view"]].dropna()
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.scatter(plot_df["netflix_runtime"], plot_df["netflix_hours_per_view"], alpha=0.6, color="#4C78A8")
            ax.set_title("Netflix Runtime vs Hours per View")
            ax.set_xlabel("netflix_runtime")
            ax.set_ylabel("netflix_hours_per_view")
            plt.tight_layout()
            plt.show()
        else:
            print("Required columns for runtime vs hours-per-view are not present.")
        """
    ),
    code(
        """
        if {"netflix_runtime", "first_halfyear_hours_per_view"}.issubset(modeling_df.columns):
            halfyear_runtime_df = modeling_df[["netflix_runtime", "first_halfyear_hours_per_view"]].dropna()
            summary = pd.DataFrame({
                "metric": ["pearson", "spearman"],
                "correlation": [
                    halfyear_runtime_df["netflix_runtime"].corr(halfyear_runtime_df["first_halfyear_hours_per_view"], method="pearson"),
                    halfyear_runtime_df["netflix_runtime"].corr(halfyear_runtime_df["first_halfyear_hours_per_view"], method="spearman"),
                ],
            })
            display(summary)
        else:
            print("Optional column first_halfyear_hours_per_view is not present for the runtime comparison table.")
        """
    ),
    md("## 15. Current-Season vs Next-Season Viewership"),
    code(
        """
        if {"netflix_log_views", "target_next_season_views"}.issubset(modeling_df.columns):
            current_next_df = modeling_df[["netflix_log_views", "target_next_season_views"]].dropna().copy()
            current_next_df["target_next_season_log_views"] = np.log1p(current_next_df["target_next_season_views"])
            pearson_current_next = current_next_df["netflix_log_views"].corr(current_next_df["target_next_season_log_views"], method="pearson")
            spearman_current_next = current_next_df["netflix_log_views"].corr(current_next_df["target_next_season_log_views"], method="spearman")
            print(f"Pearson correlation: {pearson_current_next:.4f}")
            print(f"Spearman correlation: {spearman_current_next:.4f}")
        else:
            print("Columns needed for current-season vs next-season viewership analysis are not present.")
        """
    ),
    code(
        """
        if {"netflix_log_views", "target_next_season_views"}.issubset(modeling_df.columns):
            plot_df = modeling_df[["netflix_log_views", "target_next_season_views"]].dropna().copy()
            plot_df["target_next_season_log_views"] = np.log1p(plot_df["target_next_season_views"])
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.scatter(plot_df["netflix_log_views"], plot_df["target_next_season_log_views"], alpha=0.6, color="#72B7B2")
            ax.set_title("Current-Season vs Next-Season Netflix Viewership")
            ax.set_xlabel("Current Season log1p(Views)")
            ax.set_ylabel("Next Season log1p(Views)")
            plt.tight_layout()
            plt.show()
        else:
            print("Columns needed for current-season vs next-season viewership analysis are not present.")
        """
    ),
    md(
        """
        This project distinguishes between predicting **absolute next-season viewership** and predicting **whether viewership increases**. A feature can be strongly associated with the level of next-season viewership without necessarily being equally informative for the binary growth outcome.
        """
    ),
    md("## 16. IMDb Rating vs Target"),
    code(
        """
        if "imdb_average_rating" in modeling_df.columns:
            rating_group_summary = modeling_df.groupby(TARGET_COLUMN)["imdb_average_rating"].agg(["mean", "median", "count"])
            rating_group_summary.index = rating_group_summary.index.map(outcome_label)
            display(rating_group_summary)
        else:
            print("Column imdb_average_rating is not present.")
        """
    ),
    code(
        """
        if "imdb_average_rating" in modeling_df.columns:
            box_data = [
                modeling_df.loc[modeling_df[TARGET_COLUMN] == 0, "imdb_average_rating"].dropna(),
                modeling_df.loc[modeling_df[TARGET_COLUMN] == 1, "imdb_average_rating"].dropna(),
            ]
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.boxplot(box_data, tick_labels=["No Increase", "Increase"])
            ax.set_title("IMDb Rating vs Next-Season Viewership Outcome")
            ax.set_xlabel("Target Outcome")
            ax.set_ylabel("IMDb Average Rating")
            plt.tight_layout()
            plt.show()
        else:
            print("Column imdb_average_rating is not present.")
        """
    ),
    md("## 17. Current Viewership vs Classification Target"),
    code(
        """
        if "netflix_log_views" in modeling_df.columns:
            box_data = [
                modeling_df.loc[modeling_df[TARGET_COLUMN] == 0, "netflix_log_views"].dropna(),
                modeling_df.loc[modeling_df[TARGET_COLUMN] == 1, "netflix_log_views"].dropna(),
            ]
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.boxplot(box_data, tick_labels=["No Increase", "Increase"])
            ax.set_title("Current-Season Viewership vs Next-Season Growth Outcome")
            ax.set_xlabel("Target Outcome")
            ax.set_ylabel("netflix_log_views")
            plt.tight_layout()
            plt.show()
        else:
            print("Column netflix_log_views is not present.")
        """
    ),
    md("## 18. Genre Analysis"),
    code(
        """
        genre_candidate_cols = [
            col for col in predictor_cols
            if col.startswith("genre_") and col != "genre_count"
        ]
        preferred_genres = [
            "genre_drama", "genre_comedy", "genre_action", "genre_thriller", "genre_crime",
            "genre_romance", "genre_animation", "genre_documentary", "genre_fantasy",
            "genre_horror", "genre_sci_fi", "genre_family",
        ]
        genre_cols = [col for col in preferred_genres if col in genre_candidate_cols] + [col for col in genre_candidate_cols if col not in preferred_genres]
        genre_cols = list(dict.fromkeys(genre_cols))

        genre_rows = []
        for col in genre_cols:
            subset = modeling_df.loc[modeling_df[col] == 1, [col, TARGET_COLUMN]]
            genre_rows.append({
                "genre_feature": col,
                "observation_count": len(subset),
                "increase_count": int(subset[TARGET_COLUMN].sum()),
                "increase_rate": subset[TARGET_COLUMN].mean() if len(subset) else np.nan,
            })

        genre_summary_df = pd.DataFrame(genre_rows).sort_values(["increase_rate", "observation_count"], ascending=[False, False]).reset_index(drop=True)
        display(genre_summary_df)
        """
    ),
    code(
        """
        if "genre_summary_df" in globals() and not genre_summary_df.empty:
            min_genre_n = 20
            genre_plot_df = genre_summary_df.loc[genre_summary_df["observation_count"] >= min_genre_n].sort_values("increase_rate")
            if genre_plot_df.empty:
                print(f"No genre indicators met the minimum sample size threshold of {min_genre_n}.")
            else:
                fig, ax = plt.subplots(figsize=(10, max(4, len(genre_plot_df) * 0.4)))
                ax.barh(genre_plot_df["genre_feature"], genre_plot_df["increase_rate"] * 100, color="#54A24B")
                ax.set_title("Next-Season Viewership Increase Rate by Genre")
                ax.set_xlabel("Increase Rate (%)")
                ax.set_ylabel("Genre Indicator")
                for idx, row in genre_plot_df.reset_index(drop=True).iterrows():
                    ax.text(row["increase_rate"] * 100, idx, f"  n={int(row['observation_count'])}", va="center")
                plt.tight_layout()
                plt.show()
        else:
            print("No genre indicator columns were available for analysis.")
        """
    ),
    md(
        """
        Genre relationships should be treated as descriptive rather than causal. Differences in increase rate may partly reflect release strategy, franchise effects, sample composition, or other variables that co-vary with genre.
        """
    ),
    md("## 19. Season Number Analysis"),
    code(
        """
        season_col = "netflix_season_number" if "netflix_season_number" in modeling_df.columns else ("season_order" if "season_order" in modeling_df.columns else None)
        if season_col is None:
            print("No season-number field is present for season-order analysis.")
        else:
            season_summary_df = (
                modeling_df
                .groupby(season_col)[TARGET_COLUMN]
                .agg(observation_count="count", increase_count="sum", increase_rate="mean")
                .reset_index()
                .sort_values(season_col)
            )
            display(season_summary_df)
        """
    ),
    code(
        """
        if "season_summary_df" in globals():
            plot_df = season_summary_df.copy()
            fig, ax = plt.subplots(figsize=(9, 5))
            ax.plot(plot_df[season_col], plot_df["increase_rate"] * 100, marker="o", color="#4C78A8")
            for _, row in plot_df.iterrows():
                if row["observation_count"] < 10:
                    ax.scatter(row[season_col], row["increase_rate"] * 100, color="#E45756", s=60, zorder=3)
            ax.set_title("Viewership Increase Rate by Current Season Number")
            ax.set_xlabel("Current Season Number")
            ax.set_ylabel("Next-Season Viewership Increase Rate (%)")
            plt.tight_layout()
            plt.show()
        else:
            print("Season summary table is not available.")
        """
    ),
    md("## 20. First-Observed Half-Year Feature Analysis"),
    code(
        """
        halfyear_numeric_cols = [
            col for col in [
                "first_observed_halfyear_views",
                "first_observed_halfyear_hours",
                "first_halfyear_hours_per_view",
            ]
            if col in modeling_df.columns
        ]
        halfyear_period_col = "first_observed_halfyear_period" if "first_observed_halfyear_period" in modeling_df.columns else None

        if halfyear_numeric_cols:
            halfyear_numeric_summary_rows = []
            for col in halfyear_numeric_cols:
                series = modeling_df[col]
                valid = modeling_df[[col, TARGET_COLUMN]].dropna()
                halfyear_numeric_summary_rows.append({
                    "feature": col,
                    "missing_count": int(series.isna().sum()),
                    "missing_percent": float(series.isna().mean() * 100),
                    "mean": float(series.mean()),
                    "median": float(series.median()),
                    "std": float(series.std()),
                    "min": float(series.min()),
                    "max": float(series.max()),
                    "skewness": float(series.dropna().skew()),
                    "pearson_to_target": valid[col].corr(valid[TARGET_COLUMN], method="pearson") if not valid.empty and valid[col].nunique() > 1 else np.nan,
                    "spearman_to_target": valid[col].corr(valid[TARGET_COLUMN], method="spearman") if not valid.empty and valid[col].nunique() > 1 else np.nan,
                })
            halfyear_numeric_summary_df = pd.DataFrame(halfyear_numeric_summary_rows)
            display(halfyear_numeric_summary_df)
        else:
            print("Optional half-year numeric features are not present; skipping numeric half-year analysis.")
        """
    ),
    code(
        """
        if "first_observed_halfyear_views" in modeling_df.columns:
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.hist(modeling_df["first_observed_halfyear_views"].dropna(), bins=30, color="#F58518", edgecolor="white")
            ax.set_title("Distribution of First Observed Half-Year Views")
            ax.set_xlabel("first_observed_halfyear_views")
            ax.set_ylabel("Frequency")
            plt.tight_layout()
            plt.show()
        else:
            print("Column first_observed_halfyear_views is not present.")
        """
    ),
    code(
        """
        if "first_observed_halfyear_hours" in modeling_df.columns:
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.hist(modeling_df["first_observed_halfyear_hours"].dropna(), bins=30, color="#54A24B", edgecolor="white")
            ax.set_title("Distribution of First Observed Half-Year Hours")
            ax.set_xlabel("first_observed_halfyear_hours")
            ax.set_ylabel("Frequency")
            plt.tight_layout()
            plt.show()
        else:
            print("Column first_observed_halfyear_hours is not present.")
        """
    ),
    code(
        """
        if "first_halfyear_hours_per_view" in modeling_df.columns:
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.hist(modeling_df["first_halfyear_hours_per_view"].dropna(), bins=30, color="#B279A2", edgecolor="white")
            ax.set_title("Distribution of First Half-Year Hours per View")
            ax.set_xlabel("first_halfyear_hours_per_view")
            ax.set_ylabel("Frequency")
            plt.tight_layout()
            plt.show()
        else:
            print("Column first_halfyear_hours_per_view is not present.")
        """
    ),
    code(
        """
        if "first_observed_halfyear_views" in modeling_df.columns:
            halfyear_box_df = modeling_df[[TARGET_COLUMN, "first_observed_halfyear_views"]].dropna().copy()
            halfyear_box_df["log_first_observed_halfyear_views"] = np.log1p(halfyear_box_df["first_observed_halfyear_views"])
            box_data = [
                halfyear_box_df.loc[halfyear_box_df[TARGET_COLUMN] == 0, "log_first_observed_halfyear_views"],
                halfyear_box_df.loc[halfyear_box_df[TARGET_COLUMN] == 1, "log_first_observed_halfyear_views"],
            ]
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.boxplot(box_data, tick_labels=["No Increase", "Increase"])
            ax.set_title("First Observed Half-Year Views vs Next-Season Outcome")
            ax.set_xlabel("Target Outcome")
            ax.set_ylabel("log1p(first_observed_halfyear_views)")
            plt.tight_layout()
            plt.show()
        else:
            print("Column first_observed_halfyear_views is not present.")
        """
    ),
    code(
        """
        if halfyear_period_col is not None:
            halfyear_period_summary_df = (
                modeling_df
                .groupby(halfyear_period_col)[TARGET_COLUMN]
                .agg(observation_count="count", increase_count="sum", increase_rate="mean")
                .reset_index()
                .sort_values(halfyear_period_col)
            )
            display(halfyear_period_summary_df)
        else:
            print("Column first_observed_halfyear_period is not present; skipping categorical half-year summary.")
        """
    ),
    code(
        """
        if "halfyear_period_summary_df" in globals():
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.bar(halfyear_period_summary_df[halfyear_period_col], halfyear_period_summary_df["increase_rate"] * 100, color="#72B7B2")
            ax.set_title("Viewership Increase Rate by First Observed Half-Year Period")
            ax.set_xlabel("First Observed Half-Year Period")
            ax.set_ylabel("Increase Rate (%)")
            ax.tick_params(axis="x", rotation=45)
            plt.tight_layout()
            plt.show()
        else:
            print("No half-year period summary is available for plotting.")
        """
    ),
    md("## 21. Categorical-Variable Statistical Tests"),
    code(
        """
        categorical_test_candidates = [
            col for col in categorical_predictors
            if predictor_df[col].nunique(dropna=True) > 1 and predictor_df[col].nunique(dropna=False) <= 20
        ]
        for binary_col in [
            "season_is_first", "season_is_later", "is_animation", "is_documentary",
            "is_kids_family_like", "has_prev_season_observation"
        ]:
            if binary_col in predictor_cols and binary_col not in categorical_test_candidates:
                categorical_test_candidates.append(binary_col)
        if halfyear_period_col is not None and halfyear_period_col not in categorical_test_candidates:
            categorical_test_candidates.append(halfyear_period_col)

        categorical_test_rows = []
        for col in sorted(set(categorical_test_candidates)):
            table = pd.crosstab(modeling_df[col], modeling_df[TARGET_COLUMN], dropna=False)
            if table.shape[0] < 2 or table.shape[1] < 2:
                continue
            chi2, p_value, _, _ = stats.chi2_contingency(table)
            categorical_test_rows.append({
                "variable": col,
                "chi_square": chi2,
                "p_value": p_value,
                "cramers_v": cramers_v_from_table(table),
            })

        categorical_test_df = pd.DataFrame(categorical_test_rows)
        if categorical_test_df.empty:
            categorical_test_df = pd.DataFrame(columns=["variable", "chi_square", "p_value", "cramers_v"])
        else:
            categorical_test_df = categorical_test_df.sort_values(["cramers_v", "chi_square"], ascending=[False, False]).reset_index(drop=True)
        display(categorical_test_df)
        """
    ),
    md(
        """
        Statistical significance should not be treated as a direct substitute for predictive usefulness. Large samples can make weak effects appear statistically significant, so effect size and model validation remain important.
        """
    ),
    md("## 22. Missingness Indicators vs Target"),
    code(
        """
        missing_indicator_cols = [col for col in predictor_cols if col.startswith("missing_")]
        missing_indicator_rows = []
        for col in missing_indicator_cols:
            for flag_value in [0, 1]:
                subset = modeling_df.loc[modeling_df[col] == flag_value, TARGET_COLUMN]
                missing_indicator_rows.append({
                    "missing_indicator": col,
                    "flag_value": flag_value,
                    "observation_count": int(subset.shape[0]),
                    "increase_rate": float(subset.mean()) if len(subset) else np.nan,
                })

        missing_indicator_summary_df = pd.DataFrame(missing_indicator_rows)
        display(missing_indicator_summary_df)
        """
    ),
    md("## 23. Multicollinearity / VIF"),
    code(
        """
        vif_candidate_cols = []
        excluded_prefixes = ("genre_", "missing_")
        for col in numeric_predictors:
            if col in target_cols or col in identifier_cols:
                continue
            if col.startswith(excluded_prefixes):
                continue
            if predictor_df[col].nunique(dropna=True) <= 1:
                continue
            vif_candidate_cols.append(col)

        vif_base_df = predictor_df[vif_candidate_cols].copy()
        vif_base_df = vif_base_df.loc[:, ~vif_base_df.T.duplicated()]
        vif_base_df = vif_base_df.fillna(vif_base_df.median(numeric_only=True))
        vif_base_df = vif_base_df.loc[:, vif_base_df.nunique(dropna=True) > 1]

        if vif_base_df.shape[1] < 2:
            print("Not enough stable numeric predictors were available for VIF analysis.")
            vif_df = pd.DataFrame(columns=["feature", "vif"])
        else:
            vif_rows = []
            X = vif_base_df.astype(float)
            for i, col in enumerate(X.columns):
                vif_rows.append({
                    "feature": col,
                    "vif": variance_inflation_factor(X.values, i),
                })
            vif_df = pd.DataFrame(vif_rows).sort_values("vif", ascending=False).reset_index(drop=True)
            display(vif_df.head(25))
        """
    ),
    md(
        """
        Approximate VIF interpretation:

        - `VIF < 5`: generally acceptable
        - `5 to 10`: investigate
        - `> 10`: strong multicollinearity concern

        VIF is a diagnostic, not an automatic deletion rule.
        """
    ),
    md("## 24. Candidate Feature-Quality Table"),
    code(
        """
        high_corr_feature_set = set(high_corr_pairs_df["feature_1"]).union(set(high_corr_pairs_df["feature_2"])) if "high_corr_pairs_df" in globals() and not high_corr_pairs_df.empty else set()
        feature_quality_rows = []

        corr_lookup = numeric_target_corr_df.set_index("feature") if "numeric_target_corr_df" in globals() else pd.DataFrame()
        skew_lookup = numeric_skew_df.set_index("feature") if "numeric_skew_df" in globals() else pd.DataFrame()
        constant_feature_set = set(constant_features_df["feature"]) if "constant_features_df" in globals() and not constant_features_df.empty else set()

        for col in predictor_cols:
            feature_quality_rows.append({
                "feature": col,
                "dtype": str(modeling_df[col].dtype),
                "missing_percent": float(modeling_df[col].isna().mean() * 100),
                "unique_values": int(modeling_df[col].nunique(dropna=True)),
                "skewness": float(skew_lookup.loc[col, "skewness"]) if col in skew_lookup.index else np.nan,
                "pearson_to_target": float(corr_lookup.loc[col, "pearson_correlation"]) if col in corr_lookup.index else np.nan,
                "spearman_to_target": float(corr_lookup.loc[col, "spearman_correlation"]) if col in corr_lookup.index else np.nan,
                "constant_flag": col in constant_feature_set,
                "high_correlation_flag": col in high_corr_feature_set,
            })

        feature_quality_df = pd.DataFrame(feature_quality_rows).sort_values(["high_correlation_flag", "missing_percent", "feature"], ascending=[False, False, True]).reset_index(drop=True)
        display(feature_quality_df)
        """
    ),
    md("## 25. Preliminary Feature Selection Observations"),
    code(
        """
        likely_exclude = sorted(set(identifier_cols + target_cols + list(constant_feature_set)))
        investigate_features = sorted(set(
            predictor_missing_df.loc[predictor_missing_df["missing_percent"] > 30, "feature"].tolist()
            + list(high_corr_feature_set)
        ))
        potentially_useful = numeric_target_corr_df.loc[numeric_target_corr_df["max_abs_correlation"] > 0.05, "feature"].head(15).tolist()

        summary_lines = ["# Preliminary Feature Selection Observations", ""]
        summary_lines.append("### Likely exclude")
        if likely_exclude:
            summary_lines.extend([f"- `{col}`" for col in likely_exclude])
        else:
            summary_lines.append("- No clear exclusions were detected from identifiers/targets/constants.")

        summary_lines.append("")
        summary_lines.append("### Investigate before modeling")
        if investigate_features:
            summary_lines.extend([f"- `{col}`" for col in investigate_features[:25]])
        else:
            summary_lines.append("- No major missingness or redundancy flags were detected.")

        if "netflix_hours_per_view" in modeling_df.columns:
            summary_lines.append("- `netflix_hours_per_view` should be checked for duplication of runtime-related information.")

        summary_lines.append("")
        summary_lines.append("### Potentially useful predictors")
        if potentially_useful:
            summary_lines.extend([f"- `{col}`" for col in potentially_useful])
        else:
            summary_lines.append("- No predictor exceeded the preliminary screening threshold used in this EDA.")

        summary_lines.append("")
        summary_lines.append("These are preliminary EDA-based observations only. Final feature selection should be validated using leakage-aware cross-validation and model performance rather than correlation alone.")

        display(Markdown("\\n".join(summary_lines)))
        """
    ),
    md("## 26. Leakage Review"),
    code(
        """
        forbidden_target_predictors = [col for col in TARGET_COLUMNS_KNOWN if col in predictor_cols]
        potential_leakage_flags = []

        for col in predictor_cols:
            if "next_season" in col or col.startswith("target_"):
                potential_leakage_flags.append(col)

        leakage_review = pd.DataFrame({
            "check": [
                "Known target columns accidentally present in predictor set",
                "Predictor names suggesting next-season or target leakage",
                "series_group_key available for group-aware splitting",
                "first-observed-half-year features present for review",
            ],
            "result": [
                ", ".join(forbidden_target_predictors) if forbidden_target_predictors else "None detected",
                ", ".join(sorted(set(potential_leakage_flags))) if potential_leakage_flags else "None detected",
                "Yes" if "series_group_key" in modeling_df.columns else "No",
                ", ".join([col for col in [
                    "first_observed_halfyear_period",
                    "first_observed_halfyear_views",
                    "first_observed_halfyear_hours",
                    "first_halfyear_hours_per_view",
                ] if col in modeling_df.columns]) or "None present",
            ],
        })
        display(leakage_review)

        display(Markdown(
            \"\"\"\
        The modeling row represents the **current season** and the target represents the **next season**. Predictor fields should therefore describe only the current season or earlier information. The target columns listed in the project specification should never be allowed into the predictor matrix. The first-observed-half-year fields should also be interpreted carefully to confirm they describe the current-season row rather than future behavior.
        \"\"\"
        ))
        """
    ),
    md("## 27. EDA Summary and Modeling Recommendations"),
    code(
        """
        target_positive_rate = modeling_df[TARGET_COLUMN].mean() * 100 if TARGET_COLUMN in modeling_df.columns else np.nan
        high_missing_features = predictor_missing_df.loc[predictor_missing_df["missing_percent"] > 30, "feature"].tolist()
        strong_skew_features = numeric_skew_df.loc[numeric_skew_df["abs_skewness"] > 1, "feature"].head(10).tolist()
        top_assoc_features = numeric_target_corr_df["feature"].head(10).tolist()
        top_vif_features = vif_df.head(10)["feature"].tolist() if "vif_df" in globals() and not vif_df.empty else []
        duplicate_like_features = sorted(constant_feature_set.union(high_corr_feature_set))
        genre_highlights = (
            genre_summary_df.dropna(subset=["increase_rate"]).head(5)[["genre_feature", "increase_rate", "observation_count"]]
            if "genre_summary_df" in globals() and not genre_summary_df.empty else None
        )
        season_highlights = (
            season_summary_df[["observation_count", "increase_rate"]].describe().transpose()
            if "season_summary_df" in globals() else None
        )
        halfyear_present = [col for col in [
            "first_observed_halfyear_period",
            "first_observed_halfyear_views",
            "first_observed_halfyear_hours",
            "first_halfyear_hours_per_view",
        ] if col in modeling_df.columns]

        summary_lines = [
            "# EDA Summary and Modeling Recommendations",
            "",
            f"1. Dataset size: {len(modeling_df):,} rows and {modeling_df.shape[1]:,} columns.",
            f"2. Number of predictors: {len(predictor_cols):,}.",
            f"3. Target class distribution: {target_counts.get(0, 0):,} 'No Increase' observations and {target_counts.get(1, 0):,} 'Increase' observations ({target_positive_rate:.2f}% positive class).",
            f"4. Main missing-data issues: {len(high_missing_features)} predictors exceed 30% missingness" + (f", including {', '.join(high_missing_features[:8])}." if high_missing_features else "."),
            f"5. Strongly skewed features: {', '.join(strong_skew_features[:8]) if strong_skew_features else 'No numeric predictors exceeded the strong-skew threshold used here.'}",
        ]

        raw_view_skew = numeric_skew_df.set_index("feature")["skewness"].to_dict() if "numeric_skew_df" in globals() else {}
        if "netflix_views" in raw_view_skew and "netflix_log_views" in raw_view_skew:
            summary_lines.append(
                f"6. Log transformations improve distributions for key viewership variables: `netflix_views` skew = {raw_view_skew['netflix_views']:.2f} versus `netflix_log_views` skew = {raw_view_skew['netflix_log_views']:.2f}."
            )
        else:
            summary_lines.append("6. Log transformations were reviewed where available and should generally reduce right-skew in raw engagement measures.")

        summary_lines.append(f"7. Strongest predictor-target associations from this screening include: {', '.join(top_assoc_features[:8]) if top_assoc_features else 'No numeric association table was available.'}")
        summary_lines.append(f"8. Major multicollinearity / redundancy findings include: {', '.join(top_vif_features[:8]) if top_vif_features else 'No stable VIF ranking was available.'}")

        if genre_highlights is not None and not genre_highlights.empty:
            genre_bits = [f\"{row.genre_feature} ({row.increase_rate:.1%}, n={int(row.observation_count)})\" for row in genre_highlights.itertuples()]
            summary_lines.append(f"9. Genre observations: {', '.join(genre_bits)}.")
        else:
            summary_lines.append("9. Genre observations: no qualifying genre indicators were available for summary.")

        if "season_summary_df" in globals():
            highest_season_rate = season_summary_df.sort_values("increase_rate", ascending=False).iloc[0]
            summary_lines.append(
                f"10. Season-number observations: the highest observed increase rate occurs at season {highest_season_rate[season_col]} with rate {highest_season_rate['increase_rate']:.1%} (n={int(highest_season_rate['observation_count'])})."
            )
        else:
            summary_lines.append("10. Season-number observations: season-order analysis was not available.")

        if halfyear_present:
            summary_lines.append(f"11. Half-year engagement feature observations: the dataset includes {', '.join(halfyear_present)}, and these were profiled for missingness, skewness, and target relationships.")
        else:
            summary_lines.append("11. Half-year engagement feature observations: the optional half-year features were not present.")

        summary_lines.append(f"12. Constant or duplicate-like predictors to remove or consolidate: {', '.join(duplicate_like_features[:12]) if duplicate_like_features else 'None detected by the constant/high-correlation screens.'}")
        summary_lines.append(f"13. Features requiring additional investigation: {', '.join(investigate_features[:12]) if investigate_features else 'No major feature-quality red flags were identified beyond routine modeling checks.'}")
        summary_lines.append("14. Recommended next steps before model training: finalize the predictor set, remove identifiers and targets, address missingness, reduce redundancy, preserve `series_group_key` for grouped splitting, and validate feature choices through leakage-aware cross-validation rather than EDA alone.")

        display(Markdown("\\n".join(summary_lines)))
        """
    ),
]


nb = nbf.v4.new_notebook()
nb["cells"] = cells
nb["metadata"]["kernelspec"] = {
    "display_name": "Python (.venv) - Netflix_Viewership_Group_9",
    "language": "python",
    "name": "netflix-viewership-group-9",
}
nb["metadata"]["language_info"] = {
    "name": "python",
    "version": "3.12",
}

NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
with NOTEBOOK_PATH.open("w", encoding="utf-8") as f:
    nbf.write(nb, f)

print(f"Wrote notebook to {NOTEBOOK_PATH}")
