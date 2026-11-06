"""
quick.py
========
Exploratory data analysis for tabular datasets.
Includes health metrics, missingness, cardinality, outlier detection,
and automated correlation heatmaps.
"""

from typing import Dict, List, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

try:
    from IPython.display import display
except ImportError:
    display = print


class EDAQuick:
    """Rapid tabular dataset inspector offering automated EDA routines and summary diagnostics."""

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()

    def overview(self) -> pd.DataFrame:
        """Generates a column health card covering dtypes, missing values, and unique counts."""
        total_rows = len(self.df)
        summary = []

        for col in self.df.columns:
            series = self.df[col]
            n_missing = series.isna().sum()
            pct_missing = (n_missing / total_rows) * 100 if total_rows > 0 else 0
            n_unique = series.nunique(dropna=True)
            dtype = str(series.dtype)
            sample_vals = series.dropna().unique()[:3].tolist()

            summary.append({
                "column": col,
                "dtype": dtype,
                "missing_count": n_missing,
                "missing_pct": round(pct_missing, 2),
                "unique_count": n_unique,
                "sample_values": sample_vals
            })

        return pd.DataFrame(summary).sort_values(by="missing_pct", ascending=False).reset_index(drop=True)

    def numeric_summary(self) -> pd.DataFrame:
        """Computes statistical metrics for numeric columns."""
        num_cols = self.df.select_dtypes(include=[np.number]).columns
        if len(num_cols) == 0:
            return pd.DataFrame()

        stats = []
        for col in num_cols:
            s = self.df[col].dropna()
            if len(s) == 0:
                continue

            q25, q75 = s.quantile(0.25), s.quantile(0.75)
            stats.append({
                "column": col,
                "mean": round(float(s.mean()), 3),
                "std": round(float(s.std()), 3),
                "median": round(float(s.median()), 3),
                "min": round(float(s.min()), 3),
                "q25": round(float(q25), 3),
                "q75": round(float(q75), 3),
                "max": round(float(s.max()), 3),
                "iqr": round(float(q75 - q25), 3),
                "skew": round(float(s.skew()), 3),
                "kurtosis": round(float(s.kurtosis()), 3)
            })

        return pd.DataFrame(stats).reset_index(drop=True)

    def categorical_summary(self, max_categories: int = 10) -> Dict[str, pd.DataFrame]:
        """Returns frequency distributions for non-numeric columns."""
        cat_cols = self.df.select_dtypes(exclude=[np.number]).columns
        breakdowns = {}

        for col in cat_cols:
            val_counts = self.df[col].value_counts(dropna=False).head(max_categories)
            pcts = self.df[col].value_counts(dropna=False, normalize=True).head(max_categories) * 100
            breakdowns[col] = pd.DataFrame({
                "count": val_counts,
                "percentage": pcts.round(2)
            })

        return breakdowns

    def detect_outliers(self, factor: float = 1.5) -> pd.DataFrame:
        """Identifies outliers using the Interquartile Range (IQR) rule."""
        num_cols = self.df.select_dtypes(include=[np.number]).columns
        records = []

        for col in num_cols:
            s = self.df[col].dropna()
            if len(s) == 0:
                continue
            q25, q75 = s.quantile(0.25), s.quantile(0.75)
            iqr = q75 - q25
            lower_bound = q25 - factor * iqr
            upper_bound = q75 + factor * iqr
            n_outliers = int(((s < lower_bound) | (s > upper_bound)).sum())
            pct_outliers = round((n_outliers / len(s)) * 100, 2) if len(s) > 0 else 0

            records.append({
                "column": col,
                "outlier_count": n_outliers,
                "outlier_pct": pct_outliers,
                "lower_bound": round(float(lower_bound), 3),
                "upper_bound": round(float(upper_bound), 3)
            })

        return pd.DataFrame(records).sort_values(by="outlier_count", ascending=False).reset_index(drop=True)

    def run_eda(
        self,
        show_corr: bool = True,
        show_outliers: bool = True,
        show_missing: bool = True,
        show_cardinality: bool = True,
        show_target: bool = False,
        show_predictive_power: bool = False,
        target_col: Optional[str] = None
    ) -> None:
        """Orchestrates an automated exploratory diagnostic session."""
        health_card = self.overview()

        if show_missing:
            print("\n--- [1] MISSING VALUES REPORT ---")
            missing_subset = health_card[health_card["missing_count"] > 0].reset_index(drop=True)
            if not missing_subset.empty:
                display(missing_subset[["column", "missing_count", "missing_pct"]])
            else:
                print("✓ No missing values detected in dataset.")

        if show_cardinality:
            print("\n--- [2] CARDINALITY & DATA TYPES ---")
            display(health_card[["column", "dtype", "unique_count", "sample_values"]])

        if show_outliers:
            print("\n--- [3] OUTLIER DETECTION (IQR Rule 1.5x) ---")
            outliers_df = self.detect_outliers()
            display(outliers_df)

        if show_corr:
            num_df = self.df.select_dtypes(include=[np.number])
            if num_df.shape[1] > 1:
                print("\n--- [4] CORRELATION MATRIX ---")
                corr = num_df.corr()
                plt.figure(figsize=(10, 7))
                sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", cbar=True, square=True)
                plt.title("Numerical Feature Correlation Matrix", fontsize=12)
                plt.tight_layout()
                plt.show()
            else:
            
                print("✓ Not enough numerical columns for correlation analysis.")

    def skewness_report(self, threshold: float = 0.5) -> pd.DataFrame:
        """
        Evaluates skewness across numeric features and suggests normalization transforms.
        
        Interpretation rules:
        - |skew| <= 0.5: Approximately symmetric.
        - 0.5 < |skew| <= 1.0: Moderately skewed.
        - |skew| > 1.0: Highly skewed.
        """
        num_cols = self.df.select_dtypes(include=[np.number]).columns
        if len(num_cols) == 0:
            return pd.DataFrame()

        records = []
        for col in num_cols:
            s = self.df[col].dropna()
            if len(s) == 0:
                continue

            skew_val = float(s.skew())
            abs_skew = abs(skew_val)

            if abs_skew <= threshold:
                status = "Symmetric"
                recommendation = "None"
            elif threshold < abs_skew <= 1.0:
                status = "Moderately Skewed"
                recommendation = "Sqrt / Yeo-Johnson" if (s <= 0).any() else "Log1p / Sqrt"
            else:
                status = "Highly Skewed"
                recommendation = "Yeo-Johnson" if (s <= 0).any() else "Log / Box-Cox"

            records.append({
                "column": col,
                "skewness": round(skew_val, 3),
                "status": status,
                "recommended_transform": recommendation
            })

        return (
            pd.DataFrame(records)
            .sort_values(by="skewness", key=abs, ascending=False)
            .reset_index(drop=True)
        )
