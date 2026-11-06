"""
quick.py
========
Fast exploratory data analysis for tabular datasets.
Provides summary metrics on missingness, cardinality, data types, and numeric distributions.
"""

from typing import Dict, List, Optional, Union
import numpy as np
import pandas as pd


class EDAQuick:
    """Rapid tabular dataset inspector offering column-level health and distribution diagnostics."""

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()

    def overview(self) -> pd.DataFrame:
        """Generates a column-by-column health card covering dtypes, missing values, and unique counts."""
        total_rows = len(self.df)
        summary = []

        for col in self.df.columns:
            series = self.df[col]
            n_missing = series.isna().sum()
            pct_missing = (n_missing / total_rows) * 100 if total_rows > 0 else 0
            n_unique = series.nunique(dropna=True)
            dtype = str(series.dtype)

            # Sample non-null values for quick context
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
        """Computes statistical metrics (mean, std, skewness, kurtosis, IQR) for numeric columns."""
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
                "mean": s.mean(),
                "std": s.std(),
                "median": s.median(),
                "min": s.min(),
                "q25": q25,
                "q75": q75,
                "max": s.max(),
                "iqr": q75 - q25,
                "skew": s.skew(),
                "kurtosis": s.kurtosis()
            })

        return pd.DataFrame(stats).reset_index(drop=True)

    def categorical_summary(self, max_categories: int = 10) -> Dict[str, pd.DataFrame]:
        """Returns frequency distributions for non-numeric columns."""
        cat_cols = self.df.select_dtypes(exclude=[np.number]).columns
        breakdowns = {}

        for col in cat_cols:
            val_counts = self.df[col].value_counts(dropna=False).head(max_categories)
            pcts = self.df[col].value_counts(dropna=False, normalize=True).head(max_categories) * 100
            breakdown_df = pd.DataFrame({
                "count": val_counts,
                "percentage": pcts.round(2)
            })
            breakdowns[col] = breakdown_df

        return breakdowns