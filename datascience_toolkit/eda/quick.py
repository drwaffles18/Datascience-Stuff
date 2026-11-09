"""
quick.py
========
Exploratory data analysis for tabular datasets.
Includes health metrics, missingness, cardinality, outlier detection,
and automated interactive Plotly correlation and distribution charts.
"""

from typing import Dict, List, Optional, Union, Any
import numpy as np
import pandas as pd
import plotly.express as px

try:
    from IPython.display import display
except ImportError:
    display = print


def edaquick(
    df: pd.DataFrame,
    target: Optional[str] = None,
    corr_threshold: float = 0.80,
    auto_categorical: bool = True
) -> pd.DataFrame:
    """
    Inspección exploratoria rápida automatizada (EDA Quick).
    Restaura el formato original con separadores formales, tablas estadísticas a 4 decimales
    y visualizaciones interactivas de correlación y outliers en Plotly.
    """
    df_analyzed = df.copy()

    # 0. Conversión automática de columnas object/string a category
    if auto_categorical:
        converted_cols = []
        for col in df_analyzed.select_dtypes(include=["object", "string"]).columns:
            df_analyzed[col] = df_analyzed[col].astype("category")
            converted_cols.append(col)
        if converted_cols:
            print(f"Columnas convertidas a category: {converted_cols}")

    # 1. DATASET INFO
    print("========== DATASET INFO ==========")
    n_rows, n_cols = df_analyzed.shape
    mem_mb = df_analyzed.memory_usage(deep=True).sum() / (1024 * 1024)
    dup_rows = int(df_analyzed.duplicated().sum())

    n_numeric = len(df_analyzed.select_dtypes(include=[np.number]).columns)
    n_categorical = len(df_analyzed.select_dtypes(include=["category"]).columns)
    n_datetime = len(df_analyzed.select_dtypes(include=["datetime"]).columns)
    n_boolean = len(df_analyzed.select_dtypes(include=["bool", "boolean"]).columns)
    n_other = n_cols - (n_numeric + n_categorical + n_datetime + n_boolean)

    print(f"rows: {n_rows}")
    print(f"cols: {n_cols}")
    print(f"memory_mb: {mem_mb:.2f}")
    print(f"duplicates_rows: {dup_rows}")
    print(f"target: {target}")
    print(f"n_numeric: {n_numeric}")
    print(f"n_categorical: {n_categorical}")
    print(f"n_datetime: {n_datetime}")
    print(f"n_boolean: {n_boolean}")
    print(f"n_other: {n_other}")

    # 2. SUMMARY (top)
    print("\n========== SUMMARY (top) ==========")
    summary_rows = []
    for col in df_analyzed.columns:
        n_null = int(df_analyzed[col].isnull().sum())
        pct_null = (n_null / n_rows) * 100 if n_rows > 0 else 0.0
        n_unique = int(df_analyzed[col].nunique())
        pct_unique = (n_unique / n_rows) * 100 if n_rows > 0 else 0.0
        summary_rows.append({
            "feature": col,
            "dtype": str(df_analyzed[col].dtype),
            "n_null": n_null,
            "pct_null": round(pct_null, 4),
            "n_unique": n_unique,
            "pct_unique": round(pct_unique, 4),
        })
    summary_df = pd.DataFrame(summary_rows).set_index("feature")
    summary_df = summary_df.sort_values(by="n_unique", ascending=False)
    display(summary_df)

    # 3. MISSING VALUES
    print("\n========== MISSING VALUES ==========")
    null_counts = df_analyzed.isnull().sum()
    if null_counts.sum() == 0:
        print("No se detectaron valores faltantes.")
    else:
        missing_df = pd.DataFrame({
            "feature": null_counts[null_counts > 0].index,
            "n_null": null_counts[null_counts > 0].values,
            "pct_null": ((null_counts[null_counts > 0].values / n_rows) * 100).round(4)
        }).sort_values(by="n_null", ascending=False).reset_index(drop=True)
        display(missing_df)

    # 4. CARDINALITY
    print("\n========== CARDINALITY ==========")
    card_rows = []
    for col in df_analyzed.columns:
        n_unique = int(df_analyzed[col].nunique())
        pct_unique = round((n_unique / n_rows) * 100, 4) if n_rows > 0 else 0.0
        is_cat = str(df_analyzed[col].dtype) in ("category", "object")
        is_constant = (n_unique <= 1)
        is_high_cardinality = is_cat and (n_unique > 50)
        is_id_like = (n_unique == n_rows) and not np.issubdtype(df_analyzed[col].dtype, np.floating)
        card_rows.append({
            "feature": col,
            "dtype": str(df_analyzed[col].dtype),
            "n_unique": n_unique,
            "pct_unique": pct_unique,
            "is_constant": is_constant,
            "is_high_cardinality": is_high_cardinality,
            "is_id_like": is_id_like
        })
    card_df = pd.DataFrame(card_rows).sort_values(by="n_unique", ascending=False).reset_index(drop=True)
    display(card_df)

    # 5. DESCRIBE (numeric)
    print("\n========== DESCRIBE (numeric) ==========")
    num_df = df_analyzed.select_dtypes(include=[np.number])
    if not num_df.empty:
        desc = num_df.describe().T.round(4)
        display(desc)

    # 6. CORRELATION
    print("\n========== CORRELATION ==========")
    if not num_df.empty and num_df.shape[1] > 1:
        corr = num_df.corr(method="pearson")

        # Matriz interactiva en Plotly con paleta plasma
        fig_corr = px.imshow(
            corr,
            text_auto=True,
            color_continuous_scale="plasma",
            aspect="auto",
            title="Correlación (pearson) — variables numéricas"
        )
        fig_corr.update_layout(
            margin=dict(l=40, r=40, t=50, b=40),
            height=380
        )
        fig_corr.show()

        # Detección de pares con correlación fuerte
        high_corr = []
        cols = corr.columns.tolist()
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                val = corr.iloc[i, j]
                if abs(val) >= corr_threshold:
                    high_corr.append((cols[i], cols[j], val))

        if not high_corr:
            print(f"No se detectaron pares con |correlación| >= {corr_threshold:.2f}.")
        else:
            print(f"Pares con |correlación| >= {corr_threshold:.2f} ({len(high_corr)}):")
            for col1, col2, val in high_corr:
                print(f"- {col1} y {col2}: {val:.4f}")

    # 7. OUTLIERS
    print("\n========== OUTLIERS ==========")
    if not num_df.empty:
        outlier_rows = []
        for col in num_df.columns:
            series = num_df[col].dropna()
            q1 = series.quantile(0.25)
            q3 = series.quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            outliers = series[(series < lower_bound) | (series > upper_bound)]
            n_outliers = int(len(outliers))
            pct_outliers = (n_outliers / len(series)) * 100 if len(series) > 0 else 0.0

            outlier_rows.append({
                "feature": col,
                "n": len(series),
                "n_outliers": n_outliers,
                "pct_outliers": round(pct_outliers, 4),
                "lower_bound": round(lower_bound, 4),
                "upper_bound": round(upper_bound, 4),
                "min": round(float(series.min()), 4),
                "max": round(float(series.max()), 4)
            })
        outlier_df = pd.DataFrame(outlier_rows).sort_values(by="n_outliers", ascending=False).reset_index(drop=True)
        display(outlier_df)

        # Boxplot interactivo en Plotly
        fig_box = px.box(
            num_df.melt(),
            x="variable",
            y="value",
            title="Outliers — Boxplot (Plotly)"
        )
        fig_box.update_layout(
            margin=dict(l=40, r=40, t=50, b=40),
            height=400
        )
        fig_box.show()

    return df_analyzed


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
            n_missing = int(series.isna().sum())
            pct_missing = (n_missing / total_rows) * 100 if total_rows > 0 else 0.0
            n_unique = int(series.nunique(dropna=True))
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
            pct_outliers = round((n_outliers / len(s)) * 100, 2) if len(s) > 0 else 0.0

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
        target_col: Optional[str] = None,
        corr_threshold: float = 0.80,
        auto_categorical: bool = True,
        **kwargs
    ) -> pd.DataFrame:
        """Orchestrates an automated exploratory diagnostic session with original format and Plotly charts."""
        return edaquick(
            df=self.df,
            target=target_col,
            corr_threshold=corr_threshold,
            auto_categorical=auto_categorical
        )

    def __call__(self, *args, **kwargs) -> pd.DataFrame:
        return self.run_eda(*args, **kwargs)

    def skewness_report(self, threshold: float = 0.5) -> pd.DataFrame:
        """Evaluates skewness across numeric features and suggests normalization transforms."""
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