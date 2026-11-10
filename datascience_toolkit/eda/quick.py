"""
quick.py
========
Exploratory data analysis for tabular datasets (EDAQuick).
Soporta tanto flujos no supervisados (target=None) como supervisados (target definido),
con formateo a 4 decimales, reportes de calidad, correlación, outliers y gráficos interactivos.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import gaussian_kde
from pandas.api.types import (
    is_bool_dtype,
    is_datetime64_any_dtype,
    is_numeric_dtype,
    is_object_dtype,
)

try:
    from IPython.display import display
except ImportError:
    display = print


@dataclass
class ColumnGroups:
    numeric: List[str]
    categorical: List[str]
    datetime: List[str]
    boolean: List[str]
    other: List[str]


class DFCore:
    """Núcleo base para inspección de tipos, agrupamiento de columnas y renderizado."""

    def __init__(
        self,
        df: pd.DataFrame,
        target: Optional[str] = None,
        features: Optional[List[str]] = None,
    ):
        if not isinstance(df, pd.DataFrame):
            raise TypeError("df debe ser un pandas.DataFrame")

        self.df = df.copy()
        self.target = target

        if features is None:
            self.features = [c for c in self.df.columns if c != target]
        else:
            missing = [c for c in features if c not in self.df.columns]
            if missing:
                raise ValueError(f"Estas columnas no existen en df: {missing}")
            self.features = features

        self.cols = self._infer_column_groups()

    def _infer_column_groups(self) -> ColumnGroups:
        df = self.df
        datetime_cols = [c for c in df.columns if is_datetime64_any_dtype(df[c])]
        bool_cols = [c for c in df.columns if is_bool_dtype(df[c])]
        numeric_cols = [
            c for c in df.columns
            if is_numeric_dtype(df[c]) and c not in datetime_cols and c not in bool_cols
        ]
        cat_cols = [
            c for c in df.columns
            if (is_object_dtype(df[c]) or isinstance(df[c].dtype, pd.CategoricalDtype))
            and c not in datetime_cols
            and c not in bool_cols
        ]
        used = set(datetime_cols + bool_cols + numeric_cols + cat_cols)
        other_cols = [c for c in df.columns if c not in used]

        return ColumnGroups(
            numeric=numeric_cols,
            categorical=cat_cols,
            datetime=datetime_cols,
            boolean=bool_cols,
            other=other_cols,
        )

    def objects_to_category(
        self, cols: Optional[List[str]] = None, exclude: Optional[List[str]] = None
    ) -> Dict[str, List[str]]:
        exclude = set(exclude or [])
        if cols is None:
            cols = self.df.select_dtypes(include=["object"]).columns.tolist()

        converted, skipped = [], []
        for c in cols:
            if c not in self.df.columns or c in exclude:
                skipped.append(c)
                continue
            if is_object_dtype(self.df[c]):
                self.df[c] = self.df[c].astype("category")
                converted.append(c)
            else:
                skipped.append(c)

        self.cols = self._infer_column_groups()
        if converted:
            print(f"Columnas convertidas a category: {converted}")
        return {"converted": converted, "skipped": skipped}

    def _in_notebook(self) -> bool:
        try:
            from IPython import get_ipython
            ip = get_ipython()
            return ip is not None and "IPKernelApp" in ip.config
        except Exception:
            return False

    def _display_df(
        self,
        df: pd.DataFrame,
        title: Optional[str] = None,
        head: Optional[int] = None,
        precision: int = 4,
        max_colwidth: int = 60,
    ):
        if df is None:
            return
        if not isinstance(df, pd.DataFrame):
            if title:
                print(title)
            print(df)
            return

        view = df.head(head) if head is not None else df
        if title:
            print(title)

        if self._in_notebook():
            try:
                sty = (
                    view.style.format(precision=precision)
                    .set_properties(**{"text-align": "left"})
                    .set_table_styles([{"selector": "th", "props": [("text-align", "left")]}])
                )
                display(sty)
                return
            except Exception:
                pass

        with pd.option_context(
            "display.max_rows", 200,
            "display.max_columns", 200,
            "display.width", 120,
            "display.max_colwidth", max_colwidth,
        ):
            print(view)

    @staticmethod
    def _style_strength(val: Any) -> str:
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return ""
        styles = {
            "Strong": "background-color: #d9f2d9; color: black; font-weight: 600;",
            "Moderate": "background-color: #fff2cc; color: black; font-weight: 600;",
            "Weak": "background-color: #ffe0cc; color: black; font-weight: 600;",
            "Negligible": "background-color: #eeeeee; color: black;",
            "Unknown": "background-color: #e6e0f8; color: black;",
        }
        return styles.get(str(val), "")

    def pretty_table(
        self,
        df: pd.DataFrame,
        title: Optional[str] = None,
        head: Optional[int] = None,
        precision: int = 4,
        highlight_strength_col: Optional[str] = "strength",
        max_colwidth: int = 60,
    ):
        if df is None or not isinstance(df, pd.DataFrame):
            return self._display_df(df, title=title, head=head, precision=precision)

        view = df.head(head) if head is not None else df
        if title:
            print(title)

        if self._in_notebook():
            try:
                sty = (
                    view.style.format(precision=precision)
                    .set_properties(**{"text-align": "left"})
                    .set_table_styles([{"selector": "th", "props": [("text-align", "left")]}])
                )
                if highlight_strength_col and highlight_strength_col in view.columns:
                    sty = sty.map(self._style_strength, subset=[highlight_strength_col])
                display(sty)
                return
            except Exception:
                pass

        print(view)


class EDAQuick(DFCore):
    """
    Inspección exploratoria rápida automatizada (EDA Quick).
    Compatible con flujos no supervisados (target opcional) y supervisados (target str).
    """

    def __init__(
        self,
        df: pd.DataFrame,
        target: Optional[str] = None,
        features: Optional[List[str]] = None,
    ):
        # Desacoplamos super() para inmunizar frente a %autoreload 2
        DFCore.__init__(self, df=df, target=target, features=features)

    def summary(self) -> pd.DataFrame:
        df = self.df
        out = pd.DataFrame({
            "dtype": df.dtypes.astype(str),
            "n_null": df.isna().sum(),
            "pct_null": (df.isna().mean() * 100).round(4),
            "n_unique": df.nunique(dropna=True),
        })
        out["pct_unique"] = (out["n_unique"] / len(df) * 100).round(4) if len(df) > 0 else 0.0
        return out.sort_values(["pct_null", "pct_unique"], ascending=False)

    def dataset_info(self) -> Dict[str, Any]:
        df = self.df
        return {
            "rows": df.shape[0],
            "cols": df.shape[1],
            "memory_mb": round(df.memory_usage(deep=True).sum() / 1024**2, 2),
            "duplicates_rows": int(df.duplicated().sum()),
            "target": self.target,
            "n_numeric": len(self.cols.numeric),
            "n_categorical": len(self.cols.categorical),
            "n_datetime": len(self.cols.datetime),
            "n_boolean": len(self.cols.boolean),
            "n_other": len(self.cols.other),
        }

    def describe_numeric(self) -> pd.DataFrame:
        if len(self.cols.numeric) == 0:
            return pd.DataFrame()
        return self.df[self.cols.numeric].describe().T

    def missing_report(self, include_complete: bool = False, sort: bool = True) -> pd.DataFrame:
        df = self.df
        out = pd.DataFrame({
            "dtype": df.dtypes.astype(str),
            "n_missing": df.isna().sum(),
            "pct_missing": (df.isna().mean() * 100).round(4),
        })
        out["n_present"] = len(df) - out["n_missing"]
        if not include_complete:
            out = out[out["n_missing"] > 0]
        if sort and not out.empty:
            out = out.sort_values(["pct_missing", "n_missing"], ascending=False)
        return out

    def cardinality_report(
        self,
        high_cardinality_threshold: int = 50,
        high_cardinality_ratio: float = 0.50,
        id_like_ratio: float = 0.95,
        sort: bool = True,
    ) -> pd.DataFrame:
        n = max(len(self.df), 1)
        rows = []
        for col in self.df.columns:
            n_unique = int(self.df[col].nunique(dropna=True))
            ratio = float(n_unique / n)
            is_cat = col in self.cols.categorical or col in self.cols.boolean
            rows.append({
                "feature": col,
                "dtype": str(self.df[col].dtype),
                "n_unique": n_unique,
                "pct_unique": round(ratio * 100, 4),
                "is_constant": n_unique <= 1,
                "is_high_cardinality": bool(
                    is_cat and (n_unique >= high_cardinality_threshold or ratio >= high_cardinality_ratio)
                ),
                "is_id_like": bool(n_unique > 1 and ratio >= id_like_ratio),
            })
        out = pd.DataFrame(rows)
        if sort and not out.empty:
            out = out.sort_values(
                ["is_id_like", "is_high_cardinality", "pct_unique"],
                ascending=[False, False, False],
            ).reset_index(drop=True)
        return out

    def outlier_report(
        self, cols: Optional[List[str]] = None, iqr_multiplier: float = 1.5
    ) -> pd.DataFrame:
        cols = cols or self.cols.numeric
        rows = []
        for col in cols:
            if col not in self.df.columns or not is_numeric_dtype(self.df[col]):
                continue
            s = self.df[col].dropna().astype(float)
            if s.empty:
                continue
            q1, q3 = s.quantile([0.25, 0.75])
            iqr = q3 - q1
            lower = q1 - iqr_multiplier * iqr
            upper = q3 + iqr_multiplier * iqr
            mask = (s < lower) | (s > upper)
            rows.append({
                "feature": col,
                "n": int(s.shape[0]),
                "n_outliers": int(mask.sum()),
                "pct_outliers": round(float(mask.mean() * 100), 4),
                "lower_bound": float(lower),
                "upper_bound": float(upper),
                "min": float(s.min()),
                "max": float(s.max()),
            })
        out = pd.DataFrame(rows)
        if not out.empty:
            out = out.sort_values(["pct_outliers", "n_outliers"], ascending=False).reset_index(drop=True)
        return out

    def skewness_report(
        self, cols: Optional[List[str]] = None, threshold: float = 1.0
    ) -> pd.DataFrame:
        cols = cols or self.cols.numeric
        rows = []
        for col in cols:
            if col not in self.df.columns or not is_numeric_dtype(self.df[col]):
                continue
            s = self.df[col].dropna().astype(float)
            if s.empty:
                continue
            skew = float(s.skew())
            abs_skew = abs(skew)
            severity = "low" if abs_skew < 0.5 else ("moderate" if abs_skew < 1.0 else "high")
            if abs_skew >= threshold:
                rec = "consider log1p / power transform; validate impact" if float(s.min()) >= 0 else "consider Yeo-Johnson / robust scaling; validate impact"
            else:
                rec = "none required from skewness alone"
            rows.append({
                "feature": col,
                "skewness": skew,
                "abs_skewness": abs_skew,
                "severity": severity,
                "flagged": bool(abs_skew >= threshold),
                "recommendation": rec,
            })
        out = pd.DataFrame(rows)
        if not out.empty:
            out = out.sort_values("abs_skewness", ascending=False).reset_index(drop=True)
        return out

    def correlation_report(
        self, method: str = "pearson", threshold: float = 0.80, show_plot: bool = True
    ) -> Dict[str, pd.DataFrame]:
        if len(self.cols.numeric) < 2:
            print("Se requieren al menos 2 columnas numéricas para correlación.")
            return {"matrix": pd.DataFrame(), "strong_pairs": pd.DataFrame()}

        corr = self.df[self.cols.numeric].corr(method=method)
        rows = []
        cols = corr.columns.tolist()
        for i, left in enumerate(cols):
            for right in cols[i + 1:]:
                val = corr.loc[left, right]
                if pd.notna(val) and abs(float(val)) >= threshold:
                    rows.append({
                        "feature_1": left,
                        "feature_2": right,
                        "correlation": float(val),
                        "abs_correlation": abs(float(val)),
                    })
        pairs = pd.DataFrame(rows)
        if not pairs.empty:
            pairs = pairs.sort_values("abs_correlation", ascending=False).reset_index(drop=True)

        if show_plot:
            self.correlation_plot(method=method)
        return {"matrix": corr, "strong_pairs": pairs}

    def correlation_plot(self, method: str = "pearson"):
        if len(self.cols.numeric) < 2:
            return None
        corr = self.df[self.cols.numeric].corr(method=method)
        fig = px.imshow(
            corr,
            text_auto=True,
            aspect="auto",
            title=f"Correlación ({method}) — variables numéricas",
        )
        fig.show()
        return corr

    def outliers_boxplot(self, cols: Optional[List[str]] = None, points: str = "outliers"):
        cols = cols or self.cols.numeric
        if len(cols) == 0:
            print("No hay columnas numéricas para boxplot.")
            return
        df_melt = self.df[cols].melt(var_name="variable", value_name="value")
        fig = px.box(df_melt, x="variable", y="value", points=points, title="Outliers — Boxplot (Plotly)")
        fig.show()

    def _imbalance_severity(self, ratio: float) -> str:
        if ratio < 1.5:
            return "Balanced"
        elif ratio < 3:
            return "Mild imbalance"
        elif ratio < 10:
            return "Moderate imbalance"
        return "Severe imbalance"

    def _compute_imbalance_info(self) -> Optional[Dict[str, Any]]:
        if self.target is None:
            return None
        s = self.df[self.target]
        is_cat = is_object_dtype(s) or isinstance(s.dtype, pd.CategoricalDtype) or is_bool_dtype(s)
        if not is_cat:
            return None
        counts = s.value_counts(dropna=False)
        if len(counts) <= 1:
            return {"counts": counts, "ratio": 1.0, "severity": "Balanced", "recommendations": []}
        ratio = float(counts.max() / counts.min())
        recs = []
        if ratio >= 1.5:
            recs.extend(["Prefer F1-macro / balanced accuracy over plain accuracy.", "Use stratified split / stratified CV."])
        if ratio >= 3:
            recs.append("Consider class_weight='balanced' (where applicable).")
        return {"counts": counts, "ratio": ratio, "severity": self._imbalance_severity(ratio), "recommendations": recs}

    def target_distribution(self):
        if self.target is None:
            print("No definiste target.")
            return None

        s = self.df[self.target]
        if is_object_dtype(s) or isinstance(s.dtype, pd.CategoricalDtype) or is_bool_dtype(s):
            tab = s.value_counts(dropna=False)
            pct = (tab / tab.sum() * 100).round(2)
            out = pd.DataFrame({"category": tab.index.astype(str), "count": tab.values, "pct": pct.values})
            imb = self._compute_imbalance_info()
            title = f"Distribución target: {self.target} | {imb['severity']} ({imb['ratio']:.2f}:1)" if imb and len(tab) > 1 else f"Distribución target: {self.target}"

            fig = go.Figure()
            for _, row in out.iterrows():
                fig.add_bar(
                    x=[row["pct"]],
                    y=[""],
                    orientation="h",
                    name=row["category"],
                    text=f'{row["pct"]:.1f}%<br>({row["count"]})',
                    textposition="inside",
                )
            fig.update_layout(barmode="stack", title=title, xaxis_title="Porcentaje (%)", yaxis_title="", showlegend=True, height=250)
            fig.update_xaxes(range=[0, 100])
            fig.show()
            return out.set_index("category")
        else:
            fig = px.histogram(self.df, x=self.target, nbins=30, title=f"Distribución target: {self.target}")
            fig.show()
            return s.describe()

    def imbalance_report(self) -> Optional[pd.DataFrame]:
        if self.target is None:
            return None
        imb = self._compute_imbalance_info()
        if imb is None:
            print("El target no es categórico (o no aplica imbalance).")
            return None

        counts = imb["counts"]
        pct = (counts / counts.sum() * 100).round(2)
        report = pd.DataFrame({"count": counts, "pct": pct})
        if len(counts) > 1:
            print("========== IMBALANCE REPORT ==========")
            print(f"Majority class : {counts.idxmax()}")
            print(f"Minority class : {counts.idxmin()}")
            print(f"Imbalance ratio: {imb['ratio']:.2f} : 1")
            print(f"Severity       : {imb['severity']}")
            if imb["recommendations"]:
                print("Recommendations:")
                for r in imb["recommendations"]:
                    print(f" - {r}")
            report["severity"] = imb["severity"]
            report["imbalance_ratio"] = round(imb["ratio"], 4)
        return report

    def predictive_power_categorical(
        self, var: str, normalize: str = "index", min_count: int = 10, sort_by: str = "count", show: bool = True
    ) -> pd.DataFrame:
        if self.target is None:
            raise ValueError("No definiste target.")
        if var not in self.df.columns:
            raise ValueError(f"'{var}' no existe en el DataFrame.")

        df = self.df[[var, self.target]].copy()
        df[var] = df[var].astype("object").fillna("__NA__")
        df[self.target] = df[self.target].astype("object").fillna("__NA__")

        vc = df[var].value_counts(dropna=False)
        rare = vc[vc < min_count].index
        if len(rare) > 0:
            df[var] = df[var].where(~df[var].isin(rare), "__OTHER__")

        ct_counts = pd.crosstab(df[var], df[self.target], dropna=False)
        ct_pct = ct_counts.div(ct_counts.sum(axis=1), axis=0) if normalize == "index" else ct_counts / ct_counts.values.sum()

        order = ct_counts.sum(axis=1).sort_values(ascending=False).index if sort_by == "count" else sorted(ct_counts.index, key=str)
        ct_counts = ct_counts.loc[order]
        ct_pct = ct_pct.loc[order]

        out = (ct_pct * 100).round(2)
        out["count_total"] = ct_counts.sum(axis=1)
        out = out.rename_axis(var).reset_index()

        if show:
            long_df = ct_pct.reset_index().melt(id_vars=[var], var_name="target_class", value_name="pct")
            long_df["pct"] = long_df["pct"] * 100
            fig = px.bar(
                long_df, y=var, x="pct", color="target_class", orientation="h",
                text=long_df["pct"].map(lambda x: f"{x:.1f}%"),
                title=f"Poder predictivo (categórica): {var} vs {self.target}",
            )
            fig.update_layout(barmode="stack", xaxis_title="Porcentaje (%)", yaxis_title=var)
            fig.update_xaxes(range=[0, 100])
            fig.show()
        return out

    def predictive_power_numeric(
        self, var: str, show: bool = True, grid_size: int = 256, clip_quantiles: tuple = (0.01, 0.99), fill_opacity: float = 0.35
    ) -> pd.DataFrame:
        if self.target is None:
            raise ValueError("No definiste target.")
        if var not in self.df.columns or not is_numeric_dtype(self.df[var]):
            return pd.DataFrame()

        df = self.df[[var, self.target]].dropna(subset=[var, self.target]).copy()
        stats = df.groupby(self.target, observed=True)[var].describe()
        if not show:
            return stats

        lo, hi = df[var].quantile(list(clip_quantiles)).values
        if lo == hi:
            lo, hi = df[var].min(), df[var].max()
            if lo == hi:
                return stats

        xgrid = np.linspace(lo, hi, grid_size)
        fig = go.Figure()
        for cls, sub in df.groupby(self.target, observed=True):
            x = sub[var].astype(float).values
            if len(x) < 2 or np.std(x) == 0:
                continue
            kde = gaussian_kde(x)
            y = kde(xgrid)
            fig.add_trace(go.Scatter(x=xgrid, y=y, mode="lines", name=str(cls)))
            fig.add_trace(go.Scatter(x=xgrid, y=y, mode="lines", name=str(cls), fill="tozeroy", opacity=fill_opacity, showlegend=False, hoverinfo="skip"))

        fig.update_layout(title=f"Densidad (KDE): {var} por clase de {self.target}", xaxis_title=var, yaxis_title="Densidad", height=400)
        fig.show()
        return stats

    def predictive_power_all(
        self,
        categorical_vars: Optional[List[str]] = None,
        numeric_vars: Optional[List[str]] = None,
        exclude: Optional[List[str]] = None,
        max_categorical: int = 12,
        max_numeric: int = 8,
        min_count_cat: int = 10,
        sort_by_cat: str = "count",
        show_plots: bool = True,
        **kwargs,
    ) -> Dict[str, Any]:
        if self.target is None:
            raise ValueError("No definiste target.")

        exclude = set(exclude or [])
        exclude.add(self.target)
        cat_vars = [c for c in (categorical_vars or self.cols.categorical) if c not in exclude][:max_categorical]
        num_vars = [c for c in (numeric_vars or self.cols.numeric) if c not in exclude][:max_numeric]

        results = {"categorical": {}, "numeric": {}, "skipped": {"categorical": [], "numeric": []}}
        for var in cat_vars:
            try:
                results["categorical"][var] = self.predictive_power_categorical(var=var, min_count=min_count_cat, sort_by=sort_by_cat, show=show_plots)
            except Exception as e:
                results["skipped"]["categorical"].append((var, str(e)))

        for var in num_vars:
            try:
                results["numeric"][var] = self.predictive_power_numeric(var=var, show=show_plots)
            except Exception as e:
                results["skipped"]["numeric"].append((var, str(e)))

        return results

    def run_eda(
        self,
        do_objects_to_category: bool = True,
        show_summary: bool = True,
        show_target: bool = True,
        show_imbalance: bool = True,
        show_corr: bool = True,
        show_outliers: bool = True,
        corr_method: str = "pearson",
        outlier_points: str = "outliers",
        summary_head: int = 30,
        show_predictive_power: bool = False,
        use_power_all: bool = True,
        max_categorical: int = 12,
        max_numeric: int = 8,
        min_count_cat: int = 10,
        sort_by_cat: str = "count",
        show_power_plots: bool = True,
        show_info: bool = True,
        show_describe: bool = True,
        show_missing: bool = False,
        show_cardinality: bool = False,
        show_skewness: bool = False,
        corr_threshold: float = 0.80,
        skewness_threshold: float = 1.0,
        high_cardinality_threshold: int = 50,
        high_cardinality_ratio: float = 0.50,
        id_like_ratio: float = 0.95,
        iqr_multiplier: float = 1.5,
        **kwargs,
    ) -> Dict[str, Any]:
        """Ejecuta un flujo modular y completo de diagnóstico exploratorio."""
        results: Dict[str, Any] = {}

        if do_objects_to_category:
            results["category_conversion"] = self.objects_to_category()

        if show_info:
            info = self.dataset_info()
            results["dataset_info"] = info
            print("========== DATASET INFO ==========")
            for k, v in info.items():
                print(f"{k}: {v}")

        if show_summary:
            print("\n========== SUMMARY (top) ==========")
            summ = self.summary()
            results["summary"] = summ
            self._display_df(summ, head=summary_head)

        if show_missing:
            print("\n========== MISSING VALUES ==========")
            missing = self.missing_report(include_complete=False)
            results["missing_report"] = missing
            if missing.empty:
                print("No se detectaron valores faltantes.")
            else:
                self._display_df(missing)

        if show_cardinality:
            print("\n========== CARDINALITY ==========")
            cardinality = self.cardinality_report(
                high_cardinality_threshold=high_cardinality_threshold,
                high_cardinality_ratio=high_cardinality_ratio,
                id_like_ratio=id_like_ratio,
            )
            results["cardinality_report"] = cardinality
            self._display_df(cardinality)

        if show_describe:
            desc = self.describe_numeric()
            results["describe_numeric"] = desc
            if not desc.empty:
                print("\n========== DESCRIBE (numeric) ==========")
                self._display_df(desc)

        if self.target is not None and show_target:
            print("\n========== TARGET DISTRIBUTION ==========")
            results["target_distribution"] = self.target_distribution()
            if show_imbalance:
                results["imbalance_report"] = self.imbalance_report()

        if show_corr:
            print("\n========== CORRELATION ==========")
            corr_out = self.correlation_report(method=corr_method, threshold=corr_threshold, show_plot=True)
            results["correlation"] = corr_out["matrix"]
            results["strong_correlations"] = corr_out["strong_pairs"]
            if corr_out["strong_pairs"].empty:
                print(f"No se detectaron pares con |correlación| >= {corr_threshold:.2f}.")
            else:
                print(f"\nPares con |correlación| >= {corr_threshold:.2f}")
                self._display_df(corr_out["strong_pairs"])

        if show_outliers:
            print("\n========== OUTLIERS ==========")
            outlier_rep = self.outlier_report(iqr_multiplier=iqr_multiplier)
            results["outlier_report"] = outlier_rep
            if outlier_rep.empty:
                print("No hay variables numéricas disponibles para el reporte.")
            else:
                self._display_df(outlier_rep)
            self.outliers_boxplot(points=outlier_points)

        if show_skewness:
            print("\n========== SKEWNESS ==========")
            skew = self.skewness_report(threshold=skewness_threshold)
            results["skewness_report"] = skew
            if not skew.empty:
                self._display_df(skew)

        if show_predictive_power and self.target is not None:
            print("\n========== PREDICTIVE POWER ==========")
            if use_power_all:
                results["predictive_power"] = self.predictive_power_all(
                    max_categorical=max_categorical,
                    max_numeric=max_numeric,
                    min_count_cat=min_count_cat,
                    sort_by_cat=sort_by_cat,
                    show_plots=show_power_plots,
                )

        return results


def edaquick(
    df: pd.DataFrame,
    target: Optional[str] = None,
    corr_threshold: float = 0.80,
    auto_categorical: bool = True,
    **kwargs,
) -> Dict[str, Any]:
    """Helper funcional para ejecutar EDAQuick directamente."""
    eda = EDAQuick(df=df, target=target)
    return eda.run_eda(corr_threshold=corr_threshold, do_objects_to_category=auto_categorical, **kwargs)