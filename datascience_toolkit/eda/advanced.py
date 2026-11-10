"""
advanced.py
===========
Análisis exploratorio avanzado: rankings de poder predictivo,
Cramér's V, Chi-cuadrado, Mutual Information y Feature Importance con Random Forest.
"""

from typing import Any, Dict, List, Optional, Union
import numpy as np
import pandas as pd
import plotly.express as px

from pandas.api.types import (
    is_bool_dtype,
    is_numeric_dtype,
    is_object_dtype,
)

from sklearn.feature_selection import chi2, f_classif, mutual_info_classif
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

from .quick import EDAQuick


class EDAAdvanced(EDAQuick):
    """
    EDA avanzado orientado a 'separabilidad' / poder predictivo.
    Diseñado para target categórico (clasificación).
    """

    # ----------------------------
    # Helpers
    # ----------------------------
    def _is_target_categorical(self) -> bool:
        if self.target is None:
            return False
        s = self.df[self.target]
        return (
            is_object_dtype(s)
            or isinstance(s.dtype, pd.CategoricalDtype)
            or is_bool_dtype(s)
        )

    def _require_target(self):
        if self.target is None:
            raise ValueError("No definiste target.")

    def _require_target_categorical(self):
        self._require_target()
        if not self._is_target_categorical():
            raise TypeError("EDAAdvanced (por ahora) requiere target categórico (clasificación).")

    def _cramers_v(self, confusion: pd.DataFrame) -> float:
        obs = confusion.values
        n = obs.sum()
        if n == 0:
            return np.nan

        row_sum = obs.sum(axis=1, keepdims=True)
        col_sum = obs.sum(axis=0, keepdims=True)
        expected = row_sum @ col_sum / n
        expected = np.where(expected == 0, 1e-12, expected)

        chi2_stat = ((obs - expected) ** 2 / expected).sum()

        r, k = obs.shape
        denom = n * (min(k - 1, r - 1))
        if denom == 0:
            return np.nan

        return float(np.sqrt(chi2_stat / denom))

    # ----------------------------
    # 1) Ranking numéricas vs target
    # ----------------------------
    def rank_numeric_features(
        self,
        features: Optional[List[str]] = None,
        top_n: int = 20,
        random_state: Optional[int] = None,
        verbose: bool = True
    ) -> pd.DataFrame:

        self._require_target_categorical()

        if features is None:
            features = self.cols.numeric.copy()

        features = [c for c in features if c in self.df.columns and is_numeric_dtype(self.df[c])]
        if len(features) == 0:
            return pd.DataFrame()

        X = self.df[features].copy()
        y = self.df[self.target].astype(str).values

        X = X.fillna(X.median(numeric_only=True))

        F, p = f_classif(X.values, y)
        mi = mutual_info_classif(X.values, y, random_state=random_state)

        out = pd.DataFrame({
            "feature": features,
            "anova_F": F,
            "anova_p": p,
            "mutual_info": mi,
        }).sort_values(["mutual_info", "anova_F"], ascending=False)

        mi_min = float(out["mutual_info"].min())
        mi_max = float(out["mutual_info"].max())
        out["mi_score"] = (out["mutual_info"] - mi_min) / (mi_max - mi_min + 1e-12)

        def _interpret_strength(score: float) -> str:
            if score >= 0.8:
                return "Strong"
            elif score >= 0.5:
                return "Moderate"
            elif score >= 0.2:
                return "Weak"
            else:
                return "Negligible"

        out["strength"] = out["mi_score"].apply(_interpret_strength)
        out = out[["feature", "strength", "mi_score", "mutual_info", "anova_F", "anova_p"]]

        if verbose:
            self.pretty_table(
                out,
                title="========== NUMERIC FEATURE RANKING (interpretable) ==========",
                head=top_n,
                precision=4
            )

        return out

    # ----------------------------
    # 2) Ranking categóricas vs target
    # ----------------------------
    def rank_categorical_features(
        self,
        features: Optional[List[str]] = None,
        top_n: int = 20,
        min_count: int = 10,
        verbose: bool = True
    ) -> pd.DataFrame:

        self._require_target_categorical()

        if features is None:
            features = [c for c in self.cols.categorical if c != self.target]

        features = [c for c in features if c in self.df.columns]
        if len(features) == 0:
            return pd.DataFrame()

        df = self.df[features + [self.target]].copy()
        df[self.target] = df[self.target].astype(str)

        for c in features:
            df[c] = df[c].astype("object").fillna("__NA__")
            vc = df[c].value_counts(dropna=False)
            rare = vc[vc < min_count].index
            if len(rare) > 0:
                df[c] = df[c].where(~df[c].isin(rare), "__OTHER__")

        cv_list = []
        for c in features:
            ct = pd.crosstab(df[c], df[self.target], dropna=False)
            cv_list.append((c, self._cramers_v(ct)))
        cv_df = pd.DataFrame(cv_list, columns=["feature", "cramers_v"])

        y = pd.Categorical(df[self.target]).codes

        chi2_rows = []
        for c in features:
            Xc = df[[c]]
            enc = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
            Xohe = enc.fit_transform(Xc)

            chi2_stat, pvals = chi2(Xohe, y)

            chi2_rows.append({
                "feature": c,
                "chi2_max": float(np.nanmax(chi2_stat)) if len(chi2_stat) else np.nan,
                "chi2_mean": float(np.nanmean(chi2_stat)) if len(chi2_stat) else np.nan,
                "chi2_min_p": float(np.nanmin(pvals)) if len(pvals) else np.nan,
                "n_levels": int(Xc[c].nunique(dropna=False))
            })

        chi2_df = pd.DataFrame(chi2_rows)
        out = chi2_df.merge(cv_df, on="feature", how="left")
        out = out.sort_values(["chi2_max", "cramers_v"], ascending=False)

        def _interpret_cramers(v: float) -> str:
            if pd.isna(v):
                return "Unknown"
            if v >= 0.3:
                return "Strong"
            elif v >= 0.1:
                return "Moderate"
            elif v >= 0.05:
                return "Weak"
            else:
                return "Negligible"

        out["strength"] = out["cramers_v"].apply(_interpret_cramers)
        out = out[["feature", "strength", "cramers_v", "chi2_max", "chi2_mean", "chi2_min_p", "n_levels"]]

        if verbose:
            self.pretty_table(
                out,
                title="========== CATEGORICAL FEATURE RANKING (interpretable) ==========",
                head=top_n,
                precision=4
            )

        return out

    # ----------------------------
    # 3) Plot helper — Feature Importance
    # ----------------------------
    def plot_feature_importance(
        self,
        importance_df: pd.DataFrame,
        top_n: int = 25,
        title: str = "Random Forest — Feature Importance",
        show: bool = True
    ):
        if importance_df is None or importance_df.empty:
            return None

        d = importance_df.head(top_n).copy()
        d = d.sort_values("importance", ascending=True)

        fig = px.bar(
            d,
            x="importance",
            y="feature",
            orientation="h",
            title=title
        )
        fig.update_layout(
            yaxis_title="",
            xaxis_title="Importance",
            height=max(450, 18 * len(d))
        )

        if show:
            fig.show()
        return fig

    # ----------------------------
    # 4) Feature importance con Random Forest
    # ----------------------------
    def rf_feature_importance(
        self,
        features: Optional[List[str]] = None,
        test_size: float = 0.2,
        n_estimators: int = 300,
        max_depth: Optional[int] = None,
        random_state: Optional[int] = None,
        n_jobs: int = -1,
        top_n: int = 25,
        show_plot: bool = True,
        verbose: bool = True
    ) -> pd.DataFrame:

        self._require_target_categorical()

        if features is None:
            features = [c for c in self.features if c != self.target]

        features = [c for c in features if c in self.df.columns]
        if len(features) == 0:
            return pd.DataFrame()

        X = self.df[features].copy()
        y = self.df[self.target].astype(str)

        num_cols = [c for c in X.columns if is_numeric_dtype(X[c])]
        cat_cols = [
            c for c in X.columns
            if (is_object_dtype(X[c]) or isinstance(X[c].dtype, pd.CategoricalDtype) or is_bool_dtype(X[c]))
        ]

        for c in num_cols:
            X[c] = X[c].fillna(X[c].median())
        for c in cat_cols:
            X[c] = X[c].astype("object").fillna("__NA__")

        pre = ColumnTransformer(
            transformers=[
                ("num", "passthrough", num_cols),
                ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
            ],
            remainder="drop"
        )

        clf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            n_jobs=n_jobs,
            random_state=random_state
        )

        pipe = Pipeline(steps=[("pre", pre), ("rf", clf)])

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, stratify=y, random_state=random_state
        )

        pipe.fit(X_train, y_train)

        feature_names = []
        feature_names.extend(num_cols)

        if len(cat_cols) > 0:
            ohe = pipe.named_steps["pre"].named_transformers_["cat"]
            ohe_names = ohe.get_feature_names_out(cat_cols).tolist()
            feature_names.extend(ohe_names)

        importances = pipe.named_steps["rf"].feature_importances_
        out = pd.DataFrame({"feature": feature_names, "importance": importances})
        out = out.sort_values("importance", ascending=False)

        if verbose:
            print("========== RANDOM FOREST FEATURE IMPORTANCE ==========")

        if show_plot:
            self.plot_feature_importance(
                out,
                top_n=top_n,
                title=f"Random Forest — Feature Importance (top {top_n})",
                show=True
            )

        return out

    # ----------------------------
    # 5) Runner EDA avanzado
    # ----------------------------
    def run_eda_advanced(
        self,
        do_numeric_rank: bool = True,
        do_categorical_rank: bool = True,
        do_rf_importance: bool = True,
        top_n: int = 20,
        rf_n_estimators: int = 300,
        rf_n_jobs: int = -1,
        random_state: Optional[int] = None,
        return_results: bool = False,
        show_outputs_summary: bool = True
    ) -> Optional[Dict[str, Any]]:

        self._require_target_categorical()

        results: Dict[str, Any] = {}

        if do_numeric_rank:
            results["rank_numeric"] = self.rank_numeric_features(
                top_n=top_n,
                random_state=random_state,
                verbose=True
            )

        if do_categorical_rank:
            results["rank_categorical"] = self.rank_categorical_features(
                top_n=top_n,
                verbose=True
            )

        if do_rf_importance:
            results["rf_importance"] = self.rf_feature_importance(
                n_estimators=rf_n_estimators,
                n_jobs=rf_n_jobs,
                random_state=random_state,
                top_n=top_n,
                show_plot=True,
                verbose=True
            )

        self.last_results = results

        if show_outputs_summary:
            summary = pd.DataFrame({
                "artifact": list(results.keys()),
                "rows": [results[k].shape[0] if isinstance(results[k], pd.DataFrame) else None for k in results],
                "cols": [results[k].shape[1] if isinstance(results[k], pd.DataFrame) else None for k in results],
            })
            self.pretty_table(summary, title="========== EDA ADVANCED — OUTPUTS ==========", head=None)

        if return_results:
            return results

        return None