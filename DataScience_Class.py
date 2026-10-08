# ============================================================
# DataScience Vic — MASTER CLASS v22 (auto HPO + SHAP interpretability)
# Core + EDA + Preprocess + Unsupervised + Supervised (Cls/Reg)
# - Ordenado, sin duplicados
# - Incluye HPO opcional + DL opcional + interpretabilidad SHAP/LIME + SHAP opcional
# ============================================================

from __future__ import annotations

# ============================================================
# (0) Standard library
# ============================================================
import os
import time
import json
import pickle
import warnings
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
import importlib.util
import sys


# ============================================================
# (1) Core scientific stack
# ============================================================
import numpy as np
import pandas as pd

# ============================================================
# (2) Parallelism / persistence
# ============================================================
import joblib
from joblib import Parallel, delayed

# ============================================================
# (3) Plotting and visualization
# ============================================================
import plotly.express as px
import plotly.graph_objects as go

import matplotlib

def _setup_matplotlib_backend_for_notebook():
    try:
        from IPython import get_ipython
        ip = get_ipython()
        if ip is None:
            return
        if "IPKernelApp" in ip.config:  # notebook/jupyter
            matplotlib.use("module://matplotlib_inline.backend_inline")
    except Exception:
        pass

_setup_matplotlib_backend_for_notebook()

import matplotlib.pyplot as plt

# ============================================================
# (4) Statistics / signals
# ============================================================
from scipy.stats import gaussian_kde
from scipy import signal

# ============================================================
# (5) Pandas dtype helpers
# ============================================================
from pandas.api.types import (
    is_bool_dtype,
    is_datetime64_any_dtype,
    is_numeric_dtype,
    is_object_dtype,
)

# ============================================================
# (6) Scikit-learn: base / pipelines / preprocessing
# ============================================================
from sklearn.base import clone, BaseEstimator, RegressorMixin, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (
    OneHotEncoder,
    OrdinalEncoder,
    LabelEncoder,
    StandardScaler,
    MinMaxScaler,
    RobustScaler,
)

# ============================================================
# (7) Scikit-learn: model selection / CV / search
# ============================================================
from sklearn.model_selection import (
    train_test_split,
    KFold,
    StratifiedKFold,
    cross_val_score,
    cross_validate,
    GridSearchCV,
    RandomizedSearchCV,
)

# Halving Search (optional)
try:
    from sklearn.experimental import enable_halving_search_cv  # noqa: F401
    from sklearn.model_selection import HalvingRandomSearchCV
    _HAS_HALVING = True
except Exception:
    _HAS_HALVING = False

# BayesSearch (optional)
try:
    from skopt import BayesSearchCV
    _HAS_SKOPT = True
except Exception:
    _HAS_SKOPT = False

# ============================================================
# (8) Scikit-learn: feature scoring / reduction / clustering
# ============================================================
from sklearn.feature_selection import chi2, f_classif, mutual_info_classif
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors
from sklearn.ensemble import RandomForestClassifier

# ============================================================
# (9) Scikit-learn: metrics
# ============================================================
from sklearn.metrics import (
    # Unsupervised
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    adjusted_rand_score,
    normalized_mutual_info_score,
    homogeneity_score,
    completeness_score,
    v_measure_score,
    # Classification
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
    log_loss,
    classification_report,
    # Regression
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    make_scorer,
)

# ============================================================
# (10) Supervised models — Classification
# ============================================================
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn.discriminant_analysis import (
    QuadraticDiscriminantAnalysis,
    LinearDiscriminantAnalysis,
)
from sklearn.naive_bayes import GaussianNB

# XGBoost (optional)
try:
    from xgboost import XGBClassifier
    _HAS_XGB = True
except Exception:
    _HAS_XGB = False

try:
    import xgboost as xgb #xgboost para series de tiempo
    _HAS_XGB = True
except Exception:
    _HAS_XGB = False

# SHAP (optional): model interpretability
try:
    import shap
    _HAS_SHAP = True
except Exception:
    shap = None
    _HAS_SHAP = False

# LIME (optional): local model-agnostic interpretability
try:
    from lime.lime_tabular import LimeTabularExplainer
    _HAS_LIME = True
except Exception:
    LimeTabularExplainer = None
    _HAS_LIME = False

# ============================================================
# (11) Supervised models — Regression
# ============================================================
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

# ============================================================
# (12) Deep Learning (optional): TensorFlow / Keras + Keras Tuner
# ============================================================
_HAS_TF = False
_HAS_KT = False

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import regularizers
    from tensorflow.keras import layers 
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Input, Dense, LSTM, Conv1D, GlobalAveragePooling1D
    _HAS_TF = True
except Exception:
    _HAS_TF = False

try:
    import keras_tuner as kt
    _HAS_KT = True
except Exception:
    _HAS_KT = False

# ============================================================
# (13) Time Series (statsmodels)
# (Opcional en master: si TODO vive en 1 solo archivo)
# ============================================================
from statsmodels.tsa.api import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX
try:
    from prophet import Prophet
    _HAS_PROPHET = True
except Exception:
    _HAS_PROPHET = False

# ============================================================
# (13) Utilities (framework-level)
# ============================================================
def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def _rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))

# sklearn scoring: "higher is better" => usamos negativo para que minimice RMSE
RMSE_SCORER = make_scorer(
    lambda yt, yp: -_rmse(yt, yp),
    greater_is_better=True
)

# ============================================================
# (X) Optional: joblib for parallel SARIMA tuning
# ============================================================
try:
    from joblib import Parallel, delayed
    _HAS_JOBLIB = True
except Exception:
    _HAS_JOBLIB = False
    Parallel = None
    delayed = None

# ============================================================
# Custom transformers
# ============================================================

class FrequencyEncoder(BaseEstimator, TransformerMixin):
    """Frequency encoding para variables categóricas nominales.

    Cada categoría se sustituye por su frecuencia relativa observada durante
    ``fit``. Categorías desconocidas durante ``transform`` reciben
    ``unknown_value`` (0.0 por defecto).
    """

    def __init__(self, unknown_value: float = 0.0):
        self.unknown_value = float(unknown_value)
        self.frequency_maps_: Dict[str, Dict[Any, float]] = {}
        self.feature_names_in_: Optional[np.ndarray] = None

    def fit(self, X, y=None):
        Xdf = pd.DataFrame(X).copy()
        if hasattr(X, "columns"):
            Xdf.columns = list(X.columns)
        else:
            Xdf.columns = [f"x{i}" for i in range(Xdf.shape[1])]
        self.feature_names_in_ = np.asarray(Xdf.columns, dtype=object)
        self.frequency_maps_ = {}
        for c in Xdf.columns:
            values = Xdf[c].astype("object").where(Xdf[c].notna(), "__NA__")
            self.frequency_maps_[str(c)] = values.value_counts(normalize=True, dropna=False).to_dict()
        return self

    def transform(self, X):
        Xdf = pd.DataFrame(X).copy()
        if self.feature_names_in_ is None:
            raise RuntimeError("FrequencyEncoder must be fitted before transform().")
        Xdf.columns = list(self.feature_names_in_)
        out = pd.DataFrame(index=Xdf.index)
        for c in Xdf.columns:
            values = Xdf[c].astype("object").where(Xdf[c].notna(), "__NA__")
            fmap = self.frequency_maps_[str(c)]
            out[str(c)] = values.map(fmap).fillna(self.unknown_value).astype(float)
        return out.to_numpy(dtype=float)

    def get_feature_names_out(self, input_features=None):
        if input_features is not None:
            return np.asarray(input_features, dtype=object)
        if self.feature_names_in_ is None:
            raise RuntimeError("FrequencyEncoder has not been fitted.")
        return self.feature_names_in_


# ============================================================
# Clase DataScience Vic
# ============================================================

@dataclass
class ColumnGroups:
    numeric: List[str]
    categorical: List[str]
    datetime: List[str]
    boolean: List[str]
    other: List[str]

class DFCore:
    """
    Núcleo mínimo:
    - almacena df (copia)
    - detecta grupos de columnas
    - utilidades comunes para el resto de módulos
    """
    def __init__(self, df: pd.DataFrame, target: Optional[str] = None, features: Optional[List[str]] = None):
        if not isinstance(df, pd.DataFrame):
            raise TypeError("df debe ser un pandas.DataFrame")

        self.df = df.copy()
        self.target = target

        if features is None:
            self.features = [c for c in self.df.columns if c != target]
        else:
            # Validación básica
            missing = [c for c in features if c not in self.df.columns]
            if missing:
                raise ValueError(f"Estas columnas no existen en df: {missing}")
            self.features = features

        self.cols = self._infer_column_groups()

    def _infer_column_groups(self) -> ColumnGroups:
        df = self.df

        datetime_cols = [c for c in df.columns if is_datetime64_any_dtype(df[c])]
        bool_cols = [c for c in df.columns if is_bool_dtype(df[c])]

        numeric_cols = [c for c in df.columns if is_numeric_dtype(df[c]) and c not in datetime_cols and c not in bool_cols]
        cat_cols = cat_cols = [
                c for c in df.columns
                if (
                    is_object_dtype(df[c])
                    or isinstance(df[c].dtype, pd.CategoricalDtype)
                )
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
            other=other_cols
        )

    def objects_to_category(self, cols: Optional[List[str]] = None, exclude: Optional[List[str]] = None):
        """
        Convierte columnas object a category.
        Modifica self.df y actualiza grupos de columnas.
        """
        exclude = set(exclude or [])

        if cols is None:
            cols = self.df.select_dtypes(include=["object"]).columns.tolist()

        converted = []
        skipped = []

        for c in cols:
            if c not in self.df.columns:
                skipped.append(c)
                continue
            if c in exclude:
                skipped.append(c)
                continue

            if is_object_dtype(self.df[c]):
                self.df[c] = self.df[c].astype("category")
                converted.append(c)
            else:
                skipped.append(c)

        self.cols = self._infer_column_groups()
        print(f"Columnas convertidas a category: {converted}")

        return {"converted": converted, "skipped": skipped}

    def cast_datetime(self, cols: List[str], errors: str = "coerce", format: Optional[str] = None, dayfirst: bool = False):
        """
        Convierte columnas a datetime.
        """
        missing = [c for c in cols if c not in self.df.columns]
        if missing:
            raise ValueError(f"Estas columnas no existen en df: {missing}")

        for c in cols:
            self.df[c] = pd.to_datetime(self.df[c], errors=errors, format=format, dayfirst=dayfirst)

        self.cols = self._infer_column_groups()

    def _in_notebook(self) -> bool:
        """Detecta si estamos en un entorno tipo Jupyter."""
        try:
            from IPython import get_ipython
            ip = get_ipython()
            if ip is None:
                return False
            # kernel típico de notebook/lab
            return "IPKernelApp" in ip.config
        except Exception:
            return False

    def _display_df(
        self,
        df: pd.DataFrame,
        title: Optional[str] = None,
        head: Optional[int] = None,
        precision: int = 4,
        max_colwidth: int = 60
    ):
        """
        Muestra un DataFrame de forma bonita en Jupyter (Styler),
        con fallback a print en consola.

        - title: imprime encabezado
        - head: si se pasa, muestra solo top N filas
        """
        if df is None:
            return

        if not isinstance(df, pd.DataFrame):
            # fallback simple
            if title:
                print(title)
            print(df)
            return

        view = df.head(head) if head is not None else df

        if title:
            print(title)

        # --- Notebook: usar Styler ---
        if self._in_notebook():
            try:
                from IPython.display import display
                sty = (
                    view.style
                        .format(precision=precision)
                        .set_properties(**{"text-align": "left"})
                        .set_table_styles([
                            {"selector": "th", "props": [("text-align", "left")]},
                        ])
                )
                # Nota: max_colwidth depende de pandas version; si falla, lo ignoramos
                try:
                    pd.set_option("display.max_colwidth", max_colwidth)
                except Exception:
                    pass

                display(sty)
                return
            except Exception:
                # si algo falla, caemos al print
                pass

        # --- Consola: print friendly ---
        with pd.option_context(
            "display.max_rows", 200,
            "display.max_columns", 200,
            "display.width", 120,
            "display.max_colwidth", max_colwidth
        ):
            print(view)    

    @staticmethod
    def _style_strength(val: Any) -> str:
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return ""

        v = str(val)

        styles = {
            "Strong":     "background-color: #d9f2d9; color: black; font-weight: 600;",
            "Moderate":   "background-color: #fff2cc; color: black; font-weight: 600;",
            "Weak":       "background-color: #ffe0cc; color: black; font-weight: 600;",
            "Negligible": "background-color: #eeeeee; color: black;",
            "Unknown":    "background-color: #e6e0f8; color: black;"
        }

        return styles.get(v, "")

    @staticmethod
    def _style_status(val: Any) -> str:
        """
        Colorea status tipo dependency doctor.
        - OK: verde con letras negras (bold)
        - MISSING: rojo con letras blancas (bold)
        - MISSING_FLAG: naranja con letras negras (bold)
        - WARN: amarillo con letras negras (bold) (opcional)
        """
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return ""

        v = str(val).strip().upper()

        styles = {
            "OK":           "background-color: #b7e1cd; color: black; font-weight: 700;",
            "MISSING":      "background-color: #c00000; color: white; font-weight: 700;",
            "MISSING_FLAG": "background-color: #ff8c00; color: black; font-weight: 700;",
            "WARN":         "background-color: #ffd966; color: black; font-weight: 700;",
        }
        return styles.get(v, "")
    
    @staticmethod
    def pretty_table_static(
        df: pd.DataFrame,
        title: Optional[str] = None,
        head: Optional[int] = None,
        precision: int = 4,
        max_colwidth: int = 60,
        style_funcs: Optional[Dict[str, Any]] = None,  # {"col": func(val)->css}
    ):
        """
        Render bonito SIN necesitar instancia DFCore.
        style_funcs: dict columna -> función estilo (CSS string)
        """
        if df is None:
            return None

        # fallback si no es DataFrame
        if not isinstance(df, pd.DataFrame):
            if title:
                print(title)
            print(df)
            return None

        view = df.head(head) if head is not None else df

        if title:
            print(title)

        # ¿Notebook?
        try:
            from IPython import get_ipython
            ip = get_ipython()
            in_nb = (ip is not None) and ("IPKernelApp" in ip.config)
        except Exception:
            in_nb = False

        if in_nb:
            try:
                from IPython.display import display
                sty = (
                    view.style
                        .format(precision=precision)
                        .set_properties(**{"text-align": "left"})
                        .set_table_styles([{"selector": "th", "props": [("text-align", "left")]}])
                )

                if style_funcs:
                    for col, fn in style_funcs.items():
                        if col in view.columns and callable(fn):
                            sty = sty.map(fn, subset=[col])

                try:
                    pd.set_option("display.max_colwidth", max_colwidth)
                except Exception:
                    pass

                display(sty)
                # Método puramente visual: evitar que Jupyter vuelva a
                # representar el mismo Styler como salida de la celda.
                return None
            except Exception:
                pass

        # consola fallback
        with pd.option_context(
            "display.max_rows", 200,
            "display.max_columns", 200,
            "display.width", 120,
            "display.max_colwidth", max_colwidth
        ):
            print(view)
        return None

    def _has_pkg(pkg_name: str) -> bool:
        """True si el paquete es importable (sin importarlo)."""
        try:
            return importlib.util.find_spec(pkg_name) is not None
        except Exception:
            return False

    @classmethod
    def dependency_doctor(cls, verbose: bool = True, return_df: bool = False):
        """
        Revisa dependencias por capabilities del framework y reporta faltantes.
        - No revienta si faltan; solo reporta.
        - Usa find_spec (ligero) + banderas _HAS_* cuando existan.
        """

        def _has_pkg(pkg: str) -> bool:
            try:
                return importlib.util.find_spec(pkg) is not None
            except Exception:
                return False

        checks = {
            "core": ["numpy", "pandas"],
            "viz": ["plotly", "matplotlib"],
            "sklearn_core": ["sklearn"],
            "hpo_halving": ["sklearn"],
            "hpo_bayes_skopt": ["skopt"],
            "xgboost": ["xgboost"],
            "model_interpretability_shap": ["shap"],
            "model_interpretability_lime": ["lime"],
            "deeplearning_tf": ["tensorflow"],
            "deeplearning_kt": ["keras_tuner"],
            "timeseries_statsmodels": ["statsmodels"],
            "timeseries_prophet": ["prophet"],
        }

        rows = []
        for cap, pkgs in checks.items():
            missing = [p for p in pkgs if not _has_pkg(p)]
            rows.append({
                "capability": cap,
                "status": "OK" if not missing else "MISSING",
                "missing": ", ".join(missing) if missing else ""
            })

        df = pd.DataFrame(rows)

        # banderas internas (si existen)
        flag_map = {
            "_HAS_HALVING": "hpo_halving",
            "_HAS_SKOPT": "hpo_bayes_skopt",
            "_HAS_XGB": "xgboost",
            "_HAS_SHAP": "model_interpretability_shap",
            "_HAS_LIME": "model_interpretability_lime",
            "_HAS_TF": "deeplearning_tf",
            "_HAS_KT": "deeplearning_kt",
            "_HAS_PROPHET": "timeseries_prophet",
        }
        g = globals()
        for flag, cap in flag_map.items():
            if flag in g and (not bool(g[flag])):
                df.loc[df["capability"] == cap, "status"] = "MISSING_FLAG"
                cur = df.loc[df["capability"] == cap, "missing"].iloc[0]
                if cur == "":
                    df.loc[df["capability"] == cap, "missing"] = f"(flag {flag}=False)"

        # ordenar (faltantes arriba)
        order_map = {"MISSING": 0, "MISSING_FLAG": 1, "OK": 2}
        df["_ord"] = df["status"].map(order_map).fillna(9).astype(int)
        df = (
            df[["capability", "status", "missing", "_ord"]]
            .sort_values(["_ord", "capability"])
            .drop(columns=["_ord"])
            .reset_index(drop=True)
        )

        if verbose:
            missing_df = df[df["status"] != "OK"]

            if missing_df.empty:
                print("✅ Dependency doctor: todo OK (para los módulos chequeados).")
                cls.pretty_table_static(
                    df,
                    title="Dependency doctor (OK)",
                    precision=4,
                    max_colwidth=80,
                    style_funcs={"status": lambda v: cls(pd.DataFrame({"_":[0]}))._style_status(v)}  # ver nota abajo
                )
            else:
                print("⚠️ Dependency doctor: faltantes detectados:")
                cls.pretty_table_static(
                    missing_df,
                    title="Dependency doctor (FALTANTES)",
                    precision=4,
                    max_colwidth=80,
                    style_funcs={"status": lambda v: cls(pd.DataFrame({"_":[0]}))._style_status(v)}  # ver nota abajo
                )

                pkgs_missing = sorted({
                    x.strip()
                    for m in missing_df["missing"].tolist()
                    for x in m.split(",")
                    if x.strip() and not x.strip().startswith("(")
                })
                if pkgs_missing:
                    print("\nSugerencia (pip):")
                    print("pip install " + " ".join(pkgs_missing))

        return df if return_df else None

    def pretty_table(
        self,
        df: pd.DataFrame,
        title: Optional[str] = None,
        head: Optional[int] = None,
        precision: int = 4,
        highlight_strength_col: Optional[str] = "strength",
        highlight_status_col: Optional[str] = None,
        max_colwidth: int = 60
    ):
        """
        API principal (NO romper código viejo).
        Internamente delega a pretty_table_static.
        """

        # construir style_funcs según columnas
        style_funcs = {}

        # strength
        if highlight_strength_col is not None and isinstance(df, pd.DataFrame) and highlight_strength_col in df.columns:
            style_funcs[highlight_strength_col] = DFCore._style_strength

        # status
        status_col = highlight_status_col
        if status_col is None and isinstance(df, pd.DataFrame) and "status" in df.columns:
            status_col = "status"
        if status_col is not None and isinstance(df, pd.DataFrame) and status_col in df.columns:
            style_funcs[status_col] = DFCore._style_status

        return DFCore.pretty_table_static(
            df=df,
            title=title,
            head=head,
            precision=precision,
            max_colwidth=max_colwidth,
            style_funcs=style_funcs if style_funcs else None
        )

# ============================================================
# EDA Rápida + Target Analysis + Imbalance Report + Predictive Power
# ============================================================

class EDAQuick(DFCore):
    """
    EDA rápido: shape, missing, duplicates, cardinalidad, describe numérico,
    correlación y outliers (plotly).
    """

    def summary(self) -> pd.DataFrame:
        df = self.df
        out = pd.DataFrame({
            "dtype": df.dtypes.astype(str),
            "n_null": df.isna().sum(),
            "pct_null": (df.isna().mean() * 100).round(2),
            "n_unique": df.nunique(dropna=True),
        })
        out["pct_unique"] = (out["n_unique"] / len(df) * 100).round(2)
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

    def missing_report(
        self,
        include_complete: bool = False,
        sort: bool = True
    ) -> pd.DataFrame:
        """Reporte independiente de valores faltantes por columna."""
        df = self.df
        out = pd.DataFrame({
            "dtype": df.dtypes.astype(str),
            "n_missing": df.isna().sum(),
            "pct_missing": (df.isna().mean() * 100).round(2),
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
        sort: bool = True
    ) -> pd.DataFrame:
        """
        Reporte de cardinalidad y posibles identificadores.

        Las banderas son diagnósticas: no eliminan columnas automáticamente.
        """
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
                "pct_unique": round(ratio * 100, 2),
                "is_constant": n_unique <= 1,
                "is_high_cardinality": bool(
                    is_cat and (
                        n_unique >= high_cardinality_threshold
                        or ratio >= high_cardinality_ratio
                    )
                ),
                "is_id_like": bool(n_unique > 1 and ratio >= id_like_ratio),
            })

        out = pd.DataFrame(rows)
        if sort and not out.empty:
            out = out.sort_values(
                ["is_id_like", "is_high_cardinality", "pct_unique"],
                ascending=[False, False, False]
            ).reset_index(drop=True)
        return out

    def outlier_report(
        self,
        cols: Optional[List[str]] = None,
        iqr_multiplier: float = 1.5
    ) -> pd.DataFrame:
        """Detecta outliers univariados mediante la regla del IQR."""
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
                "pct_outliers": round(float(mask.mean() * 100), 2),
                "lower_bound": float(lower),
                "upper_bound": float(upper),
                "min": float(s.min()),
                "max": float(s.max()),
            })
        out = pd.DataFrame(rows)
        if not out.empty:
            out = out.sort_values(
                ["pct_outliers", "n_outliers"], ascending=False
            ).reset_index(drop=True)
        return out

    def skewness_report(
        self,
        cols: Optional[List[str]] = None,
        threshold: float = 1.0
    ) -> pd.DataFrame:
        """Reporte de asimetría para variables numéricas."""
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
            if abs_skew < 0.5:
                severity = "low"
            elif abs_skew < 1.0:
                severity = "moderate"
            else:
                severity = "high"

            if abs_skew >= threshold:
                if float(s.min()) >= 0:
                    recommendation = "consider log1p / power transform; validate impact"
                else:
                    recommendation = "consider Yeo-Johnson / robust scaling; validate impact"
            else:
                recommendation = "none required from skewness alone"

            rows.append({
                "feature": col,
                "skewness": skew,
                "abs_skewness": abs_skew,
                "severity": severity,
                "flagged": bool(abs_skew >= threshold),
                "recommendation": recommendation,
            })
        out = pd.DataFrame(rows)
        if not out.empty:
            out = out.sort_values("abs_skewness", ascending=False).reset_index(drop=True)
        return out

    def correlation_report(
        self,
        method: str = "pearson",
        threshold: float = 0.80,
        show_plot: bool = True
    ) -> Dict[str, pd.DataFrame]:
        """Devuelve matriz y pares de correlación que superan |threshold|."""
        if len(self.cols.numeric) < 2:
            print("Se requieren al menos 2 columnas numéricas para correlación.")
            return {"matrix": pd.DataFrame(), "strong_pairs": pd.DataFrame()}

        corr = self.df[self.cols.numeric].corr(method=method)
        rows = []
        cols = corr.columns.tolist()
        for i, left in enumerate(cols):
            for right in cols[i + 1:]:
                value = corr.loc[left, right]
                if pd.notna(value) and abs(float(value)) >= threshold:
                    rows.append({
                        "feature_1": left,
                        "feature_2": right,
                        "correlation": float(value),
                        "abs_correlation": abs(float(value)),
                    })
        pairs = pd.DataFrame(rows)
        if not pairs.empty:
            pairs = pairs.sort_values("abs_correlation", ascending=False).reset_index(drop=True)

        if show_plot:
            fig = px.imshow(
                corr,
                text_auto=True,
                aspect="auto",
                title=f"Correlación ({method}) — variables numéricas"
            )
            fig.show()

        return {"matrix": corr, "strong_pairs": pairs}

    def correlation_plot(self, method: str = "pearson"):
        """
        Heatmap de correlación para numéricas (Plotly).
        """
        if len(self.cols.numeric) < 2:
            print("Se requieren al menos 2 columnas numéricas para correlación.")
            return None

        corr = self.df[self.cols.numeric].corr(method=method)
        fig = px.imshow(
            corr,
            text_auto=True,
            aspect="auto",
            title=f"Correlación ({method}) — variables numéricas"
        )
        fig.show()
        return corr

    def outliers_boxplot(self, cols: Optional[List[str]] = None, points: str = "outliers"):
        """
        Boxplot interactivo Plotly.
        points: 'outliers' | 'all' | False
        """
        if cols is None:
            cols = self.cols.numeric

        if len(cols) == 0:
            print("No hay columnas numéricas para boxplot.")
            return

        df_melt = self.df[cols].melt(var_name="variable", value_name="value")
        fig = px.box(df_melt, x="variable", y="value", points=points, title="Outliers — Boxplot (Plotly)")
        fig.show()

    def _imbalance_severity(self, ratio: float) -> str:
        # Regla práctica (ajustable después)
        if ratio < 1.5:
            return "Balanced"
        elif ratio < 3:
            return "Mild imbalance"
        elif ratio < 10:
            return "Moderate imbalance"
        else:
            return "Severe imbalance"

    def _compute_imbalance_info(self):
        """
        Calcula imbalance del target SIN imprimir nada.
        Retorna dict o None si no aplica.
        """
        if self.target is None:
            return None

        s = self.df[self.target]

        is_cat = (
            is_object_dtype(s)
            or isinstance(s.dtype, pd.CategoricalDtype)
            or is_bool_dtype(s)
        )
        if not is_cat:
            return None

        counts = s.value_counts(dropna=False)
        if len(counts) <= 1:
            return {"counts": counts, "ratio": 1.0, "severity": "Balanced", "recommendations": []}

        ratio = float(counts.max() / counts.min())
        severity = self._imbalance_severity(ratio)
        recs = self._imbalance_recommendations(ratio)

        return {
            "counts": counts,
            "ratio": ratio,
            "severity": severity,
            "recommendations": recs
        }

    def _imbalance_recommendations(self, ratio: float) -> List[str]:
        # Recomendaciones típicas para clasificación
        recs = []
        if ratio >= 1.5:
            recs.append("Prefer F1-macro / balanced accuracy over plain accuracy.")
            recs.append("Use stratified split / stratified CV.")
        if ratio >= 3:
            recs.append("Consider class_weight='balanced' (where applicable).")
            recs.append("Consider resampling (RandomOverSampler / SMOTE) for training only.")
        if ratio >= 10:
            recs.append("Review data collection; extremely imbalanced targets can be unstable.")
        return recs
    
    def target_distribution(self):
        """
        Distribución del target si existe.

        - Si es categórico:
            Barra horizontal única segmentada (%)
        - Si es numérico:
            Histograma
        """

        if self.target is None:
            print("No definiste target.")
            return

        s = self.df[self.target]

        # ---------------------------------------------------------
        # TARGET CATEGÓRICO
        # ---------------------------------------------------------
        if (
            is_object_dtype(s)
            or isinstance(s.dtype, pd.CategoricalDtype)
            or is_bool_dtype(s)
        ):

            tab = s.value_counts(dropna=False)
            pct = (tab / tab.sum() * 100).round(2)

            out = pd.DataFrame({
                "category": tab.index.astype(str),
                "count": tab.values,
                "pct": pct.values
            })

            # -------------------------------------------------
            # CALCULO DEL IMBALANCE (SOLO CATEGORICO)
            # -------------------------------------------------

            imb = self._compute_imbalance_info()

            if imb is not None and len(tab) > 1:
                title_text = (
                    f"Distribución target: {self.target} | "
                    f"{imb['severity']} ({imb['ratio']:.2f}:1)"
                )
            else:
                title_text = f"Distribución target: {self.target}"


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

            fig.update_layout(
                barmode="stack",
                title=title_text,
                xaxis_title="Porcentaje (%)",
                yaxis_title="",
                showlegend=True,
                height=250
            )

            fig.update_xaxes(range=[0, 100])
            fig.show()

            return out.set_index("category")


        # ---------------------------------------------------------
        # TARGET NUMÉRICO
        # ---------------------------------------------------------
        else:
            fig = px.histogram(
                self.df,
                x=self.target,
                nbins=30,
                title=f"Distribución target: {self.target}"
            )
            fig.show()

            return s.describe()
        
    def imbalance_report(self) -> Optional[pd.DataFrame]:
        if self.target is None:
            print("No definiste target.")
            return None

        imb = self._compute_imbalance_info()
        if imb is None:
            print("El target no es categórico (o no aplica imbalance).")
            return None

        counts = imb["counts"]
        pct = (counts / counts.sum() * 100).round(2)
        report = pd.DataFrame({"count": counts, "pct": pct})

        if len(counts) > 1:
            majority_class = counts.idxmax()
            minority_class = counts.idxmin()

            print("========== IMBALANCE REPORT ==========")
            print(f"Majority class : {majority_class}")
            print(f"Minority class : {minority_class}")
            print(f"Imbalance ratio: {imb['ratio']:.2f} : 1")
            print(f"Severity       : {imb['severity']}")
            if imb["recommendations"]:
                print("Recommendations:")
                for r in imb["recommendations"]:
                    print(f" - {r}")

            report["severity"] = imb["severity"]
            report["imbalance_ratio"] = round(imb["ratio"], 4)
        else:
            print("========== IMBALANCE REPORT ==========")
            print("Solo hay una clase en el target; no se puede calcular imbalance.")
            report["severity"] = "Balanced"
            report["imbalance_ratio"] = 1.0

        return report

    def predictive_power_categorical(
        self,
        var: str,
        normalize: str = "index",   # "index" => % por fila (cada categoría suma 100%)
        min_count: int = 10,
        sort_by: str = "count",     # "count" | "alpha"
        show: bool = True
    ) -> pd.DataFrame:
        """
        Poder predictivo para variable categórica vs target.
        - Crea tabla de contingencia y la normaliza por fila (%)
        - Grafica stacked horizontal en Plotly
        - Devuelve el DataFrame (porcentajes) + conteos

        min_count: agrupa categorías raras en '__OTHER__' si su count < min_count
        """

        if self.target is None:
            raise ValueError("No definiste target.")

        if var not in self.df.columns:
            raise ValueError(f"'{var}' no existe en el DataFrame.")

        df = self.df[[var, self.target]].copy()

        # Manejo de NaN como categoría explícita
        df[var] = df[var].astype("object").fillna("__NA__")
        df[self.target] = df[self.target].astype("object").fillna("__NA__")

        # Agrupar categorías raras
        vc = df[var].value_counts(dropna=False)
        rare = vc[vc < min_count].index
        if len(rare) > 0:
            df[var] = df[var].where(~df[var].isin(rare), "__OTHER__")

        # Conteos
        ct_counts = pd.crosstab(df[var], df[self.target], dropna=False)

        # Porcentajes por fila (cada categoría suma 1.0)
        if normalize == "index":
            ct_pct = ct_counts.div(ct_counts.sum(axis=1), axis=0)
        elif normalize == "all":
            ct_pct = ct_counts / ct_counts.values.sum()
        else:
            raise ValueError("normalize debe ser 'index' o 'all'.")

        # Orden
        if sort_by == "count":
            order = ct_counts.sum(axis=1).sort_values(ascending=False).index
        elif sort_by == "alpha":
            order = sorted(ct_counts.index, key=lambda x: str(x))
        else:
            raise ValueError("sort_by debe ser 'count' o 'alpha'.")

        ct_counts = ct_counts.loc[order]
        ct_pct = ct_pct.loc[order]

        # DataFrame de salida (pct + count_total)
        out = (ct_pct * 100).round(2)
        out["count_total"] = ct_counts.sum(axis=1)
        out = out.rename_axis(var).reset_index()

        # Plot
        if show:
            # a formato largo para plotly
            long_df = ct_pct.reset_index().melt(
                id_vars=[var],
                var_name="target_class",
                value_name="pct"
            )
            long_df["pct"] = (long_df["pct"] * 100)

            fig = px.bar(
                long_df,
                y=var,
                x="pct",
                color="target_class",
                orientation="h",
                text=long_df["pct"].map(lambda x: f"{x:.1f}%"),
                title=f"Poder predictivo (categórica): {var} vs {self.target}",
            )
            fig.update_layout(barmode="stack", xaxis_title="Porcentaje (%)", yaxis_title=var)
            fig.update_xaxes(range=[0, 100])
            fig.show()

        return out

    def predictive_power_numeric(
        self,
        var: str,
        show: bool = True,
        grid_size: int = 256,
        clip_quantiles: tuple = (0.01, 0.99),
        bw_method=None,            # None usa default; podés poner "scott" o "silverman"
        fill_opacity: float = 0.35
    ) -> pd.DataFrame:
        """
        Poder predictivo numérico vs target con KDE (estilo seaborn fill=True) usando Plotly.

        Devuelve describe() por clase.
        """

        if self.target is None:
            raise ValueError("No definiste target.")
        if var not in self.df.columns:
            raise ValueError(f"'{var}' no existe en el DataFrame.")
        if not pd.api.types.is_numeric_dtype(self.df[var]):
            raise TypeError(f"'{var}' no es numérica. dtype={self.df[var].dtype}")

        df = self.df[[var, self.target]].dropna(subset=[var, self.target]).copy()

        # Stats (silencia warning observed)
        stats = df.groupby(self.target, observed=True)[var].describe()

        if not show:
            return stats

        # Rango robusto para evitar colas locas
        lo, hi = df[var].quantile(list(clip_quantiles)).values
        if lo == hi:
            lo, hi = df[var].min(), df[var].max()
            if lo == hi:
                # variable constante
                fig = go.Figure()
                fig.add_annotation(text=f"{var} es constante; no se puede estimar KDE.", showarrow=False)
                fig.update_layout(title=f"KDE: {var} por clase de {self.target}")
                fig.show()
                return stats

        xgrid = np.linspace(lo, hi, grid_size)

        fig = go.Figure()

        # Graficar KDE por clase
        for cls, sub in df.groupby(self.target, observed=True):
            x = sub[var].astype(float).values

            if len(x) < 2 or np.std(x) == 0:
                # muy pocos datos o varianza 0 → no KDE confiable
                continue

            kde = gaussian_kde(x, bw_method=bw_method)
            y = kde(xgrid)

            # línea
            fig.add_trace(go.Scatter(
                x=xgrid,
                y=y,
                mode="lines",
                name=str(cls),
            ))
            # relleno (duplicamos con fill tozeroy)
            fig.add_trace(go.Scatter(
                x=xgrid,
                y=y,
                mode="lines",
                name=str(cls),
                fill="tozeroy",
                opacity=fill_opacity,
                showlegend=False,
                hoverinfo="skip"
            ))

        fig.update_layout(
            title=f"Densidad (KDE): {var} por clase de {self.target}",
            xaxis_title=var,
            yaxis_title="Densidad",
            height=400
        )
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
        sort_by_cat: str = "count",   # "count" | "alpha"
        show_plots: bool = True,
        points_numeric: str = "outliers",
    ) -> Dict[str, Any]:
        """
        Ejecuta poder predictivo para múltiples variables:
        - Categóricas: stacked horizontal (% target por categoría)
        - Numéricas: violin por clase

        Devuelve:
        {
            "categorical": {var: df_out, ...},
            "numeric": {var: stats_df, ...},
            "skipped": {"categorical": [...], "numeric": [...]}
        }
        """

        if self.target is None:
            raise ValueError("No definiste target.")

        exclude = set(exclude or [])
        exclude.add(self.target)

        # Si el usuario no pasa lista, usamos lo detectado por DFCore
        if categorical_vars is None:
            categorical_vars = [c for c in self.cols.categorical if c not in exclude]
        else:
            categorical_vars = [c for c in categorical_vars if c not in exclude]

        if numeric_vars is None:
            numeric_vars = [c for c in self.cols.numeric if c not in exclude]
        else:
            numeric_vars = [c for c in numeric_vars if c not in exclude]

        # Limitar para evitar explotar el notebook
        categorical_vars = categorical_vars[:max_categorical]
        numeric_vars = numeric_vars[:max_numeric]

        results = {
            "categorical": {},
            "numeric": {},
            "skipped": {"categorical": [], "numeric": []}
        }

        # ---------- Categóricas ----------
        for var in categorical_vars:
            try:
                out = self.predictive_power_categorical(
                    var=var,
                    min_count=min_count_cat,
                    sort_by=sort_by_cat,
                    show=show_plots
                )
                results["categorical"][var] = out
            except Exception as e:
                results["skipped"]["categorical"].append((var, str(e)))

        # ---------- Numéricas ----------
        for var in numeric_vars:
            try:
                stats = self.predictive_power_numeric(
                    var=var,
                    show=show_plots
                )
                results["numeric"][var] = stats
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

        # Poder predictivo (compatibilidad)
        show_predictive_power: bool = False,
        use_power_all: bool = True,
        max_categorical: int = 12,
        max_numeric: int = 8,
        min_count_cat: int = 10,
        sort_by_cat: str = "count",
        points_numeric: str = "outliers",
        show_power_plots: bool = True,

        # Controles modulares nuevos (agregados al final para no romper llamadas posicionales)
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
    ) -> Dict[str, Any]:
        """
        Ejecuta un EDA modular. Mantiene compatibilidad con la API anterior.

        `summary()` sigue incluyendo missing y cardinalidad de forma compacta,
        mientras `missing_report()` y `cardinality_report()` permiten analizarlos
        por separado.
        """
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
            corr_out = self.correlation_report(
                method=corr_method,
                threshold=corr_threshold,
                show_plot=True,
            )
            results["correlation"] = corr_out["matrix"]
            results["strong_correlations"] = corr_out["strong_pairs"]
            if corr_out["strong_pairs"].empty:
                print(f"No se detectaron pares con |correlación| >= {corr_threshold:.2f}.")
            else:
                print(f"\nPares con |correlación| >= {corr_threshold:.2f}")
                self._display_df(corr_out["strong_pairs"])

        if show_outliers:
            print("\n========== OUTLIERS ==========")
            outlier_report = self.outlier_report(iqr_multiplier=iqr_multiplier)
            results["outlier_report"] = outlier_report
            if outlier_report.empty:
                print("No hay variables numéricas disponibles para el reporte.")
            else:
                self._display_df(outlier_report)
            self.outliers_boxplot(points=outlier_points)

        if show_skewness:
            print("\n========== SKEWNESS ==========")
            skew = self.skewness_report(threshold=skewness_threshold)
            results["skewness_report"] = skew
            if skew.empty:
                print("No hay variables numéricas disponibles para el reporte.")
            else:
                self._display_df(skew)

        if show_predictive_power and self.target is not None:
            print("\n========== PREDICTIVE POWER ==========")
            if use_power_all:
                results["predictive_power"] = self.predictive_power_all(
                    max_categorical=max_categorical,
                    max_numeric=max_numeric,
                    min_count_cat=min_count_cat,
                    sort_by_cat=sort_by_cat,
                    points_numeric=points_numeric,
                    show_plots=show_power_plots,
                )
            else:
                results["predictive_power"] = {
                    "categorical": {}, "numeric": {}, "skipped": {}
                }
                print("use_power_all=False: corre los métodos individuales manualmente.")

        return results

class EDAAdvanced(EDAQuick):
    """
    EDA avanzado orientado a "separabilidad" / poder predictivo.
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

        # --------- NUEVO: score interpretable 0-1 + strength ----------
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

        # (opcional) reordenar columnas para lectura
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

        # Cramér's V
        cv_list = []
        for c in features:
            ct = pd.crosstab(df[c], df[self.target], dropna=False)
            cv_list.append((c, self._cramers_v(ct)))
        cv_df = pd.DataFrame(cv_list, columns=["feature", "cramers_v"])

        # Chi²
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

        # --------- NUEVO: strength por Cramér's V ----------
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

        # (opcional) ordenar columnas para lectura
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
    # 4) Feature importance con Random Forest (SOLO gráfico)
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
    # 5) Runner EDA avanzado (NO imprime dict)
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
        return_results: bool = False,      # ✅ si lo pones True, te devuelve el dict
        show_outputs_summary: bool = True  # ✅ tabla resumen bonita (opcional)
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

        # ✅ guardar para inspección posterior (sin ensuciar output)
        self.last_results = results

        # ✅ resumen bonito para el usuario (en vez del dict crudo)
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

# ============================================================
# PreProcess modular (sin mezclar con EDA)
# ============================================================

class DFPreprocess(DFCore):
    """
    Preprocessing modular (sin mezclar con EDA):
    - Genera 2 matrices:
        * X_model: para supervised modeling (num + cat)
        * X_embed: para unsupervised/embeddings (num + cat compacto)
    - Soporta encoding: "ohe" | "ordinal" | "auto"
    - Reproducible: guarda preprocessors fitteados + feature names
    """

    def __init__(
        self,
        df: pd.DataFrame,
        target: Optional[str] = None,
        features: Optional[List[str]] = None
    ):
        super().__init__(df=df, target=target, features=features)

        # outputs y estado
        self.preprocessor_model_ = None
        self.preprocessor_embed_ = None
        self.feature_names_model_ = None
        self.feature_names_embed_ = None

        self.last_output_ = None  # para inspección rápida

    # ============================================================
    # Helpers: seleccionar X / y
    # ============================================================
    def _require_target(self):
        if self.target is None:
            raise ValueError("Esta operación requiere target/validation_label.")

    def get_X_y(
        self,
        features: Optional[List[str]] = None,
        dropna_target: bool = True
    ):
        """Devuelve X y, opcionalmente, y.

        En flujos no supervisados ``target`` puede ser None. En ese caso se
        devuelve ``y=None`` y todas las columnas seleccionadas forman X.
        """
        df = self.df.copy()
        if self.target is not None and dropna_target:
            df = df.dropna(subset=[self.target])

        if features is None:
            X_cols = [c for c in df.columns if c != self.target]
        else:
            missing = [c for c in features if c not in df.columns]
            if missing:
                raise ValueError(f"Estas columnas no existen en df: {missing}")
            X_cols = [c for c in features if c != self.target]

        X = df[X_cols].copy()
        y = None if self.target is None else df[self.target].copy()
        return X, y

    def _split_num_cat(
        self,
        X: pd.DataFrame,
        include_bool_as_cat: bool = True
    ) -> Dict[str, List[str]]:
        dt_cols = [c for c in X.columns if is_datetime64_any_dtype(X[c])]

        bool_cols = [c for c in X.columns if is_bool_dtype(X[c]) and c not in dt_cols]
        num_cols = [c for c in X.columns if is_numeric_dtype(X[c]) and c not in dt_cols and c not in bool_cols]

        cat_cols = [
            c for c in X.columns
            if (
                is_object_dtype(X[c])
                or isinstance(X[c].dtype, pd.CategoricalDtype)
            )
            and c not in dt_cols
            and c not in bool_cols
        ]

        if include_bool_as_cat:
            cat_cols = cat_cols + bool_cols
        else:
            num_cols = num_cols + bool_cols

        return {"numeric": num_cols, "categorical": cat_cols, "datetime": dt_cols}

    # ============================================================
    # Encoding strategy helpers
    # ============================================================
    def _build_numeric_pipeline(self, num_impute: str, scale_num: str):
        steps = [("imputer", SimpleImputer(strategy=num_impute))]

        if scale_num is None:
            pass
        elif scale_num == "standard":
            steps.append(("scaler", StandardScaler()))
        elif scale_num == "robust":
            steps.append(("scaler", RobustScaler()))
        elif scale_num == "minmax":
            steps.append(("scaler", MinMaxScaler()))
        elif scale_num in {"none", "passthrough"}:
            pass
        else:
            raise ValueError("scale_num debe ser: 'standard', 'robust', 'minmax' o None.")

        return Pipeline(steps=steps)

    def _build_cat_pipeline_ohe(
        self,
        cat_impute: str,
        ohe_sparse: bool,
        ohe_min_frequency: Optional[int] = None
    ):
        steps = [("imputer", SimpleImputer(strategy=cat_impute))]

        ohe_kwargs = dict(handle_unknown="ignore", sparse_output=ohe_sparse)
        if ohe_min_frequency is not None:
            ohe_kwargs["min_frequency"] = ohe_min_frequency

        steps.append(("ohe", OneHotEncoder(**ohe_kwargs)))
        return Pipeline(steps=steps)

    def _build_cat_pipeline_ordinal(
        self,
        cat_impute: str
    ):
        # OrdinalEncoder necesita que no haya NaNs -> imputamos primero
        steps = [
            ("imputer", SimpleImputer(strategy=cat_impute)),
            ("ord", OrdinalEncoder(
                handle_unknown="use_encoded_value",
                unknown_value=-1
            ))
        ]
        return Pipeline(steps=steps)

    def _auto_split_cats_by_cardinality(
        self,
        X: pd.DataFrame,
        cat_cols: List[str],
        threshold: int
    ) -> Dict[str, List[str]]:
        """
        Divide cat_cols en:
        - low_card: <= threshold (OHE)
        - high_card: > threshold (Ordinal)
        """
        low_card, high_card = [], []
        for c in cat_cols:
            n_levels = X[c].nunique(dropna=False)
            if n_levels <= threshold:
                low_card.append(c)
            else:
                high_card.append(c)
        return {"low_card": low_card, "high_card": high_card}

    # ============================================================
    # Build preprocessors
    # ============================================================
    def build_preprocessor_model(
        self,
        X: pd.DataFrame,
        include_bool_as_cat: bool = True,
        encoding: str = "auto",              # "ohe" | "ordinal" | "auto"
        auto_cardinality_threshold: int = 15,
        num_impute: str = "median",
        cat_impute: str = "most_frequent",
        scale_num: str = "standard",
        ohe_sparse: bool = False,
        ohe_min_frequency: Optional[int] = None
    ) -> ColumnTransformer:
        """
        Preprocessor para supervised modeling.
        - num: impute + scaler
        - cat: OHE/Ordinal/Auto
        """
        split = self._split_num_cat(X, include_bool_as_cat=include_bool_as_cat)
        num_cols, cat_cols = split["numeric"], split["categorical"]

        num_pipe = self._build_numeric_pipeline(num_impute=num_impute, scale_num=scale_num)

        encoding = encoding.lower().strip()
        transformers = [("num", num_pipe, num_cols)]

        if len(cat_cols) > 0:
            if encoding == "ohe":
                cat_pipe = self._build_cat_pipeline_ohe(cat_impute, ohe_sparse, ohe_min_frequency)
                transformers.append(("cat", cat_pipe, cat_cols))

            elif encoding == "ordinal":
                cat_pipe = self._build_cat_pipeline_ordinal(cat_impute)
                transformers.append(("cat", cat_pipe, cat_cols))

            elif encoding == "auto":
                split_c = self._auto_split_cats_by_cardinality(X, cat_cols, auto_cardinality_threshold)
                low_card = split_c["low_card"]
                high_card = split_c["high_card"]

                if len(low_card) > 0:
                    cat_ohe = self._build_cat_pipeline_ohe(cat_impute, ohe_sparse, ohe_min_frequency)
                    transformers.append(("cat_ohe", cat_ohe, low_card))

                if len(high_card) > 0:
                    cat_ord = self._build_cat_pipeline_ordinal(cat_impute)
                    transformers.append(("cat_ord", cat_ord, high_card))
            else:
                raise ValueError("encoding debe ser: 'ohe', 'ordinal' o 'auto'.")

        pre = ColumnTransformer(transformers=transformers, remainder="drop")
        return pre

    def _categorical_auto_plan(
        self,
        X: pd.DataFrame,
        cat_cols: List[str],
        ohe_max_levels: int = 15,
        frequency_max_levels: int = 200,
        id_like_ratio: float = 0.95,
        high_cardinality_ratio: float = 0.50,
    ) -> Dict[str, List[str]]:
        """Clasifica categóricas para OHE, frequency encoding o exclusión."""
        plan = {"ohe": [], "frequency": [], "exclude": []}
        n = max(len(X), 1)
        for c in cat_cols:
            n_unique = int(X[c].nunique(dropna=False))
            ratio = n_unique / n
            if ratio >= id_like_ratio:
                plan["exclude"].append(c)
            elif n_unique <= ohe_max_levels:
                plan["ohe"].append(c)
            elif n_unique <= frequency_max_levels and ratio < high_cardinality_ratio:
                plan["frequency"].append(c)
            else:
                plan["exclude"].append(c)
        return plan

    def _numeric_auto_plan(
        self,
        X: pd.DataFrame,
        num_cols: List[str],
        skewness_threshold: float = 1.0,
        outlier_pct_threshold: float = 1.0,
        iqr_multiplier: float = 1.5,
    ) -> Dict[str, List[str]]:
        """Divide numéricas entre StandardScaler y RobustScaler."""
        plan = {"standard": [], "robust": []}
        for c in num_cols:
            s = pd.to_numeric(X[c], errors="coerce").dropna()
            if s.empty or s.nunique() <= 1:
                plan["standard"].append(c)
                continue
            skew = float(s.skew()) if len(s) >= 3 else 0.0
            q1, q3 = s.quantile([0.25, 0.75])
            iqr = float(q3 - q1)
            if iqr > 0:
                low = q1 - iqr_multiplier * iqr
                high = q3 + iqr_multiplier * iqr
                outlier_pct = float(((s < low) | (s > high)).mean() * 100.0)
            else:
                outlier_pct = 0.0
            if abs(skew) >= skewness_threshold or outlier_pct >= outlier_pct_threshold:
                plan["robust"].append(c)
            else:
                plan["standard"].append(c)
        return plan

    def build_preprocessor_embed(
        self,
        X: pd.DataFrame,
        include_bool_as_cat: bool = True,
        cat_encoding: str = "auto",
        num_impute: str = "median",
        cat_impute: str = "most_frequent",
        scale_num: Optional[str] = "auto",
        ohe_max_levels: int = 15,
        frequency_max_levels: int = 200,
        id_like_ratio: float = 0.95,
        high_cardinality_ratio: float = 0.50,
        skewness_threshold: float = 1.0,
        outlier_pct_threshold: float = 1.0,
        iqr_multiplier: float = 1.5,
    ) -> Tuple[ColumnTransformer, Dict[str, Any]]:
        """Preprocessor específico para embeddings/clustering.

        ``scale_num='auto'`` aplica StandardScaler a variables aproximadamente
        simétricas y RobustScaler a variables con asimetría/outliers.

        ``cat_encoding='auto'`` aplica OHE a baja cardinalidad, frequency
        encoding a cardinalidad media y excluye columnas ID-like o de
        cardinalidad excesiva.
        """
        split = self._split_num_cat(X, include_bool_as_cat=include_bool_as_cat)
        num_cols, cat_cols = split["numeric"], split["categorical"]
        transformers = []
        report: Dict[str, Any] = {
            "numeric": {},
            "categorical": {},
            "excluded_columns": [],
        }

        scale_key = None if scale_num is None else str(scale_num).lower().strip()
        if scale_key == "auto":
            nplan = self._numeric_auto_plan(
                X, num_cols, skewness_threshold=skewness_threshold,
                outlier_pct_threshold=outlier_pct_threshold,
                iqr_multiplier=iqr_multiplier,
            )
            if nplan["standard"]:
                transformers.append(("num_standard", self._build_numeric_pipeline(num_impute, "standard"), nplan["standard"]))
            if nplan["robust"]:
                transformers.append(("num_robust", self._build_numeric_pipeline(num_impute, "robust"), nplan["robust"]))
            report["numeric"] = nplan
        else:
            if num_cols:
                transformers.append(("num", self._build_numeric_pipeline(num_impute, scale_num), num_cols))
            report["numeric"] = {str(scale_num): list(num_cols)}

        cat_key = str(cat_encoding).lower().strip()
        if cat_cols:
            if cat_key == "auto":
                cplan = self._categorical_auto_plan(
                    X, cat_cols, ohe_max_levels=ohe_max_levels,
                    frequency_max_levels=frequency_max_levels,
                    id_like_ratio=id_like_ratio,
                    high_cardinality_ratio=high_cardinality_ratio,
                )
                if cplan["ohe"]:
                    transformers.append(("cat_ohe", self._build_cat_pipeline_ohe(cat_impute, False, None), cplan["ohe"]))
                if cplan["frequency"]:
                    freq_pipe = Pipeline([
                        ("imputer", SimpleImputer(strategy=cat_impute)),
                        ("frequency", FrequencyEncoder()),
                    ])
                    transformers.append(("cat_frequency", freq_pipe, cplan["frequency"]))
                report["categorical"] = cplan
                report["excluded_columns"].extend(cplan["exclude"])
            elif cat_key == "ordinal":
                transformers.append(("cat_ordinal", self._build_cat_pipeline_ordinal(cat_impute), cat_cols))
                report["categorical"] = {"ordinal": list(cat_cols)}
            elif cat_key == "ohe":
                transformers.append(("cat_ohe", self._build_cat_pipeline_ohe(cat_impute, False, None), cat_cols))
                report["categorical"] = {"ohe": list(cat_cols)}
            elif cat_key == "frequency":
                freq_pipe = Pipeline([
                    ("imputer", SimpleImputer(strategy=cat_impute)),
                    ("frequency", FrequencyEncoder()),
                ])
                transformers.append(("cat_frequency", freq_pipe, cat_cols))
                report["categorical"] = {"frequency": list(cat_cols)}
            elif cat_key in {"exclude", "none"}:
                report["categorical"] = {"exclude": list(cat_cols)}
                report["excluded_columns"].extend(cat_cols)
            else:
                raise ValueError("cat_encoding debe ser: 'auto', 'ordinal', 'ohe', 'frequency' o 'exclude'.")

        if not transformers:
            raise ValueError("No quedaron variables para preparar después de aplicar las reglas de exclusión.")

        pre = ColumnTransformer(transformers=transformers, remainder="drop")
        return pre, report

    # ============================================================
    # Fit / Transform exclusivo para embeddings / clustering
    # ============================================================
    def fit_transform_embed(
        self,
        features: Optional[List[str]] = None,
        dropna_target: bool = True,
        include_bool_as_cat: bool = True,
        num_impute: str = "median",
        cat_impute: str = "most_frequent",
        scale_num: Optional[str] = "auto",
        cat_encoding: str = "auto",
        return_df: bool = True,
        ohe_max_levels: int = 15,
        frequency_max_levels: int = 200,
        id_like_ratio: float = 0.95,
        high_cardinality_ratio: float = 0.50,
        skewness_threshold: float = 1.0,
        outlier_pct_threshold: float = 1.0,
        iqr_multiplier: float = 1.5,
    ) -> Dict[str, Any]:
        """Construye únicamente la matriz usada por análisis no supervisado."""
        X, y = self.get_X_y(features=features, dropna_target=dropna_target)
        pre, preprocessing_report = self.build_preprocessor_embed(
            X=X,
            include_bool_as_cat=include_bool_as_cat,
            cat_encoding=cat_encoding,
            num_impute=num_impute,
            cat_impute=cat_impute,
            scale_num=scale_num,
            ohe_max_levels=ohe_max_levels,
            frequency_max_levels=frequency_max_levels,
            id_like_ratio=id_like_ratio,
            high_cardinality_ratio=high_cardinality_ratio,
            skewness_threshold=skewness_threshold,
            outlier_pct_threshold=outlier_pct_threshold,
            iqr_multiplier=iqr_multiplier,
        )
        Xt = pre.fit_transform(X)
        names = self.get_feature_names(pre)

        self.preprocessor_embed_ = pre
        self.feature_names_embed_ = names

        if return_df:
            Xt = self._to_dense_df(Xt, names)

        out = {
            "X_embed": Xt,
            "y": None if y is None else y.reset_index(drop=True),
            "feature_names_embed": names,
            "preprocessor_embed": pre,
            "preprocessing_report": preprocessing_report,
            "raw_index": X.index.to_numpy(),
        }
        self.last_output_ = out
        return out

    # ============================================================
    # Fit / Transform: outputs completos
    # ============================================================
    def fit_transform_all(
        self,
        features: Optional[List[str]] = None,
        dropna_target: bool = True,

        # columnas
        include_bool_as_cat: bool = True,

        # MODEL matrix config
        model_encoding: str = "auto",                # "ohe" | "ordinal" | "auto"
        auto_cardinality_threshold: int = 15,
        num_impute: str = "median",
        cat_impute: str = "most_frequent",
        scale_num: str = "standard",
        ohe_sparse: bool = False,
        ohe_min_frequency: Optional[int] = None,

        # EMBED matrix config
        embed_cat_encoding: str = "ordinal",         # "ordinal" (default) | "ohe"

        # outputs
        return_dfs: bool = True
    ) -> Dict[str, Any]:
        """
        Fit + Transform para:
          - X_model (supervised)
          - X_embed (unsupervised)
          - y
        """
        X, y = self.get_X_y(features=features, dropna_target=dropna_target)

        # 1) build + fit preprocessors
        pre_model = self.build_preprocessor_model(
            X=X,
            include_bool_as_cat=include_bool_as_cat,
            encoding=model_encoding,
            auto_cardinality_threshold=auto_cardinality_threshold,
            num_impute=num_impute,
            cat_impute=cat_impute,
            scale_num=scale_num,
            ohe_sparse=ohe_sparse,
            ohe_min_frequency=ohe_min_frequency
        )

        pre_embed = self.build_preprocessor_embed(
            X=X,
            include_bool_as_cat=include_bool_as_cat,
            cat_encoding=embed_cat_encoding,
            num_impute=num_impute,
            cat_impute=cat_impute,
            scale_num=scale_num
        )

        pre_model.fit(X)
        pre_embed.fit(X)

        self.preprocessor_model_ = pre_model
        self.preprocessor_embed_ = pre_embed

        # 2) transform
        Xt_model = pre_model.transform(X)
        Xt_embed = pre_embed.transform(X)

        # 3) feature names
        fn_model = self.get_feature_names(pre_model)
        fn_embed = self.get_feature_names(pre_embed)

        self.feature_names_model_ = fn_model
        self.feature_names_embed_ = fn_embed

        # 4) return as DataFrames if requested
        if return_dfs:
            Xt_model = self._to_dense_df(Xt_model, fn_model)
            Xt_embed = self._to_dense_df(Xt_embed, fn_embed)

        out = {
            "X_model": Xt_model,
            "X_embed": Xt_embed,
            "y": None if y is None else y.reset_index(drop=True),
            "feature_names_model": fn_model,
            "feature_names_embed": fn_embed,
            "preprocessor_model": pre_model,
            "preprocessor_embed": pre_embed,
        }

        self.last_output_ = out
        return out

    # ============================================================
    # Utilities
    # ============================================================
    def _to_dense_df(self, Xt, feature_names: Optional[List[str]] = None) -> pd.DataFrame:
        try:
            Xt_arr = Xt.toarray()
        except Exception:
            Xt_arr = np.asarray(Xt)

        if feature_names is None:
            return pd.DataFrame(Xt_arr)

        return pd.DataFrame(Xt_arr, columns=feature_names)

    def get_feature_names(self, pre: ColumnTransformer) -> List[str]:
        """
        Intenta reconstruir nombres de features post-transform:
        - num pipeline -> columnas originales numéricas
        - cat OHE -> get_feature_names_out
        - cat ordinal -> columnas originales cat
        - auto -> combina ambos
        """
        names: List[str] = []

        for name, transformer, cols in pre.transformers_:
            if cols is None or len(cols) == 0:
                continue

            if name.startswith("num"):
                names.extend(list(cols))
                continue

            # transformer puede ser Pipeline
            if isinstance(transformer, Pipeline):
                last_step = transformer.steps[-1][1]

                if isinstance(last_step, OneHotEncoder):
                    names.extend(last_step.get_feature_names_out(cols).tolist())
                elif isinstance(last_step, (OrdinalEncoder, FrequencyEncoder)):
                    names.extend(list(cols))
                else:
                    names.extend(list(cols))
            else:
                # fallback
                names.extend(list(cols))

        return names



@dataclass
class ClusterSolution:
    """Objeto de resultado para una solución de clustering.

    Conserva compatibilidad con el antiguo diccionario mediante ``get`` y
    ``__getitem__``, pero añade una API orientada a resultados.
    """
    analysis: Any
    name: str
    result: Dict[str, Any]

    def __getitem__(self, key: str) -> Any:
        return self.result[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self.result.get(key, default)

    def keys(self):
        return self.result.keys()

    @property
    def labels(self) -> np.ndarray:
        return np.asarray(self.result["labels"])

    @property
    def model(self) -> Any:
        return self.result.get("model")

    def dataset(
        self,
        cluster_column: str = "cluster",
        include_embedding: bool = False,
        embedding_prefix: str = "cluster_space_",
        copy: bool = True,
    ) -> pd.DataFrame:
        """Devuelve el DataFrame original con la asignación de cluster."""
        df_out = self.analysis.df.copy() if copy else self.analysis.df
        if len(df_out) != len(self.labels):
            raise ValueError("La cantidad de etiquetas no coincide con las filas del DataFrame.")
        df_out[cluster_column] = self.labels

        if include_embedding:
            X_used = self.result.get("X_used", "embed")
            Xmat = self.analysis._resolve_X(X_used)
            Xdf = Xmat.copy() if isinstance(Xmat, pd.DataFrame) else pd.DataFrame(np.asarray(Xmat))
            Xdf = Xdf.reset_index(drop=True)
            Xdf.columns = [f"{embedding_prefix}{c}" for c in Xdf.columns]
            df_out = pd.concat([df_out.reset_index(drop=True), Xdf], axis=1)
        return df_out

    def profile(self, top_n_categories: int = 5, include_target: bool = True) -> Dict[str, Any]:
        """Perfiles descriptivos por cluster sobre las variables originales."""
        return self.analysis.profile_clusters(
            labels=self.labels,
            top_n_categories=top_n_categories,
            include_target=include_target,
        )

    def summary(self) -> pd.DataFrame:
        """Resumen tabular de algoritmo, parámetros, métricas y convergencia."""
        excluded = {"labels", "model"}
        row = {"solution": self.name}
        for key, value in self.result.items():
            if key in excluded:
                continue
            if np.isscalar(value) or value is None or isinstance(value, (str, bool)):
                row[key] = value
        return pd.DataFrame([row])

    def centroids(
        self,
        statistic: str = "mean",
    ) -> pd.DataFrame:
        """Devuelve el perfil numérico por cluster en la escala original.

        Parameters
        ----------
        statistic:
            ``"mean"`` para medias o ``"median"`` para medianas.
        """
        statistic = statistic.lower().strip()
        profiles = self.profile()
        if statistic in {"mean", "means", "average"}:
            return profiles.get("numeric_means", pd.DataFrame()).copy()
        if statistic in {"median", "medians"}:
            return profiles.get("numeric_medians", pd.DataFrame()).copy()
        raise ValueError("statistic debe ser 'mean' o 'median'.")

    def plot_cluster_bars(
        self,
        centroids: Optional[pd.DataFrame] = None,
        statistic: str = "mean",
        normalize: bool = True,
        top_n: Optional[int] = None,
        clusters: Optional[List[int]] = None,
        title: Optional[str] = None,
        figsize: Optional[tuple] = None,
    ):
        """Atajo orientado a la solución para el gráfico de barras."""
        if centroids is None:
            centroids = self.centroids(statistic=statistic)
        return self.analysis.plot_cluster_bars(
            centroids=centroids,
            normalize=normalize,
            top_n=top_n,
            clusters=clusters,
            title=title or f"{self.name} — centroides por cluster",
            figsize=figsize,
        )

    def plot_cluster_radar(
        self,
        centroids: Optional[pd.DataFrame] = None,
        statistic: str = "mean",
        normalize: str = "minmax",
        vars: Optional[List[str]] = None,
        top_n: Optional[int] = 10,
        clusters: Optional[List[int]] = None,
        title: Optional[str] = None,
        figsize: tuple = (8, 8),
    ):
        """Atajo orientado a la solución para el gráfico radar."""
        if centroids is None:
            centroids = self.centroids(statistic=statistic)
        return self.analysis.plot_cluster_radar(
            centroids=centroids,
            normalize=normalize,
            vars=vars,
            top_n=top_n,
            clusters=clusters,
            title=title or f"{self.name} — perfil radar",
            figsize=figsize,
        )

    def describe(
        self,
        top_n: int = 3,
        cluster_column: str = "cluster",
        format: str = "long",
        min_abs_z: float = 0.0,
    ) -> pd.DataFrame:
        """Resume qué variables numéricas distinguen más a cada cluster.

        ``format='long'`` devuelve una fila por variable, evitando columnas de
        texto truncadas en Jupyter. ``format='wide'`` conserva el resumen
        compacto anterior.
        """
        data = self.dataset(cluster_column=cluster_column)
        numeric = [c for c in self.analysis.features if c in data.columns and is_numeric_dtype(data[c])]
        if not numeric:
            return pd.DataFrame()

        means = data.groupby(cluster_column)[numeric].mean()
        global_mean = data[numeric].mean()
        global_std = data[numeric].std(ddof=0).replace(0, np.nan)
        z = (means - global_mean) / global_std
        sizes = data[cluster_column].value_counts().sort_index()

        format = format.lower().strip()
        if format == "long":
            rows = []
            for cluster_id, values in z.iterrows():
                clean = values.dropna()
                high = clean[clean >= min_abs_z].sort_values(ascending=False).head(top_n)
                low = clean[clean <= -min_abs_z].sort_values().head(top_n)

                for rank, (feature, score) in enumerate(high.items(), start=1):
                    rows.append({
                        cluster_column: cluster_id,
                        "n": int(sizes.get(cluster_id, 0)),
                        "direction": "high",
                        "rank": rank,
                        "feature": feature,
                        "z_score": float(score),
                    })
                for rank, (feature, score) in enumerate(low.items(), start=1):
                    rows.append({
                        cluster_column: cluster_id,
                        "n": int(sizes.get(cluster_id, 0)),
                        "direction": "low",
                        "rank": rank,
                        "feature": feature,
                        "z_score": float(score),
                    })

            out = pd.DataFrame(rows)
            if not out.empty:
                out["z_score"] = out["z_score"].round(3)
                out = out.set_index([cluster_column, "direction", "rank"])
            return out

        if format == "wide":
            rows = []
            for cluster_id, values in z.iterrows():
                clean = values.dropna()
                high = clean.sort_values(ascending=False).head(top_n)
                low = clean.sort_values().head(top_n)
                rows.append({
                    cluster_column: cluster_id,
                    "n": int(sizes.get(cluster_id, 0)),
                    "high_features": "\n".join(f"{k}: {v:+.2f}σ" for k, v in high.items()),
                    "low_features": "\n".join(f"{k}: {v:+.2f}σ" for k, v in low.items()),
                })
            return pd.DataFrame(rows).set_index(cluster_column)

        raise ValueError("format debe ser 'long' o 'wide'.")

    def plot(
        self,
        embedding: Optional[str] = None,
        title: Optional[str] = None,
        save: bool = True,
        filename: Optional[str] = None,
    ):
        """Grafica la solución en 2D.

        Si el espacio usado no es 2D, crea una proyección PCA únicamente para
        visualización, sin alterar el espacio donde se ajustó el clustering.
        """
        if embedding is None:
            used = self.result.get("X_used", "feature_space")
            canonical_used = self.analysis._canonical_space_name(used)
            try:
                source = self.analysis._resolve_X(canonical_used)
                n_dims = source.shape[1]
            except Exception:
                n_dims = None

            if n_dims == 2:
                embedding = canonical_used
            else:
                embedding = f"__plot_{self.name}"
                if embedding not in self.analysis.embeddings_:
                    self.analysis.reduce("pca", n_components=2, name=embedding, X=canonical_used)
        else:
            canonical_requested = self.analysis._canonical_space_name(embedding)
            if canonical_requested == "feature_space":
                source = self.analysis._resolve_X("feature_space")
                if source.shape[1] != 2:
                    auto_name = f"__plot_{self.name}_feature_space"
                    if auto_name not in self.analysis.embeddings_:
                        self.analysis.reduce("pca", n_components=2, name=auto_name, X="feature_space")
                    embedding = auto_name
                else:
                    embedding = "feature_space"

        return self.analysis.plot_embedding(
            embedding=embedding,
            color_by="cluster",
            cluster_labels=self.labels,
            title=title or f"{self.name} — clusters",
            save=save,
            filename=filename,
        )

    def export_excel(
        self,
        path: Optional[str] = None,
        cluster_column: str = "cluster",
    ) -> str:
        """Exporta dataset, resumen y perfiles a un archivo Excel."""
        if path is None:
            path = os.path.join(self.analysis.output_dir, f"{self.name}.xlsx")
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

        profiles = self.profile()
        with pd.ExcelWriter(path) as writer:
            self.dataset(cluster_column=cluster_column).to_excel(writer, sheet_name="clustered_data", index=False)
            self.summary().to_excel(writer, sheet_name="solution_summary", index=False)
            self.describe(cluster_column=cluster_column).to_excel(writer, sheet_name="cluster_description", index=False)
            for key, value in profiles.items():
                if isinstance(value, pd.DataFrame):
                    value.to_excel(writer, sheet_name=str(key)[:31])
        return path


# ============================================================
# Unsupervised Analysis: Dimensionality Reduction + Clustering
# ============================================================



@dataclass
class ClusterModelSpec:
    """Metadata and search defaults for one clustering family."""
    name: str
    family: str
    requires_k: bool
    supports_predict: bool
    supports_noise: bool
    parameter_grid: Dict[str, Any]
    recommended_for: Tuple[str, ...] = ()
    limitations: Tuple[str, ...] = ()
    optional_dependency: Optional[str] = None


@dataclass
class ClusteringCompetition:
    """Result object produced by :meth:`UnsupervisedAnalysis.compete_clustering`."""
    analysis: Any
    leaderboard_df: pd.DataFrame
    candidates_df: pd.DataFrame
    config: Dict[str, Any]

    def leaderboard(self, top_n: Optional[int] = None) -> pd.DataFrame:
        out = self.leaderboard_df.copy()
        return out.head(top_n) if top_n is not None else out

    def candidates(self, valid_only: bool = False) -> pd.DataFrame:
        out = self.candidates_df.copy()
        if valid_only and "valid" in out.columns:
            out = out[out["valid"]].copy()
        return out

    @property
    def best_row(self) -> Dict[str, Any]:
        if self.leaderboard_df.empty:
            raise RuntimeError("La competencia no produjo candidatos válidos.")
        return self.leaderboard_df.iloc[0].to_dict()

    @property
    def best_model(self) -> str:
        return str(self.best_row["model"])

    @property
    def best_space(self) -> str:
        return str(self.best_row["space"])

    @property
    def best_params(self) -> Dict[str, Any]:
        value = self.best_row.get("params", {})
        return dict(value) if isinstance(value, dict) else {}

    def fit_best(self, name: str = "best_clustering_solution") -> ClusterSolution:
        row = self.best_row
        return self.analysis._fit_cluster_candidate(
            method=str(row["model"]),
            X=str(row["space"]),
            params=dict(row.get("params", {})),
            name=name,
        )

    def recommendation(self) -> Dict[str, Any]:
        row = self.best_row
        return {
            "recommended_model": row.get("model"),
            "recommended_space": row.get("space"),
            "recommended_params": row.get("params"),
            "n_clusters": row.get("n_clusters"),
            "silhouette": row.get("silhouette"),
            "stability_ari": row.get("stability_ari"),
            "reason": (
                "Mejor ranking compuesto entre calidad interna, estabilidad, "
                "ruido, balance de tamaños y validez de la solución."
            ),
        }


@dataclass
class ClusteringExperimentPlan:
    """Transparent experiment plan built before the clustering competition.

    The plan does not declare a winner. It organizes available model families
    into methodological roles (baseline, robust, hierarchical, probabilistic,
    density) and can launch :meth:`compete_clustering` with the recorded setup.
    """
    analysis: Any
    plan_df: pd.DataFrame
    recommendations_df: pd.DataFrame
    config: Dict[str, Any]

    def plan(self, available_only: bool = False) -> pd.DataFrame:
        out = self.plan_df.copy()
        if available_only and "available" in out.columns:
            out = out[out["available"]].copy()
        return out.reset_index(drop=True)

    def summary(self) -> Dict[str, Any]:
        usable = self.plan(available_only=True)
        return {
            "spaces": list(self.config.get("spaces", [])),
            "n_planned_models": int(len(self.plan_df)),
            "n_available_models": int(len(usable)),
            "models": usable["model"].tolist() if not usable.empty else [],
            "k_range": list(self.config.get("k_range", [])),
            "evaluate_stability": bool(self.config.get("evaluate_stability", True)),
        }

    def explain(self, available_only: bool = False) -> pd.DataFrame:
        """Print a readable, non-truncated experiment plan and return it."""
        df = self.plan(available_only=available_only)
        for _, row in df.iterrows():
            print("=" * 68)
            print(f"STEP {int(row['step'])}: {str(row['role']).upper()}")
            print(f"Model     : {row['model']}")
            print(f"Family    : {row['family']}")
            print(f"Priority  : {row['priority_label']} ({row['priority_score']:.1f})")
            print(f"Available : {'Yes' if bool(row['available']) else 'No'}")
            if row.get("note"):
                print(f"Note      : {row['note']}")
            print("Reasons:")
            reasons = row.get("reasons", [])
            if not isinstance(reasons, (list, tuple)):
                reasons = [x.strip() for x in str(reasons).split(";") if x.strip()]
            for reason in reasons:
                print(f"  • {reason}")
            print("Purpose in experiment:")
            print(f"  {row['experiment_purpose']}")
        print("=" * 68)
        print("The plan prioritizes candidates; the competition determines performance.")
        return df

    def run(self, **overrides) -> "ClusteringCompetition":
        """Run the recorded experiment, allowing explicit parameter overrides."""
        cfg = dict(self.config)
        cfg.update(overrides)
        models = cfg.pop("models", None)
        if models is None:
            models = self.plan(available_only=True)["model"].tolist()
        return self.analysis.compete_clustering(models=models, **cfg)



class UnsupervisedAnalysis(DFCore):
    """
    Unsupervised module:
    - prepara data con DFPreprocess (X_embed, y)
    - Dimensionality Reduction (PCA / RPCA / UMAP / t-SNE)
    - Clustering (KMeans / DBSCAN ...)
    - Permite clusterizar sobre:
        * feature_space / X_embed (preprocesado)
        * o el embedding generado por DR (ej: PCA/UMAP)
    """

    def __init__(
        self,
        df: pd.DataFrame,
        validation_label: Optional[str] = None,
        features: Optional[List[str]] = None,
        id_columns: Optional[List[str]] = None,
        output_dir: str = "unsupervised_outputs",
        target: Optional[str] = None,  # backward compatibility
    ):
        if validation_label is not None and target is not None and validation_label != target:
            raise ValueError("Usa validation_label o target, pero no valores diferentes en ambos.")
        label = validation_label if validation_label is not None else target
        self.validation_label = label
        self.id_columns = list(id_columns or [])

        missing_ids = [c for c in self.id_columns if c not in df.columns]
        if missing_ids:
            raise ValueError(f"Estas id_columns no existen en df: {missing_ids}")

        if features is None:
            features = [c for c in df.columns if c != label and c not in self.id_columns]
        else:
            features = [c for c in features if c != label and c not in self.id_columns]

        super().__init__(df=df, target=label, features=features)
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        self.X_embed_ = None
        self.y_ = None
        self.ids_ = self.df[self.id_columns].copy() if self.id_columns else None

        self.preprocessor_: Optional[DFPreprocess] = None
        self.preprocessor_embed_ = None
        self.feature_names_embed_: Optional[List[str]] = None
        self.prep_output_ = None
        self.preprocessing_report_: Optional[Dict[str, Any]] = None

        self.embeddings_: Dict[str, pd.DataFrame] = {}
        self.reducers_: Dict[str, Any] = {}
        self.reduction_outputs_: Dict[str, Dict[str, Any]] = {}
        # Parámetros efectivos usados por cada embedding/reductor
        self.embedding_parameters_: Dict[str, Dict[str, Any]] = {}
        self.cluster_results_: Dict[str, ClusterSolution] = {}
        self.last_solution_: Optional[ClusterSolution] = None
        self.cluster_model_specs_: Dict[str, ClusterModelSpec] = {}
        self.last_clustering_competition_: Optional[ClusteringCompetition] = None
        self.last_clustering_experiment_: Optional[ClusteringExperimentPlan] = None
        self.best_space_: Optional[str] = None
        self.best_k_: Optional[int] = None
        self.add_default_cluster_models()

    # ============================================================
    # 0) Preparar matriz exclusiva para unsupervised
    # ============================================================
    def prepare(
        self,
        scale_num: Optional[str] = "auto",
        cat_encoding: str = "auto",
        num_impute: str = "median",
        cat_impute: str = "most_frequent",
        include_bool_as_cat: bool = True,
        return_dfs: bool = True,
        ohe_max_levels: int = 15,
        frequency_max_levels: int = 200,
        id_like_ratio: float = 0.95,
        high_cardinality_ratio: float = 0.50,
        skewness_threshold: float = 1.0,
        outlier_pct_threshold: float = 1.0,
        iqr_multiplier: float = 1.5,
        verbose: bool = True,
        # backward compatibility
        model_encoding: Optional[str] = None,
        auto_cardinality_threshold: int = 15,
        embed_cat_encoding: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Prepara la matriz de clustering con reglas explícitas o automáticas."""
        if embed_cat_encoding is not None:
            cat_encoding = embed_cat_encoding
        if ohe_max_levels == 15 and auto_cardinality_threshold != 15:
            ohe_max_levels = auto_cardinality_threshold

        prep = DFPreprocess(self.df, target=self.validation_label, features=self.features)
        out = prep.fit_transform_embed(
            features=self.features,
            include_bool_as_cat=include_bool_as_cat,
            num_impute=num_impute,
            cat_impute=cat_impute,
            scale_num=scale_num,
            cat_encoding=cat_encoding,
            return_df=return_dfs,
            ohe_max_levels=ohe_max_levels,
            frequency_max_levels=frequency_max_levels,
            id_like_ratio=id_like_ratio,
            high_cardinality_ratio=high_cardinality_ratio,
            skewness_threshold=skewness_threshold,
            outlier_pct_threshold=outlier_pct_threshold,
            iqr_multiplier=iqr_multiplier,
        )

        self.preprocessor_ = prep
        self.preprocessor_embed_ = out["preprocessor_embed"]
        self.feature_names_embed_ = out["feature_names_embed"]
        self.X_embed_ = out["X_embed"]
        self.y_ = out["y"]
        self.prep_output_ = out
        self.preprocessing_report_ = out["preprocessing_report"]

        if verbose:
            print("========== UNSUPERVISED PREPROCESSING PLAN ==========")
            print("Numeric:", self.preprocessing_report_.get("numeric", {}))
            print("Categorical:", self.preprocessing_report_.get("categorical", {}))
            excluded = self.preprocessing_report_.get("excluded_columns", [])
            print("Excluded:", excluded if excluded else "None")
            print("Output shape:", getattr(self.X_embed_, "shape", None))

        return {
            "X_embed": self.X_embed_,
            "validation_label": self.y_,
            "y": self.y_,
            "ids": self.ids_,
            "feature_names_embed": self.feature_names_embed_,
            "preprocessor_embed": self.preprocessor_embed_,
            "preprocessing_report": self.preprocessing_report_,
        }

    # ============================================================
    # 1) Dimensionality Reduction
    # ============================================================
    def reduce(
        self,
        method: str = "pca",
        n_components: int = 2,
        random_state: Optional[int] = None,
        name: Optional[str] = None,
        X: str = "embed",
        **kwargs
    ) -> pd.DataFrame:
        source = self._resolve_X(X)
        method_key = (method or "").lower().strip()
        key = name or method_key

        if method_key == "pca":
            model = PCA(n_components=n_components, **kwargs)
            emb = model.fit_transform(source)
            metadata = {
                "explained_variance_ratio": model.explained_variance_ratio_.copy(),
                "cumulative_explained_variance": np.cumsum(model.explained_variance_ratio_),
                "components": model.components_.copy(),
            }
        elif method_key in {"tsne", "t-sne"}:
            model = TSNE(n_components=n_components, random_state=random_state, **kwargs)
            emb = model.fit_transform(source)
            metadata = {}
        elif method_key == "umap":
            try:
                import umap
            except Exception as e:
                raise ImportError("Falta umap-learn. Instala: pip install umap-learn") from e
            model = umap.UMAP(n_components=n_components, random_state=random_state, **kwargs)
            emb = model.fit_transform(source)
            metadata = {}
        else:
            raise ValueError("method no soportado. Usa: 'pca', 'tsne' o 'umap'.")

        emb_arr = np.asarray(emb)
        emb_df = pd.DataFrame(emb_arr, columns=[f"Dim{i+1}" for i in range(emb_arr.shape[1])])

        # Guardar los parámetros efectivos del estimador para auditoría y reproducibilidad.
        try:
            effective_params = model.get_params(deep=False)
        except Exception:
            effective_params = {"n_components": n_components, **kwargs}
            if method_key in {"tsne", "t-sne", "umap"}:
                effective_params["random_state"] = random_state

        self.embeddings_[key] = emb_df
        self.reducers_[key] = model
        self.embedding_parameters_[key] = dict(effective_params)
        self.reduction_outputs_[key] = {
            "method": method_key,
            "source": X,
            "model": model,
            "embedding": emb_df,
            "parameters": dict(effective_params),
            **metadata,
        }
        self.embedding_ = emb_df  # backward compatibility
        self.reducer_ = model
        return emb_df


    # ============================================================
    # 1B) Exploration report for unsupervised learning
    # ============================================================
    def explore(
        self,
        methods: Tuple[str, ...] = ("pca", "umap", "tsne"),
        visual_n_components: int = 2,
        pca_model_variance: float = 0.90,
        random_state: Optional[int] = None,
        plot: bool = True,
        color_by: Optional[str] = "none",
        cluster_labels: Optional[np.ndarray] = None,
        legend: bool = True,
        cmap: str = "viridis",
        alpha: float = 0.72,
        point_size: int = 22,
        pca_params: Optional[Dict[str, Any]] = None,
        pca_model_params: Optional[Dict[str, Any]] = None,
        umap_params: Optional[Dict[str, Any]] = None,
        tsne_params: Optional[Dict[str, Any]] = None,
        method_kwargs: Optional[Dict[str, Dict[str, Any]]] = None,
        adaptive_defaults: bool = True,
    ) -> Dict[str, Any]:
        """Explora representaciones útiles antes del clustering.

        Los parámetros de PCA, UMAP y t-SNE son explícitos y quedan guardados en
        ``embedding_parameters_`` y ``reduction_outputs_[name]["parameters"]``.

        Si ``adaptive_defaults=True`` y el usuario no proporciona valores:
        - UMAP usa ``n_neighbors ~= sqrt(n_samples)`` (acotado entre 5 y 30).
        - t-SNE usa ``perplexity ~= sqrt(n_samples)`` (acotada entre 5 y 30 y
          siempre menor que ``n_samples``).

        ``method_kwargs`` se conserva por compatibilidad con notebooks anteriores;
        los diccionarios específicos tienen prioridad.
        """
        if self.X_embed_ is None:
            raise ValueError("Primero ejecuta prepare() para crear X_embed_.")
        if not 0 < pca_model_variance <= 1:
            raise ValueError("pca_model_variance debe estar en (0, 1].")

        method_kwargs = method_kwargs or {}
        n_samples = int(len(self.X_embed_))
        if n_samples < 3:
            raise ValueError("Se requieren al menos 3 observaciones para explorar embeddings.")

        # Compatibilidad y precedencia: parámetros específicos > method_kwargs > defaults.
        pca_cfg = {**method_kwargs.get("pca", {}), **(pca_params or {})}
        pca_model_cfg = {**method_kwargs.get("pca_model", {}), **(pca_model_params or {})}
        umap_cfg = {**method_kwargs.get("umap", {}), **(umap_params or {})}
        tsne_cfg = {**method_kwargs.get("tsne", {}), **(tsne_params or {})}

        if adaptive_defaults:
            local_scale = int(round(np.sqrt(n_samples)))
            default_neighbors = min(30, max(5, local_scale), n_samples - 1)
            default_perplexity = min(30.0, max(5.0, float(local_scale)), float(n_samples - 1))
        else:
            default_neighbors = min(15, n_samples - 1)
            default_perplexity = min(30.0, float(n_samples - 1))

        umap_defaults = {
            "n_neighbors": default_neighbors,
            "min_dist": 0.1,
            "metric": "euclidean",
        }
        tsne_defaults = {
            "perplexity": default_perplexity,
            "learning_rate": "auto",
            "max_iter": 1000,
            "init": "pca",
            "metric": "euclidean",
        }
        umap_cfg = {**umap_defaults, **umap_cfg}
        tsne_cfg = {**tsne_defaults, **tsne_cfg}

        # Validaciones explícitas para errores más claros.
        if not 2 <= int(umap_cfg["n_neighbors"]) < n_samples:
            raise ValueError(f"UMAP n_neighbors debe estar entre 2 y {n_samples - 1}.")
        if not 0 <= float(umap_cfg["min_dist"]) <= 1:
            raise ValueError("UMAP min_dist debe estar entre 0 y 1.")
        if not 0 < float(tsne_cfg["perplexity"]) < n_samples:
            raise ValueError(f"t-SNE perplexity debe ser > 0 y < n_samples ({n_samples}).")

        requested = [str(m).lower().strip() for m in methods]
        rows: List[Dict[str, Any]] = []
        created: List[str] = []
        skipped: Dict[str, str] = {}

        def _params_text(params: Dict[str, Any], keys: Tuple[str, ...]) -> str:
            vals = []
            for key in keys:
                if key in params:
                    vals.append(f"{key}={params[key]}")
            return ", ".join(vals)

        if "pca" in requested:
            pca_visual = self.reduce(
                method="pca",
                n_components=visual_n_components,
                name="pca_visual",
                **pca_cfg,
            )
            created.append("pca_visual")
            pca_meta = self.reduction_outputs_["pca_visual"]
            cum = pca_meta.get("cumulative_explained_variance")
            pca_effective = self.embedding_parameters_["pca_visual"]
            rows.append({
                "embedding": "pca_visual",
                "method": "pca",
                "n_components": int(pca_visual.shape[1]),
                "variance_explained": float(cum[-1]) if cum is not None and len(cum) else np.nan,
                "parameters": _params_text(pca_effective, ("svd_solver", "whiten", "random_state")),
                "purpose": "visualization",
                "status": "created",
            })

            pca_model = self.reduce(
                method="pca",
                n_components=pca_model_variance,
                name="pca_model",
                **pca_model_cfg,
            )
            created.append("pca_model")
            model_meta = self.reduction_outputs_["pca_model"]
            cum_model = model_meta.get("cumulative_explained_variance")
            pca_model_effective = self.embedding_parameters_["pca_model"]
            rows.append({
                "embedding": "pca_model",
                "method": "pca",
                "n_components": int(pca_model.shape[1]),
                "variance_explained": float(cum_model[-1]) if cum_model is not None and len(cum_model) else np.nan,
                "parameters": _params_text(pca_model_effective, ("svd_solver", "whiten", "random_state")),
                "purpose": "clustering candidate",
                "status": "created",
            })

        for method in requested:
            if method == "pca":
                continue
            if method not in {"umap", "tsne", "t-sne"}:
                skipped[method] = "unsupported exploration method"
                continue
            canonical = "tsne" if method in {"tsne", "t-sne"} else "umap"
            name = f"{canonical}_visual"
            cfg = umap_cfg if canonical == "umap" else tsne_cfg
            try:
                emb = self.reduce(
                    method=canonical,
                    n_components=visual_n_components,
                    random_state=random_state,
                    name=name,
                    **cfg,
                )
                created.append(name)
                effective = self.embedding_parameters_[name]
                param_keys = ("n_neighbors", "min_dist", "metric", "random_state") if canonical == "umap" else (
                    "perplexity", "learning_rate", "max_iter", "init", "metric", "random_state"
                )
                rows.append({
                    "embedding": name,
                    "method": canonical,
                    "n_components": int(emb.shape[1]),
                    "variance_explained": np.nan,
                    "parameters": _params_text(effective, param_keys),
                    "purpose": "visualization / local structure",
                    "status": "created",
                })
            except Exception as exc:
                skipped[canonical] = str(exc)
                rows.append({
                    "embedding": name,
                    "method": canonical,
                    "n_components": visual_n_components,
                    "variance_explained": np.nan,
                    "parameters": _params_text(cfg, tuple(cfg.keys())),
                    "purpose": "visualization / local structure",
                    "status": f"skipped: {exc}",
                })

        report = pd.DataFrame(rows)
        if plot and created:
            visual_names = [n for n in created if self.embeddings_[n].shape[1] >= 2 and n != "pca_model"]
            if visual_names:
                self.plot_embeddings(
                    visual_names,
                    color_by=color_by,
                    cluster_labels=cluster_labels,
                    legend=legend,
                    cmap=cmap,
                    alpha=alpha,
                    s=point_size,
                )

        self.exploration_report_ = report
        return {
            "report": report,
            "created_embeddings": created,
            "skipped": skipped,
            "embeddings": {name: self.embeddings_[name] for name in created},
            "parameters": {name: self.embedding_parameters_[name] for name in created},
            "adaptive_defaults": {
                "enabled": adaptive_defaults,
                "n_samples": n_samples,
                "umap_n_neighbors": default_neighbors,
                "tsne_perplexity": default_perplexity,
            },
        }

    def _resolve_plot_colors(
        self,
        color_by: Optional[str] = "none",
        cluster_labels: Optional[np.ndarray] = None,
        categorical_max_unique: int = 20,
    ) -> Dict[str, Any]:
        """Resolve plotting colors from a label, cluster assignment or DataFrame column.

        Parameters
        ----------
        color_by:
            - ``None`` or ``"none"``: one color for all observations.
            - ``"validation_label"`` / ``"target"`` / ``"label"``: use ``self.y_``.
            - ``"cluster"``: use ``cluster_labels``.
            - any existing column name in ``self.df``: color by that original column.
        categorical_max_unique:
            Numeric columns with at most this many unique values are treated as
            categorical for legends. Larger numeric columns are treated as continuous.
        """
        key = "none" if color_by is None else str(color_by).strip()
        key_lower = key.lower()

        if key_lower in {"none", ""}:
            return {
                "values": None,
                "codes": None,
                "labels": None,
                "categories": None,
                "categorical": False,
                "source": "none",
            }

        if key_lower in {"target", "validation_label", "label"}:
            if self.y_ is None:
                raise ValueError("No hay validation_label disponible.")
            values = pd.Series(self.y_).reset_index(drop=True)
            source = "validation_label"
        elif key_lower == "cluster":
            if cluster_labels is None:
                raise ValueError("Para color_by='cluster' debes pasar cluster_labels.")
            values = pd.Series(np.asarray(cluster_labels)).reset_index(drop=True)
            source = "cluster"
        elif key in self.df.columns:
            values = self.df[key].reset_index(drop=True)
            source = key
        else:
            available = list(self.df.columns)
            raise ValueError(
                f"color_by='{color_by}' no es válido. Usa 'none', "
                "'validation_label', 'cluster' o una columna del DataFrame. "
                f"Columnas disponibles: {available}"
            )

        if len(values) != len(self.df):
            raise ValueError("La variable usada para color no coincide con el número de filas.")

        n_unique = int(values.nunique(dropna=False))
        is_categorical = (
            not is_numeric_dtype(values)
            or is_bool_dtype(values)
            or isinstance(values.dtype, pd.CategoricalDtype)
            or n_unique <= categorical_max_unique
        )

        if is_categorical:
            display_values = values.astype("object").where(values.notna(), "__NA__").astype(str)
            codes, uniques = pd.factorize(display_values, sort=True)
            return {
                "values": display_values.to_numpy(),
                "codes": codes,
                "labels": display_values.to_numpy(),
                "categories": [str(x) for x in uniques.tolist()],
                "categorical": True,
                "source": source,
            }

        numeric = pd.to_numeric(values, errors="coerce")
        return {
            "values": numeric.to_numpy(dtype=float),
            "codes": numeric.to_numpy(dtype=float),
            "labels": None,
            "categories": None,
            "categorical": False,
            "source": source,
        }

    @staticmethod
    def _add_categorical_legend(ax, scatter, categories: List[str], title: str) -> None:
        """Add a compact categorical legend using the scatter colormap."""
        if not categories:
            return
        from matplotlib.lines import Line2D

        cmap_obj = scatter.cmap
        norm_obj = scatter.norm
        handles = []
        for idx, category in enumerate(categories):
            handles.append(
                Line2D(
                    [0], [0],
                    marker="o",
                    linestyle="",
                    markerfacecolor=cmap_obj(norm_obj(idx)),
                    markeredgecolor="none",
                    markersize=7,
                    label=str(category),
                )
            )
        ax.legend(
            handles=handles,
            title=title,
            loc="best",
            frameon=True,
            fontsize="small",
            title_fontsize="small",
        )

    def plot_embeddings(
        self,
        embeddings: Optional[List[str]] = None,
        color_by: Optional[str] = "none",
        cluster_labels: Optional[np.ndarray] = None,
        figsize_per_plot: Tuple[float, float] = (5.5, 4.5),
        alpha: float = 0.72,
        s: int = 22,
        cmap: str = "viridis",
        legend: bool = True,
        categorical_max_unique: int = 20,
    ):
        """Muestra varios embeddings 2D lado a lado.

        ``color_by`` puede ser ``none``, ``validation_label``, ``cluster`` o
        cualquier columna del DataFrame original, por ejemplo ``color``.
        Las columnas categóricas y numéricas de baja cardinalidad reciben una
        leyenda; las numéricas continuas reciben una barra de color.
        """
        names = embeddings or [k for k, v in self.embeddings_.items() if v.shape[1] >= 2]
        if not names:
            raise ValueError("No hay embeddings 2D disponibles.")

        for name in names:
            if name not in self.embeddings_:
                raise ValueError(f"No existe embedding '{name}'.")
            if self.embeddings_[name].shape[1] < 2:
                raise ValueError(f"El embedding '{name}' no tiene al menos 2 dimensiones.")

        color_info = self._resolve_plot_colors(
            color_by=color_by,
            cluster_labels=cluster_labels,
            categorical_max_unique=categorical_max_unique,
        )

        n = len(names)
        fig, axes = plt.subplots(
            1, n,
            figsize=(figsize_per_plot[0] * n, figsize_per_plot[1]),
            squeeze=False,
        )
        axes = axes.ravel()

        for ax, name in zip(axes, names):
            z = self.embeddings_[name]
            scatter_kwargs = {"alpha": alpha, "s": s}
            if color_info["codes"] is not None:
                scatter_kwargs.update({"c": color_info["codes"], "cmap": cmap})

            scatter = ax.scatter(z.iloc[:, 0], z.iloc[:, 1], **scatter_kwargs)
            ax.set_title(name)
            ax.set_xlabel(z.columns[0])
            ax.set_ylabel(z.columns[1])

            if color_info["codes"] is not None:
                if color_info["categorical"] and legend:
                    self._add_categorical_legend(
                        ax,
                        scatter,
                        color_info["categories"],
                        title=str(color_info["source"]),
                    )
                elif not color_info["categorical"]:
                    cbar = fig.colorbar(scatter, ax=ax)
                    cbar.set_label(str(color_info["source"]))

        fig.tight_layout()
        plt.show()
        return fig

    @staticmethod
    def _rank_k_solutions(dfk: pd.DataFrame) -> pd.DataFrame:
        """Crea un ranking de consenso; menor ``consensus_rank`` es mejor."""
        out = dfk.copy()
        out["rank_silhouette"] = out["silhouette"].rank(ascending=False, method="min")
        out["rank_calinski"] = out["calinski_harabasz"].rank(ascending=False, method="min")
        out["rank_davies"] = out["davies_bouldin"].rank(ascending=True, method="min")
        out["consensus_rank"] = out[["rank_silhouette", "rank_calinski", "rank_davies"]].mean(axis=1)
        if "converged" in out.columns:
            out.loc[out["converged"] == False, "consensus_rank"] += 1000
        return out.sort_values(["consensus_rank", "k"]).reset_index(drop=True)

    def compare_spaces(
        self,
        spaces: Optional[List[str]] = None,
        method: str = "kmeans",
        k_range: range = range(2, 11),
        plot_each: bool = False,
        n_init: int = 50,
        max_iter: int = 500,
        tol: float = 1e-4,
        init: str = "k-means++",
        algorithm: str = "lloyd",
        random_state: Optional[int] = None,
        metric: str = "cityblock",
        include_external_validation: bool = True,
        show_interpretation: bool = True,
    ) -> Dict[str, Any]:
        """Compara el espacio de variables y embeddings creados.

        ``feature_space`` es la matriz original ya imputada, escalada y codificada.
        ``embed`` se mantiene como alias para compatibilidad.

        El mejor K por espacio se elige únicamente con métricas internas:
        Silhouette (máximo), Calinski-Harabasz (máximo) y Davies-Bouldin
        (mínimo). Si existe ``validation_label``, se reportan además ARI, NMI,
        homogeneity, completeness y V-measure, pero no alteran la selección
        interna del mejor K.
        """
        if spaces is None:
            spaces = ["feature_space"] + list(self.embeddings_.keys())
        if not spaces:
            raise ValueError("Debes indicar al menos un espacio.")

        method_key = (method or "").lower().strip()
        leaderboard_rows: List[Dict[str, Any]] = []
        details: Dict[str, pd.DataFrame] = {}
        skipped: Dict[str, str] = {}

        y_validation = None
        if include_external_validation and self.y_ is not None:
            y_validation = pd.Series(self.y_).reset_index(drop=True)

        for requested_space in spaces:
            canonical_space = self._canonical_space_name(requested_space)
            try:
                dfk = self.find_best_k(
                    X=canonical_space,
                    k_range=k_range,
                    method=method_key,
                    plot=plot_each,
                    n_init=n_init,
                    max_iter=max_iter,
                    tol=tol,
                    init=init,
                    algorithm=algorithm,
                    random_state=random_state,
                    metric=metric,
                )
                ranked = self._rank_k_solutions(dfk)
                details[canonical_space] = ranked
                best = ranked.iloc[0]
                best_k = int(best["k"])

                row = {
                    "space": canonical_space,
                    "method": method_key,
                    "best_k": best_k,
                    "silhouette": best.get("silhouette"),
                    "calinski_harabasz": best.get("calinski_harabasz"),
                    "davies_bouldin": best.get("davies_bouldin"),
                    "inertia": best.get("inertia"),
                    "n_iter": best.get("n_iter"),
                    "converged": best.get("converged"),
                    "consensus_rank": best.get("consensus_rank"),
                }

                # Validación externa opcional: nunca participa en la selección de K.
                if y_validation is not None:
                    Xmat = self._resolve_X(canonical_space)
                    Xvals = Xmat.values if isinstance(Xmat, pd.DataFrame) else np.asarray(Xmat)

                    if method_key == "kmeans":
                        model = KMeans(
                            n_clusters=best_k,
                            init=init,
                            n_init=n_init,
                            max_iter=max_iter,
                            tol=tol,
                            algorithm=algorithm,
                            random_state=random_state,
                        )
                        labels = model.fit_predict(Xvals)
                    elif method_key == "kmedoids":
                        try:
                            from sklearn_extra.cluster import KMedoids
                        except Exception as exc:
                            raise ImportError(
                                "Falta sklearn-extra para validar KMedoids. "
                                "Instala: pip install scikit-learn-extra"
                            ) from exc
                        model = KMedoids(
                            n_clusters=best_k,
                            metric=metric,
                            max_iter=max_iter,
                            random_state=random_state,
                        )
                        labels = model.fit_predict(Xvals)
                    else:
                        labels = None

                    if labels is not None:
                        mask = y_validation.notna().to_numpy()
                        y_true = y_validation.loc[mask].astype(str).to_numpy()
                        y_pred = np.asarray(labels)[mask]
                        row.update({
                            "validation_label": self.validation_label,
                            "validation_n_classes": int(pd.Series(y_true).nunique()),
                            "k_matches_validation": bool(best_k == pd.Series(y_true).nunique()),
                            "adjusted_rand_index": float(adjusted_rand_score(y_true, y_pred)),
                            "normalized_mutual_info": float(normalized_mutual_info_score(y_true, y_pred)),
                            "homogeneity": float(homogeneity_score(y_true, y_pred)),
                            "completeness": float(completeness_score(y_true, y_pred)),
                            "v_measure": float(v_measure_score(y_true, y_pred)),
                        })

                leaderboard_rows.append(row)
            except Exception as exc:
                skipped[str(requested_space)] = str(exc)

        leaderboard = pd.DataFrame(leaderboard_rows)
        if not leaderboard.empty:
            leaderboard = leaderboard.sort_values(
                ["silhouette", "davies_bouldin"],
                ascending=[False, True],
                na_position="last",
            ).reset_index(drop=True)

        interpretation = self._build_space_comparison_interpretation(
            leaderboard=leaderboard,
            external_validation_used=bool(y_validation is not None),
        )

        if show_interpretation:
            self._print_space_comparison_interpretation(interpretation)

        self.space_comparison_ = leaderboard
        self.space_comparison_interpretation_ = interpretation
        if interpretation.get("status") == "ok":
            self.best_space_ = str(interpretation.get("best_space_internal"))
            self.best_k_ = int(interpretation.get("best_k_internal"))
        return {
            "leaderboard": leaderboard,
            "details": details,
            "skipped": skipped,
            "space_aliases": {
                "feature_space": "Matriz original después de imputación, escalado y encoding.",
                "embed": "Alias legado de feature_space.",
            },
            "external_validation_used": bool(y_validation is not None),
            "interpretation": interpretation,
        }


    def _build_space_comparison_interpretation(
        self,
        leaderboard: pd.DataFrame,
        external_validation_used: bool,
    ) -> Dict[str, Any]:
        """Construye una lectura automática y prudente de compare_spaces().

        La interpretación no cambia el ranking ni selecciona el modelo final.
        Resume la relación entre métricas internas, granularidad y, cuando
        existe, la etiqueta externa de validación.
        """
        if leaderboard is None or leaderboard.empty:
            return {
                "status": "no_results",
                "summary": "No se generaron resultados válidos para comparar espacios.",
                "signals": [],
                "recommended_checks": [],
            }

        best_row = leaderboard.iloc[0]
        best_space = str(best_row.get("space"))
        best_k = int(best_row.get("best_k"))
        signals: List[str] = []
        checks: List[str] = []

        sil = best_row.get("silhouette")
        db = best_row.get("davies_bouldin")
        signals.append(
            f"El espacio mejor posicionado por métricas internas es '{best_space}' "
            f"con k={best_k}, silhouette={sil:.3f} y Davies-Bouldin={db:.3f}."
        )

        unique_k = sorted(pd.to_numeric(leaderboard["best_k"], errors="coerce").dropna().astype(int).unique().tolist())
        if len(unique_k) == 1:
            signals.append(
                f"Todos los espacios evaluados coinciden en k={unique_k[0]}, lo que aporta consistencia entre representaciones."
            )
        else:
            signals.append(
                f"Los espacios no coinciden en la granularidad óptima; los k sugeridos fueron {unique_k}."
            )

        if external_validation_used and "validation_n_classes" in leaderboard.columns:
            n_classes = int(pd.to_numeric(leaderboard["validation_n_classes"], errors="coerce").dropna().iloc[0])
            label_name = str(leaderboard["validation_label"].dropna().iloc[0])
            matches = leaderboard["k_matches_validation"].fillna(False).astype(bool)

            if not matches.any():
                signals.append(
                    f"La solución interna no coincide con las {n_classes} clases conocidas de '{label_name}'. "
                    "Esto puede indicar subestructura real, sobresegmentación o diferentes niveles válidos de granularidad."
                )
            else:
                matched_spaces = leaderboard.loc[matches, "space"].astype(str).tolist()
                signals.append(
                    f"Al menos un espacio recupera el mismo número de grupos que '{label_name}' ({n_classes}): {matched_spaces}."
                )

            hom = float(best_row.get("homogeneity", float("nan")))
            comp = float(best_row.get("completeness", float("nan")))
            ari = float(best_row.get("adjusted_rand_index", float("nan")))
            v = float(best_row.get("v_measure", float("nan")))

            if pd.notna(hom) and pd.notna(comp):
                if hom >= 0.90 and comp < 0.70:
                    signals.append(
                        "La homogeneidad es alta pero la completitud es baja: los clusters son puros respecto a la etiqueta, "
                        "pero cada clase conocida fue fragmentada en varios subclusters."
                    )
                    checks.append(
                        f"Comparar explícitamente k={n_classes} contra k={best_k} antes de elegir la granularidad final."
                    )
                elif hom < 0.70 and comp >= 0.90:
                    signals.append(
                        "La completitud es alta pero la homogeneidad es baja: cada clase conocida tiende a mantenerse unida, "
                        "pero los clusters mezclan clases diferentes."
                    )
                elif hom >= 0.90 and comp >= 0.90:
                    signals.append(
                        "Homogeneidad y completitud son altas, por lo que la estructura descubierta se alinea fuertemente con la etiqueta conocida."
                    )

            if pd.notna(ari):
                if ari >= 0.90:
                    signals.append(f"ARI={ari:.3f}: concordancia externa muy alta.")
                elif ari >= 0.50:
                    signals.append(f"ARI={ari:.3f}: concordancia externa moderada.")
                else:
                    signals.append(f"ARI={ari:.3f}: la partición interna difiere sustancialmente de la etiqueta conocida.")
            if pd.notna(v):
                signals.append(f"V-measure={v:.3f}, resumen del balance entre homogeneidad y completitud.")

            checks.append(
                "No usar la etiqueta de validación para entrenar; emplearla únicamente para contrastar e interpretar la solución."
            )
        else:
            signals.append(
                "No se proporcionó validation_label, por lo que la comparación se basa únicamente en métricas internas."
            )
            checks.append(
                "Validar estabilidad, tamaños de cluster, perfiles y utilidad de negocio antes de seleccionar la solución final."
            )

        checks.append(
            "La representación con mejor ranking no debe elegirse solo por verse mejor; debe confirmarse con estabilidad e interpretabilidad."
        )

        return {
            "status": "ok",
            "best_space_internal": best_space,
            "best_k_internal": best_k,
            "summary": signals[0],
            "signals": signals,
            "recommended_checks": checks,
        }

    @staticmethod
    def _print_space_comparison_interpretation(interpretation: Dict[str, Any]) -> None:
        """Imprime la interpretación en un formato legible para notebook."""
        print("\n========== SPACE COMPARISON INTERPRETATION ==========")
        if interpretation.get("status") != "ok":
            print(interpretation.get("summary", "Sin resultados."))
            return

        print("Signals:")
        for item in interpretation.get("signals", []):
            print(f"  • {item}")

        checks = interpretation.get("recommended_checks", [])
        if checks:
            print("Recommended checks:")
            for item in checks:
                print(f"  • {item}")

    def explain_space_comparison(self) -> Dict[str, Any]:
        """Vuelve a mostrar la última interpretación creada por compare_spaces()."""
        interpretation = getattr(self, "space_comparison_interpretation_", None)
        if interpretation is None:
            raise ValueError("Primero ejecuta compare_spaces().")
        self._print_space_comparison_interpretation(interpretation)
        return interpretation

    def run_unsupervised_workflow(
        self,
        prepare_kwargs: Optional[Dict[str, Any]] = None,
        explore_kwargs: Optional[Dict[str, Any]] = None,
        compare_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Orquestador opcional: prepare -> explore -> compare_spaces.

        No ajusta automáticamente una solución final, porque la selección debe
        incorporar estabilidad, perfiles y criterio de negocio.
        """
        prepared = self.prepare(**(prepare_kwargs or {}))
        explored = self.explore(**(explore_kwargs or {}))
        compared = self.compare_spaces(**(compare_kwargs or {}))
        return {"prepared": prepared, "explored": explored, "compared": compared}

    # ============================================================
    # 2) Clustering Router
    # ============================================================
    def cluster(
            self,
            method: str = "kmeans",
            X: str = "embed",
            name: Optional[str] = None,
            **kwargs
        ) -> ClusterSolution:
            """
            Router de clustering:
            - kmeans: usa cluster_kmeans (con n_init/max_iter)
            - kmedoids: usa cluster_kmedoids
            - dbscan: usa cluster_dbscan
            """
            method = method.lower().strip()

            if method == "kmeans":
                return self.cluster_kmeans(X=X, name=name, **kwargs)

            if method == "kmedoids":
                return self.cluster_kmedoids(X=X, name=name, **kwargs)

            if method == "dbscan":
                return self.cluster_dbscan(X=X, name=name, **kwargs)

            raise ValueError("method debe ser: 'kmeans', 'kmedoids' o 'dbscan'.")


    # ============================================================
    # 3) Plotting (matplotlib)
    # ============================================================
    def plot_embedding(
        self,
        embedding: str,
        color_by: Optional[str] = "validation_label",
        cluster_labels: Optional[np.ndarray] = None,
        title: Optional[str] = None,
        save: bool = True,
        filename: Optional[str] = None,
        alpha: float = 0.7,
        s: int = 18,
        cmap: str = "viridis",
        legend: bool = True,
        categorical_max_unique: int = 20,
    ):
        """Plotea un embedding 2D con color flexible.

        ``color_by`` puede ser:
        - ``"validation_label"`` / ``"target"``
        - ``"cluster"`` junto con ``cluster_labels``
        - cualquier columna del DataFrame original, por ejemplo ``"color"``
        - ``"none"`` para no colorear
        """
        requested = str(embedding).lower().strip()
        canonical = self._canonical_space_name(requested)

        if canonical == "feature_space":
            if self.X_embed_ is None:
                raise ValueError("No hay feature_space preparado. Ejecuta prepare() primero.")
            Z = self.X_embed_.copy() if hasattr(self.X_embed_, "copy") else np.asarray(self.X_embed_)
            if not isinstance(Z, pd.DataFrame):
                Z = pd.DataFrame(Z, columns=[f"Feature{i+1}" for i in range(np.asarray(Z).shape[1])])
        elif canonical in self.embeddings_:
            Z = self.embeddings_[canonical]
        else:
            raise ValueError(
                f"No existe espacio o embedding '{embedding}'. "
                f"Disponibles: ['feature_space', {', '.join(repr(k) for k in self.embeddings_.keys())}]"
            )

        if Z.shape[1] != 2:
            raise ValueError(
                f"El espacio '{canonical}' tiene {Z.shape[1]} dimensiones. "
                "Para graficarlo directamente debe tener exactamente 2. "
                "Usa solution.plot() sin embedding para crear una proyección PCA 2D automática, "
                "o crea un embedding visual con uns.reduce(..., n_components=2)."
            )

        color_info = self._resolve_plot_colors(
            color_by=color_by,
            cluster_labels=cluster_labels,
            categorical_max_unique=categorical_max_unique,
        )

        if title is None:
            title = f"{canonical.upper()} — colored by {color_info['source']}"

        fig, ax = plt.subplots(figsize=(8, 6))
        scatter_kwargs = {"alpha": alpha, "s": s}
        if color_info["codes"] is not None:
            scatter_kwargs.update({"c": color_info["codes"], "cmap": cmap})

        scatter = ax.scatter(Z.iloc[:, 0], Z.iloc[:, 1], **scatter_kwargs)
        ax.set_title(title)
        ax.set_xlabel(Z.columns[0])
        ax.set_ylabel(Z.columns[1])

        if color_info["codes"] is not None:
            if color_info["categorical"] and legend:
                self._add_categorical_legend(
                    ax,
                    scatter,
                    color_info["categories"],
                    title=str(color_info["source"]),
                )
            elif not color_info["categorical"]:
                cbar = fig.colorbar(scatter, ax=ax)
                cbar.set_label(str(color_info["source"]))

        fig.tight_layout()

        if save:
            if filename is None:
                safe_color = str(color_info["source"]).replace(" ", "_")
                filename = f"{embedding}_{safe_color}.png"
            path = os.path.join(self.output_dir, filename)
            fig.savefig(path, dpi=160)
        plt.show()
        return fig

    # ============================================================
    # 4) Resolver X: embed o embedding previo
    # ============================================================
    @staticmethod
    def _canonical_space_name(X: str) -> str:
        """Normaliza aliases del espacio preprocesado sin romper notebooks viejos."""
        key = str(X).lower().strip()
        if key in {"embed", "feature_space", "prepared_data", "preprocessed"}:
            return "feature_space"
        return key

    def _resolve_X(self, X: str) -> Union[pd.DataFrame, np.ndarray]:
        requested = str(X).lower().strip()
        key = self._canonical_space_name(requested)

        if key == "feature_space":
            if self.X_embed_ is None:
                raise ValueError("No hay feature_space preparado. Ejecuta prepare() primero.")
            return self.X_embed_

        # embedding previo
        if key in self.embeddings_:
            return self.embeddings_[key]

        raise ValueError(
            f"X='{X}' inválido. Usa 'feature_space' (alias legado: 'embed') "
            f"o uno de embeddings: {list(self.embeddings_.keys())}"
        )
    
    # ============================================================
    # 5) Metricas internas de clustering
    # ============================================================

    def _internal_metrics(self, Xvals: np.ndarray, labels: np.ndarray) -> Dict[str, Any]:
        out = {
            "silhouette": None,
            "calinski_harabasz": None,
            "davies_bouldin": None,
            "n_clusters": None,
            "n_noise": None
        }

        labels = np.asarray(labels)
        uniq = set(labels)

        n_noise = int(np.sum(labels == -1)) if (-1 in uniq) else 0
        n_clusters = int(len(uniq) - (1 if -1 in uniq else 0))

        out["n_clusters"] = n_clusters
        out["n_noise"] = n_noise

        try:
            if -1 in uniq:
                mask = labels != -1
                if n_clusters >= 2 and mask.sum() > 2:
                    out["silhouette"] = float(silhouette_score(Xvals[mask], labels[mask]))
                    out["calinski_harabasz"] = float(calinski_harabasz_score(Xvals[mask], labels[mask]))
                    out["davies_bouldin"] = float(davies_bouldin_score(Xvals[mask], labels[mask]))
            else:
                if n_clusters >= 2:
                    out["silhouette"] = float(silhouette_score(Xvals, labels))
                    out["calinski_harabasz"] = float(calinski_harabasz_score(Xvals, labels))
                    out["davies_bouldin"] = float(davies_bouldin_score(Xvals, labels))
        except Exception:
            pass

        return out
    
    # ============================================================
    # 6) Metodos de clustering
    # ============================================================
    
    def cluster_kmeans(
            self,
            space: Optional[str] = None,
            k: int = 3,
            init: str = "k-means++",
            n_init: int = 20,          # ✅ importante
            max_iter: int = 400,       # ✅ importante
            tol: float = 1e-4,
            algorithm: str = "lloyd",
            random_state: Optional[int] = None,
            name: Optional[str] = None,
            X: Optional[str] = None,  # legacy alias for space
        ) -> ClusterSolution:

            selected_space = self._resolve_space_argument(space=space, X=X)
            Xmat = self._resolve_X(selected_space)
            Xvals = Xmat.values if isinstance(Xmat, pd.DataFrame) else np.asarray(Xmat)

            model = KMeans(
                n_clusters=k,
                init=init,
                n_init=n_init,
                max_iter=max_iter,
                tol=tol,
                algorithm=algorithm,
                random_state=random_state
            )
            labels = model.fit_predict(Xvals)

            if name is None:
                name = f"kmeans_k{k}_on_{selected_space}"

            n_iter_used = int(getattr(model, "n_iter_", -1))
            converged = n_iter_used >= 0 and n_iter_used < max_iter
            if not converged:
                warnings.warn(
                    f"KMeans alcanzó max_iter={max_iter}; revisa convergencia, escalado o tolerancia.",
                    RuntimeWarning
                )

            res = {
                "labels": labels,
                "model": model,
                "method": "kmeans",
                "X_used": selected_space,
                "k": k,
                "inertia": float(model.inertia_),
                "n_iter": n_iter_used,
                "max_iter": int(max_iter),
                "converged": bool(converged),
                "n_init": int(n_init),
                "init": init,
                "tol": float(tol),
                "algorithm": algorithm,
                "random_state": random_state,
            }

            res.update(self._internal_metrics(Xvals, labels))

            solution = ClusterSolution(self, name=name, result=res)
            self.cluster_results_[name] = solution
            self.last_solution_ = solution
            return solution
    
    def cluster_kmedoids(
            self,
            space: Optional[str] = None,
            k: int = 3,
            metric: str = "cityblock",  # como tu ejemplo
            max_iter: int = 400,
            init: str = "k-medoids++",
            random_state: Optional[int] = None,
            name: Optional[str] = None,
            X: Optional[str] = None,  # legacy alias for space
            **kwargs
        ) -> ClusterSolution:

            selected_space = self._resolve_space_argument(space=space, X=X)
            try:
                from sklearn_extra.cluster import KMedoids
            except Exception as e:
                raise ImportError("Falta sklearn-extra. Instala: pip install scikit-learn-extra") from e

            Xmat = self._resolve_X(selected_space)
            Xvals = Xmat.values if isinstance(Xmat, pd.DataFrame) else np.asarray(Xmat)

            model = KMedoids(
                n_clusters=k,
                metric=metric,
                max_iter=max_iter,
                init=init,
                random_state=random_state,
                **kwargs
            )
            labels = model.fit_predict(Xvals)

            if name is None:
                name = f"kmedoids_k{k}_on_{selected_space}"

            res = {
                "labels": labels,
                "model": model,
                "method": "kmedoids",
                "X_used": selected_space,
                "k": k,
                "metric": metric
            }

            res.update(self._internal_metrics(Xvals, labels))

            solution = ClusterSolution(self, name=name, result=res)
            self.cluster_results_[name] = solution
            self.last_solution_ = solution
            return solution
    
    def cluster_dbscan(
            self,
            X: str = "embed",
            eps: float = 0.5,
            min_samples: int = 5,
            metric: str = "euclidean",
            name: Optional[str] = None,
            **kwargs
        ) -> ClusterSolution:

            Xmat = self._resolve_X(X)
            Xvals = Xmat.values if isinstance(Xmat, pd.DataFrame) else np.asarray(Xmat)

            model = DBSCAN(eps=eps, min_samples=min_samples, metric=metric, **kwargs)
            labels = model.fit_predict(Xvals)

            if name is None:
                name = f"dbscan_on_{X}"

            res = {
                "labels": labels,
                "model": model,
                "method": "dbscan",
                "X_used": X,
                "eps": eps,
                "min_samples": min_samples,
                "metric": metric
            }

            res.update(self._internal_metrics(Xvals, labels))

            solution = ClusterSolution(self, name=name, result=res)
            self.cluster_results_[name] = solution
            self.last_solution_ = solution
            return solution
    
    def get_solution(self, name: Optional[str] = None) -> ClusterSolution:
        """Recupera una solución guardada; sin nombre devuelve la última."""
        if name is None:
            if self.last_solution_ is None:
                raise ValueError("Todavía no existe una solución de clustering.")
            return self.last_solution_
        if name not in self.cluster_results_:
            raise ValueError(f"No existe la solución '{name}'. Disponibles: {list(self.cluster_results_)}")
        return self.cluster_results_[name]

    def get_clustered_dataset(
        self,
        name: Optional[str] = None,
        cluster_column: str = "cluster",
        include_embedding: bool = False,
    ) -> pd.DataFrame:
        """Atajo: devuelve el dataset original con sus clusters."""
        return self.get_solution(name).dataset(
            cluster_column=cluster_column,
            include_embedding=include_embedding,
        )

    def find_best_k(
            self,
            X: str = "embed",
            k_range: range = range(2, 11),
            method: str = "kmeans",      # "kmeans" o "kmedoids"
            plot: bool = True,
            # kmeans
            n_init: int = 20,
            max_iter: int = 400,
            tol: float = 1e-4,
            init: str = "k-means++",
            algorithm: str = "lloyd",
            random_state: Optional[int] = None,
            # kmedoids
            metric: str = "cityblock"
        ) -> pd.DataFrame:

            method = method.lower().strip()
            rows = []
            previous_last_solution = self.last_solution_

            for k in k_range:
                if method == "kmeans":
                    res = self.cluster_kmeans(
                        X=X, k=k, n_init=n_init, max_iter=max_iter, tol=tol,
                        init=init, algorithm=algorithm, random_state=random_state,
                        name=f"_tmp_kmeans_{k}"
                    )
                    inertia = res.get("inertia", None)

                elif method == "kmedoids":
                    res = self.cluster_kmedoids(
                        X=X, k=k, metric=metric, max_iter=max_iter,
                        random_state=random_state, name=f"_tmp_kmedoids_{k}"
                    )
                    inertia = None
                else:
                    raise ValueError("method debe ser 'kmeans' o 'kmedoids'")

                rows.append({
                    "k": k,
                    "silhouette": res.get("silhouette"),
                    "calinski_harabasz": res.get("calinski_harabasz"),
                    "davies_bouldin": res.get("davies_bouldin"),
                    "inertia": inertia,
                    "n_iter": res.get("n_iter"),
                    "max_iter": res.get("max_iter", max_iter),
                    "converged": res.get("converged"),
                    "n_init": res.get("n_init"),
                })

            dfk = pd.DataFrame(rows)

            # limpiar temporales
            for k in k_range:
                self.cluster_results_.pop(f"_tmp_kmeans_{k}", None)
                self.cluster_results_.pop(f"_tmp_kmedoids_{k}", None)
            self.last_solution_ = previous_last_solution

            if plot:
                self._plot_k_search(dfk, method=method, X=X)

            return dfk

    def evaluate_stability(
        self,
        X: str = "embed",
        k: int = 3,
        n_runs: int = 10,
        n_init: int = 20,
        max_iter: int = 400,
        tol: float = 1e-4,
        init: str = "k-means++",
        algorithm: str = "lloyd",
        master_seed: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Evalúa estabilidad de K-Means entre inicializaciones independientes.

        ``master_seed`` solo vuelve reproducible el experimento completo; no
        convierte una semilla particular en parte de la solución final.
        """
        if n_runs < 2:
            raise ValueError("n_runs debe ser al menos 2.")

        previous_last_solution = self.last_solution_
        rng = np.random.default_rng(master_seed)
        seeds = rng.integers(0, np.iinfo(np.int32).max, size=n_runs)
        runs = []
        labels_list = []

        for i, seed in enumerate(seeds):
            res = self.cluster_kmeans(
                X=X, k=k, init=init, n_init=n_init, max_iter=max_iter,
                tol=tol, algorithm=algorithm, random_state=int(seed),
                name=f"_stability_{i}"
            )
            labels_list.append(np.asarray(res["labels"]))
            runs.append({
                "run": i,
                "seed": int(seed),
                "silhouette": res.get("silhouette"),
                "inertia": res.get("inertia"),
                "n_iter": res.get("n_iter"),
                "converged": res.get("converged"),
            })
            self.cluster_results_.pop(f"_stability_{i}", None)

        self.last_solution_ = previous_last_solution
        pairwise_ari = [
            adjusted_rand_score(labels_list[i], labels_list[j])
            for i in range(n_runs) for j in range(i + 1, n_runs)
        ]
        runs_df = pd.DataFrame(runs)
        summary = {
            "k": k,
            "X_used": X,
            "n_runs": n_runs,
            "silhouette_mean": float(runs_df["silhouette"].mean()),
            "silhouette_std": float(runs_df["silhouette"].std(ddof=1)),
            "inertia_mean": float(runs_df["inertia"].mean()),
            "inertia_std": float(runs_df["inertia"].std(ddof=1)),
            "mean_pairwise_ari": float(np.mean(pairwise_ari)),
            "min_pairwise_ari": float(np.min(pairwise_ari)),
            "all_converged": bool(runs_df["converged"].all()),
            "runs": runs_df,
            "pairwise_ari": np.asarray(pairwise_ari),
        }
        return summary

    def _plot_k_search(self, dfk: pd.DataFrame, method: str, X: str):
        k = dfk["k"].values

        has_inertia = ("inertia" in dfk.columns and dfk["inertia"].notna().any())
        n_plots = 4 if has_inertia else 3
        fig, axes = plt.subplots(1, n_plots, figsize=(4.6*n_plots, 3.6), dpi=130)

        ax_i = 0
        if has_inertia:
            axes[ax_i].plot(k, dfk["inertia"], marker="o")
            axes[ax_i].set_title(f"Elbow (inertia)\n{method} on {X}")
            axes[ax_i].set_xlabel("k")
            axes[ax_i].set_ylabel("inertia (↓)")
            ax_i += 1

        axes[ax_i].plot(k, dfk["silhouette"], marker="o")
        axes[ax_i].set_title("Silhouette (↑)")
        axes[ax_i].set_xlabel("k")
        ax_i += 1

        axes[ax_i].plot(k, dfk["calinski_harabasz"], marker="o")
        axes[ax_i].set_title("Calinski-Harabasz (↑)")
        axes[ax_i].set_xlabel("k")
        ax_i += 1

        axes[ax_i].plot(k, dfk["davies_bouldin"], marker="o")
        axes[ax_i].set_title("Davies-Bouldin (↓)")
        axes[ax_i].set_xlabel("k")

        plt.tight_layout()
        plt.show()

    def profile_clusters(
        self,
        labels: Union[np.ndarray, pd.Series],
        top_n_categories: int = 5,
        include_target: bool = True
    ) -> Dict[str, Any]:

        labels = np.asarray(labels)
        dfp = self.df.copy()
        dfp["_cluster"] = labels

        out: Dict[str, Any] = {}

        # tamaños
        out["sizes"] = dfp["_cluster"].value_counts().sort_index().to_frame("count")

        # numéricas
        num_cols = self.cols.numeric
        if num_cols:
            out["numeric_means"] = dfp.groupby("_cluster")[num_cols].mean()
            out["numeric_medians"] = dfp.groupby("_cluster")[num_cols].median()

        # categóricas (top niveles)
        cat_cols = [c for c in self.cols.categorical if c != self.target]
        cat_summary = {}
        for c in cat_cols:
            s = dfp[c].astype("object").fillna("__NA__")
            top_levels = s.value_counts().head(top_n_categories).index
            tab = pd.crosstab(dfp["_cluster"], s, normalize="index") * 100
            cat_summary[c] = tab.loc[:, tab.columns.intersection(top_levels)].round(2)
        out["categorical_pct"] = cat_summary

        # target vs cluster
        if include_target and self.target is not None and self.target in dfp.columns:
            t = dfp[self.target].astype("object").fillna("__NA__")
            out["target_by_cluster"] = (pd.crosstab(dfp["_cluster"], t, normalize="index") * 100).round(2)

        return out

    # ============================================================
    # 7) Cluster visualization (interpretability)
    # ============================================================
    def plot_cluster_bars(
        self,
        centroids: pd.DataFrame,
        normalize: bool = True,
        top_n: Optional[int] = None,
        clusters: Optional[List[int]] = None,
        title: str = "Cluster centroids (bar)",
        figsize: Optional[tuple] = None,
    ):
        """
        Bar plots horizontales por cluster.

        centroids: DataFrame index=cluster, cols=variables (ej: numeric_means)
        normalize:
            - True: escala cada variable a [0,1] usando max abs global por columna (comparables)
            - False: usa valores originales
        top_n:
            - si se pasa, usa solo las top_n variables con mayor |media| promedio (para legibilidad)
        clusters:
            - lista de clusters a mostrar; si None, muestra todos.
        """
        if centroids is None or centroids.empty:
            raise ValueError("centroids está vacío.")

        C = centroids.copy()

        # Elegir clusters
        if clusters is not None:
            C = C.loc[clusters]

        # Selección de variables para no explotar el plot
        if top_n is not None and top_n < C.shape[1]:
            score = C.abs().mean(axis=0).sort_values(ascending=False)
            vars_keep = score.head(top_n).index
            C = C[vars_keep]

        # Normalización por columna (max abs) para comparar magnitudes
        if normalize:
            denom = C.abs().max(axis=0).replace(0, 1.0)
            C = C / denom

        n_clusters = C.shape[0]
        if figsize is None:
            figsize = (4.5 * n_clusters, max(4.0, 0.25 * C.shape[1] + 1.5))

        fig, axes = plt.subplots(1, n_clusters, figsize=figsize, dpi=130, sharey=True)
        if n_clusters == 1:
            axes = [axes]

        for ax, (cl, row) in zip(axes, C.iterrows()):
            row_sorted = row.sort_values()
            ax.barh(row_sorted.index, row_sorted.values)
            ax.set_title(f"Cluster {cl}")
            ax.axvline(0, linewidth=1)
            ax.set_xlabel("scaled" if normalize else "value")

        fig.suptitle(title, y=1.02, fontsize=14, fontweight="bold")
        plt.tight_layout()
        plt.show()

    def plot_cluster_radar(
        self,
        centroids: pd.DataFrame,
        normalize: str = "minmax",   # "minmax" | "maxabs" | "none"
        vars: Optional[List[str]] = None,
        top_n: Optional[int] = 10,
        clusters: Optional[List[int]] = None,
        title: str = "Cluster centroids (radar)",
        figsize: tuple = (8, 8),
    ):
        """
        Radar plot para centroides por cluster.

        normalize:
        - "minmax": por variable -> (x - min)/(max - min) en [0,1]
        - "maxabs": por variable -> x / max(abs(x))  (puede dejar negativos)
        - "none": sin normalizar (no recomendado si escalas distintas)
        vars:
        - lista explícita de variables a graficar.
        top_n:
        - si vars es None, elige top_n variables por |media| promedio.
        """
        if centroids is None or centroids.empty:
            raise ValueError("centroids está vacío.")

        C = centroids.copy()

        # Subset clusters
        if clusters is not None:
            C = C.loc[clusters]

        # Selección de variables
        if vars is not None:
            C = C[vars]
        else:
            if top_n is not None and top_n < C.shape[1]:
                score = C.abs().mean(axis=0).sort_values(ascending=False)
                C = C[score.head(top_n).index]

        # Normalización
        if normalize == "minmax":
            vmin = C.min(axis=0)
            vmax = C.max(axis=0)
            denom = (vmax - vmin).replace(0, 1.0)
            Cn = (C - vmin) / denom
        elif normalize == "maxabs":
            denom = C.abs().max(axis=0).replace(0, 1.0)
            Cn = C / denom
            # para radar, si hay negativos, los desplazamos a [0,1] por estética
            vmin = Cn.min(axis=0)
            vmax = Cn.max(axis=0)
            denom2 = (vmax - vmin).replace(0, 1.0)
            Cn = (Cn - vmin) / denom2
        elif normalize == "none":
            Cn = C.copy()
        else:
            raise ValueError("normalize debe ser: 'minmax', 'maxabs', 'none'.")

        labels_vars = list(Cn.columns)
        n_vars = len(labels_vars)

        angles = np.linspace(0, 2*np.pi, n_vars, endpoint=False).tolist()
        angles += angles[:1]  # cerrar

        fig = plt.figure(figsize=figsize, dpi=130)
        ax = plt.subplot(111, polar=True)
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)

        plt.xticks(angles[:-1], labels_vars)

        ax.set_rlabel_position(0)
        ax.set_ylim(0, 1)

        for cl, row in Cn.iterrows():
            values = row.values.tolist()
            values += values[:1]
            ax.plot(angles, values, linewidth=1, label=f"Cluster {cl}")
            ax.fill(angles, values, alpha=0.25)

        plt.title(title, y=1.08)
        plt.legend(loc="upper right", bbox_to_anchor=(1.25, 1.15))
        plt.tight_layout()
        plt.show()


    # ============================================================
    # 10) Registry, recommendations and clustering competition
    # ============================================================
    def add_default_cluster_models(self) -> Dict[str, ClusterModelSpec]:
        """Register the default clustering families and their search metadata."""
        self.cluster_model_specs_ = {
            "kmeans": ClusterModelSpec(
                name="kmeans", family="centroid", requires_k=True,
                supports_predict=True, supports_noise=False,
                parameter_grid={"k": range(2, 11)},
                recommended_for=("scaled numeric data", "compact clusters", "large datasets"),
                limitations=("requires k", "sensitive to outliers and non-spherical geometry"),
            ),
            "kmedoids": ClusterModelSpec(
                name="kmedoids", family="medoid", requires_k=True,
                supports_predict=True, supports_noise=False,
                parameter_grid={"k": range(2, 11), "metric": ["euclidean", "manhattan"]},
                recommended_for=("outlier-prone data", "non-euclidean distances", "small/medium datasets"),
                limitations=("slower than kmeans", "optional sklearn-extra dependency"),
                optional_dependency="sklearn-extra",
            ),
            "agglomerative": ClusterModelSpec(
                name="agglomerative", family="hierarchical", requires_k=True,
                supports_predict=False, supports_noise=False,
                parameter_grid={"k": range(2, 11), "linkage": ["ward", "complete", "average"]},
                recommended_for=("small/medium datasets", "hierarchical structure", "deterministic baseline"),
                limitations=("no native predict", "memory cost grows with sample size"),
            ),
            "gaussian_mixture": ClusterModelSpec(
                name="gaussian_mixture", family="probabilistic", requires_k=True,
                supports_predict=True, supports_noise=False,
                parameter_grid={"k": range(2, 11), "covariance_type": ["full", "diag"]},
                recommended_for=("elliptical clusters", "overlapping segments", "soft membership"),
                limitations=("distributional assumptions", "can converge to local optima"),
            ),
            "dbscan": ClusterModelSpec(
                name="dbscan", family="density", requires_k=False,
                supports_predict=False, supports_noise=True,
                parameter_grid={},
                recommended_for=("irregular shapes", "noise detection", "unknown number of clusters"),
                limitations=("sensitive to eps", "struggles with varying density"),
            ),
            "hdbscan": ClusterModelSpec(
                name="hdbscan", family="density", requires_k=False,
                supports_predict=True, supports_noise=True,
                parameter_grid={"min_cluster_size": [5, 10, 15]},
                recommended_for=("varying densities", "noise detection", "unknown number of clusters"),
                limitations=("optional dependency", "small samples may be unstable"),
                optional_dependency="hdbscan",
            ),
        }
        return self.cluster_model_specs_

    def available_cluster_models(self) -> pd.DataFrame:
        rows = []
        for name, spec in self.cluster_model_specs_.items():
            available = True
            note = ""
            if name == "kmedoids":
                try:
                    import sklearn_extra  # noqa: F401
                except Exception:
                    available, note = False, "pip install scikit-learn-extra"
            elif name == "hdbscan":
                try:
                    import hdbscan  # noqa: F401
                except Exception:
                    available, note = False, "pip install hdbscan"
            rows.append({
                "model": name, "family": spec.family, "available": available,
                "requires_k": spec.requires_k, "supports_noise": spec.supports_noise,
                "recommended_for": "; ".join(spec.recommended_for),
                "limitations": "; ".join(spec.limitations), "note": note,
            })
        return pd.DataFrame(rows)

    def _resolve_space_argument(
        self,
        space: Optional[str] = None,
        X: Optional[str] = None,
        default: str = "feature_space",
    ) -> str:
        """Resolve the representation used by downstream clustering methods.

        ``space`` is the preferred public name. ``X`` remains a legacy alias.
        If neither is supplied, the best space from the most recent
        ``compare_spaces()`` call is used; otherwise ``default`` is used.
        """
        if space is not None and X is not None:
            if self._canonical_space_name(space) != self._canonical_space_name(X):
                raise ValueError("Usa 'space' o el alias legado 'X', pero no valores diferentes en ambos.")
        selected = space if space is not None else X
        if selected is None:
            selected = self.best_space_ or default
        return self._canonical_space_name(str(selected))

    @property
    def best_space(self) -> Optional[str]:
        """Best representation selected by the latest ``compare_spaces()`` call."""
        return self.best_space_

    @property
    def best_k(self) -> Optional[int]:
        """Best K selected by the latest ``compare_spaces()`` call."""
        return self.best_k_

    def recommend_cluster_models(
        self,
        space: Optional[str] = None,
        top_n: int = 6,
        X: Optional[str] = None,
    ) -> pd.DataFrame:
        """Prioritize clustering families for a representation.

        Parameters
        ----------
        space:
            Preferred representation name, for example ``feature_space``,
            ``pca_model`` or ``umap_visual``. If omitted, uses the winner from
            the latest ``compare_spaces()`` call.
        X:
            Legacy alias for ``space``.
        """
        selected_space = self._resolve_space_argument(space=space, X=X)
        Xmat = self._resolve_X(selected_space)
        Xvals = Xmat.values if isinstance(Xmat, pd.DataFrame) else np.asarray(Xmat)
        n, d = Xvals.shape
        report = self.preprocessing_report_ or {}
        robust_cols = len(report.get("numeric", {}).get("robust", []))
        scores = {
            "kmeans": 0.0,
            "kmedoids": 0.0,
            "agglomerative": 0.0,
            "gaussian_mixture": 0.0,
            "dbscan": 0.0,
            "hdbscan": 0.0,
        }
        reasons = {k: [] for k in scores}

        if n <= 3000:
            scores["agglomerative"] += 2; reasons["agglomerative"].append("dataset pequeño/mediano")
            scores["kmedoids"] += 1; reasons["kmedoids"].append("costo viable para este tamaño")
        else:
            scores["kmeans"] += 2; reasons["kmeans"].append("escala bien a muchas filas")
        if d <= 20:
            scores["kmeans"] += 1; scores["gaussian_mixture"] += 1
            reasons["kmeans"].append("espacio de baja dimensión")
            reasons["gaussian_mixture"].append("covarianzas estimables")
        if robust_cols > 0:
            scores["kmedoids"] += 2; reasons["kmedoids"].append("se detectaron variables robust-scaled/outliers")
            scores["dbscan"] += 1; reasons["dbscan"].append("puede identificar observaciones aisladas")
        if n >= 50:
            scores["dbscan"] += 1; scores["hdbscan"] += 1
            reasons["dbscan"].append("muestra suficiente para explorar densidad")
            reasons["hdbscan"].append("puede explorar densidades variables")
        scores["kmeans"] += 1
        reasons["kmeans"].append("baseline interpretable")
        scores["gaussian_mixture"] += 1
        reasons["gaussian_mixture"].append("evalúa pertenencia probabilística")

        availability = self.available_cluster_models().set_index("model")
        rows = []
        for model, score in scores.items():
            rows.append({
                "space": selected_space,
                "model": model,
                "priority_score": score,
                "available": bool(availability.loc[model, "available"]),
                "reason": "; ".join(reasons[model]) or "modelo complementario",
                "family": self.cluster_model_specs_[model].family,
            })
        out = pd.DataFrame(rows).sort_values(["available", "priority_score"], ascending=[False, False])
        return out.head(top_n).reset_index(drop=True)

    def _display_long_text_table(
        self, df: pd.DataFrame, text_columns: Optional[List[str]] = None,
        precision: int = 3, title: Optional[str] = None,
    ):
        """Notebook-friendly display that wraps long-text columns."""
        text_columns = text_columns or []
        if title:
            print(title)
        try:
            from IPython.display import display
            sty = df.style.format(precision=precision)
            for col in text_columns:
                if col in df.columns:
                    sty = sty.set_properties(
                        subset=[col],
                        **{
                            "white-space": "pre-wrap",
                            "text-align": "left",
                            "min-width": "280px",
                            "max-width": "520px",
                        },
                    )
            display(sty)
            return sty
        except Exception:
            with pd.option_context("display.max_colwidth", None, "display.width", 220):
                print(df)
            return None

    def explain_cluster_model_recommendations(
        self,
        space: Optional[str] = None,
        top_n: int = 6,
        available_only: bool = False,
        X: Optional[str] = None,
    ) -> pd.DataFrame:
        """Show recommendations as readable blocks instead of truncated cells."""
        selected_space = self._resolve_space_argument(space=space, X=X)
        rec = self.recommend_cluster_models(space=selected_space, top_n=top_n)
        print(f"Representation: {selected_space}")
        if available_only:
            rec = rec[rec["available"]].copy()
        for _, row in rec.iterrows():
            print("=" * 64)
            print(str(row["model"]).upper())
            print(f"Priority : {float(row['priority_score']):.1f}")
            print(f"Family   : {row['family']}")
            print(f"Available: {'Yes' if bool(row['available']) else 'No'}")
            print("Reasons:")
            for reason in [x.strip() for x in str(row["reason"]).split(";") if x.strip()]:
                print(f"  • {reason}")
        print("=" * 64)
        return rec.reset_index(drop=True)

    def build_experiment(
        self,
        space: Optional[str] = None,
        spaces: Optional[List[str]] = None,
        models: Union[str, List[str], Tuple[str, ...]] = "recommended",
        k_range=range(2, 11),
        evaluate_stability: bool = True,
        n_stability_runs: int = 5,
        min_cluster_pct: float = 2.0,
        max_noise_pct: float = 35.0,
        n_init: int = 50,
        max_iter: int = 500,
        tol: float = 1e-4,
        random_state: Optional[int] = 42,
        master_seed: int = 123,
        custom_grids: Optional[Dict[str, List[Dict[str, Any]]]] = None,
        include_exploratory: bool = True,
        verbose: bool = True,
        X: Optional[str] = None,  # legacy alias for space
    ) -> ClusteringExperimentPlan:
        """Build a transparent model-family experiment plan.

        This method recommends what to test and why; it does not use validation
        metrics and therefore does not select a winner. Call ``plan.run()`` to
        execute the actual competition.
        """
        if spaces is None:
            spaces = [self._resolve_space_argument(space=space, X=X)]
        else:
            if space is not None or X is not None:
                raise ValueError("Usa 'spaces' para varias representaciones o 'space' para una sola, no ambos.")
            spaces = [self._canonical_space_name(s) for s in spaces]
        for selected_space in spaces:
            self._resolve_X(selected_space)

        rec = self.recommend_cluster_models(space=spaces[0], top_n=len(self.cluster_model_specs_))
        availability = self.available_cluster_models().set_index("model")

        if isinstance(models, str):
            if models == "recommended":
                selected = rec.loc[rec["available"], "model"].tolist()
                if not include_exploratory:
                    selected = rec.loc[(rec["available"]) & (rec["priority_score"] >= 2), "model"].tolist()
            elif models == "all":
                selected = availability.loc[availability["available"], :].index.tolist()
            else:
                selected = [models.lower().strip()]
        else:
            selected = [str(m).lower().strip() for m in models]

        role_map = {
            "kmeans": ("baseline", "Establish a fast, interpretable centroid benchmark."),
            "kmedoids": ("robust alternative", "Test a medoid solution that is less sensitive to extreme observations."),
            "agglomerative": ("hierarchical alternative", "Test whether a nested or linkage-based structure better represents the sample."),
            "gaussian_mixture": ("probabilistic alternative", "Test elliptical clusters and soft membership probabilities."),
            "dbscan": ("density/noise diagnostic", "Test irregular shapes and whether isolated observations should be treated as noise."),
            "hdbscan": ("variable-density diagnostic", "Explore density-varying groups and noise without fixing k."),
        }
        preferred_order = ["kmeans", "kmedoids", "agglomerative", "gaussian_mixture", "dbscan", "hdbscan"]
        selected = [m for m in preferred_order if m in selected] + [m for m in selected if m not in preferred_order]

        rows = []
        rec_idx = rec.set_index("model")
        for step, model in enumerate(selected, start=1):
            if model not in self.cluster_model_specs_:
                raise ValueError(f"Modelo desconocido en experiment plan: {model}")
            spec = self.cluster_model_specs_[model]
            r = rec_idx.loc[model] if model in rec_idx.index else None
            score = float(r["priority_score"]) if r is not None else 0.0
            if score >= 3:
                label = "High"
            elif score >= 2:
                label = "Recommended"
            elif score >= 1:
                label = "Exploratory"
            else:
                label = "Low"
            available = bool(availability.loc[model, "available"])
            reason_text = str(r["reason"]) if r is not None else "complementary model family"
            role, purpose = role_map.get(model, (spec.family, "Complement the experiment with a different clustering assumption."))
            rows.append({
                "step": step,
                "role": role,
                "model": model,
                "family": spec.family,
                "priority_score": score,
                "priority_label": label,
                "available": available,
                "reasons": [x.strip() for x in reason_text.split(";") if x.strip()],
                "reason": "\n• ".join([x.strip() for x in reason_text.split(";") if x.strip()]),
                "experiment_purpose": purpose,
                "requires_k": spec.requires_k,
                "supports_noise": spec.supports_noise,
                "note": str(availability.loc[model, "note"] or ""),
            })

        plan_df = pd.DataFrame(rows)
        config = {
            "spaces": list(spaces),
            "k_range": list(k_range),
            "custom_grids": custom_grids,
            "evaluate_stability": evaluate_stability,
            "n_stability_runs": n_stability_runs,
            "min_cluster_pct": min_cluster_pct,
            "max_noise_pct": max_noise_pct,
            "n_init": n_init,
            "max_iter": max_iter,
            "tol": tol,
            "random_state": random_state,
            "master_seed": master_seed,
            "verbose": verbose,
            "models": plan_df.loc[plan_df["available"], "model"].tolist(),
        }
        plan = ClusteringExperimentPlan(
            analysis=self, plan_df=plan_df, recommendations_df=rec, config=config
        )
        self.last_clustering_experiment_ = plan
        if verbose:
            cols = ["step", "role", "model", "priority_label", "available", "reason", "experiment_purpose"]
            self._display_long_text_table(
                plan_df[cols], text_columns=["reason", "experiment_purpose"],
                title="========== CLUSTERING EXPERIMENT PLAN ==========",
            )
            print("Note: this is a test plan, not a model ranking. Use plan.run() for the competition.")
        return plan

    def cluster_agglomerative(
        self, space: Optional[str] = None, k: int = 3, linkage: str = "ward",
        metric: str = "euclidean", name: Optional[str] = None,
        X: Optional[str] = None,
    ) -> ClusterSolution:
        selected_space = self._resolve_space_argument(space=space, X=X)
        Xmat = self._resolve_X(selected_space)
        Xvals = Xmat.values if isinstance(Xmat, pd.DataFrame) else np.asarray(Xmat)
        if linkage == "ward" and metric != "euclidean":
            raise ValueError("linkage='ward' requiere metric='euclidean'.")
        kwargs = {"n_clusters": k, "linkage": linkage}
        # sklearn changed affinity -> metric; current v1.6 accepts metric.
        kwargs["metric"] = metric
        model = AgglomerativeClustering(**kwargs)
        labels = model.fit_predict(Xvals)
        name = name or f"agglomerative_k{k}_{linkage}_on_{selected_space}"
        res = {"labels": labels, "model": model, "method": "agglomerative", "X_used": selected_space,
               "k": k, "linkage": linkage, "metric": metric, "converged": True}
        res.update(self._internal_metrics(Xvals, labels))
        solution = ClusterSolution(self, name=name, result=res)
        self.cluster_results_[name] = solution; self.last_solution_ = solution
        return solution

    def cluster_gaussian_mixture(
        self, space: Optional[str] = None, k: int = 3, covariance_type: str = "full",
        n_init: int = 10, max_iter: int = 500, tol: float = 1e-3,
        reg_covar: float = 1e-6, random_state: Optional[int] = None,
        name: Optional[str] = None, X: Optional[str] = None,
    ) -> ClusterSolution:
        selected_space = self._resolve_space_argument(space=space, X=X)
        Xmat = self._resolve_X(selected_space)
        Xvals = Xmat.values if isinstance(Xmat, pd.DataFrame) else np.asarray(Xmat)
        model = GaussianMixture(
            n_components=k, covariance_type=covariance_type, n_init=n_init,
            max_iter=max_iter, tol=tol, reg_covar=reg_covar,
            random_state=random_state,
        )
        labels = model.fit_predict(Xvals)
        name = name or f"gmm_k{k}_{covariance_type}_on_{selected_space}"
        res = {"labels": labels, "model": model, "method": "gaussian_mixture", "X_used": X,
               "k": k, "covariance_type": covariance_type, "n_init": n_init,
               "max_iter": max_iter, "n_iter": int(model.n_iter_),
               "converged": bool(model.converged_), "tol": tol,
               "aic": float(model.aic(Xvals)), "bic": float(model.bic(Xvals)),
               "random_state": random_state}
        res.update(self._internal_metrics(Xvals, labels))
        solution = ClusterSolution(self, name=name, result=res)
        self.cluster_results_[name] = solution; self.last_solution_ = solution
        return solution

    def cluster_hdbscan(
        self, X: str = "embed", min_cluster_size: int = 5,
        min_samples: Optional[int] = None, metric: str = "euclidean",
        name: Optional[str] = None, **kwargs,
    ) -> ClusterSolution:
        try:
            import hdbscan
        except Exception as e:
            raise ImportError("Falta hdbscan. Instala: pip install hdbscan") from e
        Xmat = self._resolve_X(X)
        Xvals = Xmat.values if isinstance(Xmat, pd.DataFrame) else np.asarray(Xmat)
        model = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples,
                                metric=metric, prediction_data=True, **kwargs)
        labels = model.fit_predict(Xvals)
        name = name or f"hdbscan_mcs{min_cluster_size}_on_{X}"
        res = {"labels": labels, "model": model, "method": "hdbscan", "X_used": X,
               "min_cluster_size": min_cluster_size, "min_samples": min_samples,
               "metric": metric, "converged": True}
        res.update(self._internal_metrics(Xvals, labels))
        solution = ClusterSolution(self, name=name, result=res)
        self.cluster_results_[name] = solution; self.last_solution_ = solution
        return solution

    def _fit_cluster_candidate(self, method: str, X: str, params: Dict[str, Any],
                               name: Optional[str] = None) -> ClusterSolution:
        method = method.lower().strip()
        p = dict(params)
        if method == "kmeans": return self.cluster_kmeans(X=X, name=name, **p)
        if method == "kmedoids": return self.cluster_kmedoids(X=X, name=name, **p)
        if method == "agglomerative": return self.cluster_agglomerative(X=X, name=name, **p)
        if method in {"gaussian_mixture", "gmm"}: return self.cluster_gaussian_mixture(X=X, name=name, **p)
        if method == "dbscan": return self.cluster_dbscan(X=X, name=name, **p)
        if method == "hdbscan": return self.cluster_hdbscan(X=X, name=name, **p)
        raise ValueError(f"Modelo de clustering no soportado: {method}")

    def _candidate_stability(self, method: str, X: str, params: Dict[str, Any],
                             n_runs: int, master_seed: int) -> Optional[float]:
        if n_runs < 2:
            return None
        # Deterministic methods return perfect run-to-run repeatability.
        if method in {"agglomerative", "dbscan", "hdbscan"}:
            return 1.0
        rng = np.random.default_rng(master_seed)
        labels_runs = []
        previous = self.last_solution_
        for i, seed in enumerate(rng.integers(0, 2**31 - 1, size=n_runs)):
            p = dict(params)
            p["random_state"] = int(seed)
            try:
                sol = self._fit_cluster_candidate(method, X, p, name=f"_stability_{method}_{i}")
                labels_runs.append(sol.labels)
                self.cluster_results_.pop(f"_stability_{method}_{i}", None)
            except Exception:
                self.last_solution_ = previous
                return None
        self.last_solution_ = previous
        aris = [adjusted_rand_score(labels_runs[i], labels_runs[j])
                for i in range(len(labels_runs)) for j in range(i + 1, len(labels_runs))]
        return float(np.mean(aris)) if aris else None

    def _dbscan_parameter_grid(self, X: str, min_samples_values: Optional[List[int]] = None,
                               eps_quantiles: Tuple[float, ...] = (0.70, 0.80, 0.90, 0.95)) -> List[Dict[str, Any]]:
        Xmat = self._resolve_X(X)
        Xvals = Xmat.values if isinstance(Xmat, pd.DataFrame) else np.asarray(Xmat)
        n = len(Xvals)
        mins = min_samples_values or sorted(set([max(3, int(np.log(max(n, 3)))), 5, 10]))
        rows = []
        for m in mins:
            m = min(max(2, m), max(2, n - 1))
            nn = NearestNeighbors(n_neighbors=m).fit(Xvals)
            distances, _ = nn.kneighbors(Xvals)
            kth = distances[:, -1]
            for q in eps_quantiles:
                eps = float(np.quantile(kth, q))
                if eps > 0:
                    rows.append({"eps": eps, "min_samples": m, "metric": "euclidean"})
        # deduplicate rounded eps/min_samples
        seen, out = set(), []
        for r in rows:
            key = (round(r["eps"], 8), r["min_samples"])
            if key not in seen:
                seen.add(key); out.append(r)
        return out

    def _parameter_candidates(self, model: str, X: str, k_range,
                              custom_grids: Optional[Dict[str, List[Dict[str, Any]]]],
                              n_init: int, max_iter: int, tol: float,
                              random_state: Optional[int]) -> List[Dict[str, Any]]:
        if custom_grids and model in custom_grids:
            return [dict(x) for x in custom_grids[model]]
        ks = list(k_range)
        if model == "kmeans":
            return [{"k": k, "n_init": n_init, "max_iter": max_iter, "tol": tol,
                     "random_state": random_state} for k in ks]
        if model == "kmedoids":
            return [{"k": k, "metric": metric, "max_iter": max_iter,
                     "random_state": random_state} for k in ks for metric in ("euclidean", "manhattan")]
        if model == "agglomerative":
            return ([{"k": k, "linkage": "ward", "metric": "euclidean"} for k in ks] +
                    [{"k": k, "linkage": link, "metric": "euclidean"}
                     for k in ks for link in ("complete", "average")])
        if model == "gaussian_mixture":
            return [{"k": k, "covariance_type": cov, "n_init": max(5, n_init // 5),
                     "max_iter": max_iter, "tol": max(tol, 1e-4),
                     "random_state": random_state}
                    for k in ks for cov in ("full", "diag")]
        if model == "dbscan":
            return self._dbscan_parameter_grid(X)
        if model == "hdbscan":
            n = len(self._resolve_X(X))
            vals = sorted(set([5, 10, max(5, int(np.sqrt(n))) ]))
            return [{"min_cluster_size": v, "min_samples": None} for v in vals if v < n]
        return []

    def compete_clustering(
        self,
        X: Optional[str] = None,
        spaces: Optional[List[str]] = None,
        models: Union[str, List[str], Tuple[str, ...]] = "recommended",
        k_range=range(2, 11),
        custom_grids: Optional[Dict[str, List[Dict[str, Any]]]] = None,
        evaluate_stability: bool = True,
        n_stability_runs: int = 5,
        min_cluster_pct: float = 2.0,
        max_noise_pct: float = 35.0,
        n_init: int = 50,
        max_iter: int = 500,
        tol: float = 1e-4,
        random_state: Optional[int] = 42,
        master_seed: int = 123,
        verbose: bool = True,
    ) -> ClusteringCompetition:
        """Compete clustering algorithm + representation candidates without a target.

        Ranking combines internal validation, run-to-run stability, cluster-size
        health, noise and convergence. Raw metrics remain visible in the output.
        """
        if spaces is None:
            spaces = [X or "embed"]
        if isinstance(models, str):
            if models == "recommended":
                rec = self.recommend_cluster_models(spaces[0], top_n=6)
                models_list = rec.loc[rec["available"], "model"].tolist()
            elif models == "all":
                avail = self.available_cluster_models()
                models_list = avail.loc[avail["available"], "model"].tolist()
            else:
                models_list = [models]
        else:
            models_list = [str(m).lower() for m in models]

        rows = []
        previous = self.last_solution_
        counter = 0
        for space in spaces:
            self._resolve_X(space)  # validate early
            for model in models_list:
                model = model.lower().strip()
                candidates = self._parameter_candidates(model, space, k_range, custom_grids,
                                                        n_init, max_iter, tol, random_state)
                for params in candidates:
                    counter += 1
                    tmp_name = f"_compete_{space}_{model}_{counter}"
                    try:
                        sol = self._fit_cluster_candidate(model, space, params, name=tmp_name)
                        labels = sol.labels
                        counts = pd.Series(labels[labels != -1]).value_counts()
                        n = len(labels)
                        noise_pct = float(np.mean(labels == -1) * 100)
                        min_pct = float(counts.min() / n * 100) if len(counts) else 0.0
                        stability = (self._candidate_stability(model, space, params,
                                                               n_stability_runs, master_seed + counter)
                                     if evaluate_stability else None)
                        valid = bool(
                            (sol.get("n_clusters") or 0) >= 2
                            and noise_pct <= max_noise_pct
                            and min_pct >= min_cluster_pct
                            and sol.get("silhouette") is not None
                            and sol.get("converged", True)
                        )
                        rows.append({
                            "space": space, "model": model, "params": dict(params),
                            "n_clusters": sol.get("n_clusters"), "n_noise": sol.get("n_noise"),
                            "noise_pct": noise_pct, "min_cluster_pct": min_pct,
                            "silhouette": sol.get("silhouette"),
                            "calinski_harabasz": sol.get("calinski_harabasz"),
                            "davies_bouldin": sol.get("davies_bouldin"),
                            "stability_ari": stability,
                            "inertia": sol.get("inertia"), "aic": sol.get("aic"), "bic": sol.get("bic"),
                            "n_iter": sol.get("n_iter"), "converged": sol.get("converged", True),
                            "valid": valid, "error": None,
                        })
                    except Exception as e:
                        rows.append({"space": space, "model": model, "params": dict(params),
                                     "valid": False, "error": str(e)})
                    finally:
                        self.cluster_results_.pop(tmp_name, None)
                        self.last_solution_ = previous

        candidates_df = pd.DataFrame(rows)
        metric_cols = ["silhouette", "calinski_harabasz", "davies_bouldin", "stability_ari"]
        for c in metric_cols + ["noise_pct", "min_cluster_pct"]:
            if c not in candidates_df: candidates_df[c] = np.nan

        valid_df = candidates_df[candidates_df["valid"]].copy()
        if not valid_df.empty:
            valid_df["rank_silhouette"] = valid_df["silhouette"].rank(ascending=False, method="min")
            valid_df["rank_calinski"] = valid_df["calinski_harabasz"].rank(ascending=False, method="min")
            valid_df["rank_davies"] = valid_df["davies_bouldin"].rank(ascending=True, method="min")
            if valid_df["stability_ari"].notna().any():
                valid_df["rank_stability"] = valid_df["stability_ari"].fillna(-1).rank(ascending=False, method="min")
            else:
                valid_df["rank_stability"] = 1.0
            valid_df["rank_balance"] = valid_df["min_cluster_pct"].rank(ascending=False, method="min")
            valid_df["competition_score"] = valid_df[[
                "rank_silhouette", "rank_calinski", "rank_davies", "rank_stability", "rank_balance"
            ]].mean(axis=1)
            leaderboard = valid_df.sort_values(
                ["competition_score", "silhouette", "stability_ari"],
                ascending=[True, False, False], na_position="last"
            ).reset_index(drop=True)
            leaderboard.insert(0, "rank", np.arange(1, len(leaderboard) + 1))
        else:
            leaderboard = pd.DataFrame()

        competition = ClusteringCompetition(
            analysis=self, leaderboard_df=leaderboard,
            candidates_df=candidates_df,
            config={"spaces": spaces, "models": models_list, "k_range": list(k_range),
                    "evaluate_stability": evaluate_stability,
                    "n_stability_runs": n_stability_runs,
                    "min_cluster_pct": min_cluster_pct, "max_noise_pct": max_noise_pct},
        )
        self.last_clustering_competition_ = competition
        if verbose:
            print("========== CLUSTERING COMPETITION ==========")
            if leaderboard.empty:
                print("No hubo candidatos válidos. Revisa candidates() y los umbrales.")
            else:
                display_cols = ["rank", "space", "model", "params", "n_clusters", "silhouette",
                                "calinski_harabasz", "davies_bouldin", "stability_ari",
                                "noise_pct", "min_cluster_pct", "competition_score"]
                try:
                    from IPython.display import display
                    display(leaderboard[display_cols].head(20))
                except Exception:
                    print(leaderboard[display_cols].head(20))
        return competition

# ============================================================
# SupervisedClassification (DataScience Vic) — FULL CLASS
# - Safe preprocessing (fit only on train)
# - HPO: grid / random / halving / bayes (optional) / keras_tuner (placeholder)
# - Competition leaderboard
# - Final fit + save (pickle) / save NN (h5)
# - Threshold optimization for binary classification
# - LabelEncoder integrated (fixes XGB string-label issue)
# ============================================================

@dataclass
class ClfModelSpec:
    name: str
    estimator: Any
    hpo_spaces: Dict[str, Any]
    supports_proba: bool = True
    tags: Tuple[str, ...] = ()


class ModelInterpretabilityMixin:
    """SHAP interpretability layer shared by supervised classification/regression.

    The mixin explains the already-fitted final estimator. It never trains or
    selects a model. Main methods:
      - prepare_explainer()
      - explain_global()
      - explain_instance()
      - explain_dependence()
      - explain_summary()

    SHAP is optional. Install with ``pip install shap``.
    """

    def _require_shap(self) -> None:
        if not _HAS_SHAP or shap is None:
            raise ImportError(
                "SHAP is not installed. Run: pip install shap"
            )
        if getattr(self, "best_estimator_", None) is None:
            raise RuntimeError("Run fit() before creating SHAP explanations.")

    def _shap_is_classification(self) -> bool:
        return isinstance(self, SupervisedClassification) if "SupervisedClassification" in globals() else hasattr(self, "label_encoder_")

    def _shap_raw_split(self, data: Union[str, pd.DataFrame] = "test") -> pd.DataFrame:
        if isinstance(data, pd.DataFrame):
            return data.copy()

        key = str(data).lower().strip()
        attr_map = {
            "train": "X_train_",
            "valid": "X_valid_",
            "validation": "X_valid_",
            "test": "X_test_",
        }
        if key not in attr_map:
            raise ValueError("data must be 'train', 'valid', 'test', or a DataFrame.")
        value = getattr(self, attr_map[key], None)
        if value is None:
            raise RuntimeError(f"Split '{key}' is unavailable.")
        return value.copy() if isinstance(value, pd.DataFrame) else pd.DataFrame(value)

    @staticmethod
    def _shap_dense(X):
        if hasattr(X, "toarray"):
            X = X.toarray()
        return np.asarray(X)

    def _shap_model_space(
        self,
        raw_df: pd.DataFrame,
    ) -> Tuple[Any, pd.DataFrame, Optional[Any]]:
        """Return (model, transformed dataframe, preprocessor)."""
        est = self.best_estimator_

        # Regression stores a fitted sklearn Pipeline(preprocess, model).
        if isinstance(est, Pipeline) and "model" in est.named_steps:
            model = est.named_steps["model"]
            pre = est.named_steps.get("preprocess")
            if pre is None:
                X_arr = self._shap_dense(raw_df)
                names = list(raw_df.columns)
            else:
                X_arr = self._shap_dense(pre.transform(raw_df))
                try:
                    names = list(pre.get_feature_names_out())
                except Exception:
                    names = [f"feature_{i}" for i in range(X_arr.shape[1])]
            X_df = pd.DataFrame(X_arr, columns=names, index=raw_df.index)
            return model, X_df, pre

        # Classification stores fitted preprocessing and a fitted estimator separately.
        pre = getattr(self, "preprocessor_model_", None)
        if pre is not None:
            X_arr = self._shap_dense(pre.transform(raw_df))
            names = getattr(self, "feature_names_model_", None)
            if names is None:
                try:
                    names = pre.get_feature_names_out()
                except Exception:
                    names = [f"feature_{i}" for i in range(X_arr.shape[1])]
            names = list(names)
            X_df = pd.DataFrame(X_arr, columns=names, index=raw_df.index)
            return est, X_df, pre

        # Final fallback: estimator already consumes raw numeric data.
        X_df = raw_df.copy()
        return est, X_df, None

    def _shap_positive_index(self) -> Optional[int]:
        if not self._shap_is_classification():
            return None
        idx = getattr(self, "positive_encoded_", None)
        if idx is not None:
            return int(idx)
        return 1 if getattr(self, "n_classes_", 2) == 2 else None

    def _shap_prediction_function(self, model: Any) -> Callable:
        """Model-agnostic callable used by permutation SHAP."""
        if self._shap_is_classification():
            pos_idx = self._shap_positive_index()

            def predict_fn(X):
                X = np.asarray(X)
                if hasattr(model, "predict_proba"):
                    p = np.asarray(model.predict_proba(X))
                    if p.ndim == 2 and pos_idx is not None and pos_idx < p.shape[1]:
                        return p[:, pos_idx]
                    return p
                if hasattr(model, "decision_function"):
                    d = np.asarray(model.decision_function(X))
                    return d
                return np.asarray(model.predict(X))

            return predict_fn

        def predict_fn(X):
            return np.asarray(model.predict(np.asarray(X))).reshape(-1)

        return predict_fn

    @staticmethod
    def _shap_model_family(model: Any) -> str:
        name = type(model).__name__.lower()
        module = type(model).__module__.lower()
        if any(k in name for k in (
            "xgb", "randomforest", "extratrees", "decisiontree",
            "gradientboosting", "histgradientboosting", "lightgbm", "catboost"
        )) or any(k in module for k in ("xgboost", "lightgbm", "catboost")):
            return "tree"
        if any(k in name for k in (
            "linear", "logistic", "ridge", "lasso", "elasticnet"
        )):
            return "linear"
        return "generic"

    def prepare_explainer(
        self,
        background: Union[str, pd.DataFrame] = "train",
        max_background: int = 300,
        algorithm: str = "auto",
        random_state: Optional[int] = 42,
        force: bool = False,
    ) -> Any:
        """Create and cache a SHAP explainer for the fitted final model.

        Tree models use TreeExplainer when possible; linear models use
        LinearExplainer; all other models fall back to a model-agnostic
        permutation explainer. Generic explanations are slower.
        """
        self._require_shap()
        if getattr(self, "shap_explainer_", None) is not None and not force:
            return self.shap_explainer_

        raw_bg = self._shap_raw_split(background)
        if len(raw_bg) > int(max_background):
            raw_bg = raw_bg.sample(n=int(max_background), random_state=random_state)

        model, X_bg, pre = self._shap_model_space(raw_bg)
        family = self._shap_model_family(model)

        explainer = None
        method = None
        if family == "tree":
            try:
                if self._shap_is_classification() and hasattr(model, "predict_proba"):
                    explainer = shap.TreeExplainer(
                        model,
                        data=X_bg,
                        model_output="probability",
                        feature_perturbation="interventional",
                    )
                else:
                    explainer = shap.TreeExplainer(model, data=X_bg)
                method = "TreeExplainer"
            except Exception:
                explainer = shap.Explainer(model, X_bg, algorithm="auto")
                method = "shap.Explainer(auto-tree)"
        elif family == "linear":
            try:
                explainer = shap.LinearExplainer(model, X_bg)
                method = "LinearExplainer"
            except Exception:
                explainer = shap.Explainer(self._shap_prediction_function(model), X_bg, algorithm="permutation")
                method = "PermutationExplainer"
        else:
            resolved = "permutation" if algorithm == "auto" else algorithm
            explainer = shap.Explainer(
                self._shap_prediction_function(model),
                X_bg,
                algorithm=resolved,
            )
            method = f"shap.Explainer({resolved})"

        self.shap_explainer_ = explainer
        self.shap_model_ = model
        self.shap_preprocessor_ = pre
        self.shap_background_ = X_bg
        self.shap_feature_names_ = list(X_bg.columns)
        self.shap_explainer_method_ = method
        self.shap_model_family_ = family
        self.shap_last_explanation_ = None
        self.shap_global_importance_ = None
        self.shap_last_data_name_ = None
        return explainer

    def _shap_select_output(
        self,
        explanation: Any,
        class_label: Any = None,
    ) -> Any:
        """Reduce multiclass/multioutput SHAP values to one requested output."""
        values = np.asarray(explanation.values)
        if values.ndim <= 2:
            return explanation

        # Common SHAP shape: rows x features x outputs.
        if class_label is not None and self._shap_is_classification():
            if getattr(self, "label_encoder_", None) is not None:
                output_idx = int(self.label_encoder_.transform([class_label])[0])
            else:
                output_idx = int(class_label)
        else:
            output_idx = self._shap_positive_index()
            if output_idx is None:
                output_idx = 0

        if values.ndim == 3:
            # Prefer last axis as outputs; handle legacy outputs x rows x features.
            if values.shape[-1] <= max(2, getattr(self, "n_classes_", 2) or 2):
                return explanation[..., output_idx]
            return explanation[output_idx]
        return explanation

    def _shap_compute(
        self,
        X_df: pd.DataFrame,
        class_label: Any = None,
        max_evals: Optional[int] = None,
    ) -> Any:
        if getattr(self, "shap_explainer_", None) is None:
            self.prepare_explainer()
        kwargs = {}
        if max_evals is not None:
            kwargs["max_evals"] = int(max_evals)
        try:
            explanation = self.shap_explainer_(X_df, **kwargs)
        except TypeError:
            explanation = self.shap_explainer_(X_df)
        return self._shap_select_output(explanation, class_label=class_label)

    @staticmethod
    def _shap_original_feature_name(name: str, original_features: List[str]) -> str:
        # Longest-first avoids matching "age" before "age_group".
        for feature in sorted(original_features, key=len, reverse=True):
            if name == feature or name.startswith(feature + "_") or name.startswith(feature + "="):
                return feature
        return name

    def explain_global(
        self,
        data: Union[str, pd.DataFrame] = "test",
        sample_size: int = 500,
        plots: Tuple[str, ...] = ("beeswarm", "bar"),
        top_n: int = 15,
        class_label: Any = None,
        aggregate_ohe: bool = False,
        max_evals: Optional[int] = None,
        random_state: Optional[int] = 42,
        show: bool = True,
    ) -> Dict[str, Any]:
        """Explain the model globally and return ranked mean absolute SHAP values."""
        self._require_shap()
        raw = self._shap_raw_split(data)
        if len(raw) > int(sample_size):
            raw = raw.sample(n=int(sample_size), random_state=random_state)
        _, X_df, _ = self._shap_model_space(raw)

        explanation = self._shap_compute(X_df, class_label=class_label, max_evals=max_evals)
        values = np.asarray(explanation.values)
        if values.ndim != 2:
            raise ValueError(f"Expected a 2D SHAP matrix after output selection, got {values.shape}.")

        importance = pd.DataFrame({
            "feature": list(X_df.columns),
            "mean_abs_shap": np.mean(np.abs(values), axis=0),
            "mean_shap": np.mean(values, axis=0),
        }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
        importance.insert(0, "rank", np.arange(1, len(importance) + 1))

        aggregated = None
        if aggregate_ohe:
            original = list(getattr(self, "features", []) or raw.columns)
            tmp = importance.copy()
            tmp["original_feature"] = tmp["feature"].map(
                lambda n: self._shap_original_feature_name(str(n), original)
            )
            aggregated = (
                tmp.groupby("original_feature", as_index=False)
                .agg(
                    mean_abs_shap=("mean_abs_shap", "sum"),
                    mean_shap=("mean_shap", "sum"),
                )
                .sort_values("mean_abs_shap", ascending=False)
                .reset_index(drop=True)
            )
            aggregated.insert(0, "rank", np.arange(1, len(aggregated) + 1))

        if show:
            for plot_name in plots:
                p = str(plot_name).lower().strip()
                if p == "beeswarm":
                    shap.plots.beeswarm(explanation, max_display=int(top_n), show=True)
                elif p == "bar":
                    shap.plots.bar(explanation, max_display=int(top_n), show=True)
                else:
                    raise ValueError("plots supports: 'beeswarm' and 'bar'.")

        self.shap_last_explanation_ = explanation
        self.shap_global_importance_ = importance
        self.shap_global_importance_aggregated_ = aggregated
        self.shap_last_data_name_ = data if isinstance(data, str) else "custom_dataframe"

        return {
            "importance": importance,
            "importance_aggregated": aggregated,
            "explanation": explanation,
            "X_explained": X_df,
            "explainer_method": getattr(self, "shap_explainer_method_", None),
        }

    def explain_instance(
        self,
        index: Any,
        data: Union[str, pd.DataFrame] = "test",
        plot: str = "waterfall",
        top_n: int = 12,
        class_label: Any = None,
        max_evals: Optional[int] = None,
        show: bool = True,
    ) -> Dict[str, Any]:
        """Explain one row by original index or positional index."""
        self._require_shap()
        raw = self._shap_raw_split(data)
        if index in raw.index:
            row_raw = raw.loc[[index]]
            source_index = index
        else:
            pos = int(index)
            row_raw = raw.iloc[[pos]]
            source_index = raw.index[pos]

        model, X_row, _ = self._shap_model_space(row_raw)
        explanation = self._shap_compute(X_row, class_label=class_label, max_evals=max_evals)
        one = explanation[0]
        values = np.asarray(one.values).reshape(-1)

        local_df = pd.DataFrame({
            "feature": list(X_row.columns),
            "value": X_row.iloc[0].to_numpy(),
            "shap_value": values,
        })
        local_df["abs_shap"] = local_df["shap_value"].abs()
        local_df["direction"] = np.where(
            local_df["shap_value"] > 0,
            "increases output",
            np.where(local_df["shap_value"] < 0, "decreases output", "neutral"),
        )
        local_df = local_df.sort_values("abs_shap", ascending=False).reset_index(drop=True)

        prediction = None
        predicted_label = None
        if self._shap_is_classification():
            if hasattr(model, "predict_proba"):
                proba = np.asarray(model.predict_proba(X_row))
                pos_idx = self._shap_positive_index()
                if class_label is not None and getattr(self, "label_encoder_", None) is not None:
                    pos_idx = int(self.label_encoder_.transform([class_label])[0])
                if proba.ndim == 2 and pos_idx is not None:
                    prediction = float(proba[0, pos_idx])
            pred_enc = np.asarray(model.predict(X_row)).reshape(-1)
            if getattr(self, "label_encoder_", None) is not None:
                predicted_label = self.label_encoder_.inverse_transform(pred_enc.astype(int))[0]
            else:
                predicted_label = pred_enc[0]
        else:
            prediction = float(np.asarray(model.predict(X_row)).reshape(-1)[0])

        if show:
            p = str(plot).lower().strip()
            if p == "waterfall":
                shap.plots.waterfall(one, max_display=int(top_n), show=True)
            elif p == "force":
                try:
                    shap.plots.force(one, matplotlib=True, show=True)
                except Exception:
                    shap.force_plot(one.base_values, one.values, one.data, feature_names=X_row.columns, matplotlib=True)
            elif p in ("none", "false"):
                pass
            else:
                raise ValueError("plot supports: 'waterfall', 'force', or 'none'.")

        report = {
            "index": source_index,
            "prediction": prediction,
            "predicted_label": predicted_label,
            "base_value": float(np.asarray(one.base_values).reshape(-1)[0]),
            "contributions": local_df,
            "explanation": one,
            "raw_row": row_raw,
            "model_row": X_row,
        }
        self.shap_last_instance_ = report
        return report

    def explain_dependence(
        self,
        feature: str,
        data: Union[str, pd.DataFrame] = "test",
        sample_size: int = 500,
        class_label: Any = None,
        max_evals: Optional[int] = None,
        random_state: Optional[int] = 42,
        show: bool = True,
    ) -> Any:
        """Plot SHAP value versus feature value to reveal nonlinear behavior."""
        raw = self._shap_raw_split(data)
        if len(raw) > int(sample_size):
            raw = raw.sample(n=int(sample_size), random_state=random_state)
        _, X_df, _ = self._shap_model_space(raw)
        if feature not in X_df.columns:
            matches = [c for c in X_df.columns if str(c).startswith(str(feature) + "_")]
            raise ValueError(
                f"Feature '{feature}' is not in model space. Exact transformed matches: {matches[:20]}"
            )
        explanation = self._shap_compute(X_df, class_label=class_label, max_evals=max_evals)
        feature_exp = explanation[:, feature]
        if show:
            shap.plots.scatter(feature_exp, show=True)
        return feature_exp

    # ------------------------------------------------------------
    # LIME: local model-agnostic explanations
    # ------------------------------------------------------------
    def _require_lime(self) -> None:
        if not _HAS_LIME or LimeTabularExplainer is None:
            raise ImportError("LIME is not installed. Run: pip install lime")
        if getattr(self, "best_estimator_", None) is None:
            raise RuntimeError("Run fit() before creating LIME explanations.")

    def prepare_lime_explainer(
        self,
        background: Union[str, pd.DataFrame] = "train",
        max_background: int = 2000,
        discretize_continuous: bool = True,
        discretizer: str = "quartile",
        feature_selection: str = "auto",
        random_state: Optional[int] = 42,
        force: bool = False,
    ) -> Any:
        """Create and cache a tabular LIME explainer for the fitted model.

        LIME operates in the same transformed numeric feature space consumed by
        the final estimator. It is intended for local explanations only.
        """
        self._require_lime()
        if getattr(self, "lime_explainer_", None) is not None and not force:
            return self.lime_explainer_

        raw_bg = self._shap_raw_split(background)
        if len(raw_bg) > int(max_background):
            raw_bg = raw_bg.sample(n=int(max_background), random_state=random_state)

        model, X_bg, pre = self._shap_model_space(raw_bg)
        X_arr = np.asarray(X_bg, dtype=float)
        is_classification = self._shap_is_classification()

        class_names = None
        if is_classification:
            encoder = getattr(self, "label_encoder_", None)
            if encoder is not None:
                class_names = [str(x) for x in encoder.classes_]
            elif hasattr(model, "classes_"):
                class_names = [str(x) for x in model.classes_]

        self.lime_explainer_ = LimeTabularExplainer(
            training_data=X_arr,
            feature_names=list(X_bg.columns),
            class_names=class_names,
            mode="classification" if is_classification else "regression",
            discretize_continuous=bool(discretize_continuous),
            discretizer=discretizer,
            feature_selection=feature_selection,
            random_state=random_state,
        )
        self.lime_model_ = model
        self.lime_preprocessor_ = pre
        self.lime_background_ = X_bg
        self.lime_feature_names_ = list(X_bg.columns)
        self.lime_mode_ = "classification" if is_classification else "regression"
        self.lime_class_names_ = class_names
        self.lime_last_explanation_ = None

        print("LIME explainer prepared")
        print(f"Mode            : {self.lime_mode_}")
        print(f"Background rows : {len(X_bg)}")
        print(f"Model features  : {X_bg.shape[1]}")
        return self.lime_explainer_

    def explain_instance_lime(
        self,
        index: int,
        data: Union[str, pd.DataFrame] = "test",
        num_features: int = 10,
        num_samples: int = 5000,
        class_label: Any = None,
        show: bool = True,
    ) -> Dict[str, Any]:
        """Explain one row using LIME and return diagnostics plus a table.

        ``index`` is positional within the selected split. LIME weights are
        coefficients of the local surrogate; they should not automatically be
        interpreted as percentage-point changes.
        """
        self._require_lime()
        if getattr(self, "lime_explainer_", None) is None:
            self.prepare_lime_explainer()

        raw = self._shap_raw_split(data)
        model, X_df, _ = self._shap_model_space(raw)
        if not 0 <= int(index) < len(X_df):
            raise IndexError(f"index={index} is outside the valid positional range 0..{len(X_df)-1}.")

        row = np.asarray(X_df.iloc[int(index)], dtype=float).reshape(-1)
        original_index = X_df.index[int(index)]
        target_label = None
        target_class_index = None
        model_prediction = None

        if self.lime_mode_ == "classification":
            if not hasattr(model, "predict_proba"):
                raise RuntimeError("LIME classification explanations require predict_proba().")
            probabilities = np.asarray(model.predict_proba(row.reshape(1, -1)))[0]
            model_classes = list(getattr(model, "classes_", range(len(probabilities))))

            if class_label is not None:
                if getattr(self, "label_encoder_", None) is not None:
                    encoded = int(self.label_encoder_.transform([class_label])[0])
                else:
                    encoded = class_label
                if encoded not in model_classes:
                    raise ValueError(f"Class {class_label!r} is not available. Model classes: {model_classes}")
                target_class_index = model_classes.index(encoded)
                target_label = class_label
            elif getattr(self, "positive_encoded_", None) is not None and self.positive_encoded_ in model_classes:
                target_class_index = model_classes.index(self.positive_encoded_)
                target_label = getattr(self, "positive_label_", None) or getattr(self, "positive_label_requested_", None)
            else:
                target_class_index = int(np.argmax(probabilities))
                encoded = model_classes[target_class_index]
                if getattr(self, "label_encoder_", None) is not None:
                    target_label = self.label_encoder_.inverse_transform([int(encoded)])[0]
                else:
                    target_label = encoded

            exp = self.lime_explainer_.explain_instance(
                data_row=row,
                predict_fn=model.predict_proba,
                labels=(int(target_class_index),),
                num_features=int(num_features),
                num_samples=int(num_samples),
            )
            items = exp.as_list(label=int(target_class_index))
            model_prediction = float(probabilities[int(target_class_index)])
        else:
            exp = self.lime_explainer_.explain_instance(
                data_row=row,
                predict_fn=model.predict,
                num_features=int(num_features),
                num_samples=int(num_samples),
            )
            items = exp.as_list()
            model_prediction = float(np.asarray(model.predict(row.reshape(1, -1))).reshape(-1)[0])

        table = pd.DataFrame(items, columns=["rule", "lime_weight"])
        table["abs_weight"] = table["lime_weight"].abs()
        table["direction"] = np.where(
            table["lime_weight"] >= 0,
            "increases output",
            "decreases output",
        )
        table = table.sort_values("abs_weight", ascending=False).reset_index(drop=True)

        local_pred_arr = np.asarray(getattr(exp, "local_pred", []), dtype=float).reshape(-1)
        local_prediction = float(local_pred_arr[0]) if local_pred_arr.size else np.nan
        fidelity = float(getattr(exp, "score", np.nan))

        if show:
            try:
                fig = exp.as_pyplot_figure(label=target_class_index) if self.lime_mode_ == "classification" else exp.as_pyplot_figure()
                plt.tight_layout()
                plt.show()
            except Exception as exc:
                warnings.warn(f"LIME plot could not be displayed: {exc}")

        result = {
            "index": int(index),
            "original_index": original_index,
            "data_used": data if isinstance(data, str) else "DataFrame",
            "mode": self.lime_mode_,
            "class_label": target_label,
            "class_index": target_class_index,
            "model_prediction": model_prediction,
            "local_prediction": local_prediction,
            "local_fidelity": fidelity,
            "num_samples": int(num_samples),
            "explanation": table,
            "raw_explanation": exp,
        }
        self.lime_last_explanation_ = result
        return result

    def lime_summary(self, show: bool = True) -> pd.DataFrame:
        """Return a compact summary of the latest local LIME explanation."""
        result = getattr(self, "lime_last_explanation_", None)
        if result is None:
            raise RuntimeError("Run explain_instance_lime() first.")
        if show:
            print("=" * 58)
            print("LIME LOCAL EXPLANATION SUMMARY")
            print("=" * 58)
            print(f"Observation       : {result['index']} (original index={result['original_index']})")
            if result.get("class_label") is not None:
                print(f"Class explained   : {result['class_label']}")
            print(f"Model prediction  : {result['model_prediction']:.6f}")
            print(f"Local prediction  : {result['local_prediction']:.6f}")
            print(f"Local fidelity R² : {result['local_fidelity']:.4f}")
            print("-" * 58)
            self.pretty_table(result["explanation"], title="Local LIME drivers")
        return result["explanation"].copy()

    def explain_summary(self, top_n: int = 10, show: bool = True) -> pd.DataFrame:
        """Compact numerical summary of the latest global SHAP explanation."""
        importance = getattr(self, "shap_global_importance_", None)
        if importance is None:
            result = self.explain_global(show=False, top_n=top_n)
            importance = result["importance"]

        top = importance.head(int(top_n)).copy()
        if show:
            print("=" * 58)
            print("MODEL INTERPRETABILITY SUMMARY")
            print("=" * 58)
            print(f"Model              : {getattr(self, 'best_model_name_', type(getattr(self, 'best_estimator_', None)).__name__)}")
            print(f"Target             : {getattr(self, 'target', None)}")
            print(f"Explainer          : {getattr(self, 'shap_explainer_method_', None)}")
            print(f"Model family       : {getattr(self, 'shap_model_family_', None)}")
            print(f"Rows last explained: {len(getattr(self, 'shap_last_explanation_', [])) if getattr(self, 'shap_last_explanation_', None) is not None else 'n/a'}")
            print("-" * 58)
            self.pretty_table(top, title=f"Top {min(top_n, len(top))} global SHAP drivers")
        return top


class SupervisedClassification(ModelInterpretabilityMixin, DFCore):
    """
    Supervised classification module (DataScience Vic).

    Key points:
    - Uses DFPreprocess to build preprocessor, but fits it only on train (no leakage)
    - Encodes y internally using LabelEncoder (needed for XGB with string labels)
    - HPO per model + competition leaderboard
    - Threshold optimization for binary classification
    """

    def __init__(
        self,
        df: pd.DataFrame,
        target: str,
        features: Optional[List[str]] = None,
        output_dir: str = "supervised_outputs",
        positive_label: Any = None,
        class_of_interest: Any = None,
    ):
        super().__init__(df=df, target=target, features=features)

        self.output_dir = output_dir

        if positive_label is not None and class_of_interest is not None and positive_label != class_of_interest:
            raise ValueError(
                "Use only one of positive_label or class_of_interest, or provide the same value to both."
            )
        self.positive_label_requested_ = (
            positive_label if positive_label is not None else class_of_interest
        )
        self.class_of_interest_ = self.positive_label_requested_
        self.positive_encoded_: Optional[int] = None
        self.negative_label_: Any = None
        self.negative_encoded_: Optional[int] = None
        self.n_classes_: Optional[int] = None
        self.metric_recommendations_: Optional[pd.DataFrame] = None
        self.business_goal_: Optional[str] = None
        self.last_test_default_: Optional[pd.DataFrame] = None
        self.last_test_threshold_: Optional[pd.DataFrame] = None
        self.refit_on_: Optional[str] = None
        self.saved_path_: Optional[str] = None

        # raw data
        self.X_raw_ = None
        self.y_raw_ = None

        # splits (raw)
        self.X_train_ = None
        self.X_valid_ = None
        self.X_test_ = None

        self.y_train_ = None
        self.y_valid_ = None
        self.y_test_ = None

        # label encoding (for training)
        self.label_encoder_: Optional[LabelEncoder] = None
        self.y_train_enc_: Optional[pd.Series] = None
        self.y_valid_enc_: Optional[pd.Series] = None
        self.y_test_enc_: Optional[pd.Series] = None

        # preprocessing
        self.preprocessor_model_ = None
        self.feature_names_model_ = None

        # transformed splits (model-ready)
        self.X_train_model_ = None
        self.X_valid_model_ = None
        self.X_test_model_ = None

        # CV config
        self.cv_ = None
        self.scoring_ = "f1_macro"

        # registry + results
        self.model_specs_: Dict[str, ClfModelSpec] = {}
        self.hpo_results_: Dict[str, pd.DataFrame] = {}
        self.best_params_: Dict[str, Dict[str, Any]] = {}
        self.leaderboard_: Optional[pd.DataFrame] = None

        # fitted best
        self.best_model_name_: Optional[str] = None
        self.best_estimator_: Any = None

        # threshold
        self.threshold_: float = 0.5
        self.threshold_report_: Optional[pd.DataFrame] = None
        self.positive_label_: Any = self.positive_label_requested_  # ORIGINAL label space

    # ============================================================
    # Helpers
    # ============================================================
    def _y_train_for_fit(self) -> pd.Series:
        return self.y_train_enc_ if self.y_train_enc_ is not None else self.y_train_

    def _y_valid_for_fit(self) -> Optional[pd.Series]:
        return self.y_valid_enc_ if self.y_valid_enc_ is not None else self.y_valid_

    def _y_test_for_fit(self) -> pd.Series:
        return self.y_test_enc_ if self.y_test_enc_ is not None else self.y_test_

    def _decode_labels(self, y_enc: np.ndarray) -> np.ndarray:
        if self.label_encoder_ is None:
            return y_enc
        return self.label_encoder_.inverse_transform(np.array(y_enc))

    def _encode_label(self, label: Any) -> Any:
        if self.label_encoder_ is None:
            return label
        return int(self.label_encoder_.transform([label])[0])

    def _resolve_positive_class(self) -> None:
        """Resolve and validate the business class of interest after y encoding."""
        if self.label_encoder_ is None:
            raise RuntimeError("Label encoder is not fitted.")

        classes = list(self.label_encoder_.classes_)
        self.n_classes_ = len(classes)

        if self.n_classes_ < 2:
            raise ValueError(f"Target '{self.target}' must contain at least two classes.")

        requested = self.positive_label_requested_
        if requested is None:
            # Standard encoded-order fallback. Explicitly surfaced to avoid a silent business assumption.
            requested = classes[1] if self.n_classes_ == 2 else None

        if requested is not None and requested not in classes:
            raise ValueError(
                f"positive_label={requested!r} is not present in target '{self.target}'. "
                f"Available classes: {classes}"
            )

        self.positive_label_ = requested
        self.class_of_interest_ = requested

        if requested is not None:
            self.positive_encoded_ = int(self.label_encoder_.transform([requested])[0])
        else:
            self.positive_encoded_ = None

        if self.n_classes_ == 2 and requested is not None:
            self.negative_label_ = next(c for c in classes if c != requested)
            self.negative_encoded_ = int(self.label_encoder_.transform([self.negative_label_])[0])
        else:
            self.negative_label_ = None
            self.negative_encoded_ = None

    def _resolve_scorer(self, metric: Any = None) -> Any:
        """Return a sklearn scorer, including business-oriented class-specific metrics."""
        metric = self.scoring_ if metric is None else metric
        if not isinstance(metric, str):
            return metric

        key = metric.lower().strip()
        aliases = {
            "precision_class": "precision_positive",
            "recall_class": "recall_positive",
            "f1_class": "f1_positive",
            "precision_interest": "precision_positive",
            "recall_interest": "recall_positive",
            "f1_interest": "f1_positive",
        }
        key = aliases.get(key, key)

        if key in {"precision_positive", "recall_positive", "f1_positive"}:
            if self.positive_encoded_ is None:
                raise ValueError(
                    f"Metric '{metric}' requires positive_label/class_of_interest in the constructor."
                )
            fn = {
                "precision_positive": precision_score,
                "recall_positive": recall_score,
                "f1_positive": f1_score,
            }[key]
            return make_scorer(fn, pos_label=self.positive_encoded_, zero_division=0)

        return metric

    def recommend_metrics(
        self,
        business_goal: Optional[str] = None,
        show: bool = True,
    ) -> Optional[pd.DataFrame]:
        """Recommend metrics from class balance and an optional business objective.

        Parameters
        ----------
        business_goal:
            Business orientation used to prioritize the metrics.
        show:
            If ``True``, display the table and return ``None`` so Jupyter does
            not render it a second time. If ``False``, return the DataFrame
            without displaying it. The complete table is always stored in
            ``self.metric_recommendations_``.
        """
        if self.y_raw_ is None:
            y = self.df[self.target].dropna()
        else:
            y = self.y_raw_.dropna()

        counts = y.value_counts()
        if len(counts) < 2:
            raise ValueError("Metric recommendations require at least two target classes.")
        ratio = float(counts.max() / max(1, counts.min()))
        balanced = ratio < 1.5
        goal = (business_goal or "balanced").lower().strip()
        self.business_goal_ = goal

        rows = []
        def add(metric, priority, purpose, note):
            rows.append({
                "metric": metric,
                "priority": priority,
                "purpose": purpose,
                "note": note,
            })

        if goal in {"detect_positives", "maximize_recall", "recall"}:
            add("recall_positive", "Primary", "Detect as many positive cases as possible", "Track precision and false-positive cost.")
            add("f2_positive", "Secondary", "Weight recall more than precision", "Useful when false negatives are more costly.")
            add("average_precision", "Secondary", "Evaluate positive-class ranking quality", "Especially useful with imbalance.")
        elif goal in {"avoid_false_positives", "maximize_precision", "precision"}:
            add("precision_positive", "Primary", "Make positive alerts reliable", "Use a minimum recall constraint.")
            add("recall_positive", "Guardrail", "Prevent the model from becoming too conservative", "Set a business minimum.")
            add("average_precision", "Secondary", "Evaluate precision-recall ranking", "Threshold-independent summary.")
        elif goal in {"probability_ranking", "ranking", "auc"}:
            add("roc_auc", "Primary", "Evaluate ranking across thresholds", "Works well when classes are reasonably balanced.")
            add("average_precision", "Primary", "Focus ranking quality on the positive class", "More informative as imbalance grows.")
            add("log_loss", "Secondary", "Evaluate probability quality", "Lower is better; calibration-sensitive.")
        else:
            add("accuracy", "Primary" if balanced else "Secondary", "Overall correctness", "Appropriate here because classes are balanced." if balanced else "Can hide minority-class errors.")
            add("f1_positive", "Primary", "Balance precision and recall for the class of interest", "Business-oriented binary metric.")
            add("f1_macro", "Secondary", "Give equal weight to every class", "Useful for model-family comparison.")
            add("roc_auc", "Secondary", "Compare probability ranking independent of one threshold", "Inspect threshold metrics too.")

        df = pd.DataFrame(rows)
        df.insert(0, "business_goal", goal)
        df.insert(1, "positive_label", self.positive_label_)
        df.insert(2, "imbalance_ratio", ratio)
        df.insert(3, "balance_status", "Balanced" if balanced else "Imbalanced")
        self.metric_recommendations_ = df
        if show:
            try:
                from IPython.display import display
                styled = df.style.set_properties(
                    subset=["purpose", "note"],
                    **{"white-space": "normal", "text-align": "left"},
                )
                display(styled)
            except Exception:
                print(df.to_string(index=False))
            # Se mostró explícitamente; no devolver el DataFrame evita la
            # segunda representación automática de Jupyter.
            return None

        return df

    # ============================================================
    # 0) Prepare (split + fit preprocessor on train only)
    # ============================================================
    def prepare(
        self,
        test_size: float = 0.20,
        valid_size: float = 0.0,
        stratify: bool = True,

        # DFPreprocess config (keep aligned with your existing signature)
        model_encoding: str = "auto",
        auto_cardinality_threshold: int = 15,
        include_bool_as_cat: bool = True,
        num_impute: str = "median",
        cat_impute: str = "most_frequent",
        scale_num: str = "standard",
        ohe_sparse: bool = False,
        ohe_min_frequency: Optional[int] = None,

        # CV
        cv_splits: int = 5,
        cv_shuffle: bool = True,
        scoring: str = "f1_macro"
    ) -> Dict[str, Any]:

        self.scoring_ = scoring

        # Get X,y using DFPreprocess (your framework)
        prep = DFPreprocess(self.df, target=self.target, features=self.features)
        X, y = prep.get_X_y(features=self.features, dropna_target=True)

        self.X_raw_ = X.reset_index(drop=True)
        self.y_raw_ = y.reset_index(drop=True)

        strat = self.y_raw_ if stratify else None

        # Split test
        X_tr, X_te, y_tr, y_te = train_test_split(
            self.X_raw_, self.y_raw_,
            test_size=test_size,
            stratify=strat
        )

        # Optional valid split from train
        if valid_size and valid_size > 0:
            strat2 = y_tr if stratify else None
            X_tr, X_va, y_tr, y_va = train_test_split(
                X_tr, y_tr,
                test_size=valid_size,
                stratify=strat2
            )
            self.X_valid_ = X_va.reset_index(drop=True)
            self.y_valid_ = y_va.reset_index(drop=True)
        else:
            self.X_valid_, self.y_valid_ = None, None

        self.X_train_ = X_tr.reset_index(drop=True)
        self.y_train_ = y_tr.reset_index(drop=True)
        self.X_test_ = X_te.reset_index(drop=True)
        self.y_test_ = y_te.reset_index(drop=True)

        # -----------------------------
        # Encode y (fix XGB string labels)
        # Fit encoder only on train
        # -----------------------------
        self.label_encoder_ = LabelEncoder()
        self.label_encoder_.fit(self.y_train_)
        self._resolve_positive_class()
        self.scoring_ = self._resolve_scorer(scoring)

        self.y_train_enc_ = pd.Series(self.label_encoder_.transform(self.y_train_), name=self.target)
        self.y_test_enc_ = pd.Series(self.label_encoder_.transform(self.y_test_), name=self.target)
        if self.y_valid_ is not None:
            self.y_valid_enc_ = pd.Series(self.label_encoder_.transform(self.y_valid_), name=self.target)
        else:
            self.y_valid_enc_ = None

        # -----------------------------
        # Build preprocessor model + fit ONLY on train (no leakage)
        # -----------------------------
        kwargs = dict(
            X=self.X_train_,
            include_bool_as_cat=include_bool_as_cat,
            encoding=model_encoding,
            auto_cardinality_threshold=auto_cardinality_threshold,
            num_impute=num_impute,
            cat_impute=cat_impute,
            scale_num=scale_num,
            ohe_sparse=ohe_sparse,
        )
        # only pass if the method supports it
        try:
            if "ohe_min_frequency" in prep.build_preprocessor_model.__code__.co_varnames:
                kwargs["ohe_min_frequency"] = ohe_min_frequency
        except Exception:
            pass

        pre_model = prep.build_preprocessor_model(**kwargs)
        pre_model.fit(self.X_train_)

        self.preprocessor_model_ = pre_model
        self.feature_names_model_ = prep.get_feature_names(pre_model)

        # Transform splits
        self.X_train_model_ = pd.DataFrame(
            pre_model.transform(self.X_train_),
            columns=self.feature_names_model_
        )
        self.X_test_model_ = pd.DataFrame(
            pre_model.transform(self.X_test_),
            columns=self.feature_names_model_
        )
        if self.X_valid_ is not None:
            self.X_valid_model_ = pd.DataFrame(
                pre_model.transform(self.X_valid_),
                columns=self.feature_names_model_
            )
        else:
            self.X_valid_model_ = None

        # CV
        self.cv_ = StratifiedKFold(
            n_splits=cv_splits,
            shuffle=cv_shuffle
        )

        return {
            "X_train_model": self.X_train_model_,
            "X_valid_model": self.X_valid_model_,
            "X_test_model": self.X_test_model_,
            "y_train": self.y_train_,
            "y_valid": self.y_valid_,
            "y_test": self.y_test_,
            "y_train_enc": self.y_train_enc_,
            "y_valid_enc": self.y_valid_enc_,
            "y_test_enc": self.y_test_enc_,
            "feature_names_model": self.feature_names_model_,
            "preprocessor_model": self.preprocessor_model_,
            "label_encoder": self.label_encoder_,
            "positive_label": self.positive_label_,
            "positive_encoded": self.positive_encoded_,
            "negative_label": self.negative_label_,
            "negative_encoded": self.negative_encoded_,
            "scoring": scoring,
        }

    def summary(self) -> pd.DataFrame:
        """Compact machine-readable state summary for the classification workflow."""
        classes = (
            list(self.label_encoder_.classes_)
            if self.label_encoder_ is not None
            else list(pd.Series(self.df[self.target]).dropna().unique())
        )
        best_cv = np.nan
        best_cv_std = np.nan
        if self.leaderboard_ is not None and len(self.leaderboard_) > 0:
            best_cv = self.leaderboard_.iloc[0].get("cv_mean", np.nan)
            best_cv_std = self.leaderboard_.iloc[0].get("cv_std", np.nan)

        return pd.DataFrame([{
            "rows": len(self.df),
            "features": len(self.features) if self.features is not None else len(self.df.columns) - 1,
            "target": self.target,
            "classes": classes,
            "n_classes": len(classes),
            "positive_label": self.positive_label_,
            "positive_encoded": self.positive_encoded_,
            "negative_label": self.negative_label_,
            "business_goal": self.business_goal_,
            "scoring": str(self.scoring_),
            "threshold": self.threshold_,
            "best_model": self.best_model_name_,
            "best_cv_mean": best_cv,
            "best_cv_std": best_cv_std,
            "refit_on": self.refit_on_,
            "saved_path": self.saved_path_,
        }])

    def executive_report(
        self,
        show: bool = True,
        decimals: int = 4,
    ) -> pd.DataFrame:
        """Create an executive summary of the completed classification workflow.

        The report combines the configured business objective, winning model,
        cross-validation result, threshold, and the latest available test metrics.
        It does not recalculate or retrain anything.
        """
        best_cv = np.nan
        best_cv_std = np.nan
        metric = str(self.scoring_)
        tuned = None
        if self.leaderboard_ is not None and len(self.leaderboard_) > 0:
            row = self.leaderboard_.iloc[0]
            best_cv = row.get("cv_mean", np.nan)
            best_cv_std = row.get("cv_std", np.nan)
            metric = row.get("metric", metric)
            tuned = row.get("tuned", None)

        default_row = None
        threshold_row = None
        if self.last_test_default_ is not None and len(self.last_test_default_) > 0:
            default_row = self.last_test_default_.iloc[0]
        if self.last_test_threshold_ is not None and len(self.last_test_threshold_) > 0:
            threshold_row = self.last_test_threshold_.iloc[0]

        final_row = threshold_row if threshold_row is not None else default_row
        final_mode = "optimized_threshold" if threshold_row is not None else "default_threshold"

        def val(row, key):
            if row is None:
                return np.nan
            return row.get(key, np.nan)

        report = pd.DataFrame([{
            "target": self.target,
            "positive_label": self.positive_label_,
            "business_goal": self.business_goal_,
            "winning_model": self.best_model_name_,
            "competition_metric": metric,
            "cv_mean": best_cv,
            "cv_std": best_cv_std,
            "winner_tuned": tuned,
            "refit_on": self.refit_on_,
            "decision_mode": final_mode,
            "threshold": self.threshold_ if threshold_row is not None else 0.50,
            "accuracy": val(final_row, "accuracy"),
            "balanced_accuracy": val(final_row, "balanced_accuracy"),
            "precision_positive": val(final_row, "precision_positive"),
            "recall_positive": val(final_row, "recall_positive"),
            "f1_positive": val(final_row, "f1_positive"),
            "roc_auc": val(default_row, "roc_auc"),
            "average_precision": val(default_row, "average_precision"),
            "log_loss": val(default_row, "log_loss"),
            "saved_path": self.saved_path_,
        }])
        self.executive_report_ = report

        if show:
            r = report.iloc[0]
            def fmt(x):
                try:
                    if pd.isna(x):
                        return "Not available"
                    if isinstance(x, (float, np.floating)):
                        return f"{float(x):.{decimals}f}"
                except Exception:
                    pass
                return str(x)

            width = 58
            print("=" * width)
            print("CLASSIFICATION EXECUTIVE REPORT")
            print("=" * width)
            print(f"Target              : {fmt(r['target'])}")
            print(f"Positive class      : {fmt(r['positive_label'])}")
            print(f"Business goal       : {fmt(r['business_goal'])}")
            print(f"Winning model       : {fmt(r['winning_model'])}")
            print(f"Competition metric  : {fmt(r['competition_metric'])}")
            print(f"Cross-validation    : {fmt(r['cv_mean'])} ± {fmt(r['cv_std'])}")
            print(f"Winner tuned        : {fmt(r['winner_tuned'])}")
            print(f"Refit data          : {fmt(r['refit_on'])}")
            print("-" * width)
            print(f"Decision mode       : {fmt(r['decision_mode'])}")
            print(f"Threshold           : {fmt(r['threshold'])}")
            print(f"Accuracy            : {fmt(r['accuracy'])}")
            print(f"Balanced accuracy   : {fmt(r['balanced_accuracy'])}")
            print(f"Precision positive  : {fmt(r['precision_positive'])}")
            print(f"Recall positive     : {fmt(r['recall_positive'])}")
            print(f"F1 positive         : {fmt(r['f1_positive'])}")
            print(f"ROC AUC             : {fmt(r['roc_auc'])}")
            print(f"Average precision   : {fmt(r['average_precision'])}")
            print(f"Log loss            : {fmt(r['log_loss'])}")
            print(f"Saved model         : {fmt(r['saved_path'])}")
            print("=" * width)

        return report

    # ============================================================
    # 1) Register models you requested
    # ============================================================
    def add_default_models(self) -> None:
        self.model_specs_.clear()

        # KNN
        self.model_specs_["KNN"] = ClfModelSpec(
            name="KNN",
            estimator=KNeighborsClassifier(),
            hpo_spaces={
                "grid": {
                    "n_neighbors": [3, 5, 7, 11, 15, 21],
                    "weights": ["uniform", "distance"],
                    "metric": ["minkowski", "manhattan", "euclidean"]
                },
                "random": {
                    "n_neighbors": list(range(3, 61, 2)),
                    "weights": ["uniform", "distance"],
                    "metric": ["minkowski", "manhattan", "euclidean"]
                }
            },
            supports_proba=True,
            tags=("baseline",)
        )

        # SVM (probability=True enables ROC + thresholding)
        self.model_specs_["SVM"] = ClfModelSpec(
            name="SVM",
            estimator=SVC(probability=True),
            hpo_spaces={
                "grid": {
                    "C": [2**p for p in [-5, -3, -1, 1, 3, 5]],
                    "kernel": ["rbf", "linear", "poly", "sigmoid"],
                    "gamma": ["scale", "auto"],
                },
                "random": {
                    "C": [2**p for p in range(-7, 8, 2)],
                    "kernel": ["rbf", "linear", "poly", "sigmoid"],
                    "gamma": ["scale", "auto"],
                }
            },
            supports_proba=True,
            tags=("baseline",)
        )

        # Decision Tree (criterion log_loss may not exist on older sklearn; keep safe)
        self.model_specs_["DecisionTree"] = ClfModelSpec(
            name="DecisionTree",
            estimator=DecisionTreeClassifier(),
            hpo_spaces={
                "grid": {
                    "max_depth": [None, 3, 5, 7, 10, 15],
                    "min_samples_split": [2, 5, 10, 20],
                    "min_samples_leaf": [1, 2, 4, 8],
                    "criterion": ["gini", "entropy"]
                },
                "random": {
                    "max_depth": [None] + list(range(2, 25)),
                    "min_samples_split": [2, 3, 5, 8, 10, 15, 20, 30],
                    "min_samples_leaf": [1, 2, 3, 4, 6, 8, 10],
                    "criterion": ["gini", "entropy"]
                }
            },
            supports_proba=True,
            tags=("tree",)
        )

        # AdaBoost
        self.model_specs_["AdaBoost"] = ClfModelSpec(
            name="AdaBoost",
            estimator=AdaBoostClassifier(),
            hpo_spaces={
                "grid": {
                    "n_estimators": [50, 100, 200, 400],
                    "learning_rate": [0.01, 0.05, 0.1, 0.5, 1.0],
                },
                "random": {
                    "n_estimators": [50, 100, 150, 200, 300, 400, 600],
                    "learning_rate": [0.01, 0.03, 0.05, 0.1, 0.2, 0.5, 1.0],
                }
            },
            supports_proba=True,
            tags=("boosting",)
        )

        # GradientBoostingClassifier
        self.model_specs_["GBC"] = ClfModelSpec(
            name="GBC",
            estimator=GradientBoostingClassifier(),
            hpo_spaces={
                "grid": {
                    "n_estimators": [100, 200, 400],
                    "learning_rate": [0.01, 0.05, 0.1],
                    "max_depth": [2, 3, 5],
                    "subsample": [0.7, 0.85, 1.0]
                },
                "random": {
                    "n_estimators": [100, 200, 300, 400, 600],
                    "learning_rate": [0.01, 0.03, 0.05, 0.1, 0.2],
                    "max_depth": [2, 3, 4, 5],
                    "subsample": [0.6, 0.75, 0.85, 1.0]
                }
            },
            supports_proba=True,
            tags=("boosting",)
        )

        # XGBoost (optional)
        if _HAS_XGB:
            self.model_specs_["XGB"] = ClfModelSpec(
                name="XGB",
                estimator=XGBClassifier(
                    eval_metric="logloss",
                    n_jobs=1
                ),
                hpo_spaces={
                    "grid": {
                        "n_estimators": [200, 400, 800],
                        "max_depth": [2, 3, 5, 7],
                        "learning_rate": [0.01, 0.05, 0.1],
                        "subsample": [0.7, 0.85, 1.0],
                        "colsample_bytree": [0.7, 0.85, 1.0],
                    },
                    "random": {
                        "n_estimators": [200, 300, 400, 600, 800, 1200],
                        "max_depth": [2, 3, 4, 5, 6, 7, 8],
                        "learning_rate": [0.01, 0.03, 0.05, 0.1, 0.2],
                        "subsample": [0.6, 0.7, 0.85, 1.0],
                        "colsample_bytree": [0.6, 0.7, 0.85, 1.0],
                    }
                },
                supports_proba=True,
                tags=("boosting", "xgb")
            )

        # QDA / LDA / Naive Bayes
        self.model_specs_["QDA"] = ClfModelSpec(
            name="QDA",
            estimator=QuadraticDiscriminantAnalysis(),
            hpo_spaces={
                "grid": {"reg_param": [0.0, 0.01, 0.05, 0.1, 0.2]},
                "random": {"reg_param": [0.0, 0.01, 0.03, 0.05, 0.1, 0.2]}
            },
            supports_proba=True,
            tags=("generative",)
        )

        self.model_specs_["LDA"] = ClfModelSpec(
            name="LDA",
            estimator=LinearDiscriminantAnalysis(),
            hpo_spaces={
                "grid": {"solver": ["svd", "lsqr", "eigen"]},
                "random": {"solver": ["svd", "lsqr", "eigen"]}
            },
            supports_proba=True,
            tags=("generative",)
        )

        self.model_specs_["NaiveBayes"] = ClfModelSpec(
            name="NaiveBayes",
            estimator=GaussianNB(),
            hpo_spaces={
                "grid": {},
                "random": {}
            },
            supports_proba=True,
            tags=("baseline", "bayes")
        )

        # MLP (Keras) — se entrena con KerasTuner (no con grid/random)
        self.model_specs_["MLP"] = ClfModelSpec(
            name="MLP",
            estimator=None,  # <- IMPORTANT: keras model se construye vía tuner
            hpo_spaces={
                "keras_tuner": {
                    # defaults; puedes sobreescribir vía tune(..., param_space={...})
                    "max_trials": 30,
                    "executions_per_trial": 1,
                    "epochs": 40,
                    "batch_size": 64,
                    "patience": 6,
                    "val_metric": "val_f1_macro",
                    "objective": "val_f1_macro",
                }
            },
            supports_proba=True,
            tags=("nn", "keras")
)

    # ============================================================
    # 2) HPO (grid / random / halving / bayes / keras_tuner)
    # ============================================================
    def tune(
        self,
        model_name: str,
        method: str = "auto",
        param_space: Optional[Dict[str, Any]] = None,
        n_iter: int = 30,
        factor: int = 3,
        n_jobs: int = -1,
        verbose: int = 0
    ) -> pd.DataFrame:

        if self.X_train_model_ is None:
            raise RuntimeError("Run prepare() first.")
        if model_name not in self.model_specs_:
            raise ValueError(f"Model '{model_name}' not registered. Run add_default_models().")

        spec = self.model_specs_[model_name]
        method = str(method or "auto").lower().strip()

        # ``auto`` chooses a practical search strategy without hiding the choice.
        # - MLP -> KerasTuner.
        # - Small finite grid (<= 40 combinations) -> GridSearchCV.
        # - Larger/continuous spaces -> RandomizedSearchCV.
        if method == "auto":
            if model_name == "MLP":
                method = "keras_tuner"
            else:
                grid_space = param_space if param_space is not None else spec.hpo_spaces.get("grid")
                random_space = param_space if param_space is not None else spec.hpo_spaces.get("random")

                def _finite_grid_size(space):
                    if not isinstance(space, dict) or not space:
                        return None
                    total = 1
                    for values in space.values():
                        if isinstance(values, (str, bytes)) or not hasattr(values, "__len__"):
                            return None
                        try:
                            total *= len(values)
                        except TypeError:
                            return None
                    return int(total)

                grid_size = _finite_grid_size(grid_space)
                if grid_size is not None and grid_size <= 40:
                    method = "grid"
                elif random_space:
                    method = "random"
                elif grid_space:
                    method = "grid"
                else:
                    raise ValueError(f"No HPO space available for model '{model_name}'.")

            print(f"HPO auto-selection | model={model_name} | method={method}")

        if method == "none":
            dfres = pd.DataFrame([{
                "model": model_name,
                "search_method": "none",
                "status": "skipped",
                "message": "Hyperparameter optimization was skipped by request.",
            }])
            self.hpo_results_[model_name] = dfres
            return dfres

        # KerasTuner path
        if method == "keras_tuner":
            return self._tune_keras_tuner(model_name=model_name, verbose=verbose, param_space=param_space)

        est = clone(spec.estimator)
        space = param_space if param_space is not None else spec.hpo_spaces.get(method, None)

        if space is None:
            raise ValueError(f"No param space found for model='{model_name}' method='{method}'.")

        t0 = time.time()

        if method == "grid":
            search = GridSearchCV(
                estimator=est,
                param_grid=space,
                scoring=self.scoring_,
                cv=self.cv_,
                n_jobs=n_jobs,
                verbose=verbose,
                refit=True
            )
        elif method == "random":
            search = RandomizedSearchCV(
                estimator=est,
                param_distributions=space,
                n_iter=n_iter,
                scoring=self.scoring_,
                cv=self.cv_,
                n_jobs=n_jobs,
                verbose=verbose,
                refit=True
            )
        elif method == "halving":
            if not _HAS_HALVING:
                raise ImportError("HalvingRandomSearchCV not available in this environment.")
            search = HalvingRandomSearchCV(
                estimator=est,
                param_distributions=space,
                factor=factor,
                scoring=self.scoring_,
                cv=self.cv_,
                n_jobs=n_jobs,
                verbose=verbose,
                refit=True
            )
        elif method == "bayes":
            if not _HAS_SKOPT:
                raise ImportError("BayesSearchCV not available (install scikit-optimize).")
            search = BayesSearchCV(
                estimator=est,
                search_spaces=space,
                n_iter=n_iter,
                scoring=self.scoring_,
                cv=self.cv_,
                n_jobs=n_jobs,
                verbose=verbose,
                refit=True
            )
        else:
            raise ValueError("method must be: auto | grid | random | halving | bayes | keras_tuner | none")

        y_fit = self._y_train_for_fit()
        search.fit(self.X_train_model_, y_fit)

        elapsed = time.time() - t0

        dfres = pd.DataFrame(search.cv_results_).copy()
        dfres["model"] = model_name
        dfres["search_method"] = method
        dfres["elapsed_s"] = float(elapsed)
        dfres = dfres.sort_values("rank_test_score").reset_index(drop=True)

        self.best_params_[model_name] = {
            "method": method,
            "params": search.best_params_,
            "best_score": float(search.best_score_),
            "elapsed_s": float(elapsed),
            "estimator": search.best_estimator_
        }
        self.hpo_results_[model_name] = dfres
        return dfres

    def tune_models(
        self,
        models: List[str],
        method: str = "auto",
        n_iter: int = 25,
        continue_on_error: bool = True,
        **kwargs,
    ) -> Dict[str, pd.DataFrame]:
        """Tune several registered models with one compact call."""
        results: Dict[str, pd.DataFrame] = {}
        for model_name in models:
            if model_name not in self.model_specs_:
                if continue_on_error:
                    print(f"⚠️ {model_name} skipped: model is not registered.")
                    continue
                raise ValueError(f"Model '{model_name}' is not registered.")
            try:
                results[model_name] = self.tune(
                    model_name=model_name,
                    method=method,
                    n_iter=n_iter,
                    **kwargs,
                )
            except Exception as exc:
                if continue_on_error:
                    print(f"⚠️ {model_name} tuning skipped: {exc}")
                else:
                    raise
        return results

    def tune_top(
        self,
        top: int = 3,
        method: str = "auto",
        n_iter: int = 25,
        include_already_tuned: bool = False,
        continue_on_error: bool = True,
        **kwargs,
    ) -> Dict[str, pd.DataFrame]:
        """Tune the top candidates from the latest competition leaderboard."""
        if self.leaderboard_ is None or len(self.leaderboard_) == 0:
            raise RuntimeError("Run compete() before tune_top().")
        names = self.leaderboard_["model"].astype(str).tolist()
        if not include_already_tuned:
            names = [n for n in names if n not in self.best_params_]
        names = names[:max(1, int(top))]
        return self.tune_models(
            models=names,
            method=method,
            n_iter=n_iter,
            continue_on_error=continue_on_error,
            **kwargs,
        )

    # ============================================================
    # 3) Competition (CV on train)
    # ============================================================

    def compete(
        self,
        models: Optional[List[str]] = None,
        metric: str = "f1_macro",
        n_jobs: int = -1
    ) -> pd.DataFrame:

        if self.X_train_model_ is None:
            raise RuntimeError("Run prepare() first.")

        if not self.model_specs_:
            self.add_default_models()

        if models is None:
            models = list(self.model_specs_.keys())

        y_cv = self._y_train_for_fit()
        scorer = self._resolve_scorer(metric)

        rows = []
        for name in models:
            if name not in self.model_specs_:
                continue

            # ------------------------------------------------------
            # 1) KerasTuner models: cannot do sklearn CV
            # ------------------------------------------------------
            if name in self.best_params_ and self.best_params_[name].get("method") == "keras_tuner":
                best_score = float(self.best_params_[name].get("best_score", np.nan))
                rows.append({
                    "model": name,
                    "metric": metric,
                    "cv_mean": best_score,
                    "cv_std": np.nan,
                    "n_splits": 0,
                    "tuned": True
                })
                continue

            # ------------------------------------------------------
            # 2) Select estimator (tuned or base) — sklearn only
            # ------------------------------------------------------
            if name in self.best_params_ and "estimator" in self.best_params_[name]:
                est = clone(self.best_params_[name]["estimator"])
                tuned = True
            else:
                base_est = self.model_specs_[name].estimator

                # ✅ IMPORTANT: skip models with estimator=None (e.g., MLP not tuned)
                if base_est is None:
                    continue

                est = clone(base_est)
                tuned = False

            # ------------------------------------------------------
            # 3) Cross-validation
            # ------------------------------------------------------
            scores = cross_val_score(
                est,
                self.X_train_model_,
                y_cv,
                scoring=scorer,
                cv=self.cv_,
                n_jobs=n_jobs,
                error_score=np.nan
            )

            rows.append({
                "model": name,
                "metric": metric,
                "cv_mean": float(np.nanmean(scores)),
                "cv_std": float(np.nanstd(scores)),
                "n_splits": int(len(scores)),
                "tuned": bool(tuned)
            })

        lb = pd.DataFrame(rows).sort_values("cv_mean", ascending=False).reset_index(drop=True)
        self.leaderboard_ = lb
        return lb

    # ============================================================
    # 4) Fit final model
    # ============================================================
    def fit(self, model: str = "best", refit_on: str = "train") -> Any:

        if self.X_train_model_ is None:
            raise RuntimeError("Run prepare() first.")

        if model == "best":
            if self.leaderboard_ is None or len(self.leaderboard_) == 0:
                raise RuntimeError("Run compete() first.")
            chosen = str(self.leaderboard_.iloc[0]["model"])
        else:
            chosen = model

        if chosen in self.best_params_ and "estimator" in self.best_params_[chosen]:
            est0 = self.best_params_[chosen]["estimator"]

            # ✅ Keras model (from keras_tuner): do NOT clone
            if (chosen in self.best_params_) and (self.best_params_[chosen].get("method") == "keras_tuner"):
                est = est0
            else:
                est = clone(est0)
        else:
            est = clone(self.model_specs_[chosen].estimator)


        # choose refit data
        X_fit = self.X_train_model_
        y_fit = self._y_train_for_fit()

        if refit_on == "train_valid" and self.X_valid_model_ is not None:
            X_fit = pd.concat([self.X_train_model_, self.X_valid_model_], axis=0).reset_index(drop=True)
            y_fit = pd.concat([self.y_train_enc_, self.y_valid_enc_], axis=0).reset_index(drop=True)

        elif refit_on == "full":
            Xs = [self.X_train_model_]
            ys = [self.y_train_enc_]
            if self.X_valid_model_ is not None:
                Xs.append(self.X_valid_model_)
                ys.append(self.y_valid_enc_)
            Xs.append(self.X_test_model_)
            ys.append(self.y_test_enc_)
            X_fit = pd.concat(Xs, axis=0).reset_index(drop=True)
            y_fit = pd.concat(ys, axis=0).reset_index(drop=True)

        est.fit(X_fit, y_fit)

        self.best_model_name_ = chosen
        self.best_estimator_ = est
        self.refit_on_ = refit_on
        return est

    # ============================================================
    # 5) Predict (decoded by default)
    # ============================================================
    def predict(self, X_model: pd.DataFrame, use_threshold: bool = False) -> np.ndarray:
        if self.best_estimator_ is None:
            raise RuntimeError("Run fit() first.")

        # -----------------------------
        # Normal predict (no threshold)
        # -----------------------------
        if not use_threshold:
            est = self.best_estimator_

            # Keras path: predict -> logits/proba -> class labels
            t = str(type(est)).lower()
            is_keras = ("keras" in t) or ("tensorflow" in t)

            if is_keras:
                X_np = np.asarray(X_model).astype("float32")
                pred = est.predict(X_np, verbose=0)
                pred = np.asarray(pred)

                # binary: (n,), (n,1) -> threshold 0.5 ; (n,2)-> argmax
                if pred.ndim == 1:
                    y_pred_enc = (pred >= 0.5).astype("int32")
                elif pred.ndim == 2 and pred.shape[1] == 1:
                    y_pred_enc = (pred.reshape(-1) >= 0.5).astype("int32")
                else:
                    y_pred_enc = np.argmax(pred, axis=1).astype("int32")

                return self._decode_labels(y_pred_enc)

            # sklearn path
            y_pred_enc = est.predict(X_model)
            return self._decode_labels(np.asarray(y_pred_enc).ravel())

        # -----------------------------
        # Threshold path (binary only)
        # -----------------------------
        # Usar proba universal (sklearn o keras)
        proba = self._predict_proba_any(X_model)

        if proba.ndim != 2 or proba.shape[1] < 2:
            raise ValueError(f"Expected proba shape (n,2+) for thresholding, got {proba.shape}")

        # positive label is stored in ORIGINAL label space; convert to encoded
        if self.positive_encoded_ is None:
            pos_enc = 1  # backward-compatible fallback
        else:
            pos_enc = int(self.positive_encoded_)

        if pos_enc not in (0, 1):
            raise ValueError(
                f"Thresholding expects encoded labels {{0,1}}. Got pos_enc={pos_enc}. "
                "Check LabelEncoder mapping."
            )

        # the other label:
        neg_enc = 1 - pos_enc

        proba_pos = proba[:, pos_enc]
        y_pred_enc = np.where(proba_pos >= float(self.threshold_), pos_enc, neg_enc).astype("int32")

        return self._decode_labels(y_pred_enc)


    # ============================================================
    # 6) Evaluate test (reports on ORIGINAL labels)
    # ============================================================

    def evaluate_test(self, use_threshold: bool = False) -> pd.DataFrame:
        if self.best_estimator_ is None:
            raise RuntimeError("Run fit() first.")

        def _is_keras_model(est) -> bool:
            # robust check without importing keras types
            t = str(type(est)).lower()
            return ("keras" in t) or ("tensorflow" in t)

        def _keras_predict_labels_enc(est, X_df, n_classes: int) -> np.ndarray:
            X_np = X_df.to_numpy().astype("float32")
            pred = est.predict(X_np, verbose=0)

            # Binary: can be (n,1) sigmoid or (n,2) softmax
            if n_classes == 2:
                pred = np.array(pred)
                if pred.ndim == 2 and pred.shape[1] == 2:
                    return np.argmax(pred, axis=1).astype("int32")
                else:
                    p = pred.reshape(-1)
                    return (p >= 0.5).astype("int32")

            # Multiclass: softmax -> argmax
            return np.argmax(pred, axis=1).astype("int32")

        def _keras_predict_proba(est, X_df, n_classes: int) -> np.ndarray:
            X_np = X_df.to_numpy().astype("float32")
            pred = est.predict(X_np, verbose=0)
            pred = np.array(pred)

            if n_classes == 2:
                # If sigmoid (n,1) or (n,), convert to 2-col proba [P0, P1]
                if pred.ndim == 1:
                    p1 = pred
                elif pred.ndim == 2 and pred.shape[1] == 1:
                    p1 = pred.reshape(-1)
                elif pred.ndim == 2 and pred.shape[1] == 2:
                    # already 2-col
                    return pred
                else:
                    p1 = pred.reshape(-1)

                p1 = np.clip(p1, 1e-7, 1 - 1e-7)
                p0 = 1.0 - p1
                return np.column_stack([p0, p1])

            # Multiclass softmax expected
            return pred

        # true labels (encoded) for proba metrics
        y_true_enc = self._y_test_for_fit().to_numpy().ravel()
        y_true = self._decode_labels(y_true_enc)

        n_classes = int(len(np.unique(y_true_enc)))
        is_keras = _is_keras_model(self.best_estimator_)

        # --------------------------------------------------
        # Predicted labels
        # --------------------------------------------------
        if use_threshold:
            # your threshold path already returns decoded labels
            y_pred = self.predict(self.X_test_model_, use_threshold=True)

        else:
            if is_keras:
                y_pred_enc = _keras_predict_labels_enc(self.best_estimator_, self.X_test_model_, n_classes)
            else:
                y_pred_enc = self.best_estimator_.predict(self.X_test_model_)

            y_pred = self._decode_labels(np.array(y_pred_enc).ravel())

        # -----------------------------
        # Global metrics (macro)
        # -----------------------------
        out = {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
            "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
            "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
            "f1_macro": float(f1_score(y_true, y_pred, average="macro")),
            "use_threshold": bool(use_threshold),
            "threshold": float(self.threshold_) if use_threshold else np.nan,
        }

        # -----------------------------
        # Per-class metrics (dynamic)
        # -----------------------------
        rep = classification_report(
            y_true,
            y_pred,
            output_dict=True,
            zero_division=0
        )

        out["precision_weighted"] = float(rep["weighted avg"]["precision"])
        out["recall_weighted"] = float(rep["weighted avg"]["recall"])
        out["f1_weighted"] = float(rep["weighted avg"]["f1-score"])

        class_keys = [k for k in rep.keys() if k not in ["accuracy", "macro avg", "weighted avg"]]
        for cls_name in sorted(class_keys):
            stats = rep[cls_name]
            out[f"{cls_name}_precision"] = float(stats["precision"])
            out[f"{cls_name}_recall"] = float(stats["recall"])
            out[f"{cls_name}_f1"] = float(stats["f1-score"])
            out[f"{cls_name}_support"] = int(stats["support"])

        if self.positive_label_ is not None:
            pos_key = str(self.positive_label_)
            if pos_key in rep:
                out["positive_label"] = self.positive_label_
                out["precision_positive"] = float(rep[pos_key]["precision"])
                out["recall_positive"] = float(rep[pos_key]["recall"])
                out["f1_positive"] = float(rep[pos_key]["f1-score"])
                out["support_positive"] = int(rep[pos_key]["support"])

        # -----------------------------
        # Proba metrics (encoded space)
        # -----------------------------
        proba = None
        if not use_threshold:
            if hasattr(self.best_estimator_, "predict_proba") and not is_keras:
                try:
                    proba = self.best_estimator_.predict_proba(self.X_test_model_)
                except Exception:
                    proba = None
            elif is_keras:
                try:
                    proba = _keras_predict_proba(self.best_estimator_, self.X_test_model_, n_classes)
                except Exception:
                    proba = None

        if proba is not None:
            try:
                out["log_loss"] = float(log_loss(y_true_enc, proba))
            except Exception:
                out["log_loss"] = np.nan

            if len(np.unique(y_true_enc)) == 2:
                try:
                    pos_enc = 1 if self.positive_encoded_ is None else int(self.positive_encoded_)
                    y_true_pos = (y_true_enc == pos_enc).astype(int)
                    proba_pos = proba[:, pos_enc]
                    out["roc_auc"] = float(roc_auc_score(y_true_pos, proba_pos))
                    out["average_precision"] = float(average_precision_score(y_true_pos, proba_pos))
                except Exception:
                    out["roc_auc"] = np.nan
                    out["average_precision"] = np.nan
            else:
                out["roc_auc"] = np.nan
                out["average_precision"] = np.nan
        else:
            out["log_loss"] = np.nan
            out["roc_auc"] = np.nan
            out["average_precision"] = np.nan

        # -----------------------------
        # Make output "pretty": reorder columns
        # -----------------------------
        base_cols = [
            "accuracy", "balanced_accuracy",
            "precision_macro", "recall_macro", "f1_macro",
            "precision_weighted", "recall_weighted", "f1_weighted",
            "log_loss", "roc_auc", "average_precision",
            "positive_label", "precision_positive", "recall_positive", "f1_positive", "support_positive",
            "use_threshold", "threshold",
        ]

        class_cols = []
        for cls_name in sorted(class_keys):
            class_cols += [
                f"{cls_name}_precision",
                f"{cls_name}_recall",
                f"{cls_name}_f1",
                f"{cls_name}_support",
            ]

        ordered_cols = [c for c in base_cols if c in out] + [c for c in class_cols if c in out]
        df_out = pd.DataFrame([out])

        remaining = [c for c in df_out.columns if c not in ordered_cols]
        df_out = df_out[ordered_cols + remaining]

        if use_threshold:
            self.last_test_threshold_ = df_out.copy()
        else:
            self.last_test_default_ = df_out.copy()

        return df_out

    # ============================================================
    # 7) Optimize threshold (binary; objective on positive class)
    # ============================================================
    def optimize_threshold(
        self,
        positive_label: Any = None,
        objective: str = "precision",     # precision | recall | f1
        constraint: Optional[Dict[str, float]] = None,
        grid: Optional[np.ndarray] = None,
        use: str = "valid"                # valid | test
    ) -> pd.DataFrame:

        if self.best_estimator_ is None:
            raise RuntimeError("Run fit() first.")

        if objective not in ("precision", "recall", "f1"):
            raise ValueError("objective must be one of: 'precision', 'recall', 'f1'")

        if positive_label is None:
            positive_label = self.positive_label_
        if positive_label is None:
            raise ValueError(
                "No positive class is configured. Pass positive_label in the constructor "
                "or to optimize_threshold()."
            )

        if grid is None:
            grid = np.round(np.arange(0.05, 0.96, 0.01), 2)

        # -----------------------------
        # Select eval split
        # -----------------------------
        if use == "valid":
            if self.X_valid_model_ is None:
                raise RuntimeError("No validation split available. Re-run prepare(valid_size>0).")
            X_eval = self.X_valid_model_
            y_eval_enc = self.y_valid_enc_.to_numpy()
        else:
            X_eval = self.X_test_model_
            y_eval_enc = self.y_test_enc_.to_numpy()

        # Ensure binary based on encoded labels (robust for keras)
        uniq = np.unique(y_eval_enc)
        if len(uniq) != 2:
            raise ValueError("Threshold optimization supported only for binary classification.")

        # Encoded label for the positive class
        pos_enc = int(self._encode_label(positive_label))

        if pos_enc not in list(uniq):
            raise ValueError(
                f"positive_label='{positive_label}' encodes to {pos_enc}, but y_eval has labels {list(uniq)}"
            )

        # Neg label is "the other one"
        neg_enc = int(uniq[0] if uniq[1] == pos_enc else uniq[1])

        # -----------------------------
        # Probabilities (sklearn or keras)
        # -----------------------------
        proba = self._predict_proba_any(X_eval)

        # We need P(pos)
        # - If proba has 2 columns, assume column index == encoded label (0/1)
        # - If it doesn't, fail explicitly
        if proba.ndim != 2 or proba.shape[1] < 2:
            raise ValueError(f"Expected proba shape (n,2+) for binary, got {proba.shape}")

        if pos_enc not in (0, 1):
            # extremely uncommon with LabelEncoder, but guard anyway
            raise ValueError(
                f"Encoded labels are {list(uniq)}; expected {{0,1}} for thresholding with proba[:, pos_index]."
            )

        proba_pos = proba[:, pos_enc]

        # -----------------------------
        # Grid search over thresholds
        # -----------------------------
        rows = []
        for th in grid:
            y_pred_enc = np.where(proba_pos >= th, pos_enc, neg_enc)

            prec = precision_score(y_eval_enc, y_pred_enc, pos_label=pos_enc, zero_division=0)
            rec  = recall_score(y_eval_enc, y_pred_enc, pos_label=pos_enc, zero_division=0)
            f1v  = f1_score(y_eval_enc, y_pred_enc, pos_label=pos_enc, zero_division=0)

            ok = True
            if constraint:
                if "recall_min" in constraint:
                    ok = ok and (rec >= float(constraint["recall_min"]))
                if "precision_min" in constraint:
                    ok = ok and (prec >= float(constraint["precision_min"]))

            score = {"precision": prec, "recall": rec, "f1": f1v}[objective]

            rows.append({
                "threshold": float(th),
                "precision_pos": float(prec),
                "recall_pos": float(rec),
                "f1_pos": float(f1v),
                "objective": objective,
                "objective_value": float(score),
                "passes_constraint": bool(ok)
            })

        df = pd.DataFrame(rows)
        df_ok = df[df["passes_constraint"]].copy()
        if len(df_ok) == 0:
            df_ok = df.copy()

        best_row = df_ok.sort_values("objective_value", ascending=False).iloc[0]
        self.threshold_ = float(best_row["threshold"])
        self.threshold_report_ = df
        self.positive_label_ = positive_label

        return df

    def _predict_proba_any(self, X) -> np.ndarray:
        """
        Returns probabilities in sklearn-like format:
        - binary: shape (n, 2) with columns [P(class0), P(class1)] in encoded label order [0, 1]
        - multiclass: shape (n, K)
        """
        est = self.best_estimator_

        # sklearn path
        if hasattr(est, "predict_proba"):
            return np.asarray(est.predict_proba(X))

        # keras/tf path
        if hasattr(est, "predict"):
            X_np = np.asarray(X).astype("float32")
            p = est.predict(X_np, verbose=0)
            p = np.asarray(p)

            # (n,) -> (n,2)
            if p.ndim == 1:
                p1 = p.reshape(-1, 1)
                p1 = np.clip(p1, 1e-7, 1 - 1e-7)
                return np.hstack([1 - p1, p1])

            # (n,1) -> (n,2)
            if p.ndim == 2 and p.shape[1] == 1:
                p1 = np.clip(p, 1e-7, 1 - 1e-7)
                return np.hstack([1 - p1, p1])

            # (n,2) or (n,K) already
            if p.ndim == 2:
                return p

        raise ValueError("Model does not support probability prediction (predict_proba or keras predict).")


    # ============================================================
    # 8) Save (pickle) / Save keras (h5)
    # ============================================================

    def save(self, path: str, kind: str = "pickle") -> str:
        if self.best_estimator_ is None:
            raise RuntimeError("Run fit() first.")

        # --------------------------------------------------------
        # Helper: best CV from leaderboard (if available)
        # --------------------------------------------------------
        best_cv = None
        if self.leaderboard_ is not None and len(self.leaderboard_) > 0:
            try:
                best_cv = float(self.leaderboard_.iloc[0]["cv_mean"])
            except Exception:
                best_cv = None

        # --------------------------------------------------------
        # Detect if model is Keras (robust)
        # --------------------------------------------------------
        est = self.best_estimator_
        is_keras = False
        try:
            # works for tf.keras.Model and keras.Model
            if hasattr(est, "save") and hasattr(est, "fit") and hasattr(est, "predict"):
                mod = type(est).__module__.lower()
                if ("keras" in mod) or ("tensorflow" in mod):
                    is_keras = True
        except Exception:
            is_keras = False

        kind = (kind or "pickle").lower()

        # --------------------------------------------------------
        # If Keras model: prefer h5/keras saving
        # --------------------------------------------------------
        if is_keras and kind == "pickle":
            print("⚠️ Detected a Keras model. Switching save kind from 'pickle' to 'h5'.")
            kind = "h5"

            # if path ends with .pkl, switch to .h5 to avoid confusion
            if path.lower().endswith(".pkl"):
                path = path[:-4] + ".h5"

        # --------------------------------------------------------
        # Pickle path: sklearn-like models
        # --------------------------------------------------------
        if kind == "pickle":
            bundle = {
                # core objects needed for inference
                "preprocessor_model": self.preprocessor_model_,
                "feature_names_model": self.feature_names_model_,
                "label_encoder": self.label_encoder_,

                # model
                "model_name": self.best_model_name_,
                "model": self.best_estimator_,

                # thresholding (optional at inference time)
                "threshold": float(self.threshold_),
                "positive_label": self.positive_label_,
                "positive_encoded": self.positive_encoded_,
                "negative_label": self.negative_label_,
                "negative_encoded": self.negative_encoded_,
                "class_of_interest": self.class_of_interest_,

                # diagnostics / reproducibility
                "leaderboard": self.leaderboard_,
                "best_cv": best_cv,

                "meta": {
                    "scoring": self.scoring_,
                    "classes_original": list(self.label_encoder_.classes_) if self.label_encoder_ is not None else None,
                }
            }

            joblib.dump(bundle, path)

            print(
                f"✅ Saved -> {path}\n"
                f"   best_model: {self.best_model_name_}\n"
                f"   best_cv: {best_cv}\n"
                f"   threshold: {self.threshold_} | positive_label: {self.positive_label_}"
            )
            self.saved_path_ = path
            return path

        # --------------------------------------------------------
        # Keras save path: .h5 (or SavedModel depending on TF)
        # --------------------------------------------------------
        elif kind in ("h5", "keras"):
            model = self.best_estimator_
            try:
                model.save(path)
            except Exception as e:
                raise RuntimeError(f"Failed to save keras model: {e}")

            sidecar = path + ".preprocess.pkl"
            meta = path + ".meta.json"

            joblib.dump(
                {
                    "preprocessor_model": self.preprocessor_model_,
                    "feature_names_model": self.feature_names_model_,
                    "label_encoder": self.label_encoder_,
                    "leaderboard": self.leaderboard_,
                    "best_cv": best_cv,
                },
                sidecar
            )

            with open(meta, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "threshold": float(self.threshold_),
                        "positive_label": None if self.positive_label_ is None else str(self.positive_label_),
                        "model_name": self.best_model_name_,
                        "best_cv": best_cv,
                        "scoring": self.scoring_,
                        "classes_original": list(self.label_encoder_.classes_) if self.label_encoder_ is not None else None,
                    },
                    f,
                    ensure_ascii=False,
                    indent=2
                )

            print(
                f"✅ Saved Keras -> {path}\n"
                f"   sidecar preprocess: {sidecar}\n"
                f"   meta json: {meta}\n"
                f"   best_model: {self.best_model_name_} | best_cv: {best_cv}\n"
                f"   threshold: {self.threshold_} | positive_label: {self.positive_label_}"
            )
            self.saved_path_ = path
            return path

        else:
            raise ValueError("kind must be 'pickle' or 'h5'/'keras'.")

    # ============================================================
    # (Keras) Helpers: build model + f1 macro metric
    # ============================================================
    def _keras_f1_macro_metric(self):
        """
        Keras metric: F1 macro for binary/multiclass.
        We implement it as a callback-like metric via tf ops.
        Note: this is an approximation on batch-level; good enough for tuning.
        """
        if not _HAS_TF:
            raise ImportError("TensorFlow is not available.")

        def f1_macro(y_true, y_pred):
            y_true = tf.cast(y_true, tf.int32)

            # If y_pred is probability/logits, convert to class predictions
            if tf.rank(y_pred) == 2:
                # multiclass: argmax
                y_pred_labels = tf.argmax(y_pred, axis=1, output_type=tf.int32)
            else:
                # binary: threshold at 0.5
                y_pred_labels = tf.cast(y_pred >= 0.5, tf.int32)

            y_true_flat = tf.reshape(y_true, [-1])
            y_pred_flat = tf.reshape(y_pred_labels, [-1])

            # Number of classes inferred from y_true in this batch
            n_classes = tf.reduce_max(y_true_flat) + 1

            def per_class_f1(c):
                c = tf.cast(c, tf.int32)
                tp = tf.reduce_sum(tf.cast((y_true_flat == c) & (y_pred_flat == c), tf.float32))
                fp = tf.reduce_sum(tf.cast((y_true_flat != c) & (y_pred_flat == c), tf.float32))
                fn = tf.reduce_sum(tf.cast((y_true_flat == c) & (y_pred_flat != c), tf.float32))

                precision = tp / (tp + fp + 1e-8)
                recall = tp / (tp + fn + 1e-8)
                return 2.0 * precision * recall / (precision + recall + 1e-8)

            f1s = tf.map_fn(per_class_f1, tf.range(n_classes), fn_output_signature=tf.float32)
            return tf.reduce_mean(f1s)

        return f1_macro

    def _build_mlp_from_hp(self, hp, input_dim: int, n_classes: int):

        model = keras.Sequential()
        model.add(layers.Input(shape=(input_dim,)))

        # HPO: capas / units / dropout / l2
        n_layers = hp.Int("n_layers", 1, 3)
        for i in range(n_layers):
            units = hp.Int(f"units_{i}", min_value=32, max_value=256, step=32)
            dropout = hp.Float(f"dropout_{i}", 0.0, 0.5, step=0.1)
            l2v = hp.Float(f"l2_{i}", 1e-6, 1e-3, sampling="log")

            model.add(layers.Dense(units, activation="relu",
                                kernel_regularizer=regularizers.l2(l2v)))
            if dropout > 0:
                model.add(layers.Dropout(dropout))

        if n_classes == 2:
            model.add(layers.Dense(1, activation="sigmoid"))
            loss = "binary_crossentropy"
            metrics = [
                keras.metrics.BinaryAccuracy(name="accuracy"),
                keras.metrics.AUC(name="auc"),
            ]
        else:
            model.add(layers.Dense(n_classes, activation="softmax"))
            loss = "sparse_categorical_crossentropy"
            metrics = [
                keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
            ]

        lr = hp.Float("lr", 1e-4, 5e-3, sampling="log")
        opt = keras.optimizers.Adam(learning_rate=lr)

        model.compile(optimizer=opt, loss=loss, metrics=metrics)
        return model

    # ============================================================
    # 9) KerasTuner
    # ============================================================
    def _tune_keras_tuner(
        self,
        model_name: str,
        verbose: int = 0,
        param_space: Optional[Dict[str, Any]] = None
    ) -> pd.DataFrame:

        if self.X_train_model_ is None:
            raise RuntimeError("Run prepare() first.")
        if not _HAS_TF:
            raise ImportError("TensorFlow/Keras not available. Install tensorflow.")
        if not _HAS_KT:
            raise ImportError("keras-tuner not available. Install keras-tuner.")

        if model_name not in self.model_specs_:
            raise ValueError(f"Model '{model_name}' not registered.")
        if model_name != "MLP":
            raise ValueError("keras_tuner currently implemented for model_name='MLP' only.")

        # --------------------------------------------------------
        # Defaults from spec + overrides
        # --------------------------------------------------------
        spec_defaults = self.model_specs_[model_name].hpo_spaces.get("keras_tuner", {})
        cfg = dict(spec_defaults)
        if param_space:
            cfg.update(param_space)

        max_trials = int(cfg.get("max_trials", 30))
        executions_per_trial = int(cfg.get("executions_per_trial", 1))
        epochs = int(cfg.get("epochs", 40))
        batch_size = int(cfg.get("batch_size", 64))
        patience = int(cfg.get("patience", 6))

        # --------------------------------------------------------
        # ✅ Tuner directory: ALWAYS local
        # --------------------------------------------------------
        tuner_dir = cfg.get("tuner_dir", None)
        if tuner_dir is None:
            tuner_dir = os.path.join(os.path.expanduser("~"), "kt_dir")
        if not os.path.isabs(tuner_dir):
            tuner_dir = os.path.join(os.path.expanduser("~"), tuner_dir)

        os.makedirs(tuner_dir, exist_ok=True)
        project_name = cfg.get("project_name", f"{self.target}_MLP")

        # --------------------------------------------------------
        # Train/valid split for tuner
        # --------------------------------------------------------
        if self.X_valid_model_ is None:
            X_tr, X_va, y_tr, y_va = train_test_split(
                self.X_train_model_,
                self.y_train_enc_,
                test_size=0.10,
                stratify=self.y_train_enc_
            )
        else:
            X_tr, y_tr = self.X_train_model_, self.y_train_enc_
            X_va, y_va = self.X_valid_model_, self.y_valid_enc_

        X_tr_np = X_tr.to_numpy().astype("float32")
        X_va_np = X_va.to_numpy().astype("float32")
        y_tr_np = np.array(y_tr).astype("int32")
        y_va_np = np.array(y_va).astype("int32")

        input_dim = X_tr_np.shape[1]
        n_classes = int(len(np.unique(y_tr_np)))

        # --------------------------------------------------------
        # ✅ Objective MUST exist in Keras logs
        #    - binario: val_auc
        #    - multiclase: val_accuracy
        #    Si cfg trae algo inválido (ej val_f1_macro), hacemos fallback
        # --------------------------------------------------------
        allowed_objectives = {"val_auc", "val_accuracy", "val_loss"}

        default_objective = "val_auc" if n_classes == 2 else "val_accuracy"
        objective = cfg.get("objective", default_objective)

        if objective not in allowed_objectives:
            # fallback (NO romper la corrida)
            objective = default_objective

        direction = "min" if objective == "val_loss" else "max"

        # --------------------------------------------------------
        # Build model function
        # --------------------------------------------------------
        def build_model(hp):
            return self._build_mlp_from_hp(hp, input_dim=input_dim, n_classes=n_classes)

        # ✅ stability
        tf.keras.backend.clear_session()

        tuner = kt.RandomSearch(
            hypermodel=build_model,
            objective=kt.Objective(objective, direction=direction),
            max_trials=max_trials,
            executions_per_trial=executions_per_trial,
            directory=tuner_dir,
            project_name=project_name,
            overwrite=True
        )

        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor=objective,
                patience=patience,
                mode="min" if objective == "val_loss" else "max",
                restore_best_weights=True
            )
        ]

        t0 = time.time()
        tuner.search(
            X_tr_np, y_tr_np,
            validation_data=(X_va_np, y_va_np),
            epochs=epochs,
            batch_size=batch_size,
            verbose=verbose,
            callbacks=callbacks
        )
        elapsed = time.time() - t0

        best_hp = tuner.get_best_hyperparameters(num_trials=1)[0]
        best_model = tuner.get_best_models(num_models=1)[0]

        # --------------------------------------------------------
        # ✅ Compute best_score with sklearn (F1 macro)
        # --------------------------------------------------------
        pred = best_model.predict(X_va_np, verbose=0)

        if n_classes == 2:
            # soporta salida sigmoid (N,) o (N,1) o softmax (N,2)
            if pred.ndim == 2 and pred.shape[1] == 2:
                y_hat = np.argmax(pred, axis=1).astype("int32")
            else:
                p = pred.reshape(-1)
                y_hat = (p >= 0.5).astype("int32")
        else:
            y_hat = np.argmax(pred, axis=1).astype("int32")

        best_score = float(f1_score(y_va_np, y_hat, average="macro"))

        # Store for compete()/fit()
        self.best_params_[model_name] = {
            "method": "keras_tuner",
            "params": best_hp.values,
            "best_score": best_score,
            "elapsed_s": float(elapsed),
            "estimator": best_model,
            "objective": objective  # <- útil para debug
        }

        dfres = pd.DataFrame([{
            "model": model_name,
            "search_method": "keras_tuner",
            "objective": objective,
            "best_score": best_score,
            "elapsed_s": float(elapsed),
            "params": json.dumps(best_hp.values, ensure_ascii=False)
        }])

        self.hpo_results_[model_name] = dfres
        return dfres

# ==========================================================
# SupervisedRegression
# - Split: train / valid / test
# - Preprocess: num scaling (StandardScaler por default) + cat encoding (OneHot/Ordinal)
# - Model registry: Linear, Lasso, Ridge, SVR, DecisionTree, RandomForest, GradientBoosting (+ opcional MLP Keras)
# - HPO: GridSearchCV / RandomizedSearchCV (n_jobs=-1)
# - Competition: CV leaderboard (RMSE por default)
# - Fit best + evaluate_test (RMSE/MAE/R2) + save/load
# ==========================================================

# -----------------------------
# Utilities
# -----------------------------
def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def _rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))

RMSE_SCORER = make_scorer(
    lambda yt, yp: -_rmse(yt, yp),  # sklearn scoring: "higher is better"
    greater_is_better=True
)

class KerasRegressorSK(BaseEstimator, RegressorMixin):
    """
    Wrapper sklearn-compatible:
    - Implementa API fit/predict
    - Es cloneable (por BaseEstimator)
    - NO guarda objetos no-cloneables en __init__ (solo configs)
    """
    def __init__(
        self,
        preprocess,
        hp_values: dict,
        build_fn,                 # callable: (hp_values, input_dim) -> keras.Model
        epochs: int = 40,
        batch_size: int = 64,
        patience: int = 6,
        random_state: int | None = None,
        verbose: int = 0,
    ):
        self.preprocess = preprocess
        self.hp_values = hp_values
        self.build_fn = build_fn
        self.epochs = epochs
        self.batch_size = batch_size
        self.patience = patience
        self.random_state = random_state
        self.verbose = verbose

        # se setean en fit()
        self.preprocess_ = None
        self.model_ = None

    def fit(self, X, y):
        self.preprocess_ = clone(self.preprocess)

        # split interno para early stopping (siempre crea val_rmse)
        X_tr, X_va, y_tr, y_va = train_test_split(
            X, y, test_size=0.15, random_state=self.random_state
        )

        Xt = self.preprocess_.fit_transform(X_tr)
        Xv = self.preprocess_.transform(X_va)

        input_dim = Xt.shape[1]
        self.model_ = self.build_fn(self.hp_values, input_dim)

        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor="val_rmse",
                patience=self.patience,
                restore_best_weights=True,
            )
        ]

        self.model_.fit(
            Xt, y_tr,
            validation_data=(Xv, y_va),
            epochs=self.epochs,
            batch_size=self.batch_size,
            callbacks=callbacks,
            verbose=self.verbose
        )
        return self

    def predict(self, X):
        Xt = self.preprocess_.transform(X)
        return self.model_.predict(Xt, verbose=0).ravel()

@dataclass
class ModelSpec:
    estimator: Any
    param_grid: Optional[Dict[str, Any]] = None
    param_dist: Optional[Dict[str, Any]] = None

class SupervisedRegression(ModelInterpretabilityMixin, DFCore):
    """
    SupervisedRegression(df, target="y")

    Workflow:
      sup = SupervisedRegression(df, target="resistencia_compresion")
      sup.prepare(...)
      sup.add_default_models()
      sup.tune("GBR", method="random", n_iter=25)
      lb = sup.compete(metric="rmse")
      sup.fit(model="best", refit_on="train_valid")
      sup.evaluate_test()
      sup.save("best_model.pkl")

    Notas:
      - scoring interno para CV: por defecto usa NEG_RMSE (equivalente a minimizar RMSE)
      - compete() reporta RMSE positivo (más bajo es mejor) aunque internamente use negativo.
    """

    # -----------------------------
    # Init
    # -----------------------------
    def __init__(
        self,
        df: pd.DataFrame,
        target: str,
        features: Optional[List[str]] = None,
        output_dir: str = "outputs_reg",
    ):
        # ✅ hereda y centraliza: df, target, features, cols, validación, etc.
        super().__init__(df=df, target=target, features=features)

        self.output_dir = output_dir
        _ensure_dir(self.output_dir)

        # Learned artifacts
        self.preprocess_: Optional[ColumnTransformer] = None
        self.model_specs_: Dict[str, ModelSpec] = {}
        self.tuned_models_: Dict[str, Dict[str, Any]] = {}  # {name: {"best_estimator":..., "best_params":..., ...}}

        # Data splits
        self.X_train_: Optional[pd.DataFrame] = None
        self.y_train_: Optional[np.ndarray] = None
        self.X_valid_: Optional[pd.DataFrame] = None
        self.y_valid_: Optional[np.ndarray] = None
        self.X_test_: Optional[pd.DataFrame] = None
        self.y_test_: Optional[np.ndarray] = None

        # CV / objective
        self.cv_: Optional[KFold] = None
        self.scoring_: str = "neg_root_mean_squared_error"  # default sklearn string
        self.random_state_: Optional[int] = None  # por default None (no forzar reproducibilidad)

        # Best
        self.best_model_name_: Optional[str] = None
        self.best_estimator_: Optional[Pipeline] = None
        self.best_cv_: Optional[Dict[str, Any]] = None


    # -----------------------------
    # Core helpers
    # -----------------------------
    def _infer_features(self) -> List[str]:
        if self.features is not None:
            return list(self.features)
        return [c for c in self.df.columns if c != self.target]

    def _split_data(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        test_size: float,
        valid_size: float,
        stratify: bool = False,  # en regresión normalmente False
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
        # Test split
        strat = None
        if stratify:
            # En regresión no es común; si lo activan, discretizamos y estratificamos por bins
            y_bins = pd.qcut(y, q=10, duplicates="drop")
            strat = y_bins

        X_train_valid, X_test, y_train_valid, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state_, stratify=strat
        )

        # Valid split from train_valid
        # valid_size es proporción del TOTAL; convertir a proporción dentro de train_valid:
        tv_size = 1.0 - test_size
        valid_ratio_in_tv = valid_size / tv_size

        strat_tv = None
        if stratify:
            ytv_bins = pd.qcut(y_train_valid, q=10, duplicates="drop")
            strat_tv = ytv_bins

        X_train, X_valid, y_train, y_valid = train_test_split(
            X_train_valid,
            y_train_valid,
            test_size=valid_ratio_in_tv,
            random_state=self.random_state_,
            stratify=strat_tv,
        )

        return X_train, X_valid, X_test, y_train, y_valid, y_test

    def _build_preprocess(
        self,
        X: pd.DataFrame,
        model_encoding: str = "auto",
        scale_num: str = "standard",
        auto_cardinality_threshold: int = 15,
    ) -> ColumnTransformer:
        # Separar num/cat por dtype
        num_cols = X.select_dtypes(include=["number", "bool"]).columns.tolist()
        cat_cols = [c for c in X.columns if c not in num_cols]

        # Numeric pipeline
        if scale_num == "standard":
            scaler = StandardScaler()
        elif scale_num == "minmax":
            scaler = MinMaxScaler()
        elif scale_num == "robust":
            scaler = RobustScaler()
        elif scale_num in (None, "none", "passthrough"):
            scaler = "passthrough"
        else:
            raise ValueError(f"scale_num='{scale_num}' no soportado. Use: standard|minmax|robust|none")

        num_pipe = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", scaler),
            ]
        )

        # Categorical pipeline
        # auto: onehot para baja cardinalidad; ordinal para alta (evitar explosión)
        cat_steps = [("imputer", SimpleImputer(strategy="most_frequent"))]

        if model_encoding == "auto":
            low_card = []
            high_card = []
            for c in cat_cols:
                nunique = X[c].nunique(dropna=True)
                if nunique <= auto_cardinality_threshold:
                    low_card.append(c)
                else:
                    high_card.append(c)

            transformers = []
            if low_card:
                transformers.append(
                    ("cat_onehot", Pipeline(steps=cat_steps + [
                        ("enc", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
                    ]), low_card)
                )
            if high_card:
                transformers.append(
                    ("cat_ordinal", Pipeline(steps=cat_steps + [
                        ("enc", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))
                    ]), high_card)
                )

            # Combinar con numeric
            pre = ColumnTransformer(
                transformers=[("num", num_pipe, num_cols)] + transformers,
                remainder="drop",
                verbose_feature_names_out=False,
            )
            return pre

        elif model_encoding == "onehot":
            cat_pipe = Pipeline(steps=cat_steps + [
                ("enc", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
            ])
        elif model_encoding == "ordinal":
            cat_pipe = Pipeline(steps=cat_steps + [
                ("enc", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))
            ])
        elif model_encoding in (None, "none", "passthrough"):
            cat_pipe = "passthrough"
        else:
            raise ValueError(f"model_encoding='{model_encoding}' no soportado. Use: auto|onehot|ordinal|none")

        pre = ColumnTransformer(
            transformers=[
                ("num", num_pipe, num_cols),
                ("cat", cat_pipe, cat_cols),
            ],
            remainder="drop",
            verbose_feature_names_out=False,
        )
        return pre

    def _make_pipeline(self, estimator) -> Pipeline:
        if self.preprocess_ is None:
            raise RuntimeError("Primero ejecute prepare() para construir preprocess_.")
        return Pipeline(steps=[("preprocess", self.preprocess_), ("model", estimator)])

    # -----------------------------
    # Public API
    # -----------------------------
    def prepare(
        self,
        test_size: float = 0.20,
        valid_size: float = 0.15,
        stratify: bool = False,
        scoring: str = "rmse",         # "rmse"|"mae"|"r2"
        cv_splits: int = 5,
        model_encoding: str = "auto",
        scale_num: str = "standard",
        auto_cardinality_threshold: int = 15,
        random_state: Optional[int] = None,  # si quieres fijar, pásalo; si no, queda None
    ) -> "SupervisedRegression":
        self.random_state_ = random_state

        feats = self._infer_features()
        X = self.df[feats].copy()
        y = self.df[self.target].values.astype(float)

        # splits
        X_train, X_valid, X_test, y_train, y_valid, y_test = self._split_data(
            X, y, test_size=test_size, valid_size=valid_size, stratify=stratify
        )

        self.X_train_, self.y_train_ = X_train, y_train
        self.X_valid_, self.y_valid_ = X_valid, y_valid
        self.X_test_, self.y_test_ = X_test, y_test

        # preprocess (fit happens inside CV/pipeline; we store transformer object configured)
        self.preprocess_ = self._build_preprocess(
            X_train,
            model_encoding=model_encoding,
            scale_num=scale_num,
            auto_cardinality_threshold=auto_cardinality_threshold,
        )

        # CV
        self.cv_ = KFold(n_splits=cv_splits, shuffle=True, random_state=self.random_state_)

        # scoring mapping (sklearn uses higher=better)
        scoring = (scoring or "rmse").lower()
        if scoring == "rmse":
            self.scoring_ = "neg_root_mean_squared_error"
        elif scoring == "mae":
            self.scoring_ = "neg_mean_absolute_error"
        elif scoring == "r2":
            self.scoring_ = "r2"
        else:
            raise ValueError("scoring debe ser: rmse | mae | r2")

        return self

    def add_default_models(self) -> "SupervisedRegression":
        """
        Registra SIEMPRE los modelos:
          - LinearRegression
          - Lasso
          - Ridge
          - SVR (rbf)
          - DecisionTreeRegressor
          - RandomForestRegressor
          - GradientBoostingRegressor

        Nota: params están pensados para HPO; si no tuneas un modelo, igual compite con defaults.
        """
        self.model_specs_.clear()

        # 1) Linear
        self.model_specs_["Linear"] = ModelSpec(
            estimator=LinearRegression(),
            param_grid=None,
            param_dist=None,
        )

        # 2) Lasso
        self.model_specs_["Lasso"] = ModelSpec(
            estimator=Lasso(max_iter=5000),
            param_grid={"model__alpha": [1e-4, 1e-3, 1e-2, 1e-1, 1.0]},
            param_dist={"model__alpha": np.logspace(-4, 0, 50)},
        )

        # 3) Ridge
        self.model_specs_["Ridge"] = ModelSpec(
            estimator=Ridge(solver="auto"),
            param_grid={"model__alpha": [1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0]},
            param_dist={"model__alpha": np.logspace(-3, 2, 80)},
        )

        # 4) SVR (RBF)
        self.model_specs_["SVR"] = ModelSpec(
            estimator=SVR(kernel="rbf"),
            param_grid={
                "model__C": [0.1, 1, 10, 100],
                "model__epsilon": [0.01, 0.05, 0.1, 0.2],
                "model__gamma": ["scale", "auto"],
            },
            param_dist={
                "model__C": np.logspace(-1, 2, 60),
                "model__epsilon": np.linspace(0.01, 0.3, 60),
                "model__gamma": ["scale", "auto"],
            },
        )

        # 5) Decision Tree
        self.model_specs_["DecisionTree"] = ModelSpec(
            estimator=DecisionTreeRegressor(),
            param_grid={
                "model__max_depth": [2, 3, 4, 5, None],
                "model__min_samples_split": [2, 5, 10, 20],
                "model__min_samples_leaf": [1, 2, 4, 8],
            },
            param_dist={
                "model__max_depth": [None] + list(range(2, 21)),
                "model__min_samples_split": list(range(2, 31)),
                "model__min_samples_leaf": list(range(1, 21)),
            },
        )

        # 6) Random Forest
        self.model_specs_["RandomForest"] = ModelSpec(
            estimator=RandomForestRegressor(n_estimators=300),
            param_grid={
                "model__max_depth": [2, 3, 4, 5, None],
                "model__min_samples_split": [2, 5, 10],
                "model__min_samples_leaf": [1, 2, 4],
                "model__max_features": ["sqrt", "log2", None],
            },
            param_dist={
                "model__n_estimators": [200, 300, 500, 800],
                "model__max_depth": [None] + list(range(2, 31)),
                "model__min_samples_split": list(range(2, 31)),
                "model__min_samples_leaf": list(range(1, 21)),
                "model__max_features": ["sqrt", "log2", None],
            },
        )

        # 7) Gradient Boosting
        self.model_specs_["GBR"] = ModelSpec(
            estimator=GradientBoostingRegressor(),
            param_grid={
                "model__n_estimators": [200, 300, 500],
                "model__learning_rate": [0.01, 0.05, 0.1],
                "model__max_depth": [2, 3, 4],
                "model__subsample": [0.7, 0.9, 1.0],
            },
            param_dist={
                "model__n_estimators": [100, 200, 300, 500, 800],
                "model__learning_rate": np.logspace(-2.5, -0.7, 50),
                "model__max_depth": [2, 3, 4, 5],
                "model__subsample": np.linspace(0.6, 1.0, 9),
            },
        )

        # 8) Optional: MLP (Keras) + KerasTuner (regression)
        if _HAS_TF and _HAS_KT:
            self.model_specs_["MLP"] = ModelSpec(
                estimator="keras_mlp",  # placeholder; lo construimos en tune()
                param_grid=None,
                param_dist=None,
            )

        return self

    # -----------------------------
    # Keras regression builder (optional)
    # -----------------------------
    def _build_mlp_reg_from_hp(self, hp, input_dim: int) -> "keras.Model":
        model = keras.Sequential()
        model.add(layers.Input(shape=(input_dim,)))

        n_layers = hp.Int("n_layers", 1, 4)
        for i in range(n_layers):
            units = hp.Int(f"units_{i}", min_value=32, max_value=256, step=32)
            dropout = hp.Float(f"dropout_{i}", 0.0, 0.5, step=0.1)
            l2v = hp.Float(f"l2_{i}", 1e-6, 1e-3, sampling="log")

            model.add(layers.Dense(units, activation="relu",
                                   kernel_regularizer=regularizers.l2(l2v)))
            if dropout > 0:
                model.add(layers.Dropout(dropout))

        # Regression head
        model.add(layers.Dense(1, activation="linear"))

        lr = hp.Float("lr", 1e-4, 5e-3, sampling="log")
        opt = keras.optimizers.Adam(learning_rate=lr)

        # RMSE metric
        model.compile(
            optimizer=opt,
            loss="mse",
            metrics=[keras.metrics.RootMeanSquaredError(name="rmse")]
        )
        return model

    # -----------------------------
    # Tuning
    # -----------------------------

    def _resolve_tuner_dir(self, tuner_dir: Optional[str], project_name: str) -> str:
        """
        Garantiza que tuner_dir sea un DIRECTORIO válido para KerasTuner.
        - Si tuner_dir es relativo => lo ancla en HOME (para evitar cwd raros).
        - Si existe como archivo => crea un directorio alternativo.
        - Siempre crea la carpeta.
        """
        base = os.path.expanduser("~")

        # default sano
        if tuner_dir is None or str(tuner_dir).strip() == "":
            tuner_dir = os.path.join(base, "kt_dir")

        tuner_dir = str(tuner_dir)

        # si no es absoluto, lo anclamos a HOME
        if not os.path.isabs(tuner_dir):
            tuner_dir = os.path.join(base, tuner_dir)

        # si existe pero ES ARCHIVO, no sirve
        if os.path.exists(tuner_dir) and not os.path.isdir(tuner_dir):
            # alternativa segura y única
            safe_name = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in project_name)
            tuner_dir = tuner_dir + f"__{safe_name}__dir"

        os.makedirs(tuner_dir, exist_ok=True)
        return tuner_dir

    @staticmethod
    def _build_mlp_reg_from_values(hp_values: dict, input_dim: int):
        hp = kt.HyperParameters()
        for k, v in hp_values.items():
            hp.values[k] = v

        model = keras.Sequential()
        model.add(layers.Input(shape=(input_dim,)))

        n_layers = hp.values["n_layers"]
        for i in range(n_layers):
            units = hp.values[f"units_{i}"]
            dropout = hp.values[f"dropout_{i}"]
            l2v = hp.values[f"l2_{i}"]

            model.add(layers.Dense(units, activation="relu",
                                kernel_regularizer=regularizers.l2(l2v)))
            if dropout > 0:
                model.add(layers.Dropout(dropout))

        model.add(layers.Dense(1, activation="linear"))

        lr = hp.values["lr"]
        opt = keras.optimizers.Adam(learning_rate=lr)

        model.compile(
            optimizer=opt,
            loss="mse",
            metrics=[keras.metrics.RootMeanSquaredError(name="rmse")]
        )
        return model

    def tune(
        self,
        model_name: str,
        method: str = "auto",  # auto|random|grid|keras_tuner|none
        n_iter: int = 25,
        param_space: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
    ) -> Dict[str, Any]:
        if self.X_train_ is None or self.y_train_ is None:
            raise RuntimeError("Primero ejecute prepare().")
        if self.cv_ is None:
            raise RuntimeError("prepare() no inicializó CV.")
        if model_name not in self.model_specs_:
            raise KeyError(f"Modelo '{model_name}' no existe. Use add_default_models() o registre el spec.")

        spec = self.model_specs_[model_name]
        method = str(method or "auto").lower().strip()

        # ``auto`` makes the strategy explicit in the output:
        # - MLP -> KerasTuner.
        # - Finite grid with <= 40 combinations -> GridSearchCV.
        # - Otherwise -> RandomizedSearchCV.
        if method == "auto":
            if model_name == "MLP":
                method = "keras_tuner"
            else:
                grid_space = param_space if param_space is not None else spec.param_grid
                random_space = param_space if param_space is not None else (spec.param_dist or spec.param_grid)

                def _finite_grid_size(space):
                    if not isinstance(space, dict) or not space:
                        return None
                    total = 1
                    for values in space.values():
                        if isinstance(values, (str, bytes)) or not hasattr(values, "__len__"):
                            return None
                        try:
                            total *= len(values)
                        except TypeError:
                            return None
                    return int(total)

                grid_size = _finite_grid_size(grid_space)
                if grid_size is not None and grid_size <= 40:
                    method = "grid"
                elif random_space:
                    method = "random"
                elif grid_space:
                    method = "grid"
                else:
                    raise ValueError(f"'{model_name}' has no HPO search space.")

            print(f"HPO auto-selection | model={model_name} | method={method}")

        if method == "none":
            out = {
                "model": model_name,
                "method": "none",
                "best_params": {},
                "best_score": np.nan,
                "best_estimator": self._make_pipeline(spec.estimator),
                "tuned": False,
                "status": "skipped",
            }
            return out

        # Special: keras_tuner
        if method == "keras_tuner":
            if model_name != "MLP":
                raise ValueError("keras_tuner solo aplica a 'MLP'.")
            if not (_HAS_TF and _HAS_KT):
                raise RuntimeError("TensorFlow/KerasTuner no disponibles en este ambiente.")

            ps = param_space or {}

            tuner_dir_raw = ps.get("tuner_dir", None)
            project_name = ps.get("project_name", "reg_mlp")
            tuner_dir = self._resolve_tuner_dir(tuner_dir_raw, project_name=project_name)

            max_trials = int(ps.get("max_trials", 25))
            epochs = int(ps.get("epochs", 40))
            batch_size = int(ps.get("batch_size", 64))
            patience = int(ps.get("patience", 6))

            # ------------------------------------------------------------
            # (1) NO tocar self.preprocess_ (evita side-effects)
            # ------------------------------------------------------------
            preproc = clone(self.preprocess_)

            # ------------------------------------------------------------
            # (2) Dataset para tuner: usa valid si existe; si no, split interno
            # ------------------------------------------------------------
            if self.X_valid_ is not None and self.y_valid_ is not None:
                X_tuner_train, y_tuner_train = self.X_train_, self.y_train_
                X_tuner_val, y_tuner_val = self.X_valid_, self.y_valid_
            else:
                X_tuner_train, X_tuner_val, y_tuner_train, y_tuner_val = train_test_split(
                    self.X_train_,
                    self.y_train_,
                    test_size=0.15,
                    random_state=self.random_state_,
                )

            Xt = preproc.fit_transform(X_tuner_train)
            Xv = preproc.transform(X_tuner_val)

            input_dim = Xt.shape[1]

            def model_builder(hp):
                return self._build_mlp_reg_from_hp(hp, input_dim=input_dim)

            tuner = kt.RandomSearch(
                model_builder,
                objective=kt.Objective("val_rmse", direction="min"),
                max_trials=max_trials,
                directory=tuner_dir,
                project_name=project_name,
                overwrite=True,
            )

            callbacks = [
                keras.callbacks.EarlyStopping(
                    monitor="val_rmse",
                    patience=patience,
                    restore_best_weights=True
                )
            ]

            tuner.search(
                Xt, y_tuner_train,
                validation_data=(Xv, y_tuner_val),
                epochs=epochs,
                batch_size=batch_size,
                callbacks=callbacks,
                verbose=verbose,
            )

            best_hp = tuner.get_best_hyperparameters(1)[0]

            wrapped = KerasRegressorSK(
                preprocess=clone(self.preprocess_),
                hp_values=best_hp.values,
                build_fn=SupervisedRegression._build_mlp_reg_from_values,  # <- estable
                epochs=epochs,
                batch_size=batch_size,
                patience=patience,
                random_state=self.random_state_,
                verbose=0,
            )

            out = {
                "model": model_name,
                "method": "keras_tuner",
                "best_params": best_hp.values,
                "best_estimator": wrapped,
                "tuned": True,
            }
            self.tuned_models_[model_name] = out
            return out
        


##----------------------------------------------------------------------------------------------------


        # sklearn tuning
        pipe = self._make_pipeline(spec.estimator)

        if method == "grid":
            grid = param_space or spec.param_grid
            if not grid:
                raise ValueError(f"'{model_name}' no tiene param_grid. Proporcione param_space.")
            search = GridSearchCV(
                estimator=pipe,
                param_grid=grid,
                scoring=self.scoring_,
                cv=self.cv_,
                n_jobs=-1,
                verbose=verbose,
                refit=True,
            )
        elif method == "random":
            dist = param_space or spec.param_dist or spec.param_grid
            if not dist:
                raise ValueError(f"'{model_name}' no tiene param_dist/grid. Proporcione param_space.")
            search = RandomizedSearchCV(
                estimator=pipe,
                param_distributions=dist,
                n_iter=int(n_iter),
                scoring=self.scoring_,
                cv=self.cv_,
                n_jobs=-1,
                verbose=verbose,
                random_state=self.random_state_,
                refit=True,
            )
        else:
            raise ValueError("method debe ser: auto | grid | random | keras_tuner | none")

        search.fit(self.X_train_, self.y_train_)

        out = {
            "model": model_name,
            "method": method,
            "best_params": search.best_params_,
            "best_score": float(search.best_score_),  # negative if rmse/mae
            "best_estimator": search.best_estimator_,
            "tuned": True,
        }
        self.tuned_models_[model_name] = out
        return out

    def tune_models(
        self,
        models: List[str],
        method: str = "auto",
        n_iter: int = 20,
        param_spaces: Optional[Dict[str, Dict[str, Any]]] = None,
        continue_on_error: bool = True,
        verbose: int = 0,
    ) -> pd.DataFrame:
        """
        Ajusta varios modelos de regresión en una sola llamada.

        Parameters
        ----------
        models:
            Nombres registrados en ``model_specs_``.
        method:
            ``"random"`` o ``"grid"`` para sklearn.
            MLP requiere ``"keras_tuner"`` y conviene ajustarlo por separado.
        n_iter:
            Número de combinaciones para RandomizedSearchCV.
        param_spaces:
            Diccionario opcional ``{modelo: espacio_de_búsqueda}``.
        continue_on_error:
            Si True, registra el error y continúa con los demás modelos.
        """
        if not self.model_specs_:
            self.add_default_models()

        param_spaces = param_spaces or {}
        rows = []

        for model_name in models:
            if model_name not in self.model_specs_:
                msg = f"Modelo '{model_name}' no está registrado."
                if continue_on_error:
                    print(f"⚠️ {msg} Se omite.")
                    rows.append({
                        "model": model_name,
                        "status": "skipped",
                        "method": method,
                        "error": msg,
                    })
                    continue
                raise KeyError(msg)

            if model_name == "MLP" and method not in {"keras_tuner", "auto"}:
                msg = "MLP requiere method='keras_tuner' o method='auto'; se omite en tune_models()."
                if continue_on_error:
                    print(f"⚠️ {msg}")
                    rows.append({
                        "model": model_name,
                        "status": "skipped",
                        "method": method,
                        "error": msg,
                    })
                    continue
                raise ValueError(msg)

            try:
                result = self.tune(
                    model_name=model_name,
                    method=method,
                    n_iter=n_iter,
                    param_space=param_spaces.get(model_name),
                    verbose=verbose,
                )
                rows.append({
                    "model": model_name,
                    "status": "tuned",
                    "method": result.get("method", method),
                    "best_score": result.get("best_score", np.nan),
                    "best_params": result.get("best_params"),
                    "error": None,
                })
            except Exception as exc:
                if not continue_on_error:
                    raise
                print(f"⚠️ {model_name} tuning skipped: {exc}")
                rows.append({
                    "model": model_name,
                    "status": "error",
                    "method": method,
                    "best_score": np.nan,
                    "best_params": None,
                    "error": str(exc),
                })

        report = pd.DataFrame(rows)
        self.last_tuning_batch_ = report.copy()
        return report

    def tune_top(
        self,
        top: int = 3,
        method: str = "auto",
        n_iter: int = 20,
        exclude: Optional[List[str]] = None,
        param_spaces: Optional[Dict[str, Dict[str, Any]]] = None,
        continue_on_error: bool = True,
        verbose: int = 0,
    ) -> pd.DataFrame:
        """
        Ajusta los mejores modelos del último leaderboard de regresión.

        Debe ejecutarse después de ``compete()``. El orden del leaderboard ya
        respeta la métrica usada: menor RMSE/MAE o mayor R².

        Por defecto excluye MLP porque requiere KerasTuner y una configuración
        diferente. Puede ajustarse después mediante ``tune('MLP',
        method='keras_tuner', ...)``.
        """
        if getattr(self, "leaderboard_", None) is None or self.leaderboard_.empty:
            raise RuntimeError(
                "No existe un leaderboard. Ejecute compete(metric=...) antes de tune_top()."
            )

        if top < 1:
            raise ValueError("top debe ser >= 1.")

        excluded = set(exclude or [])
        if method not in {"keras_tuner", "auto"}:
            excluded.add("MLP")

        ordered = [
            str(name)
            for name in self.leaderboard_["model"].tolist()
            if str(name) not in excluded
            and str(name) in self.model_specs_
            and pd.notna(name)
        ]

        # Evitar duplicados conservando el ranking original.
        selected = list(dict.fromkeys(ordered))[: int(top)]

        if not selected:
            raise RuntimeError(
                "No quedaron modelos elegibles para ajustar después de aplicar exclusiones."
            )

        print(f"Top {len(selected)} modelos seleccionados para HPO: {selected}")
        self.top_models_selected_ = selected

        return self.tune_models(
            models=selected,
            method=method,
            n_iter=n_iter,
            param_spaces=param_spaces,
            continue_on_error=continue_on_error,
            verbose=verbose,
        )

    # -----------------------------
    # Competition (CV)
    # -----------------------------
    def _cv_one(self, model_name: str, metric: str = "rmse") -> Dict[str, Any]:
        spec = self.model_specs_[model_name]

        # Use tuned estimator if exists
        if model_name in self.tuned_models_ and "best_estimator" in self.tuned_models_[model_name]:
            est = self.tuned_models_[model_name]["best_estimator"]
        else:
            if spec.estimator == "keras_mlp":
                # no tuned keras => skip
                return {
                    "model": model_name,
                    "metric": metric,
                    "cv_mean": np.nan,
                    "cv_std": np.nan,
                    "n_splits": self.cv_.get_n_splits(),
                    "tuned": False,
                    "note": "MLP requires tune(method='keras_tuner')"
                }
            est = self._make_pipeline(spec.estimator)

        # Choose sklearn scoring
        metric = (metric or "rmse").lower()
        if metric == "rmse":
            scoring = "neg_root_mean_squared_error"
        elif metric == "mae":
            scoring = "neg_mean_absolute_error"
        elif metric == "r2":
            scoring = "r2"
        else:
            raise ValueError("metric debe ser: rmse | mae | r2")

        cvres = cross_validate(
            est,
            self.X_train_,
            self.y_train_,
            cv=self.cv_,
            scoring=scoring,
            n_jobs=-1,
            return_train_score=False,
        )

        scores = cvres["test_score"]
        if metric in ("rmse", "mae"):
            # convert from negative to positive error
            scores_pos = -scores
        else:
            scores_pos = scores

        return {
            "model": model_name,
            "metric": metric,
            "cv_mean": float(np.mean(scores_pos)),
            "cv_std": float(np.std(scores_pos)),
            "n_splits": int(self.cv_.get_n_splits()),
            "tuned": bool(model_name in self.tuned_models_),
        }

    def compete(self, metric: str = "rmse") -> pd.DataFrame:
        if self.X_train_ is None or self.y_train_ is None:
            raise RuntimeError("Primero ejecute prepare().")
        if not self.model_specs_:
            raise RuntimeError("No hay modelos registrados. Ejecute add_default_models().")

        metric = (metric or "rmse").lower()
        names = list(self.model_specs_.keys())

        # --- separar MLP (Keras) para evitar procesos en Windows/TF ---
        has_mlp = "MLP" in names
        sk_names = [n for n in names if n != "MLP"]

        # Paralelo SOLO para modelos sklearn
        rows = Parallel(n_jobs=-1, prefer="processes")(
            delayed(self._cv_one)(name, metric=metric) for name in sk_names
        )

        # MLP en el proceso principal (más estable)
        if has_mlp:
            rows.append(self._cv_one("MLP", metric=metric))

        lb = pd.DataFrame(rows)

        # Orden: menor RMSE/MAE mejor; mayor R2 mejor
        if metric in ("rmse", "mae"):
            lb = lb.sort_values("cv_mean", ascending=True).reset_index(drop=True)
        else:
            lb = lb.sort_values("cv_mean", ascending=False).reset_index(drop=True)

        # persist leaderboard
        self.leaderboard_ = lb.copy()

        # set best
        self.best_model_name_ = str(lb.loc[0, "model"]) if len(lb) else None
        self.best_row_ = lb.loc[0].to_dict() if len(lb) else None

        try:
            self.best_cv_ = float(lb.loc[0, "cv_mean"]) if len(lb) else None
        except Exception:
            self.best_cv_ = None        

        return lb

    # -----------------------------
    # Fit + Evaluate
    # -----------------------------
    def fit(self, model: str = "best", refit_on: str = "train_valid") -> "SupervisedRegression":
        if self.X_train_ is None or self.y_train_ is None:
            raise RuntimeError("Primero ejecute prepare().")

        if model == "best":
            if self.best_model_name_ is None:
                raise RuntimeError("Aún no hay best_model_name_. Ejecute compete() primero.")
            model_name = self.best_model_name_
        else:
            model_name = model

        # Choose estimator
        if model_name in self.tuned_models_ and "best_estimator" in self.tuned_models_[model_name]:
            est = self.tuned_models_[model_name]["best_estimator"]
        else:
            spec = self.model_specs_.get(model_name)
            if spec is None:
                raise KeyError(f"Modelo '{model_name}' no existe.")
            if spec.estimator == "keras_mlp":
                raise RuntimeError("MLP requiere tune(method='keras_tuner') antes de fit(best).")
            est = self._make_pipeline(spec.estimator)

        # Data to refit
        if refit_on == "train_valid":
            X_refit = pd.concat([self.X_train_, self.X_valid_], axis=0)
            y_refit = np.concatenate([self.y_train_, self.y_valid_], axis=0)
        elif refit_on == "train":
            X_refit, y_refit = self.X_train_, self.y_train_
        else:
            raise ValueError("refit_on debe ser: train_valid | train")

        # Fit
        est.fit(X_refit, y_refit)
        self.best_estimator_ = est
        self.best_model_name_ = model_name
        return self

    def evaluate_test(self) -> pd.DataFrame:
        if self.best_estimator_ is None:
            raise RuntimeError("Primero ejecute fit().")
        if self.X_test_ is None or self.y_test_ is None:
            raise RuntimeError("prepare() no definió test split.")

        y_pred = self.best_estimator_.predict(self.X_test_)

        rmse = _rmse(self.y_test_, y_pred)
        mae = float(mean_absolute_error(self.y_test_, y_pred))
        r2 = float(r2_score(self.y_test_, y_pred))

        out = pd.DataFrame([{
            "model": self.best_model_name_,
            "rmse": rmse,
            "mae": mae,
            "r2": r2,
            "n_test": int(len(self.y_test_)),
        }])
        return out

    def y_distribution_summary(self) -> pd.DataFrame:
        """Resumen tipo 'ResVarPred' para comparar RMSE vs escala de la variable."""
        y = pd.Series(self.df[self.target].values, name=self.target)
        q1, q3 = np.quantile(y, [0.25, 0.75])
        return pd.DataFrame([{
            "Variable": y.name,
            "Min": float(np.min(y)),
            "Q1": float(q1),
            "Q3": float(q3),
            "Max": float(np.max(y)),
        }])

    # -----------------------------
    # Save / Load
    # -----------------------------

    def save(self, path: str, kind: str = "pickle") -> str:
        if self.best_estimator_ is None:
            raise RuntimeError("No hay best_estimator_. Ejecute fit() primero.")

        kind = (kind or "pickle").lower()
        if kind != "pickle":
            raise ValueError("Por ahora solo soportamos kind='pickle'.")

        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

        est = self.best_estimator_

        # --------------------------------------------------------
        # Helper: best CV desde leaderboard (si existe)
        # --------------------------------------------------------
        best_cv = None
        if getattr(self, "leaderboard_", None) is not None and len(self.leaderboard_) > 0:
            try:
                best_cv = float(self.leaderboard_.iloc[0]["cv_mean"])
            except Exception:
                best_cv = None
        # fallback a self.best_cv_ si lo tienes guardado
        if best_cv is None:
            try:
                best_cv = None if self.best_cv_ is None else float(self.best_cv_)
            except Exception:
                best_cv = None

        bundle = {
            "task": "regression",
            "best_model_name": self.best_model_name_,
            "best_cv": best_cv,
            "target": self.target,
            "features": self._infer_features(),
            "estimator_type": type(est).__name__,
        }

        base = re.sub(r"\.(pkl|pickle|joblib)$", "", path, flags=re.IGNORECASE)

        # -------------------------
        # Caso MLP (KerasRegressorSK)
        # -------------------------
        if isinstance(est, KerasRegressorSK):
            if est.model_ is None or est.preprocess_ is None:
                raise RuntimeError("KerasRegressorSK no está entrenado (model_/preprocess_).")

            keras_path = base + ".keras"
            sidecar_path = base + ".pkl"

            est.model_.save(keras_path)

            bundle.update({
                "estimator_type": "KerasRegressorSK",
                "keras_model_path": os.path.basename(keras_path),
                "keras_format": "keras",
                "build_fn_name": "_build_mlp_reg_from_values",
                "hp_values": est.hp_values,
                "epochs": est.epochs,
                "batch_size": est.batch_size,
                "patience": est.patience,
                "random_state": est.random_state,
                "verbose": est.verbose,
                "preprocess_fitted": est.preprocess_,
            })

            joblib.dump(bundle, sidecar_path)

            print(
                f"✅ Saved -> {sidecar_path}\n"
                f"   best_model: {self.best_model_name_}\n"
                f"   best_cv: {best_cv}\n"
                f"   keras_model: {os.path.basename(keras_path)}"
            )
            return sidecar_path

        # -------------------------
        # Caso sklearn
        # -------------------------
        bundle["sk_estimator"] = est
        joblib.dump(bundle, path)

        print(
            f"✅ Saved -> {path}\n"
            f"   best_model: {self.best_model_name_}\n"
            f"   best_cv: {best_cv}"
        )
        return path

    @staticmethod
    def load(path: str) -> Dict[str, Any]:
        """
        Carga el bundle guardado.
        Nota: si es MLP, este método retorna bundle y un helper `estimator_ready`
        listo para inferencia.
        """
        import os
        import joblib

        obj = joblib.load(path)

        # Si NO es KerasRegressorSK, devolvemos tal cual (sk_estimator listo)
        if obj.get("estimator_type") != "KerasRegressorSK":
            return obj

        # Si es KerasRegressorSK: reconstruir estimador listo para predict()
        keras_file = obj.get("keras_model_path")
        if keras_file is None:
            raise RuntimeError("Bundle KerasRegressorSK no trae 'keras_model_path'.")

        keras_path = os.path.join(os.path.dirname(path), keras_file)

        try:
            from tensorflow import keras
            model = keras.models.load_model(keras_path, compile=False)
        except Exception as e:
            raise RuntimeError(f"No se pudo cargar el modelo Keras desde {keras_path}: {e}")

        est = KerasRegressorSK(
            preprocess=obj["preprocess_fitted"],
            hp_values=obj.get("hp_values", {}),
            build_fn=SupervisedRegression._build_mlp_reg_from_values,
            epochs=int(obj.get("epochs", 40)),
            batch_size=int(obj.get("batch_size", 64)),
            patience=int(obj.get("patience", 6)),
            random_state=obj.get("random_state", None),
            verbose=int(obj.get("verbose", 0)),
        )
        est.preprocess_ = obj["preprocess_fitted"]
        est.model_ = model

        obj["estimator_ready"] = est
        obj["keras_model_path_full"] = keras_path
        return obj


# ============================================================
# TimeSeriesForecaster (sklearn-like) — DataScience Vic
# - anti-leakage temporal
# - periodograma sobre train
# - modelos: ETS, SARIMA (grid), LSTM (lookback)
# - tune / fit / predict / compete
# ============================================================



# ============================================================
# Helpers: métricas
# ============================================================

def rmse(y_true: pd.Series, y_pred: pd.Series) -> float:
    y_true = pd.Series(y_true).astype(float)
    y_pred = pd.Series(y_pred).astype(float)
    return float(np.sqrt(mean_squared_error(y_true.values, y_pred.values)))

# ============================================================
# Periodogram helper (equivalente a tu Periodograma)
# ============================================================

class PeriodogramHelper:
    def __init__(self, ts: pd.Series):
        self.ts = ts.astype(float).values
        self.freq, self.spec = signal.periodogram(self.ts)

    def best_freq(self, best: int = 3) -> np.ndarray:
        res = np.argsort(-self.spec)
        # Evitar freq=0 o freq=1 si sale raro; tu clase evitaba 1
        res = res[(res != 0) & (res != 1)]
        res = res[:best]
        return self.freq[res]

    def best_periods(self, best: int = 3) -> np.ndarray:
        f = self.best_freq(best=best)
        # evitar división por 0
        f = np.where(f == 0, np.nan, f)
        return 1.0 / f


# ============================================================
# Model Spec (similar a tu patrón de framework)
# ============================================================

@dataclass
class TSModelResult:
    name: str
    params: Dict[str, Any]
    y_pred: pd.Series
    score: float


class BaseTSModel:
    """Interfaz mínima tipo sklearn para forecasting."""
    def fit(self, y_train: pd.Series, **kwargs) -> "BaseTSModel":
        raise NotImplementedError

    def predict(self, steps: int, **kwargs) -> pd.Series:
        raise NotImplementedError


# ============================================================
# ETS Model
# ============================================================

# Nota: alpha/beta/gamma se pueden fijar o dejar optimizar internamente (si no se fijan).

class ETSModel(BaseTSModel):
    def __init__(
        self,
        trend: Optional[str] = "add",
        seasonal: Optional[str] = None,
        seasonal_periods: Optional[int] = None,
        # ✅ Holt-Winters smoothing
        alpha: Optional[float] = None,
        beta: Optional[float] = None,
        gamma: Optional[float] = None,
        optimized: bool = True,   # si False, fija smoothing_*
    ):
        self.trend = trend
        self.seasonal = seasonal
        self.seasonal_periods = seasonal_periods

        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.optimized = bool(optimized)

        self._fit = None
        self._y_train = None

        # para reportar “lo que quedó” en el fit
        self.alpha_ = alpha
        self.beta_  = beta
        self.gamma_ = gamma

    def fit(self, y_train: pd.Series, **kwargs) -> "ETSModel":
        self._y_train = y_train

        model = ExponentialSmoothing(
            y_train,
            trend=self.trend,
            seasonal=self.seasonal,
            seasonal_periods=self.seasonal_periods
        )

        fit_kwargs = {"optimized": self.optimized}

        # Si fijamos params (optimized=False), pasarlos
        if not self.optimized:
            fit_kwargs["smoothing_level"] = self.alpha
            fit_kwargs["smoothing_trend"] = self.beta
            fit_kwargs["smoothing_seasonal"] = self.gamma

        self._fit = model.fit(**fit_kwargs)

        # guardar lo que quedó en el fit (sea fijo u optimizado)
        lvl = getattr(self._fit, "smoothing_level", None)
        tr  = getattr(self._fit, "smoothing_trend", None)
        sea = getattr(self._fit, "smoothing_seasonal", None)

        self.alpha_ = self.alpha if (lvl is None) else float(lvl)
        self.beta_  = self.beta  if (tr  is None) else float(tr)
        self.gamma_ = self.gamma if (sea is None) else float(sea)

        return self

    def predict(self, steps: int, **kwargs) -> pd.Series:
        if self._fit is None:
            raise RuntimeError("ETSModel: call fit() first.")
        yhat = self._fit.forecast(steps)
        return pd.Series(yhat, index=_future_index(self._y_train.index, steps))


# ============================================================
# SARIMA Model (grid tuning externo)
# ============================================================

class SARIMAModel(BaseTSModel):
    def __init__(self, order=(1, 0, 0), seasonal_order=(0, 0, 0, 0)):
        self.order = tuple(order)
        self.seasonal_order = tuple(seasonal_order)
        self._fit = None
        self._y_train = None

    def fit(self, y_train: pd.Series, **kwargs) -> "SARIMAModel":
        self._y_train = y_train
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = SARIMAX(y_train, order=self.order, seasonal_order=self.seasonal_order, enforce_stationarity=False, enforce_invertibility=False)
            self._fit = model.fit(disp=False)
        return self

    def predict(self, steps: int, **kwargs) -> pd.Series:
        if self._fit is None:
            raise RuntimeError("SARIMAModel: call fit() first.")
        yhat = self._fit.forecast(steps)
        return pd.Series(yhat, index=_future_index(self._y_train.index, steps))


# ============================================================
# LSTM Model (REAL) — fixed (Input + no leakage scaling)
# ============================================================

class LSTMModel(BaseTSModel):

    def __init__(
        self,
        lookback: int = 12,
        epochs: int = 30,
        batch_size: int = 16,
        verbose: int = 0,

        # EarlyStopping
        validation_split: float = 0.2,
        early_stopping: bool = True,
        patience: int = 10,
        restore_best_weights: bool = True,

        # ReduceLROnPlateau
        reduce_lr: bool = True,
        lr_monitor: str = "val_loss",
        lr_factor: float = 0.5,
        lr_patience: int = 5,
        lr_min: float = 1e-6,

        # Dropout + L2
        dropout: float = 0.1,
        recurrent_dropout: float = 0.0,
        l2: float = 1e-4,
    ):
        self.lookback = int(lookback)
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.verbose = int(verbose)

        self.validation_split = float(validation_split)
        self.early_stopping = bool(early_stopping)
        self.patience = int(patience)
        self.restore_best_weights = bool(restore_best_weights)

        self.reduce_lr = bool(reduce_lr)
        self.lr_monitor = str(lr_monitor)
        self.lr_factor = float(lr_factor)
        self.lr_patience = int(lr_patience)
        self.lr_min = float(lr_min)

        self.dropout = float(dropout)
        self.recurrent_dropout = float(recurrent_dropout)
        self.l2 = float(l2)

        self._y_train = None
        self._model = None
        self._scaler_min = None
        self._scaler_max = None

    # -----------------------------
    # scaler helpers
    # -----------------------------
    def _fit_scaler(self, x: np.ndarray):
        self._scaler_min = float(np.min(x))
        self._scaler_max = float(np.max(x))

        # evitar división por cero
        if (self._scaler_max - self._scaler_min) == 0:
            self._scaler_max = self._scaler_min + 1.0

    def _scale(self, x: np.ndarray) -> np.ndarray:
        if self._scaler_min is None or self._scaler_max is None:
            # por seguridad, si alguien llama _scale sin fit_scaler
            self._fit_scaler(x)

        denom = self._scaler_max - self._scaler_min
        return (x - self._scaler_min) / denom

    def _inv_scale(self, x: np.ndarray) -> np.ndarray:
        denom = self._scaler_max - self._scaler_min
        return x * denom + self._scaler_min

    # -----------------------------
    # ventanas supervisadas
    # -----------------------------
    def _make_windows(self, arr: np.ndarray):
        X, y = [], []
        for i in range(len(arr) - self.lookback):
            X.append(arr[i:i + self.lookback])
            y.append(arr[i + self.lookback])

        X = np.array(X)
        y = np.array(y)

        # LSTM espera [samples, timesteps, features]
        X = X.reshape((X.shape[0], X.shape[1], 1))
        return X, y

    # -----------------------------
    # FIT
    # -----------------------------
    def fit(self, y_train: pd.Series, **kwargs):

        if not _HAS_TF:
            raise ImportError("TensorFlow no disponible. Instala tensorflow para usar LSTM.")

        # permitir override desde tune()
        if "lookback" in kwargs and kwargs["lookback"] is not None:
            self.lookback = int(kwargs["lookback"])
        if "epochs" in kwargs and kwargs["epochs"] is not None:
            self.epochs = int(kwargs["epochs"])
        if "batch_size" in kwargs and kwargs["batch_size"] is not None:
            self.batch_size = int(kwargs["batch_size"])

        self._y_train = y_train

        arr = y_train.astype(float).values

        # ✅ evitar leakage: fit scaler SOLO con el tramo train real
        # Keras validation_split toma el último % como val cuando shuffle=False,
        # y cuando shuffle=True, mezcla; de todos modos, el scaler debe ajustarse al "train" teórico.
        n = len(arr)
        split = self.validation_split if (self.validation_split is not None) else 0.0
        split = float(split)
        if split < 0 or split >= 1:
            split = 0.0

        cut = int(np.floor(n * (1 - split)))
        # asegurar mínimo de datos para entrenar ventanas
        cut = max(cut, self.lookback + 2)

        self._fit_scaler(arr[:cut])

        arr_scaled = self._scale(arr)
        X, y = self._make_windows(arr_scaled)

        # ---------
        # Modelo
        # ---------
        model = Sequential()

        reg = regularizers.l2(self.l2) if self.l2 > 0 else None

        # ✅ FIX warning: Input() primero, sin input_shape en LSTM
        model.add(Input(shape=(self.lookback, 1)))

        model.add(
            LSTM(
                32,
                dropout=self.dropout,
                recurrent_dropout=self.recurrent_dropout,
                kernel_regularizer=reg,
                recurrent_regularizer=reg,
            )
        )

        model.add(Dense(1, kernel_regularizer=reg))
        model.compile(optimizer="adam", loss="mse")

        callbacks = []
        if self.early_stopping:
            callbacks.append(
                tf.keras.callbacks.EarlyStopping(
                    monitor="val_loss",
                    patience=self.patience,
                    restore_best_weights=self.restore_best_weights,
                )
            )
        if self.reduce_lr:
            callbacks.append(
                tf.keras.callbacks.ReduceLROnPlateau(
                    monitor=self.lr_monitor,
                    factor=self.lr_factor,
                    patience=self.lr_patience,
                    min_lr=self.lr_min,
                    verbose=0,
                )
            )

        # Nota: para series de tiempo, típicamente shuffle=False
        # (si tu framework ya lo controla, podés quitarlo)
        model.fit(
            X,
            y,
            epochs=self.epochs,
            batch_size=self.batch_size,
            verbose=self.verbose,
            validation_split=self.validation_split,
            callbacks=callbacks,
            shuffle=False,
        )

        self._model = model
        return self

    # -----------------------------
    # PREDICT (recursive forecast)
    # -----------------------------
    def predict(self, steps: int, **kwargs):

        if self._model is None:
            raise RuntimeError("LSTMModel: call fit() first.")

        # usar scaler ya fit
        history = self._scale(self._y_train.astype(float).values.copy())

        preds = []
        for _ in range(int(steps)):
            x_input = history[-self.lookback:].reshape((1, self.lookback, 1))
            yhat = float(self._model.predict(x_input, verbose=0)[0][0])
            preds.append(yhat)
            history = np.append(history, yhat)

        preds = self._inv_scale(np.array(preds))

        return pd.Series(
            preds,
            index=_future_index(self._y_train.index, int(steps)),
        )

# ============================================================
# CNN1D Model (REAL) — fixed (Input + no leakage scaling)
# ============================================================

class CNN1DModel(BaseTSModel):

    def __init__(
        self,
        lookback: int = 12,
        epochs: int = 30,
        batch_size: int = 16,
        verbose: int = 0,

        # callbacks
        validation_split: float = 0.2,
        early_stopping: bool = True,
        patience: int = 10,
        restore_best_weights: bool = True,

        reduce_lr: bool = True,
        lr_monitor: str = "val_loss",
        lr_factor: float = 0.5,
        lr_patience: int = 5,
        lr_min: float = 1e-6,

        # CNN params
        filters: int = 32,
        kernel_size: int = 3,
        l2: float = 0.0,
    ):
        self.lookback = int(lookback)
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.verbose = int(verbose)

        self.validation_split = float(validation_split)
        self.early_stopping = bool(early_stopping)
        self.patience = int(patience)
        self.restore_best_weights = bool(restore_best_weights)

        self.reduce_lr = bool(reduce_lr)
        self.lr_monitor = str(lr_monitor)
        self.lr_factor = float(lr_factor)
        self.lr_patience = int(lr_patience)
        self.lr_min = float(lr_min)

        self.filters = int(filters)
        self.kernel_size = int(kernel_size)
        self.l2 = float(l2)

        self._y_train = None
        self._model = None
        self._scaler_min = None
        self._scaler_max = None

    # -----------------------------
    # scaler helpers (no leakage)
    # -----------------------------
    def _fit_scaler(self, x: np.ndarray):
        self._scaler_min = float(np.min(x))
        self._scaler_max = float(np.max(x))
        if (self._scaler_max - self._scaler_min) == 0:
            self._scaler_max = self._scaler_min + 1.0

    def _scale(self, x: np.ndarray) -> np.ndarray:
        if self._scaler_min is None or self._scaler_max is None:
            self._fit_scaler(x)
        denom = self._scaler_max - self._scaler_min
        return (x - self._scaler_min) / denom

    def _inv_scale(self, x: np.ndarray) -> np.ndarray:
        denom = self._scaler_max - self._scaler_min
        return x * denom + self._scaler_min

    # -----------------------------
    # ventanas supervisadas
    # -----------------------------
    def _make_windows(self, arr: np.ndarray):
        X, y = [], []
        for i in range(len(arr) - self.lookback):
            X.append(arr[i:i + self.lookback])
            y.append(arr[i + self.lookback])

        X = np.array(X).reshape((-1, self.lookback, 1))
        y = np.array(y)
        return X, y

    # -----------------------------
    # FIT
    # -----------------------------
    def fit(self, y_train: pd.Series, **kwargs) -> "CNN1DModel":

        if not _HAS_TF:
            raise ImportError("TensorFlow no disponible. Instala tensorflow para usar CNN1D.")

        # ✅ permitir override desde tune()
        if "lookback" in kwargs and kwargs["lookback"] is not None:
            self.lookback = int(kwargs["lookback"])
        if "epochs" in kwargs and kwargs["epochs"] is not None:
            self.epochs = int(kwargs["epochs"])
        if "batch_size" in kwargs and kwargs["batch_size"] is not None:
            self.batch_size = int(kwargs["batch_size"])
        if "filters" in kwargs and kwargs["filters"] is not None:
            self.filters = int(kwargs["filters"])
        if "kernel_size" in kwargs and kwargs["kernel_size"] is not None:
            self.kernel_size = int(kwargs["kernel_size"])

        self._y_train = y_train

        arr = y_train.astype(float).values

        # ✅ evitar leakage: scaler se ajusta solo al tramo train real
        n = len(arr)
        split = self.validation_split if (self.validation_split is not None) else 0.0
        split = float(split)
        if split < 0 or split >= 1:
            split = 0.0

        cut = int(np.floor(n * (1 - split)))
        cut = max(cut, self.lookback + 2)

        self._fit_scaler(arr[:cut])

        arr_scaled = self._scale(arr)
        X, y = self._make_windows(arr_scaled)

        reg = regularizers.l2(self.l2) if self.l2 > 0 else None

        model = Sequential()

        # ✅ FIX warning: Input() primero, sin input_shape en Conv1D
        model.add(Input(shape=(self.lookback, 1)))

        model.add(
            Conv1D(
                filters=self.filters,
                kernel_size=self.kernel_size,
                activation="relu",
                padding="causal",
                kernel_regularizer=reg,
            )
        )
        model.add(GlobalAveragePooling1D())
        model.add(Dense(1, kernel_regularizer=reg))

        model.compile(optimizer="adam", loss="mse")

        callbacks = []
        if self.early_stopping:
            callbacks.append(
                tf.keras.callbacks.EarlyStopping(
                    monitor="val_loss",
                    patience=self.patience,
                    restore_best_weights=self.restore_best_weights,
                )
            )
        if self.reduce_lr:
            callbacks.append(
                tf.keras.callbacks.ReduceLROnPlateau(
                    monitor=self.lr_monitor,
                    factor=self.lr_factor,
                    patience=self.lr_patience,
                    min_lr=self.lr_min,
                    verbose=0,
                )
            )

        model.fit(
            X,
            y,
            epochs=self.epochs,
            batch_size=self.batch_size,
            verbose=self.verbose,
            validation_split=self.validation_split,
            callbacks=callbacks,
            shuffle=False,  # ✅ recomendado para series
        )

        self._model = model
        return self

    # -----------------------------
    # PREDICT (recursive forecast)
    # -----------------------------
    def predict(self, steps: int, **kwargs) -> pd.Series:

        if self._model is None or self._y_train is None:
            raise RuntimeError("CNN1DModel: call fit() first.")

        history = self._scale(self._y_train.astype(float).values.copy())
        preds = []

        for _ in range(int(steps)):
            x_input = history[-self.lookback:].reshape((1, self.lookback, 1))
            yhat = float(self._model.predict(x_input, verbose=0)[0][0])
            preds.append(yhat)
            history = np.append(history, yhat)

        preds = self._inv_scale(np.array(preds))

        return pd.Series(
            preds,
            index=_future_index(self._y_train.index, int(steps)),
        )

# ============================================================
# Prophet Model
# ============================================================

class ProphetModel(BaseTSModel):
    def __init__(
        self,
        seasonality_mode: str = "additive",           # additive | multiplicative
        yearly_seasonality: str | bool = "auto",
        weekly_seasonality: str | bool = "auto",
        daily_seasonality: str | bool = "auto",
        changepoint_prior_scale: float = 0.05,
        seasonality_prior_scale: float = 10.0,
        holidays_prior_scale: float = 10.0,
        interval_width: float = 0.8,
        add_country_holidays: str | None = None,      # e.g. "US", "CO"...
    ):
        self.seasonality_mode = seasonality_mode
        self.yearly_seasonality = yearly_seasonality
        self.weekly_seasonality = weekly_seasonality
        self.daily_seasonality = daily_seasonality
        self.changepoint_prior_scale = float(changepoint_prior_scale)
        self.seasonality_prior_scale = float(seasonality_prior_scale)
        self.holidays_prior_scale = float(holidays_prior_scale)
        self.interval_width = float(interval_width)
        self.add_country_holidays = add_country_holidays

        self._m = None
        self._y_train = None

    def fit(self, y_train: pd.Series, **kwargs) -> "ProphetModel":
        if not _HAS_PROPHET:
            raise ImportError("Prophet no disponible. Instala 'prophet' para usar ProphetModel.")

        self._y_train = y_train

        df = pd.DataFrame({
            "ds": y_train.index.to_pydatetime(),
            "y": y_train.astype(float).values
        })

        m = Prophet(
            seasonality_mode=self.seasonality_mode,
            yearly_seasonality=self.yearly_seasonality,
            weekly_seasonality=self.weekly_seasonality,
            daily_seasonality=self.daily_seasonality,
            changepoint_prior_scale=self.changepoint_prior_scale,
            seasonality_prior_scale=self.seasonality_prior_scale,
            holidays_prior_scale=self.holidays_prior_scale,
            interval_width=self.interval_width
        )

        if self.add_country_holidays:
            m.add_country_holidays(country_name=self.add_country_holidays)

        m.fit(df)
        self._m = m
        return self

    def predict(self, steps: int, **kwargs) -> pd.Series:
        if self._m is None or self._y_train is None:
            raise RuntimeError("ProphetModel: call fit() first.")

        # Prophet arma future con freq inferido por pandas, usamos el index helper
        future_idx = _future_index(self._y_train.index, int(steps))
        future = pd.DataFrame({"ds": future_idx.to_pydatetime()})

        fcst = self._m.predict(future)
        yhat = fcst["yhat"].values.astype(float)

        return pd.Series(yhat, index=future_idx)

# ============================================================
# XGBoost Model
# ============================================================

class XGBTSModel(BaseTSModel):
    def __init__(
        self,
        lags: list[int] = [1, 2, 3, 6, 12],
        rolling_windows: list[int] = [3, 6, 12],
        use_date_parts: bool = True,

        # XGB params
        n_estimators: int = 400,
        max_depth: int = 4,
        learning_rate: float = 0.05,
        subsample: float = 0.9,
        colsample_bytree: float = 0.9,
        reg_lambda: float = 1.0,
        reg_alpha: float = 0.0,
        random_state: int | None = None,
    ):
        self.lags = [int(x) for x in lags]
        self.rolling_windows = [int(x) for x in rolling_windows]
        self.use_date_parts = bool(use_date_parts)

        self.n_estimators = int(n_estimators)
        self.max_depth = int(max_depth)
        self.learning_rate = float(learning_rate)
        self.subsample = float(subsample)
        self.colsample_bytree = float(colsample_bytree)
        self.reg_lambda = float(reg_lambda)
        self.reg_alpha = float(reg_alpha)
        self.random_state = random_state

        self._model = None
        self._y_train = None

    def _make_features(self, y: pd.Series) -> pd.DataFrame:
        df = pd.DataFrame({"y": y.astype(float).values}, index=y.index)

        # lags
        for L in self.lags:
            df[f"lag_{L}"] = df["y"].shift(L)

        # rolling stats (solo pasado)
        for w in self.rolling_windows:
            df[f"roll_mean_{w}"] = df["y"].shift(1).rolling(w, min_periods=1).mean()
            df[f"roll_std_{w}"]  = df["y"].shift(1).rolling(w, min_periods=1).std()

        # date parts
        if self.use_date_parts:
            idx = df.index
            df["month"] = idx.month
            df["quarter"] = idx.quarter
            df["year"] = idx.year

        return df

    def fit(self, y_train: pd.Series, **kwargs) -> "XGBTSModel":
        if not _HAS_XGB:
            raise ImportError("xgboost no disponible. Instala 'xgboost' para usar XGBTSModel.")

        self._y_train = y_train

        df = self._make_features(y_train)
        df = df.dropna()  # se pierden las primeras filas por lags
        X = df.drop(columns=["y"])
        y = df["y"]

        model = xgb.XGBRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            reg_lambda=self.reg_lambda,
            reg_alpha=self.reg_alpha,
            random_state=self.random_state,
            objective="reg:squarederror",
            n_jobs=1,
        )
        model.fit(X, y)

        self._model = model
        return self

    def predict(self, steps: int, **kwargs) -> pd.Series:
        if self._model is None or self._y_train is None:
            raise RuntimeError("XGBTSModel: call fit() first.")

        steps = int(steps)
        hist = self._y_train.astype(float).copy()

        preds = []
        for _ in range(steps):
            # crear “una fila” de features usando la historia (sin fuga)
            df_feat = self._make_features(hist).iloc[[-1]]  # última fila
            X_next = df_feat.drop(columns=["y"])
            yhat = float(self._model.predict(X_next)[0])

            # append
            next_idx = _future_index(hist.index, 1)[0]
            hist = pd.concat([hist, pd.Series([yhat], index=pd.DatetimeIndex([next_idx]))])
            preds.append(yhat)

        future_idx = _future_index(self._y_train.index, steps)
        return pd.Series(np.array(preds, dtype=float), index=future_idx)

# ============================================================
# Index helper
# ============================================================

def _future_index(idx: pd.DatetimeIndex, steps: int) -> pd.DatetimeIndex:
    # idx debe tener freq o inferible
    if getattr(idx, "freq", None) is None:
        freq = pd.infer_freq(idx)
        if freq is None:
            raise ValueError("No se pudo inferir frecuencia del DatetimeIndex. Define freq en to_series().")
    else:
        freq = idx.freq

    start = idx[-1]
    future = pd.date_range(start=start, periods=steps + 1, freq=freq)
    return future[1:]


# ============================================================
# Forecaster
# ============================================================

class TimeSeriesForecaster(DFCore):
    """
    sklearn-like forecaster (anti-leakage temporal)
    """

    def _params_to_str(self, p: Any) -> str:

        """
        Convierte dict de params a string JSON estable,
        evitando problemas con numpy types.
        """

        if p is None:
            return ""

        def _json_default(o):
            if isinstance(o, (np.integer,)):
                return int(o)
            if isinstance(o, (np.floating,)):
                return float(o)
            if isinstance(o, (np.ndarray,)):
                return o.tolist()
            if isinstance(o, (set, tuple)):
                return list(o)
            return str(o)

        try:
            return json.dumps(
                p,
                ensure_ascii=False,
                sort_keys=True,
                default=_json_default
            )
        except Exception:
            return str(p)

    def __init__(self, df: pd.DataFrame, date_col: str, target_col: str, freq: str = "MS"):
        # ✅ DFCore: target = target_col; features = todo menos target (incluye date_col)
        super().__init__(df=df, target=target_col)

        # OJO: DFCore ya guarda self.df (copia). Usamos esa.
        self.date_col = date_col
        self.target_col = target_col
        self.freq = freq

        # serie
        self.y_: Optional[pd.Series] = None

        # splits
        self.y_train_: Optional[pd.Series] = None
        self.y_test_: Optional[pd.Series] = None
        self.test_size_: Optional[int] = None

        # candidates
        self.period_candidates_: Optional[List[int]] = None

        # modelos
        self.models_: Dict[str, BaseTSModel] = {}
        self.leaderboard_: Optional[pd.DataFrame] = None
        self.best_model_name_: Optional[str] = None
        self.best_score_: Optional[float] = None
        self.best_params_: Optional[Dict[str, Any]] = None 

        # fitted
        self.fitted_model_: Optional[BaseTSModel] = None
    
    # -----------------------------
    # Helper valor efectivo
    # -----------------------------

    def _effective(self, obj, fitted_attr: str, spec_attr: str):
        v = getattr(obj, fitted_attr, None)
        return v if v is not None else getattr(obj, spec_attr, None)
    
    @staticmethod

    def format_forecast(df: pd.DataFrame, col: str, millions: bool = True) -> pd.DataFrame:
        out = df.copy()
        if millions:
            out[col + "_M"] = (out[col] / 1e6).round(2)
        out[col + "_fmt"] = out[col].map(lambda x: f"{x:,.0f}")
        return out

    # -----------------------------
    # 1) Data → Serie
    # -----------------------------
    def to_series(self, agg: str = "sum") -> "TimeSeriesForecaster":
        d = self.df.copy()
        d[self.date_col] = pd.to_datetime(d[self.date_col])
        d = d.sort_values(self.date_col)

        # agregación si hay duplicados por fecha
        if agg is not None:
            if agg == "sum":
                d = d.groupby(self.date_col, as_index=False)[self.target_col].sum()
            elif agg == "mean":
                d = d.groupby(self.date_col, as_index=False)[self.target_col].mean()
            else:
                raise ValueError("agg soportado: 'sum' | 'mean'")

        idx = pd.DatetimeIndex(d[self.date_col])  # sin freq aquí
        y = pd.Series(d[self.target_col].values, index=idx, name=self.target_col).astype(float)

        # aseguramos frecuencia y orden
        y = y.asfreq(self.freq)
        self.y_ = y
        return self

    # -----------------------------
    # 2) Split temporal (anti-leakage)
    # -----------------------------
    def prepare(self, test_size: Optional[int] = None, train_size: Optional[float] = None) -> "TimeSeriesForecaster":
        if self.y_ is None:
            raise RuntimeError("Ejecuta to_series() primero.")

        n = len(self.y_)
        if test_size is None:
            if train_size is None:
                train_size = 0.95
            cut = int(np.floor(n * float(train_size)))
            test_size = n - cut

        if test_size <= 0 or test_size >= n:
            raise ValueError("test_size inválido.")

        self.test_size_ = int(test_size)
        self.y_train_ = self.y_.iloc[:-self.test_size_]
        self.y_test_ = self.y_.iloc[-self.test_size_:]
        return self

    # -----------------------------
    # 3) Periodograma (solo train)
    # -----------------------------
    def periodogram_candidates(self, best: int = 3, round_int: bool = True, min_period: int = 2, max_period: Optional[int] = None) -> List[int]:
        if self.y_train_ is None:
            raise RuntimeError("Ejecuta prepare() primero (split temporal).")

        p = PeriodogramHelper(self.y_train_)
        periods = p.best_periods(best=best)

        # limpieza
        periods = periods[np.isfinite(periods)]
        periods = periods[periods > 0]

        if round_int:
            cand = [int(np.round(x)) for x in periods]
        else:
            cand = [float(x) for x in periods]

        # filtros
        cand = [int(x) for x in cand if int(x) >= min_period]
        if max_period is not None:
            cand = [int(x) for x in cand if int(x) <= int(max_period)]

        # unique manteniendo orden
        seen = set()
        uniq = []
        for x in cand:
            if x not in seen:
                uniq.append(x)
                seen.add(x)

        self.period_candidates_ = uniq
        return uniq

    # -----------------------------
    # 4) Model registry
    # -----------------------------
    def add_default_models(
        self,
        include_lstm: bool = True,
        include_cnn: bool = True,
        include_prophet: bool = True,
        include_xgb: bool = True
        ) -> "TimeSeriesForecaster":

        self.models_["ETS"] = ETSModel(trend="add", seasonal=None, seasonal_periods=None)
        self.models_["SARIMA"] = SARIMAModel(order=(1, 0, 0), seasonal_order=(0, 0, 0, 0))

        if include_lstm:
            self.models_["LSTM"] = LSTMModel(lookback=12, epochs=20, batch_size=16, verbose=0)

        if include_cnn:
            self.models_["CNN1D"] = CNN1DModel(lookback=12, epochs=20, batch_size=16, verbose=0)

        if include_prophet:
            self.models_["PROPHET"] = ProphetModel()

        if include_xgb:
            self.models_["XGBTS"] = XGBTSModel()

        return self

    # -----------------------------
    # 5) Tune
    # -----------------------------
    def tune(
        self,
        model_name: str,
        metric: str = "rmse",
        # SARIMA grid (legacy)
        ar: List[int] = [0, 1, 2],
        seas: List[int] = [0, 1],
        S_candidates: Optional[List[int]] = None,
        n_jobs: int = -1,
        # LSTM/CNN candidates
        p_candidates: Optional[List[int]] = None,
        dl_epochs: int = 20,
        **kwargs,  # ✅ IMPORTANTE: habilita kwargs.get(...)
        ) -> "TimeSeriesForecaster":

        if self.y_train_ is None or self.y_test_ is None:
            raise RuntimeError("Ejecuta prepare() antes de tune().")

        if model_name not in self.models_:
            raise ValueError(f"Modelo '{model_name}' no existe. Usa add_default_models() primero.")

        if metric != "rmse":
            raise ValueError("Por ahora solo metric='rmse'.")

        name = model_name.upper()

        # ============================================================
        # ETS
        # ============================================================
        if name == "ETS":
            # --------- defaults de seasonal_periods -----------
            if self.period_candidates_:
                sp_default = int(self.period_candidates_[0])
            else:
                sp_default = None

            # seasonal_periods candidates (prioridad: explícito > periodograma > None)
            seasonal_periods_candidates = kwargs.get("seasonal_periods_candidates", None)

            if seasonal_periods_candidates is None:
                if S_candidates is not None:
                    seasonal_periods_candidates = list(S_candidates)
                elif self.period_candidates_:
                    seasonal_periods_candidates = list(self.period_candidates_)
                else:
                    seasonal_periods_candidates = [None]

            # trend/seasonal “tipo Holt-Winters”
            trend = kwargs.get("trend", "add")
            seasonal = kwargs.get("seasonal", "add")

            # grillas alpha/beta/gamma (paso tipo HW_calibrado)
            alpha_candidates = kwargs.get("alpha_candidates", None)
            beta_candidates  = kwargs.get("beta_candidates", None)
            gamma_candidates = kwargs.get("gamma_candidates", None)

            paso = kwargs.get("paso", 0.1)
            if alpha_candidates is None:
                alpha_candidates = np.round(np.arange(0, 1 + paso, paso), 10).tolist()
            if beta_candidates is None:
                beta_candidates = np.round(np.arange(0, 1 + paso, paso), 10).tolist()
            if gamma_candidates is None:
                gamma_candidates = np.round(np.arange(0, 1 + paso, paso), 10).tolist()

            # si trend/seasonal no aplican, “apagamos” beta/gamma
            if trend is None:
                beta_candidates = [None]
            if seasonal is None:
                gamma_candidates = [None]

            n_train = len(self.y_train_)

            # Si seasonal="add" o "mul", necesitamos seasonal_periods válido y “suficiente data”
            if seasonal is not None:
                seasonal_periods_candidates = [
                    sp for sp in seasonal_periods_candidates
                    if sp is not None and int(sp) >= 2 and (2 * int(sp) <= n_train)
                ]
                if not seasonal_periods_candidates:
                    # fallback: sin estacionalidad si no hay ciclos suficientes
                    seasonal = None
                    seasonal_periods_candidates = [None]
                    gamma_candidates = [None]
            else:
                seasonal_periods_candidates = [None]
                gamma_candidates = [None]

            # --------- build param grid -----------
            grid = []
            for sp in seasonal_periods_candidates:
                for a in alpha_candidates:
                    for b in beta_candidates:
                        for g in gamma_candidates:
                            grid.append((sp, a, b, g))

            # --------- parallel eval -----------
            def _eval_ets(tup):
                sp, a, b, g = tup
                try:
                    m = ETSModel(
                        trend=trend,
                        seasonal=seasonal,
                        seasonal_periods=(int(sp) if sp is not None else None),
                        alpha=(None if a is None else float(a)),
                        beta=(None if b is None else float(b)),
                        gamma=(None if g is None else float(g)),
                        optimized=False,   # ✅ agrega esto
                    ).fit(self.y_train_)
                    yhat = m.predict(steps=len(self.y_test_))
                    sc = rmse(self.y_test_, yhat)
                    # params finales (incluye alpha_/beta_/gamma_)
                    params = {
                        "trend": trend,
                        "seasonal": seasonal,
                        "seasonal_periods": (int(sp) if sp is not None else None),
                        "alpha": self._effective(m, "alpha_", "alpha"),
                        "beta":  self._effective(m, "beta_",  "beta"),
                        "gamma": self._effective(m, "gamma_", "gamma"),
                    }
                    return sc, params
                except Exception:
                    return float("inf"), None

            # joblib paralelo
            has_joblib = False
            if n_jobs != 1:
                try:
                    from joblib import Parallel, delayed
                    has_joblib = True
                except Exception:
                    has_joblib = False

            if has_joblib and n_jobs != 1:
                results = Parallel(n_jobs=n_jobs)(delayed(_eval_ets)(t) for t in grid)
            else:
                results = [_eval_ets(t) for t in grid]

            best_score = float("inf")
            best_params = None
            for sc, params in results:
                if sc < best_score:
                    best_score = sc
                    best_params = params
            
            if best_params is None:
                sp_fb = seasonal_periods_candidates[0] if seasonal_periods_candidates and seasonal_periods_candidates[0] is not None else None
                self.models_["ETS"] = ETSModel(
                    trend=trend,
                    seasonal=seasonal,
                    seasonal_periods=(int(sp_fb) if sp_fb is not None else None),
                    optimized=True
                )
                return self

            # Guardar el “mejor ETS” como spec, igual que los otros modelos
            self.models_["ETS"] = ETSModel(
                trend=best_params["trend"],
                seasonal=best_params["seasonal"],
                seasonal_periods=best_params["seasonal_periods"],
                alpha=best_params["alpha"],
                beta=best_params["beta"],
                gamma=best_params["gamma"],
                optimized=False,   # porque fijamos smoothing
            )
            return self

        # ============================================================
        # LSTM: escoger mejor lookback p
        # ============================================================
        if name == "LSTM":
            if p_candidates is None:
                if self.period_candidates_ is None:
                    raise RuntimeError("Corre periodogram_candidates() o pasa p_candidates manualmente.")
                p_candidates = self.period_candidates_

            best_score = float("inf")
            best_p = None

            for p in p_candidates:
                try:
                    m = LSTMModel(
                        lookback=int(p),
                        epochs=int(dl_epochs),
                        batch_size=16,
                        verbose=0,
                    ).fit(self.y_train_)
                    yhat = m.predict(steps=len(self.y_test_))
                    sc = rmse(self.y_test_, yhat)
                    if sc < best_score:
                        best_score = sc
                        best_p = int(p)
                except Exception:
                    continue

            if best_p is None:
                raise RuntimeError("LSTM tune: no se encontró un lookback válido.")

            self.models_["LSTM"] = LSTMModel(
                lookback=best_p,
                epochs=int(dl_epochs),
                batch_size=16,
                verbose=0,
            )
            return self

        # ============================================================
        # CNN1D: escoger mejor lookback p
        # ============================================================
        if name == "CNN1D":
            if p_candidates is None:
                if self.period_candidates_ is None:
                    raise RuntimeError("Corre periodogram_candidates() o pasa p_candidates manualmente.")
                p_candidates = self.period_candidates_

            best_score = float("inf")
            best_p = None

            for p in p_candidates:
                try:
                    m = CNN1DModel(
                        lookback=int(p),
                        epochs=int(dl_epochs),
                        batch_size=16,
                        verbose=0,
                    ).fit(self.y_train_)
                    yhat = m.predict(steps=len(self.y_test_))
                    sc = rmse(self.y_test_, yhat)
                    if sc < best_score:
                        best_score = sc
                        best_p = int(p)
                except Exception:
                    continue

            if best_p is None:
                raise RuntimeError("CNN1D tune: no se encontró un lookback válido.")

            self.models_["CNN1D"] = CNN1DModel(
                lookback=best_p,
                epochs=int(dl_epochs),
                batch_size=16,
                verbose=0,
            )
            return self

        # ============================================================
        # SARIMA: grid over (p,d,q,P,D,Q,S)
        # ============================================================
        if name == "SARIMA":
            if S_candidates is None:
                if self.period_candidates_ is None:
                    raise RuntimeError("Corre periodogram_candidates() o pasa S_candidates manualmente.")
                S_candidates = self.period_candidates_

            # --- bounds & estrategia (desde kwargs con defaults) ---
            max_p = int(kwargs.get("max_p", 2))
            max_d = int(kwargs.get("max_d", 2))
            max_q = int(kwargs.get("max_q", 2))
            max_P = int(kwargs.get("max_P", 1))
            max_D = int(kwargs.get("max_D", 1))
            max_Q = int(kwargs.get("max_Q", 1))

            search = kwargs.get("search", "grid")            # "grid" | "random"
            max_evals = int(kwargs.get("max_evals", 150))    # usado si random
            random_state = kwargs.get("random_state", None)  # opcional

            # bounds
            p_list = list(range(0, int(max_p) + 1))
            d_list = list(range(0, int(max_d) + 1))
            q_list = list(range(0, int(max_q) + 1))
            P_list = list(range(0, int(max_P) + 1))
            D_list = list(range(0, int(max_D) + 1))
            Q_list = list(range(0, int(max_Q) + 1))

            # generate all combos
            param_grid = []
            for S in S_candidates:
                S = int(S)
                for p in p_list:
                    for d in d_list:
                        for q in q_list:
                            for P in P_list:
                                for D in D_list:
                                    for Q in Q_list:
                                        param_grid.append(((p, d, q), (P, D, Q, S)))

            # search strategy
            search = str(search).lower()
            if search not in ("grid", "random"):
                raise ValueError("search debe ser 'grid' o 'random'.")

            if search == "random":
                rng = np.random.default_rng(random_state)
                if len(param_grid) > int(max_evals):
                    idx = rng.choice(len(param_grid), size=int(max_evals), replace=False)
                    param_grid = [param_grid[i] for i in idx]

            # eval
            def _eval(params):
                order, seas_order = params
                try:
                    m = SARIMAModel(order=order, seasonal_order=seas_order).fit(self.y_train_)
                    yhat = m.predict(steps=len(self.y_test_))
                    sc = rmse(self.y_test_, yhat)
                    return sc, order, seas_order
                except Exception:
                    return float("inf"), None, None

            has_joblib = False
            if n_jobs != 1:
                try:
                    from joblib import Parallel, delayed
                    has_joblib = True
                except Exception:
                    has_joblib = False

            if has_joblib and n_jobs != 1:
                # ✅ Evitar loky/processes en Windows (statsmodels no siempre es picklable)
                # prefer="threads" evita el error de buffer read-only en unpickle.
                try:
                    results = Parallel(n_jobs=n_jobs, prefer="threads")(delayed(_eval)(p) for p in param_grid)
                except Exception:
                    # fallback ultra-robusto
                    results = [_eval(p) for p in param_grid]
            else:
                results = [_eval(p) for p in param_grid]

            best_score = float("inf")
            best_params = None
            for sc, order, seas_order in results:
                if sc < best_score:
                    best_score = sc
                    best_params = {"order": order, "seasonal_order": seas_order}

            if best_params is None or best_params["order"] is None:
                raise RuntimeError("SARIMA tune: no se encontró un modelo válido.")

            self.models_["SARIMA"] = SARIMAModel(
                order=best_params["order"],
                seasonal_order=best_params["seasonal_order"],
            )
            return self

        # ============================================================
        # Prophet
        # ============================================================

        import logging

        # Silenciar cmdstanpy y Prophet
        logging.getLogger("cmdstanpy").setLevel(logging.WARNING)
        logging.getLogger("prophet").setLevel(logging.WARNING)

        # (opcional) si tienes muchos warnings ruidosos
        logging.getLogger().setLevel(logging.WARNING)

        if name == "PROPHET":
            if not _HAS_PROPHET:
                raise ImportError("Prophet no disponible.")

            search = str(kwargs.get("search", "grid")).lower()   # grid | random
            max_evals = int(kwargs.get("max_evals", 60))

            cps_grid = kwargs.get("changepoint_prior_scale_grid", [0.01, 0.05, 0.1, 0.2])
            sps_grid = kwargs.get("seasonality_prior_scale_grid", [1.0, 5.0, 10.0, 20.0])
            mode_grid = kwargs.get("seasonality_mode_grid", ["additive", "multiplicative"])

            # si tu serie es mensual (freq="MS"), weekly/daily casi siempre sobran
            yearly = kwargs.get("yearly_seasonality", "auto")
            weekly = kwargs.get("weekly_seasonality", "auto")
            daily  = kwargs.get("daily_seasonality", "auto")

            combos = [(cps, sps, mode) for cps in cps_grid for sps in sps_grid for mode in mode_grid]

            if search == "random" and len(combos) > max_evals:
                rng = np.random.default_rng(kwargs.get("random_state", None))
                idx = rng.choice(len(combos), size=max_evals, replace=False)
                combos = [combos[i] for i in idx]

            best_score = float("inf")
            best_params = None

            for cps, sps, mode in combos:
                try:
                    m = ProphetModel(
                        seasonality_mode=mode,
                        yearly_seasonality=yearly,
                        weekly_seasonality=weekly,
                        daily_seasonality=daily,
                        changepoint_prior_scale=float(cps),
                        seasonality_prior_scale=float(sps),
                    ).fit(self.y_train_)
                    yhat = m.predict(steps=len(self.y_test_))
                    sc = rmse(self.y_test_, yhat)
                    if sc < best_score:
                        best_score = sc
                        best_params = {"changepoint_prior_scale": float(cps),
                                    "seasonality_prior_scale": float(sps),
                                    "seasonality_mode": mode,
                                    "yearly_seasonality": yearly,
                                    "weekly_seasonality": weekly,
                                    "daily_seasonality": daily}
                except Exception:
                    continue

            if best_params is None:
                raise RuntimeError("PROPHET tune: no se encontró configuración válida.")

            self.models_["PROPHET"] = ProphetModel(**best_params)
            return self

        # ============================================================
        # XGBoost TS
        # ============================================================

        if name == "XGBTS":
            if not _HAS_XGB:
                raise ImportError("xgboost no disponible.")

            search = str(kwargs.get("search", "random")).lower()   # XGB suele ir mejor random
            max_evals = int(kwargs.get("max_evals", 60))
            rng = np.random.default_rng(kwargs.get("random_state", None))

            # grids
            lag_sets = kwargs.get("lag_sets", [[1,2,3,6,12], [1,2,3,12], [1,2,3,4,5,6,12]])
            roll_sets = kwargs.get("roll_sets", [[3,6,12], [3,6], [6,12]])
            depth_grid = kwargs.get("max_depth_grid", [3,4,5,6])
            lr_grid = kwargs.get("learning_rate_grid", [0.02, 0.05, 0.1])
            n_est_grid = kwargs.get("n_estimators_grid", [300, 500, 800])
            subs_grid = kwargs.get("subsample_grid", [0.8, 0.9, 1.0])
            col_grid = kwargs.get("colsample_bytree_grid", [0.8, 0.9, 1.0])

            combos = []
            for lags in lag_sets:
                for rolls in roll_sets:
                    for md in depth_grid:
                        for lr in lr_grid:
                            for ne in n_est_grid:
                                for ss in subs_grid:
                                    for cc in col_grid:
                                        combos.append((lags, rolls, md, lr, ne, ss, cc))

            if search == "random" and len(combos) > max_evals:
                idx = rng.choice(len(combos), size=max_evals, replace=False)
                combos = [combos[i] for i in idx]

            best_score = float("inf")
            best_params = None

            for lags, rolls, md, lr, ne, ss, cc in combos:
                try:
                    m = XGBTSModel(
                        lags=lags,
                        rolling_windows=rolls,
                        n_estimators=int(ne),
                        max_depth=int(md),
                        learning_rate=float(lr),
                        subsample=float(ss),
                        colsample_bytree=float(cc),
                    ).fit(self.y_train_)
                    yhat = m.predict(steps=len(self.y_test_))
                    sc = rmse(self.y_test_, yhat)
                    if sc < best_score:
                        best_score = sc
                        best_params = {
                            "lags": lags,
                            "rolling_windows": rolls,
                            "n_estimators": int(ne),
                            "max_depth": int(md),
                            "learning_rate": float(lr),
                            "subsample": float(ss),
                            "colsample_bytree": float(cc),
                        }
                except Exception:
                    continue

            if best_params is None:
                raise RuntimeError("XGBTS tune: no se encontró configuración válida.")

            self.models_["XGBTS"] = XGBTSModel(**best_params)
            return self

    # -----------------------------
    # 6) Compete (eval todos)
    # -----------------------------

    def compete(self, metric: str = "rmse", verbose: bool = True) -> pd.DataFrame:
        #"""
        #Evalúa todos los modelos registrados y retorna el leaderboard.

        #verbose:
        #  - True  -> imprime tablas resumen con pretty_table (leaderboard + best params + distribuciones).
        #  - False -> no imprime nada; solo devuelve el DataFrame.
        #"""

        if self.y_train_ is None or self.y_test_ is None:
            raise RuntimeError("Ejecuta prepare() antes de compete().")

        if metric != "rmse":
            raise ValueError("Por ahora solo metric='rmse'.")

        # -----------------------------
        # 1) Evaluar todos los modelos
        # -----------------------------
        rows = []
        for name, model in self.models_.items():
            try:
                m_fit = _clone_model(model).fit(self.y_train_)
                yhat = m_fit.predict(steps=len(self.y_test_))
                sc = rmse(self.y_test_, yhat)

                rows.append({
                    "model": name,
                    "rmse": float(sc),
                    "params": self._params_of_model(model, fitted=m_fit)
                })
            except Exception:
                rows.append({
                    "model": name,
                    "rmse": np.nan,
                    "params": None
                })

        lb = pd.DataFrame(rows).sort_values("rmse", ascending=True).reset_index(drop=True)
        self.leaderboard_ = lb

        # -----------------------------
        # 2) Determinar best
        # -----------------------------
        valid = lb.dropna(subset=["rmse"])
        if len(valid) > 0:
            best_row = valid.iloc[0]
            self.best_model_name_ = str(best_row["model"])
            self.best_score_ = float(best_row["rmse"])
            self.best_params_ = dict(best_row["params"]) if isinstance(best_row["params"], dict) else None
        else:
            self.best_model_name_ = None
            self.best_score_ = None
            self.best_params_ = None

        # -----------------------------
        # 3) Mostrar leaderboard (DFCore pretty_table)
        # -----------------------------
        lb["params_str"] = lb["params"].apply(self._params_to_str)
        show_lb = lb[["model", "rmse", "params_str"]].copy()

        if verbose:
            self.pretty_table(
                show_lb,
                title="Leaderboard (RMSE) — menor es mejor",
                precision=4,
                max_colwidth=500
            )

        # -----------------------------
        # 4) Mostrar summary del ganador (params + distribuciones)
        # -----------------------------
        if verbose and self.best_model_name_ is not None:
            bp = self.best_params()
            self.pretty_table(pd.DataFrame([bp]), title="Best model params", precision=4)

            dist_all = self.value_distribution(use="all")
            dist_test = self.value_distribution(use="test")

            self.pretty_table(pd.DataFrame([dist_all]), title="Target distribution (ALL)", precision=4)
            self.pretty_table(pd.DataFrame([dist_test]), title="Target distribution (TEST)", precision=4)

            iqr = dist_all["iqr"]
            rng = dist_all["max"] - dist_all["min"]

            rel = {
                "rmse": float(self.best_score_),
                "rmse_over_iqr": float(self.best_score_ / iqr) if iqr and iqr != 0 else np.nan,
                "rmse_over_range": float(self.best_score_ / rng) if rng and rng != 0 else np.nan,
            }
            self.pretty_table(pd.DataFrame([rel]), title="RMSE contextualizado (ALL)", precision=4)

        return lb

    def _params_of_model(self, spec: BaseTSModel, fitted: Optional[BaseTSModel] = None) -> Dict[str, Any]:
        
        #Params para leaderboard.
        #- Por defecto toma del spec (lo configurado).
        #- Si pasas fitted, toma del fitted cuando aplica (ETS alpha_/beta_/gamma_).
        
        m = fitted if fitted is not None else spec

        if isinstance(spec, SARIMAModel):
            return {"order": spec.order, "seasonal_order": spec.seasonal_order}

        if isinstance(spec, LSTMModel):
            return {"lookback": spec.lookback, "epochs": spec.epochs, "batch_size": spec.batch_size}

        if isinstance(spec, CNN1DModel):
            return {
                "lookback": spec.lookback, "epochs": spec.epochs, "batch_size": spec.batch_size,
                "filters": spec.filters, "kernel_size": spec.kernel_size, "l2": spec.l2
            }

        if isinstance(spec, ETSModel):
            # para ETS: preferimos lo que quedó en el FIT
            alpha = getattr(m, "alpha_", None)
            beta  = getattr(m, "beta_", None)
            gamma = getattr(m, "gamma_", None)

            # fallback al spec si fitted no trae nada
            if alpha is None: alpha = getattr(spec, "alpha", None)
            if beta  is None: beta  = getattr(spec, "beta", None)
            if gamma is None: gamma = getattr(spec, "gamma", None)

            return {
                "trend": spec.trend,
                "seasonal": spec.seasonal,
                "seasonal_periods": spec.seasonal_periods,
                "alpha": alpha,
                "beta": beta,
                "gamma": gamma,
                "optimized": getattr(spec, "optimized", None),
            }
        
        if isinstance(spec, ProphetModel):
            return {
                "seasonality_mode": spec.seasonality_mode,
                "yearly_seasonality": spec.yearly_seasonality,
                "weekly_seasonality": spec.weekly_seasonality,
                "daily_seasonality": spec.daily_seasonality,
                "changepoint_prior_scale": spec.changepoint_prior_scale,
                "seasonality_prior_scale": spec.seasonality_prior_scale,
                "holidays_prior_scale": spec.holidays_prior_scale,
                "interval_width": spec.interval_width,
                "add_country_holidays": spec.add_country_holidays,
            }

        if isinstance(spec, XGBTSModel):
            return {
                "lags": spec.lags,
                "rolling_windows": spec.rolling_windows,
                "use_date_parts": spec.use_date_parts,
                "n_estimators": spec.n_estimators,
                "max_depth": spec.max_depth,
                "learning_rate": spec.learning_rate,
                "subsample": spec.subsample,
                "colsample_bytree": spec.colsample_bytree,
                "reg_lambda": spec.reg_lambda,
                "reg_alpha": spec.reg_alpha,
            }

        return {}

    def value_distribution(self, use: str = "all") -> Dict[str, float]:
        # Distribución de la variable objetivo para contextualizar métricas (RMSE).
        # use: 'train' | 'test' | 'all'
       
        if use not in ("train", "test", "all"):
            raise ValueError("use soportado: 'train' | 'test' | 'all'")

        if self.y_train_ is None or self.y_test_ is None:
            raise RuntimeError("Ejecuta prepare() antes de value_distribution().")

        if use == "train":
            y = self.y_train_
        elif use == "test":
            y = self.y_test_
        else:
            y = pd.concat([self.y_train_, self.y_test_])

        yv = y.astype(float).values

        dist = {
            "min": float(np.min(yv)),
            "q1": float(np.quantile(yv, 0.25)),
            "median": float(np.quantile(yv, 0.50)),
            "q3": float(np.quantile(yv, 0.75)),
            "max": float(np.max(yv)),
            "iqr": float(np.quantile(yv, 0.75) - np.quantile(yv, 0.25)),
        }
        return dist

    # -----------------------------
    # 6b) Best params summary
    # -----------------------------
    def best_params(self) -> Dict[str, Any]:
        if self.best_model_name_ is None:
            raise RuntimeError("No hay best model. Corre compete() primero.")

        out = {"model": self.best_model_name_}

        bp = getattr(self, "best_params_", None)
        if isinstance(bp, dict):
            out.update(bp)

        return out

    # -----------------------------
    # 7) Fit final
    # -----------------------------
    def fit(self, model: str = "best", refit_on: str = "train") -> "TimeSeriesForecaster":
        if self.y_train_ is None:
            raise RuntimeError("Ejecuta prepare() antes de fit().")

        if model == "best":
            if self.best_model_name_ is None:
                raise RuntimeError("No hay best model. Corre compete() primero.")
            model_name = self.best_model_name_
        else:
            model_name = model

        if model_name not in self.models_:
            raise ValueError(f"Modelo '{model_name}' no existe.")

        if refit_on not in ["train", "train_test_full"]:
            raise ValueError("refit_on soportado: 'train' | 'train_test_full'")

        if refit_on == "train":
            yfit = self.y_train_
        else:
            # Para producir forecast final con toda la serie (como tu punto e)
            if self.y_ is None:
                raise RuntimeError("No existe y_.")
            yfit = self.y_

        self.fitted_model_ = _clone_model(self.models_[model_name]).fit(yfit)
        self.best_model_name_ = model_name
        return self

    # -----------------------------
    # 8) Predict
    # -----------------------------
    def predict(self, steps: int) -> pd.Series:
        if self.fitted_model_ is None:
            raise RuntimeError("Ejecuta fit() antes de predict().")
        return self.fitted_model_.predict(steps=int(steps))
    
    # -----------------------------
    # 8b) Forecast as DataFrame
    # -----------------------------
    def forecast_df(
        self,
        steps: int,
        date_col: Optional[str] = None,
        pred_col: Optional[str] = None,
        include_model: bool = True
    ) -> pd.DataFrame:
        #"""
        #Devuelve el forecast futuro como DataFrame con fechas y predicción.

        #Parameters
        #----------
        #steps : int
        #    Número de pasos futuros a predecir.
        #date_col : str, optional
        #    Nombre de la columna de fechas en el output. Default: self.date_col.
        #pred_col : str, optional
        #    Nombre de la columna de predicción. Default: f"pred_{self.target_col}".
        #include_model : bool, default=True
        #    Si True, agrega columna 'model' con el nombre del modelo usado.

        #Returns
        #-------
        #pd.DataFrame con columnas: [date_col, pred_col] (+ 'model' si include_model=True)
        #"""
        yhat = self.predict(steps=int(steps))  # pd.Series con DatetimeIndex

        out_date_col = date_col if date_col is not None else self.date_col
        out_pred_col = pred_col if pred_col is not None else f"pred_{self.target_col}"

        df_pred = pd.DataFrame({
            out_date_col: yhat.index,
            out_pred_col: yhat.values
        })

        if include_model:
            df_pred["model"] = self.best_model_name_

        return df_pred

    # -----------------------------
    # 9) Evaluate (sobre test) 
    # -----------------------------
    def evaluate_test(self) -> Dict[str, Any]:
        if self.y_test_ is None:
            raise RuntimeError("Ejecuta prepare() primero.")
        if self.fitted_model_ is None:
            raise RuntimeError("Ejecuta fit() primero.")

        yhat = self.fitted_model_.predict(steps=len(self.y_test_))
        score = float(rmse(self.y_test_, yhat))

        # distribución para contextualizar RMSE
        y_all = (
            pd.concat([self.y_train_, self.y_test_])
            if (self.y_train_ is not None and self.y_test_ is not None)
            else self.y_test_
        )
        yv = y_all.astype(float).values

        dist = {
            "min": float(np.min(yv)),
            "q1": float(np.quantile(yv, 0.25)),
            "median": float(np.quantile(yv, 0.50)),
            "q3": float(np.quantile(yv, 0.75)),
            "max": float(np.max(yv)),
        }

        return {
            "model": self.best_model_name_,
            "rmse": score,
            "dist": dist,
        }

    # -----------------------------
    # 10) Plot forecast
    # -----------------------------
    def plot_forecast(self, steps: int = 30, show_test_pred: bool = True, figsize=(14, 6)):
        """
        Plotea:
        - Train (real)
        - Test (real)
        - (opcional) Predicción sobre test (fit en train, forecast len(test))
        - Forecast futuro (requiere fit() previo)
        """
        if self.y_train_ is None or self.y_test_ is None:
            raise RuntimeError("Ejecuta prepare() antes de plot_forecast().")

        import matplotlib.pyplot as plt

        plt.figure(figsize=figsize)

        # 1) Real: train y test
        plt.plot(self.y_train_, label="Train", linewidth=2)
        plt.plot(self.y_test_, label="Test real", linewidth=2)

        # 2) Pred test (backtest visual)
        if show_test_pred:
            if self.best_model_name_ is None:
                raise RuntimeError("Corre compete() para definir best_model antes de show_test_pred.")
            m_test = _clone_model(self.models_[self.best_model_name_]).fit(self.y_train_)
            yhat_test = m_test.predict(steps=len(self.y_test_))
            plt.plot(yhat_test, label="Pred test", linestyle="--", linewidth=2)

        # 3) Forecast futuro (usa fitted_model_)
        if self.fitted_model_ is not None:
            yhat_future = self.fitted_model_.predict(steps=int(steps))
            plt.plot(yhat_future, label=f"Forecast futuro (+{steps})", linewidth=2)
            plt.axvline(self.y_.index[-1], linestyle="--")  # separador último real
        else:
            # separador entre train y test (si aún no fit final)
            plt.axvline(self.y_train_.index[-1], linestyle="--")

        plt.title(f"Forecast — best={self.best_model_name_}")
        plt.legend()
        plt.grid(True)
        plt.show()

    # ------------------------------------------------------------
    # (TS) 1) Reporte de fechas faltantes (SERIES CORE)
    # ------------------------------------------------------------
    def _missing_report_series(
        self,
        series: pd.Series,
        freq: str = None,
    ) -> pd.DataFrame:

        if not isinstance(series.index, pd.DatetimeIndex):
            raise ValueError("series debe tener DatetimeIndex. Use to_series() primero.")

        freq_use = freq or getattr(self, "freq", None)
        if freq_use is None:
            raise ValueError("No hay freq definida. Pase freq=... o defina self.freq.")

        s = series.sort_index()
        full_idx = pd.date_range(start=s.index.min(), end=s.index.max(), freq=freq_use)
        missing = full_idx.difference(s.index)

        report = {
            "freq": freq_use,
            "start": s.index.min(),
            "end": s.index.max(),
            "n_expected": int(len(full_idx)),
            "n_present": int(s.index.nunique()),
            "n_missing": int(len(missing)),
            "missing_dates": list(missing)
        }
        return pd.DataFrame([report])

    # ------------------------------------------------------------
    # (TS) 2) Agregar fechas faltantes (SERIES CORE)
    # ------------------------------------------------------------
    def _add_missing_dates_series(
        self,
        series: pd.Series,
        freq: str = None
    ) -> pd.Series:

        if not isinstance(series.index, pd.DatetimeIndex):
            raise ValueError("series debe tener DatetimeIndex. Use to_series() primero.")

        freq_use = freq or getattr(self, "freq", None)
        if freq_use is None:
            raise ValueError("No hay freq definida. Pase freq=... o defina self.freq.")

        s = series.sort_index()
        full_idx = pd.date_range(start=s.index.min(), end=s.index.max(), freq=freq_use)
        s2 = s.reindex(full_idx)
        s2.index.name = series.index.name
        return s2

    # ------------------------------------------------------------
    # (TS) 3) Imputación + suavizado (SERIES CORE)
    # ------------------------------------------------------------
    def _impute_smoothed_series(
        self,
        series: pd.Series,
        *,
        smooth: str = "rolling_mean",
        window: int = 5,
        center: bool = True,
        impute: str = "time",
        limit: int = None,
        keep_original_name: bool = True
    ) -> pd.Series:

        if not isinstance(series.index, pd.DatetimeIndex):
            raise ValueError("series debe tener DatetimeIndex. Use to_series() primero.")

        s = series.sort_index().copy()

        # (A) suavizado
        if smooth is None or str(smooth).lower() in ("none", ""):
            s_smooth = s
        else:
            sm = str(smooth).lower()
            if sm == "rolling_mean":
                s_smooth = s.rolling(window, min_periods=1, center=center).mean()
            elif sm == "rolling_median":
                s_smooth = s.rolling(window, min_periods=1, center=center).median()
            elif sm == "ewm":
                span = max(2, int(window))
                s_smooth = s.ewm(span=span, adjust=False).mean()
            else:
                raise ValueError("smooth debe ser rolling_mean | rolling_median | ewm | None")

        # (B) imputación
        if impute is None or str(impute).lower() in ("none", ""):
            out = s_smooth
        else:
            imp = str(impute).lower()
            if imp in ("time", "linear"):
                method = "time" if imp == "time" else "linear"
                out = s_smooth.interpolate(method=method, limit=limit)
            elif imp == "ffill":
                out = s_smooth.ffill(limit=limit)
            elif imp == "bfill":
                out = s_smooth.bfill(limit=limit)
            elif imp == "zero":
                out = s_smooth.fillna(0.0)
            else:
                raise ValueError("impute debe ser time | linear | ffill | bfill | zero | None")

        if keep_original_name:
            out.name = series.name
        return out

    # ------------------------------------------------------------
    # (TS) API PÚBLICA (SIN PASAR SERIES)
    # ------------------------------------------------------------
    def missing_report(self, freq: str = None) -> pd.DataFrame:
        """
        Reporte de fechas faltantes sobre self.y_.
        (No modifica la serie)
        """
        if self.y_ is None:
            raise RuntimeError("Ejecuta to_series() primero.")
        return self._missing_report_series(self.y_, freq=freq)

    def add_missing_dates(self, freq: str = None) -> "TimeSeriesForecaster":
        """
        Reindexa self.y_ a freq fija, creando NaNs donde falten fechas.
        (Modifica self.y_)
        """
        if self.y_ is None:
            raise RuntimeError("Ejecuta to_series() primero.")
        self.y_ = self._add_missing_dates_series(self.y_, freq=freq)
        return self

    def impute_smoothed(
        self,
        *,
        smooth: str = "rolling_mean",
        window: int = 5,
        center: bool = True,
        impute: str = "time",
        limit: int = None
    ) -> "TimeSeriesForecaster":
        """
        Suaviza + imputa sobre self.y_.
        (Modifica self.y_)
        """
        if self.y_ is None:
            raise RuntimeError("Ejecuta to_series() primero.")
        self.y_ = self._impute_smoothed_series(
            self.y_,
            smooth=smooth,
            window=window,
            center=center,
            impute=impute,
            limit=limit,
            keep_original_name=True
        )
        return self
    

    # -----------------------------
    # 11) One-shot pipeline
    # -----------------------------
    def run_auto(
        self,
        *,
        # --- data & missing dates
        agg: str = "sum",
        check_missing: bool = True,
        add_missing: bool = False,                 # si ya haces asfreq en to_series, puede ser False
        impute_missing: bool = False,
        impute_kwargs: Optional[Dict[str, Any]] = None,

        # --- split
        train_size: float = 0.95,
        test_size: Optional[int] = None,

        # --- periodogram
        use_periodogram: bool = True,
        best_periods: int = 3,
        min_period: int = 2,
        max_period: Optional[int] = None,

        # --- models
        include_lstm: bool = True,
        include_cnn: bool = True,
        include_prophet: bool = True,
        include_xgb: bool = True,

        # --- tuning
        tune_models: bool = True,
        p_candidates: Optional[List[int]] = None,
        S_candidates: Optional[List[int]] = None,
        dl_epochs: int = 20,
        n_jobs: int = -1,
        tune_prophet: bool = True,
        tune_xgb: bool = True,
        prophet_max_evals: int = 40,
        xgb_max_evals: int = 60,

        # --- compete & fit
        metric: str = "rmse",
        verbose: bool = True,
        refit_on: str = "train_test_full",

        # --- optional forecast
        forecast_steps: Optional[int] = None,
        return_forecast_df: bool = True,
    ) -> Dict[str, Any]:
        #"""
        #Ejecuta el pipeline completo de forecasting en una sola llamada.

        #Returns
        #-------
        #dict con llaves típicas:
        #  - "missing_report": DataFrame (si check_missing)
        #  - "leaderboard": DataFrame
        #  - "best_model": str
        #  - "best_score": float
        #  - "forecast": DataFrame (si forecast_steps y return_forecast_df)
        #"""

        # defaults
        if impute_kwargs is None:
            impute_kwargs = dict(smooth="rolling_mean", window=5, center=True, impute="time", limit=None)

        out: Dict[str, Any] = {}

        # 1) df -> serie
        self.to_series(agg=agg)

        # 2) missing dates (reporte)
        if check_missing:
            out["missing_report"] = self.missing_report()

        # 3) agregar faltantes (opcional)
        if add_missing:
            self.add_missing_dates()

        # 4) imputación (opcional)
        if impute_missing:
            self.impute_smoothed(**impute_kwargs)

        # 5) split temporal
        self.prepare(test_size=test_size, train_size=train_size)

        # 6) periodograma (opcional)
        if use_periodogram:
            self.periodogram_candidates(
                best=best_periods,
                round_int=True,
                min_period=min_period,
                max_period=max_period
            )

        # 7) registrar modelos
        self.add_default_models(
            include_lstm=include_lstm,
            include_cnn=include_cnn,
            include_prophet=include_prophet,
            include_xgb=include_xgb
        )

        # 8) tuning (opcional)
        if tune_models:
            # candidatos por defecto: periodograma si existe
            if p_candidates is None:
                p_candidates = self.period_candidates_ if self.period_candidates_ else [7]
            if S_candidates is None:
                S_candidates = self.period_candidates_ if self.period_candidates_ else [7]

            if include_cnn and "CNN1D" in self.models_:
                self.tune("CNN1D", p_candidates=p_candidates, dl_epochs=dl_epochs)
            if include_lstm and "LSTM" in self.models_:
                self.tune("LSTM", p_candidates=p_candidates, dl_epochs=dl_epochs)
            if "SARIMA" in self.models_:
                self.tune(
                    "SARIMA",
                    n_jobs=n_jobs,
                    S_candidates=S_candidates,
                    search="grid",
                    max_p=2, max_d=1, max_q=2,
                    max_P=1, max_D=1, max_Q=1,
                    # opcional: max_evals si random
                )
            if include_prophet and "PROPHET" in self.models_ and tune_prophet:
                if self.freq in ("MS", "M", "QS", "Q", "A", "Y"):
                    yearly, weekly, daily = "auto", False, False
                elif self.freq in ("D", "B"):
                    yearly, weekly, daily = "auto", "auto", False
                else:
                    yearly, weekly, daily = "auto", "auto", "auto"

                self.tune(
                    "PROPHET",
                    search="random",
                    max_evals=int(prophet_max_evals),
                    yearly_seasonality=yearly,
                    weekly_seasonality=weekly,
                    daily_seasonality=daily,
                )
            if include_xgb and "XGBTS" in self.models_ and tune_xgb:
                self.tune("XGBTS", search="random", max_evals=int(xgb_max_evals))

        # 9) competir
        lb = self.compete(metric=metric, verbose=verbose)
        out["leaderboard"] = lb
        out["best_model"] = self.best_model_name_
        out["best_score"] = self.best_score_

        # 10) fit final (por defecto con toda la serie para forecast futuro)
        self.fit(model="best", refit_on=refit_on)

        # 11) forecast opcional
        if forecast_steps is not None and int(forecast_steps) > 0:
            if return_forecast_df and hasattr(self, "forecast_df"):
                out["forecast"] = self.forecast_df(steps=int(forecast_steps))
            else:
                out["forecast"] = self.predict(steps=int(forecast_steps))

        return out

# ============================================================
# Clone helper (porque sklearn.clone no aplica directo)
# ============================================================

def _clone_model(m: BaseTSModel) -> BaseTSModel:
    if isinstance(m, ETSModel):
        return ETSModel(
            trend=m.trend,
            seasonal=m.seasonal,
            seasonal_periods=m.seasonal_periods,
            alpha=getattr(m, "alpha", None),
            beta=getattr(m, "beta", None),
            gamma=getattr(m, "gamma", None),
            optimized=getattr(m, "optimized", True),
        )
    if isinstance(m, SARIMAModel):
        return SARIMAModel(order=m.order, seasonal_order=m.seasonal_order)
    if isinstance(m, LSTMModel):
        return LSTMModel(lookback=m.lookback, epochs=m.epochs, batch_size=m.batch_size, verbose=m.verbose)
    if isinstance(m, CNN1DModel):
        return CNN1DModel(
            lookback=m.lookback,
            epochs=m.epochs,
            batch_size=m.batch_size,
            verbose=m.verbose,
            validation_split=m.validation_split,
            early_stopping=m.early_stopping,
            patience=m.patience,
            restore_best_weights=m.restore_best_weights,
            reduce_lr=m.reduce_lr,
            lr_monitor=m.lr_monitor,
            lr_factor=m.lr_factor,
            lr_patience=m.lr_patience,
            lr_min=m.lr_min,
            filters=m.filters,
            kernel_size=m.kernel_size,
            l2=m.l2,
        )
    if isinstance(m, ProphetModel):
        return ProphetModel(
            seasonality_mode=m.seasonality_mode,
            yearly_seasonality=m.yearly_seasonality,
            weekly_seasonality=m.weekly_seasonality,
            daily_seasonality=m.daily_seasonality,
            changepoint_prior_scale=m.changepoint_prior_scale,
            seasonality_prior_scale=m.seasonality_prior_scale,
            holidays_prior_scale=m.holidays_prior_scale,
            interval_width=m.interval_width,
            add_country_holidays=m.add_country_holidays,
        )

    if isinstance(m, XGBTSModel):
        return XGBTSModel(
            lags=list(m.lags),
            rolling_windows=list(m.rolling_windows),
            use_date_parts=m.use_date_parts,
            n_estimators=m.n_estimators,
            max_depth=m.max_depth,
            learning_rate=m.learning_rate,
            subsample=m.subsample,
            colsample_bytree=m.colsample_bytree,
            reg_lambda=m.reg_lambda,
            reg_alpha=m.reg_alpha,
            random_state=m.random_state,
        )
    raise TypeError("No se puede clonar el modelo.")
