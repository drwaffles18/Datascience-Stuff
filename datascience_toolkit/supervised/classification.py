"""
classification.py
=================
Motor de clasificación supervisada modular y reproducible (ClassificationTournament / SupervisedClassification).
Incluye preprocesamiento libre de fuga (data leakage), HPO, torneos de modelos,
optimización de umbrales para clasificación binaria, interpretabilidad (SHAP / LIME) y reportes ejecutivos.
"""

from __future__ import annotations

import os
import time
import json
import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union, Callable

import numpy as np
import pandas as pd
import joblib

from pandas.api.types import (
    is_bool_dtype,
    is_datetime64_any_dtype,
    is_numeric_dtype,
    is_object_dtype,
)

from sklearn.base import clone, BaseEstimator
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
from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_val_score,
    GridSearchCV,
    RandomizedSearchCV,
)
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    log_loss,
    classification_report,
    make_scorer,
)

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn.discriminant_analysis import (
    QuadraticDiscriminantAnalysis,
    LinearDiscriminantAnalysis,
)
from sklearn.naive_bayes import GaussianNB

# Dependencias opcionales
try:
    from xgboost import XGBClassifier
    _HAS_XGB = True
except Exception:
    _HAS_XGB = False

try:
    import shap
    _HAS_SHAP = True
except Exception:
    shap = None
    _HAS_SHAP = False

try:
    from lime.lime_tabular import LimeTabularExplainer
    _HAS_LIME = True
except Exception:
    LimeTabularExplainer = None
    _HAS_LIME = False

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import regularizers, layers
    _HAS_TF = True
except Exception:
    _HAS_TF = False

try:
    import keras_tuner as kt
    _HAS_KT = True
except Exception:
    _HAS_KT = False


@dataclass
class ColumnGroups:
    numeric: List[str]
    categorical: List[str]
    datetime: List[str]
    boolean: List[str]
    other: List[str]


class DFCore:
    def __init__(self, df: pd.DataFrame, target: Optional[str] = None, features: Optional[List[str]] = None):
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
        numeric_cols = [c for c in df.columns if is_numeric_dtype(df[c]) and c not in datetime_cols and c not in bool_cols]
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

    def _in_notebook(self) -> bool:
        try:
            from IPython import get_ipython
            ip = get_ipython()
            return ip is not None and "IPKernelApp" in ip.config
        except Exception:
            return False

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
            if title:
                print(title)
            print(df)
            return

        view = df.head(head) if head is not None else df
        if title:
            print(title)

        if self._in_notebook():
            try:
                from IPython.display import display
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

        with pd.option_context(
            "display.max_rows", 200,
            "display.max_columns", 200,
            "display.width", 120,
            "display.max_colwidth", max_colwidth,
        ):
            print(view)


class _SupervisedPreprocessor:
    def __init__(self, df: pd.DataFrame, target: str, features: Optional[List[str]] = None):
        self.df = df.copy()
        self.target = target
        self.features = features or [c for c in df.columns if c != target]

    def get_X_y(self, dropna_target: bool = True) -> Tuple[pd.DataFrame, pd.Series]:
        df = self.df.dropna(subset=[self.target]) if dropna_target else self.df
        X_cols = [c for c in self.features if c != self.target]
        return df[X_cols].copy(), df[self.target].copy()

    def build_preprocessor_model(
        self,
        X: pd.DataFrame,
        include_bool_as_cat: bool = True,
        encoding: str = "auto",
        auto_cardinality_threshold: int = 15,
        num_impute: str = "median",
        cat_impute: str = "most_frequent",
        scale_num: str = "standard",
        ohe_sparse: bool = False,
        **kwargs,
    ) -> ColumnTransformer:
        dt_cols = [c for c in X.columns if is_datetime64_any_dtype(X[c])]
        bool_cols = [c for c in X.columns if is_bool_dtype(X[c]) and c not in dt_cols]
        num_cols = [c for c in X.columns if is_numeric_dtype(X[c]) and c not in dt_cols and c not in bool_cols]
        cat_cols = [
            c for c in X.columns
            if (is_object_dtype(X[c]) or isinstance(X[c].dtype, pd.CategoricalDtype))
            and c not in dt_cols and c not in bool_cols
        ]
        if include_bool_as_cat:
            cat_cols += bool_cols
        else:
            num_cols += bool_cols

        steps_num: List[Tuple[str, Any]] = [("imputer", SimpleImputer(strategy=num_impute))]
        if scale_num == "standard":
            steps_num.append(("scaler", StandardScaler()))
        elif scale_num == "robust":
            steps_num.append(("scaler", RobustScaler()))
        elif scale_num == "minmax":
            steps_num.append(("scaler", MinMaxScaler()))
        num_pipe = Pipeline(steps=steps_num)

        transformers = [("num", num_pipe, num_cols)]

        if cat_cols:
            enc = encoding.lower().strip()
            if enc == "ohe":
                pipe_cat = Pipeline([
                    ("imputer", SimpleImputer(strategy=cat_impute)),
                    ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=ohe_sparse)),
                ])
                transformers.append(("cat", pipe_cat, cat_cols))
            elif enc == "ordinal":
                pipe_cat = Pipeline([
                    ("imputer", SimpleImputer(strategy=cat_impute)),
                    ("ord", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
                ])
                transformers.append(("cat", pipe_cat, cat_cols))
            else:  # auto
                low_card = [c for c in cat_cols if X[c].nunique(dropna=False) <= auto_cardinality_threshold]
                high_card = [c for c in cat_cols if X[c].nunique(dropna=False) > auto_cardinality_threshold]
                if low_card:
                    pipe_ohe = Pipeline([
                        ("imputer", SimpleImputer(strategy=cat_impute)),
                        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=ohe_sparse)),
                    ])
                    transformers.append(("cat_ohe", pipe_ohe, low_card))
                if high_card:
                    pipe_ord = Pipeline([
                        ("imputer", SimpleImputer(strategy=cat_impute)),
                        ("ord", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
                    ])
                    transformers.append(("cat_ord", pipe_ord, high_card))

        return ColumnTransformer(transformers=transformers, remainder="drop")

    def get_feature_names(self, pre: ColumnTransformer) -> List[str]:
        names: List[str] = []
        for name, transformer, cols in pre.transformers_:
            if cols is None or len(cols) == 0:
                continue
            if name.startswith("num"):
                names.extend(list(cols))
            elif isinstance(transformer, Pipeline):
                last = transformer.steps[-1][1]
                if isinstance(last, OneHotEncoder):
                    names.extend(last.get_feature_names_out(cols).tolist())
                else:
                    names.extend(list(cols))
            else:
                names.extend(list(cols))
        return names


@dataclass
class ClfModelSpec:
    name: str
    estimator: Any
    hpo_spaces: Dict[str, Any]
    supports_proba: bool = True
    tags: Tuple[str, ...] = ()


class ModelInterpretabilityMixin:
    def _require_shap(self) -> None:
        if not _HAS_SHAP or shap is None:
            raise ImportError("SHAP no está instalado. Ejecuta: pip install shap")
        if getattr(self, "best_estimator_", None) is None:
            raise RuntimeError("Debes ejecutar fit() antes de generar explicaciones SHAP.")

    def _shap_is_classification(self) -> bool:
        return True

    def _shap_raw_split(self, data: Union[str, pd.DataFrame] = "test") -> pd.DataFrame:
        if isinstance(data, pd.DataFrame):
            return data.copy()
        key = str(data).lower().strip()
        attr_map = {"train": "X_train_", "valid": "X_valid_", "validation": "X_valid_", "test": "X_test_"}
        if key not in attr_map:
            raise ValueError("data debe ser 'train', 'valid', 'test' o un DataFrame.")
        val = getattr(self, attr_map[key], None)
        if val is None:
            raise RuntimeError(f"El split '{key}' no está disponible.")
        return val.copy() if isinstance(val, pd.DataFrame) else pd.DataFrame(val)

    @staticmethod
    def _shap_dense(X):
        return X.toarray() if hasattr(X, "toarray") else np.asarray(X)

    def _shap_model_space(self, raw_df: pd.DataFrame) -> Tuple[Any, pd.DataFrame, Optional[Any]]:
        est = self.best_estimator_
        pre = getattr(self, "preprocessor_model_", None)
        if pre is not None:
            X_arr = self._shap_dense(pre.transform(raw_df))
            names = getattr(self, "feature_names_model_", None)
            if names is None:
                try:
                    names = pre.get_feature_names_out().tolist()
                except Exception:
                    names = [f"f_{i}" for i in range(X_arr.shape[1])]
            X_df = pd.DataFrame(X_arr, columns=list(names), index=raw_df.index)
            return est, X_df, pre
        return est, raw_df.copy(), None

    def _shap_positive_index(self) -> Optional[int]:
        idx = getattr(self, "positive_encoded_", None)
        return int(idx) if idx is not None else (1 if getattr(self, "n_classes_", 2) == 2 else None)

    def _shap_prediction_function(self, model: Any) -> Callable:
        pos_idx = self._shap_positive_index()
        def predict_fn(X):
            X = np.asarray(X)
            if hasattr(model, "predict_proba"):
                p = np.asarray(model.predict_proba(X))
                if p.ndim == 2 and pos_idx is not None and pos_idx < p.shape[1]:
                    return p[:, pos_idx]
                return p
            return np.asarray(model.predict(X))
        return predict_fn

    @staticmethod
    def _shap_model_family(model: Any) -> str:
        name = type(model).__name__.lower()
        if any(k in name for k in ("xgb", "randomforest", "extratrees", "decisiontree", "gradientboosting")):
            return "tree"
        if any(k in name for k in ("linear", "logistic", "ridge", "lasso")):
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
        self._require_shap()
        if getattr(self, "shap_explainer_", None) is not None and not force:
            return self.shap_explainer_

        raw_bg = self._shap_raw_split(background)
        if len(raw_bg) > int(max_background):
            raw_bg = raw_bg.sample(n=int(max_background), random_state=random_state)

        model, X_bg, pre = self._shap_model_space(raw_bg)
        family = self._shap_model_family(model)

        if family == "tree":
            try:
                explainer = shap.TreeExplainer(model, data=X_bg, model_output="probability", feature_perturbation="interventional")
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
            explainer = shap.Explainer(self._shap_prediction_function(model), X_bg, algorithm=resolved)
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
        return explainer

    def _shap_select_output(self, explanation: Any, class_label: Any = None) -> Any:
        vals = np.asarray(explanation.values)
        if vals.ndim <= 2:
            return explanation
        if class_label is not None and getattr(self, "label_encoder_", None) is not None:
            out_idx = int(self.label_encoder_.transform([class_label])[0])
        else:
            out_idx = self._shap_positive_index() or 0
        if vals.ndim == 3:
            return explanation[..., out_idx] if vals.shape[-1] <= max(2, getattr(self, "n_classes_", 2) or 2) else explanation[out_idx]
        return explanation

    def _shap_compute(self, X_df: pd.DataFrame, class_label: Any = None, max_evals: Optional[int] = None) -> Any:
        if getattr(self, "shap_explainer_", None) is None:
            self.prepare_explainer()
        kwargs = {"max_evals": int(max_evals)} if max_evals is not None else {}
        try:
            exp = self.shap_explainer_(X_df, **kwargs)
        except TypeError:
            exp = self.shap_explainer_(X_df)
        return self._shap_select_output(exp, class_label=class_label)

    def explain_global(
        self,
        data: Union[str, pd.DataFrame] = "test",
        sample_size: int = 500,
        plots: Tuple[str, ...] = ("beeswarm", "bar"),
        top_n: int = 15,
        class_label: Any = None,
        max_evals: Optional[int] = None,
        random_state: Optional[int] = 42,
        show: bool = True,
    ) -> Dict[str, Any]:
        self._require_shap()
        raw = self._shap_raw_split(data)
        if len(raw) > int(sample_size):
            raw = raw.sample(n=int(sample_size), random_state=random_state)
        _, X_df, _ = self._shap_model_space(raw)

        exp = self._shap_compute(X_df, class_label=class_label, max_evals=max_evals)
        vals = np.asarray(exp.values)

        importance = pd.DataFrame({
            "feature": list(X_df.columns),
            "mean_abs_shap": np.mean(np.abs(vals), axis=0),
            "mean_shap": np.mean(vals, axis=0),
        }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
        importance.insert(0, "rank", np.arange(1, len(importance) + 1))

        if show:
            for p in plots:
                if p == "beeswarm":
                    shap.plots.beeswarm(exp, max_display=int(top_n), show=True)
                elif p == "bar":
                    shap.plots.bar(exp, max_display=int(top_n), show=True)

        self.shap_last_explanation_ = exp
        self.shap_global_importance_ = importance
        return {"importance": importance, "explanation": exp, "X_explained": X_df}

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
        self._require_shap()
        raw = self._shap_raw_split(data)
        if index in raw.index:
            row_raw = raw.loc[[index]]
            src_idx = index
        else:
            pos = int(index)
            row_raw = raw.iloc[[pos]]
            src_idx = raw.index[pos]

        model, X_row, _ = self._shap_model_space(row_raw)
        exp = self._shap_compute(X_row, class_label=class_label, max_evals=max_evals)
        one = exp[0]
        vals = np.asarray(one.values).reshape(-1)

        local_df = pd.DataFrame({
            "feature": list(X_row.columns),
            "value": X_row.iloc[0].to_numpy(),
            "shap_value": vals,
        })
        local_df["abs_shap"] = local_df["shap_value"].abs()
        local_df["direction"] = np.where(local_df["shap_value"] > 0, "increases output", "decreases output")
        local_df = local_df.sort_values("abs_shap", ascending=False).reset_index(drop=True)

        pred_val = None
        if hasattr(model, "predict_proba"):
            p = model.predict_proba(X_row)
            pos_idx = self._shap_positive_index()
            if p.ndim == 2 and pos_idx is not None and pos_idx < p.shape[1]:
                pred_val = float(p[0, pos_idx])

        if show:
            if plot == "waterfall":
                shap.plots.waterfall(one, max_display=int(top_n), show=True)
            elif plot == "force":
                shap.plots.force(one, matplotlib=True, show=True)

        report = {
            "index": src_idx,
            "prediction": pred_val,
            "base_value": float(np.asarray(one.base_values).reshape(-1)[0]),
            "contributions": local_df,
            "explanation": one,
        }
        return report

    def explain_summary(self, top_n: int = 10, show: bool = True) -> pd.DataFrame:
        imp = getattr(self, "shap_global_importance_", None)
        if imp is None:
            res = self.explain_global(show=False, top_n=top_n)
            imp = res["importance"]
        top = imp.head(int(top_n)).copy()
        if show:
            print("=" * 58)
            print("MODEL INTERPRETABILITY SUMMARY")
            print("=" * 58)
            print(f"Model: {getattr(self, 'best_model_name_', 'best_model')}")
            print(f"Target: {getattr(self, 'target', None)}")
            self.pretty_table(top, title=f"Top {min(top_n, len(top))} global SHAP drivers")
        return top


class ClassificationTournament(ModelInterpretabilityMixin, DFCore):
    """Motor de torneos, búsqueda de hiperparámetros y optimización de clasificación."""

    def __init__(
        self,
        df: pd.DataFrame,
        target: str,
        features: Optional[List[str]] = None,
        output_dir: str = "supervised_outputs",
        positive_label: Any = None,
        class_of_interest: Any = None,
    ):
        DFCore.__init__(self, df=df, target=target, features=features)

        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        self.positive_label_requested_ = positive_label if positive_label is not None else class_of_interest
        self.class_of_interest_ = self.positive_label_requested_
        self.positive_encoded_: Optional[int] = None
        self.negative_label_: Any = None
        self.negative_encoded_: Optional[int] = None
        self.n_classes_: Optional[int] = None
        self.business_goal_: Optional[str] = None
        self.refit_on_: Optional[str] = None
        self.saved_path_: Optional[str] = None

        self.X_raw_ = None
        self.y_raw_ = None
        self.X_train_ = None
        self.X_valid_ = None
        self.X_test_ = None
        self.y_train_ = None
        self.y_valid_ = None
        self.y_test_ = None

        self.label_encoder_: Optional[LabelEncoder] = None
        self.y_train_enc_: Optional[pd.Series] = None
        self.y_valid_enc_: Optional[pd.Series] = None
        self.y_test_enc_: Optional[pd.Series] = None

        self.preprocessor_model_ = None
        self.feature_names_model_ = None
        self.X_train_model_ = None
        self.X_valid_model_ = None
        self.X_test_model_ = None

        self.cv_ = None
        self.scoring_ = "f1_macro"

        self.model_specs_: Dict[str, ClfModelSpec] = {}
        self.hpo_results_: Dict[str, pd.DataFrame] = {}
        self.best_params_: Dict[str, Dict[str, Any]] = {}
        self.leaderboard_: Optional[pd.DataFrame] = None

        self.best_model_name_: Optional[str] = None
        self.best_estimator_: Any = None
        self.threshold_: float = 0.5
        self.threshold_report_: Optional[pd.DataFrame] = None
        self.positive_label_: Any = self.positive_label_requested_

        self.last_test_default_: Optional[pd.DataFrame] = None
        self.last_test_threshold_: Optional[pd.DataFrame] = None

    def _y_train_for_fit(self) -> pd.Series:
        return self.y_train_enc_ if self.y_train_enc_ is not None else self.y_train_

    def _decode_labels(self, y_enc: np.ndarray) -> np.ndarray:
        return self.label_encoder_.inverse_transform(np.asarray(y_enc, dtype=int)) if self.label_encoder_ else y_enc

    def _encode_label(self, label: Any) -> int:
        return int(self.label_encoder_.transform([label])[0]) if self.label_encoder_ else label

    def _resolve_positive_class(self) -> None:
        if self.label_encoder_ is None:
            raise RuntimeError("LabelEncoder no ajustado.")
        classes = list(self.label_encoder_.classes_)
        self.n_classes_ = len(classes)
        if self.n_classes_ < 2:
            raise ValueError(f"Target '{self.target}' debe contener al menos dos clases.")

        req = self.positive_label_requested_
        if req is None and self.n_classes_ == 2:
            req = classes[1]
        if req is not None and req not in classes:
            raise ValueError(f"positive_label={req!r} no existe en target '{self.target}'. Clases: {classes}")

        self.positive_label_ = req
        self.class_of_interest_ = req
        if req is not None:
            self.positive_encoded_ = int(self.label_encoder_.transform([req])[0])
            self.negative_label_ = next(c for c in classes if c != req) if self.n_classes_ == 2 else None
            self.negative_encoded_ = int(self.label_encoder_.transform([self.negative_label_])[0]) if self.negative_label_ else None

    def _resolve_scorer(self, metric: Any = None) -> Any:
        metric = self.scoring_ if metric is None else metric
        if not isinstance(metric, str):
            return metric
        key = metric.lower().strip()
        alias = {
            "precision_class": "precision_positive",
            "recall_class": "recall_positive",
            "f1_class": "f1_positive",
            "precision_interest": "precision_positive",
            "recall_interest": "recall_positive",
            "f1_interest": "f1_positive",
        }.get(key, key)

        if alias in ("precision_positive", "recall_positive", "f1_positive"):
            if self.positive_encoded_ is None:
                raise ValueError(f"Métrica '{metric}' requiere positive_label en la instanciación.")
            fn = {
                "precision_positive": precision_score,
                "recall_positive": recall_score,
                "f1_positive": f1_score,
            }[alias]
            return make_scorer(fn, pos_label=self.positive_encoded_, zero_division=0)
        return metric

    def recommend_metrics(
        self,
        business_goal: Optional[str] = None,
        show: bool = True,
    ) -> Optional[pd.DataFrame]:
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

        df_rec = pd.DataFrame(rows)
        df_rec.insert(0, "business_goal", goal)
        df_rec.insert(1, "positive_label", self.positive_label_)
        df_rec.insert(2, "imbalance_ratio", ratio)
        df_rec.insert(3, "balance_status", "Balanced" if balanced else "Imbalanced")
        self.metric_recommendations_ = df_rec

        if show:
            try:
                from IPython.display import display
                styled = df_rec.style.set_properties(
                    subset=["purpose", "note"],
                    **{"white-space": "normal", "text-align": "left"},
                )
                display(styled)
            except Exception:
                print(df_rec.to_string(index=False))
            return None

        return df_rec

    def prepare(
        self,
        test_size: float = 0.20,
        valid_size: float = 0.0,
        stratify: bool = True,
        model_encoding: str = "auto",
        auto_cardinality_threshold: int = 15,
        include_bool_as_cat: bool = True,
        num_impute: str = "median",
        cat_impute: str = "most_frequent",
        scale_num: str = "standard",
        ohe_sparse: bool = False,
        cv_splits: int = 5,
        cv_shuffle: bool = True,
        scoring: str = "f1_macro",
        **kwargs,
    ) -> Dict[str, Any]:
        prep = _SupervisedPreprocessor(self.df, target=self.target, features=self.features)
        X, y = prep.get_X_y(dropna_target=True)

        self.X_raw_ = X.reset_index(drop=True)
        self.y_raw_ = y.reset_index(drop=True)
        strat = self.y_raw_ if stratify else None

        X_tr, X_te, y_tr, y_te = train_test_split(self.X_raw_, self.y_raw_, test_size=test_size, stratify=strat)

        if valid_size > 0:
            strat_val = y_tr if stratify else None
            X_tr, X_va, y_tr, y_va = train_test_split(X_tr, y_tr, test_size=valid_size, stratify=strat_val)
            self.X_valid_ = X_va.reset_index(drop=True)
            self.y_valid_ = y_va.reset_index(drop=True)
        else:
            self.X_valid_, self.y_valid_ = None, None

        self.X_train_ = X_tr.reset_index(drop=True)
        self.y_train_ = y_tr.reset_index(drop=True)
        self.X_test_ = X_te.reset_index(drop=True)
        self.y_test_ = y_te.reset_index(drop=True)

        self.label_encoder_ = LabelEncoder()
        self.label_encoder_.fit(self.y_train_)
        self._resolve_positive_class()
        self.scoring_ = self._resolve_scorer(scoring)

        self.y_train_enc_ = pd.Series(self.label_encoder_.transform(self.y_train_), name=self.target)
        self.y_test_enc_ = pd.Series(self.label_encoder_.transform(self.y_test_), name=self.target)
        self.y_valid_enc_ = pd.Series(self.label_encoder_.transform(self.y_valid_), name=self.target) if self.y_valid_ is not None else None

        pre_model = prep.build_preprocessor_model(
            X=self.X_train_,
            include_bool_as_cat=include_bool_as_cat,
            encoding=model_encoding,
            auto_cardinality_threshold=auto_cardinality_threshold,
            num_impute=num_impute,
            cat_impute=cat_impute,
            scale_num=scale_num,
            ohe_sparse=ohe_sparse,
        )
        pre_model.fit(self.X_train_)
        self.preprocessor_model_ = pre_model
        self.feature_names_model_ = prep.get_feature_names(pre_model)

        self.X_train_model_ = pd.DataFrame(pre_model.transform(self.X_train_), columns=self.feature_names_model_)
        self.X_test_model_ = pd.DataFrame(pre_model.transform(self.X_test_), columns=self.feature_names_model_)
        self.X_valid_model_ = pd.DataFrame(pre_model.transform(self.X_valid_), columns=self.feature_names_model_) if self.X_valid_ is not None else None

        self.cv_ = StratifiedKFold(n_splits=cv_splits, shuffle=cv_shuffle)

        return {
            "X_train_model": self.X_train_model_,
            "X_valid_model": self.X_valid_model_,
            "X_test_model": self.X_test_model_,
            "y_train": self.y_train_,
            "positive_label": self.positive_label_,
            "scoring": scoring,
        }

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

        # SVM
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

        # Decision Tree
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

        # GBC
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

        # XGBoost
        if _HAS_XGB:
            self.model_specs_["XGB"] = ClfModelSpec(
                name="XGB",
                estimator=XGBClassifier(eval_metric="logloss", n_jobs=1),
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

        # QDA con regularización por defecto para mitigar multicolinealidad
        self.model_specs_["QDA"] = ClfModelSpec(
            name="QDA",
            estimator=QuadraticDiscriminantAnalysis(reg_param=0.05),
            hpo_spaces={
                "grid": {"reg_param": [0.001, 0.01, 0.05, 0.1, 0.2]},
                "random": {"reg_param": [0.001, 0.01, 0.03, 0.05, 0.1, 0.2]}
            },
            supports_proba=True,
            tags=("generative",)
        )

        # LDA
        self.model_specs_["LDA"] = ClfModelSpec(
            name="LDA",
            estimator=LinearDiscriminantAnalysis(solver="svd"),
            hpo_spaces={
                "grid": {"solver": ["svd", "lsqr", "eigen"]},
                "random": {"solver": ["svd", "lsqr", "eigen"]}
            },
            supports_proba=True,
            tags=("generative",)
        )

        # Naive Bayes
        self.model_specs_["NaiveBayes"] = ClfModelSpec(
            name="NaiveBayes",
            estimator=GaussianNB(),
            hpo_spaces={"grid": {}, "random": {}},
            supports_proba=True,
            tags=("baseline", "bayes")
        )

        # MLP (Keras)
        self.model_specs_["MLP"] = ClfModelSpec(
            name="MLP",
            estimator=None,
            hpo_spaces={
                "keras_tuner": {
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

    def compete(self, models: Optional[List[str]] = None, metric: Optional[str] = None, n_jobs: int = -1) -> pd.DataFrame:
        if self.X_train_model_ is None:
            raise RuntimeError("Debes ejecutar prepare() antes de compete().")
        if not self.model_specs_:
            self.add_default_models()

        candidate_models = models or list(self.model_specs_.keys())
        y_cv = self._y_train_for_fit()
        metric_name = metric or str(self.scoring_)
        scorer = self._resolve_scorer(metric_name)

        rows = []
        for name in candidate_models:
            if name not in self.model_specs_:
                continue

            if name in self.best_params_ and self.best_params_[name].get("method") == "keras_tuner":
                best_score = float(self.best_params_[name].get("best_score", np.nan))
                rows.append({
                    "model": name,
                    "metric": metric_name,
                    "cv_mean": best_score,
                    "cv_std": np.nan,
                    "n_splits": 0,
                    "tuned": True
                })
                continue

            if name in self.best_params_ and "estimator" in self.best_params_[name]:
                est = clone(self.best_params_[name]["estimator"])
                tuned = True
            else:
                base_est = self.model_specs_[name].estimator
                if base_est is None:
                    continue
                est = clone(base_est)
                tuned = False

            try:
                scores = cross_val_score(
                    est,
                    self.X_train_model_,
                    y_cv,
                    scoring=scorer,
                    cv=self.cv_,
                    n_jobs=n_jobs,
                    error_score=np.nan
                )
                cv_mean = float(np.nanmean(scores))
                cv_std = float(np.nanstd(scores))
            except Exception:
                if name == "QDA" and hasattr(est, "set_params"):
                    try:
                        est_fallback = clone(est)
                        est_fallback.set_params(reg_param=0.08)
                        scores = cross_val_score(
                            est_fallback,
                            self.X_train_model_,
                            y_cv,
                            scoring=scorer,
                            cv=self.cv_,
                            n_jobs=n_jobs,
                            error_score=np.nan
                        )
                        cv_mean = float(np.nanmean(scores))
                        cv_std = float(np.nanstd(scores))
                    except Exception:
                        cv_mean, cv_std = np.nan, np.nan
                else:
                    cv_mean, cv_std = np.nan, np.nan

            rows.append({
                "model": name,
                "metric": metric_name,
                "cv_mean": cv_mean,
                "cv_std": cv_std,
                "n_splits": int(self.cv_.get_n_splits()),
                "tuned": tuned,
            })

        lb = pd.DataFrame(rows).sort_values("cv_mean", ascending=False).reset_index(drop=True)
        self.leaderboard_ = lb
        return lb

    def _build_mlp_from_hp(self, hp, input_dim: int, n_classes: int):
        model = keras.Sequential()
        model.add(layers.Input(shape=(input_dim,)))
        n_layers = hp.Int("n_layers", 1, 3)
        for i in range(n_layers):
            units = hp.Int(f"units_{i}", min_value=32, max_value=256, step=32)
            dropout = hp.Float(f"dropout_{i}", 0.0, 0.5, step=0.1)
            l2v = hp.Float(f"l2_{i}", 1e-6, 1e-3, sampling="log")
            model.add(layers.Dense(units, activation="relu", kernel_regularizer=regularizers.l2(l2v)))
            if dropout > 0:
                model.add(layers.Dropout(dropout))

        if n_classes == 2:
            model.add(layers.Dense(1, activation="sigmoid"))
            loss = "binary_crossentropy"
            metrics = [keras.metrics.BinaryAccuracy(name="accuracy"), keras.metrics.AUC(name="auc")]
        else:
            model.add(layers.Dense(n_classes, activation="softmax"))
            loss = "sparse_categorical_crossentropy"
            metrics = [keras.metrics.SparseCategoricalAccuracy(name="accuracy")]

        lr = hp.Float("lr", 1e-4, 5e-3, sampling="log")
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr), loss=loss, metrics=metrics)
        return model

    def _tune_keras_tuner(self, model_name: str, verbose: int = 0, param_space: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        if not _HAS_TF or not _HAS_KT:
            raise ImportError("TensorFlow o KerasTuner no están disponibles.")
        spec_defaults = self.model_specs_[model_name].hpo_spaces.get("keras_tuner", {})
        cfg = dict(spec_defaults)
        if param_space:
            cfg.update(param_space)

        max_trials = int(cfg.get("max_trials", 20))
        epochs = int(cfg.get("epochs", 30))
        batch_size = int(cfg.get("batch_size", 64))
        patience = int(cfg.get("patience", 5))

        tuner_dir = os.path.join(os.path.expanduser("~"), "kt_dir")
        os.makedirs(tuner_dir, exist_ok=True)
        project_name = cfg.get("project_name", f"{self.target}_MLP")

        if self.X_valid_model_ is None:
            X_tr, X_va, y_tr, y_va = train_test_split(self.X_train_model_, self.y_train_enc_, test_size=0.15, stratify=self.y_train_enc_)
        else:
            X_tr, y_tr = self.X_train_model_, self.y_train_enc_
            X_va, y_va = self.X_valid_model_, self.y_valid_enc_

        X_tr_np, X_va_np = X_tr.to_numpy().astype("float32"), X_va.to_numpy().astype("float32")
        y_tr_np, y_va_np = y_tr.to_numpy().astype("int32"), y_va.to_numpy().astype("int32")
        input_dim = X_tr_np.shape[1]
        n_classes = len(np.unique(y_tr_np))

        obj_name = "val_auc" if n_classes == 2 else "val_accuracy"
        tf.keras.backend.clear_session()
        tuner = kt.RandomSearch(
            lambda hp: self._build_mlp_from_hp(hp, input_dim=input_dim, n_classes=n_classes),
            objective=kt.Objective(obj_name, direction="max"),
            max_trials=max_trials,
            directory=tuner_dir,
            project_name=project_name,
            overwrite=True
        )

        t0 = time.time()
        tuner.search(
            X_tr_np, y_tr_np,
            validation_data=(X_va_np, y_va_np),
            epochs=epochs,
            batch_size=batch_size,
            verbose=verbose,
            callbacks=[keras.callbacks.EarlyStopping(monitor=obj_name, patience=patience, mode="max", restore_best_weights=True)]
        )
        elapsed = time.time() - t0

        best_hp = tuner.get_best_hyperparameters(1)[0]
        best_model = tuner.get_best_models(1)[0]
        pred = best_model.predict(X_va_np, verbose=0)
        y_hat = (pred.reshape(-1) >= 0.5).astype("int32") if n_classes == 2 else np.argmax(pred, axis=1).astype("int32")
        best_score = float(f1_score(y_va_np, y_hat, average="macro"))

        self.best_params_[model_name] = {
            "method": "keras_tuner",
            "params": best_hp.values,
            "best_score": best_score,
            "elapsed_s": float(elapsed),
            "estimator": best_model,
        }

        dfres = pd.DataFrame([{
            "model": model_name,
            "search_method": "keras_tuner",
            "best_score": best_score,
            "elapsed_s": float(elapsed),
            "params": json.dumps(best_hp.values, ensure_ascii=False)
        }])
        self.hpo_results_[model_name] = dfres
        return dfres

    def tune(self, model_name: str, method: str = "auto", param_space: Optional[Dict[str, Any]] = None, n_iter: int = 25, n_jobs: int = -1, verbose: int = 0) -> pd.DataFrame:
        if self.X_train_model_ is None:
            raise RuntimeError("Ejecuta prepare() primero.")
        if model_name not in self.model_specs_:
            raise ValueError(f"Modelo '{model_name}' no registrado.")

        spec = self.model_specs_[model_name]
        method_res = method.lower().strip()
        if method_res == "auto":
            method_res = "keras_tuner" if model_name == "MLP" else "random"

        if method_res == "keras_tuner":
            return self._tune_keras_tuner(model_name=model_name, verbose=verbose, param_space=param_space)

        space = param_space or spec.hpo_spaces.get(method_res, spec.hpo_spaces.get("grid", {}))
        est = clone(spec.estimator)
        t0 = time.time()

        if method_res == "grid" or not space:
            search = GridSearchCV(est, space, scoring=self.scoring_, cv=self.cv_, n_jobs=n_jobs, verbose=verbose, refit=True)
        else:
            search = RandomizedSearchCV(est, space, n_iter=n_iter, scoring=self.scoring_, cv=self.cv_, n_jobs=n_jobs, verbose=verbose, refit=True)

        search.fit(self.X_train_model_, self._y_train_for_fit())
        elapsed = time.time() - t0

        self.best_params_[model_name] = {
            "method": method_res,
            "params": search.best_params_,
            "best_score": float(search.best_score_),
            "elapsed_s": float(elapsed),
            "estimator": search.best_estimator_,
        }
        res_df = pd.DataFrame(search.cv_results_).sort_values("rank_test_score").reset_index(drop=True)
        self.hpo_results_[model_name] = res_df
        return res_df

    def tune_models(self, models: List[str], method: str = "auto", n_iter: int = 25, **kwargs) -> Dict[str, pd.DataFrame]:
        results = {}
        for name in models:
            if name in self.model_specs_:
                try:
                    results[name] = self.tune(name, method=method, n_iter=n_iter, **kwargs)
                except Exception as e:
                    warnings.warn(f"No se pudo optimizar {name}: {e}")
        return results

    def tune_top(self, top: int = 3, method: str = "auto", n_iter: int = 25, **kwargs) -> Dict[str, pd.DataFrame]:
        if self.leaderboard_ is None or self.leaderboard_.empty:
            raise RuntimeError("Ejecuta compete() antes de tune_top().")
        names = self.leaderboard_["model"].tolist()[:max(1, int(top))]
        return self.tune_models(models=names, method=method, n_iter=n_iter, **kwargs)

    def fit(self, model: str = "best", refit_on: str = "train") -> Any:
        if self.X_train_model_ is None:
            raise RuntimeError("Ejecuta prepare() primero.")

        chosen = str(self.leaderboard_.iloc[0]["model"]) if model == "best" else model
        if chosen in self.best_params_ and self.best_params_[chosen].get("method") == "keras_tuner":
            est = self.best_params_[chosen]["estimator"]
        elif chosen in self.best_params_:
            est = clone(self.best_params_[chosen]["estimator"])
        else:
            est = clone(self.model_specs_[chosen].estimator)

        X_fit = self.X_train_model_
        y_fit = self._y_train_for_fit()
        if refit_on == "train_valid" and self.X_valid_model_ is not None:
            X_fit = pd.concat([self.X_train_model_, self.X_valid_model_], axis=0).reset_index(drop=True)
            y_fit = pd.concat([self.y_train_enc_, self.y_valid_enc_], axis=0).reset_index(drop=True)

        if not str(type(est)).lower().__contains__("keras"):
            est.fit(X_fit, y_fit)

        self.best_model_name_ = chosen
        self.best_estimator_ = est
        self.refit_on_ = refit_on
        return est

    def _predict_proba_any(self, X) -> np.ndarray:
        est = self.best_estimator_
        if hasattr(est, "predict_proba"):
            return np.asarray(est.predict_proba(X))
        if hasattr(est, "predict"):
            p = np.asarray(est.predict(np.asarray(X).astype("float32"), verbose=0))
            if p.ndim == 1 or (p.ndim == 2 and p.shape[1] == 1):
                p1 = np.clip(p.reshape(-1, 1), 1e-7, 1 - 1e-7)
                return np.hstack([1 - p1, p1])
            return p
        raise ValueError("El modelo no soporta probabilidades predict_proba.")

    def predict(self, X_model: pd.DataFrame, use_threshold: bool = False) -> np.ndarray:
        if self.best_estimator_ is None:
            raise RuntimeError("Ejecuta fit() primero.")
        if not use_threshold:
            est = self.best_estimator_
            if str(type(est)).lower().__contains__("keras"):
                p = np.asarray(est.predict(np.asarray(X_model).astype("float32"), verbose=0))
                y_pred = (p.reshape(-1) >= 0.5).astype("int32") if p.shape[-1] <= 2 else np.argmax(p, axis=1).astype("int32")
                return self._decode_labels(y_pred)
            return self._decode_labels(est.predict(X_model))

        proba = self._predict_proba_any(X_model)
        pos_enc = self.positive_encoded_ if self.positive_encoded_ is not None else 1
        neg_enc = 1 - pos_enc
        preds_enc = np.where(proba[:, pos_enc] >= float(self.threshold_), pos_enc, neg_enc)
        return self._decode_labels(preds_enc)

    def optimize_threshold(
        self,
        positive_label: Any = None,
        objective: str = "precision",
        constraint: Optional[Dict[str, float]] = None,
        grid: Optional[np.ndarray] = None,
        use: str = "valid",
    ) -> pd.DataFrame:
        if self.best_estimator_ is None:
            raise RuntimeError("Ejecuta fit() primero.")
        if use == "valid" and self.X_valid_model_ is not None:
            X_eval = self.X_valid_model_
            y_eval_enc = self.y_valid_enc_.to_numpy()
        else:
            X_eval = self.X_test_model_
            y_eval_enc = self.y_test_enc_.to_numpy()

        pos_label = positive_label or self.positive_label_
        pos_enc = self._encode_label(pos_label)
        neg_enc = 1 - pos_enc
        proba_pos = self._predict_proba_any(X_eval)[:, pos_enc]
        th_grid = grid if grid is not None else np.round(np.arange(0.05, 0.96, 0.01), 2)

        rows = []
        for th in th_grid:
            y_pred = np.where(proba_pos >= th, pos_enc, neg_enc)
            prec = precision_score(y_eval_enc, y_pred, pos_label=pos_enc, zero_division=0)
            rec = recall_score(y_eval_enc, y_pred, pos_label=pos_enc, zero_division=0)
            f1 = f1_score(y_eval_enc, y_pred, pos_label=pos_enc, zero_division=0)

            ok = True
            if constraint:
                if "recall_min" in constraint and rec < float(constraint["recall_min"]):
                    ok = False
                if "precision_min" in constraint and prec < float(constraint["precision_min"]):
                    ok = False

            score = {"precision": prec, "recall": rec, "f1": f1}.get(objective, f1)
            rows.append({
                "threshold": float(th),
                "precision_pos": float(prec),
                "recall_pos": float(rec),
                "f1_pos": float(f1),
                "objective_value": float(score),
                "passes_constraint": ok,
            })

        df_th = pd.DataFrame(rows)
        df_valid = df_th[df_th["passes_constraint"]]
        if df_valid.empty:
            df_valid = df_th

        best = df_valid.sort_values("objective_value", ascending=False).iloc[0]
        self.threshold_ = float(best["threshold"])
        self.threshold_report_ = df_th
        return df_th

    def evaluate_test(self, use_threshold: bool = False) -> pd.DataFrame:
        if self.best_estimator_ is None:
            raise RuntimeError("Ejecuta fit() primero.")
        y_true = self.y_test_.to_numpy()
        y_pred = self.predict(self.X_test_model_, use_threshold=use_threshold)

        rep = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
        out = {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
            "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
            "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
            "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
            "use_threshold": use_threshold,
            "threshold": float(self.threshold_) if use_threshold else 0.5,
        }

        if self.positive_label_ in rep:
            p_key = str(self.positive_label_)
            out["precision_positive"] = float(rep[p_key]["precision"])
            out["recall_positive"] = float(rep[p_key]["recall"])
            out["f1_positive"] = float(rep[p_key]["f1-score"])

        try:
            proba = self._predict_proba_any(self.X_test_model_)
            pos_enc = self.positive_encoded_ if self.positive_encoded_ is not None else 1
            out["roc_auc"] = float(roc_auc_score(self.y_test_enc_, proba[:, pos_enc]))
            out["average_precision"] = float(average_precision_score(self.y_test_enc_, proba[:, pos_enc]))
        except Exception:
            out["roc_auc"] = np.nan
            out["average_precision"] = np.nan

        df_out = pd.DataFrame([out])
        if use_threshold:
            self.last_test_threshold_ = df_out.copy()
        else:
            self.last_test_default_ = df_out.copy()
        return df_out

    def save(self, path: str) -> str:
        if self.best_estimator_ is None:
            raise RuntimeError("Ejecuta fit() primero.")
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        bundle = {
            "preprocessor_model": self.preprocessor_model_,
            "feature_names_model": self.feature_names_model_,
            "label_encoder": self.label_encoder_,
            "model_name": self.best_model_name_,
            "model": self.best_estimator_,
            "threshold": float(self.threshold_),
            "positive_label": self.positive_label_,
            "positive_encoded": self.positive_encoded_,
            "leaderboard": self.leaderboard_,
        }
        joblib.dump(bundle, path)
        self.saved_path_ = path
        print(f"Modelo guardado exitosamente en: {path}")
        return path

    def executive_report(self, show: bool = True) -> pd.DataFrame:
        default_row = self.last_test_default_.iloc[0] if self.last_test_default_ is not None else None
        th_row = self.last_test_threshold_.iloc[0] if self.last_test_threshold_ is not None else default_row

        report = pd.DataFrame([{
            "target": self.target,
            "positive_label": self.positive_label_,
            "winning_model": self.best_model_name_,
            "decision_mode": "optimized_threshold" if self.last_test_threshold_ is not None else "default_threshold",
            "threshold": self.threshold_ if self.last_test_threshold_ is not None else 0.50,
            "accuracy": th_row.get("accuracy", np.nan) if th_row is not None else np.nan,
            "precision_positive": th_row.get("precision_positive", np.nan) if th_row is not None else np.nan,
            "recall_positive": th_row.get("recall_positive", np.nan) if th_row is not None else np.nan,
            "f1_positive": th_row.get("f1_positive", np.nan) if th_row is not None else np.nan,
            "roc_auc": default_row.get("roc_auc", np.nan) if default_row is not None else np.nan,
            "saved_path": self.saved_path_,
        }])
        if show:
            self.pretty_table(report, title="Resumen Ejecutivo de Clasificación")
        return report


# Alias transparente
SupervisedClassification = ClassificationTournament

__all__ = [
    "ClfModelSpec",
    "ModelInterpretabilityMixin",
    "ClassificationTournament",
    "SupervisedClassification",
]