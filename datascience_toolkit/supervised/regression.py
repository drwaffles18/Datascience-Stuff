"""
regression.py
=============
Motor de regresión supervisada modular y reproducible (RegressionTournament / SupervisedRegression).
Incluye preprocesamiento libre de fuga, torneos de modelos (RMSE), HPO, explicabilidad LIME/SHAP y reportes.
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

from sklearn.base import clone, BaseEstimator, RegressorMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (
    OneHotEncoder,
    OrdinalEncoder,
    StandardScaler,
    MinMaxScaler,
    RobustScaler,
)
from sklearn.model_selection import (
    train_test_split,
    KFold,
    cross_val_score,
    GridSearchCV,
    RandomizedSearchCV,
)
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    make_scorer,
)

from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

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


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


RMSE_SCORER = make_scorer(
    lambda yt, yp: -_rmse(yt, yp),
    greater_is_better=True
)


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
                raise ValueError(f"Columnas faltantes en df: {missing}")
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
            and c not in datetime_cols and c not in bool_cols
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

    def pretty_table(
        self,
        df: pd.DataFrame,
        title: Optional[str] = None,
        head: Optional[int] = None,
        precision: int = 4,
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

                def _color_strength(val):
                    if not isinstance(val, str):
                        return ""
                    v = val.strip().lower()
                    if v == "strong":
                        return "background-color: #d4edda; color: #155724; font-weight: bold;"
                    elif v == "moderate":
                        return "background-color: #fff3cd; color: #856404; font-weight: bold;"
                    elif v == "weak":
                        return "background-color: #ffe5d0; color: #842029; font-weight: bold;"
                    elif v in ("negligible", "poor"):
                        return "background-color: #f8d7da; color: #721c24; font-weight: bold;"
                    return ""

                sty = (
                    view.style.format(precision=precision)
                    .set_properties(**{"text-align": "left"})
                    .set_table_styles([{"selector": "th", "props": [("text-align", "left")]}])
                )
                if "strength" in view.columns:
                    try:
                        sty = sty.map(_color_strength, subset=["strength"])
                    except AttributeError:
                        sty = sty.applymap(_color_strength, subset=["strength"])

                display(sty)
                return
            except Exception:
                pass

        with pd.option_context(
            "display.max_rows", 200, "display.max_columns", 200,
            "display.width", 120, "display.max_colwidth", max_colwidth,
        ):
            print(view)


class ModelInterpretabilityMixin:
    def _require_shap(self) -> None:
        if not _HAS_SHAP or shap is None:
            raise ImportError("SHAP no está instalado. Ejecuta: pip install shap")
        if getattr(self, "best_estimator_", None) is None:
            raise RuntimeError("Debes ejecutar fit() antes de usar SHAP.")

    def _require_lime(self) -> None:
        if not _HAS_LIME or LimeTabularExplainer is None:
            raise ImportError("LIME no está instalado. Ejecuta: pip install lime")
        if getattr(self, "best_estimator_", None) is None:
            raise RuntimeError("Debes ejecutar fit() antes de usar LIME.")

    def _shap_raw_split(self, data: Union[str, pd.DataFrame] = "test") -> pd.DataFrame:
        if isinstance(data, pd.DataFrame):
            return data.copy()
        key = str(data).lower().strip()
        attr_map = {"train": "X_train_", "valid": "X_valid_", "test": "X_test_"}
        if key not in attr_map:
            raise ValueError("data debe ser 'train', 'valid', 'test' o un DataFrame.")
        val = getattr(self, attr_map[key], None)
        if val is None:
            raise RuntimeError(f"Split '{key}' no disponible.")
        return val.copy() if isinstance(val, pd.DataFrame) else pd.DataFrame(val)

    def _shap_model_space(self, raw_df: pd.DataFrame) -> Tuple[Any, pd.DataFrame, Optional[Any]]:
        est = self.best_estimator_
        if isinstance(est, Pipeline) and "model" in est.named_steps:
            model = est.named_steps["model"]
            pre = est.named_steps.get("preprocess")
            X_arr = pre.transform(raw_df) if pre is not None else np.asarray(raw_df)
            X_arr = X_arr.toarray() if hasattr(X_arr, "toarray") else np.asarray(X_arr)
            try:
                names = list(pre.get_feature_names_out())
            except Exception:
                names = [f"f_{i}" for i in range(X_arr.shape[1])]
            return model, pd.DataFrame(X_arr, columns=names, index=raw_df.index), pre
        return est, raw_df.copy(), None

    def prepare_lime_explainer(
        self,
        background: Union[str, pd.DataFrame] = "train",
        max_background: int = 2000,
        discretize_continuous: bool = True,
        discretizer: str = "quartile",
        random_state: Optional[int] = 42,
        force: bool = False,
    ) -> Any:
        self._require_lime()
        if getattr(self, "lime_explainer_", None) is not None and not force:
            return self.lime_explainer_

        raw_bg = self._shap_raw_split(background)
        if len(raw_bg) > int(max_background):
            raw_bg = raw_bg.sample(n=int(max_background), random_state=random_state)

        model, X_bg, pre = self._shap_model_space(raw_bg)
        X_arr = np.asarray(X_bg, dtype=float)

        self.lime_explainer_ = LimeTabularExplainer(
            training_data=X_arr,
            feature_names=list(X_bg.columns),
            mode="regression",
            discretize_continuous=bool(discretize_continuous),
            discretizer=discretizer,
            random_state=random_state,
        )
        self.lime_model_ = model
        self.lime_preprocessor_ = pre
        self.lime_background_ = X_bg
        return self.lime_explainer_

    def explain_instance_lime(
        self,
        index: int,
        data: Union[str, pd.DataFrame] = "test",
        num_features: int = 10,
        num_samples: int = 5000,
        show: bool = True,
    ) -> Dict[str, Any]:
        self._require_lime()
        if getattr(self, "lime_explainer_", None) is None:
            self.prepare_lime_explainer()

        raw = self._shap_raw_split(data)
        model, X_df, _ = self._shap_model_space(raw)
        row = np.asarray(X_df.iloc[int(index)], dtype=float).reshape(-1)

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
        table["direction"] = np.where(table["lime_weight"] >= 0, "increases output", "decreases output")
        table = table.sort_values("abs_weight", ascending=False).reset_index(drop=True)

        local_pred_arr = np.asarray(getattr(exp, "local_pred", []), dtype=float).reshape(-1)
        local_prediction = float(local_pred_arr[0]) if local_pred_arr.size else np.nan
        fidelity = float(getattr(exp, "score", np.nan))

        if show:
            try:
                import matplotlib.pyplot as plt
                fig = exp.as_pyplot_figure()
                plt.tight_layout()
                plt.show()
            except Exception as exc:
                warnings.warn(f"No se pudo renderizar el gráfico LIME: {exc}")

        res = {
            "index": int(index),
            "class_label": None,
            "model_prediction": model_prediction,
            "local_prediction": local_prediction,
            "local_fidelity": fidelity,
            "explanation": table,
        }
        self.lime_last_explanation_ = res
        return res

    def lime_summary(self, show: bool = True) -> pd.DataFrame:
        res = getattr(self, "lime_last_explanation_", None)
        if res is None:
            raise RuntimeError("Ejecuta explain_instance_lime() primero.")
        if show:
            print("=" * 58)
            print("LIME LOCAL EXPLANATION SUMMARY")
            print("=" * 58)
            print(f"Model prediction : {res['model_prediction']:.4f}")
            print(f"Local prediction : {res['local_prediction']:.4f}")
            print(f"Local fidelity R²: {res['local_fidelity']:.4f}")
            self.pretty_table(res["explanation"], title="Reglas locales LIME")
        return res["explanation"].copy()


@dataclass
class ModelSpec:
    estimator: Any
    param_grid: Optional[Dict[str, Any]] = None
    param_dist: Optional[Dict[str, Any]] = None


class SupervisedRegression(ModelInterpretabilityMixin, DFCore):
    """Motor integral de regresión supervisada con torneos y optimización."""

    def __init__(
        self,
        df: pd.DataFrame,
        target: str,
        features: Optional[List[str]] = None,
        output_dir: str = "outputs_reg",
    ):
        DFCore.__init__(self, df=df, target=target, features=features)
        self.output_dir = output_dir
        _ensure_dir(self.output_dir)

        self.preprocess_: Optional[ColumnTransformer] = None
        self.model_specs_: Dict[str, ModelSpec] = {}
        self.tuned_models_: Dict[str, Dict[str, Any]] = {}
        self.hpo_results_: Dict[str, pd.DataFrame] = {}

        self.X_train_: Optional[pd.DataFrame] = None
        self.y_train_: Optional[np.ndarray] = None
        self.X_valid_: Optional[pd.DataFrame] = None
        self.y_valid_: Optional[np.ndarray] = None
        self.X_test_: Optional[pd.DataFrame] = None
        self.y_test_: Optional[np.ndarray] = None

        self.cv_: Optional[KFold] = None
        self.scoring_: str = "neg_root_mean_squared_error"
        self.leaderboard_: Optional[pd.DataFrame] = None
        self.best_model_name_: Optional[str] = None
        self.best_estimator_: Any = None
        self.refit_on_: Optional[str] = None
        self.saved_path_: Optional[str] = None

    def _build_preprocess(
        self,
        X: pd.DataFrame,
        scale_num: str = "standard",
        encoding: str = "auto",
        auto_cardinality_threshold: int = 15,
        num_impute: str = "median",
        cat_impute: str = "most_frequent",
    ) -> ColumnTransformer:
        dt_cols = [c for c in X.columns if is_datetime64_any_dtype(X[c])]
        bool_cols = [c for c in X.columns if is_bool_dtype(X[c]) and c not in dt_cols]
        num_cols = [c for c in X.columns if is_numeric_dtype(X[c]) and c not in dt_cols and c not in bool_cols]
        cat_cols = [c for c in X.columns if (is_object_dtype(X[c]) or isinstance(X[c].dtype, pd.CategoricalDtype)) and c not in dt_cols and c not in bool_cols]
        cat_cols += bool_cols

        steps_num: List[Tuple[str, Any]] = [("imputer", SimpleImputer(strategy=num_impute))]
        if scale_num == "standard":
            steps_num.append(("scaler", StandardScaler()))
        elif scale_num == "robust":
            steps_num.append(("scaler", RobustScaler()))
        elif scale_num == "minmax":
            steps_num.append(("scaler", MinMaxScaler()))

        transformers = [("num", Pipeline(steps_num), num_cols)]

        if cat_cols:
            enc = encoding.lower().strip()
            if enc == "ohe":
                transformers.append(("cat", Pipeline([("imputer", SimpleImputer(strategy=cat_impute)), ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), cat_cols))
            elif enc == "ordinal":
                transformers.append(("cat", Pipeline([("imputer", SimpleImputer(strategy=cat_impute)), ("ord", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))]), cat_cols))
            else:
                low = [c for c in cat_cols if X[c].nunique(dropna=False) <= auto_cardinality_threshold]
                high = [c for c in cat_cols if X[c].nunique(dropna=False) > auto_cardinality_threshold]
                if low:
                    transformers.append(("cat_ohe", Pipeline([("imputer", SimpleImputer(strategy=cat_impute)), ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), low))
                if high:
                    transformers.append(("cat_ord", Pipeline([("imputer", SimpleImputer(strategy=cat_impute)), ("ord", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))]), high))

        return ColumnTransformer(transformers=transformers, remainder="drop")

    def _make_pipeline(self, estimator: Any) -> Pipeline:
        return Pipeline([("preprocess", clone(self.preprocess_)), ("model", clone(estimator))])

    def prepare(
        self,
        test_size: float = 0.20,
        valid_size: float = 0.15,
        cv_splits: int = 5,
        cv_shuffle: bool = True,
        scale_num: str = "standard",
        model_encoding: str = "auto",
        auto_cardinality_threshold: int = 15,
        num_impute: str = "median",
        cat_impute: str = "most_frequent",
        **kwargs,
    ) -> Dict[str, Any]:
        df_clean = self.df.dropna(subset=[self.target]).reset_index(drop=True)
        X = df_clean[[c for c in self.features if c != self.target]].copy()
        y = df_clean[self.target].to_numpy().astype(float)

        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_size)

        if valid_size > 0:
            X_tr, X_va, y_tr, y_va = train_test_split(X_tr, y_tr, test_size=valid_size)
            self.X_valid_ = X_va.reset_index(drop=True)
            self.y_valid_ = y_va
        else:
            self.X_valid_, self.y_valid_ = None, None

        self.X_train_ = X_tr.reset_index(drop=True)
        self.y_train_ = y_tr
        self.X_test_ = X_te.reset_index(drop=True)
        self.y_test_ = y_te

        self.preprocess_ = self._build_preprocess(
            X=self.X_train_,
            scale_num=scale_num,
            encoding=model_encoding,
            auto_cardinality_threshold=auto_cardinality_threshold,
            num_impute=num_impute,
            cat_impute=cat_impute,
        )
        self.preprocess_.fit(self.X_train_)
        self.cv_ = KFold(n_splits=cv_splits, shuffle=cv_shuffle)

        return {
            "X_train": self.X_train_,
            "y_train": self.y_train_,
            "X_valid": self.X_valid_,
            "y_valid": self.y_valid_,
            "X_test": self.X_test_,
            "y_test": self.y_test_,
            "preprocessor": self.preprocess_,
        }

    def add_default_models(self) -> SupervisedRegression:
        self.model_specs_.clear()
        self.model_specs_["Linear"] = ModelSpec(estimator=LinearRegression())
        self.model_specs_["Lasso"] = ModelSpec(
            estimator=Lasso(max_iter=5000),
            param_grid={"model__alpha": [1e-4, 1e-3, 1e-2, 1e-1, 1.0]},
            param_dist={"model__alpha": np.logspace(-4, 0, 50)},
        )
        self.model_specs_["Ridge"] = ModelSpec(
            estimator=Ridge(),
            param_grid={"model__alpha": [1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0]},
            param_dist={"model__alpha": np.logspace(-3, 2, 80)},
        )
        self.model_specs_["SVR"] = ModelSpec(
            estimator=SVR(kernel="rbf"),
            param_grid={"model__C": [0.1, 1, 10, 100], "model__epsilon": [0.01, 0.05, 0.1, 0.2]},
            param_dist={"model__C": np.logspace(-1, 2, 60), "model__epsilon": np.linspace(0.01, 0.3, 60)},
        )
        self.model_specs_["DecisionTree"] = ModelSpec(
            estimator=DecisionTreeRegressor(),
            param_grid={"model__max_depth": [2, 3, 5, 10, None], "model__min_samples_split": [2, 5, 10]},
            param_dist={"model__max_depth": [None] + list(range(2, 21)), "model__min_samples_split": list(range(2, 31))},
        )
        self.model_specs_["RandomForest"] = ModelSpec(
            estimator=RandomForestRegressor(n_estimators=300),
            param_grid={"model__max_depth": [3, 5, 10, None], "model__min_samples_split": [2, 5, 10]},
            param_dist={"model__n_estimators": [200, 300, 500], "model__max_depth": [None] + list(range(2, 31))},
        )
        self.model_specs_["GBR"] = ModelSpec(
            estimator=GradientBoostingRegressor(),
            param_grid={"model__n_estimators": [100, 200, 300], "model__learning_rate": [0.01, 0.05, 0.1], "model__max_depth": [2, 3, 4]},
            param_dist={"model__n_estimators": [100, 200, 300, 500], "model__learning_rate": np.logspace(-2.5, -0.7, 50), "model__max_depth": [2, 3, 4, 5]},
        )
        return self

    def compete(self, models: Optional[List[str]] = None, metric: str = "rmse", n_jobs: int = -1) -> pd.DataFrame:
        if self.X_train_ is None:
            raise RuntimeError("Ejecuta prepare() primero.")
        if not self.model_specs_:
            self.add_default_models()

        candidate_names = models or list(self.model_specs_.keys())
        rows = []
        for name in candidate_names:
            if name not in self.model_specs_:
                continue
            if name in self.tuned_models_ and "best_estimator" in self.tuned_models_[name]:
                est = self.tuned_models_[name]["best_estimator"]
                tuned = True
            else:
                base = self.model_specs_[name].estimator
                est = self._make_pipeline(base)
                tuned = False

            scores = cross_val_score(est, self.X_train_, self.y_train_, scoring=RMSE_SCORER, cv=self.cv_, n_jobs=n_jobs)
            rmse_scores = -scores
            rows.append({
                "model": name,
                "metric": metric,
                "cv_mean": float(np.mean(rmse_scores)),
                "cv_std": float(np.std(rmse_scores)),
                "n_splits": int(self.cv_.get_n_splits()),
                "tuned": tuned,
            })

        lb = pd.DataFrame(rows).sort_values("cv_mean", ascending=True).reset_index(drop=True)
        self.leaderboard_ = lb
        return lb
    def tune(self, model_name: str, method: str = "random", n_iter: int = 25, n_jobs: int = -1, verbose: int = 0) -> pd.DataFrame:
        if self.X_train_ is None:
            raise RuntimeError("Ejecuta prepare() primero.")
        if model_name not in self.model_specs_:
            raise ValueError(f"Modelo '{model_name}' no registrado.")

        spec = self.model_specs_[model_name]
        base_pipe = self._make_pipeline(spec.estimator)
        base_rmse = -float(np.mean(cross_val_score(base_pipe, self.X_train_, self.y_train_, scoring=RMSE_SCORER, cv=self.cv_, n_jobs=n_jobs)))

        t0 = time.time()
        space = spec.param_grid or {} if method == "grid" or spec.param_dist is None else spec.param_dist or {}
        search_cls = GridSearchCV if method == "grid" or spec.param_dist is None else RandomizedSearchCV
        search_kwargs = {"cv": self.cv_, "scoring": RMSE_SCORER, "n_jobs": n_jobs, "verbose": verbose, "refit": True}
        if search_cls is RandomizedSearchCV:
            search_kwargs["n_iter"] = n_iter
        search = search_cls(base_pipe, space, **search_kwargs)
        search.fit(self.X_train_, self.y_train_)
        elapsed = time.time() - t0
        tuned_rmse = -float(search.best_score_)

        # Champion Fallback: se adopta el tuning solo si supera estrictamente al baseline
        improved = tuned_rmse < base_rmse
        chosen_est = search.best_estimator_ if improved else base_pipe.fit(self.X_train_, self.y_train_)
        chosen_rmse = tuned_rmse if improved else base_rmse
        msg = f"✅ {model_name}: HPO mejoró ({base_rmse:.4f} -> {tuned_rmse:.4f})" if improved else f"⚠️ {model_name}: HPO ({tuned_rmse:.4f}) no superó baseline ({base_rmse:.4f}). Se preserva baseline."
        print(msg)

        self.tuned_models_[model_name] = {
            "method": method if improved else "baseline_champion",
            "params": search.best_params_ if improved else {},
            "best_score": chosen_rmse,
            "elapsed_s": float(elapsed),
            "best_estimator": chosen_est,
            "tuned": improved,
        }
        res_df = pd.DataFrame(search.cv_results_).sort_values("rank_test_score").reset_index(drop=True)
        self.hpo_results_[model_name] = res_df
        return res_df

    def tune_models(self, models: List[str], method: str = "random", n_iter: int = 25, **kwargs) -> Dict[str, pd.DataFrame]:
        results = {}
        for name in models:
            if name in self.model_specs_:
                try:
                    results[name] = self.tune(name, method=method, n_iter=n_iter, **kwargs)
                except Exception as e:
                    warnings.warn(f"No se pudo optimizar {name}: {e}")
        return results

    def tune_top(self, top: int = 3, method: str = "random", n_iter: int = 25, **kwargs) -> Dict[str, pd.DataFrame]:
        if self.leaderboard_ is None or self.leaderboard_.empty:
            raise RuntimeError("Ejecuta compete() antes de tune_top().")
        names = self.leaderboard_["model"].tolist()[:max(1, int(top))]
        return self.tune_models(models=names, method=method, n_iter=n_iter, **kwargs)

    def fit(self, model: str = "best", refit_on: str = "train_valid") -> SupervisedRegression:
        if self.X_train_ is None or self.y_train_ is None:
            raise RuntimeError("Ejecuta prepare() primero.")

        chosen = str(self.leaderboard_.iloc[0]["model"]) if model == "best" else model
        if chosen in self.tuned_models_ and "best_estimator" in self.tuned_models_[chosen]:
            est = self.tuned_models_[chosen]["best_estimator"]
        else:
            base = self.model_specs_[chosen].estimator
            est = self._make_pipeline(base)

        if refit_on == "train_valid" and self.X_valid_ is not None:
            X_refit = pd.concat([self.X_train_, self.X_valid_], axis=0).reset_index(drop=True)
            y_refit = np.concatenate([self.y_train_, self.y_valid_], axis=0)
        else:
            X_refit, y_refit = self.X_train_, self.y_train_

        est.fit(X_refit, y_refit)
        self.best_model_name_ = chosen
        self.best_estimator_ = est
        self.refit_on_ = refit_on
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.best_estimator_ is None:
            raise RuntimeError("Ejecuta fit() primero.")
        return np.asarray(self.best_estimator_.predict(X)).ravel()

    def evaluate_test(self) -> pd.DataFrame:
        if self.best_estimator_ is None:
            raise RuntimeError("Ejecuta fit() primero.")
        y_pred = self.predict(self.X_test_)
        y_true = self.y_test_

        report = pd.DataFrame([{
            "model": self.best_model_name_,
            "rmse": _rmse(y_true, y_pred),
            "mae": float(mean_absolute_error(y_true, y_pred)),
            "r2": float(r2_score(y_true, y_pred)),
            "n_test": len(y_true),
        }])
        self.test_report_ = report
        return report

    def y_distribution_summary(self) -> pd.DataFrame:
        s = pd.Series(self.df[self.target].dropna())
        val_min = float(s.min())
        val_max = float(s.max())
        q1 = float(s.quantile(0.25))
        q3 = float(s.quantile(0.75))

        # Márgenes seguros hacia los extremos históricos
        dist_min = max(0.0001, q1 - val_min)
        dist_max = max(0.0001, val_max - q3)
        safe_margin = min(dist_min, dist_max)

        row = {
            "Variable": self.target,
            "Min": val_min,
            "Q1": q1,
            "Q3": q3,
            "Max": val_max,
        }

        # 1. Obtener la métrica ganadora (priorizando Test sobre CV)
        if getattr(self, "test_report_", None) is not None and not self.test_report_.empty:
            rep = self.test_report_.iloc[0]
            chosen_metric = "rmse"
            if self.leaderboard_ is not None and not self.leaderboard_.empty:
                chosen_metric = str(self.leaderboard_.iloc[0]["metric"]).lower().strip()

            err = float(rep[chosen_metric]) if chosen_metric in rep else float(rep["rmse"])
            row["Winner_Model"] = str(rep["model"])
            row["Metric"] = chosen_metric.upper()
            row["Error"] = err
            row["Evaluation"] = "Test"
        elif self.leaderboard_ is not None and not self.leaderboard_.empty:
            winner = self.leaderboard_.iloc[0]
            chosen_metric = str(winner["metric"]).lower().strip()
            err = float(winner["cv_mean"])
            row["Winner_Model"] = str(winner["model"])
            row["Metric"] = chosen_metric.upper()
            row["Error"] = err
            row["Evaluation"] = "CV"
        else:
            return pd.DataFrame([row])

        # 2. Diagnóstico según la métrica elegida
        if chosen_metric in ("rmse", "neg_root_mean_squared_error"):
            # Si el error excede el margen al mínimo o al máximo, desborda los datos históricos
            if (q1 - err < val_min) or (q3 + err > val_max):
                strength = "Negligible"
            else:
                ratio = err / safe_margin
                if ratio < 0.35:
                    strength = "Strong"
                elif ratio < 0.70:
                    strength = "Moderate"
                elif ratio <= 1.00:
                    strength = "Weak"
                else:
                    strength = "Negligible"

        elif chosen_metric in ("mae", "neg_mean_absolute_error"):
            if (q1 - err < val_min) or (q3 + err > val_max):
                strength = "Negligible"
            else:
                ratio = err / safe_margin
                if ratio < 0.25:
                    strength = "Strong"
                elif ratio < 0.50:
                    strength = "Moderate"
                elif ratio < 0.75:
                    strength = "Weak"
                else:
                    strength = "Negligible"

        elif chosen_metric in ("r2", "r2_score"):
            if err >= 0.75:
                strength = "Strong"
            elif err >= 0.50:
                strength = "Moderate"
            elif err >= 0.25:
                strength = "Weak"
            else:
                strength = "Negligible"

        elif chosen_metric in ("mape", "mean_absolute_percentage_error"):
            if err <= 0.10:
                strength = "Strong"
            elif err <= 0.20:
                strength = "Moderate"
            elif err <= 0.35:
                strength = "Weak"
            else:
                strength = "Negligible"

        else:
            ratio = err / safe_margin
            strength = "Strong" if ratio < 0.35 else ("Moderate" if ratio < 0.70 else "Weak")

        row["strength"] = strength
        return pd.DataFrame([row])

    def save(self, path: str, kind: str = "pickle") -> str:
        if self.best_estimator_ is None:
            raise RuntimeError("Ejecuta fit() primero.")
        _ensure_dir(os.path.dirname(path) or ".")
        bundle = {
            "model_name": self.best_model_name_,
            "model": self.best_estimator_,
            "preprocess": self.preprocess_,
            "leaderboard": self.leaderboard_,
            "target": self.target,
        }
        joblib.dump(bundle, path)
        self.saved_path_ = path
        print(f"Modelo guardado exitosamente en: {path}")
        return path


RegressionTournament = SupervisedRegression

__all__ = [
    "SupervisedRegression",
    "RegressionTournament",
    "ModelSpec",
    "RMSE_SCORER",
]