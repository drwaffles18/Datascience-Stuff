"""
forecasting.py
==============
Temporal forecasting engine (TimeSeriesForecaster) with strict anti-leakage splits,
spectral periodogram analysis, multi-family model tournament, and Plotly Dark charts.
"""

from __future__ import annotations

import json
import logging
import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from pandas.api.types import (
    is_bool_dtype,
    is_datetime64_any_dtype,
    is_numeric_dtype,
    is_object_dtype,
)
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

try:
    import plotly.graph_objects as go
    _HAS_PLOTLY = True
except Exception:
    go = None
    _HAS_PLOTLY = False

from .diagnostics import PeriodogramHelper
from .models import (
    BaseTSModel,
    ETSModel,
    SARIMAModel,
    LSTMModel,
    CNN1DModel,
    ProphetModel,
    XGBTSModel,
    _clone_model,
    _HAS_TF,
    _HAS_PROPHET,
    _HAS_XGB,
)


def rmse(y_true: pd.Series, y_pred: pd.Series) -> float:
    y_true = pd.Series(y_true).astype(float)
    y_pred = pd.Series(y_pred).astype(float)
    return float(np.sqrt(mean_squared_error(y_true.values, y_pred.values)))


@dataclass
class _ColumnGroups:
    numeric: List[str]
    categorical: List[str]
    datetime: List[str]
    boolean: List[str]
    other: List[str]


class _TSDFCore:
    """Base core desacoplada para evitar colisiones de super() con %autoreload 2."""

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

    def _infer_column_groups(self) -> _ColumnGroups:
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
        return _ColumnGroups(
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
                from IPython.display import display
                sty = (
                    view.style
                    .format(precision=precision)
                    .set_properties(**{"text-align": "left"})
                    .set_table_styles([{"selector": "th", "props": [("text-align", "left")]}])
                )
                try:
                    pd.set_option("display.max_colwidth", max_colwidth)
                except Exception:
                    pass
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


class TimeSeriesForecaster(_TSDFCore):
    """
    sklearn-like forecaster (anti-leakage temporal)
    - Periodograma sobre train
    - Modelos: ETS, SARIMA, LSTM, CNN1D, PROPHET, XGBTS
    - Tune / Compete / Fit / Predict / Forecast / Plot (Plotly Dark)
    """

    def _params_to_str(self, p: Any) -> str:
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
            return json.dumps(p, ensure_ascii=False, sort_keys=True, default=_json_default)
        except Exception:
            return str(p)

    def __init__(self, df: pd.DataFrame, date_col: str, target_col: str, freq: str = "MS"):
        _TSDFCore.__init__(self, df=df, target=target_col)

        self.date_col = date_col
        self.target_col = target_col
        self.freq = freq

        self.y_: Optional[pd.Series] = None
        self.y_train_: Optional[pd.Series] = None
        self.y_test_: Optional[pd.Series] = None
        self.test_size_: Optional[int] = None

        self.period_candidates_: Optional[List[int]] = None

        self.models_: Dict[str, BaseTSModel] = {}
        self.leaderboard_: Optional[pd.DataFrame] = None
        self.best_model_name_: Optional[str] = None
        self.best_score_: Optional[float] = None
        self.best_params_: Optional[Dict[str, Any]] = None

        self.fitted_model_: Optional[BaseTSModel] = None

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
    # 1) Data -> Serie
    # -----------------------------
    def to_series(self, agg: str = "sum") -> "TimeSeriesForecaster":
        d = self.df.copy()
        d[self.date_col] = pd.to_datetime(d[self.date_col])
        d = d.sort_values(self.date_col)

        if agg is not None:
            if agg == "sum":
                d = d.groupby(self.date_col, as_index=False)[self.target_col].sum()
            elif agg == "mean":
                d = d.groupby(self.date_col, as_index=False)[self.target_col].mean()
            else:
                raise ValueError("agg soportado: 'sum' | 'mean'")

        idx = pd.DatetimeIndex(d[self.date_col])
        y = pd.Series(d[self.target_col].values, index=idx, name=self.target_col).astype(float)
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
        self.y_train_ = self.y_.iloc[: -self.test_size_]
        self.y_test_ = self.y_.iloc[-self.test_size_ :]
        return self

    # -----------------------------
    # 3) Periodograma (solo train)
    # -----------------------------
    def periodogram_candidates(
        self,
        best: int = 3,
        round_int: bool = True,
        min_period: int = 2,
        max_period: Optional[int] = None,
    ) -> List[int]:
        if self.y_train_ is None:
            raise RuntimeError("Ejecuta prepare() primero (split temporal).")

        p = PeriodogramHelper(self.y_train_)
        periods = p.best_periods(best=best)
        periods = periods[np.isfinite(periods)]
        periods = periods[periods > 0]

        cand = [int(np.round(x)) for x in periods] if round_int else [float(x) for x in periods]
        cand = [int(x) for x in cand if int(x) >= min_period]
        if max_period is not None:
            cand = [int(x) for x in cand if int(x) <= int(max_period)]

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
        include_xgb: bool = True,
    ) -> "TimeSeriesForecaster":
        self.models_.clear()
        self.models_["ETS"] = ETSModel(trend="add", seasonal=None, seasonal_periods=None)
        self.models_["SARIMA"] = SARIMAModel(order=(1, 0, 0), seasonal_order=(0, 0, 0, 0))

        if include_lstm:
            if _HAS_TF:
                self.models_["LSTM"] = LSTMModel(lookback=12, epochs=20, batch_size=16, verbose=0)
            else:
                warnings.warn("TensorFlow no instalado; se omite LSTM.")

        if include_cnn:
            if _HAS_TF:
                self.models_["CNN1D"] = CNN1DModel(lookback=12, epochs=20, batch_size=16, verbose=0)
            else:
                warnings.warn("TensorFlow no instalado; se omite CNN1D.")

        if include_prophet:
            if _HAS_PROPHET:
                self.models_["PROPHET"] = ProphetModel()
            else:
                warnings.warn("Prophet no instalado; se omite PROPHET.")

        if include_xgb:
            if _HAS_XGB:
                self.models_["XGBTS"] = XGBTSModel()
            else:
                warnings.warn("XGBoost no instalado; se omite XGBTS.")

        return self

    # -----------------------------
    # 5) Tune
    # -----------------------------
    def tune(
        self,
        model_name: str,
        metric: str = "rmse",
        ar: List[int] = [0, 1, 2],
        seas: List[int] = [0, 1],
        S_candidates: Optional[List[int]] = None,
        n_jobs: int = -1,
        p_candidates: Optional[List[int]] = None,
        dl_epochs: int = 20,
        **kwargs,
    ) -> "TimeSeriesForecaster":
        if self.y_train_ is None or self.y_test_ is None:
            raise RuntimeError("Ejecuta prepare() antes de tune().")
        if model_name not in self.models_:
            raise ValueError(f"Modelo '{model_name}' no existe. Usa add_default_models() primero.")
        if metric != "rmse":
            raise ValueError("Por ahora solo metric='rmse'.")

        name = model_name.upper()

        # --- ETS ---
        if name == "ETS":
            seasonal_periods_candidates = kwargs.get("seasonal_periods_candidates", None)
            if seasonal_periods_candidates is None:
                if S_candidates is not None:
                    seasonal_periods_candidates = list(S_candidates)
                elif self.period_candidates_:
                    seasonal_periods_candidates = list(self.period_candidates_)
                else:
                    seasonal_periods_candidates = [None]

            trend = kwargs.get("trend", "add")
            seasonal = kwargs.get("seasonal", "add")
            paso = kwargs.get("paso", 0.1)

            alpha_candidates = kwargs.get("alpha_candidates") or np.round(np.arange(0, 1 + paso, paso), 10).tolist()
            beta_candidates = kwargs.get("beta_candidates") or np.round(np.arange(0, 1 + paso, paso), 10).tolist()
            gamma_candidates = kwargs.get("gamma_candidates") or np.round(np.arange(0, 1 + paso, paso), 10).tolist()

            if trend is None:
                beta_candidates = [None]
            if seasonal is None:
                gamma_candidates = [None]

            n_train = len(self.y_train_)
            if seasonal is not None:
                seasonal_periods_candidates = [
                    sp for sp in seasonal_periods_candidates
                    if sp is not None and int(sp) >= 2 and (2 * int(sp) <= n_train)
                ]
                if not seasonal_periods_candidates:
                    seasonal = None
                    seasonal_periods_candidates = [None]
                    gamma_candidates = [None]
            else:
                seasonal_periods_candidates = [None]
                gamma_candidates = [None]

            grid = [
                (sp, a, b, g)
                for sp in seasonal_periods_candidates
                for a in alpha_candidates
                for b in beta_candidates
                for g in gamma_candidates
            ]

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
                        optimized=False,
                    ).fit(self.y_train_)
                    yhat = m.predict(steps=len(self.y_test_))
                    sc = rmse(self.y_test_, yhat)
                    params = {
                        "trend": trend,
                        "seasonal": seasonal,
                        "seasonal_periods": (int(sp) if sp is not None else None),
                        "alpha": self._effective(m, "alpha_", "alpha"),
                        "beta": self._effective(m, "beta_", "beta"),
                        "gamma": self._effective(m, "gamma_", "gamma"),
                    }
                    return sc, params
                except Exception:
                    return float("inf"), None

            try:
                from joblib import Parallel, delayed
                results = Parallel(n_jobs=n_jobs)(delayed(_eval_ets)(t) for t in grid) if n_jobs != 1 else [_eval_ets(t) for t in grid]
            except Exception:
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
                    optimized=True,
                )
                return self

            self.models_["ETS"] = ETSModel(
                trend=best_params["trend"],
                seasonal=best_params["seasonal"],
                seasonal_periods=best_params["seasonal_periods"],
                alpha=best_params["alpha"],
                beta=best_params["beta"],
                gamma=best_params["gamma"],
                optimized=False,
            )
            return self

        # --- LSTM ---
        if name == "LSTM":
            if p_candidates is None:
                if self.period_candidates_ is None:
                    raise RuntimeError("Corre periodogram_candidates() o pasa p_candidates manualmente.")
                p_candidates = self.period_candidates_

            best_score = float("inf")
            best_p = None
            for p in p_candidates:
                try:
                    m = LSTMModel(lookback=int(p), epochs=int(dl_epochs), batch_size=16, verbose=0).fit(self.y_train_)
                    yhat = m.predict(steps=len(self.y_test_))
                    sc = rmse(self.y_test_, yhat)
                    if sc < best_score:
                        best_score = sc
                        best_p = int(p)
                except Exception:
                    continue

            if best_p is None:
                raise RuntimeError("LSTM tune: no se encontró un lookback válido.")

            self.models_["LSTM"] = LSTMModel(lookback=best_p, epochs=int(dl_epochs), batch_size=16, verbose=0)
            return self

        # --- CNN1D ---
        if name == "CNN1D":
            if p_candidates is None:
                if self.period_candidates_ is None:
                    raise RuntimeError("Corre periodogram_candidates() o pasa p_candidates manualmente.")
                p_candidates = self.period_candidates_

            best_score = float("inf")
            best_p = None
            for p in p_candidates:
                try:
                    m = CNN1DModel(lookback=int(p), epochs=int(dl_epochs), batch_size=16, verbose=0).fit(self.y_train_)
                    yhat = m.predict(steps=len(self.y_test_))
                    sc = rmse(self.y_test_, yhat)
                    if sc < best_score:
                        best_score = sc
                        best_p = int(p)
                except Exception:
                    continue

            if best_p is None:
                raise RuntimeError("CNN1D tune: no se encontró un lookback válido.")

            self.models_["CNN1D"] = CNN1DModel(lookback=best_p, epochs=int(dl_epochs), batch_size=16, verbose=0)
            return self

        # --- SARIMA ---
        if name == "SARIMA":
            if S_candidates is None:
                if self.period_candidates_ is None:
                    raise RuntimeError("Corre periodogram_candidates() o pasa S_candidates manualmente.")
                S_candidates = self.period_candidates_

            max_p = int(kwargs.get("max_p", 2))
            max_d = int(kwargs.get("max_d", 2))
            max_q = int(kwargs.get("max_q", 2))
            max_P = int(kwargs.get("max_P", 1))
            max_D = int(kwargs.get("max_D", 1))
            max_Q = int(kwargs.get("max_Q", 1))

            search = str(kwargs.get("search", "grid")).lower()
            max_evals = int(kwargs.get("max_evals", 150))
            random_state = kwargs.get("random_state", None)

            param_grid = [
                ((p, d, q), (P, D, Q, int(S)))
                for S in S_candidates
                for p in range(max_p + 1)
                for d in range(max_d + 1)
                for q in range(max_q + 1)
                for P in range(max_P + 1)
                for D in range(max_D + 1)
                for Q in range(max_Q + 1)
            ]

            if search == "random" and len(param_grid) > max_evals:
                rng = np.random.default_rng(random_state)
                idx = rng.choice(len(param_grid), size=max_evals, replace=False)
                param_grid = [param_grid[i] for i in idx]

            def _eval_sarima(params):
                order, seas_order = params
                try:
                    m = SARIMAModel(order=order, seasonal_order=seas_order).fit(self.y_train_)
                    yhat = m.predict(steps=len(self.y_test_))
                    return rmse(self.y_test_, yhat), order, seas_order
                except Exception:
                    return float("inf"), None, None

            try:
                from joblib import Parallel, delayed
                results = Parallel(n_jobs=n_jobs, prefer="threads")(delayed(_eval_sarima)(p) for p in param_grid) if n_jobs != 1 else [_eval_sarima(p) for p in param_grid]
            except Exception:
                results = [_eval_sarima(p) for p in param_grid]

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

        # --- PROPHET ---
        if name == "PROPHET":
            if not _HAS_PROPHET:
                raise ImportError("Prophet no disponible.")

            logging.getLogger("cmdstanpy").setLevel(logging.WARNING)
            logging.getLogger("prophet").setLevel(logging.WARNING)

            search = str(kwargs.get("search", "grid")).lower()
            max_evals = int(kwargs.get("max_evals", 60))

            cps_grid = kwargs.get("changepoint_prior_scale_grid", [0.01, 0.05, 0.1, 0.2])
            sps_grid = kwargs.get("seasonality_prior_scale_grid", [1.0, 5.0, 10.0, 20.0])
            mode_grid = kwargs.get("seasonality_mode_grid", ["additive", "multiplicative"])

            yearly = kwargs.get("yearly_seasonality", "auto")
            weekly = kwargs.get("weekly_seasonality", "auto")
            daily = kwargs.get("daily_seasonality", "auto")

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
                        best_params = {
                            "changepoint_prior_scale": float(cps),
                            "seasonality_prior_scale": float(sps),
                            "seasonality_mode": mode,
                            "yearly_seasonality": yearly,
                            "weekly_seasonality": weekly,
                            "daily_seasonality": daily,
                        }
                except Exception:
                    continue

            if best_params is None:
                raise RuntimeError("PROPHET tune: no se encontró configuración válida.")

            self.models_["PROPHET"] = ProphetModel(**best_params)
            return self

        # --- XGBTS ---
        if name == "XGBTS":
            if not _HAS_XGB:
                raise ImportError("xgboost no disponible.")

            search = str(kwargs.get("search", "random")).lower()
            max_evals = int(kwargs.get("max_evals", 60))
            rng = np.random.default_rng(kwargs.get("random_state", None))

            lag_sets = kwargs.get("lag_sets", [[1, 2, 3, 6, 12], [1, 2, 3, 12], [1, 2, 3, 4, 5, 6, 12]])
            roll_sets = kwargs.get("roll_sets", [[3, 6, 12], [3, 6], [6, 12]])
            depth_grid = kwargs.get("max_depth_grid", [3, 4, 5, 6])
            lr_grid = kwargs.get("learning_rate_grid", [0.02, 0.05, 0.1])
            n_est_grid = kwargs.get("n_estimators_grid", [300, 500, 800])
            subs_grid = kwargs.get("subsample_grid", [0.8, 0.9, 1.0])
            col_grid = kwargs.get("colsample_bytree_grid", [0.8, 0.9, 1.0])

            combos = [
                (lags, rolls, md, lr, ne, ss, cc)
                for lags in lag_sets
                for rolls in roll_sets
                for md in depth_grid
                for lr in lr_grid
                for ne in n_est_grid
                for ss in subs_grid
                for cc in col_grid
            ]

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

        return self

    # -----------------------------
    # 6) Compete
    # -----------------------------
    def compete(self, metric: str = "rmse", verbose: bool = True) -> pd.DataFrame:
        if self.y_train_ is None or self.y_test_ is None:
            raise RuntimeError("Ejecuta prepare() antes de compete().")
        if metric != "rmse":
            raise ValueError("Por ahora solo metric='rmse'.")

        rows = []
        for name, model in self.models_.items():
            try:
                m_fit = _clone_model(model).fit(self.y_train_)
                yhat = m_fit.predict(steps=len(self.y_test_))
                sc = rmse(self.y_test_, yhat)
                rows.append({
                    "model": name,
                    "rmse": float(sc),
                    "params": self._params_of_model(model, fitted=m_fit),
                })
            except Exception:
                rows.append({
                    "model": name,
                    "rmse": np.nan,
                    "params": None,
                })

        lb = pd.DataFrame(rows).sort_values("rmse", ascending=True).reset_index(drop=True)
        self.leaderboard_ = lb

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

        lb["params_str"] = lb["params"].apply(self._params_to_str)
        show_lb = lb[["model", "rmse", "params_str"]].copy()

        if verbose:
            self.pretty_table(
                show_lb,
                title="Leaderboard (RMSE) — menor es mejor",
                precision=4,
                max_colwidth=500,
            )

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
        m = fitted if fitted is not None else spec

        if isinstance(spec, SARIMAModel):
            return {"order": spec.order, "seasonal_order": spec.seasonal_order}
        if isinstance(spec, LSTMModel):
            return {"lookback": spec.lookback, "epochs": spec.epochs, "batch_size": spec.batch_size}
        if isinstance(spec, CNN1DModel):
            return {
                "lookback": spec.lookback,
                "epochs": spec.epochs,
                "batch_size": spec.batch_size,
                "filters": spec.filters,
                "kernel_size": spec.kernel_size,
                "l2": spec.l2,
            }
        if isinstance(spec, ETSModel):
            alpha = getattr(m, "alpha_", None)
            beta = getattr(m, "beta_", None)
            gamma = getattr(m, "gamma_", None)
            if alpha is None:
                alpha = getattr(spec, "alpha", None)
            if beta is None:
                beta = getattr(spec, "beta", None)
            if gamma is None:
                gamma = getattr(spec, "gamma", None)
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
        return {
            "min": float(np.min(yv)),
            "q1": float(np.quantile(yv, 0.25)),
            "median": float(np.quantile(yv, 0.50)),
            "q3": float(np.quantile(yv, 0.75)),
            "max": float(np.max(yv)),
            "iqr": float(np.quantile(yv, 0.75) - np.quantile(yv, 0.25)),
        }

    def best_params(self) -> Dict[str, Any]:
        if self.best_model_name_ is None:
            raise RuntimeError("No hay best model. Corre compete() primero.")
        out = {"model": self.best_model_name_}
        if isinstance(self.best_params_, dict):
            out.update(self.best_params_)
        return out

    # -----------------------------
    # 7) Fit final & Predict
    # -----------------------------
    def fit(self, model: str = "best", refit_on: str = "train") -> "TimeSeriesForecaster":
        if self.y_train_ is None:
            raise RuntimeError("Ejecuta prepare() antes de fit().")

        model_name = self.best_model_name_ if model == "best" else model
        if model_name is None:
            raise RuntimeError("No hay best model. Corre compete() primero.")
        if model_name not in self.models_:
            raise ValueError(f"Modelo '{model_name}' no existe.")
        if refit_on not in ["train", "train_test_full"]:
            raise ValueError("refit_on soportado: 'train' | 'train_test_full'")

        yfit = self.y_train_ if refit_on == "train" else self.y_
        if yfit is None:
            raise RuntimeError("Serie no disponible para fit.")

        self.fitted_model_ = _clone_model(self.models_[model_name]).fit(yfit)
        self.best_model_name_ = model_name
        return self

    def predict(self, steps: int) -> pd.Series:
        if self.fitted_model_ is None:
            raise RuntimeError("Ejecuta fit() antes de predict().")
        return self.fitted_model_.predict(steps=int(steps))

    def forecast_df(
        self,
        steps: int,
        date_col: Optional[str] = None,
        pred_col: Optional[str] = None,
        include_model: bool = True,
    ) -> pd.DataFrame:
        yhat = self.predict(steps=int(steps))
        out_date_col = date_col if date_col is not None else self.date_col
        out_pred_col = pred_col if pred_col is not None else f"pred_{self.target_col}"

        df_pred = pd.DataFrame({
            out_date_col: yhat.index,
            out_pred_col: yhat.values,
        })
        if include_model:
            df_pred["model"] = self.best_model_name_
        return df_pred

    def evaluate_test(self) -> Dict[str, Any]:
        if self.y_test_ is None:
            raise RuntimeError("Ejecuta prepare() primero.")
        if self.fitted_model_ is None:
            raise RuntimeError("Ejecuta fit() primero.")

        yhat = self.fitted_model_.predict(steps=len(self.y_test_))
        score = float(rmse(self.y_test_, yhat))

        y_all = pd.concat([self.y_train_, self.y_test_]) if self.y_train_ is not None else self.y_test_
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
    # 10) Plot forecast (Plotly Dark)
    # -----------------------------
    def plot_forecast(
            self,
            steps: int = 30,
            show_test_pred: bool = True,
            height: int = 520,
            width: Optional[int] = None,
            show: bool = True,
            return_fig: bool = False,
            **kwargs,
        ):
        """
        Plotea de forma interactiva con tema plotly_dark:
        - Train (real)
        - Test (real)
        - (opcional) Predicción sobre test (fit en train, forecast len(test))
        - Forecast futuro (requiere fit() previo)
        """
        if self.y_train_ is None or self.y_test_ is None:
            raise RuntimeError("Ejecuta prepare() antes de plot_forecast().")
        if not _HAS_PLOTLY or go is None:
            raise ImportError("Plotly no disponible. Instala 'plotly' para usar plot_forecast().")

        fig = go.Figure()

        # 1) Real: train y test
        fig.add_trace(
            go.Scatter(
                x=self.y_train_.index,
                y=self.y_train_.values,
                mode="lines",
                name="Train",
                line=dict(width=2, color="#38bdf8"),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=self.y_test_.index,
                y=self.y_test_.values,
                mode="lines",
                name="Test real",
                line=dict(width=2, color="#34d399"),
            )
        )

        # 2) Pred test (backtest visual)
        if show_test_pred:
            if self.best_model_name_ is None:
                raise RuntimeError("Corre compete() para definir best_model antes de show_test_pred.")
            m_test = _clone_model(self.models_[self.best_model_name_]).fit(self.y_train_)
            yhat_test = m_test.predict(steps=len(self.y_test_))
            fig.add_trace(
                go.Scatter(
                    x=yhat_test.index,
                    y=yhat_test.values,
                    mode="lines",
                    name="Pred test",
                    line=dict(width=2, dash="dash", color="#fb923c"),
                )
            )

        # 3) Forecast futuro (usa fitted_model_)
        if self.fitted_model_ is not None:
            yhat_future = self.fitted_model_.predict(steps=int(steps))
            fig.add_trace(
                go.Scatter(
                    x=yhat_future.index,
                    y=yhat_future.values,
                    mode="lines+markers",
                    name=f"Forecast futuro (+{steps})",
                    line=dict(width=2.5, color="#c084fc"),
                    marker=dict(size=5),
                )
            )
            sep_date = self.y_.index[-1]
        else:
            sep_date = self.y_train_.index[-1]

        fig.add_shape(
            type="line",
            x0=sep_date,
            x1=sep_date,
            y0=0,
            y1=1,
            yref="paper",
            line=dict(color="rgba(255, 255, 255, 0.45)", width=1.5, dash="dash"),
        )

        layout_kwargs: Dict[str, Any] = dict(
            title=f"Forecast — best={self.best_model_name_}",
            template="plotly_dark",
            hovermode="x unified",
            xaxis_title="Fecha",
            yaxis_title=self.target_col,
            height=height,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=50, r=30, t=70, b=50),
        )
        if width is not None:
            layout_kwargs["width"] = width

        fig.update_layout(**layout_kwargs)
        if show:
            fig.show()
        if return_fig:
            return fig
        return None

    # ------------------------------------------------------------
    # Missing dates & smoothing helpers
    # ------------------------------------------------------------
    def _missing_report_series(self, series: pd.Series, freq: str = None) -> pd.DataFrame:
        if not isinstance(series.index, pd.DatetimeIndex):
            raise ValueError("series debe tener DatetimeIndex. Use to_series() primero.")

        freq_use = freq or getattr(self, "freq", None)
        if freq_use is None:
            raise ValueError("No hay freq definida. Pase freq=... o defina self.freq.")

        s = series.sort_index()
        full_idx = pd.date_range(start=s.index.min(), end=s.index.max(), freq=freq_use)
        missing = full_idx.difference(s.index)

        return pd.DataFrame([{
            "freq": freq_use,
            "start": s.index.min(),
            "end": s.index.max(),
            "n_expected": int(len(full_idx)),
            "n_present": int(s.index.nunique()),
            "n_missing": int(len(missing)),
            "missing_dates": list(missing),
        }])

    def _add_missing_dates_series(self, series: pd.Series, freq: str = None) -> pd.Series:
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

    def _impute_smoothed_series(
        self,
        series: pd.Series,
        *,
        smooth: str = "rolling_mean",
        window: int = 5,
        center: bool = True,
        impute: str = "time",
        limit: int = None,
        keep_original_name: bool = True,
    ) -> pd.Series:
        if not isinstance(series.index, pd.DatetimeIndex):
            raise ValueError("series debe tener DatetimeIndex. Use to_series() primero.")

        s = series.sort_index().copy()

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

    def missing_report(self, freq: str = None) -> pd.DataFrame:
        if self.y_ is None:
            raise RuntimeError("Ejecuta to_series() primero.")
        return self._missing_report_series(self.y_, freq=freq)

    def add_missing_dates(self, freq: str = None) -> "TimeSeriesForecaster":
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
        limit: int = None,
    ) -> "TimeSeriesForecaster":
        if self.y_ is None:
            raise RuntimeError("Ejecuta to_series() primero.")
        self.y_ = self._impute_smoothed_series(
            self.y_,
            smooth=smooth,
            window=window,
            center=center,
            impute=impute,
            limit=limit,
            keep_original_name=True,
        )
        return self

    # -----------------------------
    # 11) One-shot pipeline
    # -----------------------------
    def run_auto(
        self,
        *,
        agg: str = "sum",
        check_missing: bool = True,
        add_missing: bool = False,
        impute_missing: bool = False,
        impute_kwargs: Optional[Dict[str, Any]] = None,
        train_size: float = 0.95,
        test_size: Optional[int] = None,
        use_periodogram: bool = True,
        best_periods: int = 3,
        min_period: int = 2,
        max_period: Optional[int] = None,
        include_lstm: bool = True,
        include_cnn: bool = True,
        include_prophet: bool = True,
        include_xgb: bool = True,
        tune_models: bool = True,
        p_candidates: Optional[List[int]] = None,
        S_candidates: Optional[List[int]] = None,
        dl_epochs: int = 20,
        n_jobs: int = -1,
        tune_prophet: bool = True,
        tune_xgb: bool = True,
        prophet_max_evals: int = 40,
        xgb_max_evals: int = 60,
        metric: str = "rmse",
        verbose: bool = True,
        refit_on: str = "train_test_full",
        forecast_steps: Optional[int] = None,
        return_forecast_df: bool = True,
    ) -> Dict[str, Any]:
        if impute_kwargs is None:
            impute_kwargs = dict(smooth="rolling_mean", window=5, center=True, impute="time", limit=None)

        out: Dict[str, Any] = {}

        self.to_series(agg=agg)
        if check_missing:
            out["missing_report"] = self.missing_report()
        if add_missing:
            self.add_missing_dates()
        if impute_missing:
            self.impute_smoothed(**impute_kwargs)

        self.prepare(test_size=test_size, train_size=train_size)

        if use_periodogram:
            self.periodogram_candidates(
                best=best_periods,
                round_int=True,
                min_period=min_period,
                max_period=max_period,
            )

        self.add_default_models(
            include_lstm=include_lstm,
            include_cnn=include_cnn,
            include_prophet=include_prophet,
            include_xgb=include_xgb,
        )

        if tune_models:
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

        lb = self.compete(metric=metric, verbose=verbose)
        out["leaderboard"] = lb
        out["best_model"] = self.best_model_name_
        out["best_score"] = self.best_score_

        self.fit(model="best", refit_on=refit_on)

        if forecast_steps is not None and int(forecast_steps) > 0:
            if return_forecast_df:
                out["forecast"] = self.forecast_df(steps=int(forecast_steps))
            else:
                out["forecast"] = self.predict(steps=int(forecast_steps))

        return out


# Alias para compatibilidad hacia atrás
TimeSeriesTournament = TimeSeriesForecaster