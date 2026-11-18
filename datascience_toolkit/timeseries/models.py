"""
models.py
=========
Time series forecasting model wrappers (ETS, SARIMA, LSTM, CNN1D, Prophet, XGBTS)
with strict anti-leakage scaling and unified fit/predict interfaces.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from .features import _future_index

try:
    from statsmodels.tsa.api import ExponentialSmoothing
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    _HAS_STATSMODELS = True
except Exception:
    ExponentialSmoothing = None
    SARIMAX = None
    _HAS_STATSMODELS = False

try:
    import xgboost as xgb
    _HAS_XGB = True
except Exception:
    xgb = None
    _HAS_XGB = False

try:
    import tensorflow as tf
    from tensorflow.keras import regularizers
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Input, Dense, LSTM, Conv1D, GlobalAveragePooling1D
    _HAS_TF = True
except Exception:
    tf = None
    _HAS_TF = False

try:
    from prophet import Prophet
    _HAS_PROPHET = True
except Exception:
    Prophet = None
    _HAS_PROPHET = False


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


class ETSModel(BaseTSModel):
    def __init__(
        self,
        trend: Optional[str] = "add",
        seasonal: Optional[str] = None,
        seasonal_periods: Optional[int] = None,
        alpha: Optional[float] = None,
        beta: Optional[float] = None,
        gamma: Optional[float] = None,
        optimized: bool = True,
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

        self.alpha_ = alpha
        self.beta_ = beta
        self.gamma_ = gamma

    def fit(self, y_train: pd.Series, **kwargs) -> "ETSModel":
        if not _HAS_STATSMODELS:
            raise ImportError("statsmodels no disponible. Instala 'statsmodels' para usar ETSModel.")

        self._y_train = y_train
        model = ExponentialSmoothing(
            y_train,
            trend=self.trend,
            seasonal=self.seasonal,
            seasonal_periods=self.seasonal_periods,
        )

        fit_kwargs = {"optimized": self.optimized}
        if not self.optimized:
            fit_kwargs["smoothing_level"] = self.alpha
            fit_kwargs["smoothing_trend"] = self.beta
            fit_kwargs["smoothing_seasonal"] = self.gamma

        self._fit = model.fit(**fit_kwargs)

        lvl = getattr(self._fit, "smoothing_level", None)
        tr = getattr(self._fit, "smoothing_trend", None)
        sea = getattr(self._fit, "smoothing_seasonal", None)

        self.alpha_ = self.alpha if (lvl is None) else float(lvl)
        self.beta_ = self.beta if (tr is None) else float(tr)
        self.gamma_ = self.gamma if (sea is None) else float(sea)
        return self

    def predict(self, steps: int, **kwargs) -> pd.Series:
        if self._fit is None:
            raise RuntimeError("ETSModel: call fit() first.")
        yhat = np.asarray(self._fit.forecast(steps), dtype=float)
        return pd.Series(yhat, index=_future_index(self._y_train.index, steps))


class SARIMAModel(BaseTSModel):
    def __init__(self, order=(1, 0, 0), seasonal_order=(0, 0, 0, 0)):
        self.order = tuple(order)
        self.seasonal_order = tuple(seasonal_order)
        self._fit = None
        self._y_train = None

    def fit(self, y_train: pd.Series, **kwargs) -> "SARIMAModel":
        if not _HAS_STATSMODELS:
            raise ImportError("statsmodels no disponible. Instala 'statsmodels' para usar SARIMAModel.")

        self._y_train = y_train
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = SARIMAX(
                y_train,
                order=self.order,
                seasonal_order=self.seasonal_order,
                enforce_stationarity=False,
                enforce_invertibility=False,
            )
            self._fit = model.fit(disp=False)
        return self

    def predict(self, steps: int, **kwargs) -> pd.Series:
        if self._fit is None:
            raise RuntimeError("SARIMAModel: call fit() first.")
        yhat = np.asarray(self._fit.forecast(steps), dtype=float)
        return pd.Series(yhat, index=_future_index(self._y_train.index, steps))


class LSTMModel(BaseTSModel):
    def __init__(
        self,
        lookback: int = 12,
        epochs: int = 30,
        batch_size: int = 16,
        verbose: int = 0,
        validation_split: float = 0.2,
        early_stopping: bool = True,
        patience: int = 10,
        restore_best_weights: bool = True,
        reduce_lr: bool = True,
        lr_monitor: str = "val_loss",
        lr_factor: float = 0.5,
        lr_patience: int = 5,
        lr_min: float = 1e-6,
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

    def _fit_scaler(self, x: np.ndarray):
        self._scaler_min = float(np.min(x))
        self._scaler_max = float(np.max(x))
        if (self._scaler_max - self._scaler_min) == 0:
            self._scaler_max = self._scaler_min + 1.0

    def _scale(self, x: np.ndarray) -> np.ndarray:
        if self._scaler_min is None or self._scaler_max is None:
            self._fit_scaler(x)
        return (x - self._scaler_min) / (self._scaler_max - self._scaler_min)

    def _inv_scale(self, x: np.ndarray) -> np.ndarray:
        return x * (self._scaler_max - self._scaler_min) + self._scaler_min

    def _make_windows(self, arr: np.ndarray):
        X, y = [], []
        for i in range(len(arr) - self.lookback):
            X.append(arr[i : i + self.lookback])
            y.append(arr[i + self.lookback])
        X = np.array(X).reshape((-1, self.lookback, 1))
        y = np.array(y)
        return X, y

    def fit(self, y_train: pd.Series, **kwargs):
        if not _HAS_TF:
            raise ImportError("TensorFlow no disponible. Instala tensorflow para usar LSTM.")

        if kwargs.get("lookback") is not None:
            self.lookback = int(kwargs["lookback"])
        if kwargs.get("epochs") is not None:
            self.epochs = int(kwargs["epochs"])
        if kwargs.get("batch_size") is not None:
            self.batch_size = int(kwargs["batch_size"])

        self._y_train = y_train
        arr = y_train.astype(float).values

        n = len(arr)
        split = float(self.validation_split) if self.validation_split is not None else 0.0
        if split < 0 or split >= 1:
            split = 0.0
        cut = max(int(np.floor(n * (1 - split))), self.lookback + 2)
        self._fit_scaler(arr[:cut])

        arr_scaled = self._scale(arr)
        X, y = self._make_windows(arr_scaled)

        reg = regularizers.l2(self.l2) if self.l2 > 0 else None
        model = Sequential([
            Input(shape=(self.lookback, 1)),
            LSTM(
                32,
                dropout=self.dropout,
                recurrent_dropout=self.recurrent_dropout,
                kernel_regularizer=reg,
                recurrent_regularizer=reg,
            ),
            Dense(1, kernel_regularizer=reg),
        ])
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
            X, y,
            epochs=self.epochs,
            batch_size=self.batch_size,
            verbose=self.verbose,
            validation_split=self.validation_split,
            callbacks=callbacks,
            shuffle=False,
        )
        self._model = model
        return self

    def predict(self, steps: int, **kwargs) -> pd.Series:
        if self._model is None:
            raise RuntimeError("LSTMModel: call fit() first.")

        history = self._scale(self._y_train.astype(float).values.copy())
        preds = []
        for _ in range(int(steps)):
            x_input = history[-self.lookback :].reshape((1, self.lookback, 1))
            yhat = float(self._model.predict(x_input, verbose=0)[0][0])
            preds.append(yhat)
            history = np.append(history, yhat)

        preds = self._inv_scale(np.array(preds))
        return pd.Series(preds, index=_future_index(self._y_train.index, int(steps)))


class CNN1DModel(BaseTSModel):
    def __init__(
        self,
        lookback: int = 12,
        epochs: int = 30,
        batch_size: int = 16,
        verbose: int = 0,
        validation_split: float = 0.2,
        early_stopping: bool = True,
        patience: int = 10,
        restore_best_weights: bool = True,
        reduce_lr: bool = True,
        lr_monitor: str = "val_loss",
        lr_factor: float = 0.5,
        lr_patience: int = 5,
        lr_min: float = 1e-6,
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

    def _fit_scaler(self, x: np.ndarray):
        self._scaler_min = float(np.min(x))
        self._scaler_max = float(np.max(x))
        if (self._scaler_max - self._scaler_min) == 0:
            self._scaler_max = self._scaler_min + 1.0

    def _scale(self, x: np.ndarray) -> np.ndarray:
        if self._scaler_min is None or self._scaler_max is None:
            self._fit_scaler(x)
        return (x - self._scaler_min) / (self._scaler_max - self._scaler_min)

    def _inv_scale(self, x: np.ndarray) -> np.ndarray:
        return x * (self._scaler_max - self._scaler_min) + self._scaler_min

    def _make_windows(self, arr: np.ndarray):
        X, y = [], []
        for i in range(len(arr) - self.lookback):
            X.append(arr[i : i + self.lookback])
            y.append(arr[i + self.lookback])
        return np.array(X).reshape((-1, self.lookback, 1)), np.array(y)

    def fit(self, y_train: pd.Series, **kwargs) -> "CNN1DModel":
        if not _HAS_TF:
            raise ImportError("TensorFlow no disponible. Instala tensorflow para usar CNN1D.")

        for attr in ("lookback", "epochs", "batch_size", "filters", "kernel_size"):
            if kwargs.get(attr) is not None:
                setattr(self, attr, int(kwargs[attr]))

        self._y_train = y_train
        arr = y_train.astype(float).values

        n = len(arr)
        split = float(self.validation_split) if self.validation_split is not None else 0.0
        if split < 0 or split >= 1:
            split = 0.0
        cut = max(int(np.floor(n * (1 - split))), self.lookback + 2)
        self._fit_scaler(arr[:cut])

        arr_scaled = self._scale(arr)
        X, y = self._make_windows(arr_scaled)

        reg = regularizers.l2(self.l2) if self.l2 > 0 else None
        model = Sequential([
            Input(shape=(self.lookback, 1)),
            Conv1D(
                filters=self.filters,
                kernel_size=self.kernel_size,
                activation="relu",
                padding="causal",
                kernel_regularizer=reg,
            ),
            GlobalAveragePooling1D(),
            Dense(1, kernel_regularizer=reg),
        ])
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
            X, y,
            epochs=self.epochs,
            batch_size=self.batch_size,
            verbose=self.verbose,
            validation_split=self.validation_split,
            callbacks=callbacks,
            shuffle=False,
        )
        self._model = model
        return self

    def predict(self, steps: int, **kwargs) -> pd.Series:
        if self._model is None or self._y_train is None:
            raise RuntimeError("CNN1DModel: call fit() first.")

        history = self._scale(self._y_train.astype(float).values.copy())
        preds = []
        for _ in range(int(steps)):
            x_input = history[-self.lookback :].reshape((1, self.lookback, 1))
            yhat = float(self._model.predict(x_input, verbose=0)[0][0])
            preds.append(yhat)
            history = np.append(history, yhat)

        preds = self._inv_scale(np.array(preds))
        return pd.Series(preds, index=_future_index(self._y_train.index, int(steps)))


class ProphetModel(BaseTSModel):
    def __init__(
        self,
        seasonality_mode: str = "additive",
        yearly_seasonality: str | bool = "auto",
        weekly_seasonality: str | bool = "auto",
        daily_seasonality: str | bool = "auto",
        changepoint_prior_scale: float = 0.05,
        seasonality_prior_scale: float = 10.0,
        holidays_prior_scale: float = 10.0,
        interval_width: float = 0.8,
        add_country_holidays: str | None = None,
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
            "y": y_train.astype(float).values,
        })

        m = Prophet(
            seasonality_mode=self.seasonality_mode,
            yearly_seasonality=self.yearly_seasonality,
            weekly_seasonality=self.weekly_seasonality,
            daily_seasonality=self.daily_seasonality,
            changepoint_prior_scale=self.changepoint_prior_scale,
            seasonality_prior_scale=self.seasonality_prior_scale,
            holidays_prior_scale=self.holidays_prior_scale,
            interval_width=self.interval_width,
        )
        if self.add_country_holidays:
            m.add_country_holidays(country_name=self.add_country_holidays)

        m.fit(df)
        self._m = m
        return self

    def predict(self, steps: int, **kwargs) -> pd.Series:
        if self._m is None or self._y_train is None:
            raise RuntimeError("ProphetModel: call fit() first.")

        future_idx = _future_index(self._y_train.index, int(steps))
        future = pd.DataFrame({"ds": future_idx.to_pydatetime()})
        fcst = self._m.predict(future)
        yhat = fcst["yhat"].values.astype(float)
        return pd.Series(yhat, index=future_idx)


class XGBTSModel(BaseTSModel):
    def __init__(
        self,
        lags: list[int] = [1, 2, 3, 6, 12],
        rolling_windows: list[int] = [3, 6, 12],
        use_date_parts: bool = True,
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

        for L in self.lags:
            df[f"lag_{L}"] = df["y"].shift(L)

        for w in self.rolling_windows:
            df[f"roll_mean_{w}"] = df["y"].shift(1).rolling(w, min_periods=1).mean()
            df[f"roll_std_{w}"] = df["y"].shift(1).rolling(w, min_periods=1).std()

        if self.use_date_parts:
            idx = df.index
            df["month"] = idx.month
            df["quarter"] = idx.quarter
            df["year"] = idx.year
            df["dayofweek"] = idx.dayofweek

        return df

    def fit(self, y_train: pd.Series, **kwargs) -> "XGBTSModel":
        if not _HAS_XGB:
            raise ImportError("xgboost no disponible. Instala 'xgboost' para usar XGBTSModel.")

        self._y_train = y_train
        df = self._make_features(y_train).dropna()
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
            df_feat = self._make_features(hist).iloc[[-1]]
            X_next = df_feat.drop(columns=["y"])
            yhat = float(self._model.predict(X_next)[0])
            next_idx = _future_index(hist.index, 1)[0]
            hist = pd.concat([hist, pd.Series([yhat], index=pd.DatetimeIndex([next_idx]))])
            preds.append(yhat)

        future_idx = _future_index(self._y_train.index, steps)
        return pd.Series(np.array(preds, dtype=float), index=future_idx)


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
        return LSTMModel(
            lookback=m.lookback,
            epochs=m.epochs,
            batch_size=m.batch_size,
            verbose=m.verbose,
        )
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