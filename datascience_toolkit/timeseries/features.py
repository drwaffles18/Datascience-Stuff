"""
features.py
===========
Feature engineering for time-indexed data: Lags, rolling window aggregates,
and cyclical calendar encoding.
"""

from typing import List, Optional
import numpy as np
import pandas as pd


class TimeSeriesFeatureExtractor:
    """Generates temporal, cyclical, lag, and rolling statistics features."""

    @staticmethod
    def add_calendar_features(
        df: pd.DataFrame,
        date_col: Optional[str] = None,
        cyclical_encoding: bool = True
    ) -> pd.DataFrame:
        """Decomposes datetime index or column into calendar attributes with cyclical sin/cos transformations."""
        out = df.copy()

        if date_col:
            dt_series = pd.to_datetime(out[date_col])
        elif isinstance(out.index, pd.DatetimeIndex):
            dt_series = out.index.to_series()
        else:
            raise TypeError("A date_col must be specified or DataFrame must have a DatetimeIndex.")

        out["year"] = dt_series.dt.year
        out["quarter"] = dt_series.dt.quarter
        out["month"] = dt_series.dt.month
        out["day"] = dt_series.dt.day
        out["dayofweek"] = dt_series.dt.dayofweek
        out["is_weekend"] = dt_series.dt.dayofweek.isin([5, 6]).astype(int)

        if cyclical_encoding:
            # Month cyclical encoding (1 to 12)
            out["month_sin"] = np.round(np.sin(2 * np.pi * dt_series.dt.month / 12), 4)
            out["month_cos"] = np.round(np.cos(2 * np.pi * dt_series.dt.month / 12), 4)

            # Day of week cyclical encoding (0 to 6)
            out["dow_sin"] = np.round(np.sin(2 * np.pi * dt_series.dt.dayofweek / 7), 4)
            out["dow_cos"] = np.round(np.cos(2 * np.pi * dt_series.dt.dayofweek / 7), 4)

        return out

    @staticmethod
    def add_lag_features(
        df: pd.DataFrame,
        target_col: str,
        lags: List[int] = [1, 2, 7, 14]
    ) -> pd.DataFrame:
        """Constructs temporal autoregressive lags for a target feature."""
        out = df.copy()
        for lag in lags:
            out[f"{target_col}_lag_{lag}"] = out[target_col].shift(lag)
        return out

    @staticmethod
    def add_rolling_features(
        df: pd.DataFrame,
        target_col: str,
        windows: List[int] = [7, 14, 30],
        aggregates: List[str] = ["mean", "std", "min", "max"]
    ) -> pd.DataFrame:
        """Constructs rolling window aggregates shifted by 1 period to prevent data leakage."""
        out = df.copy()
        shifted_target = out[target_col].shift(1)

        for w in windows:
            roll = shifted_target.rolling(window=w)
            if "mean" in aggregates:
                out[f"{target_col}_roll_mean_{w}"] = roll.mean()
            if "std" in aggregates:
                out[f"{target_col}_roll_std_{w}"] = roll.std()
            if "min" in aggregates:
                out[f"{target_col}_roll_min_{w}"] = roll.min()
            if "max" in aggregates:
                out[f"{target_col}_roll_max_{w}"] = roll.max()

        return out