"""
forecasting.py
==============
Temporal cross-validation and forecasting tournament avoiding data leakage.
"""

from typing import Dict, Any, Optional
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor


class TimeSeriesTournament:
    """Evaluates time series forecasting models across forward-chaining TimeSeriesSplit folds."""

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.default_models: Dict[str, Any] = {
            "Ridge": Ridge(),
            "RandomForest": RandomForestRegressor(n_estimators=100, random_state=random_state),
            "GradientBoosting": GradientBoostingRegressor(random_state=random_state)
        }
        self.leaderboard_: Optional[pd.DataFrame] = None

    def run(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        n_splits: int = 5,
        models: Optional[Dict[str, Any]] = None
    ) -> pd.DataFrame:
        """Executes forward-chaining validation and calculates RMSE, MAE, MAPE, and WAPE."""
        # Drop initial rows with NaNs resulting from lag generation
        valid_idx = X.dropna().index.intersection(y.dropna().index)
        X_clean = X.loc[valid_idx]
        y_clean = y.loc[valid_idx]

        candidate_models = models or self.default_models
        tscv = TimeSeriesSplit(n_splits=n_splits)

        results = []
        for name, model in candidate_models.items():
            fold_rmse, fold_mae, fold_mape, fold_wape = [], [], [], []

            for train_idx, test_idx in tscv.split(X_clean):
                X_train, X_test = X_clean.iloc[train_idx], X_clean.iloc[test_idx]
                y_train, y_test = y_clean.iloc[train_idx], y_clean.iloc[test_idx]

                model.fit(X_train, y_train)
                preds = model.predict(X_test)

                # Metrics per fold
                err = y_test.values - preds
                abs_err = np.abs(err)

                rmse = np.sqrt(np.mean(err ** 2))
                mae = np.mean(abs_err)

                # Percentage errors
                non_zero = y_test.values != 0
                mape = np.mean(abs_err[non_zero] / np.abs(y_test.values[non_zero])) * 100 if np.any(non_zero) else np.nan
                wape = (np.sum(abs_err) / np.sum(np.abs(y_test.values))) * 100 if np.sum(np.abs(y_test.values)) != 0 else np.nan

                fold_rmse.append(rmse)
                fold_mae.append(mae)
                fold_mape.append(mape)
                fold_wape.append(wape)

            results.append({
                "model": name,
                "rmse_mean": round(float(np.mean(fold_rmse)), 4),
                "mae_mean": round(float(np.mean(fold_mae)), 4),
                "mape_mean_pct": round(float(np.nanmean(fold_mape)), 2),
                "wape_mean_pct": round(float(np.nanmean(fold_wape)), 2),
                "folds_evaluated": n_splits
            })

        self.leaderboard_ = pd.DataFrame(results).sort_values(by="rmse_mean", ascending=True).reset_index(drop=True)
        return self.leaderboard_