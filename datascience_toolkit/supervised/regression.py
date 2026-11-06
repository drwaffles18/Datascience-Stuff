"""
regression.py
=============
Supervised regression model evaluation and tournament engine.
"""

from typing import Dict, List, Optional, Any
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_validate, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import Ridge, Lasso, LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor


class RegressionTournament:
    """Evaluates multiple regression algorithms across cross-validation splits."""

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.default_models: Dict[str, Any] = {
            "LinearRegression": LinearRegression(),
            "Ridge": Ridge(random_state=random_state),
            "Lasso": Lasso(random_state=random_state),
            "RandomForest": RandomForestRegressor(n_estimators=100, random_state=random_state),
            "GradientBoosting": GradientBoostingRegressor(random_state=random_state)
        }
        self.leaderboard_: Optional[pd.DataFrame] = None

    def run(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        models: Optional[Dict[str, Any]] = None,
        cv_splits: int = 5
    ) -> pd.DataFrame:
        """Executes cross-validation tournament and produces performance leaderboard."""
        candidate_models = models or self.default_models
        cv = KFold(n_splits=cv_splits, shuffle=True, random_state=self.random_state)

        scoring = {
            "r2": "r2",
            "rmse": "neg_root_mean_squared_error",
            "mae": "neg_mean_absolute_error"
        }

        results = []
        for name, model in candidate_models.items():
            cv_res = cross_validate(model, X, y, cv=cv, scoring=scoring, n_jobs=-1)

            results.append({
                "model": name,
                "r2_mean": round(float(np.mean(cv_res["test_r2"])), 4),
                "r2_std": round(float(np.std(cv_res["test_r2"])), 4),
                "rmse_mean": round(float(-np.mean(cv_res["test_rmse"])), 4),
                "rmse_std": round(float(np.std(cv_res["test_rmse"])), 4),
                "mae_mean": round(float(-np.mean(cv_res["test_mae"])), 4)
            })

        self.leaderboard_ = pd.DataFrame(results).sort_values(by="r2_mean", ascending=False).reset_index(drop=True)
        return self.leaderboard_

    @staticmethod
    def get_residual_analysis(y_true: pd.Series, y_pred: np.ndarray) -> pd.DataFrame:
        """Computes residuals and error metrics per observation."""
        residuals = y_true.values - y_pred
        return pd.DataFrame({
            "actual": y_true.values,
            "predicted": y_pred,
            "residual": residuals,
            "abs_error": np.abs(residuals),
            "pct_error": np.where(y_true.values != 0, np.abs(residuals) / y_true.values * 100, np.nan)
        }, index=y_true.index)