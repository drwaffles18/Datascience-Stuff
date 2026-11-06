"""
classification.py
=================
Supervised classification model evaluation and tournament engine.
"""

from typing import Dict, List, Optional, Any
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_validate, StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier


class ClassificationTournament:
    """Evaluates multiple classification algorithms across stratified folds."""

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.default_models: Dict[str, Any] = {
            "LogisticRegression": LogisticRegression(max_iter=1000, random_state=random_state),
            "RandomForest": RandomForestClassifier(n_estimators=100, random_state=random_state),
            "GradientBoosting": GradientBoostingClassifier(random_state=random_state)
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
        cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=self.random_state)
        is_binary = y.nunique() == 2

        results = []
        for name, model in candidate_models.items():
            scoring = ["accuracy", "precision_weighted", "recall_weighted", "f1_weighted"]
            if is_binary:
                scoring.append("roc_auc")

            cv_res = cross_validate(model, X, y, cv=cv, scoring=scoring, n_jobs=-1)

            entry = {
                "model": name,
                "accuracy_mean": round(float(np.mean(cv_res["test_accuracy"])), 4),
                "accuracy_std": round(float(np.std(cv_res["test_accuracy"])), 4),
                "f1_mean": round(float(np.mean(cv_res["test_f1_weighted"])), 4),
                "precision_mean": round(float(np.mean(cv_res["test_precision_weighted"])), 4),
                "recall_mean": round(float(np.mean(cv_res["test_recall_weighted"])), 4)
            }
            if is_binary:
                entry["roc_auc_mean"] = round(float(np.mean(cv_res["test_roc_auc"])), 4)

            results.append(entry)

        self.leaderboard_ = pd.DataFrame(results).sort_values(by="f1_mean", ascending=False).reset_index(drop=True)
        return self.leaderboard_

    @staticmethod
    def get_confusion_matrix_df(y_true: pd.Series, y_pred: np.ndarray) -> pd.DataFrame:
        """Returns a labeled confusion matrix DataFrame."""
        labels = sorted(y_true.unique())
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        return pd.DataFrame(
            cm,
            index=[f"Actual_{lbl}" for lbl in labels],
            columns=[f"Pred_{lbl}" for lbl in labels]
        )