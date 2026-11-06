"""
profiling.py
============
Audience profiling and affinity indexing engine for cluster characterization.
Computes Index Scores (Base = 100), Lift, and segment scorecards.
"""

from typing import Dict, List, Optional
import numpy as np
import pandas as pd


class AudienceProfiler:
    """Computes marketing affinity scores, demographic lift, and over/under-indexing ratios."""

    def __init__(self, data: pd.DataFrame, cluster_col: str):
        if cluster_col not in data.columns:
            raise KeyError(f"Cluster column '{cluster_col}' not found in DataFrame.")
        self.data = data.copy()
        self.cluster_col = cluster_col
        self.clusters = sorted(self.data[cluster_col].dropna().unique())

    def compute_index(
        self,
        target_cluster: any,
        feature: str,
        is_categorical: bool = False
    ) -> pd.DataFrame:
        """
        Calculates Affinity Index for a given feature.
        Formula: Index = (% in Segment / % in Base) * 100
        Index > 120: High affinity (over-indexed)
        Index < 80: Low affinity (under-indexed)
        """
        base_df = self.data.dropna(subset=[feature])
        seg_df = base_df[base_df[self.cluster_col] == target_cluster]

        if is_categorical:
            base_dist = base_df[feature].value_counts(normalize=True) * 100
            seg_dist = seg_df[feature].value_counts(normalize=True) * 100

            comparison = pd.DataFrame({
                "segment_pct": seg_dist,
                "base_pct": base_dist
            }).fillna(0)

            comparison["index_score"] = ((comparison["segment_pct"] / comparison["base_pct"]) * 100).round(1)
            comparison["lift"] = (comparison["segment_pct"] / comparison["base_pct"]).round(2)
            comparison["target_cluster"] = target_cluster
            return comparison.sort_values(by="index_score", ascending=False)
        else:
            mean_seg = seg_df[feature].mean()
            mean_base = base_df[feature].mean()
            median_seg = seg_df[feature].median()
            median_base = base_df[feature].median()
            index_score = (mean_seg / mean_base) * 100 if mean_base != 0 else np.nan

            return pd.DataFrame([{
                "feature": feature,
                "segment_mean": round(mean_seg, 2),
                "base_mean": round(mean_base, 2),
                "segment_median": round(median_seg, 2),
                "base_median": round(median_base, 2),
                "index_score": round(index_score, 1),
                "lift": round(mean_seg / mean_base, 2) if mean_base != 0 else np.nan,
                "target_cluster": target_cluster
            }])

    def build_scorecard(
        self,
        numeric_features: List[str],
        categorical_features: Optional[List[str]] = None
    ) -> Dict[any, pd.DataFrame]:
        """Generates comprehensive affinity scorecards for all clusters."""
        scorecards = {}

        for cl in self.clusters:
            num_rows = []
            for num_col in numeric_features:
                if num_col in self.data.columns:
                    num_rows.append(self.compute_index(target_cluster=cl, feature=num_col, is_categorical=False))

            df_num = pd.concat(num_rows, ignore_index=True) if num_rows else pd.DataFrame()

            cat_rows = []
            if categorical_features:
                for cat_col in categorical_features:
                    if cat_col in self.data.columns:
                        cat_res = self.compute_index(target_cluster=cl, feature=cat_col, is_categorical=True)
                        cat_res = cat_res.reset_index().rename(columns={"index": "category_value"})
                        cat_res["feature"] = cat_col
                        cat_rows.append(cat_res)

            df_cat = pd.concat(cat_rows, ignore_index=True) if cat_rows else pd.DataFrame()

            scorecards[cl] = {
                "numeric_metrics": df_num,
                "categorical_affinity": df_cat
            }

        return scorecards