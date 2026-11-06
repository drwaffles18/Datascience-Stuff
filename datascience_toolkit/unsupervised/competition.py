"""
competition.py
==============
Evaluator that benchmarks clustering partitions across multiple vector representations.
"""

from typing import Dict, List, Optional, Any
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    adjusted_rand_score,
    normalized_mutual_info_score,
    homogeneity_completeness_v_measure
)


class SpaceCompetition:
    """Runs clustering across spaces and generates performance leaderboards with diagnostic signals."""

    @staticmethod
    def evaluate(
        spaces: Dict[str, pd.DataFrame],
        requested_spaces: List[str],
        k_range: range = range(2, 8),
        validation_label: Optional[pd.Series] = None,
        n_init: int = 20,
        max_iter: int = 300,
        random_state: Optional[int] = None
    ) -> Dict[str, Any]:
        results = []

        for space_name in requested_spaces:
            if space_name not in spaces:
                continue

            X = spaces[space_name].values
            best_k = None
            best_sil = -1.0
            best_model = None
            best_labels = None

            # Sample subset for silhouette if volume > 20,000 to keep execution fast
            sample_size = 20000 if len(X) > 20000 else None

            for k in k_range:
                km = KMeans(n_clusters=k, n_init=n_init, max_iter=max_iter, random_state=random_state)
                labels = km.fit_predict(X)
                sil = silhouette_score(X, labels, sample_size=sample_size, random_state=random_state)

                if sil > best_sil:
                    best_sil = sil
                    best_k = k
                    best_model = km
                    best_labels = labels

            # Compute definitive metrics on the optimal k
            ch = calinski_harabasz_score(X, best_labels)
            db = davies_bouldin_score(X, best_labels)

            row = {
                "space": space_name,
                "method": "kmeans",
                "best_k": best_k,
                "silhouette": round(best_sil, 6),
                "calinski_harabasz": round(ch, 6),
                "davies_bouldin": round(db, 6),
                "inertia": round(best_model.inertia_, 6),
                "n_iter": best_model.n_iter_,
                "converged": True,
            }

            # External validation label evaluation if provided
            if validation_label is not None:
                val = validation_label.dropna()
                common_idx = val.index.intersection(spaces[space_name].index)
                if len(common_idx) > 0:
                    y_true = val.loc[common_idx]
                    y_pred = best_labels[spaces[space_name].index.get_indexer(common_idx)]

                    homo, comp, v_meas = homogeneity_completeness_v_measure(y_true, y_pred)
                    row.update({
                        "validation_label": validation_label.name or "target",
                        "validation_n_classes": val.nunique(),
                        "k_matches_validation": best_k == val.nunique(),
                        "adjusted_rand_index": round(adjusted_rand_score(y_true, y_pred), 6),
                        "normalized_mutual_info": round(normalized_mutual_info_score(y_true, y_pred), 6),
                        "homogeneity": round(homo, 6),
                        "completeness": round(comp, 6),
                        "v_measure": round(v_meas, 6)
                    })

            results.append(row)

        leaderboard = pd.DataFrame(results).sort_values(by="silhouette", ascending=False).reset_index(drop=True)
        signals = SpaceCompetition._generate_signals(leaderboard)

        return {
            "leaderboard": leaderboard,
            "interpretation": signals
        }

    @staticmethod
    def _generate_signals(df: pd.DataFrame) -> List[str]:
        if df.empty:
            return ["No spaces were evaluated."]

        top = df.iloc[0]
        signals = [
            f"El espacio mejor posicionado por métricas internas es '{top['space']}' con k={top['best_k']}, silhouette={top['silhouette']:.3f} y Davies-Bouldin={top['davies_bouldin']:.3f}."
        ]

        if "best_k" in df.columns and df["best_k"].nunique() == 1:
            signals.append(f"Todos los espacios evaluados coinciden en k={top['best_k']}, lo que aporta consistencia entre representaciones.")

        if "adjusted_rand_index" in df.columns:
            signals.append(f"ARI={top['adjusted_rand_index']:.3f}: la partición interna difiere de la etiqueta conocida.")
            signals.append(f"V-measure={top['v_measure']:.3f}, resumen del balance entre homogeneidad y completitud.")

        return signals