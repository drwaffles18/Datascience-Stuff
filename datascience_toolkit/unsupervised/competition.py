"""
competition.py
==============
Multi-space clustering competition engine.
Benchmarks vector spaces across k ranges, consolidates geometric metrics,
and computes consensus rankings.
"""

from typing import Dict, List, Optional, Any, Union
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    adjusted_rand_score,
    normalized_mutual_info_score,
)


class SpaceCompetition:
    """Evaluates and compares clustering viability across different vector spaces."""

    @classmethod
    def evaluate(
        cls,
        spaces: Dict[str, pd.DataFrame],
        requested_spaces: List[str],
        method: str = "kmeans",
        k_range: range = range(2, 8),
        validation_label: Optional[pd.Series] = None,
        n_init: int = 50,
        max_iter: int = 500,
        tol: float = 1e-4,
        random_state: Optional[int] = 42,
        **kwargs
    ) -> Dict[str, Any]:
        valid_spaces = [s for s in requested_spaces if s in spaces]
        if not valid_spaces:
            raise ValueError(f"Ninguno de los espacios {requested_spaces} existe en el registro.")

        all_records = []
        best_per_space = []

        for space_name in valid_spaces:
            X = spaces[space_name]
            space_records = []

            for k in k_range:
                if method.lower() in ("kmeans", "k-means"):
                    model = KMeans(
                        n_clusters=k,
                        n_init=n_init,
                        max_iter=max_iter,
                        tol=tol,
                        random_state=random_state,
                        **kwargs
                    )
                    labels = model.fit_predict(X)
                    inertia = float(model.inertia_)
                    n_iter = int(model.n_iter_)
                    converged = bool(n_iter < max_iter)
                else:
                    raise NotImplementedError(f"Método '{method}' aún no soportado en SpaceCompetition.")

                sil = float(silhouette_score(X, labels))
                ch = float(calinski_harabasz_score(X, labels))
                db = float(davies_bouldin_score(X, labels))

                rec = {
                    "space": space_name,
                    "method": method,
                    "k": k,
                    "best_k": k,
                    "silhouette": sil,
                    "calinski_harabasz": ch,
                    "davies_bouldin": db,
                    "inertia": inertia,
                    "n_iter": n_iter,
                    "converged": converged,
                }

                if validation_label is not None:
                    val_idx = validation_label.dropna().index.intersection(X.index)
                    if len(val_idx) > 0:
                        y_true = validation_label.loc[val_idx]
                        y_pred = labels[X.index.get_indexer(val_idx)]
                        rec["ari"] = float(adjusted_rand_score(y_true, y_pred))
                        rec["nmi"] = float(normalized_mutual_info_score(y_true, y_pred))

                space_records.append(rec)
                all_records.append(rec)

            # Seleccionar la mejor configuración para este espacio (por silueta)
            space_df = pd.DataFrame(space_records)
            best_row = space_df.sort_values(by="silhouette", ascending=False).iloc[0].to_dict()
            best_per_space.append(best_row)

        summary_df = pd.DataFrame(best_per_space)

        # Cálculo de consensus_rank (promedio de los rangos ordinales de las métricas)
        # Mayor silueta = mejor (1), Mayor CH = mejor (1), Menor DB = mejor (1)
        rank_sil = summary_df["silhouette"].rank(ascending=False, method="min")
        rank_ch = summary_df["calinski_harabasz"].rank(ascending=False, method="min")
        rank_db = summary_df["davies_bouldin"].rank(ascending=True, method="min")

        summary_df["consensus_rank"] = ((rank_sil + rank_ch + rank_db) / 3.0).round(6)

        # Seleccionar y ordenar las columnas exactamente como en el contrato original
        output_cols = [
            "space", "method", "best_k", "silhouette", "calinski_harabasz",
            "davies_bouldin", "inertia", "n_iter", "converged", "consensus_rank"
        ]
        if "ari" in summary_df.columns:
            output_cols.append("ari")
        if "nmi" in summary_df.columns:
            output_cols.append("nmi")

        leaderboard = summary_df[output_cols].sort_values(by="silhouette", ascending=False).reset_index(drop=True)

        # Diagnóstico narrativo
        interpretation = []
        best_overall = leaderboard.iloc[0]
        interpretation.append(
            f"Configuración óptima global: Espacio '{best_overall['space']}' con k={int(best_overall['best_k'])} "
            f"(Silhouette: {best_overall['silhouette']:.4f}, Davies-Bouldin: {best_overall['davies_bouldin']:.4f})."
        )
        for _, row in leaderboard.iterrows():
            interpretation.append(
                f"Espacio '{row['space']}': Mejor balance en k={int(row['best_k'])} (Silueta: {row['silhouette']:.4f})."
            )

        return {
            "leaderboard": leaderboard,
            "all_runs": pd.DataFrame(all_records),
            "interpretation": interpretation,
            "best_space": best_overall["space"],
            "best_k": int(best_overall["best_k"])
        }