"""
competition.py
==============
Benchmarking module for comparing clustering quality across vector spaces.
Evaluates clustering stability and separation metrics across geometric spaces and k ranges.
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
)


class SpaceCompetition:
    """Benchmark engine comparing clustering performance across representation spaces."""

    @classmethod
    def evaluate(
        cls,
        spaces: Dict[str, pd.DataFrame],
        requested_spaces: List[str],
        method: str = "kmeans",
        k_range: range = range(2, 8),
        validation_label: Optional[pd.Series] = None,
        n_init: int = 20,
        max_iter: int = 300,
        random_state: Optional[int] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Runs clustering benchmarks across selected vector spaces and cluster counts."""
        records = []
        fitted_models: Dict[str, Any] = {}
        labels_by_run: Dict[str, np.ndarray] = {}

        # 1. Validar qué espacios solicitados están presentes en el registro
        available_spaces = [s for s in requested_spaces if s in spaces]
        missing_spaces = [s for s in requested_spaces if s not in spaces]

        if missing_spaces:
            print(f"⚠️ Espacios no encontrados en el registro y omitidos: {missing_spaces}")

        if not available_spaces:
            raise ValueError(
                f"Ninguno de los espacios solicitados {requested_spaces} está disponible en el registro. "
                f"Espacios disponibles: {list(spaces.keys())}"
            )

        # 2. Extraer parámetros específicos para KMeans admitiendo kwargs arbitrarios
        tol = kwargs.pop("tol", 1e-4)

        # 3. Torneo de clustering
        for space_name in available_spaces:
            X = spaces[space_name]

            for k in k_range:
                run_key = f"{space_name}__k{k}"

                if method.lower() == "kmeans":
                    model = KMeans(
                        n_clusters=k,
                        n_init=n_init,
                        max_iter=max_iter,
                        tol=tol,
                        random_state=random_state,
                        **kwargs
                    )
                    cluster_labels = model.fit_predict(X)
                else:
                    raise NotImplementedError(f"Método de clustering '{method}' no soportado todavía.")

                # Métricas internas de validación geométrica
                sil = float(silhouette_score(X, cluster_labels))
                ch = float(calinski_harabasz_score(X, cluster_labels))
                db = float(davies_bouldin_score(X, cluster_labels))

                row = {
                    "space": space_name,
                    "k": k,
                    "silhouette": round(sil, 4),
                    "calinski_harabasz": round(ch, 2),
                    "davies_bouldin": round(db, 4),
                }

                # Métricas externas opcionales si se proporcionó validación externa (ground truth)
                if validation_label is not None:
                    valid_idx = validation_label.dropna().index.intersection(X.index)
                    if len(valid_idx) > 0:
                        y_true = validation_label.loc[valid_idx]
                        y_pred = cluster_labels[X.index.get_indexer(valid_idx)]
                        row["ari"] = round(float(adjusted_rand_score(y_true, y_pred)), 4)
                        row["nmi"] = round(float(normalized_mutual_info_score(y_true, y_pred)), 4)

                records.append(row)
                fitted_models[run_key] = model
                labels_by_run[run_key] = cluster_labels

        leaderboard = pd.DataFrame(records)

        # Ordenar el leaderboard por Silhouette Score descendente
        if not leaderboard.empty:
            leaderboard = leaderboard.sort_values(by="silhouette", ascending=False).reset_index(drop=True)

        # 4. Generar interpretaciones automáticas
        interpretations = cls._generate_interpretation(leaderboard)

        return {
            "leaderboard": leaderboard,
            "models": fitted_models,
            "labels": labels_by_run,
            "interpretation": interpretations,
        }

    @staticmethod
    def _generate_interpretation(df: pd.DataFrame) -> List[str]:
        """Synthesizes high-level takeaways from benchmark metrics."""
        if df.empty:
            return ["No se generaron resultados para interpretar."]

        takeaways = []

        # Mejor configuración global por silueta
        best_row = df.iloc[0]
        takeaways.append(
            f"Configuración óptima global: Espacio '{best_row['space']}' con k={int(best_row['k'])} "
            f"(Silhouette: {best_row['silhouette']:.4f}, Davies-Bouldin: {best_row['davies_bouldin']:.4f})."
        )

        # Mejor k dentro de cada espacio evaluado
        for space, grp in df.groupby("space", sort=False):
            best_in_space = grp.sort_values(by="silhouette", ascending=False).iloc[0]
            takeaways.append(
                f"Espacio '{space}': Mejor balance en k={int(best_in_space['k'])} "
                f"(Silueta: {best_in_space['silhouette']:.4f})."
            )

        # Indicador de Davies-Bouldin
        best_db = df.sort_values(by="davies_bouldin", ascending=True).iloc[0]
        if best_db["space"] != best_row["space"] or best_db["k"] != best_row["k"]:
            takeaways.append(
                f"Mayor compacidad de clústeres según Davies-Bouldin: Espacio '{best_db['space']}' con k={int(best_db['k'])} "
                f"({best_db['davies_bouldin']:.4f})."
            )

        return takeaways