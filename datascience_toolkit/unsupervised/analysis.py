"""
analysis.py
===========
Orchestrator class for unsupervised data exploration, dimensionality reduction,
and multi-space clustering tournaments.
"""

from typing import Dict, List, Optional, Tuple, Any, Union
import itertools
import numpy as np
import pandas as pd

from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    adjusted_rand_score,
    normalized_mutual_info_score,
)

from .embeddings import EmbeddingManager
from .competition import SpaceCompetition
from ..preprocessing import DFPreprocess


from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors

class ExperimentResult(dict):
    """
    Contenedor de resultados del torneo de clustering compatible con acceso dict y atributo.
    Ejecuta el torneo diferido con .run(), renderiza la tabla de competición y calcula el leaderboard.
    """
    def __init__(self, runner_fn=None, plan_df: Optional[pd.DataFrame] = None, *args, **kwargs):
        dict.__init__(self, *args, **kwargs)
        object.__setattr__(self, "_runner_fn", runner_fn)
        object.__setattr__(self, "_is_executed", False)
        object.__setattr__(self, "plan_df", plan_df if plan_df is not None else pd.DataFrame())

    def __getattr__(self, name: str) -> Any:
        if name in self.__dict__:
            return self.__dict__[name]
        if dict.__contains__(self, name):
            return dict.__getitem__(self, name)
        if not self._is_executed and self._runner_fn is not None:
            self.run(verbose=False)
            if dict.__contains__(self, name):
                return dict.__getitem__(self, name)
        raise AttributeError(f"'ExperimentResult' no contiene el atributo '{name}'")

    def __setattr__(self, name: str, value: Any) -> None:
        if name.startswith("_") or name == "plan_df":
            object.__setattr__(self, name, value)
        else:
            self[name] = value

    def __getitem__(self, key: Any) -> Any:
        if not self._is_executed and not dict.__contains__(self, key) and self._runner_fn is not None:
            self.run(verbose=False)
        return dict.__getitem__(self, key)

    def run(self, verbose: bool = True) -> "ExperimentResult":
        """Ejecuta el torneo de modelos, muestra la tabla completa de la competición y consolida métricas."""
        from IPython.display import display

        if not self._is_executed and self._runner_fn is not None:
            results = self._runner_fn()
            self.update(results)
            object.__setattr__(self, "_is_executed", True)

        lb = self.get("leaderboard", pd.DataFrame())

        if verbose and not lb.empty:
            print("========== CLUSTERING COMPETITION ==========")
            with pd.option_context("display.max_colwidth", None):
                display(lb)

        return self

    def leaderboard(self, top_n: Optional[int] = None) -> pd.DataFrame:
        """Retorna el DataFrame del leaderboard ordenado por puntaje de competición."""
        if not self._is_executed and self._runner_fn is not None:
            self.run(verbose=False)
        lb = dict.get(self, "leaderboard", pd.DataFrame())
        if top_n is not None and isinstance(lb, pd.DataFrame):
            return lb.head(top_n)
        return lb

    def plan(self, top_n: int = 5) -> pd.DataFrame:
        """Genera un plan de decisión comparativo y accionable post-competición."""
        if not self._is_executed and self._runner_fn is not None:
            self.run(verbose=False)
        lb = self.leaderboard().copy()
        if lb.empty:
            return pd.DataFrame()

        candidates = []
        seen_keys = set()

        def add_candidate(row, role, rationale, next_step):
            k_val = row.get("n_clusters", row.get("k", 2))
            key = (row["space"], row["model"], int(k_val))
            if key not in seen_keys:
                seen_keys.add(key)
                rec = {
                    "role": role,
                    "space": row["space"],
                    "model": row["model"],
                    "k": int(k_val),
                    "silhouette": row["silhouette"],
                    "davies_bouldin": row["davies_bouldin"],
                    "stability": row.get("stability_ari", row.get("stability", np.nan)),
                    "strategic_rationale": rationale,
                    "next_action": next_step
                }
                candidates.append(rec)

        best = lb.iloc[0]
        add_candidate(
            best,
            "Primary Champion",
            f"Mayor puntuación global en el torneo (Score: {best.get('competition_score', 'N/A')}, Silueta: {best['silhouette']:.4f}).",
            "Proceder a perfilamiento de arquetipos y análisis de medias por clúster."
        )

        if "stability_ari" in lb.columns:
            best_stab = lb.sort_values(by="stability_ari", ascending=False).iloc[0]
            add_candidate(
                best_stab,
                "Stability Champion",
                f"Máxima consistencia (ARI: {best_stab['stability_ari']:.4f}).",
                "Usar si la reproducibilidad en producción es prioritaria sobre la separación pura."
            )

        best_db = lb.sort_values(by="davies_bouldin", ascending=True).iloc[0]
        add_candidate(
            best_db,
            "Compactness Champion",
            f"Mínima dispersión intra-clúster (DB: {best_db['davies_bouldin']:.4f}).",
            "Examinar fronteras de decisión y dispersión de outliers."
        )

        for _, row in lb.iterrows():
            if len(candidates) >= top_n:
                break
            add_candidate(
                row,
                f"Runner-up (#{len(candidates)+1})",
                f"Candidato competitivo con silueta {row['silhouette']:.4f}.",
                "Considerar como alternativa de respaldo."
            )

        return pd.DataFrame(candidates)

    def explain(self) -> None:
        """Imprime un diagnóstico analítico y narrativo detallado sobre el torneo."""
        if not self._is_executed and self._runner_fn is not None:
            self.run(verbose=False)
        lb = self.leaderboard()
        if lb.empty:
            print("No hay resultados registrados en el experimento para explicar.")
            return

        best = lb.iloc[0]
        k_val = best.get("n_clusters", best.get("k", 2))

        print("=" * 75)
        print("📊 CLUSTERING EXPERIMENT DIAGNOSIS & INTERPRETATION")
        print("=" * 75)
        print("\n1. CONFIGURACIÓN GANADORA (CHAMPION)")
        print(f"   - Espacio vectorial:   {best['space']}")
        print(f"   - Familia de modelo:   {best['model'].upper()}")
        print(f"   - Número de clusters:  k = {int(k_val)}")
        print(f"   - Silhouette Score:    {best['silhouette']:.4f}")
        print(f"   - Davies-Bouldin:      {best['davies_bouldin']:.4f}")
        if "stability_ari" in best:
            print(f"   - Estabilidad (ARI):   {best['stability_ari']:.4f}")
        print("=" * 75)

class ClusterSolution(dict):
    """
    Contenedor de la solución final de clustering ajustada.
    Permite acceso transparente como atributo y diccionario, e incorpora
    métodos de perfilamiento, conteo, extracción de datos y graficación de arquetipos.
    """
    def __getattr__(self, name: str) -> Any:
        if name in self:
            return self[name]
        raise AttributeError(f"'ClusterSolution' no contiene el atributo '{name}'")

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value

    def dataset(self, which: str = "raw") -> pd.DataFrame:
        """
        Retorna el DataFrame con la columna de clústeres incorporada.
        Garantiza que la columna 'cluster' siempre esté presente.
        """
        if which.lower() in ("prepared", "prep", "features"):
            df = self.get("prep_with_clusters", self.get("data_with_clusters"))
        else:
            df = self.get("data_with_clusters")

        if df is not None:
            df = df.copy()
            if "cluster" not in df.columns and "labels" in self:
                df["cluster"] = self["labels"].values
            return df
        return pd.DataFrame()

    def summary(self) -> pd.DataFrame:
        """Retorna la distribución de frecuencias y porcentajes por clúster."""
        labels = self.get("labels", pd.Series(dtype=int))
        counts = labels.value_counts().sort_index()
        pcts = (labels.value_counts(normalize=True).sort_index() * 100).round(2)
        return pd.DataFrame({"count": counts, "percentage": pcts})

    def profile(self, df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Genera el perfil medio de las variables numéricas agrupadas por clúster."""
        target_df = df if df is not None else self.get("raw_data", self.get("prepared_data"))
        if target_df is None:
            return pd.DataFrame()
        num_df = target_df.select_dtypes(include=[np.number]).copy()

        cols_to_drop = [c for c in ["cluster", self.get("name", "")] if c in num_df.columns]
        if cols_to_drop:
            num_df = num_df.drop(columns=cols_to_drop)

        num_df["cluster"] = np.asarray(self["labels"])
        return num_df.groupby("cluster").mean().round(3)

    def describe(
        self,
        top_n: int = 5,
        min_abs_z: float = 0.4,
        which: str = "raw",
        verbose: bool = True
    ) -> pd.DataFrame:
        """
        Genera el perfil diferencial de cada clúster basándose en el Z-Score
        respecto a la media poblacional.
        """
        target_df = self.get("raw_data" if which == "raw" else "prepared_data")
        if target_df is None:
            target_df = self.get("data_with_clusters")

        labels = self.get("labels")
        if labels is None or target_df is None:
            return pd.DataFrame()

        num_df = target_df.select_dtypes(include=[np.number]).copy()
        cols_to_drop = [c for c in ["cluster", self.get("name", "")] if c in num_df.columns]
        if cols_to_drop:
            num_df = num_df.drop(columns=cols_to_drop)

        global_means = num_df.mean()
        global_stds = num_df.std(ddof=0).replace(0, np.nan)

        records = []
        cluster_ids = sorted(pd.Series(labels).unique())

        for c_id in cluster_ids:
            c_mask = (np.asarray(labels) == c_id)
            c_df = num_df.iloc[c_mask]
            c_means = c_df.mean()

            z_scores = (c_means - global_means) / global_stds
            filtered_z = z_scores[z_scores.abs() >= min_abs_z]
            top_features = filtered_z.abs().sort_values(ascending=False).head(top_n).index

            for feat in top_features:
                z_val = z_scores[feat]
                records.append({
                    "cluster": c_id,
                    "feature": feat,
                    "cluster_mean": round(float(c_means[feat]), 3),
                    "global_mean": round(float(global_means[feat]), 3),
                    "z_score": round(float(z_val), 3),
                    "direction": "High (+)" if z_val > 0 else "Low (-)"
                })

        result_df = pd.DataFrame(records)

        if verbose and not result_df.empty:
            print("=" * 70)
            print("📋 DISTINCTIVE CLUSTER PROFILES (Z-SCORE DIFFERENTIAL)")
            print("=" * 70)
            for c_id in cluster_ids:
                c_rows = result_df[result_df["cluster"] == c_id]
                size = int((np.asarray(labels) == c_id).sum())
                pct = (size / len(labels)) * 100
                print(f"\n🔹 Clúster {c_id} ({size} observaciones, {pct:.1f}%):")
                if c_rows.empty:
                    print(f"   (Sin variables que superen |z| >= {min_abs_z})")
                else:
                    for _, r in c_rows.iterrows():
                        arrow = "▲" if r["z_score"] > 0 else "▼"
                        print(f"   {arrow} {r['feature']:<25} | z = {r['z_score']:+.2f} | Clúster: {r['cluster_mean']} (Global: {r['global_mean']})")
            print("=" * 70)

        return result_df

    def _get_cluster_means(self, normalize: Union[bool, str] = True, which: str = "raw") -> pd.DataFrame:
        """Calcula las medias de variables por clúster con o sin normalización."""
        target_df = self.get("raw_data" if which == "raw" else "prepared_data")
        if target_df is None:
            target_df = self.get("data_with_clusters")

        num_df = target_df.select_dtypes(include=[np.number]).copy()
        cols_to_drop = [c for c in ["cluster", self.get("name", "")] if c in num_df.columns]
        if cols_to_drop:
            num_df = num_df.drop(columns=cols_to_drop)

        if normalize in (True, "minmax"):
            denom = (num_df.max() - num_df.min()).replace(0, np.nan)
            scaled_df = (num_df - num_df.min()) / denom
        elif normalize in ("zscore", "standard"):
            stds = num_df.std(ddof=0).replace(0, np.nan)
            scaled_df = (num_df - num_df.mean()) / stds
        else:
            scaled_df = num_df.copy()

        scaled_df["cluster"] = np.asarray(self["labels"])
        return scaled_df.groupby("cluster").mean()

    def plot_cluster_bars(
        self,
        normalize: Union[bool, str] = True,
        which: str = "raw",
        figsize: Optional[Tuple[float, float]] = None
    ) -> None:
        """
        Genera gráficos de barras horizontales comparativos por clúster ordenados
        por los valores del clúster de referencia.
        """
        import matplotlib.pyplot as plt

        means = self._get_cluster_means(normalize=normalize, which=which)
        cluster_ids = sorted(means.index)
        k = len(cluster_ids)

        # Ordenar características según los valores del Cluster 0 de forma descendente
        sort_cluster = cluster_ids[0]
        feature_order = means.loc[sort_cluster].sort_values(ascending=False).index.tolist()

        fig_size = figsize or (4 * k, max(4.5, len(feature_order) * 0.45))
        fig, axes = plt.subplots(1, k, figsize=fig_size, sharey=True)
        if k == 1:
            axes = [axes]

        sol_name = self.get("name", "cluster_solution")
        fig.suptitle(f"{sol_name} — centroides por cluster", fontsize=14, fontweight="bold", y=0.98)

        x_label = "scaled" if normalize else "mean"
        for c_id, ax in zip(cluster_ids, axes):
            c_vals = means.loc[c_id, feature_order]
            ax.barh(feature_order, c_vals)
            ax.invert_yaxis()
            ax.set_title(f"Cluster {c_id}")
            ax.set_xlabel(x_label)
            if normalize in (True, "minmax"):
                ax.set_xlim(0, 1.05)
                ax.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])

        plt.tight_layout()
        plt.show()

    def plot_cluster_radar(
        self,
        normalize: Union[bool, str] = "minmax",
        which: str = "raw",
        figsize: Optional[Tuple[float, float]] = None
    ) -> None:
        """
        Genera el gráfico polar radar superponiendo los perfiles de todos los clústeres.
        """
        import matplotlib.pyplot as plt

        means = self._get_cluster_means(normalize=normalize, which=which)
        cluster_ids = sorted(means.index)
        features = means.columns.tolist()
        num_vars = len(features)

        angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
        angles_closed = angles + [angles[0]]

        fig_size = figsize or (6.5, 6.5)
        fig, ax = plt.subplots(figsize=fig_size, subplot_kw=dict(polar=True))

        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        ax.set_xticks(angles)
        ax.set_xticklabels(features, fontsize=9)
        ax.set_rlabel_position(0)

        if normalize in (True, "minmax"):
            ax.set_ylim(0, 1.05)
            ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
            ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=8)

        for c_id in cluster_ids:
            values = means.loc[c_id, features].tolist()
            values_closed = values + [values[0]]
            ax.plot(angles_closed, values_closed, label=f"Cluster {c_id}", linewidth=1.2)
            ax.fill(angles_closed, values_closed, alpha=0.2)

        sol_name = self.get("name", "cluster_solution")
        ax.set_title(f"{sol_name} — perfil radar", fontsize=11, pad=25)
        ax.legend(loc="upper right", bbox_to_anchor=(1.15, 1.15))

        plt.tight_layout()
        plt.show()

class UnsupervisedAnalysis:
    """Unified API for unsupervised clustering workflows."""

    def __init__(self, data: pd.DataFrame, validation_label: Optional[pd.Series] = None):
        self.raw_data = data.copy()
        self.data = data.copy()
        self.validation_label = validation_label
        self.preprocessor: Optional[DFPreprocess] = None
        self.prepared_data_: Optional[pd.DataFrame] = None
        self.manager = EmbeddingManager(self.data)
        self.manager.registry["embed"] = self.data
        self.last_competition: Optional[Dict[str, Any]] = None
        self.last_experiment: Optional[ExperimentResult] = None
        self.last_solution: Optional[ClusterSolution] = None

    def prepare(
        self,
        scale_num: str = "auto",
        cat_encoding: str = "auto",
        drop_cols: Optional[List[str]] = None,
        verbose: bool = True,
        **kwargs
    ) -> pd.DataFrame:
        """
        Prepara los datos tabulares aplicando imputación, codificación categórica
        y escalado numérico para algoritmos basados en distancias geométricas.
        """
        num_strategy = "standard" if scale_num == "auto" else scale_num
        cat_strategy = "onehot" if cat_encoding == "auto" else cat_encoding

        df_to_prep = self.raw_data.copy()
        if drop_cols:
            dropped = [c for c in drop_cols if c in df_to_prep.columns]
            df_to_prep = df_to_prep.drop(columns=dropped)
        else:
            dropped = []

        self.preprocessor = DFPreprocess(
            scale_num=num_strategy,
            encoding=cat_strategy,
            **kwargs
        )

        self.prepared_data_ = self.preprocessor.fit_transform(df_to_prep)
        self.data = self.prepared_data_.copy()

        self.manager = EmbeddingManager(self.data)
        self.manager.registry["embed"] = self.data

        if verbose:
            print("=" * 60)
            print("✓ DATA PREPARATION SUMMARY")
            print("=" * 60)
            print(f"- Filas procesadas:        {self.prepared_data_.shape[0]}")
            print(f"- Columnas originales:     {df_to_prep.shape[1]}")
            print(f"- Características finales: {self.prepared_data_.shape[1]} (tras encoding)")
            print(f"- Estrategia de escalado:  {num_strategy}")
            print(f"- Codificación categórica: {cat_strategy}")
            if dropped:
                print(f"- Columnas excluidas:      {dropped}")
            print("=" * 60)

        return self.prepared_data_

    def explore(
        self,
        methods: Tuple[str, ...] = ("pca", "umap", "tsne"),
        pca_model_variance: float = 0.90,
        umap_params: Optional[dict] = None,
        tsne_params: Optional[dict] = None,
        plot: bool = False
    ) -> Dict[str, Any]:
        """Calculates dimensional representations (PCA, UMAP, t-SNE) and catalogs them."""
        self.manager = EmbeddingManager(self.data)
        self.manager.registry["embed"] = self.data

        if "pca" in methods:
            self.manager.compute_pca(target_variance=pca_model_variance)

        if "umap" in methods:
            params = umap_params or {"n_neighbors": 15, "min_dist": 0.1, "metric": "euclidean"}
            self.manager.compute_umap(**params)

        if "tsne" in methods:
            params = tsne_params or {"perplexity": 30.0, "learning_rate": "auto"}
            self.manager.compute_tsne(**params)

        report_df = self.manager.get_catalog_df()

        if plot:
            import matplotlib.pyplot as plt

            plot_targets = [
                ("PCA", self.manager.registry.get("pca_visual", self.manager.registry.get("pca"))),
                ("UMAP", self.manager.registry.get("umap_visual", self.manager.registry.get("umap"))),
                ("t-SNE", self.manager.registry.get("tsne_visual", self.manager.registry.get("tsne"))),
            ]
            valid_plots = [(title, space) for title, space in plot_targets if space is not None and space.shape[1] >= 2]

            if valid_plots:
                fig, axes = plt.subplots(1, len(valid_plots), figsize=(5 * len(valid_plots), 4.5))
                if len(valid_plots) == 1:
                    axes = [axes]

                for ax, (title, space_df) in zip(axes, valid_plots):
                    ax.scatter(space_df.iloc[:, 0], space_df.iloc[:, 1], alpha=0.7, edgecolors="none", s=25)
                    ax.set_title(f"Espacio: {title}")
                    ax.set_xlabel(space_df.columns[0])
                    ax.set_ylabel(space_df.columns[1])
                    ax.grid(True, linestyle="--", alpha=0.5)

                plt.tight_layout()
                plt.show()

        return {
            "report": report_df,
            "registry": self.manager.registry
        }

    def compare_spaces(
        self,
        spaces: List[str],
        method: str = "kmeans",
        k_range: range = range(2, 8),
        n_init: int = 20,
        max_iter: int = 300,
        random_state: Optional[int] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Runs clustering benchmarks across specified vector spaces."""
        if "embed" not in self.manager.registry:
            self.manager.registry["embed"] = self.data

        comp_output = SpaceCompetition.evaluate(
            spaces=self.manager.registry,
            requested_spaces=spaces,
            method=method,
            k_range=k_range,
            validation_label=self.validation_label,
            n_init=n_init,
            max_iter=max_iter,
            random_state=random_state,
            **kwargs
        )
        self.last_competition = comp_output

        print("\n========== SPACE COMPARISON INTERPRETATION ==========")
        for s in comp_output.get("interpretation", []):
            print(f"- {s}")

        return comp_output

    def available_cluster_models(self) -> pd.DataFrame:
        """
        Cataloga los 6 modelos de clustering soportados respetando el esquema
        original de 8 columnas y validación dinámica de librerías opcionales.
        """
        # Validación dinámica de kmedoids
        kmedoids_available = False
        kmedoids_note = ""
        try:
            from sklearn_extra.cluster import KMedoids  # type: ignore
            kmedoids_available = True
        except ImportError:
            try:
                import kmedoids  # type: ignore
                kmedoids_available = True
            except ImportError:
                kmedoids_available = False
                kmedoids_note = "optional dependency; install via pip install scikit-learn-extra"

        # Validación dinámica de hdbscan
        hdbscan_available = False
        hdbscan_note = ""
        try:
            from sklearn.cluster import HDBSCAN  # type: ignore
            hdbscan_available = True
        except ImportError:
            try:
                import hdbscan  # type: ignore
                hdbscan_available = True
            except ImportError:
                hdbscan_available = False
                hdbscan_note = "optional dependency; install via pip install hdbscan"

        models_catalog = [
            {
                "model": "kmeans",
                "family": "centroid",
                "available": True,
                "requires_k": True,
                "supports_noise": False,
                "recommended_for": "scaled numeric data; compact clusters; large datasets",
                "limitations": "requires k; sensitive to outliers and non-spherical geometries",
                "note": ""
            },
            {
                "model": "kmedoids",
                "family": "medoid",
                "available": kmedoids_available,
                "requires_k": True,
                "supports_noise": False,
                "recommended_for": "outlier-prone data; non-euclidean distances; small/medium datasets",
                "limitations": "slower than kmeans; optional sklearn-extra dependency",
                "note": kmedoids_note
            },
            {
                "model": "agglomerative",
                "family": "hierarchical",
                "available": True,
                "requires_k": True,
                "supports_noise": False,
                "recommended_for": "small/medium datasets; hierarchical structure; deterministic clustering",
                "limitations": "no native predict; memory cost grows with sample size",
                "note": ""
            },
            {
                "model": "gaussian_mixture",
                "family": "probabilistic",
                "available": True,
                "requires_k": True,
                "supports_noise": False,
                "recommended_for": "elliptical clusters; overlapping segments; soft membership",
                "limitations": "distributional assumptions; can converge to local optima",
                "note": ""
            },
            {
                "model": "dbscan",
                "family": "density",
                "available": True,
                "requires_k": False,
                "supports_noise": True,
                "recommended_for": "irregular shapes; noise detection; unknown number of clusters",
                "limitations": "sensitive to eps; struggles with varying density",
                "note": ""
            },
            {
                "model": "hdbscan",
                "family": "density",
                "available": hdbscan_available,
                "requires_k": False,
                "supports_noise": True,
                "recommended_for": "varying densities; noise detection; unknown number of clusters",
                "limitations": "optional dependency; small samples may be unstable",
                "note": hdbscan_note
            }
        ]

        return pd.DataFrame(models_catalog)

    def recommend_cluster_models(
        self,
        X: Union[str, pd.DataFrame, np.ndarray] = "embed"
    ) -> pd.DataFrame:
        """Analiza propiedades dimensionales y recomienda familias de clustering idóneas."""
        if isinstance(X, str):
            if X in self.manager.registry:
                matrix = self.manager.registry[X]
            else:
                raise ValueError(
                    f"El espacio '{X}' no existe en el registro. "
                    f"Espacios disponibles: {list(self.manager.registry.keys())}"
                )
        elif isinstance(X, (pd.DataFrame, np.ndarray)):
            matrix = X
        else:
            matrix = self.data

        n_samples, n_features = matrix.shape
        recommendations = []

        recommendations.append({
            "algorithm": "K-Means",
            "suitability": "Highly Recommended",
            "rationale": (
                f"Excelente línea base para {n_samples} muestras y {n_features} dimensiones. "
                "Supone clústeres globulares/esféricos de varianza similar."
            ),
            "suggested_params": "k: range(2, 8), n_init=20 o 50"
        })

        if n_samples <= 2500:
            hier_suit = "Highly Recommended"
            hier_rat = (
                f"Ideal para datasets pequeños a medianos (n={n_samples}). Permite descubrir "
                "jerarquías taxonómicas sin fijar k de antemano y no sufre de convergencia local."
            )
        else:
            hier_suit = "Conditional / Slow"
            hier_rat = f"Complejidad O(n^2). Con n={n_samples} el consumo de memoria para la matriz de enlace puede ser elevado."

        recommendations.append({
            "algorithm": "Agglomerative (Hierarchical)",
            "suitability": hier_suit,
            "rationale": hier_rat,
            "suggested_params": "linkage='ward', metric='euclidean'"
        })

        params_per_cluster = (n_features * (n_features + 1)) // 2
        if n_samples > 10 * n_features:
            gmm_suit = "Recommended"
            gmm_rat = (
                f"Permite clústeres elipsoidales con asignación probabilística suave (soft clustering). "
                f"Relación muestras/variables ({n_samples}/{n_features}) adecuada."
            )
        else:
            gmm_suit = "Caution (Risk of Overfitting)"
            gmm_rat = (
                f"Relación muestras/variables reducida ({n_samples} muestras para {n_features} dimensiones). "
                f"Cada clúster con covarianza 'full' estima {params_per_cluster} parámetros. Usar covarianza 'diag' o 'spherical'."
            )

        recommendations.append({
            "algorithm": "Gaussian Mixture Models (GMM)",
            "suitability": gmm_suit,
            "rationale": gmm_rat,
            "suggested_params": "covariance_type='diag' si d es alto, n_components: 2-6"
        })

        if n_features <= 15:
            db_suit = "Recommended"
            db_rat = (
                f"Capaz de encontrar formas no esféricas y aislar ruido/outliers. "
                f"Con {n_features} dimensiones la distancia euclidiana aún mantiene contraste espacial."
            )
        else:
            db_suit = "Caution"
            db_rat = f"En alta dimensión ({n_features} variables) la distancia euclidiana tiende a equipararse (maldición de la dimensionalidad), dificultando el ajuste de eps."

        recommendations.append({
            "algorithm": "DBSCAN",
            "suitability": db_suit,
            "rationale": db_rat,
            "suggested_params": f"min_samples={max(3, n_features)}, calibrar eps con gráfica de k-distancias"
        })

        recommendations.append({
            "algorithm": "HDBSCAN",
            "suitability": "Recommended",
            "rationale": (
                "Supera a DBSCAN al tolerar densidades variables y seleccionar clusters estables "
                "automáticamente a lo largo de un árbol jerárquico."
            ),
            "suggested_params": f"min_cluster_size={max(5, int(n_samples * 0.05))}"
        })

        if n_features > 30 and isinstance(X, str) and X == "embed":
            for rec in recommendations:
                rec["rationale"] += " [Nota: Espacio de alta dimensión; se recomienda proyectar primero con PCA o UMAP]."

        return pd.DataFrame(recommendations)

    def cluster_kmeans(
        self,
        X: Union[str, pd.DataFrame, np.ndarray] = "embed",
        k: int = 3,
        n_init: int = 50,
        max_iter: int = 300,
        tol: float = 1e-4,
        random_state: Optional[int] = 42,
        name: str = "cluster_segment",
        verbose: bool = True,
        **kwargs
    ) -> ClusterSolution:
        """
        Ajusta una solución final de K-Means sobre el espacio vectorial indicado,
        vinculando las etiquetas al índice del dataset y calculando métricas geométricas.
        """
        # 1. Resolver el espacio vectorial
        if isinstance(X, str):
            if X in self.manager.registry:
                matrix = self.manager.registry[X]
                space_name = X
            else:
                raise ValueError(
                    f"El espacio '{X}' no existe en el registro. "
                    f"Espacios disponibles: {list(self.manager.registry.keys())}"
                )
        elif isinstance(X, pd.DataFrame):
            matrix = X
            space_name = "custom_dataframe"
        elif isinstance(X, np.ndarray):
            matrix = pd.DataFrame(X, index=self.data.index)
            space_name = "custom_array"
        else:
            matrix = self.data
            space_name = "prepared_data"

        # 2. Entrenar el modelo final
        model = KMeans(
            n_clusters=k,
            n_init=n_init,
            max_iter=max_iter,
            tol=tol,
            random_state=random_state,
            **kwargs
        )
        labels_arr = model.fit_predict(matrix)
        labels_series = pd.Series(labels_arr, index=matrix.index, name=name)

        # 3. Métricas geométricas de separación
        sil = float(silhouette_score(matrix, labels_arr))
        ch = float(calinski_harabasz_score(matrix, labels_arr))
        db = float(davies_bouldin_score(matrix, labels_arr))

        metrics = {
            "silhouette": round(sil, 4),
            "calinski_harabasz": round(ch, 2),
            "davies_bouldin": round(db, 4),
        }

        # 4. Construir datasets etiquetados respetando el contrato original
        data_with_clusters = self.raw_data.copy()
        data_with_clusters["cluster"] = labels_arr
        if name and name != "cluster":
            data_with_clusters[name] = labels_arr

        prep_with_clusters = self.data.copy()
        prep_with_clusters["cluster"] = labels_arr
        if name and name != "cluster":
            prep_with_clusters[name] = labels_arr

        solution = ClusterSolution({
            "name": name,
            "k": k,
            "space": space_name,
            "model": model,
            "labels": labels_series,
            "labels_array": labels_arr,
            "metrics": metrics,
            "features": matrix,
            "raw_data": self.raw_data,
            "prepared_data": self.data,
            "data_with_clusters": data_with_clusters,
            "prep_with_clusters": prep_with_clusters,
        })

        self.last_solution = solution

        if verbose:
            print("=" * 65)
            print(f"✓ K-MEANS SOLUTION FITTED: '{name}'")
            print("=" * 65)
            print(f"- Espacio vectorial:   {space_name} ({matrix.shape[1]} dimensiones)")
            print(f"- Número de clústeres: k = {k}")
            print(f"- Silhouette Score:    {sil:.4f}")
            print(f"- Davies-Bouldin:      {db:.4f}")
            print(f"- Calinski-Harabasz:   {ch:.2f}")
            print(f"- Distribución:        {dict(labels_series.value_counts().sort_index())}")
            print("=" * 65)

        return solution
    
    def build_experiment(
        self,
        spaces: List[str],
        models: Union[str, List[str]] = "recommended",
        k_range: range = range(2, 11),
        evaluate_stability: bool = True,
        n_stability_runs: int = 5,
        n_init: int = 50,
        max_iter: int = 500,
        tol: float = 1e-4,
        random_state: Optional[int] = None,
        master_seed: Optional[int] = None,
        verbose: bool = True,
        **kwargs
    ) -> ExperimentResult:
        """
        Construye el plan del experimento de clustering (Fase 1) y define el motor
        de la competición para ejecutar la parrilla de modelos completa en .run() (Fase 2).
        """
        from IPython.display import display

        # 1. Validación de dependencias opcionales
        kmedoids_avail = False
        kmedoids_driver = None
        try:
            from sklearn_extra.cluster import KMedoids  # type: ignore
            kmedoids_avail = True
            kmedoids_driver = "sklearn_extra"
        except ImportError:
            try:
                import kmedoids  # type: ignore
                kmedoids_avail = True
                kmedoids_driver = "kmedoids"
            except ImportError:
                pass

        hdbscan_avail = False
        hdbscan_cls = None
        try:
            from sklearn.cluster import HDBSCAN  # type: ignore
            hdbscan_avail = True
            hdbscan_cls = HDBSCAN
        except ImportError:
            try:
                import hdbscan  # type: ignore
                hdbscan_avail = True
                hdbscan_cls = hdbscan.HDBSCAN
            except ImportError:
                pass

        n_samples, n_features = self.data.shape

        # Detección de outliers
        num_cols = self.data.select_dtypes(include=[np.number])
        has_outliers = False
        for c in num_cols.columns:
            s = num_cols[c].dropna()
            iqr = s.quantile(0.75) - s.quantile(0.25)
            if iqr > 0 and ((s < s.quantile(0.25) - 1.5 * iqr) | (s > s.quantile(0.75) + 1.5 * iqr)).any():
                has_outliers = True
                break

        kmedoids_reason = "costo viable para este tamaño"
        if has_outliers:
            kmedoids_reason += "\n- se detectaron variables robust-scaled/outliers"

        dbscan_reason = "puede identificar observaciones aisladas"
        if n_samples >= 50:
            dbscan_reason += "\n- muestra suficiente para explorar densidad"

        plan_records = [
            {
                "step": 1,
                "role": "baseline",
                "model": "kmeans",
                "priority_label": "Exploratory",
                "available": True,
                "reason": "baseline interpretable",
                "experiment_purpose": "Establish a fast, interpretable centroid benchmark."
            },
            {
                "step": 2,
                "role": "robust alternative",
                "model": "kmedoids",
                "priority_label": "High" if has_outliers else "Recommended",
                "available": kmedoids_avail,
                "reason": kmedoids_reason,
                "experiment_purpose": "Test a medoid solution that is less sensitive to extreme observations."
            },
            {
                "step": 3,
                "role": "hierarchical alternative",
                "model": "agglomerative",
                "priority_label": "Recommended" if n_samples <= 2500 else "Conditional",
                "available": True,
                "reason": "dataset pequeño/mediano" if n_samples <= 2500 else "dataset grande (costo O(n^2))",
                "experiment_purpose": "Test whether a nested or linkage-based structure better represents the sample."
            },
            {
                "step": 4,
                "role": "probabilistic alternative",
                "model": "gaussian_mixture",
                "priority_label": "Exploratory",
                "available": True,
                "reason": "evalúa pertenencia probabilística",
                "experiment_purpose": "Test elliptical clusters and soft membership probabilities."
            },
            {
                "step": 5,
                "role": "density/noise diagnostic",
                "model": "dbscan",
                "priority_label": "Recommended" if n_features <= 15 else "Caution",
                "available": True,
                "reason": dbscan_reason,
                "experiment_purpose": "Test irregular shapes and whether isolated observations should be treated as noise."
            },
            {
                "step": 6,
                "role": "variable-density diagnostic",
                "model": "hdbscan",
                "priority_label": "Exploratory",
                "available": hdbscan_avail,
                "reason": "puede explorar densidades variables",
                "experiment_purpose": "Explore density-varying groups and noise without fixing k."
            }
        ]
        plan_df = pd.DataFrame(plan_records)

        if verbose:
            print("========== CLUSTERING EXPERIMENT PLAN ==========")
            with pd.option_context("display.max_colwidth", None):
                try:
                    display(plan_df.style.set_properties(**{"white-space": "pre-wrap", "text-align": "left"}))
                except Exception:
                    display(plan_df)
            print("Note: this is a test plan, not a model ranking. Use plan.run() for the competition.\n")

        # 2. Motor de ejecución del torneo completo (Fase 2)
        def _execute_runner() -> Dict[str, Any]:
            if "embed" not in self.manager.registry:
                self.manager.registry["embed"] = self.data

            valid_spaces = [s for s in spaces if s in self.manager.registry]
            if not valid_spaces:
                raise ValueError(f"Ninguno de los espacios {spaces} existe en el registro.")

            run_all_recommended = (models == "recommended")
            selected_models = [m.lower() for m in models] if isinstance(models, list) else ([models.lower()] if not run_all_recommended else [])

            base_seed = master_seed or random_state or 42
            raw_runs = []
            fitted_models: Dict[str, Any] = {}
            labels_by_run: Dict[str, np.ndarray] = {}

            for space_name in valid_spaces:
                X = self.manager.registry[space_name]
                n_rows = len(X)

                # --- A. K-MEANS ---
                if run_all_recommended or "kmeans" in selected_models:
                    for k in k_range:
                        params = {"k": k, "n_init": n_init, "max_iter": max_iter, "tol": tol, "random_state": base_seed}
                        m = KMeans(n_clusters=k, n_init=n_init, max_iter=max_iter, tol=tol, random_state=base_seed)
                        lbl = m.fit_predict(X)
                        
                        stab = 1.0
                        if evaluate_stability and n_stability_runs > 1:
                            sub_aris = []
                            for s in range(n_stability_runs):
                                m_s = KMeans(n_clusters=k, n_init=max(2, n_init // 5), max_iter=max_iter, random_state=base_seed + s * 17)
                                sub_aris.append(adjusted_rand_score(lbl, m_s.fit_predict(X)))
                            stab = float(np.mean(sub_aris))

                        k_key = f"{space_name}__kmeans__k{k}"
                        fitted_models[k_key] = m
                        labels_by_run[k_key] = lbl
                        raw_runs.append({"space": space_name, "model": "kmeans", "params": params, "labels": lbl, "stability_ari": stab, "model_obj": m, "run_key": k_key})

                # --- B. K-MEDOIDS ---
                if (run_all_recommended or "kmedoids" in selected_models) and kmedoids_avail:
                    for metric in ["euclidean", "manhattan"]:
                        for k in k_range:
                            params = {"k": k, "metric": metric, "max_iter": max_iter, "random_state": base_seed}
                            try:
                                if kmedoids_driver == "sklearn_extra":
                                    from sklearn_extra.cluster import KMedoids
                                    m = KMedoids(n_clusters=k, metric=metric, max_iter=max_iter, random_state=base_seed)
                                    lbl = m.fit_predict(X)
                                else:
                                    import kmedoids
                                    from scipy.spatial.distance import pdist, squareform
                                    D = squareform(pdist(X, metric=metric))
                                    res = kmedoids.fasterpam(D, k, random_state=base_seed)
                                    lbl = np.asarray(res.labels)
                                    m = res

                                stab = 1.0
                                if evaluate_stability and n_stability_runs > 1 and kmedoids_driver == "sklearn_extra":
                                    sub_aris = []
                                    for s in range(n_stability_runs):
                                        m_s = KMedoids(n_clusters=k, metric=metric, max_iter=max_iter, random_state=base_seed + s * 17)
                                        sub_aris.append(adjusted_rand_score(lbl, m_s.fit_predict(X)))
                                    stab = float(np.mean(sub_aris))

                                km_key = f"{space_name}__kmedoids__{metric}__k{k}"
                                fitted_models[km_key] = m
                                labels_by_run[km_key] = lbl
                                raw_runs.append({"space": space_name, "model": "kmedoids", "params": params, "labels": lbl, "stability_ari": stab, "model_obj": m, "run_key": km_key})
                            except Exception:
                                pass

                # --- C. AGGLOMERATIVE ---
                if run_all_recommended or "agglomerative" in selected_models:
                    for k in k_range:
                        params = {"k": k, "linkage": "ward", "metric": "euclidean"}
                        m = AgglomerativeClustering(n_clusters=k, linkage="ward")
                        lbl = m.fit_predict(X)

                        stab = 1.0
                        if evaluate_stability and n_stability_runs > 1:
                            rng = np.random.RandomState(base_seed)
                            sub_aris = []
                            for _ in range(n_stability_runs):
                                sub_idx = rng.choice(n_rows, size=int(0.85 * n_rows), replace=False)
                                m_s = AgglomerativeClustering(n_clusters=k, linkage="ward")
                                sub_aris.append(adjusted_rand_score(lbl[sub_idx], m_s.fit_predict(X.iloc[sub_idx])))
                            stab = float(np.mean(sub_aris)) if sub_aris else 1.0

                        agg_key = f"{space_name}__agglomerative__k{k}"
                        fitted_models[agg_key] = m
                        labels_by_run[agg_key] = lbl
                        raw_runs.append({"space": space_name, "model": "agglomerative", "params": params, "labels": lbl, "stability_ari": stab, "model_obj": m, "run_key": agg_key})

                # --- D. GAUSSIAN MIXTURE ---
                if run_all_recommended or "gaussian_mixture" in selected_models:
                    for cov in ["full", "diag"]:
                        for k in k_range:
                            params = {"k": k, "covariance_type": cov, "n_init": 10, "max_iter": max_iter, "random_state": base_seed}
                            m = GaussianMixture(n_components=k, covariance_type=cov, n_init=min(n_init, 10), max_iter=max_iter, random_state=base_seed)
                            lbl = m.fit_predict(X)

                            stab = 1.0
                            if evaluate_stability and n_stability_runs > 1:
                                sub_aris = []
                                for s in range(n_stability_runs):
                                    m_s = GaussianMixture(n_components=k, covariance_type=cov, n_init=1, max_iter=max_iter, random_state=base_seed + s * 17)
                                    sub_aris.append(adjusted_rand_score(lbl, m_s.fit_predict(X)))
                                stab = float(np.mean(sub_aris))

                            gmm_key = f"{space_name}__gaussian_mixture__{cov}__k{k}"
                            fitted_models[gmm_key] = m
                            labels_by_run[gmm_key] = lbl
                            raw_runs.append({"space": space_name, "model": "gaussian_mixture", "params": params, "labels": lbl, "stability_ari": stab, "model_obj": m, "run_key": gmm_key})

                # --- E. DBSCAN ---
                if run_all_recommended or "dbscan" in selected_models:
                    for ms in [4, 5, 10]:
                        if ms < n_rows:
                            nn = NearestNeighbors(n_neighbors=ms).fit(X)
                            dists, _ = nn.kneighbors(X)
                            k_d = np.sort(dists[:, -1])
                            eps_list = np.unique([float(np.percentile(k_d, p)) for p in [70, 80, 85, 90, 95]])
                            for eps_val in eps_list:
                                if eps_val <= 0:
                                    continue
                                params = {"eps": eps_val, "min_samples": ms, "metric": "euclidean"}
                                m = DBSCAN(eps=eps_val, min_samples=ms)
                                lbl = m.fit_predict(X)
                                db_key = f"{space_name}__dbscan__ms{ms}__eps{round(eps_val, 3)}"
                                fitted_models[db_key] = m
                                labels_by_run[db_key] = lbl
                                raw_runs.append({"space": space_name, "model": "dbscan", "params": params, "labels": lbl, "stability_ari": 1.0, "model_obj": m, "run_key": db_key})

                # --- F. HDBSCAN ---
                if (run_all_recommended or "hdbscan" in selected_models) and hdbscan_cls is not None:
                    for mcs in [5, 10]:
                        if mcs < n_rows:
                            params = {"min_cluster_size": mcs, "min_samples": None}
                            try:
                                m = hdbscan_cls(min_cluster_size=mcs)
                                lbl = m.fit_predict(X)
                                hdb_key = f"{space_name}__hdbscan__mcs{mcs}"
                                fitted_models[hdb_key] = m
                                labels_by_run[hdb_key] = lbl
                                raw_runs.append({"space": space_name, "model": "hdbscan", "params": params, "labels": lbl, "stability_ari": 1.0, "model_obj": m, "run_key": hdb_key})
                            except Exception:
                                pass

            # 3. Consolidar métricas geométricas
            processed_runs = []
            for r in raw_runs:
                lbl = np.asarray(r["labels"])
                noise_mask = (lbl == -1)
                noise_pct = float((noise_mask.sum() / len(lbl)) * 100)
                clean_lbl = lbl[~noise_mask]
                unique_cl = np.unique(clean_lbl)
                n_clusters = len(unique_cl)

                if n_clusters < 2:
                    continue

                clean_X = self.manager.registry[r["space"]][~noise_mask]
                sil = float(silhouette_score(clean_X, clean_lbl))
                ch = float(calinski_harabasz_score(clean_X, clean_lbl))
                db = float(davies_bouldin_score(clean_X, clean_lbl))

                counts = pd.Series(clean_lbl).value_counts()
                min_cl_pct = float((counts.min() / len(lbl)) * 100)

                processed_runs.append({
                    "space": r["space"],
                    "model": r["model"],
                    "params": r["params"],
                    "n_clusters": int(n_clusters),
                    "silhouette": round(sil, 6),
                    "calinski_harabasz": round(ch, 6),
                    "davies_bouldin": round(db, 6),
                    "stability_ari": round(float(r["stability_ari"]), 6),
                    "noise_pct": round(noise_pct, 6),
                    "min_cluster_pct": round(min_cl_pct, 6),
                    "run_key": r["run_key"]
                })

            leaderboard_df = pd.DataFrame(processed_runs)
            if not leaderboard_df.empty:
                # Ranking ponderado multi-métrica
                r_sil = leaderboard_df["silhouette"].rank(ascending=False, method="min")
                r_ch = leaderboard_df["calinski_harabasz"].rank(ascending=False, method="min")
                r_db = leaderboard_df["davies_bouldin"].rank(ascending=True, method="min")
                r_stab = leaderboard_df["stability_ari"].rank(ascending=False, method="min")
                r_noise = leaderboard_df["noise_pct"].rank(ascending=True, method="min")

                leaderboard_df["competition_score"] = ((r_sil + r_ch + r_db + r_stab + r_noise) / 5.0).round(1)
                leaderboard_df = leaderboard_df.sort_values(
                    by=["competition_score", "silhouette"],
                    ascending=[True, False]
                ).reset_index(drop=True)

                leaderboard_df.insert(0, "rank", leaderboard_df.index + 1)
                best_key = leaderboard_df.iloc[0]["run_key"]
            else:
                best_key = None

            return {
                "leaderboard": leaderboard_df,
                "models": fitted_models,
                "labels": labels_by_run,
                "best_config": leaderboard_df.iloc[0].to_dict() if not leaderboard_df.empty else {},
                "best_model": fitted_models.get(best_key),
                "spaces": {s: self.manager.registry[s] for s in valid_spaces}
            }

        result = ExperimentResult(runner_fn=_execute_runner, plan_df=plan_df)
        self.last_experiment = result
        return result