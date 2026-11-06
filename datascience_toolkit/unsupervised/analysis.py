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


class ExperimentResult(dict):
    """
    Contenedor de resultados del torneo de clustering compatible con acceso dict y atributo.
    Proporciona métodos estratégicos para generar el leaderboard, plan de decisión,
    explicación diagnóstica y extracción de artefactos.
    """
    def __getattr__(self, name: str) -> Any:
        if name in self:
            return self[name]
        raise AttributeError(f"'ExperimentResult' no contiene el atributo '{name}'")

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value

    def leaderboard(self, top_n: Optional[int] = None) -> pd.DataFrame:
        """Retorna el DataFrame del leaderboard ordenado por métrica."""
        lb = dict.get(self, "leaderboard", pd.DataFrame())
        if top_n is not None and isinstance(lb, pd.DataFrame):
            return lb.head(top_n)
        return lb

    def run(self, verbose: bool = True) -> "ExperimentResult":
        """
        Ejecuta y consolida la competición de clustering retornando el objeto
        con todos los modelos entrenados, métricas y ranking final.
        """
        if verbose:
            print("=" * 65)
            print("✓ CLUSTERING COMPETITION COMPLETED")
            print("=" * 65)
            lb = self.leaderboard()
            if not lb.empty:
                best = lb.iloc[0]
                print(f"- Configuración campeona: {best['space']} | {best['model'].upper()} | k={int(best['k'])}")
                print(f"- Silhouette Score:       {best['silhouette']:.4f}")
                print(f"- Davies-Bouldin:         {best['davies_bouldin']:.4f}")
                if "stability" in best:
                    print(f"- Estabilidad (ARI):      {best['stability']:.4f}")
            print("=" * 65)
        return self

    def plan(self, top_n: int = 5) -> pd.DataFrame:
        """Genera un plan de decisión comparativo y accionable."""
        lb = self.leaderboard().copy()
        if lb.empty:
            return pd.DataFrame()

        has_stability = "stability" in lb.columns
        candidates = []
        seen_keys = set()

        def add_candidate(row, role, rationale, next_step):
            key = (row["space"], row["model"], int(row["k"]))
            if key not in seen_keys:
                seen_keys.add(key)
                rec = {
                    "role": role,
                    "space": row["space"],
                    "model": row["model"],
                    "k": int(row["k"]),
                    "silhouette": row["silhouette"],
                    "davies_bouldin": row["davies_bouldin"],
                }
                if has_stability:
                    rec["stability"] = row.get("stability", np.nan)
                rec["strategic_rationale"] = rationale
                rec["next_action"] = next_step
                candidates.append(rec)

        best = lb.iloc[0]
        add_candidate(
            best,
            "Primary Champion",
            f"Mayor puntuación global de silueta ({best['silhouette']:.4f}). Estructura geométrica más nítida.",
            "Proceder a perfilamiento de arquetipos y análisis de medias por clúster."
        )

        if has_stability and "stability" in lb.columns:
            best_stab = lb.sort_values(by="stability", ascending=False).iloc[0]
            add_candidate(
                best_stab,
                "Stability Champion",
                f"Máxima consistencia en remuestreo/semillas (ARI: {best_stab['stability']:.4f}). Clústeres reproducibles.",
                "Usar si la reproducibilidad en producción es prioritaria sobre la separación pura."
            )

        best_db = lb.sort_values(by="davies_bouldin", ascending=True).iloc[0]
        add_candidate(
            best_db,
            "Compactness Champion",
            f"Mínima dispersión intra-clúster y máxima separación inter-centroide (DB: {best_db['davies_bouldin']:.4f}).",
            "Examinar fronteras de decisión y dispersión de outliers."
        )

        low_k = lb[lb["k"].isin([2, 3])]
        if not low_k.empty:
            best_low = low_k.sort_values(by="silhouette", ascending=False).iloc[0]
            add_candidate(
                best_low,
                "Parsimony Pick (Low k)",
                f"Configuración de baja complejidad (k={int(best_low['k'])}) con silueta {best_low['silhouette']:.4f}.",
                "Ideal si el negocio requiere segmentación comercial simple y altamente accionable."
            )

        primary_space = best["space"]
        alt_spaces = lb[lb["space"] != primary_space]
        if not alt_spaces.empty:
            best_alt = alt_spaces.sort_values(by="silhouette", ascending=False).iloc[0]
            add_candidate(
                best_alt,
                f"Alt Space ({best_alt['space']})",
                f"Mejor opción en espacio alternativo con silueta {best_alt['silhouette']:.4f}.",
                "Comparar solapamiento de asignaciones con el campeón primario (ARI cruzado)."
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
        lb = self.leaderboard()
        if lb.empty:
            print("No hay resultados registrados en el experimento para explicar.")
            return

        best = lb.iloc[0]
        has_stability = "stability" in lb.columns

        print("=" * 75)
        print("📊 CLUSTERING EXPERIMENT DIAGNOSIS & INTERPRETATION")
        print("=" * 75)
        print("\n1. CONFIGURACIÓN GANADORA (CHAMPION)")
        print(f"   - Espacio vectorial:   {best['space']}")
        print(f"   - Familia de modelo:   {best['model'].upper()}")
        print(f"   - Número de clusters:  k = {int(best['k'])}")
        print(f"   - Silhouette Score:    {best['silhouette']:.4f} (Separación global)")
        print(f"   - Davies-Bouldin:      {best['davies_bouldin']:.4f} (Compacidad)")
        if has_stability:
            print(f"   - Estabilidad (ARI):   {best['stability']:.4f}")

        print("\n2. IMPACTO DEL ESPACIO DE REPRESENTACIÓN")
        space_summary = lb.groupby("space")["silhouette"].agg(["mean", "max"]).round(4)
        for space, row in space_summary.iterrows():
            print(f"   - '{space}': Silueta promedio = {row['mean']:.4f} | Silueta máxima = {row['max']:.4f}")

        if len(space_summary) > 1:
            best_space = space_summary["max"].idxmax()
            other_spaces = [s for s in space_summary.index if s != best_space]
            print(f"   → El espacio '{best_space}' superó a {other_spaces}, confirmando que la reducción")
            print("     dimensional filtró ruido y mejoró el contraste de distancias.")

        print("\n3. DINÁMICA POR FAMILIA DE ALGORITMOS")
        model_summary = lb.groupby("model")["silhouette"].agg(["mean", "max"]).round(4)
        for model_name, row in model_summary.iterrows():
            print(f"   - {model_name.upper():<14}: Silueta promedio = {row['mean']:.4f} | Máxima = {row['max']:.4f}")

        if has_stability:
            print("\n4. EVALUACIÓN DE ESTABILIDAD Y REPRODUCIBILIDAD")
            top_stab = lb.sort_values(by="stability", ascending=False).iloc[0]
            print(f"   - Configuración más estable: {top_stab['space']} | {top_stab['model']} | k={int(top_stab['k'])} (ARI={top_stab['stability']:.4f})")

        print("\n5. RECOMENDACIONES TÁCTICAS")
        print(f"   - Adoptar '{best['model']}' con k={int(best['k'])} en el espacio '{best['space']}' como segmentación base.")
        print("   - Realizar perfilamiento de variables continuas y categóricas para validar la interpretación cualitativa.")
        print("=" * 75)

    def get_labels(
        self,
        space: Optional[str] = None,
        model: Optional[str] = None,
        k: Optional[int] = None
    ) -> np.ndarray:
        """Retorna las etiquetas de clúster para una configuración dada o el campeón."""
        if space is None or model is None or k is None:
            best = self.get("best_config", {})
            space = space or best.get("space")
            model = model or best.get("model")
            k = k or best.get("k")

        run_key = f"{space}__{model}__k{int(k)}"
        labels_dict = self.get("labels", {})
        if run_key in labels_dict:
            return labels_dict[run_key]
        raise KeyError(f"No se encontraron etiquetas para '{run_key}'.")

    def get_model(
        self,
        space: Optional[str] = None,
        model: Optional[str] = None,
        k: Optional[int] = None
    ) -> Any:
        """Retorna el modelo ajustado para una configuración dada o el campeón."""
        if space is None or model is None or k is None:
            return self.get("best_model")
        run_key = f"{space}__{model}__k{int(k)}"
        models_dict = self.get("models", {})
        if run_key in models_dict:
            return models_dict[run_key]
        raise KeyError(f"No se encontró el modelo para '{run_key}'.")


class ClusterSolution(dict):
    """
    Contenedor de la solución final de clustering ajustada.
    Permite acceso transparente como atributo y diccionario, e incorpora
    métodos rápidos de perfilamiento y conteo.
    """
    def __getattr__(self, name: str) -> Any:
        if name in self:
            return self[name]
        raise AttributeError(f"'ClusterSolution' no contiene el atributo '{name}'")

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value

    def summary(self) -> pd.DataFrame:
        """Retorna la distribución de frecuencias y porcentajes por clúster."""
        labels = self.get("labels", pd.Series())
        counts = labels.value_counts().sort_index()
        pcts = (labels.value_counts(normalize=True).sort_index() * 100).round(2)
        return pd.DataFrame({"count": counts, "percentage": pcts})

    def profile(self, df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Genera el perfil medio de las variables numéricas agrupadas por clúster."""
        target_df = df if df is not None else self.get("raw_data", self.get("prepared_data"))
        if target_df is None:
            return pd.DataFrame()
        num_df = target_df.select_dtypes(include=[np.number]).copy()
        num_df["cluster"] = self["labels"]
        return num_df.groupby("cluster").mean().round(3)

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

        # Actualizar el gestor de embeddings con la matriz transformada
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
        """Cataloga los modelos de clustering soportados y valida dinamicamente dependencias."""
        hdbscan_available = False
        try:
            from sklearn.cluster import HDBSCAN  # type: ignore
            hdbscan_available = True
        except ImportError:
            try:
                import hdbscan  # type: ignore
                hdbscan_available = True
            except ImportError:
                hdbscan_available = False

        models_catalog = [
            {
                "model": "kmeans",
                "algorithm": "K-Means",
                "family": "Centroid-based",
                "status": "Available",
                "notes": "Incluido en scikit-learn. Adecuado para grupos esfericos e hiper-esfericos."
            },
            {
                "model": "hierarchical",
                "algorithm": "Agglomerative Clustering",
                "family": "Hierarchical",
                "status": "Available",
                "notes": "Incluido en scikit-learn. Soporta enlaces 'ward', 'complete' y 'average'."
            },
            {
                "model": "gmm",
                "algorithm": "Gaussian Mixture Models",
                "family": "Probabilistic / Density",
                "status": "Available",
                "notes": "Incluido en scikit-learn. Clustering probabilistico suave (soft clustering)."
            },
            {
                "model": "dbscan",
                "algorithm": "DBSCAN",
                "family": "Density-based",
                "status": "Available",
                "notes": "Incluido en scikit-learn. Detecta ruido y clústeres de forma arbitraria."
            },
            {
                "model": "hdbscan",
                "algorithm": "HDBSCAN",
                "family": "Hierarchical Density-based",
                "status": "Available" if hdbscan_available else "Missing dependency",
                "notes": "Listo para usarse." if hdbscan_available else "Requiere instalacion: ejecuta '%pip install hdbscan'."
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

        # 4. Construir datasets etiquetados
        data_with_clusters = self.raw_data.copy()
        data_with_clusters[name] = labels_series

        prep_with_clusters = self.data.copy()
        prep_with_clusters[name] = labels_series

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
        k_range: range = range(2, 8),
        evaluate_stability: bool = False,
        n_stability_runs: int = 5,
        n_init: int = 50,
        max_iter: int = 500,
        tol: float = 1e-4,
        random_state: Optional[int] = None,
        master_seed: Optional[int] = None,
        **kwargs
    ) -> ExperimentResult:
        """
        Ejecuta un torneo multiespacio y multimodelo evaluando separación geométrica
        y estabilidad de agrupamiento.
        """
        if "embed" not in self.manager.registry:
            self.manager.registry["embed"] = self.data

        valid_spaces = [s for s in spaces if s in self.manager.registry]
        if not valid_spaces:
            raise ValueError(
                f"Ninguno de los espacios {spaces} existe en el registro. "
                f"Disponibles: {list(self.manager.registry.keys())}"
            )

        if models == "recommended":
            target_models = ["kmeans", "hierarchical", "gmm"]
        elif isinstance(models, str):
            target_models = [models.lower()]
        else:
            target_models = [m.lower() for m in models]

        base_seed = master_seed or random_state or 42
        records = []
        fitted_models: Dict[str, Any] = {}
        labels_by_run: Dict[str, np.ndarray] = {}

        for space_name in valid_spaces:
            X = self.manager.registry[space_name]

            for model_name in target_models:
                for k in k_range:
                    run_key = f"{space_name}__{model_name}__k{k}"

                    if model_name in ("kmeans", "k-means"):
                        model = KMeans(
                            n_clusters=k,
                            n_init=n_init,
                            max_iter=max_iter,
                            tol=tol,
                            random_state=base_seed,
                            **kwargs
                        )
                        labels = model.fit_predict(X)

                    elif model_name in ("hierarchical", "agglomerative"):
                        linkage = kwargs.get("linkage", "ward")
                        model = AgglomerativeClustering(
                            n_clusters=k,
                            linkage=linkage
                        )
                        labels = model.fit_predict(X)

                    elif model_name in ("gmm", "gaussian_mixture"):
                        model = GaussianMixture(
                            n_components=k,
                            n_init=min(n_init, 10),
                            max_iter=max_iter,
                            tol=tol,
                            random_state=base_seed,
                            **kwargs
                        )
                        labels = model.fit_predict(X)

                    else:
                        continue

                    sil = float(silhouette_score(X, labels))
                    ch = float(calinski_harabasz_score(X, labels))
                    db = float(davies_bouldin_score(X, labels))

                    row = {
                        "space": space_name,
                        "model": model_name,
                        "k": k,
                        "silhouette": round(sil, 4),
                        "calinski_harabasz": round(ch, 2),
                        "davies_bouldin": round(db, 4),
                    }

                    if evaluate_stability and n_stability_runs > 1:
                        seed_runs = []
                        for s_idx in range(n_stability_runs):
                            current_seed = base_seed + (s_idx * 17)
                            if model_name in ("kmeans", "k-means"):
                                m_stab = KMeans(
                                    n_clusters=k,
                                    n_init=max(2, n_init // 5),
                                    max_iter=max_iter,
                                    tol=tol,
                                    random_state=current_seed
                                )
                                seed_runs.append(m_stab.fit_predict(X))
                            elif model_name in ("gmm", "gaussian_mixture"):
                                m_stab = GaussianMixture(
                                    n_components=k,
                                    n_init=1,
                                    max_iter=max_iter,
                                    tol=tol,
                                    random_state=current_seed
                                )
                                seed_runs.append(m_stab.fit_predict(X))
                            elif model_name in ("hierarchical", "agglomerative"):
                                rng = np.random.RandomState(current_seed)
                                sub_idx = rng.choice(len(X), size=int(0.9 * len(X)), replace=False)
                                m_stab = AgglomerativeClustering(n_clusters=k, linkage=kwargs.get("linkage", "ward"))
                                sub_lbl = m_stab.fit_predict(X.iloc[sub_idx])
                                seed_runs.append((sub_idx, sub_lbl))

                        if model_name in ("hierarchical", "agglomerative"):
                            ari_scores = []
                            for (idx_a, lbl_a), (idx_b, lbl_b) in itertools.combinations(seed_runs, 2):
                                common = np.intersect1d(idx_a, idx_b)
                                if len(common) > 0:
                                    pos_a = [np.where(idx_a == c)[0][0] for c in common]
                                    pos_b = [np.where(idx_b == c)[0][0] for c in common]
                                    ari_scores.append(adjusted_rand_score(lbl_a[pos_a], lbl_b[pos_b]))
                            row["stability"] = round(float(np.mean(ari_scores)), 4) if ari_scores else 1.0
                        else:
                            ari_scores = [
                                adjusted_rand_score(seed_runs[i], seed_runs[j])
                                for i, j in itertools.combinations(range(len(seed_runs)), 2)
                            ]
                            row["stability"] = round(float(np.mean(ari_scores)), 4) if ari_scores else 1.0

                    if self.validation_label is not None:
                        val_idx = self.validation_label.dropna().index.intersection(X.index)
                        if len(val_idx) > 0:
                            y_true = self.validation_label.loc[val_idx]
                            y_pred = labels[X.index.get_indexer(val_idx)]
                            row["ari"] = round(float(adjusted_rand_score(y_true, y_pred)), 4)
                            row["nmi"] = round(float(normalized_mutual_info_score(y_true, y_pred)), 4)

                    records.append(row)
                    fitted_models[run_key] = model
                    labels_by_run[run_key] = labels

        leaderboard_df = pd.DataFrame(records)
        if not leaderboard_df.empty:
            leaderboard_df = leaderboard_df.sort_values(by="silhouette", ascending=False).reset_index(drop=True)

        best_key = (
            f"{leaderboard_df.iloc[0]['space']}__{leaderboard_df.iloc[0]['model']}__k{int(leaderboard_df.iloc[0]['k'])}"
            if not leaderboard_df.empty else None
        )

        result = ExperimentResult({
            "leaderboard": leaderboard_df,
            "models": fitted_models,
            "labels": labels_by_run,
            "best_config": leaderboard_df.iloc[0].to_dict() if not leaderboard_df.empty else {},
            "best_model": fitted_models.get(best_key),
            "spaces": {s: self.manager.registry[s] for s in valid_spaces}
        })

        self.last_experiment = result

        print("\n" + "=" * 60)
        print("✓ EXPERIMENT LEADERBOARD GENERATED")
        print("=" * 60)
        if not leaderboard_df.empty:
            best = leaderboard_df.iloc[0]
            print(f"- Mejor combinación global: Espacio '{best['space']}' | Modelo '{best['model']}' | k={int(best['k'])}")
            print(f"- Silhouette Score: {best['silhouette']:.4f} | Davies-Bouldin: {best['davies_bouldin']:.4f}")
            if "stability" in best:
                print(f"- Stability Score (ARI): {best['stability']:.4f}")
        print("=" * 60)

        return result