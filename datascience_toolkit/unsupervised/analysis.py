"""
analysis.py
===========
Orchestrator class for unsupervised data exploration, dimensionality reduction,
and multi-space clustering tournaments.
"""

from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
from .embeddings import EmbeddingManager
from .competition import SpaceCompetition


class UnsupervisedAnalysis:
    """Unified API for unsupervised clustering workflows."""

    def __init__(self, data: pd.DataFrame, validation_label: Optional[pd.Series] = None):
        self.data = data.copy()
        self.validation_label = validation_label
        self.manager = EmbeddingManager(self.data)
        self.last_competition: Optional[Dict[str, Any]] = None

    def explore(
        self,
        methods: Tuple[str, ...] = ("pca",),
        pca_model_variance: float = 0.90,
        umap_params: Optional[dict] = None,
        tsne_params: Optional[dict] = None,
        plot: bool = False
    ) -> Dict[str, Any]:
        """Calculates dimensional representations (PCA, UMAP, t-SNE) and catalogs them."""
        if "pca" in methods:
            self.manager.compute_pca(target_variance=pca_model_variance)

        if "umap" in methods:
            params = umap_params or {"n_neighbors": 15, "min_dist": 0.1, "metric": "euclidean"}
            self.manager.compute_umap(**params)

        if "tsne" in methods:
            params = tsne_params or {"perplexity": 30.0, "learning_rate": "auto"}
            self.manager.compute_tsne(**params)

        report_df = self.manager.get_catalog_df()
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
        random_state: Optional[int] = None
    ) -> Dict[str, Any]:
        """Runs clustering benchmarks across specified vector spaces."""
        comp_output = SpaceCompetition.evaluate(
            spaces=self.manager.registry,
            requested_spaces=spaces,
            k_range=k_range,
            validation_label=self.validation_label,
            n_init=n_init,
            max_iter=max_iter,
            random_state=random_state
        )
        self.last_competition = comp_output

        # Console print matching your previous notebook logs
        print("\n========== SPACE COMPARISON INTERPRETATION ==========")
        for s in comp_output["interpretation"]:
            print(f"- {s}")

        return comp_output