"""
embeddings.py
=============
Dimensionality reduction engine supporting PCA, t-SNE, and UMAP with graceful fallbacks.
"""

from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# Optional dependency check for UMAP
try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False


class EmbeddingManager:
    """Computes and registers 2D/3D representations of feature spaces."""

    def __init__(self, data: pd.DataFrame):
        self.data = data.copy()
        self.registry: Dict[str, pd.DataFrame] = {
            "feature_space": self.data.copy(),
            "embed": self.data.copy()
        }
        self.models: Dict[str, Any] = {}
        self.catalog: list = []

    def compute_pca(self, target_variance: float = 0.90) -> None:
        """Generates a 2D visualization PCA and a higher-dimensional PCA retaining target variance."""
        # 1. Visual 2D PCA
        pca_vis = PCA(n_components=2)
        vis_coords = pca_vis.fit_transform(self.data)
        self.registry["pca_visual"] = pd.DataFrame(
            vis_coords, columns=["PC1", "PC2"], index=self.data.index
        )
        self.models["pca_visual"] = pca_vis
        self.catalog.append({
            "embedding": "pca_visual",
            "method": "pca",
            "n_components": 2,
            "variance_explained": round(float(np.sum(pca_vis.explained_variance_ratio_)), 6),
            "parameters": "svd_solver=auto, whiten=False",
            "purpose": "visualization",
            "status": "created"
        })

        # 2. Model PCA (Clustering candidate)
        pca_mod = PCA(n_components=target_variance, svd_solver="full")
        mod_coords = pca_mod.fit_transform(self.data)
        n_comp = mod_coords.shape[1]
        col_names = [f"PC{i+1}" for i in range(n_comp)]
        self.registry["pca_model"] = pd.DataFrame(
            mod_coords, columns=col_names, index=self.data.index
        )
        self.models["pca_model"] = pca_mod
        self.catalog.append({
            "embedding": "pca_model",
            "method": "pca",
            "n_components": n_comp,
            "variance_explained": round(float(np.sum(pca_mod.explained_variance_ratio_)), 6),
            "parameters": f"target_variance={target_variance}",
            "purpose": "clustering candidate",
            "status": "created"
        })

    def compute_umap(self, n_neighbors: int = 15, min_dist: float = 0.1, metric: str = "euclidean") -> None:
        """Generates 2D UMAP projection if umap-learn is installed."""
        if not UMAP_AVAILABLE:
            self.catalog.append({
                "embedding": "umap_visual",
                "method": "umap",
                "n_components": 2,
                "variance_explained": np.nan,
                "parameters": f"n_neighbors={n_neighbors}, min_dist={min_dist}, metric={metric}",
                "purpose": "visualization / local structure",
                "status": "skipped: Falta umap-learn. Instala: pip install umap-learn"
            })
            return

        reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, metric=metric, n_components=2)
        umap_coords = reducer.fit_transform(self.data)
        self.registry["umap_visual"] = pd.DataFrame(
            umap_coords, columns=["UMAP1", "UMAP2"], index=self.data.index
        )
        self.models["umap_visual"] = reducer
        self.catalog.append({
            "embedding": "umap_visual",
            "method": "umap",
            "n_components": 2,
            "variance_explained": np.nan,
            "parameters": f"n_neighbors={n_neighbors}, min_dist={min_dist}, metric={metric}",
            "purpose": "visualization / local structure",
            "status": "created"
        })

    def compute_tsne(
            self,
            n_components: int = 2,
            perplexity: float = 30.0,
            learning_rate: Any = "auto",
            **kwargs
        ) -> pd.DataFrame:
            """Computes t-SNE projection supporting arbitrary sklearn parameters."""
            tsne_params = {
                "n_components": n_components,
                "perplexity": perplexity,
                "learning_rate": learning_rate,
                "random_state": getattr(self, "random_state", 42),
            }
            # Integra init, metric, max_iter, etc.
            tsne_params.update(kwargs)

            tsne = TSNE(**tsne_params)
            embedding = tsne.fit_transform(self.data)

            cols = [f"TSNE_{i+1}" for i in range(n_components)]
            emb_df = pd.DataFrame(embedding, columns=cols, index=self.data.index)
            
            self.registry["tsne"] = emb_df
            return emb_df

    def get_catalog_df(self) -> pd.DataFrame:
        return pd.DataFrame(self.catalog)