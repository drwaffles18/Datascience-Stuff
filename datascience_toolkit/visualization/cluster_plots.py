"""
cluster_plots.py
================
Visual diagnostic tools for clustering and dimensionality reduction.
"""

from typing import List, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_radar_comparison(
    df: pd.DataFrame,
    cluster_col: str,
    features: List[str],
    title: str = "Cluster Profiles (Radar Chart)"
) -> None:
    """
    Renders a radar chart comparing the mean values of features across clusters.
    Expects scaled or standardized numerical features for balanced visualization.
    """
    means = df.groupby(cluster_col)[features].mean()
    categories = list(means.columns)
    n_vars = len(categories)

    angles = [n / float(n_vars) * 2 * np.pi for n in range(n_vars)]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    plt.xticks(angles[:-1], categories, color="grey", size=10)
    ax.set_rlabel_position(0)

    for cluster_id, row in means.iterrows():
        values = row.values.flatten().tolist()
        values += values[:1]
        ax.plot(angles, values, linewidth=2, linestyle="solid", label=f"Cluster {cluster_id}")
        ax.fill(angles, values, alpha=0.15)

    plt.title(title, size=14, y=1.08)
    plt.legend(loc="upper right", bbox_to_anchor=(0.1, 0.1))
    plt.tight_layout()
    plt.show()


def plot_embedding_clusters(
    embedding_df: pd.DataFrame,
    labels: pd.Series,
    title: str = "Cluster Distribution",
    x_col: Optional[str] = None,
    y_col: Optional[str] = None
) -> None:
    """
    Plots a 2D scatter plot colored by cluster assignment.
    """
    x = x_col or embedding_df.columns[0]
    y = y_col or embedding_df.columns[1]

    plt.figure(figsize=(9, 6))
    scatter = plt.scatter(
        embedding_df[x],
        embedding_df[y],
        c=labels,
        cmap="tab10",
        alpha=0.6,
        edgecolors="w",
        s=30
    )
    plt.colorbar(scatter, label="Cluster")
    plt.xlabel(x)
    plt.ylabel(y)
    plt.title(title)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.show()