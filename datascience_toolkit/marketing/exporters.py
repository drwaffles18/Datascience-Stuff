"""
exporters.py
============
Audience segment export utilities for downstream ad tech activation,
CRM sync, and Athena/SQL query generation.
"""

from typing import List, Optional
from pathlib import Path
import pandas as pd


def export_cluster_ids(
    df: pd.DataFrame,
    id_col: str,
    cluster_col: str,
    target_cluster: any,
    output_path: Optional[str] = None
) -> pd.Series:
    """Filters unique IDs belonging to a target cluster and optionally saves to CSV/Parquet."""
    subset = df[df[cluster_col] == target_cluster][id_col].dropna().drop_duplicates()

    if output_path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix == ".parquet":
            subset.to_frame().to_parquet(path, index=False)
        else:
            subset.to_csv(path, index=False, header=[id_col])
        print(f"Exported {len(subset):,} IDs to {path}")

    return subset


def generate_sql_in_clause(
    df: pd.DataFrame,
    id_col: str,
    cluster_col: str,
    target_cluster: any,
    max_ids: Optional[int] = 1000
) -> str:
    """Generates an SQL 'WHERE id IN (...)' clause for quick Athena or database lookups."""
    ids = export_cluster_ids(df, id_col, cluster_col, target_cluster)
    if max_ids:
        ids = ids.head(max_ids)

    formatted_ids = ", ".join([f"'{str(i)}'" for i in ids])
    return f"{id_col} IN ({formatted_ids})"