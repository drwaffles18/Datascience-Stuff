"""
DataScience_Class.py
====================
DEPRECATION NOTICE: This monolithic file is deprecated.
Use the modular 'datascience_toolkit' package instead:
    from datascience_toolkit import DFPreprocess, UnsupervisedAnalysis, etc.
"""

import warnings
from datascience_toolkit import (
    DFPreprocess,
    sanitize_athena_na,
    standardize_types,
    EDAQuick,
    UnsupervisedAnalysis,
    AudienceProfiler,
    export_cluster_ids,
    generate_sql_in_clause,
    ClassificationTournament,
    RegressionTournament,
    TimeSeriesDiagnostics,
    TimeSeriesFeatureExtractor,
    TimeSeriesTournament,
    plot_radar_comparison,
    plot_embedding_clusters,
)

warnings.warn(
    "Importing from 'DataScience_Class' is deprecated and will be removed. "
    "Please update your imports to 'from datascience_toolkit import ...'",
    DeprecationWarning,
    stacklevel=2,
)