from .classification import ClassificationTournament
from .regression import RegressionTournament

# Alias para compatibilidad hacia atrás con los notebooks anteriores
SupervisedClassification = ClassificationTournament

__all__ = [
    "ClassificationTournament",
    "SupervisedClassification",
    "RegressionTournament",
]