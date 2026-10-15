"""
Evaluation of the models: predictive performance, clustering quality, size of the trees and
fairness, each as a dictionary of metrics.
"""
from .classification import evaluate_classification
from .clustering import evaluate_clustering, evaluate_clustering_internal
from .fairness import evaluate_fairness
from .interpretability import evaluate_interpretability
from .regression import evaluate_regression

__all__ = ["evaluate_classification", "evaluate_regression", "evaluate_clustering", "evaluate_clustering_internal",
           "evaluate_interpretability", "evaluate_fairness"]
