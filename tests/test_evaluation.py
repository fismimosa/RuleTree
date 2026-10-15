import numpy as np
import pytest
import sklearn.metrics as skm
from sklearn.tree import DecisionTreeClassifier

from RuleTree import RuleTreeClassifier
from RuleTree.evaluation import (evaluate_classification, evaluate_clustering, evaluate_clustering_internal,
                                 evaluate_fairness, evaluate_interpretability, evaluate_regression)


def test_classification_binary_with_string_labels(iris):
    X, y = iris
    y = np.where(y == 2, "virginica", "other")
    model = RuleTreeClassifier(max_depth=2, random_state=0).fit(X, y)

    scores = evaluate_classification(y, model.predict(X), model.predict_proba(X), labels=model.classes_)

    assert scores["f1_score"] == skm.f1_score(y, model.predict(X), pos_label="virginica")
    assert scores["roc_macro"] == skm.roc_auc_score(y == "virginica", model.predict_proba(X)[:, 1])


def test_classification_multiclass(iris):
    X, y = iris
    model = RuleTreeClassifier(max_depth=2, random_state=0).fit(X, y)
    proba = model.predict_proba(X)

    scores = evaluate_classification(y, model.predict(X), proba)

    assert scores["f1_score"] == scores["f1_micro"]
    assert scores["roc_weighted"] == pytest.approx(skm.roc_auc_score(y, proba, multi_class="ovr", average="weighted"))
    assert "roc_macro" not in evaluate_classification(y, model.predict(X))


def test_regression_with_negative_values():
    scores = evaluate_regression(np.array([-1.0, 0.0, 2.0]), np.array([-0.5, 0.5, 2.0]))

    assert np.isnan(scores["mean_squared_log_error"])
    assert scores["max_error"] == 0.5


def test_clustering_metrics(blobs):
    X, y = blobs

    assert evaluate_clustering(y, y)["adjusted_rand"] == 1.0
    assert np.isnan(evaluate_clustering_internal(X, np.zeros(len(X)))["silhouette_score"])
    distances = skm.pairwise_distances(X)
    assert evaluate_clustering_internal(distances, y, metric="precomputed")["silhouette_score"] == pytest.approx(
        evaluate_clustering_internal(X, y)["silhouette_score"])


def test_interpretability_of_ruletree_and_sklearn_trees(iris):
    X, y = iris
    sklearn_tree = DecisionTreeClassifier(max_depth=3, random_state=0).fit(X, y)

    scores = evaluate_interpretability(sklearn_tree)

    assert scores == {"n_leaves": sklearn_tree.get_n_leaves(), "n_internal_nodes": sklearn_tree.get_n_leaves() - 1,
                      "max_depth": 3}
    assert evaluate_interpretability(RuleTreeClassifier(max_depth=3).fit(X, y))["max_depth"] == 3
    assert evaluate_interpretability(object()) == {}


def test_fairness_of_a_fair_partition():
    labels = np.array([0, 0, 1, 1])
    sex = np.array(["F", "M", "F", "M"])

    assert evaluate_fairness(labels, sex) == {"balance": 1.0, "max_fairness_cost": 0.0}
    assert evaluate_fairness(labels, sex, {"F": 1.0, "M": 0.0})["max_fairness_cost"] == 0.5
