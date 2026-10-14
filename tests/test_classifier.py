import warnings

import numpy as np
import pytest
import sklearn

from RuleTree import RuleTreeClassifier


def test_fit_predict(iris):
    X, y = iris
    model = RuleTreeClassifier(max_depth=3, random_state=0).fit(X, y)

    y_pred = model.predict(X)

    assert y_pred.shape == y.shape
    assert set(y_pred) <= set(y)
    assert np.mean(y_pred == y) > 0.9


def test_string_labels(iris):
    X, y = iris
    y_str = np.array(["setosa", "versicolor", "virginica"])[y]

    model = RuleTreeClassifier(max_depth=3).fit(X, y_str)

    assert set(model.predict(X)) <= set(y_str)
    assert np.mean(model.predict(X) == y_str) > 0.9


def test_predict_proba(iris):
    X, y = iris
    model = RuleTreeClassifier(max_depth=2).fit(X, y)

    proba = model.predict_proba(X)

    assert proba.shape == (X.shape[0], len(np.unique(y)))
    assert np.allclose(proba.sum(axis=1), 1)
    assert np.array_equal(model.classes_[proba.argmax(axis=1)], model.predict(X))


def test_no_warnings(iris):
    X, y = iris
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        RuleTreeClassifier(max_depth=4).fit(X, y).predict(X)


def test_pure_node_log_odds(iris):
    X, y = iris
    model = RuleTreeClassifier(max_depth=1).fit(X, y)

    pure_leaf = model.root.node_l  # setosa is separated by the first split
    assert np.all(pure_leaf.prediction_probability[1:] == 0)
    assert pure_leaf.log_odds[0] == np.inf
    assert np.all(pure_leaf.log_odds[1:] == -np.inf)


@pytest.mark.parametrize("max_depth", [1, 2, 3])
def test_max_depth(iris, max_depth):
    X, y = iris
    model = RuleTreeClassifier(max_depth=max_depth).fit(X, y)

    assert max(leaf.get_depth() for leaf in model.get_leaf_nodes().values()) <= max_depth


@pytest.mark.parametrize("max_leaf_nodes", [2, 3, 5])
def test_max_leaf_nodes(iris, max_leaf_nodes):
    X, y = iris
    model = RuleTreeClassifier(max_leaf_nodes=max_leaf_nodes).fit(X, y)

    assert len(model.get_leaf_nodes()) <= max_leaf_nodes


def test_min_samples_split(iris):
    X, y = iris
    model = RuleTreeClassifier(min_samples_split=len(y) + 1).fit(X, y)

    assert model.root.is_leaf()
    assert np.all(model.predict(X) == model.root.prediction)


def test_best_stump_selection(iris):
    X, y = iris
    model = RuleTreeClassifier(max_depth=3, stump_selection="best").fit(X, y)

    assert np.mean(model.predict(X) == y) > 0.9


def test_categorical_features(iris_mixed):
    X, y = iris_mixed
    model = RuleTreeClassifier(max_depth=3).fit(X, y)

    assert np.mean(model.predict(X) == y) > 0.9
    assert any(node.stump.is_categorical for node in model.get_predicates().values())


def test_prune_useless_leaves(iris):
    X, y = iris
    model = RuleTreeClassifier(max_depth=6, prune_useless_leaves=True).fit(X, y)

    for node in model.get_predicates().values():
        if node.node_l.is_leaf() and node.node_r.is_leaf():
            assert node.node_l.prediction != node.node_r.prediction


def test_sklearn_clone(iris):
    X, y = iris
    model = RuleTreeClassifier(max_depth=2, random_state=0)

    clone = sklearn.clone(model)

    assert clone.get_params()["max_depth"] == 2
    assert np.array_equal(clone.fit(X, y).predict(X), model.fit(X, y).predict(X))
