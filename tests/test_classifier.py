import warnings

import numpy as np
import pytest
import sklearn

from RuleTree import RuleTreeClassifier, RuleTreeClusterClassifier
from RuleTree.stumps.Stump import Stump
from RuleTree.stumps.classification import DecisionTreeStumpClassifier
from RuleTree.stumps.splitters import ClassificationSplitter
from RuleTree.stumps.transformers import MatrixTransformer


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


def test_random_state_given_to_the_stumps(iris):
    X, y = iris
    stump = DecisionTreeStumpClassifier()

    model = RuleTreeClassifier(max_depth=3, random_state=7, base_stumps=stump).fit(X, y)

    assert all(node.stump.random_state == 7 for node in model.get_predicates().values())
    assert stump.random_state is None  # base_stumps are not modified


def test_stump_random_state_is_kept(iris):
    X, y = iris

    model = RuleTreeClassifier(max_depth=2, random_state=7,
                               base_stumps=DecisionTreeStumpClassifier(random_state=3)).fit(X, y)

    assert all(node.stump.random_state == 3 for node in model.get_predicates().values())


def test_random_state_of_composed_stumps(iris):
    X, y = iris
    stump = Stump(MatrixTransformer(), ClassificationSplitter())

    model = RuleTreeClassifier(max_depth=2, random_state=7, base_stumps=stump).fit(X, y)

    assert model.root.stump.splitter.random_state == 7


def test_without_random_state_the_stumps_are_not_seeded(iris):
    X, y = iris

    model = RuleTreeClassifier(max_depth=2).fit(X, y)

    assert model.root.stump.random_state is None


@pytest.mark.parametrize("random_state", [0, np.random.RandomState(0)], ids=["int", "RandomState"])
def test_random_state_makes_trees_reproducible(iris, random_state):
    """On iris, petal length and width have the same gain at the root: without a seed, the choice is random."""
    X, y = iris

    trees = [RuleTreeClassifier(max_depth=4, random_state=random_state).fit(X, y) for _ in range(5)]

    rules = {str(tree.get_rules()) for tree in trees} if isinstance(random_state, int) else None
    assert rules is None or len(rules) == 1
    assert all(isinstance(tree.root.stump.random_state, int) for tree in trees)  # an int drawn from RandomState


def test_update_statistics_empty_leaf_predicts_nan(iris):
    """The leaves reached by no instance predict nan, not a number from the cast to int."""
    X, y = iris
    model = RuleTreeClassifier(max_depth=2, random_state=0).fit(X, y)

    with np.errstate(invalid="raise"):  # the cast of nan to int
        model.update_statistics(X[y == 0], y[y == 0])
        predictions = model.predict(X)

    assert np.all(predictions[y == 0] == 0)
    assert np.all(np.isnan(predictions[y != 0].astype(float)))


@pytest.mark.parametrize("model_class", [RuleTreeClassifier, RuleTreeClusterClassifier])
def test_target_as_a_list(model_class, iris):
    X, y = iris

    model = model_class(max_depth=2, random_state=0).fit(X, y.tolist())

    assert np.array_equal(model.predict(X), model_class(max_depth=2, random_state=0).fit(X, y).predict(X))
