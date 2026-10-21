import json

import numpy as np
import pytest

from RuleTree import RuleTreeClassifier, RuleTreeCluster, RuleTreeRegressor
from RuleTree.stumps.classification import DecisionTreeStumpClassifier, ObliqueDecisionTreeStumpClassifier
from RuleTree.stumps.regression import ObliqueDecisionTreeStumpRegressor
from RuleTree.tree.RuleTree import RuleTree
from RuleTree.utils.data_utils import json_numpy_encoder


def json_round_trip(dictionary):
    return json.loads(json.dumps(dictionary, cls=json_numpy_encoder))


@pytest.fixture(scope="module")
def diagonal():
    """Two classes separated by the diagonal x0 + x1 = 0: an axis-parallel split cannot separate them."""
    rng = np.random.default_rng(0)
    X = rng.uniform(-1, 1, size=(300, 2))
    return X, (X.sum(axis=1) > 0).astype(int)


@pytest.mark.parametrize("oblique_split_type", ["householder", "bivariate"])
def test_rule_is_the_split(oblique_split_type, iris):
    """The combination and the threshold of the rule give the same split as apply."""
    X, y = iris
    stump = ObliqueDecisionTreeStumpClassifier(oblique_split_type, random_state=0)
    stump.fit(X, y, context=None, node_id=None)

    rule = stump.get_rule(float_precision=None)
    left = X[:, list(rule["feature_idx"])] @ np.array(rule["coefficients"]) <= rule["threshold"]

    assert np.array_equal(stump.apply(X), np.where(left, 1, 2))
    assert rule["textual_rule"].endswith(f"<= {rule['threshold']}")
    assert rule["not_textual_rule"] == rule["textual_rule"].replace("<=", ">")


def test_rule_with_columns_names(iris):
    X, y = iris
    stump = ObliqueDecisionTreeStumpClassifier("bivariate", random_state=0).fit(X, y, context=None, node_id=None)
    names = ["sepal length", "sepal width", "petal length", "petal width"]

    rule = stump.get_rule(columns_names=names)

    for feature in stump.feature_original:
        assert names[feature] in rule["textual_rule"]
    with pytest.raises(ValueError, match="Scaler"):
        stump.get_rule(scaler=object())


def test_oblique_split_on_the_diagonal(diagonal):
    """One bivariate split separates the classes almost perfectly, an axis-parallel one cannot."""
    X, y = diagonal

    oblique = RuleTreeClassifier(max_depth=1, random_state=0,
                                 base_stumps=ObliqueDecisionTreeStumpClassifier("bivariate", n_orientations=8))
    axis_parallel = RuleTreeClassifier(max_depth=1, random_state=0, base_stumps=DecisionTreeStumpClassifier())

    assert oblique.fit(X, y).score(X, y) > 0.95
    assert axis_parallel.fit(X, y).score(X, y) < 0.9
    assert set(oblique.root.stump.feature_original) == {0, 1}


def test_invalid_oblique_split_type():
    with pytest.raises(ValueError, match="oblique_split_type"):
        ObliqueDecisionTreeStumpClassifier("diagonal")
    with pytest.raises(ValueError, match="oblique_split_type"):
        ObliqueDecisionTreeStumpRegressor("diagonal")


@pytest.mark.parametrize("oblique_split_type", ["householder", "bivariate"])
def test_trees_with_oblique_stumps(oblique_split_type, iris, diabetes, blobs):
    """Classification, regression and clustering trees, also after to_dict and from_dict."""
    X, y = iris
    classifier = RuleTreeClassifier(max_depth=3, random_state=0,
                                    base_stumps=ObliqueDecisionTreeStumpClassifier(oblique_split_type)).fit(X, y)
    assert classifier.score(X, y) > 0.9

    X_reg, y_reg = diabetes
    regressor = RuleTreeRegressor(max_depth=2, random_state=0,
                                  base_stumps=ObliqueDecisionTreeStumpRegressor(oblique_split_type)).fit(X_reg, y_reg)
    assert len(regressor.get_leaf_nodes()) == 4

    X_blobs, _ = blobs
    cluster = RuleTreeCluster(max_depth=2, random_state=0,
                              base_stumps=ObliqueDecisionTreeStumpRegressor(oblique_split_type)).fit(X_blobs)
    assert len(set(cluster.predict(X_blobs))) > 1

    for model, data in [(classifier, X), (regressor, X_reg), (cluster, X_blobs)]:
        loaded = RuleTree.from_dict(json_round_trip(model.to_dict()))
        assert np.array_equal(loaded.predict(data), model.predict(data))
        rules = {node_id: node.stump.get_rule()["textual_rule"] for node_id, node in model.get_predicates().items()}
        assert {node_id: node.stump.get_rule()["textual_rule"]
                for node_id, node in loaded.get_predicates().items()} == rules


def test_best_selection_with_oblique_and_axis_parallel_stumps(diagonal):
    X, y = diagonal
    model = RuleTreeClassifier(max_depth=1, stump_selection="best", random_state=0,
                               base_stumps=[DecisionTreeStumpClassifier(),
                                            ObliqueDecisionTreeStumpClassifier("bivariate", n_orientations=8)])

    model.fit(X, y)

    assert isinstance(model.root.stump, ObliqueDecisionTreeStumpClassifier)
