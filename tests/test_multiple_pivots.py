import json

import numpy as np
import pytest
from sklearn.metrics import pairwise_distances

from RuleTree import RuleTreeClassifier
from RuleTree.stumps.classification import (MultipleObliquePivotTreeStumpClassifier, MultiplePivotTreeStumpClassifier,
                                            ObliquePivotTreeStumpClassifier, PivotTreeStumpClassifier)
from RuleTree.tree.RuleTree import RuleTree
from RuleTree.utils.data_utils import json_numpy_encoder

STUMPS = [
    MultiplePivotTreeStumpClassifier,
    ObliquePivotTreeStumpClassifier,
    lambda **kwargs: ObliquePivotTreeStumpClassifier(oblique_split_type="bivariate", **kwargs),
    MultipleObliquePivotTreeStumpClassifier,
    lambda **kwargs: MultipleObliquePivotTreeStumpClassifier(oblique_split_type="bivariate", **kwargs),
]
IDS = ["MultiplePivot", "ObliquePivot-householder", "ObliquePivot-bivariate", "MultipleObliquePivot-householder",
       "MultipleObliquePivot-bivariate"]


def json_round_trip(dictionary):
    return json.loads(json.dumps(dictionary, cls=json_numpy_encoder))


def distances_from(stump, X, pivots):
    """Distances from training instances, as computed by the stump: standardized numerical features."""
    X_scaled = stump.transformer_.scaler_.transform(X)
    return pairwise_distances(X_scaled, X_scaled[list(pivots)], metric=stump.distance_measure)


@pytest.mark.parametrize("stump_class", [MultiplePivotTreeStumpClassifier, MultipleObliquePivotTreeStumpClassifier])
def test_multiple_pivot_rule_is_the_split(stump_class, iris):
    """P_a <= P_b: the instances closer to the training instance a than to b go left."""
    X, y = iris
    stump = stump_class(random_state=0).fit(X, y, context=None, node_id=None)

    a, b = stump.feature_original
    distances = distances_from(stump, X, (a, b))

    assert stump.get_rule()["textual_rule"] == f"P_{a} <= P_{b}"
    assert np.array_equal(stump.apply(X), np.where(distances[:, 0] <= distances[:, 1], 1, 2))
    assert y[a] != y[b]  # pivots of different classes


@pytest.mark.parametrize("oblique_split_type", ["householder", "bivariate"])
def test_oblique_pivot_rule_is_the_split(oblique_split_type, iris):
    X, y = iris
    stump = ObliquePivotTreeStumpClassifier(oblique_split_type=oblique_split_type, random_state=0)
    stump = stump.fit(X, y, context=None, node_id=None)

    rule = stump.get_rule(float_precision=None)
    left = distances_from(stump, X, rule["feature_idx"]) @ np.array(rule["coefficients"]) <= rule["threshold"]

    assert np.array_equal(stump.apply(X), np.where(left, 1, 2))
    assert all(f"P_{pivot}" in rule["textual_rule"] for pivot in rule["feature_idx"])


@pytest.mark.parametrize("make_stump", STUMPS, ids=IDS)
def test_pivots_of_the_split(make_stump, iris):
    X, y = iris
    model = RuleTreeClassifier(max_depth=2, base_stumps=make_stump(), random_state=0).fit(X, y)
    loaded = RuleTree.from_dict(json_round_trip(model.to_dict()))

    assert model.score(X, y) > 0.85  # as PivotTreeStumpClassifier at depth 2
    assert np.array_equal(loaded.predict(X), model.predict(X))
    for node_id, node in model.get_predicates().items():
        pivots = node.stump.get_pivots()
        assert set(pivots["used"]) <= set(pivots["candidates"])
        assert pivots["candidates"] == pivots["descriptives"] + pivots["discriminatives"]
        assert loaded.get_node_by_id(node_id).stump.get_pivots() == pivots
        assert loaded.get_node_by_id(node_id).stump.get_rule() == node.stump.get_rule()


def test_invalid_oblique_split_type():
    for stump_class in (ObliquePivotTreeStumpClassifier, MultipleObliquePivotTreeStumpClassifier):
        with pytest.raises(ValueError, match="oblique_split_type"):
            stump_class(oblique_split_type="diagonal")


def test_pivot_stumps_together(iris):
    """All the pivot stumps in the same tree, chosen by gain."""
    X, y = iris
    model = RuleTreeClassifier(max_depth=3, stump_selection="best", random_state=0,
                               base_stumps=[PivotTreeStumpClassifier(), MultiplePivotTreeStumpClassifier(),
                                            ObliquePivotTreeStumpClassifier()])

    model.fit(X, y)

    assert model.score(X, y) >= 0.9
    assert len({type(node.stump) for node in model.get_predicates().values()}) > 1
    loaded = RuleTree.from_dict(json_round_trip(model.to_dict()))
    assert np.array_equal(loaded.predict(X), model.predict(X))
