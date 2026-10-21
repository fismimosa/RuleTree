import json
import os
import tempfile

import numpy as np
import pytest

from RuleTree import RuleTreeClassifier, RuleTreeRegressor
from RuleTree.stumps.classification import PartialProximityTreeStumpClassifier, ProximityTreeStumpClassifier
from RuleTree.stumps.regression import ProximityTreeStumpRegressor
from RuleTree.tree.RuleTree import RuleTree
from RuleTree.utils.data_utils import json_numpy_encoder
from RuleTree.utils.define import DATA_TYPE_TS


def json_round_trip(dictionary):
    return json.loads(json.dumps(dictionary, cls=json_numpy_encoder))


def shapelet_kwargs():
    return {"n_shapelets": 4, "sliding_window": 10, "selection": "random"}


@pytest.fixture(scope="module")
def time_series():
    """A bump in the middle of the series of class 0, a dip in the ones of class 1: each class is closer
    to the shapelets of its own pattern."""
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, size=80)
    X = rng.normal(scale=0.3, size=(80, 1, 40))
    X[y == 0, 0, 15:25] += 2
    X[y == 1, 0, 15:25] -= 2
    return X, y


@pytest.mark.parametrize("make_stump, dataset", [
    (lambda: ProximityTreeStumpClassifier(**shapelet_kwargs()), "time_series"),
    (lambda: PartialProximityTreeStumpClassifier(random_state=0), "iris"),
], ids=["Proximity", "PartialProximity"])
def test_closer_to_the_first_reference_goes_left(make_stump, dataset, request):
    X, y = request.getfixturevalue(dataset)
    stump = make_stump()
    stump = stump.fit(X, y, data_type=DATA_TYPE_TS if dataset == "time_series" else None, context=None, node_id=None)

    distances = stump.st.transform(X)

    assert np.array_equal(stump.apply(X), np.where(distances[:, 0] <= distances[:, 1], 1, 2))


def test_proximity_rule(time_series):
    X, y = time_series
    stump = ProximityTreeStumpClassifier(**shapelet_kwargs(), distance="cityblock")
    stump = stump.fit(X, y, data_type=DATA_TYPE_TS, context=None, node_id=None)

    rule = stump.get_rule()

    assert rule["textual_rule"] == "cityblock(TS, green_shp) <= cityblock(TS, red_shp)"
    assert rule["not_textual_rule"] == "cityblock(TS, green_shp) > cityblock(TS, red_shp)"
    with pytest.raises(ValueError, match="Scaler"):
        stump.get_rule(scaler=object())


def test_partial_proximity_rule_with_columns_names(iris):
    X, y = iris
    stump = PartialProximityTreeStumpClassifier(n_features_strategy=1, random_state=0)
    stump = stump.fit(X, y, context=None, node_id=None)

    rule = stump.get_rule(columns_names=["sl", "sw", "pl", "pw"])["textual_rule"]

    first, second = rule.split(" <= ")
    assert first.startswith("euclidean(x, SP0(d: ") and second.startswith("euclidean(x, SP1(d: ")
    assert "Feat." not in rule


@pytest.mark.parametrize("tree_class, stump_class", [(RuleTreeClassifier, ProximityTreeStumpClassifier),
                                                      (RuleTreeRegressor, ProximityTreeStumpRegressor)],
                         ids=["classifier", "regressor"])
def test_proximity_trees(tree_class, stump_class, time_series):
    X, y = time_series
    model = tree_class(max_depth=2, base_stumps=stump_class(**shapelet_kwargs())).fit(X_ts=X, y=y)

    if tree_class is RuleTreeClassifier:
        assert np.mean(model.predict(X_ts=X) == y) > 0.95  # score takes only X
    loaded = RuleTree.from_dict(json_round_trip(model.to_dict()))
    assert np.array_equal(loaded.predict(X_ts=X), model.predict(X_ts=X))


def test_partial_proximity_tree(iris):
    X, y = iris
    model = RuleTreeClassifier(max_depth=3, random_state=0, base_stumps=PartialProximityTreeStumpClassifier())

    model.fit(X, y)

    assert model.score(X, y) > 0.9
    loaded = RuleTree.from_dict(json_round_trip(model.to_dict()))
    assert np.array_equal(loaded.predict(X), model.predict(X))
    assert loaded.root.stump.get_rule()["textual_rule"] == model.root.stump.get_rule()["textual_rule"]


def test_proximity_drawing_with_both_shapelets(time_series, tmp_path, monkeypatch):
    pytest.importorskip("matplotlib")
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    X, y = time_series
    stump = ProximityTreeStumpClassifier(**shapelet_kwargs())
    stump = stump.fit(X, y, data_type=DATA_TYPE_TS, context=None, node_id=None)

    assert "image" not in stump.get_rule()["graphviz_rule"] and os.listdir(tmp_path) == []
    attributes = stump.get_graphviz_rule()
    assert attributes["label"] == stump.get_rule()["textual_rule"]
    assert os.path.dirname(attributes["image"]) == str(tmp_path) and os.path.getsize(attributes["image"]) > 0
