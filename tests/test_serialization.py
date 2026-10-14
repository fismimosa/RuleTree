import json

import numpy as np
import pytest

from RuleTree import (RuleTreeClassifier, RuleTreeCluster, RuleTreeClusterClassifier, RuleTreeClusterRegressor,
                      RuleTreeRegressor)
from RuleTree.tree.RuleTree import RuleTree
from RuleTree.utils.data_utils import json_numpy_encoder

# (estimator class, constructor arguments, dataset fixture, uses y)
MODELS = [
    (RuleTreeClassifier, {"max_depth": 3}, "iris", True),
    (RuleTreeRegressor, {"max_depth": 3}, "diabetes", True),
    (RuleTreeCluster, {"max_depth": 3}, "blobs", False),
    (RuleTreeClusterClassifier, {"max_depth": 3}, "iris", True),
    (RuleTreeClusterRegressor, {"max_depth": 3, "bic_eps": 0.5}, "diabetes", True),
]


@pytest.mark.parametrize("model_class, kwargs, dataset, supervised", MODELS, ids=[m[0].__name__ for m in MODELS])
def test_dict_round_trip(model_class, kwargs, dataset, supervised, request):
    X, y = request.getfixturevalue(dataset)
    model = model_class(**kwargs).fit(X, y if supervised else None)

    loaded = RuleTree.from_dict(json.loads(json.dumps(model.to_dict(), cls=json_numpy_encoder)))

    assert type(loaded) is model_class
    assert loaded.get_params(deep=False).keys() == model.get_params(deep=False).keys()
    assert np.array_equal(loaded.predict(X), model.predict(X))


def test_file_round_trip(iris, tmp_path):
    X, y = iris
    model = RuleTreeClassifier(max_depth=3).fit(X, y)
    filename = tmp_path / "tree.json"

    model.to_dict(filename=filename)
    loaded = RuleTree.from_dict(filename)

    assert np.array_equal(loaded.predict(X), model.predict(X))


def test_categorical_split_round_trip(iris_mixed):
    X, y = iris_mixed
    model = RuleTreeClassifier(max_depth=3).fit(X, y)
    assert any(node.stump.is_categorical for node in model.get_predicates().values())

    loaded = RuleTree.from_dict(json.loads(json.dumps(model.to_dict(), cls=json_numpy_encoder)))

    assert np.array_equal(loaded.predict(X), model.predict(X))
    assert loaded.get_rules()["textual_rule"] == model.get_rules()["textual_rule"]


@pytest.mark.parametrize("model_class, kwargs, dataset, supervised", MODELS, ids=[m[0].__name__ for m in MODELS])
def test_classes_only_for_classification(model_class, kwargs, dataset, supervised, request):
    X, y = request.getfixturevalue(dataset)
    model = model_class(**kwargs).fit(X, y if supervised else None)
    dictionary = json.loads(json.dumps(model.to_dict(), cls=json_numpy_encoder))
    is_classifier = model_class in (RuleTreeClassifier, RuleTreeClusterClassifier)

    assert hasattr(model, "classes_") == is_classifier
    assert ("classes_" in dictionary) == is_classifier
    if is_classifier:
        assert np.array_equal(model.classes_, np.unique(y))
    elif model_class is not RuleTreeCluster:
        assert all(node["prediction_classes_"] is None for node in dictionary["nodes"])
