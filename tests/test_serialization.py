import copy
import json
import pickle

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


@pytest.mark.parametrize("model_class, kwargs, dataset, supervised", MODELS, ids=[m[0].__name__ for m in MODELS])
def test_copy_and_pickle(model_class, kwargs, dataset, supervised, request):
    """Unfitted and fitted trees can be copied and pickled (the ensembles copy the base estimator), and
    the copies grow further as the original."""
    X, y = request.getfixturevalue(dataset)
    target = y if supervised else None
    kwargs = kwargs | {"random_state": 0}  # the same tree at every fit
    unfitted = model_class(**kwargs)

    for clone in (copy.deepcopy(unfitted), pickle.loads(pickle.dumps(unfitted))):
        assert np.array_equal(clone.fit(X, target).predict(X), model_class(**kwargs).fit(X, target).predict(X))

    model = model_class(**kwargs).fit(X, target)
    clones = [copy.deepcopy(model), pickle.loads(pickle.dumps(model))]
    grown = model.set_params(max_depth=5).partial_fit(X, target).predict(X)
    for clone in clones:
        assert clone.get_params()["max_depth"] == kwargs["max_depth"]  # a copy, not the same object
        assert np.array_equal(clone.set_params(max_depth=5).partial_fit(X, target).predict(X), grown)


@pytest.mark.parametrize("model_class, kwargs, dataset, supervised", MODELS, ids=[m[0].__name__ for m in MODELS])
def test_round_trip_keeps_the_statistics_of_the_nodes(model_class, kwargs, dataset, supervised, request):
    """Number of instances and balance of the splits, also used by get_balanced_stumps."""
    X, y = request.getfixturevalue(dataset)
    model = model_class(**kwargs).fit(X, y if supervised else None)

    loaded = RuleTree.from_dict(json.loads(json.dumps(model.to_dict(), cls=json_numpy_encoder)))

    for node_id, node in model.get_leaf_nodes().items() | model.get_predicates().items():
        assert loaded.get_node_by_id(node_id).samples == node.samples
    for node_id, node in model.get_predicates().items():
        assert loaded.get_node_by_id(node_id).balance_score == pytest.approx(node.balance_score)
    if isinstance(model, RuleTreeClassifier):
        assert loaded.get_balanced_stumps().keys() == model.get_balanced_stumps().keys()


def test_unfitted_tree_cannot_be_saved():
    from sklearn.exceptions import NotFittedError

    with pytest.raises(NotFittedError, match="call fit before to_dict"):
        RuleTreeClassifier().to_dict()


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
