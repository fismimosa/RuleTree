import json
import warnings

import numpy as np
import pytest
from sklearn.datasets import load_iris

from RuleTree import RuleTreeClassifier, RuleTreeCluster, RuleTreeRegressor
from RuleTree.stumps.classification import (DecisionTreeStumpClassifier, PivotTreeStumpClassifier,
                                            ShapeletTreeStumpClassifier)
from RuleTree.tree.RuleTree import RuleTree
from RuleTree.utils.data_utils import json_numpy_encoder
from RuleTree.utils.define import DATA_TYPE_TABULAR, DATA_TYPE_TS


@pytest.fixture(scope="module")
def multimodal():
    """
    Iris (tabular) plus time series: only the time series separate versicolor from virginica
    (the tabular data have the petal features removed).
    """
    X, y = load_iris(return_X_y=True)
    rng = np.random.default_rng(0)
    X_ts = rng.normal(scale=0.3, size=(len(y), 1, 30))
    X_ts[y == 2, 0, 10:20] += 2
    return X[:, :2], X_ts, y


def shapelet():
    return ShapeletTreeStumpClassifier(n_shapelets=4, sliding_window=10, selection="random")


def split_types(model):
    return {(type(node.stump).__name__, node.stump.data_type_) for node in model.get_predicates().values()}


def test_data_type_from_argument_name(multimodal):
    _, X_ts, y = multimodal

    model = RuleTreeClassifier(max_depth=2).fit(X_ts=X_ts, y=y)

    assert split_types(model) == {("DecisionTreeStumpClassifier", DATA_TYPE_TS)}
    assert model.predict(X_ts=X_ts).shape == y.shape


def test_stumps_use_all_supported_data_types(multimodal):
    X, X_ts, y = multimodal
    model = RuleTreeClassifier(max_depth=3, stump_selection="best",
                               base_stumps=[DecisionTreeStumpClassifier(), shapelet()])

    model.fit(X=X, X_ts=X_ts, y=y)

    candidates = {(type(stump).__name__, data_type) for _, stump, data_type in model.candidate_splits_}
    assert candidates == {("DecisionTreeStumpClassifier", DATA_TYPE_TABULAR),
                          ("DecisionTreeStumpClassifier", DATA_TYPE_TS),
                          ("ShapeletTreeStumpClassifier", DATA_TYPE_TS)}
    assert np.mean(model.predict(X=X, X_ts=X_ts) == y) > 0.9


def test_base_stumps_data_type(multimodal):
    X, X_ts, y = multimodal
    model = RuleTreeClassifier(max_depth=3, stump_selection="best",
                               base_stumps=[(0.5, DecisionTreeStumpClassifier(), DATA_TYPE_TABULAR),
                                            (0.5, shapelet())])

    model.fit(X=X, X_ts=X_ts, y=y)

    assert split_types(model) <= {("DecisionTreeStumpClassifier", DATA_TYPE_TABULAR),
                                  ("ShapeletTreeStumpClassifier", DATA_TYPE_TS)}
    assert np.mean(model.predict(X=X, X_ts=X_ts) == y) > 0.9


def test_base_stumps_probabilities(multimodal):
    X, X_ts, y = multimodal
    model = RuleTreeClassifier(base_stumps=[(0.6, DecisionTreeStumpClassifier()), (0.4, shapelet(), [DATA_TYPE_TS])])

    model._set_candidate_splits({DATA_TYPE_TABULAR: X, DATA_TYPE_TS: X_ts})

    probabilities = {(type(stump).__name__, data_type): p for p, stump, data_type in model.candidate_splits_}
    assert probabilities == pytest.approx({("DecisionTreeStumpClassifier", DATA_TYPE_TABULAR): 0.3,
                                           ("DecisionTreeStumpClassifier", DATA_TYPE_TS): 0.3,
                                           ("ShapeletTreeStumpClassifier", DATA_TYPE_TS): 0.4})


def test_random_selection_is_reproducible(multimodal):
    X, X_ts, y = multimodal
    base_stumps = [DecisionTreeStumpClassifier(), shapelet()]

    first = RuleTreeClassifier(max_depth=3, base_stumps=base_stumps, random_state=0).fit(X=X, X_ts=X_ts, y=y)
    second = RuleTreeClassifier(max_depth=3, base_stumps=base_stumps, random_state=0).fit(X=X, X_ts=X_ts, y=y)

    assert np.array_equal(first.apply(X=X, X_ts=X_ts), second.apply(X=X, X_ts=X_ts))


def test_no_stump_supports_data(iris):
    X, y = iris

    with pytest.raises(ValueError, match="None of the base stumps supports"):
        RuleTreeClassifier(base_stumps=shapelet()).fit(X, y)


def test_text_is_not_supported_by_decision_tree(iris):
    X, y = iris
    texts = np.array(["text"] * len(y))

    with pytest.raises(ValueError, match="None of the base stumps supports"):
        RuleTreeClassifier().fit(X_txt=texts, y=y)

    with pytest.warns(UserWarning, match="text data"):
        RuleTreeClassifier(max_depth=1).fit(X=X, X_txt=texts, y=y)


@pytest.mark.parametrize("base_stumps, error", [
    ([(1.0, shapelet(), DATA_TYPE_TABULAR)], "does not support"),
    ([(0.5, DecisionTreeStumpClassifier()), (0.2, shapelet())], "sum to 1"),
    ([("not a stump",)], "Invalid base_stumps entry"),
    ([], "empty"),
])
def test_invalid_base_stumps(iris, base_stumps, error):
    X, y = iris

    with pytest.raises(ValueError, match=error):
        RuleTreeClassifier(base_stumps=base_stumps).fit(X, y)


def test_inconsistent_number_of_samples(multimodal):
    X, X_ts, y = multimodal

    with pytest.raises(ValueError):
        RuleTreeClassifier().fit(X=X, X_ts=X_ts[:-1], y=y)


def test_cluster_supports_only_tabular_data(multimodal):
    X, X_ts, _ = multimodal

    with pytest.raises(ValueError, match="only tabular"):
        RuleTreeCluster().fit(X=X, X_ts=X_ts)


def test_serialization_keeps_data_types(multimodal):
    X, X_ts, y = multimodal
    model = RuleTreeClassifier(max_depth=3, base_stumps=[(1.0, DecisionTreeStumpClassifier(), DATA_TYPE_TS)])
    model.fit(X=X, X_ts=X_ts, y=y)

    loaded = RuleTree.from_dict(json.loads(json.dumps(model.to_dict(), cls=json_numpy_encoder)))

    assert split_types(loaded) == {("DecisionTreeStumpClassifier", DATA_TYPE_TS)}
    assert np.array_equal(loaded.predict(X=X, X_ts=X_ts), model.predict(X=X, X_ts=X_ts))
    assert loaded._parse_base_stumps()[0][2] == (DATA_TYPE_TS,)


def test_cache(iris):
    X, y = iris
    model = RuleTreeClassifier(max_depth=2).fit(X, y)

    stump = model.candidate_splits_[0][1]
    assert ("feature_types", DATA_TYPE_TABULAR) in model.get_cache(stump.cache_namespace())  # kept after fit

    model.get_cache("test")["value"] = 1
    model.fit(X, y)
    assert model.get_cache("test") == {}  # cleared by fit

    model.get_cache("test")["value"] = 1
    model.clear_cache()
    assert model.get_cache("test") == {}


def test_fit_restarts_from_scratch(iris):
    X, y = iris
    model = RuleTreeClassifier(max_depth=3).fit(X, y)
    fresh = RuleTreeClassifier(max_depth=1).fit(X, y)

    model.set_params(max_depth=1).fit(X, y)

    assert set(model.get_leaf_nodes()) == set(fresh.get_leaf_nodes())


def test_partial_fit_grows_the_tree(iris):
    X, y = iris
    model = RuleTreeClassifier(max_depth=1).fit(X, y)
    root_stump = model.root.stump

    model.set_params(max_depth=3).partial_fit(X, y)

    assert model.root.stump is root_stump
    assert max(leaf.get_depth() for leaf in model.get_leaf_nodes().values()) == 3
    fresh = RuleTreeClassifier(max_depth=3).fit(X, y)
    assert np.array_equal(model.predict(X), fresh.predict(X))


def test_partial_fit_respects_max_leaf_nodes(iris):
    X, y = iris
    model = RuleTreeClassifier(max_leaf_nodes=2).fit(X, y)

    model.set_params(max_leaf_nodes=4).partial_fit(X, y)

    assert len(model.get_leaf_nodes()) == 4


def test_partial_fit_without_fit(diabetes):
    X, y = diabetes

    model = RuleTreeRegressor(max_depth=2).partial_fit(X, y)

    assert np.array_equal(model.predict(X), RuleTreeRegressor(max_depth=2).fit(X, y).predict(X))


def test_partial_fit_keeps_cache(iris):
    X, y = iris
    model = RuleTreeClassifier(max_depth=1).fit(X, y)
    model.get_cache("test")["value"] = 1

    model.set_params(max_depth=2).partial_fit(X, y)

    assert model.get_cache("test") == {"value": 1}


def test_no_warnings_on_time_series(multimodal):
    _, X_ts, y = multimodal
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        RuleTreeClassifier(max_depth=3).fit(X_ts=X_ts, y=y).predict(X_ts=X_ts)


def test_cache_namespace_is_the_stump_class_and_its_hyperparameters(iris):
    X, y = iris
    model = RuleTreeClassifier(max_depth=2, stump_selection="best", random_state=0, base_stumps=[
        DecisionTreeStumpClassifier(), PivotTreeStumpClassifier(), PivotTreeStumpClassifier(distance_measure="cosine")
    ]).fit(X, y)
    namespaces = [stump.cache_namespace() for _, stump, _ in model.candidate_splits_]

    assert len(set(namespaces)) == 3 and set(model.cache_) == set(namespaces)
    assert [n.split("-")[0] for n in namespaces] == ["DecisionTreeStumpClassifier"] + ["PivotTreeStumpClassifier"] * 2
    assert ("pivot", DATA_TYPE_TABULAR, "euclidean") in model.get_cache(namespaces[1])
    assert ("pivot", DATA_TYPE_TABULAR, "cosine") in model.get_cache(namespaces[2])
