import json

import numpy as np
import pytest
import sklearn
from numba import UnsupportedError

from RuleTree.exceptions import NoSplitFoundWarning
from RuleTree.stumps.transformers import (MatrixTransformer, PartialPivotTransformer, PivotTransformer,
                                          ShapeletTransformer)
from RuleTree.utils.data_utils import json_numpy_encoder
from RuleTree.utils.define import DATA_TYPE_TABULAR, DATA_TYPE_TS, MODEL_TYPE_CLF


@pytest.fixture(scope="module")
def time_series():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, size=40)
    X = rng.normal(scale=0.3, size=(40, 1, 30))
    X[y == 1, 0, 10:20] += 2
    return X, y


def json_round_trip(state):
    return json.loads(json.dumps(state, cls=json_numpy_encoder))


def test_matrix_transformer(iris_mixed, time_series):
    X, y = iris_mixed
    cache = {}

    transformer = MatrixTransformer().fit(X, y, data_type=DATA_TYPE_TABULAR, cache=cache)

    assert transformer.transform(X) is X  # tabular data are used as they are
    assert (transformer.numerical_, transformer.categorical_) == ([1, 2], [0])
    assert cache[("feature_types", DATA_TYPE_TABULAR)] == ([1, 2], [0])

    X_ts, y_ts = time_series
    transformer = MatrixTransformer().fit(X_ts, y_ts, data_type=DATA_TYPE_TS)
    assert transformer.transform(X_ts).shape == (40, 30)  # flattened
    assert transformer.numerical_ == list(range(30))


def test_pivot_transformer(iris):
    X, y = iris
    idx = np.arange(0, len(y), 2)
    cache = {}

    transformer = PivotTransformer(MODEL_TYPE_CLF).fit(X, y, idx, data_type=DATA_TYPE_TABULAR, cache=cache)

    candidates = len(transformer.candidates_)
    assert transformer.transform(X[idx]).shape == (len(idx), candidates)
    assert set(transformer.candidate_names_) <= set(idx)  # pivots are instances of the node
    assert ("pivot", DATA_TYPE_TABULAR, "euclidean") in cache

    name = transformer.feature_id(1)
    distances = transformer.transform(X)[:, 1]
    assert transformer.select(1) == 0
    assert transformer.feature_id(0) == name
    assert np.allclose(transformer.transform(X)[:, 0], distances)

    loaded = sklearn.clone(transformer).set_state(json_round_trip(transformer.get_state()))
    assert np.allclose(loaded.transform(X), transformer.transform(X))
    assert loaded.feature_id(0) == name


def test_pivot_transformer_needs_numerical_features(iris_mixed):
    X, y = iris_mixed

    with pytest.raises(NoSplitFoundWarning):
        PivotTransformer().fit(X[:, [0]], y)


def test_shapelet_transformer(time_series):
    X, y = time_series

    transformer = ShapeletTransformer(n_shapelets=3, sliding_window=10, selection="random").fit(X, y)

    assert transformer.transform(X).shape == (len(y), 3)
    assert transformer.numerical_ == [0, 1, 2]

    loaded = sklearn.clone(transformer).set_state(json_round_trip(transformer.get_state()))
    assert np.allclose(loaded.transform(X), transformer.transform(X))

    with pytest.raises(UnsupportedError):
        ShapeletTransformer().fit(X, y, sample_weight=np.ones(len(y)))


def test_partial_pivot_transformer(iris):
    X, y = iris

    transformer = PartialPivotTransformer(n_shapelets=3).fit(X, y)
    assert transformer.refit_after_select
    assert transformer.transform(X).shape[1] == len(transformer.st_.shapelets)

    distances = transformer.transform(X)[:, 1]
    assert transformer.select(1) == 0
    assert np.allclose(transformer.transform(X)[:, 0], distances)

    loaded = sklearn.clone(transformer).set_state(json_round_trip(transformer.get_state()))
    assert np.allclose(loaded.transform(X), transformer.transform(X))


@pytest.mark.parametrize("transformer, data_types", [
    (MatrixTransformer(), {"tabular", "ts", "image"}),
    (PivotTransformer(), {"tabular"}),
    (PartialPivotTransformer(), {"tabular"}),
    (ShapeletTransformer(), {"ts"}),
], ids=["Matrix", "Pivot", "PartialPivot", "Shapelet"])
def test_supported_data_types(transformer, data_types):
    assert set(transformer.supported_data_types) == data_types
    assert sklearn.clone(transformer).get_params() == transformer.get_params()
