import json

import numpy as np
import pytest
import sklearn

from RuleTree.exceptions import NoSplitFoundWarning
from RuleTree.stumps.transformers import (BivariateTransformer, HouseholderTransformer, MatrixTransformer,
                                          PartialPivotTransformer, PivotTransformer, ShapeletTransformer)
from RuleTree.utils.data_utils import json_numpy_encoder
from RuleTree.utils.define import DATA_TYPE_TABULAR, DATA_TYPE_TS, MODEL_TYPE_CLF, MODEL_TYPE_REG


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

    with pytest.raises(ValueError, match="sample_weight"):
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
    (HouseholderTransformer(), {"tabular"}),
    (BivariateTransformer(), {"tabular"}),
], ids=["Matrix", "Pivot", "PartialPivot", "Shapelet", "Householder", "Bivariate"])
def test_supported_data_types(transformer, data_types):
    assert set(transformer.supported_data_types) == data_types
    assert sklearn.clone(transformer).get_params() == transformer.get_params()


@pytest.mark.parametrize("transformer", [HouseholderTransformer(), BivariateTransformer(n_orientations=6),
                                         BivariateTransformer(MODEL_TYPE_REG, n_orientations=6)],
                         ids=["Householder", "Bivariate-clf", "Bivariate-reg"])
def test_oblique_transformer(transformer, iris_mixed):
    """Each column is a linear combination of the numerical features; select keeps one of them."""
    X, y = iris_mixed  # column 0 is categorical: it is not combined
    idx = np.arange(0, len(y), 2)

    transformer.fit(X, y.astype(float) if transformer.get_params().get("ml_task") == MODEL_TYPE_REG else y, idx,
                    data_type=DATA_TYPE_TABULAR)

    X_transformed = transformer.transform(X)
    assert transformer.num_pre_transformed_ == [1, 2]
    for column in range(X_transformed.shape[1]):
        features, coefficients = transformer.combination(column)
        assert set(features) <= {1, 2}
        assert np.allclose(X_transformed[:, column], X[:, features].astype(float) @ coefficients)

    column = X_transformed.shape[1] - 1
    name = transformer.feature_id(column)
    assert transformer.select(column) == 0
    assert transformer.feature_id(0) == name
    assert np.allclose(transformer.transform(X)[:, 0], X_transformed[:, column])

    loaded = sklearn.clone(transformer).set_state(json_round_trip(transformer.get_state()))
    assert np.allclose(loaded.transform(X), transformer.transform(X))


def test_householder_transformer_reflects_the_principal_component():
    """The reflection maps the principal component of the node onto an axis."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 1)) @ np.array([[0.6, 0.8]]) + rng.normal(scale=0.01, size=(200, 2))

    transformer = HouseholderTransformer().fit(X, np.zeros(200))

    projected = transformer.transform(X)
    assert np.isclose(projected.var(axis=0).max(), X.var(axis=0).sum(), rtol=1e-3)  # all the variance on one axis


def test_oblique_transformers_need_numerical_features(iris_mixed):
    X, y = iris_mixed

    with pytest.raises(NoSplitFoundWarning):
        HouseholderTransformer().fit(X[:, [0]], y)
    with pytest.raises(NoSplitFoundWarning, match="two numerical"):
        BivariateTransformer().fit(X[:, [0, 1]], y)
