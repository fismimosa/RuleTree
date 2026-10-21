import json

import numpy as np
import pytest
import sklearn

from RuleTree import RuleTreeClassifier
from RuleTree.exceptions import NoSplitFoundWarning
from RuleTree.stumps.transformers import (BivariateTransformer, HouseholderTransformer, MatrixTransformer,
                                          MultiplePivotTransformer, ObliquePivotTransformer, PartialPivotTransformer,
                                          PartialProximityTransformer, PivotTransformer, ProximityTransformer,
                                          ShapeletTransformer)
from RuleTree.stumps.transformers._pivot_selection import PivotSelection
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
    tree = RuleTreeClassifier()  # the context, with the cache

    transformer = MatrixTransformer()
    transformer = transformer.fit(X, y, data_type=DATA_TYPE_TABULAR,
                                  context=tree, node_id="R", cache=tree._stump_cache())

    assert transformer.transform(X) is X  # tabular data are used as they are
    assert (transformer.numerical_, transformer.categorical_) == ([1, 2], [0])
    assert tree.get_cache(transformer.cache_namespace())[("feature_types", DATA_TYPE_TABULAR)] == ([1, 2], [0])

    X_ts, y_ts = time_series
    transformer = MatrixTransformer().fit(X_ts, y_ts, data_type=DATA_TYPE_TS, context=None, node_id=None, cache=None)
    assert transformer.transform(X_ts).shape == (40, 30)  # flattened
    assert transformer.numerical_ == list(range(30))


def test_pivot_transformer(iris):
    X, y = iris
    idx = np.arange(0, len(y), 2)
    tree = RuleTreeClassifier()  # the context, with the cache

    transformer = PivotTransformer(MODEL_TYPE_CLF)
    transformer.fit(X, y, idx, data_type=DATA_TYPE_TABULAR, context=tree, node_id="R", cache=tree._stump_cache())

    candidates = len(transformer.candidates_)
    assert transformer.transform(X[idx]).shape == (len(idx), candidates)
    assert set(transformer.candidate_names_) <= set(idx)  # pivots are instances of the node
    assert ("pivot", DATA_TYPE_TABULAR, "euclidean") in tree.get_cache(transformer.cache_namespace())

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
        PivotTransformer().fit(X[:, [0]], y, context=None, node_id=None, cache=None)


def test_shapelet_transformer(time_series):
    X, y = time_series

    transformer = ShapeletTransformer(n_shapelets=3, sliding_window=10, selection="random")
    transformer = transformer.fit(X, y, context=None, node_id=None, cache=None)

    assert transformer.transform(X).shape == (len(y), 3)
    assert transformer.numerical_ == [0, 1, 2]

    loaded = sklearn.clone(transformer).set_state(json_round_trip(transformer.get_state()))
    assert np.allclose(loaded.transform(X), transformer.transform(X))

    with pytest.raises(ValueError, match="sample_weight"):
        ShapeletTransformer().fit(X, y, sample_weight=np.ones(len(y)), context=None, node_id=None, cache=None)


def test_partial_pivot_transformer(iris):
    X, y = iris

    transformer = PartialPivotTransformer(n_shapelets=3).fit(X, y, context=None, node_id=None, cache=None)
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
    (ProximityTransformer(), {"ts"}),
    (PartialProximityTransformer(), {"tabular"}),
    (MultiplePivotTransformer(), {"tabular"}),
    (ObliquePivotTransformer(), {"tabular"}),
], ids=["Matrix", "Pivot", "PartialPivot", "Shapelet", "Householder", "Bivariate", "Proximity", "PartialProximity",
        "MultiplePivot", "ObliquePivot"])
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
                    data_type=DATA_TYPE_TABULAR, context=None, node_id=None, cache=None)

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

    transformer = HouseholderTransformer().fit(X, np.zeros(200), context=None, node_id=None, cache=None)

    projected = transformer.transform(X)
    assert np.isclose(projected.var(axis=0).max(), X.var(axis=0).sum(), rtol=1e-3)  # all the variance on one axis


def test_oblique_transformers_need_numerical_features(iris_mixed):
    X, y = iris_mixed

    with pytest.raises(NoSplitFoundWarning):
        HouseholderTransformer().fit(X[:, [0]], y, context=None, node_id=None, cache=None)
    with pytest.raises(NoSplitFoundWarning, match="two numerical"):
        BivariateTransformer().fit(X[:, [0, 1]], y, context=None, node_id=None, cache=None)


@pytest.mark.parametrize("transformer, dataset", [
    (ProximityTransformer(n_shapelets=4, sliding_window=10, selection="random"), "time_series"),
    (PartialProximityTransformer(n_shapelets=6, random_state=0), "iris"),
], ids=["Proximity", "PartialProximity"])
def test_proximity_transformer(transformer, dataset, request):
    """One 0/1 column per pair of references: 1 where the instance is closer to the second one."""
    X, y = request.getfixturevalue(dataset)

    transformer.fit(X, y, context=None, node_id=None, cache=None)

    distances = transformer.st_.transform(X)
    comparisons = transformer.transform(X)
    assert comparisons.shape == (len(y), len(transformer.pairs_))
    for column, (i, j) in enumerate(transformer.pairs_):
        assert np.array_equal(comparisons[:, column], distances[:, i] > distances[:, j])

    column = len(transformer.pairs_) - 1
    assert transformer.select(column) == 0
    assert len(transformer.st_.shapelets) == 2 and transformer.pairs_ == [(0, 1)]
    assert np.array_equal(transformer.transform(X)[:, 0], comparisons[:, column])

    loaded = sklearn.clone(transformer).set_state(json_round_trip(transformer.get_state()))
    assert np.array_equal(loaded.transform(X), transformer.transform(X))


def test_partial_proximity_compares_pivots_on_the_same_features(iris):
    X, y = iris

    for same_features in (True, False):
        transformer = PartialProximityTransformer(n_shapelets=6, n_features_strategy=(1, 2), random_state=0,
                                                  proximity_on_same_features=same_features)
        transformer.fit(X, y, context=None, node_id=None, cache=None)
        used = ~np.isnan(transformer.st_.shapelets)
        same = [np.array_equal(used[i], used[j]) for i, j in transformer.pairs_]
        assert all(same) if same_features else len(transformer.pairs_) == 15


def test_proximity_transformer_needs_two_shapelets(time_series):
    X, y = time_series

    with pytest.raises(NoSplitFoundWarning, match="two shapelets"):
        ProximityTransformer(n_shapelets=1, sliding_window=10, selection="random").fit(X, y, context=None,
                                                                                     node_id=None, cache=None)


@pytest.mark.parametrize("discriminative, max_per_class", [("tree", 1), ("householder", 2), ("bivariate", 2)])
def test_pivot_selection_discriminatives(discriminative, max_per_class, iris):
    """The discriminative pivots of each class are instances of the class: one, or the ones of an oblique split."""
    X, y = iris
    distance_matrix = np.linalg.norm(X[:, None] - X[None], axis=2)

    selection = PivotSelection(MODEL_TYPE_CLF, discriminative=discriminative, random_state=0)
    selection.fit(X, y, distance_matrix, np.arange(len(y)))

    labels = y[selection.discriminative_names_]
    assert all(0 < np.sum(labels == label) <= max_per_class for label in np.unique(y))
    assert np.array_equal(selection.candidate_labels_, y[selection.candidate_names_])


def test_pivot_selection_unknown_discriminative(iris):
    X, y = iris

    with pytest.raises(ValueError, match="discriminative"):
        PivotSelection(MODEL_TYPE_CLF, discriminative="forest").fit(X, y, np.ones((len(y), len(y))), np.arange(len(y)))


@pytest.mark.parametrize("ml_task", [MODEL_TYPE_CLF, MODEL_TYPE_REG])
def test_multiple_pivot_transformer(ml_task, iris):
    X, y = iris
    target = y if ml_task == MODEL_TYPE_CLF else X[:, 3]

    transformer = MultiplePivotTransformer(ml_task, tree_params={"random_state": 0})
    transformer = transformer.fit(X, target, context=None, node_id=None, cache=None)

    labels = dict(zip(transformer.candidate_names_, transformer.candidate_labels_))
    distances = PivotTransformer.transform(transformer, X)
    comparisons = transformer.transform(X)
    assert comparisons.shape == (len(y), len(transformer.pairs_))
    for column, (a, b) in enumerate(transformer.pairs_):
        assert np.array_equal(comparisons[:, column], distances[:, a] > distances[:, b])
        if ml_task == MODEL_TYPE_CLF:  # only pivots of different classes
            assert labels[transformer.candidate_names_[a]] != labels[transformer.candidate_names_[b]]

    column = len(transformer.pairs_) - 1
    pair = transformer.feature_id(column)
    assert transformer.select(column) == 0
    assert transformer.feature_id(0) == pair
    assert np.array_equal(transformer.transform(X)[:, 0], comparisons[:, column])

    loaded = sklearn.clone(transformer).set_state(json_round_trip(transformer.get_state()))
    assert np.array_equal(loaded.transform(X), transformer.transform(X))
    assert loaded.feature_id(0) == pair


@pytest.mark.parametrize("oblique_split_type", ["householder", "bivariate"])
def test_oblique_pivot_transformer(oblique_split_type, iris):
    """Each column is a linear combination of the distances from the candidate pivots."""
    X, y = iris

    transformer = ObliquePivotTransformer(oblique_split_type=oblique_split_type, n_orientations=4,
                                          tree_params={"random_state": 0})
    transformer.fit(X, y, context=None, node_id=None, cache=None)

    distances = PivotTransformer.transform(transformer, X)
    names = list(transformer.candidate_names_)
    X_transformed = transformer.transform(X)
    for column in range(X_transformed.shape[1]):
        pivots, coefficients = transformer.combination(column)
        assert np.allclose(X_transformed[:, column], distances[:, [names.index(p) for p in pivots]] @ coefficients)

    column = X_transformed.shape[1] - 1
    pivots = transformer.feature_id(column)
    assert transformer.select(column) == 0
    assert transformer.feature_id(0) == pivots
    assert np.allclose(transformer.transform(X)[:, 0], X_transformed[:, column])

    loaded = sklearn.clone(transformer).set_state(json_round_trip(transformer.get_state()))
    assert np.allclose(loaded.transform(X), transformer.transform(X))
