import json

import numpy as np
import pytest
import sklearn

from RuleTree import RuleTreeClassifier, RuleTreeRegressor
from RuleTree.stumps.classification import (DecisionTreeStumpClassifier, PartialPivotTreeStumpClassifier,
                                            PivotTreeStumpClassifier, ShapeletTreeStumpClassifier)
from RuleTree.stumps.regression import (DecisionTreeStumpRegressor, FairTreeStumpRegressor, PivotTreeStumpRegressor,
                                        ShapeletTreeStumpRegressor)
from RuleTree.stumps.regression.XGBoostStumpRegressor import XGBoostStumpRegressor
from RuleTree.stumps.Stump import Stump
from RuleTree.stumps.splitters import FairSplitter, RegressionSplitter
from RuleTree.stumps.transformers import MatrixTransformer, PivotTransformer
from RuleTree.tree.RuleTree import RuleTree
from RuleTree.utils.data_utils import json_numpy_encoder
from RuleTree.utils.define import DATA_TYPE_TABULAR, DATA_TYPE_TS, MODEL_TYPE_REG


def shapelet_kwargs():
    return {"n_shapelets": 4, "sliding_window": 10, "selection": "random"}


@pytest.fixture(scope="module")
def time_series():
    """Class 1 has a bump in the middle of the series."""
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, size=80)
    X = rng.normal(scale=0.3, size=(80, 1, 40))
    X[y == 1, 0, 15:25] += 2
    return X, y


# (stump, data type, dataset fixture, target is continuous)
STUMPS = [
    (lambda: DecisionTreeStumpClassifier(), DATA_TYPE_TABULAR, "iris", False),
    (lambda: DecisionTreeStumpClassifier(), DATA_TYPE_TS, "time_series", False),
    (lambda: PivotTreeStumpClassifier(), DATA_TYPE_TABULAR, "iris", False),
    (lambda: PartialPivotTreeStumpClassifier(), DATA_TYPE_TABULAR, "iris", False),
    (lambda: ShapeletTreeStumpClassifier(**shapelet_kwargs()), DATA_TYPE_TS, "time_series", False),
    (lambda: DecisionTreeStumpRegressor(), DATA_TYPE_TABULAR, "diabetes", True),
    (lambda: PivotTreeStumpRegressor(), DATA_TYPE_TABULAR, "diabetes", True),
    (lambda: ShapeletTreeStumpRegressor(**shapelet_kwargs(), mi_n_neighbors=5), DATA_TYPE_TS, "time_series", True),
    (lambda: XGBoostStumpRegressor(lam=1), DATA_TYPE_TABULAR, "diabetes", True),
    (lambda: FairTreeStumpRegressor(penalty="balance", sensible_attribute=1, n_jobs=1), DATA_TYPE_TABULAR,
     "diabetes", True),
]


@pytest.mark.parametrize("make_stump, data_type, dataset, continuous", STUMPS,
                         ids=[f"{s[0]().__class__.__name__}-{s[1]}" for s in STUMPS])
def test_stump_contract(make_stump, data_type, dataset, continuous, request):
    """Every stump: fit on a subset idx of the whole data, apply returns 1/2 for the requested instances."""
    X, y = request.getfixturevalue(dataset)
    data = {data_type: X}
    idx = np.arange(0, len(y), 2)

    stump = make_stump().fit(data, y.astype(float) if continuous else y, idx=idx, data_type=data_type)

    assert stump.data_type_ == data_type
    labels = stump.apply(data, idx)
    assert labels.shape == idx.shape
    assert np.issubdtype(labels.dtype, np.integer)
    assert set(labels) <= {1, 2}
    assert np.array_equal(stump.apply(data)[idx], labels)


def test_stump_rejects_unsupported_data(time_series):
    X, y = time_series

    with pytest.raises(ValueError, match="does not support"):
        PivotTreeStumpClassifier().fit({DATA_TYPE_TS: X}, y, data_type=DATA_TYPE_TS)


def test_pivot_classifier(iris):
    X, y = iris
    model = RuleTreeClassifier(max_depth=3, base_stumps=PivotTreeStumpClassifier()).fit(X, y)

    assert np.mean(model.predict(X) == y) > 0.9


def test_partial_pivot_classifier(iris):
    X, y = iris
    model = RuleTreeClassifier(max_depth=2, base_stumps=PartialPivotTreeStumpClassifier()).fit(X, y)

    assert np.mean(model.predict(X) == y) > 0.9


def test_pivot_regressor(diabetes):
    X, y = diabetes
    model = RuleTreeRegressor(max_depth=2, base_stumps=PivotTreeStumpRegressor()).fit(X, y)

    assert len(np.unique(model.predict(X))) > 1


def test_shapelet_classifier(time_series):
    X, y = time_series
    model = RuleTreeClassifier(max_depth=2, base_stumps=ShapeletTreeStumpClassifier(**shapelet_kwargs()))

    model.fit(X_ts=X, y=y)

    assert np.mean(model.predict(X_ts=X) == y) > 0.9


def test_fair_regressor(diabetes):
    X, y = diabetes
    X = np.column_stack([X, X[:, 1] > 0])
    stump = FairTreeStumpRegressor(penalty="balance", sensible_attribute=-1, n_jobs=1)

    model = RuleTreeRegressor(max_depth=2, base_stumps=stump).fit(X, y)

    assert model.predict(X).shape == y.shape


def json_round_trip(dictionary):
    return json.loads(json.dumps(dictionary, cls=json_numpy_encoder))


@pytest.mark.parametrize("make_stump, data_type, dataset, continuous", STUMPS,
                         ids=[f"{s[0]().__class__.__name__}-{s[1]}" for s in STUMPS])
def test_stump_serialization(make_stump, data_type, dataset, continuous, request):
    X, y = request.getfixturevalue(dataset)
    data = {data_type: X}

    unfitted = make_stump()
    loaded = type(unfitted).dict_to_node(json_round_trip(unfitted.node_to_dict()))
    assert loaded.get_params() == unfitted.get_params()

    fitted = make_stump().fit(data, y.astype(float) if continuous else y, data_type=data_type)
    loaded = type(fitted).dict_to_node(json_round_trip(fitted.node_to_dict()))
    assert loaded.get_params() == fitted.get_params()
    assert loaded.data_type_ == data_type
    assert np.array_equal(loaded.apply(data), fitted.apply(data))
    assert loaded.get_rule(float_precision=None)["textual_rule"] == fitted.get_rule(float_precision=None)["textual_rule"]


@pytest.mark.parametrize("stump", [
    DecisionTreeStumpClassifier(criterion="entropy"),
    PivotTreeStumpClassifier(criterion="entropy", distance_measure="cosine", k=7),
], ids=["DecisionTree", "Pivot"])
def test_serialization_keeps_parameters(stump):
    loaded = type(stump).dict_to_node(json_round_trip(stump.node_to_dict()))

    assert loaded.criterion == "entropy"
    assert loaded.get_params() == stump.get_params()


def test_composed_stump(diabetes):
    """Any transformer with any splitter: here pivots with a fairness constraint."""
    X, y = diabetes
    X = np.column_stack([X, X[:, 1] > 0])
    stump = Stump(PivotTransformer(MODEL_TYPE_REG, tree_params={"random_state": 0}),
                  FairSplitter(penalty="balance", sensible_attribute=0, n_jobs=1))
    assert stump.task == MODEL_TYPE_REG
    assert stump.supported_data_types == (DATA_TYPE_TABULAR,)

    model = RuleTreeRegressor(max_depth=2, base_stumps=stump).fit(X, y)
    loaded = RuleTree.from_dict(json_round_trip(model.to_dict()))

    assert len(model.get_leaf_nodes()) > 1
    assert np.array_equal(loaded.predict(X), model.predict(X))
    assert type(loaded.root.stump.transformer) is PivotTransformer


def test_stump_task_must_match_the_tree(iris):
    X, y = iris

    with pytest.raises(TypeError, match="not a stump for"):
        RuleTreeClassifier(base_stumps=Stump(MatrixTransformer(), RegressionSplitter())).fit(X, y)
    with pytest.raises(TypeError, match="not a stump for"):
        RuleTreeRegressor(base_stumps=DecisionTreeStumpClassifier()).fit(X, y)


def test_stumps_have_explicit_parameters():
    stump = DecisionTreeStumpClassifier(criterion="entropy", min_samples_leaf=5)

    assert stump.get_params()["min_samples_leaf"] == 5
    assert sklearn.clone(stump).get_params() == stump.get_params()
    with pytest.raises(TypeError):
        DecisionTreeStumpClassifier(max_depth=1)  # stumps always have depth 1


@pytest.mark.parametrize("base_stumps, dataset", [
    (DecisionTreeStumpClassifier(), "categorical"),
    ([DecisionTreeStumpRegressor(random_state=0),
      FairTreeStumpRegressor(penalty="balance", sensible_attribute=-1, n_jobs=1)], "fair"),
], ids=["only-categorical", "fair"])
def test_best_selection_uses_the_gain_of_any_stump(base_stumps, dataset, iris, diabetes):
    if dataset == "categorical":
        X, y = iris
        X = np.where(X > np.median(X, axis=0), "high", "low").astype(object)
        model = RuleTreeClassifier(max_depth=2, stump_selection="best", base_stumps=base_stumps)
    else:
        X, y = diabetes
        X = np.column_stack([X, X[:, 1] > 0])
        model = RuleTreeRegressor(max_depth=2, stump_selection="best", base_stumps=base_stumps)

    model.fit(X, y)

    assert len(model.get_leaf_nodes()) > 1
