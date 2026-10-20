"""Regression with several targets: y of shape (n_samples, n_targets)."""
import json

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeRegressor

from RuleTree import RuleTreeClusterRegressor, RuleTreeRegressor
from RuleTree.ensemble.EBMRegressor import EBMRegressor
from RuleTree.ensemble.GBoostRegressor import GBoostRegressor
from RuleTree.ensemble.XGBoostRegressor import XGBoostRegressor
from RuleTree.evaluation import evaluate_regression
from RuleTree.stumps.regression import (DecisionTreeStumpRegressor, FairTreeStumpRegressor,
                                        ObliqueDecisionTreeStumpRegressor, PivotTreeStumpRegressor,
                                        ProximityTreeStumpRegressor, ShapeletTreeStumpRegressor,
                                        XGBoostStumpRegressor)
from RuleTree.tree.RuleTree import RuleTree
from RuleTree.tree.RuleTreeXGBoostRegressor import RuleTreeXGBoostRegressor
from RuleTree.utils.data_utils import json_numpy_encoder
from RuleTree.utils.define import DATA_TYPE_TABULAR


@pytest.fixture(scope="module")
def targets(diabetes):
    """Two targets on very different scales: the disease progression and the body mass index."""
    X, y = diabetes
    return X, np.column_stack([y, X[:, 2]])


def json_round_trip(dictionary):
    return json.loads(json.dumps(dictionary, cls=json_numpy_encoder))


def test_same_tree_as_scikit_learn(targets):
    X, Y = targets

    model = RuleTreeRegressor(max_depth=3, base_stumps=DecisionTreeStumpRegressor(random_state=0)).fit(X, Y)
    reference = DecisionTreeRegressor(max_depth=3, random_state=0).fit(X, Y)

    assert model.n_targets_ == 2
    assert model.predict(X).shape == Y.shape
    assert np.allclose(model.predict(X), reference.predict(X))
    assert np.allclose(model.root.prediction, Y.mean(axis=0))
    assert np.allclose(model.root.prediction_probability, Y.std(axis=0))


REGRESSORS = {
    "RuleTreeRegressor": lambda: RuleTreeRegressor(max_depth=2, random_state=0),
    "RuleTreeClusterRegressor": lambda: RuleTreeClusterRegressor(max_depth=2, random_state=0),
    "RuleTreeXGBoostRegressor": lambda: RuleTreeXGBoostRegressor(max_depth=2, base_stumps=XGBoostStumpRegressor(lam=1)),
    "GBoostRegressor": lambda: GBoostRegressor(base_estimator=RuleTreeRegressor(max_depth=2, random_state=0),
                                               n_estimators=2),
    "XGBoostRegressor": lambda: XGBoostRegressor(n_estimators=2),
    "EBMRegressor": lambda: EBMRegressor(base_estimator=RuleTreeRegressor(max_depth=1, random_state=0),
                                         n_iterations=1),
}


@pytest.mark.parametrize("make_model", REGRESSORS.values(), ids=REGRESSORS.keys())
@pytest.mark.parametrize("target_shape", ["vector", "column", "list"])
def test_one_target_predicts_a_vector(make_model, target_shape, targets):
    """With one target, also given as a column (as scikit-learn) or as a list, the predictions are a vector."""
    X, Y = targets
    y = {"vector": Y[:, 0], "column": Y[:, :1], "list": Y[:, 0].tolist()}[target_shape]

    model = make_model().fit(X[:, :4], y)

    assert np.asarray(model.predict(X[:, :4])).shape == (len(X),)
    if hasattr(model, "root"):
        assert isinstance(model.root.prediction, float)


@pytest.mark.parametrize("leaf_prediction, statistic", [("mean", np.mean), ("median", np.median),
                                                        (lambda y: np.percentile(y, 90, axis=0),
                                                         lambda y, axis: np.percentile(y, 90, axis=axis))],
                         ids=["mean", "median", "callable"])
def test_leaf_prediction(leaf_prediction, statistic, targets):
    X, Y = targets
    model = RuleTreeRegressor(max_depth=2, random_state=0, leaf_prediction=leaf_prediction).fit(X, Y)

    leaves = model.apply(X)
    for leaf_id, leaf in model.get_leaf_nodes().items():
        assert np.allclose(leaf.prediction, statistic(Y[leaves == leaf_id], axis=0))

    one_target = RuleTreeRegressor(max_depth=2, random_state=0, leaf_prediction=leaf_prediction).fit(X, Y[:, 0])
    assert np.isclose(one_target.root.prediction, statistic(Y[:, 0], axis=0))


def test_unknown_leaf_prediction(targets):
    X, Y = targets

    with pytest.raises(ValueError, match="leaf_prediction"):
        RuleTreeRegressor(leaf_prediction="mode").fit(X, Y)


def test_balanced_target_weights(targets):
    """Each target counts the same: as the tree fitted on the standardized targets."""
    X, Y = targets
    standardized = (Y - Y.mean(axis=0)) / Y.std(axis=0)

    balanced = RuleTreeRegressor(max_depth=3, random_state=0, target_weights="balanced").fit(X, Y)
    on_standardized = RuleTreeRegressor(max_depth=3, random_state=0).fit(X, standardized)
    unweighted = RuleTreeRegressor(max_depth=3, random_state=0).fit(X, Y)

    assert np.array_equal(balanced.apply(X), on_standardized.apply(X))
    assert not np.array_equal(balanced.apply(X), unweighted.apply(X))  # the large target dominates otherwise
    assert np.allclose(balanced.root.prediction, Y.mean(axis=0))  # predictions in the original scale
    r2 = {name: evaluate_regression(Y, model.predict(X), multioutput="raw_values")["r2"]
          for name, model in [("balanced", balanced), ("unweighted", unweighted)]}
    assert r2["balanced"][1] > r2["unweighted"][1]


def test_target_weights_array(targets):
    """Weights on the squared errors: as the tree fitted on the targets multiplied by their square root."""
    X, Y = targets
    weights = np.array([1.0, 1e7])

    weighted = RuleTreeRegressor(max_depth=3, random_state=0, target_weights=weights).fit(X, Y)
    scaled = RuleTreeRegressor(max_depth=3, random_state=0).fit(X, Y * np.sqrt(weights))

    assert np.array_equal(weighted.apply(X), scaled.apply(X))
    assert np.allclose(weighted.target_scales_, np.sqrt(weights))


@pytest.mark.parametrize("target_weights", ["inverse", [1.0], [1.0, -1.0]])
def test_invalid_target_weights(target_weights, targets):
    X, Y = targets

    with pytest.raises(ValueError, match="target_weights"):
        RuleTreeRegressor(target_weights=target_weights).fit(X, Y)


def test_target_weights_ignored_with_one_target(targets):
    X, Y = targets

    model = RuleTreeRegressor(max_depth=2, random_state=0, target_weights="balanced").fit(X, Y[:, 0])

    assert model.target_scales_ is None


def test_saving_pruning_printing_and_updating(targets, capsys):
    X, Y = targets
    model = RuleTreeRegressor(max_depth=4, prune_useless_leaves=True, random_state=0,
                              target_weights="balanced").fit(X, Y)

    loaded = RuleTree.from_dict(json_round_trip(model.to_dict()))
    assert np.allclose(loaded.predict(X), model.predict(X))
    assert clone(model).get_params()["target_weights"] == "balanced"

    model.print_rules(model.get_rules())
    assert "output: [" in capsys.readouterr().out

    first_leaf = model.apply(X) == model.apply(X)[0]
    model.update_statistics(X[first_leaf], Y[first_leaf])
    predictions = model.predict(X)
    assert np.allclose(predictions[first_leaf], Y[first_leaf].mean(axis=0))
    assert np.isnan(predictions[~first_leaf]).all()  # the other leaves get no instance

    loaded.set_params(max_depth=5).partial_fit(X, Y)
    assert loaded.predict(X).shape == Y.shape


@pytest.mark.parametrize("stump", [DecisionTreeStumpRegressor(), PivotTreeStumpRegressor(),
                                   ObliqueDecisionTreeStumpRegressor(),
                                   ObliqueDecisionTreeStumpRegressor("bivariate", n_orientations=4)],
                         ids=["DecisionTree", "Pivot", "Oblique-householder", "Oblique-bivariate"])
def test_stumps_with_several_targets(stump, targets):
    X, Y = targets
    data = {DATA_TYPE_TABULAR: X}

    fitted = clone(stump).fit(data, Y, data_type=DATA_TYPE_TABULAR)
    loaded = type(fitted).dict_to_node(json_round_trip(fitted.node_to_dict()))
    assert set(fitted.apply(data)) == {1, 2}
    assert np.array_equal(loaded.apply(data), fitted.apply(data))

    model = RuleTreeRegressor(max_depth=2, random_state=0, base_stumps=stump).fit(X, Y)
    assert model.predict(X).shape == Y.shape


@pytest.mark.parametrize("make_model, data", [
    (lambda: RuleTreeRegressor(base_stumps=XGBoostStumpRegressor(lam=1)), "tabular"),
    (lambda: RuleTreeRegressor(base_stumps=FairTreeStumpRegressor(penalty="balance", sensible_attribute=1,
                                                                  n_jobs=1)), "tabular"),
    (lambda: RuleTreeRegressor(base_stumps=ShapeletTreeStumpRegressor(n_shapelets=3, sliding_window=5)), "ts"),
    (lambda: RuleTreeRegressor(base_stumps=ProximityTreeStumpRegressor(n_shapelets=3, sliding_window=5)), "ts"),
], ids=["XGBoost", "Fair", "Shapelet-mi_reg", "Proximity-mi_reg"])
def test_components_with_a_single_target(make_model, data, targets):
    X, Y = targets
    inputs = {"X": X} if data == "tabular" else {"X_ts": np.random.default_rng(0).normal(size=(len(Y), 1, 20))}

    with pytest.raises(ValueError, match="supports a single target"):
        make_model().fit(y=Y, **inputs)


@pytest.mark.parametrize("stump_class", [ShapeletTreeStumpRegressor, ProximityTreeStumpRegressor])
def test_shapelet_stumps_with_several_targets(stump_class, targets):
    """Without the mutual information, the shapelets are drawn from all the time series: any target works."""
    _, Y = targets
    X_ts = np.random.default_rng(0).normal(size=(len(Y), 1, 20))

    model = RuleTreeRegressor(max_depth=2, base_stumps=stump_class(n_shapelets=3, sliding_window=5,
                                                                   selection="random")).fit(X_ts=X_ts, y=Y)

    assert model.predict(X_ts=X_ts).shape == Y.shape


def test_local_interpretation_needs_one_target(targets):
    X, Y = targets

    with pytest.raises(ValueError, match="single target"):
        RuleTreeRegressor(max_depth=2, random_state=0).fit(X, Y).local_interpretation(X)


def test_cluster_regressor_and_ensembles(targets):
    X, Y = targets

    cluster = RuleTreeClusterRegressor(max_depth=2, random_state=0).fit(X, Y)
    gboost = GBoostRegressor(base_estimator=RuleTreeRegressor(max_depth=2, random_state=0), n_estimators=5,
                             random_state=0).fit(X, Y)
    ebm = EBMRegressor(base_estimator=RuleTreeRegressor(max_depth=1, random_state=0), n_iterations=2,
                       random_state=0).fit(X[:, :3], Y)

    for predictions in (cluster.predict(X), gboost.predict(X), ebm.predict(X[:, :3])):
        assert predictions.shape == Y.shape
    assert np.allclose(gboost.base_prediction_.shape, (2,))
    assert gboost.score(X, Y) > 0  # r2, uniform average over the targets


def test_scikit_learn_tools(targets):
    X, Y = targets

    scores = cross_val_score(RuleTreeRegressor(max_depth=3, random_state=0, target_weights="balanced"), X, Y, cv=3)

    assert np.all(scores > 0)


def test_evaluate_regression_with_several_targets(targets):
    X, Y = targets
    predictions = RuleTreeRegressor(max_depth=3, random_state=0).fit(X, Y).predict(X)

    raw = evaluate_regression(Y, predictions, multioutput="raw_values")
    average = evaluate_regression(Y, predictions)
    weighted = evaluate_regression(Y, predictions, multioutput=[3, 1])

    for name, value in raw.items():
        if name == "mean_squared_log_error":
            continue  # nan: the second target has negative values
        assert value.shape == (2,), name
        assert np.isclose(average[name], value.mean()), name
        assert np.isclose(weighted[name], np.average(value, weights=[3, 1])), name
    assert np.allclose(raw["max_error"], np.abs(Y - predictions).max(axis=0))
    assert np.isnan(raw["mean_squared_log_error"]).all()
