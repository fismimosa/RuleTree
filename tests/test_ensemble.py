import warnings

import numpy as np
import pytest

from sklearn.metrics import r2_score

from RuleTree import RuleTreeRegressor
from RuleTree.ensemble.EBMRegressor import EBMRegressor
from RuleTree.ensemble.GBoostClassifier import GBoostClassifier
from RuleTree.ensemble.GBoostRegressor import GBoostRegressor
from RuleTree.ensemble.XGBoostRegressor import XGBoostRegressor
from RuleTree.ensemble._boosting import validation_split
from RuleTree.stumps.regression.XGBoostStumpRegressor import XGBoostStumpRegressor
from RuleTree.tree.RuleTreeXGBoostRegressor import RuleTreeXGBoostRegressor


def test_gboost_regressor(diabetes):
    X, y = diabetes
    model = GBoostRegressor(base_estimator=RuleTreeRegressor(max_depth=2), n_estimators=10).fit(X, y)

    assert r2_score(y, model.predict(X)) > 0.4


def test_xgboost_regressor_default_base_estimator():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        model = XGBoostRegressor()
        base_estimator = model._get_base_estimator()

    assert model.get_params()["base_estimator"] is None
    assert isinstance(base_estimator, RuleTreeXGBoostRegressor)
    assert isinstance(base_estimator.base_stumps, XGBoostStumpRegressor)


def test_xgboost_tree_update_statistics(diabetes):
    X, y = diabetes
    model = RuleTreeXGBoostRegressor(max_depth=2, base_stumps=XGBoostStumpRegressor(lam=1)).fit(X, y)
    predictions = model.predict(X)

    model.update_statistics(X, y)

    assert np.allclose(model.predict(X), predictions)  # XGBoost output values, as in fit


@pytest.mark.parametrize("patience", [1, 3])
def test_early_stopping_keeps_only_the_estimators_that_improve(diabetes, patience):
    """After early stopping, the last estimator kept decreased the validation error."""
    X, y = diabetes
    _, X_val, _, y_val = validation_split(X, y, val_size=.15, random_state=0)  # as GBoostRegressor and EBMRegressor

    gboost = GBoostRegressor(base_estimator=RuleTreeRegressor(max_depth=2, random_state=0), n_estimators=200,
                             learning_rate=.5, patience=patience, random_state=0).fit(X, y)
    steps = [gboost.learning_rate * est.predict(X_val) for est in gboost.estimators_]
    predictions = gboost.base_prediction_ + np.cumsum([np.zeros(len(y_val))] + steps, axis=0)
    errors = np.abs(y_val - predictions).sum(axis=1)
    assert 0 < len(gboost.estimators_) < 200
    assert errors[-1] <= errors[-2]

    ebm = EBMRegressor(base_estimator=RuleTreeRegressor(max_depth=1, random_state=0), n_iterations=200,
                       learning_rate=.5, patience=patience, random_state=0).fit(X, y)
    n_iterations = len(ebm.estimators_[0])
    steps = [sum(ebm.learning_rate * estimators[i].predict(X_val[:, feature_idx].reshape(len(y_val), -1))
                 for feature_idx, estimators in ebm.estimators_.items())
             for i in range(n_iterations)]
    errors = np.abs(y_val - np.cumsum([np.zeros(len(y_val))] + steps, axis=0)).sum(axis=1)
    assert 0 < n_iterations < 200
    assert errors[-1] <= errors[-2]


@pytest.mark.parametrize("patience", [1, 3])
def test_gboost_classifier_early_stopping_keeps_only_the_estimators_that_improve(iris, patience):
    """As for the regressors: the last estimator kept for a class decreased its validation error."""
    from scipy.special import expit
    from sklearn.model_selection import train_test_split

    from RuleTree.ensemble.GBoostClassifier import _train_one_class

    X, y = iris
    base_log_odds = np.log((y == 1).sum() / (y != 1).sum())
    args = (1, 1, X, y, RuleTreeRegressor(max_depth=3, random_state=0), expit(base_log_odds), base_log_odds, 2.,
            200, .15, patience, 0, 0)
    _, estimators = _train_one_class(args)

    _, X_val, _, y_val = train_test_split(X, y, test_size=.15, random_state=0, stratify=y)
    steps = [2. * np.vectorize(gamma_map.get)(est.apply(X_val)) for est, gamma_map in estimators]
    log_odds = base_log_odds + np.cumsum([np.zeros(len(y_val))] + steps, axis=0)
    errors = np.abs((y_val == 1) - expit(log_odds)).mean(axis=1)
    assert 0 < len(estimators) < 200
    assert errors[-1] < errors[-2]


def test_gboost_classifier_any_labels_in_the_current_process(iris, monkeypatch):
    """Classes other than 0, ..., n - 1; with n_jobs=1 no process pool is created."""
    import RuleTree.ensemble.GBoostClassifier as gboost_module

    def no_pool(*args, **kwargs):
        raise AssertionError("n_jobs=1 must not create a process pool")

    monkeypatch.setattr(gboost_module, "ProcessPoolExecutor", no_pool)
    X, y = iris
    labels = np.array(["setosa", "versicolor", "virginica"])[y]

    model = GBoostClassifier(base_estimator=RuleTreeRegressor(max_depth=2, random_state=0), n_estimators=5,
                             n_jobs=1, random_state=0).fit(X, labels)

    assert np.mean(model.predict(X) == labels) > 0.9
