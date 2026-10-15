import warnings

import numpy as np
import pytest

from sklearn.metrics import r2_score

from RuleTree import RuleTreeRegressor
from RuleTree.ensemble.EBMRegressor import EBMRegressor
from RuleTree.ensemble.GBoostClassifier import GBoostClassifier
from RuleTree.ensemble.GBoostRegressor import GBoostRegressor
from RuleTree.ensemble.XGBoostRegressor import XGBoostRegressor
from RuleTree.stumps.regression.XGBoostStumpRegressor import XGBoostStumpRegressor
from RuleTree.tree.RuleTreeXGBoostRegressor import RuleTreeXGBoostRegressor


def test_gboost_regressor(diabetes):
    X, y = diabetes
    model = GBoostRegressor(base_estimator=RuleTreeRegressor(max_depth=2), n_estimators=10).fit(X, y)

    assert r2_score(y, model.predict(X)) > 0.4


def test_ebm_regressor(diabetes):
    X, y = diabetes
    model = EBMRegressor(base_estimator=RuleTreeRegressor(max_depth=2), n_iterations=2, random_state=0).fit(X, y)

    assert model.predict(X).shape == y.shape


def test_xgboost_regressor_default_base_estimator():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        model = XGBoostRegressor()
        base_estimator = model._get_base_estimator()

    assert model.get_params()["base_estimator"] is None
    assert isinstance(base_estimator, RuleTreeXGBoostRegressor)
    assert isinstance(base_estimator.base_stumps, XGBoostStumpRegressor)


@pytest.mark.slow
def test_gboost_classifier(iris):
    X, y = iris
    y = (y == 1).astype(int)

    model = GBoostClassifier(base_estimator=RuleTreeRegressor(max_depth=2), n_estimators=5).fit(X, y)

    assert set(model.predict(X)) <= {0, 1}


def test_xgboost_regressor(diabetes):
    X, y = diabetes
    model = XGBoostRegressor(n_estimators=3).fit(X, y)

    assert model.predict(X).shape == y.shape


def test_xgboost_tree_update_statistics(diabetes):
    X, y = diabetes
    model = RuleTreeXGBoostRegressor(max_depth=2, base_stumps=XGBoostStumpRegressor(lam=1)).fit(X, y)
    predictions = model.predict(X)

    model.update_statistics(X, y)

    assert np.allclose(model.predict(X), predictions)  # XGBoost output values, as in fit


def test_early_stopping_with_patience_one(diabetes):
    """patience=1 removes no estimator (slicing with [:-0] removed all of them)."""
    X, y = diabetes
    gboost = GBoostRegressor(base_estimator=RuleTreeRegressor(max_depth=2, random_state=0), n_estimators=200,
                             patience=1, random_state=0).fit(X, y)
    ebm = EBMRegressor(base_estimator=RuleTreeRegressor(max_depth=1, random_state=0), n_iterations=200,
                       learning_rate=.5, patience=1, random_state=0).fit(X, y)

    assert 0 < len(gboost.estimators_) < 200
    assert all(0 < len(estimators) < 200 for estimators in ebm.estimators_.values())


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
