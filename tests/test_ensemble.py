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
