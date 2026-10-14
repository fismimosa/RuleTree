import warnings

from sklearn.metrics import r2_score

from RuleTree import RuleTreeRegressor
from RuleTree.ensemble.EBMRegressor import EBMRegressor
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
