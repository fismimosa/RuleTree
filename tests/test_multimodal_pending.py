"""
Features currently broken by the unfinished multimodal API (X, X_ts, X_img, X_txt) and by the fit/context design.

They are marked as strict xfail: when the API is completed these tests start passing,
pytest reports them as XPASS(strict) failures, and the xfail marker must be removed.
"""
import numpy as np
import pytest

from RuleTree import RuleTreeClassifier, RuleTreeRegressor
from RuleTree.ensemble.GBoostClassifier import GBoostClassifier
from RuleTree.ensemble.XGBoostRegressor import XGBoostRegressor
from RuleTree.stumps.classification import (PartialPivotTreeStumpClassifier, PivotTreeStumpClassifier,
                                            ShapeletTreeStumpClassifier)
from RuleTree.stumps.regression import FairTreeStumpRegressor, PivotTreeStumpRegressor

MISSING_RESOLVE = pytest.mark.xfail(strict=True, raises=AttributeError,
                                    reason="RuleTreeBase._resolve_data_input is not implemented")


@MISSING_RESOLVE
def test_apply(iris):
    X, y = iris
    model = RuleTreeClassifier(max_depth=2).fit(X, y)

    assert set(model.apply(X)) == set(model.get_leaf_nodes())


@MISSING_RESOLVE
def test_local_interpretation(iris):
    X, y = iris
    model = RuleTreeClassifier(max_depth=2).fit(X, y)

    model.local_interpretation(X)


@MISSING_RESOLVE
def test_update_statistics(iris):
    X, y = iris
    model = RuleTreeClassifier(max_depth=2).fit(X, y)

    model.update_statistics(X, y)


@MISSING_RESOLVE
def test_pivot_classifier(iris):
    X, y = iris
    model = RuleTreeClassifier(max_depth=2, base_stumps=PivotTreeStumpClassifier()).fit(X, y)

    assert np.mean(model.predict(X) == y) > 0.9


@pytest.mark.xfail(strict=True, raises=IndexError,
                   reason="context caches the feature types of X, but the inner fit runs on the transformed features")
def test_partial_pivot_classifier(iris):
    X, y = iris
    model = RuleTreeClassifier(max_depth=2, base_stumps=PartialPivotTreeStumpClassifier()).fit(X, y)

    assert np.mean(model.predict(X) == y) > 0.9


@MISSING_RESOLVE
def test_pivot_regressor(diabetes):
    X, y = diabetes
    model = RuleTreeRegressor(max_depth=2, base_stumps=PivotTreeStumpRegressor()).fit(X, y)

    assert model.predict(X).shape == y.shape


@MISSING_RESOLVE
def test_shapelet_classifier():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(60, 1, 40))
    y = rng.integers(0, 2, size=60)
    stump = ShapeletTreeStumpClassifier(n_shapelets=4, sliding_window=10, selection="random")

    model = RuleTreeClassifier(max_depth=2, base_stumps=stump).fit(X, y)

    assert model.predict(X).shape == y.shape


@pytest.mark.xfail(strict=True, raises=TypeError, reason="FairTreeStumpRegressor.fit has the old signature")
def test_fair_regressor(diabetes):
    X, y = diabetes
    X = np.column_stack([X, X[:, 1] > 0])
    stump = FairTreeStumpRegressor(penalty="balance", sensible_attribute=-1, n_jobs=1)

    model = RuleTreeRegressor(max_depth=2, base_stumps=stump).fit(X, y)

    assert model.predict(X).shape == y.shape


@MISSING_RESOLVE
def test_gboost_classifier(iris):
    X, y = iris
    y = (y == 1).astype(int)

    model = GBoostClassifier(n_estimators=5).fit(X, y)

    assert set(model.predict(X)) <= {0, 1}


@MISSING_RESOLVE
def test_xgboost_regressor(diabetes):
    X, y = diabetes
    model = XGBoostRegressor(n_estimators=3).fit(X, y)

    assert model.predict(X).shape == y.shape
