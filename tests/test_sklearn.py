import numpy as np
import pytest
from sklearn.base import is_classifier, is_clusterer, is_regressor
from sklearn.model_selection import GridSearchCV, cross_val_score

from RuleTree import (RuleTreeClassifier, RuleTreeCluster, RuleTreeClusterClassifier, RuleTreeClusterRegressor,
                      RuleTreeRegressor)
from RuleTree.ensemble.EBMRegressor import EBMRegressor
from RuleTree.ensemble.GBoostClassifier import GBoostClassifier
from RuleTree.ensemble.GBoostRegressor import GBoostRegressor
from RuleTree.ensemble.XGBoostRegressor import XGBoostRegressor


@pytest.mark.parametrize("estimator, kind", [
    (RuleTreeClassifier(), "classifier"),
    (RuleTreeClusterClassifier(), "classifier"),
    (GBoostClassifier(), "classifier"),
    (RuleTreeRegressor(), "regressor"),
    (RuleTreeClusterRegressor(), "regressor"),
    (GBoostRegressor(), "regressor"),
    (XGBoostRegressor(), "regressor"),
    (EBMRegressor(), "regressor"),
    (RuleTreeCluster(), "clusterer"),
], ids=lambda value: type(value).__name__ if not isinstance(value, str) else value)
def test_estimator_type(estimator, kind):
    """sklearn tools (e.g. cross_val_score, scoring) behave according to the type of estimator."""
    assert is_classifier(estimator) == (kind == "classifier")
    assert is_regressor(estimator) == (kind == "regressor")
    assert is_clusterer(estimator) == (kind == "clusterer")


def test_cross_val_score(iris, diabetes):
    X, y = iris
    accuracy = cross_val_score(RuleTreeClassifier(max_depth=3, random_state=0), X, y, cv=3)
    assert np.all(accuracy > 0.8)

    X, y = diabetes
    r2 = cross_val_score(RuleTreeRegressor(max_depth=3, random_state=0), X, y, cv=3)
    assert np.all(r2 > 0)


def test_grid_search(iris):
    X, y = iris

    search = GridSearchCV(RuleTreeClassifier(random_state=0), {"max_depth": [1, 3]}, cv=3).fit(X, y)

    assert search.best_params_ == {"max_depth": 3}


def test_clone_keeps_the_parameters_of_the_xgboost_estimators():
    """All the parameters are in the signature: get_params and clone do not lose them."""
    from sklearn.base import clone

    from RuleTree.stumps.regression.XGBoostStumpRegressor import XGBoostStumpRegressor
    from RuleTree.tree.RuleTreeXGBoostRegressor import RuleTreeXGBoostRegressor

    ensemble = clone(XGBoostRegressor(n_estimators=7, patience=2, val_size=.2, random_state=0))
    assert (ensemble.n_estimators, ensemble.patience, ensemble.val_size, ensemble.random_state) == (7, 2, .2, 0)

    tree = clone(RuleTreeXGBoostRegressor(base_stumps=XGBoostStumpRegressor(lam=1), min_samples_split=50,
                                          max_leaf_nodes=4, random_state=0))
    assert (tree.min_samples_split, tree.max_leaf_nodes, tree.random_state) == (50, 4, 0)
