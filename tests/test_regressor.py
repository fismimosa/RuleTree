import warnings

import numpy as np
import pytest
from sklearn.metrics import r2_score

from RuleTree import RuleTreeRegressor


def test_fit_predict(diabetes):
    X, y = diabetes
    model = RuleTreeRegressor(max_depth=3, random_state=0).fit(X, y)

    y_pred = model.predict(X)

    assert y_pred.shape == y.shape
    assert r2_score(y, y_pred) > 0.4


def test_leaves_predict_training_mean(diabetes):
    X, y = diabetes
    model = RuleTreeRegressor(max_depth=2).fit(X, y)

    _, leaves, _ = model.root.predict(X)

    for leaf_id, leaf in model.get_leaf_nodes().items():
        assert leaf.prediction == pytest.approx(y[leaves == leaf_id].mean())


def test_no_warnings(diabetes):
    X, y = diabetes
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        RuleTreeRegressor(max_depth=4).fit(X, y).predict(X)


@pytest.mark.parametrize("max_depth", [1, 2, 4])
def test_max_depth(diabetes, max_depth):
    X, y = diabetes
    model = RuleTreeRegressor(max_depth=max_depth).fit(X, y)

    assert max(leaf.get_depth() for leaf in model.get_leaf_nodes().values()) <= max_depth


def test_max_leaf_nodes(diabetes):
    X, y = diabetes
    model = RuleTreeRegressor(max_leaf_nodes=4).fit(X, y)

    assert len(model.get_leaf_nodes()) <= 4
    assert len(np.unique(model.predict(X))) <= 4


def test_categorical_features(iris_mixed, iris):
    X, _ = iris_mixed
    y = iris[0][:, 2]  # petal length, from which the categorical feature is derived

    model = RuleTreeRegressor(max_depth=1).fit(X, y)

    assert model.root.stump.is_categorical
    assert r2_score(y, model.predict(X)) > 0.5
