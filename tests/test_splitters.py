import json
import warnings

import numpy as np
import pytest
import sklearn

from RuleTree.exceptions import NoSplitFoundWarning
from RuleTree.stumps.splitters import ClassificationSplitter, FairSplitter, RegressionSplitter, XGBoostSplitter
from RuleTree.utils.data_utils import entropy, json_numpy_encoder
from RuleTree.utils.define import MODEL_TYPE_CLF, MODEL_TYPE_REG

# column 0: noise, column 1: separates the classes at 5, column 2: categorical, "a" for the class 1
X = np.array([[0.3, 1, "b"], [0.9, 2, "b"], [0.1, 3, "b"], [0.7, 4, "b"],
              [0.2, 6, "a"], [0.8, 7, "a"], [0.5, 8, "a"], [0.6, 9, "b"]], dtype=object)
y_clf = np.array([0, 0, 0, 0, 1, 1, 1, 1])
y_reg = np.array([1.0, 1.1, 0.9, 1.0, 5.0, 5.2, 4.8, 5.0])
NUMERICAL, CATEGORICAL = [0, 1], [2]


@pytest.mark.parametrize("splitter, y", [(ClassificationSplitter(), y_clf), (RegressionSplitter(), y_reg),
                                         (XGBoostSplitter(lam=1), y_reg)], ids=["clf", "reg", "xgboost"])
def test_numerical_split(splitter, y):
    splitter.fit(X, y, NUMERICAL, CATEGORICAL)

    assert splitter.feature_ == 1
    assert splitter.threshold_ == 5
    assert not splitter.is_categorical_
    assert splitter.gain_ > 0
    assert np.array_equal(splitter.apply(X), np.where(X[:, 1] <= 5, 1, 2))


@pytest.mark.parametrize("splitter, y", [(ClassificationSplitter(), y_clf), (RegressionSplitter(), y_reg)],
                         ids=["clf", "reg"])
def test_categorical_split(splitter, y):
    """Without the numerical features, the best split is on the categorical one."""
    splitter.fit(X, y, [], CATEGORICAL)

    assert splitter.feature_ == 2
    assert splitter.is_categorical_
    assert splitter.threshold_ in ("a", "b")
    assert set(splitter.apply(X)) == {1, 2}


@pytest.mark.parametrize("splitter, y", [(ClassificationSplitter(), y_clf), (RegressionSplitter(), y_reg)],
                         ids=["clf", "reg"])
def test_categorical_split_when_no_numerical_split(splitter, y):
    """A numerical feature that cannot split (constant) does not prevent the categorical splits."""
    X_constant = X.copy()
    X_constant[:, 0] = 1.0

    splitter.fit(X_constant, y, [0], CATEGORICAL)

    assert splitter.feature_ == 2
    assert splitter.is_categorical_

@pytest.mark.parametrize("splitter", [ClassificationSplitter(), RegressionSplitter(),
                                      XGBoostSplitter(lam=1, min_cover=len(X))], ids=["clf", "reg", "xgboost"])
def test_no_split(splitter):
    with pytest.raises(NoSplitFoundWarning):
        splitter.fit(X[:, :2].astype(float), np.ones(len(X)), [0, 1], [])


def test_binary_feature_threshold_halfway():
    X_binary = np.array([[0.0], [0.0], [1.0], [1.0]])

    splitter = ClassificationSplitter().fit(X_binary, np.array([0, 0, 1, 1]), [0], [])

    assert splitter.threshold_ == 0.5


def test_fair_split():
    rng = np.random.default_rng(0)
    X_fair = np.column_stack([rng.normal(size=60), rng.normal(size=60), rng.integers(0, 2, size=60)])
    y = X_fair[:, 0] * 3 + rng.normal(scale=0.1, size=60)

    splitter = FairSplitter(penalty="balance", sensible_attribute=2, n_jobs=1).fit(X_fair, y, [0, 1, 2], [])

    assert splitter.feature_ == 0
    assert set(splitter.apply(X_fair)) == {1, 2}


@pytest.mark.parametrize("sensible_attribute", [2, -1])
def test_fair_split_never_uses_the_sensitive_attribute(sensible_attribute):
    """Also with a negative index: the sensitive attribute is excluded even if it predicts the target."""
    rng = np.random.default_rng(0)
    X_fair = np.column_stack([rng.normal(size=60), rng.normal(size=60), rng.integers(0, 2, size=60)])
    y = X_fair[:, 2] * 10 + rng.normal(scale=0.1, size=60)

    splitter = FairSplitter(penalty="balance", sensible_attribute=sensible_attribute, penalization_weight=0, n_jobs=1)
    splitter.fit(X_fair, y, [0, 1, 2], [])

    assert splitter.feature_ != 2


def test_update_statistics():
    splitter = ClassificationSplitter().fit(X, y_clf, NUMERICAL, CATEGORICAL)
    assert splitter.impurity_[1:] == [0, 0]  # pure children

    splitter.update_statistics(X[:6], np.array([0, 1, 0, 0, 1, 1]))

    assert splitter.impurity_[1] > 0


@pytest.mark.parametrize("splitter, y", [(ClassificationSplitter(), y_clf), (RegressionSplitter(), y_reg),
                                         (XGBoostSplitter(lam=1), y_reg)], ids=["clf", "reg", "xgboost"])
def test_state_round_trip(splitter, y):
    splitter.fit(X, y, NUMERICAL, CATEGORICAL)
    state = json.loads(json.dumps(splitter.get_state(), cls=json_numpy_encoder))

    loaded = sklearn.clone(splitter).set_state(state)

    assert np.array_equal(loaded.apply(X), splitter.apply(X))
    assert loaded.gain_ == pytest.approx(splitter.gain_)


def test_task():
    assert ClassificationSplitter.task == MODEL_TYPE_CLF
    assert {RegressionSplitter.task, XGBoostSplitter.task, FairSplitter.task} == {MODEL_TYPE_REG}


def test_parameters():
    splitter = ClassificationSplitter(criterion="entropy", min_samples_leaf=3)

    assert sklearn.clone(splitter).get_params() == splitter.get_params()
    assert splitter.get_params()["min_samples_leaf"] == 3


def test_entropy_with_classes_without_instances():
    """With class_weight, the impurity also counts the classes absent from the node."""
    y = np.array([0, 0, 1])

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        impurity = entropy(y, None, {0: 1.0, 1: 1.0, 2: 1.0})

    assert impurity == pytest.approx(entropy(y))


def test_mean_squared_error_as_scikit_learn():
    """The impurity of the regression splits gives the same numbers as sklearn.metrics.mean_squared_error."""
    from sklearn.metrics import mean_squared_error

    from RuleTree.stumps.splitters.RegressionSplitter import _mean_squared_error

    rng = np.random.default_rng(0)
    for shape in [(1,), (500,), (300, 3)]:
        for scale in (1e-3, 1, 1e4):
            y = rng.normal(scale=scale, size=shape)
            prediction = np.broadcast_to(y.mean(axis=0), y.shape)
            assert _mean_squared_error(y, prediction) == mean_squared_error(y, prediction)
