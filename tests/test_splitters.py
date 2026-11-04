import json
import warnings

import numpy as np
import pytest
import sklearn

from RuleTree.exceptions import NoSplitFoundWarning
from RuleTree.stumps.splitters import ClassificationSplitter, FairSplitter, RegressionSplitter, XGBoostSplitter
from RuleTree.utils.data_utils import json_numpy_encoder
from RuleTree.utils.define import MODEL_TYPE_CLF, MODEL_TYPE_REG
from RuleTree.utils.impurity import class_weights, entropy

# column 0: noise, column 1: separates the classes at 5, column 2: categorical, "a" for the class 1
X = np.array([[0.3, 1, "b"], [0.9, 2, "b"], [0.1, 3, "b"], [0.7, 4, "b"],
              [0.2, 6, "a"], [0.8, 7, "a"], [0.5, 8, "a"], [0.6, 9, "b"]], dtype=object)
y_clf = np.array([0, 0, 0, 0, 1, 1, 1, 1])
y_reg = np.array([1.0, 1.1, 0.9, 1.0, 5.0, 5.2, 4.8, 5.0])
NUMERICAL, CATEGORICAL = [0, 1], [2]


@pytest.mark.parametrize("splitter, y", [(ClassificationSplitter(), y_clf), (RegressionSplitter(), y_reg),
                                         (XGBoostSplitter(lam=1), y_reg)], ids=["clf", "reg", "xgboost"])
def test_numerical_split(splitter, y):
    splitter.fit(X, y, NUMERICAL, CATEGORICAL, context=None, node_id=None, cache=None)

    assert splitter.feature_ == 1
    assert splitter.threshold_ == 5
    assert not splitter.is_categorical_
    assert splitter.gain_ > 0
    assert np.array_equal(splitter.apply(X), np.where(X[:, 1] <= 5, 1, 2))


@pytest.mark.parametrize("splitter, y", [(ClassificationSplitter(), y_clf), (RegressionSplitter(), y_reg)],
                         ids=["clf", "reg"])
def test_categorical_split(splitter, y):
    """Without the numerical features, the best split is on the categorical one."""
    splitter.fit(X, y, [], CATEGORICAL, context=None, node_id=None, cache=None)

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

    splitter.fit(X_constant, y, [0], CATEGORICAL, context=None, node_id=None, cache=None)

    assert splitter.feature_ == 2
    assert splitter.is_categorical_

@pytest.mark.parametrize("splitter", [ClassificationSplitter(), RegressionSplitter(),
                                      XGBoostSplitter(lam=1, min_cover=len(X))], ids=["clf", "reg", "xgboost"])
def test_no_split(splitter):
    with pytest.raises(NoSplitFoundWarning):
        splitter.fit(X[:, :2].astype(float), np.ones(len(X)), [0, 1], [], context=None, node_id=None, cache=None)


@pytest.mark.parametrize("splitter, y", [(ClassificationSplitter(), np.array([0, 0, 1, 1])),
                                         (RegressionSplitter(), np.array([1.0, 1.1, 5.0, 5.2]))], ids=["clf", "reg"])
def test_binary_feature_threshold_halfway(splitter, y):
    """Not the threshold of scikit-learn, computed on the values in float32."""
    X_binary = np.array([[0.1], [0.1], [0.7], [0.7]])

    splitter.fit(X_binary, y, [0], [], context=None, node_id=None, cache=None)

    assert splitter.threshold_ == (0.1 + 0.7) / 2


@pytest.mark.parametrize("splitter, y", [(ClassificationSplitter(), y_clf), (RegressionSplitter(), y_reg)],
                         ids=["clf", "reg"])
def test_no_degenerate_categorical_split(splitter, y):
    """A categorical value of all the instances would leave the right child empty."""
    with pytest.raises(NoSplitFoundWarning):
        splitter.fit(np.full((len(y), 1), "a", dtype=object), y, [], [0], context=None, node_id=None, cache=None)


def test_fair_split():
    rng = np.random.default_rng(0)
    X_fair = np.column_stack([rng.normal(size=60), rng.normal(size=60), rng.integers(0, 2, size=60)])
    y = X_fair[:, 0] * 3 + rng.normal(scale=0.1, size=60)

    splitter = FairSplitter(penalty="balance", sensible_attribute=2)
    splitter = splitter.fit(X_fair, y, [0, 1, 2], [], context=None, node_id=None, cache=None)

    assert splitter.feature_ == 0
    assert set(splitter.apply(X_fair)) == {1, 2}


@pytest.mark.parametrize("sensible_attribute", [2, -1])
def test_fair_split_never_uses_the_sensitive_attribute(sensible_attribute):
    """Also with a negative index: the sensitive attribute is excluded even if it predicts the target."""
    rng = np.random.default_rng(0)
    X_fair = np.column_stack([rng.normal(size=60), rng.normal(size=60), rng.integers(0, 2, size=60)])
    y = X_fair[:, 2] * 10 + rng.normal(scale=0.1, size=60)

    splitter = FairSplitter(penalty="balance", sensible_attribute=sensible_attribute, penalization_weight=0)
    splitter.fit(X_fair, y, [0, 1, 2], [], context=None, node_id=None, cache=None)

    assert splitter.feature_ != 2


def test_update_statistics():
    splitter = ClassificationSplitter().fit(X, y_clf, NUMERICAL, CATEGORICAL, context=None, node_id=None, cache=None)
    assert splitter.impurity_[1:] == [0, 0]  # pure children

    splitter.update_statistics(X[:6], np.array([0, 1, 0, 0, 1, 1]), context=None, node_id=None, cache=None)

    assert splitter.impurity_[1] > 0


@pytest.mark.parametrize("splitter, y", [(ClassificationSplitter(), y_clf), (RegressionSplitter(), y_reg),
                                         (XGBoostSplitter(lam=1), y_reg)], ids=["clf", "reg", "xgboost"])
def test_state_round_trip(splitter, y):
    splitter.fit(X, y, NUMERICAL, CATEGORICAL, context=None, node_id=None, cache=None)
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
    """The impurity also counts the classes absent from the node, e.g. the ones of class_weight."""
    y = np.array([0, 0, 1])

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        impurity = entropy(class_weights(y, [0, 1, 2]))

    assert impurity == pytest.approx(entropy(class_weights(y, [0, 1])))


@pytest.mark.parametrize("criterion, n_targets", [("squared_error", 1), ("squared_error", 3), ("poisson", 1)])
def test_regression_impurities_as_scikit_learn(criterion, n_targets):
    """The impurities of RegressionSplitter are the ones of the scikit-learn tree, also with several targets."""
    from sklearn.tree import DecisionTreeRegressor

    rng = np.random.default_rng(0)
    X_random = rng.normal(size=(200, 2))
    y = rng.poisson(3, size=(200, n_targets)).astype(float) + X_random[:, :1]
    y = np.abs(y[:, 0]) if n_targets == 1 else y

    splitter = RegressionSplitter(criterion=criterion, random_state=0)
    splitter.fit(X_random, y, [0, 1], [], context=None, node_id=None, cache=None)
    tree = DecisionTreeRegressor(max_depth=1, criterion=criterion, random_state=0).fit(X_random, y)

    assert splitter.impurity_ == pytest.approx(list(tree.tree_.impurity))


def exhaustive_xgboost_split(splitter, X, y, numerical, categorical):
    """The search of XGBoostSplitter before its vectorization: every threshold of every feature, in order."""
    from RuleTree.utils.tree_utils import xgboost_similarity_score_regression as similarity

    root = similarity(y, len(y), splitter.lam)
    best = (-np.inf, -1, -1)
    for feature in range(X.shape[1]):
        for threshold in splitter.get_threshold_values(X, feature, categorical):
            left = X[:, feature] <= threshold if feature in numerical else X[:, feature] == threshold
            if min(left.sum(), (~left).sum()) < splitter.min_cover:
                continue
            gain = similarity(y[left], left.sum(), splitter.lam) + similarity(y[~left], (~left).sum(), splitter.lam) - root
            if gain > best[0]:
                best = (gain, feature, threshold)
    return best[1], best[2]


@pytest.mark.parametrize("case", ["continuous", "ties", "categorical", "min_cover", "quantile", "lam=0", "large",
                                  "small residuals", "few instances"])
def test_xgboost_splitter_as_the_exhaustive_search(case):
    """The vectorized search finds the same split as the exhaustive one, also with ties and rounding."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(400, 6))
    y = X[:, 2] * 3 + rng.normal(size=400)
    numerical, categorical, kwargs = list(range(6)), [], {"lam": 1}
    if case == "ties":
        X = np.round(X)
    elif case == "categorical":
        X = np.column_stack([X[:, :3], rng.integers(0, 4, size=(400, 2))]).astype(float)
        numerical, categorical = [0, 1, 2], [3, 4]
        y = y + 3 * (X[:, 3] == 2)
    elif case == "min_cover":
        kwargs["min_cover"] = 50
    elif case == "quantile":
        kwargs["approximation"] = "quantile"
    elif case == "lam=0":
        kwargs["lam"] = 0
    elif case == "large":
        y = y * 1e9 + 1e12
    elif case == "small residuals":
        y = y * 1e-9
    elif case == "few instances":
        X, y = X[:7], y[:7]

    for target in (y - y.mean(), y):
        splitter = XGBoostSplitter(**kwargs)
        splitter = splitter.fit(X, target, numerical, categorical, context=None, node_id=None, cache=None)
        assert (splitter.feature_, splitter.threshold_) == exhaustive_xgboost_split(splitter, X, target, numerical,
                                                                                  categorical)
