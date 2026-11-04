import json

import numpy as np
import pytest
import sklearn
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from RuleTree import RuleTreeClassifier, RuleTreeRegressor
from RuleTree.exceptions import NoSplitFoundWarning
from RuleTree.stumps.Stump import Stump, _params_from_dict, _params_to_dict
from RuleTree.stumps.splitters import (FlatMatrixClassificationSplitter, FlatMatrixRegressionSplitter, candidate_mask,
                                       information_gain)
from RuleTree.stumps.transformers import MatrixTransformer
from RuleTree.utils.data_utils import get_info_gain, json_numpy_encoder

FIT = dict(context=None, node_id=None, cache=None)

# column 0: noise, column 1: separates the classes at 5, column 2: categorical, "a" for the class 1
X = np.array([[0.3, 1, "b"], [0.9, 2, "b"], [0.1, 3, "b"], [0.7, 4, "b"],
              [0.2, 6, "a"], [0.8, 7, "a"], [0.5, 8, "a"], [0.6, 9, "b"]], dtype=object)
y_clf = np.array([0, 0, 0, 0, 1, 1, 1, 1])
y_reg = np.array([1.0, 1.1, 0.9, 1.0, 5.0, 5.2, 4.8, 5.0])
NUMERICAL, CATEGORICAL = [0, 1], [2]


def not_on_column_1(impurity, impurity_left, impurity_right, weight_left, weight_right, X, y, features, thresholds,
                    is_categorical, sample_weight=None):
    """Discards the candidates on the column 1, the best one for the data above."""
    gains = information_gain(impurity, impurity_left, impurity_right, weight_left, weight_right)
    return np.where(features == 1, -np.inf, gains)


def discard_all(impurity, impurity_left, *args):
    return np.full(len(impurity_left), np.nan)


def wrong_shape(*args):
    return np.zeros(1)


def balanced_gain(impurity, impurity_left, impurity_right, weight_left, weight_right, X, y, features, thresholds,
                  is_categorical, sample_weight=None):
    """Penalizes the children of different size, from the mask of the candidates."""
    left_fraction = candidate_mask(X, features, thresholds, is_categorical).mean(axis=0)
    return information_gain(impurity, impurity_left, impurity_right, weight_left, weight_right) - abs(left_fraction - 0.5)


@pytest.mark.parametrize("splitter, y", [(FlatMatrixClassificationSplitter(), y_clf),
                                         (FlatMatrixRegressionSplitter(), y_reg)], ids=["clf", "reg"])
def test_numerical_split(splitter, y):
    splitter.fit(X, y, NUMERICAL, CATEGORICAL, **FIT)

    assert splitter.feature_ == 1
    assert splitter.threshold_ == 5
    assert not splitter.is_categorical_
    assert splitter.gain_ > 0
    assert np.array_equal(splitter.apply(X), np.where(X[:, 1] <= 5, 1, 2))


@pytest.mark.parametrize("splitter, y", [(FlatMatrixClassificationSplitter(), y_clf),
                                         (FlatMatrixRegressionSplitter(), y_reg)], ids=["clf", "reg"])
def test_categorical_split(splitter, y):
    """The two values give the same partition, so either one can be chosen."""
    splitter.fit(X, y, [], CATEGORICAL, **FIT)

    assert splitter.feature_ == 2
    assert splitter.is_categorical_
    assert splitter.threshold_ in ("a", "b")
    assert np.array_equal(splitter.apply(X), np.where(X[:, 2] == splitter.threshold_, 1, 2))


@pytest.mark.parametrize("splitter", [FlatMatrixClassificationSplitter(), FlatMatrixRegressionSplitter()], ids=["clf", "reg"])
def test_no_split_on_a_pure_node(splitter):
    with pytest.raises(NoSplitFoundWarning):
        splitter.fit(X[:, :2].astype(float), np.ones(len(X)), [0, 1], [], **FIT)


@pytest.mark.parametrize("splitter, y", [(FlatMatrixClassificationSplitter(), y_clf),
                                         (FlatMatrixRegressionSplitter(), y_reg)], ids=["clf", "reg"])
def test_no_degenerate_split(splitter, y):
    """A constant feature, numerical or categorical, would leave a child empty."""
    with pytest.raises(NoSplitFoundWarning):
        splitter.fit(np.full((len(y), 1), 1.0), y, [0], [], **FIT)
    with pytest.raises(NoSplitFoundWarning):
        splitter.fit(np.full((len(y), 1), "a", dtype=object), y, [], [0], **FIT)


@pytest.mark.parametrize("splitter, y", [(FlatMatrixClassificationSplitter, y_clf),
                                         (FlatMatrixRegressionSplitter, y_reg)], ids=["clf", "reg"])
@pytest.mark.parametrize("min_samples_leaf", [3, 3 / 8])
def test_min_samples_leaf(splitter, y, min_samples_leaf):
    y = y.copy()
    y[-1] = y[0]  # the best split without the constraint leaves 1 instance on a side
    X_sorted = np.arange(len(y), dtype=float)[:, None]

    fitted = splitter(min_samples_leaf=min_samples_leaf).fit(X_sorted, y, [0], [], **FIT)

    assert min(np.bincount(fitted.apply(X_sorted))[1:]) >= 3
    with pytest.raises(NoSplitFoundWarning):
        splitter(min_samples_leaf=len(y)).fit(X_sorted, y, [0], [], **FIT)


@pytest.mark.parametrize("splitter, y", [(FlatMatrixClassificationSplitter(), y_clf),
                                         (FlatMatrixRegressionSplitter(), y_reg)], ids=["clf", "reg"])
def test_state_round_trip(splitter, y):
    splitter.fit(X, y, NUMERICAL, CATEGORICAL, **FIT)
    state = json.loads(json.dumps(splitter.get_state(), cls=json_numpy_encoder))

    loaded = sklearn.clone(splitter).set_state(state)

    assert np.array_equal(loaded.apply(X), splitter.apply(X))
    assert loaded.gain_ == pytest.approx(splitter.gain_)


@pytest.mark.parametrize("splitter, y", [(FlatMatrixClassificationSplitter(), y_clf),
                                         (FlatMatrixRegressionSplitter(), y_reg)], ids=["clf", "reg"])
def test_several_targets(splitter, y):
    with pytest.raises(ValueError):
        splitter.fit(X, np.column_stack([y, y]), NUMERICAL, CATEGORICAL, **FIT)


@pytest.mark.parametrize("criterion, params, weighted", [
    ("gini", {}, False), ("entropy", {}, False), ("log_loss", {}, False), ("gini", {}, True),
    ("gini", {"min_samples_leaf": 10}, False), ("entropy", {"class_weight": "balanced"}, True),
    ("gini", {"class_weight": {0: 5.0, 1: 1.0}}, True),
], ids=["gini", "entropy", "log_loss", "sample_weight", "min_samples_leaf", "balanced", "dict"])
def test_classification_as_scikit_learn(iris, criterion, params, weighted):
    """Same gain and impurity of the node as the split of a scikit-learn tree."""
    X_iris, y = iris
    sample_weight = np.random.default_rng(0).uniform(0.5, 2, len(y)) if weighted else None
    numerical = list(range(X_iris.shape[1]))

    flat = FlatMatrixClassificationSplitter(criterion=criterion, **params)
    flat.fit(X_iris, y, numerical, [], sample_weight, **FIT)
    reference = DecisionTreeClassifier(max_depth=1, criterion=criterion, random_state=0, **params)
    reference.fit(X_iris, y, sample_weight=sample_weight)

    assert flat.gain_ == pytest.approx(get_info_gain(reference))
    assert flat.impurity_[0] == pytest.approx(reference.tree_.impurity[0])


def test_classification_impurities_as_definition():
    """A child without instances or with a pure class has impurity 0, not nan or 1."""
    for criterion, node in [("gini", 0.5), ("entropy", 1.0)]:
        splitter = FlatMatrixClassificationSplitter(criterion=criterion)
        statistics = np.array([[2.0, 0.0, 2.0], [2.0, 0.0, 0.0]])

        assert splitter._impurity(statistics) == pytest.approx([node, 0, 0])
        assert splitter._weight(statistics) == pytest.approx([4, 0, 2])


def test_misclassification():
    """Maximizes the decrease of the errors of the majority classes, on an exhaustive search."""
    rng = np.random.default_rng(0)
    x = rng.normal(size=100)
    y = rng.integers(0, 3, 100)
    X_random = x[:, None]

    fitted = FlatMatrixClassificationSplitter(criterion="misclassification").fit(X_random, y, [0], [], **FIT)

    def errors(labels):
        return len(labels) - np.bincount(labels).max()

    values = np.unique(x)
    expected = max(errors(y) - errors(y[x <= t]) - errors(y[x > t]) for t in (values[:-1] + values[1:]) / 2)
    assert fitted.gain_ == pytest.approx(expected / len(y))
    assert fitted.impurity_[0] == pytest.approx(errors(y) / len(y))


def test_unknown_criterion():
    with pytest.raises(ValueError, match="criterion"):
        FlatMatrixClassificationSplitter(criterion="unknown").fit(X, y_clf, NUMERICAL, [], **FIT)


@pytest.mark.parametrize("splitter", [FlatMatrixClassificationSplitter, FlatMatrixRegressionSplitter],
                         ids=["clf", "reg"])
def test_candidates_as_the_definition(splitter):
    """The statistics of the left children are the sums over the instances that a mask sends left."""
    rng = np.random.default_rng(0)
    X_mixed = np.empty((200, 3), dtype=object)
    X_mixed[:, 0], X_mixed[:, 1] = rng.integers(0, 5, 200).astype(float), rng.normal(size=200)
    X_mixed[:, 2] = rng.choice(list("abc"), 200)
    y = rng.integers(0, 3, 200)
    sample_weight = rng.uniform(0.5, 2, 200)

    fitted = splitter()
    statistics = fitted._statistics(y, sample_weight)
    features, thresholds, is_categorical, statistics_left, n_left = fitted._candidate_splits(X_mixed, statistics,
                                                                                             [0, 1], [2])

    assert list(is_categorical) == sorted(is_categorical)
    for i, (feature, threshold) in enumerate(zip(features, thresholds)):
        left = X_mixed[:, feature] == threshold if is_categorical[i] else X_mixed[:, feature] <= threshold
        assert statistics_left[i] == pytest.approx(statistics[left].sum(axis=0))
        assert n_left[i] == left.sum()


@pytest.mark.parametrize("splitter, y", [(FlatMatrixClassificationSplitter(), y_clf),
                                         (FlatMatrixRegressionSplitter(), y_reg)], ids=["clf", "reg"])
def test_missing_values_go_right(splitter, y):
    X_missing = np.array([1, 2, 3, 4, 6, 7, 8, np.nan])[:, None]

    splitter.fit(X_missing, y, [0], [], **FIT)

    assert splitter.threshold_ == 5
    assert np.array_equal(splitter.apply(X_missing), np.where(X_missing[:, 0] <= 5, 1, 2))


@pytest.mark.parametrize("splitter", [FlatMatrixClassificationSplitter(), FlatMatrixRegressionSplitter()],
                         ids=["clf", "reg"])
def test_threshold_between_adjacent_values(splitter):
    """The midpoint of two adjacent floats is rounded to one of them: the upper one must stay on the right."""
    lower, upper = 1.0, np.nextafter(1.0, 2.0)
    X_adjacent = np.array([[lower], [lower], [upper], [upper]])
    y = np.array([0.0, 0.0, 1.0, 1.0])

    splitter.fit(X_adjacent, y, [0], [], **FIT)

    assert np.array_equal(splitter.apply(X_adjacent), [1, 1, 2, 2])


@pytest.mark.parametrize("splitter, y", [(FlatMatrixClassificationSplitter, y_clf),
                                         (FlatMatrixRegressionSplitter, y_reg)], ids=["clf", "reg"])
def test_gain_function_changes_the_split(splitter, y):
    """gain_ is the value of the function; impurity_ is the one of the split."""
    fitted = splitter(gain_function=not_on_column_1).fit(X, y, NUMERICAL, CATEGORICAL, **FIT)
    best = splitter().fit(X, y, NUMERICAL, CATEGORICAL, **FIT)

    assert best.feature_ == 1 and fitted.feature_ != 1
    assert fitted.gain_ < best.gain_


def test_gain_function_with_the_mask(iris):
    X_iris, y = iris

    fitted = FlatMatrixClassificationSplitter(gain_function=balanced_gain).fit(X_iris, y, [0, 1, 2, 3], [], **FIT)
    left = fitted.apply(X_iris) == 1

    assert abs(left.mean() - 0.5) < abs((X_iris[:, 2] <= 2.45).mean() - 0.5)


@pytest.mark.parametrize("splitter, y", [(FlatMatrixClassificationSplitter, y_clf),
                                         (FlatMatrixRegressionSplitter, y_reg)], ids=["clf", "reg"])
def test_gain_function_errors(splitter, y):
    with pytest.raises(NoSplitFoundWarning):
        splitter(gain_function=discard_all).fit(X, y, NUMERICAL, CATEGORICAL, **FIT)
    with pytest.raises(ValueError, match="shape"):
        splitter(gain_function=wrong_shape).fit(X, y, NUMERICAL, CATEGORICAL, **FIT)


def test_candidate_mask():
    features = np.array([1, 2, 2])
    thresholds = np.array([5.0, "a", "b"], dtype=object)

    left = candidate_mask(X, features, thresholds, np.array([False, True, True]))

    assert np.array_equal(left[:, 0], X[:, 1] <= 5)
    assert np.array_equal(left[:, 1], X[:, 2] == "a")
    assert np.array_equal(left[:, 2], X[:, 2] == "b")


def test_save_a_tree_with_a_gain_function(tmp_path, iris):
    """The function is saved by name, in the JSON file, and the tree predicts the same after loading it."""
    X_iris, y = iris
    model = RuleTreeClassifier(max_depth=2, random_state=0,
                               base_stumps=Stump(MatrixTransformer(),
                                                 FlatMatrixClassificationSplitter(gain_function=balanced_gain)))
    model.fit(X_iris, y)

    model.to_dict(tmp_path / "tree.json")
    loaded = RuleTreeClassifier.from_dict(tmp_path / "tree.json")

    assert np.array_equal(loaded.predict(X_iris), model.predict(X_iris))
    assert _params_from_dict(_params_to_dict({"gain_function": balanced_gain}))["gain_function"] is balanced_gain


def test_lambda_cannot_be_saved():
    with pytest.raises(ValueError, match="cannot be saved"):
        _params_to_dict({"gain_function": lambda *args: 0})


def test_update_statistics():
    splitter = FlatMatrixClassificationSplitter().fit(X, y_clf, NUMERICAL, CATEGORICAL, **FIT)
    assert splitter.impurity_[1:] == [0, 0]  # pure children

    splitter.update_statistics(X[:6], np.array([0, 1, 0, 0, 1, 1]), **FIT)

    assert splitter.impurity_[1] > 0


def test_classification_tree(iris):
    X_iris, y = iris

    model = RuleTreeClassifier(max_depth=3, random_state=0,
                               base_stumps=Stump(MatrixTransformer(), FlatMatrixClassificationSplitter())).fit(X_iris, y)
    reference = RuleTreeClassifier(max_depth=3, random_state=0).fit(X_iris, y)

    assert model.score(X_iris, y) == pytest.approx(reference.score(X_iris, y))


@pytest.mark.parametrize("criterion", ["squared_error", "poisson"])
@pytest.mark.parametrize("params", [{}, {"min_samples_leaf": 10}], ids=["default", "min_samples_leaf"])
@pytest.mark.parametrize("weighted", [False, True], ids=["unweighted", "sample_weight"])
def test_regression_as_scikit_learn(diabetes, criterion, params, weighted):
    """Same gain and impurity of the node as the split of a scikit-learn tree."""
    X_diabetes, y = diabetes
    sample_weight = np.random.default_rng(0).uniform(0.5, 2, len(y)) if weighted else None

    flat = FlatMatrixRegressionSplitter(criterion=criterion, **params)
    flat.fit(X_diabetes, y, list(range(X_diabetes.shape[1])), [], sample_weight, **FIT)
    reference = DecisionTreeRegressor(max_depth=1, criterion=criterion, random_state=0, **params)
    reference.fit(X_diabetes, y, sample_weight=sample_weight)

    assert flat.gain_ == pytest.approx(get_info_gain(reference))
    assert flat.impurity_[0] == pytest.approx(reference.tree_.impurity[0])


def test_regression_unknown_criterion():
    with pytest.raises(ValueError, match="criterion"):
        FlatMatrixRegressionSplitter(criterion="unknown").fit(X, y_reg, NUMERICAL, [], **FIT)


def test_poisson_children():
    """A child whose target sums to 0 is not allowed, and a negative target is an error."""
    X_counts = np.array([[1.0], [2.0], [3.0], [4.0]])
    splitter = FlatMatrixRegressionSplitter(criterion="poisson")

    splitter.fit(X_counts, np.array([0.0, 0.0, 1.0, 5.0]), [0], [], **FIT)
    assert splitter.threshold_ == 3.5

    with pytest.raises(NoSplitFoundWarning):
        splitter.fit(X_counts, np.array([0.0, 0.0, 0.0, 5.0]), [0], [], **FIT)
    with pytest.raises(ValueError, match="negative"):
        splitter.fit(X_counts, np.array([-1.0, 0.0, 0.0, 5.0]), [0], [], **FIT)


def test_regression_tree(diabetes):
    X_diabetes, y = diabetes

    model = RuleTreeRegressor(max_depth=3, random_state=0,
                              base_stumps=Stump(MatrixTransformer(), FlatMatrixRegressionSplitter())).fit(X_diabetes, y)
    reference = RuleTreeRegressor(max_depth=3, random_state=0).fit(X_diabetes, y)

    assert model.score(X_diabetes, y) == pytest.approx(reference.score(X_diabetes, y))
