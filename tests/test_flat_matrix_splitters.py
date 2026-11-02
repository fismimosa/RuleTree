import json

import numpy as np
import pytest
import sklearn
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from RuleTree import RuleTreeClassifier, RuleTreeRegressor
from RuleTree.exceptions import NoSplitFoundWarning
from RuleTree.stumps.Stump import Stump
from RuleTree.stumps.splitters import FlatMatrixClassificationSplitter, FlatMatrixRegressionSplitter
from RuleTree.stumps.transformers import MatrixTransformer
from RuleTree.utils.data_utils import get_info_gain, json_numpy_encoder

FIT = dict(context=None, node_id=None, cache=None)

# column 0: noise, column 1: separates the classes at 5, column 2: categorical, "a" for the class 1
X = np.array([[0.3, 1, "b"], [0.9, 2, "b"], [0.1, 3, "b"], [0.7, 4, "b"],
              [0.2, 6, "a"], [0.8, 7, "a"], [0.5, 8, "a"], [0.6, 9, "b"]], dtype=object)
y_clf = np.array([0, 0, 0, 0, 1, 1, 1, 1])
y_reg = np.array([1.0, 1.1, 0.9, 1.0, 5.0, 5.2, 4.8, 5.0])
NUMERICAL, CATEGORICAL = [0, 1], [2]


class WorstFirstSplitter(FlatMatrixClassificationSplitter):
    """Penalizes the children of the splits on the column 1, the best one for the data above."""

    def _penalize_impurity(self, X, y, left, impurity_left, impurity_right, sample_weight=None):
        penalty = (left == (X[:, 1:2].astype(float) <= 5)).all(axis=0)
        return impurity_left + penalty, impurity_right + penalty


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


@pytest.mark.parametrize("criterion", ["gini", "entropy"])
@pytest.mark.parametrize("params", [{}, {"min_samples_leaf": 10}, {"class_weight": "balanced"},
                                    {"class_weight": {0: 5.0, 1: 1.0}}],
                         ids=["default", "min_samples_leaf", "balanced", "dict"])
@pytest.mark.parametrize("weighted", [False, True], ids=["unweighted", "sample_weight"])
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
    """Empty left child and class without instances in a child give impurity 0, not nan or 1."""
    left = np.array([[True, False], [True, False], [False, False], [False, False]])
    y = np.array([0, 0, 1, 1])

    for criterion, node in [("gini", 0.5), ("entropy", 1.0)]:
        impurity, impurity_left, impurity_right, weight_left, weight_right = (
            FlatMatrixClassificationSplitter(criterion=criterion)._split_impurities(y, left))

        assert impurity == pytest.approx(node)
        assert impurity_left == pytest.approx([0, 0])
        assert impurity_right == pytest.approx([0, node])
        assert weight_left == pytest.approx([2, 0])
        assert weight_right == pytest.approx([2, 4])


def test_unknown_criterion():
    with pytest.raises(ValueError, match="criterion"):
        FlatMatrixClassificationSplitter(criterion="unknown").fit(X, y_clf, NUMERICAL, [], **FIT)


def test_penalized_impurity_changes_the_split():
    """The gain uses the penalized impurity; impurity_ is the one of the data."""
    splitter = WorstFirstSplitter().fit(X, y_clf, NUMERICAL, CATEGORICAL, **FIT)

    assert not np.array_equal(splitter.apply(X), np.where(X[:, 1] <= 5, 1, 2))
    assert splitter.gain_ < FlatMatrixClassificationSplitter().fit(X, y_clf, NUMERICAL, CATEGORICAL, **FIT).gain_


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


@pytest.mark.parametrize("params", [{}, {"min_samples_leaf": 10}], ids=["default", "min_samples_leaf"])
@pytest.mark.parametrize("weighted", [False, True], ids=["unweighted", "sample_weight"])
def test_regression_as_scikit_learn(diabetes, params, weighted):
    """Same gain and impurity of the node as the split of a scikit-learn tree."""
    X_diabetes, y = diabetes
    sample_weight = np.random.default_rng(0).uniform(0.5, 2, len(y)) if weighted else None

    flat = FlatMatrixRegressionSplitter(**params)
    flat.fit(X_diabetes, y, list(range(X_diabetes.shape[1])), [], sample_weight, **FIT)
    reference = DecisionTreeRegressor(max_depth=1, random_state=0, **params)
    reference.fit(X_diabetes, y, sample_weight=sample_weight)

    assert flat.gain_ == pytest.approx(get_info_gain(reference))
    assert flat.impurity_[0] == pytest.approx(reference.tree_.impurity[0])


def test_regression_unknown_criterion():
    with pytest.raises(ValueError, match="criterion"):
        FlatMatrixRegressionSplitter(criterion="poisson").fit(X, y_reg, NUMERICAL, [], **FIT)


def test_regression_tree(diabetes):
    X_diabetes, y = diabetes

    model = RuleTreeRegressor(max_depth=3, random_state=0,
                              base_stumps=Stump(MatrixTransformer(), FlatMatrixRegressionSplitter())).fit(X_diabetes, y)
    reference = RuleTreeRegressor(max_depth=3, random_state=0).fit(X_diabetes, y)

    assert model.score(X_diabetes, y) == pytest.approx(reference.score(X_diabetes, y))
