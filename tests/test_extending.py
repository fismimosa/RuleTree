"""New components written with the extension interfaces, as in docs/guides/extending.rst."""
import json

import numpy as np
import pytest
import sklearn

from RuleTree import RuleTreeClassifier, RuleTreeRegressor
from RuleTree.base.RuleTreeBaseGainSplitter import RuleTreeBaseGainSplitter
from RuleTree.base.RuleTreeBaseTransformer import RuleTreeBaseTransformer
from RuleTree.exceptions import NoSplitFoundWarning
from RuleTree.stumps.Stump import Stump
from RuleTree.stumps.splitters import ClassificationSplitter, RegressionSplitter
from RuleTree.stumps.transformers import MatrixTransformer
from RuleTree.tree.RuleTree import RuleTree
from RuleTree.utils.data_utils import json_numpy_encoder
from RuleTree.utils.define import DATA_TYPE_TABULAR, DATA_TYPE_TS, MODEL_TYPE_CLF


class SummaryTransformer(RuleTreeBaseTransformer):
    """Summary statistics of each time series: a transformer without fitted state."""

    supported_data_types = (DATA_TYPE_TS,)
    statistics = {"mean": np.mean, "std": np.std, "min": np.min, "max": np.max}

    def transform(self, X):
        return np.column_stack([f(X[:, 0, :], axis=1) for f in self.statistics.values()])

    def feature_id(self, feature):
        return list(self.statistics)[feature]


class RandomProjectionTransformer(RuleTreeBaseTransformer):
    """Random projections of the tabular data, drawn at each node: a transformer with fitted state."""

    def __init__(self, n_projections=10, random_state=None):
        self.n_projections = n_projections
        self.random_state = random_state

    def fit(self, X, y, idx=None, data_type=None, cache=None, sample_weight=None):
        rng = np.random.default_rng(self.random_state)
        self.directions_ = rng.normal(size=(X.shape[1], self.n_projections))
        self.numerical_, self.categorical_ = list(range(self.n_projections)), []
        return self

    def transform(self, X):
        return X @ self.directions_

    def select(self, feature):
        self.directions_ = self.directions_[:, [feature]]
        return 0

    def feature_id(self, feature):
        return "projection(" + ", ".join(f"{w:.2f}" for w in self.directions_[:, feature]) + ")"

    def get_state(self):
        return {"directions": self.directions_.tolist()}

    def set_state(self, state):
        self.directions_ = np.array(state["directions"])
        return self


class MisclassificationSplitter(RuleTreeBaseGainSplitter):
    """Split minimizing the number of errors of the majority classes of the two sides."""

    task = MODEL_TYPE_CLF

    def __init__(self, min_samples_leaf=1):
        self.min_samples_leaf = min_samples_leaf

    def gain(self, y, left, sample_weight=None):
        def errors(labels):
            return len(labels) - np.unique(labels, return_counts=True)[1].max()

        return errors(y) - errors(y[left]) - errors(y[~left])


class SummaryStumpClassifier(Stump):
    """A named stump: summary statistics of the time series, split as in a decision tree."""

    def __init__(self, criterion="gini", min_samples_leaf=1, random_state=None):
        self.criterion = criterion
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state

    def _make_transformer(self):
        return SummaryTransformer()

    def _make_splitter(self):
        return ClassificationSplitter(criterion=self.criterion, min_samples_leaf=self.min_samples_leaf,
                                      random_state=self.random_state)


@pytest.fixture(scope="module")
def time_series():
    """The class is given by the level of the series."""
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, size=100)
    X = rng.normal(scale=0.5, size=(100, 1, 30)) + 2 * y[:, None, None]
    return X, y


def json_round_trip(model):
    return RuleTree.from_dict(json.loads(json.dumps(model.to_dict(), cls=json_numpy_encoder)))


def test_stateless_transformer(time_series):
    X, y = time_series
    stump = Stump(SummaryTransformer(), ClassificationSplitter())
    assert stump.supported_data_types == (DATA_TYPE_TS,)

    model = RuleTreeClassifier(max_depth=2, base_stumps=stump).fit(X_ts=X, y=y)

    assert np.mean(model.predict(X_ts=X) == y) > 0.95
    assert model.get_rules()["textual_rule"].split(" ")[0] in SummaryTransformer.statistics
    assert np.array_equal(json_round_trip(model).predict(X_ts=X), model.predict(X_ts=X))


def test_stateful_transformer(diabetes):
    X, y = diabetes
    model = RuleTreeRegressor(max_depth=3, random_state=0,
                              base_stumps=Stump(RandomProjectionTransformer(), RegressionSplitter()))

    model.fit(X, y)

    assert model.root.stump.transformer_.directions_.shape == (X.shape[1], 1)  # only the chosen projection
    assert model.get_rules()["textual_rule"].startswith("projection(")
    assert np.allclose(json_round_trip(model).predict(X), model.predict(X))


def test_random_state_reaches_the_components(diabetes):
    X, y = diabetes
    stump = Stump(RandomProjectionTransformer(), RegressionSplitter())

    first = RuleTreeRegressor(max_depth=3, random_state=0, base_stumps=stump).fit(X, y)
    second = RuleTreeRegressor(max_depth=3, random_state=0, base_stumps=stump).fit(X, y)

    assert first.root.stump.transformer.random_state == 0
    assert np.array_equal(first.predict(X), second.predict(X))


def test_gain_splitter(iris):
    X, y = iris
    splitter = MisclassificationSplitter()

    splitter.fit(X, y, numerical=[0, 1, 2, 3], categorical=[])

    assert splitter.feature_ in (2, 3)  # petal length or width separate setosa
    assert set(splitter.apply(X)) == {1, 2}
    model = RuleTreeClassifier(max_depth=3, base_stumps=Stump(MatrixTransformer(), splitter)).fit(X, y)
    assert np.mean(model.predict(X) == y) > 0.9


def test_gain_splitter_categorical_and_min_samples_leaf(iris_mixed):
    X, y = iris_mixed

    splitter = MisclassificationSplitter().fit(X, y, numerical=[], categorical=[0])
    assert splitter.is_categorical_ and splitter.threshold_ in ("red", "green", "blue")

    with pytest.raises(NoSplitFoundWarning):
        MisclassificationSplitter(min_samples_leaf=len(y)).fit(X, y, numerical=[1, 2], categorical=[0])


def test_named_stump(time_series):
    X, y = time_series
    stump = SummaryStumpClassifier(criterion="entropy")

    model = RuleTreeClassifier(max_depth=2, stump_selection="best",
                               base_stumps=[stump, SummaryStumpClassifier()]).fit(X_ts=X, y=y)

    assert sklearn.clone(stump).get_params() == {"criterion": "entropy", "min_samples_leaf": 1,
                                                 "random_state": None}
    assert np.array_equal(json_round_trip(model).predict(X_ts=X), model.predict(X_ts=X))


def test_cache_namespace_of_composed_stumps(iris):
    X, y = iris
    model = RuleTreeClassifier(max_depth=2, stump_selection="best", base_stumps=[
        Stump(MatrixTransformer(), ClassificationSplitter()),
        Stump(RandomProjectionTransformer(), ClassificationSplitter()),
    ]).fit(X, y)

    assert "Stump(MatrixTransformer)" in model.cache_
    assert ("feature_types", DATA_TYPE_TABULAR) in model.get_cache("Stump(MatrixTransformer)")
