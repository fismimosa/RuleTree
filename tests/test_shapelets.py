"""Candidate shapelets and their selection (RuleTree.utils.shapelet_transform.Shapelets)."""
import numpy as np
import pytest

pytest.importorskip("numba")

from RuleTree.utils.define import MODEL_TYPE_REG  # noqa: E402
from RuleTree.utils.shapelet_transform.Shapelets import Shapelets  # noqa: E402


@pytest.fixture(scope="module")
def series():
    """Each time series is constant, equal to its index: a candidate tells which series it comes from."""
    n_series, length = 60, 20
    X = np.repeat(np.arange(n_series, dtype=float)[:, None, None], length, axis=2)
    y = np.repeat([0, 1, 2], [30, 20, 10])
    return X, y


def candidates(shapelets: Shapelets, X, y):
    shapelets.fit(X[:, :, :], y)  # sets the seeds as fit does
    return shapelets._Shapelets__fit_partition(X, y)


def origin(candidate_shapelets):
    """Index of the time series of each candidate."""
    return candidate_shapelets[:, 0, 0].astype(int)


def test_number_of_candidates_per_class(series):
    X, y = series

    found = candidates(Shapelets(n_shapelets=3, n_shapelets_for_selection=30, sliding_window=5), X, y)

    assert len(found) == 30  # 10 for each of the 3 classes (before, 3 for each class)
    assert np.array_equal(np.bincount(y[origin(found)]), [10, 10, 10])


def test_stratified_candidates(series):
    X, y = series

    found = candidates(Shapelets(n_shapelets=3, n_shapelets_for_selection="stratified", sliding_window=5), X, y)

    # the square root of 60 series (7), in proportion to the classes (30, 20 and 10 series)
    assert np.array_equal(np.bincount(y[origin(found)]), [4, 2, 1])


def test_time_series_per_class(series):
    X, y = series

    found = candidates(Shapelets(n_shapelets=3, n_shapelets_for_selection=300, n_ts_for_selection_per_class=2,
                                 sliding_window=5), X, y)

    for label in range(3):
        assert len(set(origin(found)[y[origin(found)] == label])) <= 2


def test_regression_draws_from_all_the_series(series):
    X, _ = series
    y = np.linspace(0, 1, len(X))  # every value once: not classes

    found = candidates(Shapelets(n_shapelets=3, n_shapelets_for_selection=40, sliding_window=5,
                                 ml_task=MODEL_TYPE_REG), X, y)

    assert len(found) == 40
    several_targets = np.column_stack([y, -y])
    assert len(candidates(Shapelets(n_shapelets=3, n_shapelets_for_selection=40, sliding_window=5,
                                    ml_task=MODEL_TYPE_REG), X, several_targets)) == 40


def test_all_the_subsequences(series):
    X, y = series

    found = candidates(Shapelets(n_shapelets=3, n_shapelets_for_selection=np.inf, sliding_window=5), X, y)

    assert len(found) == len(X) * (20 - 5 + 1)  # also the last position


@pytest.mark.parametrize("selection, task", [("mi_clf", None), ("mi_reg", MODEL_TYPE_REG)])
def test_mutual_information_selects_informative_shapelets(selection, task):
    """Each distance is paired with the target of its own time series."""
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, size=120)
    X = rng.normal(scale=0.3, size=(120, 1, 40))
    X[y == 1, 0, 15:25] += 2  # a bump only in the class 1
    target = y if task is None else y + rng.normal(scale=0.1, size=len(y))

    kwargs = {} if task is None else {"ml_task": task}
    shapelets = Shapelets(n_shapelets=2, n_shapelets_for_selection=200, sliding_window=10, selection=selection,
                          mi_n_neighbors=10, random_state=0, **kwargs).fit(X, target)

    assert np.all(shapelets.shapelets.max(axis=(1, 2)) > 1)  # both contain (part of) the bump
