"""What the boosting ensembles have in common: the validation set, the copies of the base estimator and the
early stopping."""

import copy

import numpy as np


def validation_split(X: np.ndarray, y: np.ndarray, val_size: float, random_state):
    """
    Training and validation sets of a boosting: a random permutation of the instances, whose first
    ``val_size`` fraction is the validation set.

    Parameters
    ----------
    X : numpy.ndarray of shape (n_samples, n_features)
        Data.
    y : numpy.ndarray
        Target.
    val_size : float
        Fraction of the instances in the validation set.
    random_state : int or None
        Seed of the permutation. The global random state of numpy is not changed.

    Returns
    -------
    X_train, X_val, y_train, y_val : numpy.ndarray
    """
    n_val = int(val_size * X.shape[0])
    idx = np.random.RandomState(random_state).permutation(X.shape[0])
    val_idx, train_idx = idx[:n_val], idx[n_val:]
    return X[train_idx], X[val_idx], y[train_idx], y[val_idx]


def seeded_copy(estimator, rng: np.random.RandomState):
    """
    A copy of the base estimator, for one tree of a boosting. If the estimator has no random_state, the
    copy gets one drawn from ``rng``, the generator of the boosting: so the random_state of the boosting
    makes all its trees reproducible, and each tree gets a different seed, as in the ensembles of
    scikit-learn. A random_state set on the base estimator is kept.

    Parameters
    ----------
    estimator : BaseEstimator
        The base estimator.
    rng : numpy.random.RandomState
        Generator of the boosting, created from its random_state.

    Returns
    -------
    estimator : BaseEstimator
        The copy, not fitted.

    Examples
    --------
    >>> import numpy as np
    >>> from RuleTree import RuleTreeRegressor
    >>> from RuleTree.ensemble._boosting import seeded_copy
    >>> rng = np.random.RandomState(0)
    >>> seeded_copy(RuleTreeRegressor(), rng).random_state, seeded_copy(RuleTreeRegressor(random_state=7), rng).random_state
    (209652396, 7)
    """
    estimator = copy.deepcopy(estimator)
    seed = int(rng.randint(np.iinfo(np.int32).max))  # drawn for every copy, so the seeds do not depend on the estimator
    if "random_state" in estimator.get_params(deep=False) and estimator.get_params(deep=False)["random_state"] is None:
        estimator.set_params(random_state=seed)
    return estimator


class EarlyStopping:
    """
    Early stopping of a boosting: it stops after ``patience`` consecutive iterations whose validation
    error is not lower than the one of the previous iteration.

    Parameters
    ----------
    patience : int
        Number of iterations without improvement before stopping.
    strict : bool, default=False
        Whether an equal error counts as no improvement (``error < previous``) or as an improvement
        (``error <= previous``).
    previous : float, default=numpy.inf
        Validation error before the first iteration.

    Examples
    --------
    >>> from RuleTree.ensemble._boosting import EarlyStopping
    >>> stopping = EarlyStopping(patience=2)
    >>> [stopping.stop(error) for error in [5.0, 4.0, 4.5, 4.2, 3.0, 3.5, 3.6]]
    [False, False, False, False, False, False, True]
    """

    def __init__(self, patience: int, strict: bool = False, previous: float = np.inf):
        """Initialize the early stopping; see the class documentation for the parameters."""
        self.patience = patience
        self.strict = strict
        self.previous = previous
        self.wait = 0

    def stop(self, error: float) -> bool:
        """
        Record the validation error of an iteration.

        Parameters
        ----------
        error : float
            Validation error after the iteration.

        Returns
        -------
        stop : bool
            Whether the last ``patience`` iterations did not improve the error.
        """
        improved = error < self.previous if self.strict else error <= self.previous
        self.wait = 0 if improved else self.wait + 1
        self.previous = error
        return self.wait >= self.patience
