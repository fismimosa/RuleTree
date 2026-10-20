"""What the boosting ensembles have in common: the validation set and the early stopping."""

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
        Seed of the permutation (set with ``numpy.random.seed``).

    Returns
    -------
    X_train, X_val, y_train, y_val : numpy.ndarray
    """
    np.random.seed(random_state)
    n_val = int(val_size * X.shape[0])
    idx = np.random.permutation(X.shape[0])
    val_idx, train_idx = idx[:n_val], idx[n_val:]
    return X[train_idx], X[val_idx], y[train_idx], y[val_idx]


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
