"""FlatMatrixRegressionSplitter: the best split for regression, with matrix operations."""

import numpy as np

from RuleTree.stumps.splitters.FlatMatrixSplitter import FlatMatrixSplitter
from RuleTree.utils.define import MODEL_TYPE_REG


class FlatMatrixRegressionSplitter(FlatMatrixSplitter):
    """
    Splitter for regression that evaluates all the candidate splits together with matrix operations
    (see FlatMatrixSplitter) and maximizes the decrease of impurity.

    Parameters
    ----------
    criterion : {'squared_error'}, default='squared_error'
        Impurity measure.
    min_samples_leaf : int or float, default=1
        Minimum number of instances in each child (a fraction of the instances of the node if float).

    Attributes
    ----------
    feature_ : int
        Column of the feature matrix used by the split.
    threshold_ : float or object
        Threshold of a numerical split, or value of a categorical split.
    is_categorical_ : bool
        Whether the split is categorical.
    gain_ : float
        Decrease of impurity of the split.
    impurity_ : list of float
        Impurity of the node, of the left child and of the right child.

    See Also
    --------
    RegressionSplitter : the splitter to use when the impurity is not changed.

    Examples
    --------
    >>> from sklearn.datasets import load_diabetes
    >>> from RuleTree.stumps.splitters import FlatMatrixRegressionSplitter
    >>> X, y = load_diabetes(return_X_y=True)
    >>> splitter = FlatMatrixRegressionSplitter()
    >>> splitter = splitter.fit(X, y, numerical=list(range(10)), categorical=[], context=None, node_id=None, cache=None)
    >>> splitter.feature_, round(splitter.threshold_, 4)
    (8, -0.0038)
    """

    task = MODEL_TYPE_REG

    def __init__(self, criterion='squared_error', min_samples_leaf=1):
        """Initialize the FlatMatrixRegressionSplitter; see the class documentation for the parameters."""
        super().__init__(criterion, min_samples_leaf)

    def _statistics(self, y, sample_weight=None) -> np.ndarray:
        """
        Weight of each instance, and its product with the target and with the square of the target. The
        target is centered, for smaller rounding errors.

        Parameters
        ----------
        y : numpy.ndarray of shape (n_samples,)
            Target of the instances in the node.
        sample_weight : numpy.ndarray of shape (n_samples,), default=None
            Sample weights of the instances in the node.

        Returns
        -------
        statistics : numpy.ndarray of shape (n_samples, 3)
        """
        weights = np.ones(len(y)) if sample_weight is None else np.asarray(sample_weight, dtype=float)
        y = np.asarray(y, dtype=float)
        y = y - weights @ y / max(weights.sum(), np.finfo(float).tiny)

        return np.column_stack([weights, weights * y, weights * y ** 2])

    def _impurity(self, statistics) -> np.ndarray:
        """
        Impurity from the weight and the sums of the target and of its square.

        Parameters
        ----------
        statistics : numpy.ndarray of shape (3, ...)
            Total weight, weighted sum of the target and weighted sum of its square.

        Returns
        -------
        impurity : numpy.ndarray of shape (...)

        Raises
        ------
        ValueError
            If the criterion is not 'squared_error'.
        """
        if self.criterion != "squared_error":
            raise ValueError(f"Unknown criterion {self.criterion!r}: use 'squared_error'")

        return _squared_error(self._weight(statistics), statistics[1], statistics[2])

    def _weight(self, statistics) -> np.ndarray:
        """
        Total weight from the statistics.

        Parameters
        ----------
        statistics : numpy.ndarray of shape (3, ...)
            Total weight, weighted sum of the target and weighted sum of its square.

        Returns
        -------
        weight : numpy.ndarray of shape (...)
        """
        return np.maximum(statistics[0], 0)  # no negative rounding errors


def _squared_error(weight, y_sum, y_squared) -> np.ndarray:
    """
    Variance of the target from its sums, 0 without instances.

    Parameters
    ----------
    weight : float or numpy.ndarray
        Total weight of the instances.
    y_sum : float or numpy.ndarray
        Weighted sum of the target.
    y_squared : float or numpy.ndarray
        Weighted sum of the square of the target.

    Returns
    -------
    impurity : float or numpy.ndarray
    """
    safe_weight = np.where(weight > 0, weight, 1)
    return np.maximum(y_squared / safe_weight - (y_sum / safe_weight) ** 2, 0)
