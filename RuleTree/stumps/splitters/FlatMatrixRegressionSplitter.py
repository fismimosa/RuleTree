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

    def _split_impurities(self, y, left, sample_weight=None) -> tuple:
        """
        Impurity of the node and of the children of all the candidate splits, from the weighted sums of
        the target and of its square in the left children.

        Parameters
        ----------
        y : numpy.ndarray of shape (n_samples,)
            Target of the instances in the node.
        left : numpy.ndarray of bool of shape (n_samples, n_candidates)
            For each candidate split, the instances that go to the left child.
        sample_weight : numpy.ndarray of shape (n_samples,), default=None
            Sample weights of the instances in the node.

        Returns
        -------
        impurity : float
            Impurity of the node.
        impurity_left, impurity_right : numpy.ndarray of shape (n_candidates,)
            Impurity of the children of each candidate.
        weight_left, weight_right : numpy.ndarray of shape (n_candidates,)
            Total weight of the children of each candidate.

        Raises
        ------
        ValueError
            If the criterion is not 'squared_error'.
        """
        if self.criterion != "squared_error":
            raise ValueError(f"Unknown criterion {self.criterion!r}: use 'squared_error'")

        weights = np.ones(len(y)) if sample_weight is None else np.asarray(sample_weight, dtype=float)
        y = np.asarray(y, dtype=float)
        y = y - weights @ y / max(weights.sum(), np.finfo(float).tiny)  # centered: smaller rounding errors

        weight, weight_left = weights.sum(), weights @ left
        y_sum, y_sum_left = weights @ y, (weights * y) @ left
        y_squared, y_squared_left = weights @ y ** 2, (weights * y ** 2) @ left

        weight_right = np.maximum(weight - weight_left, 0)

        return (_squared_error(weight, y_sum, y_squared),
                _squared_error(weight_left, y_sum_left, y_squared_left),
                _squared_error(weight_right, y_sum - y_sum_left, y_squared - y_squared_left),
                weight_left, weight_right)


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
