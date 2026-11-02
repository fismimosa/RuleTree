"""FlatMatrixClassificationSplitter: the best split for classification, with matrix operations."""

import numpy as np

from RuleTree.stumps.splitters.FlatMatrixSplitter import FlatMatrixSplitter
from RuleTree.utils.define import MODEL_TYPE_CLF


class FlatMatrixClassificationSplitter(FlatMatrixSplitter):
    """
    Splitter for classification that evaluates all the candidate splits together with matrix
    operations (see FlatMatrixSplitter) and maximizes the decrease of impurity.

    Parameters
    ----------
    criterion : {'gini', 'entropy'}, default='gini'
        Impurity measure.
    min_samples_leaf : int or float, default=1
        Minimum number of instances in each child (a fraction of the instances of the node if float).
    class_weight : dict, 'balanced' or None, default=None
        Weights of the classes in the impurity; 'balanced' weighs them inversely to their frequency.

    Attributes
    ----------
    feature_ : int
        Column of the feature matrix used by the split.
    threshold_ : float or object
        Threshold of a numerical split, or value of a categorical split.
    is_categorical_ : bool
        Whether the split is categorical.
    gain_ : float
        Information gain of the split.
    impurity_ : list of float
        Impurity of the node, of the left child and of the right child.

    See Also
    --------
    ClassificationSplitter : the splitter to use when the impurity is not changed.

    Examples
    --------
    >>> from sklearn.datasets import load_iris
    >>> from RuleTree.stumps.splitters import FlatMatrixClassificationSplitter
    >>> X, y = load_iris(return_X_y=True)
    >>> splitter = FlatMatrixClassificationSplitter()
    >>> splitter = splitter.fit(X, y, numerical=[0, 1, 2, 3], categorical=[], context=None, node_id=None, cache=None)
    >>> splitter.feature_, round(splitter.threshold_, 2), round(splitter.gain_, 3)
    (2, 2.45, 0.333)
    >>> splitter.apply(X[[0, 100]])
    array([1, 2])
    """

    task = MODEL_TYPE_CLF

    def __init__(self, criterion='gini', min_samples_leaf=1, class_weight=None):
        """Initialize the FlatMatrixClassificationSplitter; see the class documentation for the parameters."""
        super().__init__(criterion, min_samples_leaf)
        self.class_weight = class_weight

    def _impurity_fun(self):
        """
        Impurity function of the criterion.

        Returns
        -------
        impurity_fun : callable
            Function from the weights of the classes, of shape (n_classes, ...), to the impurities.

        Raises
        ------
        ValueError
            If the criterion is not 'gini' or 'entropy'.
        """
        if self.criterion == "gini":
            return _gini
        elif self.criterion == "entropy":
            return _entropy
        raise ValueError(f"Unknown criterion {self.criterion!r}: use 'gini' or 'entropy'")

    def _class_weights(self, classes, counts) -> np.ndarray:
        """
        Weight of each class.

        Parameters
        ----------
        classes : numpy.ndarray of shape (n_classes,)
            Classes in the node.
        counts : numpy.ndarray of shape (n_classes,)
            Number of instances of each class in the node.

        Returns
        -------
        class_weights : numpy.ndarray of shape (n_classes,)
        """
        if self.class_weight is None:
            return np.ones(len(classes))
        if self.class_weight == "balanced":
            return counts.sum() / (len(classes) * counts)
        return np.array([self.class_weight.get(class_label, 1.0) for class_label in classes])

    def _split_impurities(self, y, left, sample_weight=None) -> tuple:
        """
        Impurity of the node and of the children of all the candidate splits, from the weights of the
        classes in the left children.

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
        """
        impurity_fun = self._impurity_fun()
        classes, y_idx, counts = np.unique(y, return_inverse=True, return_counts=True)

        y_onehot = np.zeros((len(y), len(classes)))
        y_onehot[np.arange(len(y)), y_idx] = self._class_weights(classes, counts)[y_idx]
        if sample_weight is not None:
            y_onehot[np.arange(len(y)), y_idx] *= sample_weight

        weights = y_onehot.sum(axis=0)
        weights_left = y_onehot.T @ left
        weights_right = np.maximum(weights[:, None] - weights_left, 0)  # no negative rounding errors

        return (impurity_fun(weights), impurity_fun(weights_left), impurity_fun(weights_right),
                weights_left.sum(axis=0), weights_right.sum(axis=0))


def _probabilities(weights: np.ndarray) -> np.ndarray:
    """
    Probabilities of the classes, all 0 without instances.

    Parameters
    ----------
    weights : numpy.ndarray of shape (n_classes, ...)
        Weight of each class.

    Returns
    -------
    probabilities : numpy.ndarray of shape (n_classes, ...)
    """
    totals = weights.sum(axis=0)
    return weights / np.where(totals > 0, totals, 1)


def _gini(weights: np.ndarray) -> np.ndarray:
    """
    Gini impurity from the weights of the classes, 0 without instances.

    Parameters
    ----------
    weights : numpy.ndarray of shape (n_classes, ...)
        Weight of each class.

    Returns
    -------
    impurity : numpy.ndarray of shape (...)
    """
    return np.where(weights.sum(axis=0) > 0, 1 - np.sum(_probabilities(weights) ** 2, axis=0), 0.0)


def _entropy(weights: np.ndarray) -> np.ndarray:
    """
    Entropy from the weights of the classes, 0 without instances.

    Parameters
    ----------
    weights : numpy.ndarray of shape (n_classes, ...)
        Weight of each class.

    Returns
    -------
    impurity : numpy.ndarray of shape (...)
    """
    probabilities = _probabilities(weights)
    log_probabilities = np.log2(probabilities, out=np.zeros_like(probabilities), where=probabilities > 0)
    return -np.sum(probabilities * log_probabilities, axis=0)
