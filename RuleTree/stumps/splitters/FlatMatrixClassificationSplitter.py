"""FlatMatrixClassificationSplitter: the best split for classification, with matrix operations."""

import numpy as np

from RuleTree.stumps.splitters.FlatMatrixSplitter import FlatMatrixSplitter
from RuleTree.utils.define import MODEL_TYPE_CLF
from RuleTree.utils.impurity import CLASSIFICATION_IMPURITIES


class FlatMatrixClassificationSplitter(FlatMatrixSplitter):
    """
    Splitter for classification that evaluates all the candidate splits together with matrix
    operations (see FlatMatrixSplitter) and maximizes the decrease of impurity.

    Parameters
    ----------
    criterion : {'gini', 'entropy', 'log_loss', 'misclassification'}, default='gini'
        Impurity measure; 'log_loss' is the same as 'entropy', 'misclassification' is the fraction of the
        instances outside the majority class.
    min_samples_leaf : int or float, default=1
        Minimum number of instances in each child (a fraction of the instances of the node if float).
    class_weight : dict, 'balanced' or None, default=None
        Weights of the classes in the impurity; 'balanced' weighs them inversely to their frequency.
    gain_function : callable, default=None
        Gain of all the candidates of a node, instead of the information gain (see FlatMatrixSplitter).

    Attributes
    ----------
    feature_ : int
        Column of the feature matrix used by the split.
    threshold_ : float or object
        Threshold of a numerical split, or value of a categorical split.
    is_categorical_ : bool
        Whether the split is categorical.
    gain_ : float
        Gain of the split: the information gain, or the value of ``gain_function``.
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

    def __init__(self, criterion='gini', min_samples_leaf=1, class_weight=None, gain_function=None):
        """Initialize the FlatMatrixClassificationSplitter; see the class documentation for the parameters."""
        super().__init__(criterion, min_samples_leaf, gain_function)
        self.class_weight = class_weight

    def _impurity_fun(self):
        """Impurity function of the criterion, from the weights of the classes (see RuleTree.utils.impurity)."""
        if self.criterion not in CLASSIFICATION_IMPURITIES:
            raise ValueError(f"Unknown criterion {self.criterion!r}: use 'gini', 'entropy', 'log_loss' or "
                             "'misclassification'")
        return CLASSIFICATION_IMPURITIES[self.criterion]

    def _class_weights(self, classes, counts) -> np.ndarray:
        """Weight of each class."""
        if self.class_weight is None:
            return np.ones(len(classes))
        if self.class_weight == "balanced":
            return counts.sum() / (len(classes) * counts)
        return np.array([self.class_weight.get(class_label, 1.0) for class_label in classes])

    def _statistics(self, y, sample_weight=None) -> np.ndarray:
        """
        Weight of each instance in the column of its class.

        Parameters
        ----------
        y : numpy.ndarray of shape (n_samples,)
            Target of the instances in the node.
        sample_weight : numpy.ndarray of shape (n_samples,), default=None
            Sample weights of the instances in the node.

        Returns
        -------
        statistics : numpy.ndarray of shape (n_samples, n_classes)
        """
        classes, y_idx, counts = np.unique(y, return_inverse=True, return_counts=True)

        statistics = np.zeros((len(y), len(classes)))
        statistics[np.arange(len(y)), y_idx] = self._class_weights(classes, counts)[y_idx]
        if sample_weight is not None:
            statistics[np.arange(len(y)), y_idx] *= sample_weight

        return statistics

    def _impurity(self, statistics) -> np.ndarray:
        """
        Impurity from the weights of the classes.

        Parameters
        ----------
        statistics : numpy.ndarray of shape (n_classes, ...)
            Weight of each class.

        Returns
        -------
        impurity : numpy.ndarray of shape (...)
        """
        return self._impurity_fun()(np.maximum(statistics, 0))  # no negative rounding errors

    def _weight(self, statistics) -> np.ndarray:
        """
        Total weight from the weights of the classes.

        Parameters
        ----------
        statistics : numpy.ndarray of shape (n_classes, ...)
            Weight of each class.

        Returns
        -------
        weight : numpy.ndarray of shape (...)
        """
        return np.maximum(statistics, 0).sum(axis=0)
