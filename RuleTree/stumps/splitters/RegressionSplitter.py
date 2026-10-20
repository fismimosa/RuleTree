"""RegressionSplitter: the best split for regression."""

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_poisson_deviance
from sklearn.tree import DecisionTreeRegressor

from RuleTree.stumps.splitters.DecisionTreeSplitter import DecisionTreeSplitter
from RuleTree.utils.define import MODEL_TYPE_REG


class RegressionSplitter(DecisionTreeSplitter):
    """
    Splitter for regression that maximizes the decrease of impurity.

    The numerical features are split by a scikit-learn ``DecisionTreeRegressor`` of depth 1 with the
    same parameters (``x <= threshold`` goes left). Then each value of each categorical feature is
    tested (``x == value`` goes left), and the best categorical split replaces the numerical one if
    its gain is higher (see DecisionTreeSplitter).

    Parameters
    ----------
    criterion : {'squared_error', 'absolute_error', 'poisson'} or callable, default='squared_error'
        Impurity measure. A callable ``f(y_true, y_pred) -> float`` (e.g. a scikit-learn metric) is
        supported only for categorical features.
    min_samples_leaf : int or float, default=1
        Minimum number of instances in each child (a fraction of the instances of the node if
        float). Applies to the numerical features.
    min_weight_fraction_leaf : float, default=0.0
        Minimum fraction of the sample weights of the node in each child. Applies to the numerical
        features.
    max_features : int, float, {'sqrt', 'log2'} or None, default=None
        Number of numerical features considered for the split, drawn at random; None for all.
    random_state : int, RandomState instance or None, default=None
        Seed of the choice of the features (``max_features``) and of the ties between features with
        the same gain. With None, a tree gives the stump its own ``random_state``.
    min_impurity_decrease : float, default=0.0
        A numerical split is kept only if it decreases the weighted impurity by at least this value.

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
    ClassificationSplitter : the same splitter for classification.
    DecisionTreeStumpRegressor : the stump made of a MatrixTransformer and this splitter.

    Notes
    -----
    The categorical splits do not use ``sample_weight`` and ignore ``min_samples_leaf``.
    On a binary feature, the threshold is halfway between its two values; the categorical values that
    would leave a child empty are not tested.

    Examples
    --------
    >>> from sklearn.datasets import load_diabetes
    >>> from RuleTree.stumps.splitters import RegressionSplitter
    >>> X, y = load_diabetes(return_X_y=True)
    >>> splitter = RegressionSplitter(random_state=0).fit(X, y, numerical=list(range(10)), categorical=[])
    >>> splitter.feature_, round(float(splitter.threshold_), 4)
    (8, -0.0038)
    """

    task = MODEL_TYPE_REG

    _sklearn_tree = DecisionTreeRegressor

    def __init__(self, criterion='squared_error', min_samples_leaf=1, min_weight_fraction_leaf=0.0,
                 max_features=None, random_state=None, min_impurity_decrease=0.0):
        """Initialize the RegressionSplitter; see the class documentation for the parameters."""
        super().__init__(criterion, min_samples_leaf, min_weight_fraction_leaf, max_features, random_state,
                         min_impurity_decrease)

    @classmethod
    def _get_impurity_fun(cls, imp):
        """
        Impurity function of a criterion.

        Parameters
        ----------
        imp : str or callable
            Name of the criterion, or the impurity function itself.

        Returns
        -------
        impurity_fun : callable
            Function (y_true, y_pred) -> impurity.

        Raises
        ------
        NotImplementedError
            For 'friedman_mse'.
        """
        if imp == "squared_error":
            return _mean_squared_error
        elif imp == "friedman_mse":
            raise NotImplementedError("not implemented")  # TODO: implement in DecisionTreeRegressor?
        elif imp == "absolute_error":
            return mean_absolute_error
        elif imp == "poisson":
            return mean_poisson_deviance
        else:
            return imp

    @classmethod
    def _impurity_fun(cls, impurity_fun, **x):
        """
        Impurity of y_true predicted as y_pred, 0 if there are no instances.

        Parameters
        ----------
        impurity_fun : str or callable
            Criterion.
        **x : dict
            ``y_true`` and ``y_pred``.

        Returns
        -------
        impurity : float
        """
        f = cls._get_impurity_fun(impurity_fun)
        return f(**x) if len(x["y_true"]) > 0 else 0  # TODO: check

    @classmethod
    def _split_impurity(cls, criterion, y, left):
        """
        Impurity of the node and of the two children, each predicted with its mean. With several targets,
        each target is predicted with its mean and the impurity is the average over the targets.

        Parameters
        ----------
        criterion : str or callable
            Criterion.
        y : numpy.ndarray of shape (n_samples,) or (n_samples, n_targets)
            Target of the instances in the node.
        left : numpy.ndarray of bool
            Instances that go to the left child.

        Returns
        -------
        impurity : list of float
            Impurity of the node, of the left child and of the right child (0 for an empty child).
        """
        def impurity(y_true):
            y_pred = np.broadcast_to(np.mean(y_true, axis=0), y_true.shape) if len(y_true) > 0 else y_true
            return cls._impurity_fun(criterion, y_true=y_true, y_pred=y_pred)

        return [impurity(y), impurity(y[left]), impurity(y[~left])]

    def _impurities(self, y, left, sample_weight=None):
        """
        Impurity of the node and of the two children, each predicted with its mean (see _split_impurity).

        Parameters
        ----------
        y : numpy.ndarray of shape (n_samples,) or (n_samples, n_targets)
            Target of the instances in the node.
        left : numpy.ndarray of bool
            Instances that go to the left child.
        sample_weight : numpy.ndarray of shape (n_samples,), default=None
            Not used.

        Returns
        -------
        impurity : list of float
            Impurity of the node, of the left child and of the right child.
        """
        return self._split_impurity(self.criterion, y, left)


def _mean_squared_error(y_true, y_pred) -> float:
    """
    sklearn.metrics.mean_squared_error (averaged over the targets), with the same operations and so
    the same result, without its checks of the input, which are most of its time in the loops over
    the candidate splits.
    """
    errors = (np.asarray(y_true) - np.asarray(y_pred)) ** 2
    return float(np.mean(errors) if errors.ndim == 1 else np.mean(np.mean(errors, axis=0)))
