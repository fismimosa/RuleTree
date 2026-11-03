"""FlatMatrixSplitter: the best split among all the candidates, evaluated together with matrix operations."""

from abc import ABC, abstractmethod
from math import ceil

import numpy as np

from RuleTree.base.RuleTreeBaseSplitter import RuleTreeBaseSplitter
from RuleTree.exceptions import NoSplitFoundWarning
from RuleTree.utils.data_utils import check_one_target


class FlatMatrixSplitter(RuleTreeBaseSplitter, ABC):
    """
    Base class of the splitters that evaluate all the candidate splits of a node together:
    FlatMatrixClassificationSplitter and FlatMatrixRegressionSplitter.

    The candidates are the thresholds halfway between consecutive distinct values of each numerical
    feature (``x <= threshold`` goes left) and the values of each categorical feature (``x == value``
    goes left). They are described by flat arrays (feature and threshold of each candidate). The impurity
    of the children comes from statistics of the target that can be summed: cumulative sums along each
    sorted numerical feature, sums by value along each categorical feature.

    It does not replace ClassificationSplitter and RegressionSplitter, which are faster: it is meant for
    the choice of the split that a scikit-learn tree does not allow, through ``gain_function``.

    A subclass defines ``task``, :meth:`_statistics`, :meth:`_impurity` and :meth:`_weight`, and, if some
    children are not allowed, :meth:`_valid_children`.

    Parameters
    ----------
    criterion : str
        Impurity measure (see the subclasses).
    min_samples_leaf : int or float, default=1
        Minimum number of instances in each child (a fraction of the instances of the node if float).
    gain_function : callable, default=None
        Gain of all the candidates of a node, called once per node with their impurities, the data of the
        node and the conditions of the splits; None for :func:`information_gain`. The signature is
        ``f(impurity, impurity_left, impurity_right, weight_left, weight_right, X, y, features, thresholds,
        is_categorical, sample_weight=None) -> gains``: the arrays have one value per candidate, and X, y
        and sample_weight are the ones of the node. X is the matrix of the transformer of the stump, so
        with a transformer other than MatrixTransformer it may not contain the original columns.
        A gain of -inf or nan discards a candidate. :func:`candidate_mask` builds the mask of the candidates,
        for the functions that need it. To save a tree, the function must be importable, not a lambda.

    Attributes
    ----------
    feature_ : int
        Column of the feature matrix used by the split.
    threshold_ : float or object
        Threshold of a numerical split, or value of a categorical split.
    is_categorical_ : bool
        Whether the split is categorical.
    gain_ : float
        Gain of the split, as returned by ``gain_function``.
    impurity_ : list of float
        Impurity of the node, of the left child and of the right child, not changed by ``gain_function``.

    Examples
    --------
    A gain that prefers the splits with children of similar size:

    >>> from sklearn.datasets import load_iris
    >>> from RuleTree.stumps.splitters import FlatMatrixClassificationSplitter, candidate_mask, information_gain
    >>> def balanced_gain(impurity, impurity_left, impurity_right, weight_left, weight_right, X, y, features,
    ...                   thresholds, is_categorical, sample_weight=None):
    ...     left_fraction = candidate_mask(X, features, thresholds, is_categorical).mean(axis=0)
    ...     gains = information_gain(impurity, impurity_left, impurity_right, weight_left, weight_right)
    ...     return gains - abs(left_fraction - 0.5)
    >>> X, y = load_iris(return_X_y=True)
    >>> splitter = FlatMatrixClassificationSplitter(gain_function=balanced_gain)
    >>> splitter = splitter.fit(X, y, numerical=[0, 1, 2, 3], categorical=[], context=None, node_id=None, cache=None)
    >>> splitter.feature_, round(splitter.threshold_, 2)
    (2, 4.35)
    """

    def __init__(self, criterion, min_samples_leaf=1, gain_function=None):
        """Initialize the splitter; see the class documentation for the parameters."""
        self.criterion = criterion
        self.min_samples_leaf = min_samples_leaf
        self.gain_function = gain_function

    @abstractmethod
    def _statistics(self, y: np.ndarray, sample_weight=None) -> np.ndarray:
        """
        Statistics of each instance; the statistics of a child are the sums over its instances.

        Parameters
        ----------
        y : numpy.ndarray of shape (n_samples,)
            Target of the instances in the node.
        sample_weight : numpy.ndarray of shape (n_samples,), default=None
            Sample weights of the instances in the node.

        Returns
        -------
        statistics : numpy.ndarray of shape (n_samples, n_statistics)
        """

    @abstractmethod
    def _impurity(self, statistics: np.ndarray) -> np.ndarray:
        """
        Impurity from the summed statistics.

        Parameters
        ----------
        statistics : numpy.ndarray of shape (n_statistics, ...)
            Summed statistics.

        Returns
        -------
        impurity : numpy.ndarray of shape (...)
        """

    @abstractmethod
    def _weight(self, statistics: np.ndarray) -> np.ndarray:
        """
        Total weight of the instances from the summed statistics.

        Parameters
        ----------
        statistics : numpy.ndarray of shape (n_statistics, ...)
            Summed statistics.

        Returns
        -------
        weight : numpy.ndarray of shape (...)
        """

    def _valid_children(self, statistics_left, statistics_right) -> np.ndarray:
        """
        Candidates whose children are allowed by the criterion: all of them by default.

        Parameters
        ----------
        statistics_left, statistics_right : numpy.ndarray of shape (n_statistics, n_candidates)
            Statistics of the children of each candidate.

        Returns
        -------
        valid : numpy.ndarray of bool of shape (n_candidates,)
        """
        return np.ones(statistics_left.shape[1], dtype=bool)

    @staticmethod
    def _numerical_candidates(x, statistics) -> tuple:
        """
        Candidate splits of a numerical feature. Missing values go to the right child.

        Parameters
        ----------
        x : numpy.ndarray of shape (n_samples,)
            Values of the feature.
        statistics : numpy.ndarray of shape (n_samples, n_statistics)
            Statistics of each instance.

        Returns
        -------
        thresholds : numpy.ndarray of shape (n_candidates,)
            Threshold of each candidate.
        statistics_left : numpy.ndarray of shape (n_candidates, n_statistics)
            Statistics of the left child of each candidate.
        n_left : numpy.ndarray of shape (n_candidates,)
            Number of instances in the left child of each candidate.
        """
        x = x.astype(float)
        order = np.argsort(x, kind="stable")  # missing values last
        order = order[:len(x) - np.count_nonzero(np.isnan(x))]
        x_sorted = x[order]

        ends = np.flatnonzero(x_sorted[1:] != x_sorted[:-1])
        lower, upper = x_sorted[ends], x_sorted[ends + 1]
        thresholds = (lower + upper) / 2
        thresholds = np.where(thresholds == upper, lower, thresholds)  # rounded to upper: it would go left

        return thresholds, np.cumsum(statistics[order], axis=0)[ends], ends + 1

    @staticmethod
    def _categorical_candidates(x, statistics) -> tuple:
        """
        Candidate splits of a categorical feature.

        Parameters
        ----------
        x : numpy.ndarray of shape (n_samples,)
            Values of the feature.
        statistics : numpy.ndarray of shape (n_samples, n_statistics)
            Statistics of each instance.

        Returns
        -------
        values : numpy.ndarray of shape (n_candidates,)
            Value of each candidate.
        statistics_left : numpy.ndarray of shape (n_candidates, n_statistics)
            Statistics of the left child of each candidate.
        n_left : numpy.ndarray of shape (n_candidates,)
            Number of instances in the left child of each candidate.
        """
        values, inverse, n_left = np.unique(x, return_inverse=True, return_counts=True)
        statistics_left = np.zeros((len(values), statistics.shape[1]))
        np.add.at(statistics_left, inverse, statistics)

        return values, statistics_left, n_left

    def _candidate_splits(self, X, statistics, numerical, categorical) -> tuple:
        """
        All the candidate splits of a feature matrix.

        Parameters
        ----------
        X : numpy.ndarray of shape (n_samples, n_features)
            Feature matrix.
        statistics : numpy.ndarray of shape (n_samples, n_statistics)
            Statistics of each instance.
        numerical : list of int
            Indices of the numerical columns of X.
        categorical : list of int
            Indices of the categorical columns of X.

        Returns
        -------
        features : numpy.ndarray of int of shape (n_candidates,)
            Column of each candidate.
        thresholds : numpy.ndarray of object of shape (n_candidates,)
            Threshold or value of each candidate.
        is_categorical : numpy.ndarray of bool of shape (n_candidates,)
            Whether each candidate is categorical; the numerical ones come first.
        statistics_left : numpy.ndarray of shape (n_candidates, n_statistics)
            Statistics of the left child of each candidate.
        n_left : numpy.ndarray of shape (n_candidates,)
            Number of instances in the left child of each candidate.
        """
        candidates = [self._numerical_candidates(X[:, feature], statistics) for feature in numerical]
        candidates += [self._categorical_candidates(X[:, feature], statistics) for feature in categorical]

        lengths = [len(candidate[0]) for candidate in candidates]
        features = np.repeat(np.array([*numerical, *categorical], dtype=int), lengths)
        thresholds = np.concatenate([np.empty(0, dtype=object), *[c[0] for c in candidates]], dtype=object)
        is_categorical = np.repeat([False] * len(numerical) + [True] * len(categorical), lengths)
        statistics_left = np.vstack([np.empty((0, statistics.shape[1])), *[c[1] for c in candidates]])
        n_left = np.concatenate([np.empty(0, dtype=int), *[c[2] for c in candidates]])

        return features, thresholds, is_categorical, statistics_left, n_left

    def _min_samples_leaf(self, n_samples) -> int:
        """
        Minimum number of instances in each child of a node.

        Parameters
        ----------
        n_samples : int
            Number of instances in the node.

        Returns
        -------
        min_samples_leaf : int
            At least 1: a child cannot be empty.
        """
        if isinstance(self.min_samples_leaf, float):
            return max(1, ceil(self.min_samples_leaf * n_samples))
        return max(1, self.min_samples_leaf)

    def fit(self, X, y, numerical, categorical, sample_weight=None, check_input=True, *, context,
            node_id, cache):
        """
        Evaluate all the candidate splits and keep the one with the highest gain. Among splits with the
        same gain, the first one: numerical before categorical, in the order of the columns.

        Parameters
        ----------
        X : numpy.ndarray of shape (n_samples, n_features)
            Feature matrix of the instances in the node.
        y : numpy.ndarray of shape (n_samples,)
            Target of the instances in the node.
        numerical : list of int
            Indices of the numerical columns of X.
        categorical : list of int
            Indices of the categorical columns of X.
        sample_weight : numpy.ndarray of shape (n_samples,), default=None
            Sample weights of the instances in the node.
        check_input : bool, default=True
            Not used.
        context : RuleTree or None
            Tree being built, None outside a tree.
        node_id : str or None
            Identifier of the node, e.g. ``"Rl"`` (``context.get_node_by_id(node_id)``), None outside a tree.
        cache : dict or None
            Cache of the tree (namespace -> dictionary of values), None outside a tree: the component
            decides whether and how to use it (the ones of the library use ``self._get_cache(cache)``).

        Returns
        -------
        self : FlatMatrixSplitter
            The fitted splitter.

        Raises
        ------
        NoSplitFoundWarning
            If the node is pure, no candidate has at least min_samples_leaf instances on each side, or
            gain_function discards all the candidates.
        ValueError
            If y has several targets, or gain_function does not return one gain per candidate.
        """
        check_one_target(y, self.__class__.__name__)
        y = np.asarray(y).ravel()
        self.feature_, self.threshold_, self.is_categorical_, self.impurity_ = None, None, False, None
        self.gain_ = -np.inf

        statistics = self._statistics(y, sample_weight)
        statistics_node = statistics.sum(axis=0)
        impurity = self._impurity(statistics_node)
        if impurity <= np.finfo(float).eps:  # as scikit-learn, a pure node is not split
            raise NoSplitFoundWarning(f"No split found for X {X.shape} and y {np.unique(y)}")

        features, thresholds, is_categorical, statistics_left, n_left = self._candidate_splits(X, statistics, numerical,
                                                                                               categorical)

        valid = np.minimum(n_left, len(X) - n_left) >= self._min_samples_leaf(len(X))
        if not valid.any():
            raise NoSplitFoundWarning(f"No split with at least {self.min_samples_leaf} instances per side")

        statistics_left = statistics_left[valid].T
        statistics_right = statistics_node[:, None] - statistics_left
        features, thresholds, is_categorical = features[valid], thresholds[valid], is_categorical[valid]

        valid = self._valid_children(statistics_left, statistics_right)
        if not valid.any():
            raise NoSplitFoundWarning(f"No split allowed by the criterion {self.criterion!r}")

        statistics_left, statistics_right = statistics_left[:, valid], statistics_right[:, valid]
        features, thresholds, is_categorical = features[valid], thresholds[valid], is_categorical[valid]
        impurity_left, impurity_right = self._impurity(statistics_left), self._impurity(statistics_right)
        weight_left, weight_right = self._weight(statistics_left), self._weight(statistics_right)

        if self.gain_function is None:
            gains = information_gain(impurity, impurity_left, impurity_right, weight_left, weight_right)
        else:
            gains = np.asarray(self.gain_function(impurity, impurity_left, impurity_right, weight_left, weight_right,
                                                  X, y, features, thresholds, is_categorical, sample_weight),
                               dtype=float)
            if gains.shape != features.shape:
                raise ValueError(f"gain_function returned the shape {gains.shape} instead of {features.shape}")
            gains = np.where(np.isnan(gains), -np.inf, gains)

        best = int(np.argmax(gains))
        if gains[best] == -np.inf:
            raise NoSplitFoundWarning("gain_function discarded all the candidate splits")

        self.feature_ = int(features[best])
        self.threshold_ = thresholds[best]
        self.is_categorical_ = bool(is_categorical[best])
        self.gain_ = float(gains[best])
        self.impurity_ = [float(impurity), float(impurity_left[best]), float(impurity_right[best])]

        return self

    def update_statistics(self, X, y, sample_weight=None, *, context, node_id, cache):
        """
        Recompute the impurity of the node and of the two children.

        Parameters
        ----------
        X : numpy.ndarray of shape (n_samples, n_features)
            Feature matrix of the instances in the node.
        y : numpy.ndarray of shape (n_samples,)
            Target of the instances in the node.
        sample_weight : numpy.ndarray of shape (n_samples,), default=None
            Sample weights of the instances in the node.
        context : RuleTree or None
            The tree, None outside a tree.
        node_id : str or None
            Identifier of the node, e.g. ``"Rl"`` (``context.get_node_by_id(node_id)``), None outside a tree.
        cache : dict or None
            Cache of the tree (namespace -> dictionary of values), None outside a tree: the component
            decides whether and how to use it (the ones of the library use ``self._get_cache(cache)``).
        """
        left = self.apply(X, context=context, node_id=node_id, cache=cache) == 1
        statistics = self._statistics(np.asarray(y).ravel(), sample_weight)
        statistics_node, statistics_left = statistics.sum(axis=0), statistics[left].sum(axis=0)
        self.impurity_ = [float(self._impurity(statistics_node)), float(self._impurity(statistics_left)),
                          float(self._impurity(statistics_node - statistics_left))]


def information_gain(impurity, impurity_left, impurity_right, weight_left, weight_right) -> np.ndarray:
    """
    Decrease of impurity of the candidate splits.

    Parameters
    ----------
    impurity : float
        Impurity of the node.
    impurity_left, impurity_right : numpy.ndarray of shape (n_candidates,)
        Impurity of the children of each candidate.
    weight_left, weight_right : numpy.ndarray of shape (n_candidates,)
        Total weight of the children of each candidate.

    Returns
    -------
    gains : numpy.ndarray of shape (n_candidates,)
    """
    return impurity - (weight_left * impurity_left + weight_right * impurity_right) / (weight_left + weight_right)


def candidate_mask(X, features, thresholds, is_categorical) -> np.ndarray:
    """
    Instances that go to the left child of each candidate split. It uses n_samples * n_candidates bytes; a
    function can pass subsets of the candidates to process them in blocks.

    Parameters
    ----------
    X : numpy.ndarray of shape (n_samples, n_features)
        Feature matrix.
    features : numpy.ndarray of int of shape (n_candidates,)
        Column of each candidate.
    thresholds : numpy.ndarray of shape (n_candidates,)
        Threshold or value of each candidate.
    is_categorical : numpy.ndarray of bool of shape (n_candidates,)
        Whether each candidate is categorical.

    Returns
    -------
    left : numpy.ndarray of bool of shape (n_samples, n_candidates)
    """
    left = np.empty((len(X), len(features)), dtype=bool)
    for categorical in (False, True):
        for feature in np.unique(features[is_categorical == categorical]):
            columns = np.flatnonzero((features == feature) & (is_categorical == categorical))
            if categorical:
                left[:, columns] = X[:, feature, None] == thresholds[columns]
            else:
                left[:, columns] = X[:, feature, None].astype(float) <= thresholds[columns].astype(float)

    return left
