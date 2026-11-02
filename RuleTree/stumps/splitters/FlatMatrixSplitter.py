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
    goes left). They are described by flat arrays (feature and threshold of each candidate) and
    evaluated on a boolean matrix of shape (n_samples, n_candidates), so the impurity of the children of
    all the candidates comes from one matrix product.

    It does not replace ClassificationSplitter and RegressionSplitter, which are faster and use less
    memory: it is meant for the splitters that change the impurity of the children, which a scikit-learn
    tree does not allow. Such a splitter overrides :meth:`_penalize_impurity`.

    A subclass defines ``task`` and :meth:`_split_impurities`.

    Parameters
    ----------
    criterion : str
        Impurity measure (see the subclasses).
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
        Decrease of impurity of the split, computed with the impurities of :meth:`_penalize_impurity`.
    impurity_ : list of float
        Impurity of the node, of the left child and of the right child, without penalties.

    Notes
    -----
    The memory used grows with n_samples * n_candidates, and on a continuous feature there are about
    n_samples candidates.
    """

    def __init__(self, criterion, min_samples_leaf=1):
        """Initialize the splitter; see the class documentation for the parameters."""
        self.criterion = criterion
        self.min_samples_leaf = min_samples_leaf

    @abstractmethod
    def _split_impurities(self, y: np.ndarray, left: np.ndarray, sample_weight=None) -> tuple:
        """
        Impurity of the node and of the children of all the candidate splits, with their weights.

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

    def _penalize_impurity(self, X, y, left, impurity_left, impurity_right, sample_weight=None) -> tuple:
        """
        Impurity of the children used to compute the gain of the candidate splits. By default the
        impurity itself: a subclass overrides this method to penalize some splits.

        Parameters
        ----------
        X : numpy.ndarray of shape (n_samples, n_features)
            Feature matrix of the instances in the node.
        y : numpy.ndarray of shape (n_samples,)
            Target of the instances in the node.
        left : numpy.ndarray of bool of shape (n_samples, n_candidates)
            For each candidate split, the instances that go to the left child.
        impurity_left, impurity_right : numpy.ndarray of shape (n_candidates,)
            Impurity of the children of each candidate.
        sample_weight : numpy.ndarray of shape (n_samples,), default=None
            Sample weights of the instances in the node.

        Returns
        -------
        impurity_left, impurity_right : numpy.ndarray of shape (n_candidates,)
            Impurity to use for the children of each candidate.
        """
        return impurity_left, impurity_right

    def _candidate_splits(self, X, numerical, categorical) -> tuple:
        """
        All the candidate splits of a feature matrix.

        Parameters
        ----------
        X : numpy.ndarray of shape (n_samples, n_features)
            Feature matrix.
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
        n_numerical : int
            Number of numerical candidates, which come first.
        left : numpy.ndarray of bool of shape (n_samples, n_candidates)
            For each candidate, the instances that go to the left child.
        """
        candidates = []
        for feature in numerical:
            values = np.unique(X[:, feature].astype(float))
            candidates.append((values[:-1] + values[1:]) / 2)
        candidates += [np.unique(X[:, feature]) for feature in categorical]

        n_numerical = sum(len(thresholds) for thresholds in candidates[:len(numerical)])
        features = np.repeat(np.array([*numerical, *categorical], dtype=int), [len(c) for c in candidates])
        thresholds = np.concatenate([np.empty(0, dtype=object), *candidates], dtype=object)

        left = np.empty((len(X), len(features)), dtype=bool)
        left[:, :n_numerical] = (X[:, features[:n_numerical]].astype(float)
                                 <= thresholds[:n_numerical].astype(float))
        left[:, n_numerical:] = X[:, features[n_numerical:]] == thresholds[n_numerical:]

        return features, thresholds, n_numerical, left

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
            If the node is pure, or no candidate has at least min_samples_leaf instances on each side.
        ValueError
            If y has several targets.
        """
        check_one_target(y, self.__class__.__name__)
        y = np.asarray(y).ravel()
        self.feature_, self.threshold_, self.is_categorical_, self.impurity_ = None, None, False, None
        self.gain_ = -np.inf

        features, thresholds, n_numerical, left = self._candidate_splits(X, numerical, categorical)

        n_left = left.sum(axis=0)
        valid = np.flatnonzero(np.minimum(n_left, len(X) - n_left) >= self._min_samples_leaf(len(X)))
        if len(valid) == 0:
            raise NoSplitFoundWarning(f"No split with at least {self.min_samples_leaf} instances per side")

        left = left[:, valid]
        impurity, impurity_left, impurity_right, weight_left, weight_right = self._split_impurities(y, left,
                                                                                                    sample_weight)
        if impurity <= np.finfo(float).eps:  # as scikit-learn, a pure node is not split
            raise NoSplitFoundWarning(f"No split found for X {X.shape} and y {np.unique(y)}")

        penalized_left, penalized_right = self._penalize_impurity(X, y, left, impurity_left, impurity_right,
                                                                  sample_weight)
        gains = impurity - (weight_left * penalized_left + weight_right * penalized_right) / (weight_left + weight_right)

        best = int(np.argmax(gains))
        self.feature_ = int(features[valid[best]])
        self.threshold_ = thresholds[valid[best]]
        self.is_categorical_ = bool(valid[best] >= n_numerical)
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
        impurity, impurity_left, impurity_right, _, _ = self._split_impurities(np.asarray(y).ravel(), left[:, None],
                                                                               sample_weight)
        self.impurity_ = [float(impurity), float(impurity_left[0]), float(impurity_right[0])]
