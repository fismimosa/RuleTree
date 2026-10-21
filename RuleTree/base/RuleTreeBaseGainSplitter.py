"""
Base class of the splitters defined by a gain function: the simplest way to write a new splitter.
"""
from abc import ABC, abstractmethod

import numpy as np

from RuleTree.base.RuleTreeBaseSplitter import RuleTreeBaseSplitter
from RuleTree.exceptions import NoSplitFoundWarning


class RuleTreeBaseGainSplitter(RuleTreeBaseSplitter, ABC):
    """
    Base class of the splitters that test every candidate split and keep the one with the highest
    gain: a new splitter only defines the task (``task``) and the gain of a split (:meth:`gain`).

    The candidate splits are the thresholds halfway between consecutive distinct values of each
    numerical feature (``x <= threshold`` goes left) and the values of each categorical feature
    (``x == value`` goes left). Splits with fewer than ``min_samples_leaf`` instances on a side are
    skipped. The search is exhaustive and not optimized: ``gain`` is called once per candidate.

    Attributes
    ----------
    min_samples_leaf : int
        Minimum number of instances on each side of a split. 1 unless the subclass sets it (e.g.
        as a parameter of ``__init__``).

    See Also
    --------
    RuleTreeBaseSplitter : the attributes set by fit.

    Examples
    --------
    >>> import numpy as np
    >>> from RuleTree.base.RuleTreeBaseGainSplitter import RuleTreeBaseGainSplitter
    >>> from RuleTree.utils.define import MODEL_TYPE_CLF
    >>> class MisclassificationSplitter(RuleTreeBaseGainSplitter):
    ...     task = MODEL_TYPE_CLF
    ...
    ...     def gain(self, y, left, sample_weight=None):
    ...         # decrease of the number of errors when each side predicts its majority class
    ...         errors = lambda labels: len(labels) - np.unique(labels, return_counts=True)[1].max()
    ...         return errors(y) - errors(y[left]) - errors(y[~left])
    >>> X = np.array([[1.0], [2.0], [3.0], [4.0]])
    >>> splitter = MisclassificationSplitter()
    >>> splitter = splitter.fit(X, np.array([0, 0, 1, 1]), numerical=[0], categorical=[],
    ...                         context=None, node_id=None, cache=None)
    >>> splitter.feature_, splitter.threshold_
    (0, 2.5)
    """

    min_samples_leaf = 1

    @abstractmethod
    def gain(self, y: np.ndarray, left: np.ndarray, sample_weight=None) -> float:
        """
        Gain of a candidate split: the higher, the better.

        Parameters
        ----------
        y : numpy.ndarray of shape (n_samples,)
            Target of the instances in the node.
        left : numpy.ndarray of bool of shape (n_samples,)
            Instances that go to the left child.
        sample_weight : numpy.ndarray of shape (n_samples,), default=None
            Sample weights of the instances in the node.

        Returns
        -------
        gain : float
        """

    def candidate_thresholds(self, x: np.ndarray) -> np.ndarray:
        """
        Thresholds tested on a numerical feature: halfway between its consecutive distinct values.

        Parameters
        ----------
        x : numpy.ndarray of shape (n_samples,)
            Values of the feature.

        Returns
        -------
        thresholds : numpy.ndarray
        """
        values = np.unique(x.astype(float))
        return (values[:-1] + values[1:]) / 2

    def fit(self, X, y, numerical, categorical, sample_weight=None, check_input=True, *, context,
            node_id, cache):
        """
        Test every candidate split and keep the one with the highest gain.

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
            Sample weights of the instances in the node, passed to gain.
        check_input : bool, default=True
            Not used.
        context : RuleTree, default=None
            Tree being built, None outside a tree.
        node_id : str, default=None
            Identifier of the node, e.g. ``"Rl"`` (``context.get_node_by_id(node_id)``), None outside a tree.
        cache : dict, default=None
            Cache of the tree, when called by it; None otherwise.

        Returns
        -------
        self : RuleTreeBaseGainSplitter
            The fitted splitter.

        Raises
        ------
        NoSplitFoundWarning
            If no candidate split has at least min_samples_leaf instances on each side.
        """
        self.feature_, self.threshold_, self.is_categorical_, self.impurity_ = None, None, False, None
        self.gain_ = -np.inf

        candidates = [(feature, float(threshold), False)
                      for feature in numerical for threshold in self.candidate_thresholds(X[:, feature])]
        candidates += [(feature, value, True) for feature in categorical for value in np.unique(X[:, feature])]

        for feature, threshold, is_categorical in candidates:
            left = X[:, feature] == threshold if is_categorical else X[:, feature].astype(float) <= threshold
            if min(left.sum(), (~left).sum()) < self.min_samples_leaf:
                continue

            gain = self.gain(y, left, sample_weight)
            if gain > self.gain_:
                self.feature_, self.threshold_, self.is_categorical_, self.gain_ = feature, threshold, is_categorical, gain

        if self.feature_ is None:
            raise NoSplitFoundWarning(f"No split with at least {self.min_samples_leaf} instances per side")

        return self
