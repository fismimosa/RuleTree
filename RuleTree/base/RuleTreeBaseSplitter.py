"""
Abstract base class of the splitters: the component of a stump that finds the best split of a
feature matrix (the matrix produced by the stump's transformer, see RuleTreeBaseTransformer).
"""
from abc import ABC, abstractmethod

import numpy as np
from sklearn.base import BaseEstimator


class RuleTreeBaseSplitter(BaseEstimator, ABC):
    """
    Abstract base class of the splitters, the component of a stump that finds the best split of a
    feature matrix: a threshold on a numerical feature (``x <= threshold`` goes to the left child)
    or a value of a categorical feature (``x == value`` goes to the left child).

    A new splitter implements :meth:`fit`, which sets the attributes below; :meth:`apply`,
    :meth:`get_state` and :meth:`set_state` work for any split of this form. Splitters are
    scikit-learn estimators: their parameters are the arguments of ``__init__``.

    Attributes
    ----------
    task : str
        Kind of target: MODEL_TYPE_CLF (classification) or MODEL_TYPE_REG (regression). Class attribute.
    feature_ : int
        Column of the matrix used by the split.
    threshold_ : float or object
        Threshold (numerical split) or value (categorical split).
    is_categorical_ : bool
        Whether the split is categorical.
    gain_ : float
        Gain of the split: the stumps with the highest gain are chosen with stump_selection='best'.
    impurity_ : list of float or None
        Impurity of the node and of the two children, if computed.

    See Also
    --------
    RuleTreeBaseGainSplitter : a simpler interface, where only the gain of a split is defined.
    RuleTree.stumps.Stump.Stump : the stump that combines a transformer and a splitter.

    Examples
    --------
    A splitter that splits the first numerical feature at its median (see :doc:`/guides/extending`):

    >>> import numpy as np
    >>> from RuleTree.base.RuleTreeBaseSplitter import RuleTreeBaseSplitter
    >>> from RuleTree.utils.define import MODEL_TYPE_REG
    >>> class MedianSplitter(RuleTreeBaseSplitter):
    ...     task = MODEL_TYPE_REG
    ...
    ...     def fit(self, X, y, numerical, categorical, sample_weight=None, check_input=True, *, context, node_id):
    ...         self.feature_ = numerical[0]
    ...         self.threshold_ = float(np.median(X[:, self.feature_]))
    ...         self.is_categorical_, self.gain_, self.impurity_ = False, 0.0, None
    ...         return self
    >>> X = np.array([[1.0], [2.0], [3.0], [4.0]])
    >>> MedianSplitter().fit(X, None, numerical=[0], categorical=[], context=None, node_id=None).apply(X)
    array([1, 1, 2, 2])
    """

    # MODEL_TYPE_CLF or MODEL_TYPE_REG: the kind of target the splitter works with
    task = None

    # fitted attributes saved by get_state
    _state = ("feature_", "threshold_", "is_categorical_", "gain_", "impurity_")

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray, numerical: list, categorical: list, sample_weight=None,
            check_input=True, *, context, node_id):
        """
        Find the best split of X.

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
            Whether to check the input data.
        context : RuleTree or None
            Tree being built, None outside a tree.
        node_id : str or None
            Identifier of the node being split, e.g. ``"Rl"``: ``context.get_node_by_id(node_id)`` is the
            node (still a leaf; its ancestors are already split). None outside a tree.

        Returns
        -------
        self : RuleTreeBaseSplitter
            The fitted splitter.

        Raises
        ------
        NoSplitFoundWarning
            If no split is found.
        """

    def apply(self, X: np.ndarray, context=None, node_id=None) -> np.ndarray:
        """
        Apply the split to a feature matrix.

        Parameters
        ----------
        X : numpy.ndarray of shape (n_samples, n_features)
            Feature matrix.
        context : RuleTree, default=None
            The tree, when called by it (fit, predictions); None otherwise.
        node_id : str, default=None
            Identifier of the node of the stump, when called by the tree; None otherwise.

        Returns
        -------
        labels : numpy.ndarray of int
            For each row, 1 if it goes to the left child, 2 otherwise.
        """
        X_feature = X[:, self.feature_]
        if self.is_categorical_:
            return np.where(X_feature == self.threshold_, 1, 2)

        return np.where(X_feature <= self.threshold_, 1, 2)

    def update_statistics(self, X: np.ndarray, y: np.ndarray, sample_weight=None, *, context, node_id):
        """
        Recompute the statistics of the split (e.g. impurity) on new data, without changing the split.
        The default implementation does nothing.

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
        """

    def get_state(self) -> dict:
        """
        Fitted state of the splitter: the attributes listed in ``_state``.

        Returns
        -------
        state : dict
            JSON-friendly dictionary, restored by :meth:`set_state`.
        """
        return {name: getattr(self, name, None) for name in self._state}

    def set_state(self, state: dict):
        """
        Restore the fitted state saved by :meth:`get_state`.

        Parameters
        ----------
        state : dict
            Dictionary created by :meth:`get_state`.

        Returns
        -------
        self : RuleTreeBaseSplitter
        """
        for name in self._state:
            setattr(self, name, state.get(name))

        return self
