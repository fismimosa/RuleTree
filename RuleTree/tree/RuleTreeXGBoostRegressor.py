"""RuleTreeXGBoostRegressor: the trees of XGBoostRegressor."""

import warnings

import numpy as np

from RuleTree import RuleTreeRegressor
from RuleTree.tree.RuleTreeRegressor import check_target
#from RuleTree.stumps.classification.XGBoostStumpClassifier import XGBoostStumpClassifier
from RuleTree.stumps.regression.XGBoostStumpRegressor import XGBoostStumpRegressor
from RuleTree.tree.RuleTreeNode import RuleTreeNode


class EmptyXGBTreeException(Exception):
    """Raised when the gamma pruning of a RuleTreeXGBoostRegressor removes all the splits."""
    pass


class RuleTreeXGBoostRegressor(RuleTreeRegressor):
    """
    RuleTreeRegressor with the splits and the leaf values of XGBoost, used as base estimator of
    XGBoostRegressor (fitted on the residuals).

    The splits maximize the XGBoost gain (see XGBoostStumpRegressor); after fit, the splits with a
    gain lower than ``gamma`` are pruned and each leaf predicts ``sum(y) / (n + lam)``.

    Parameters
    ----------
    gamma : float, default=0
        Minimum gain of a split.
    lam : float, default=1
        L2 regularization of the leaf values and of the similarity scores.
    max_depth : int, default=6
        Maximum depth of the tree.
    min_cover : float, default=1
        Minimum cover (number of instances) of each child.
    n_jobs : int, default=1
        Passed to the stumps.
    base_stumps : XGBoostStumpRegressor, default=None
        If None or not an XGBoostStumpRegressor, XGBoostStumpRegressor(lam, min_cover, n_jobs) with a
        warning.
    max_leaf_nodes, min_samples_split, prune_useless_leaves, stump_selection, random_state
        As in RuleTreeRegressor.

    Attributes
    ----------
    root : RuleTreeNode
        Root of the fitted tree; the ``prediction`` of each leaf is its XGBoost value.

    See Also
    --------
    RuleTree.ensemble.XGBoostRegressor.XGBoostRegressor : the boosting of these trees.

    Examples
    --------
    >>> from sklearn.datasets import load_diabetes
    >>> from RuleTree.stumps.regression import XGBoostStumpRegressor
    >>> from RuleTree.tree.RuleTreeXGBoostRegressor import RuleTreeXGBoostRegressor
    >>> X, y = load_diabetes(return_X_y=True)
    >>> tree = RuleTreeXGBoostRegressor(max_depth=2, lam=1, base_stumps=XGBoostStumpRegressor(lam=1))
    >>> tree = tree.fit(X, y - y.mean())  # residuals of the mean
    >>> tree.get_rules()["textual_rule"]
    'X_8 <= -0.004'
    """
    def __init__(self, gamma=0, lam=1, max_depth=6, min_cover=1, n_jobs=1, base_stumps=None,
                 max_leaf_nodes=float('inf'), min_samples_split=2, prune_useless_leaves=False,
                 stump_selection='random', random_state=None):
        """Initialize the RuleTreeXGBoostRegressor; see the class documentation for the parameters."""
        # explicit parameters (not **kwargs): get_params, clone and to_dict see only the ones in the signature
        if base_stumps is None or not issubclass(base_stumps.__class__, XGBoostStumpRegressor):
            warnings.warn('base_stumps will be set as XGBoostTreeStumpRegressor.',)
            base_stumps = XGBoostStumpRegressor(lam=lam, n_jobs=n_jobs, min_cover=min_cover)
        super().__init__(max_leaf_nodes=max_leaf_nodes, min_samples_split=min_samples_split, max_depth=max_depth,
                         prune_useless_leaves=prune_useless_leaves, base_stumps=base_stumps,
                         stump_selection=stump_selection, random_state=random_state)
        self.gamma = gamma
        self.lam = lam
        self.min_cover = min_cover
        self.n_jobs = n_jobs


    def fit(self, X: np.array = None, y: np.array = None,
            X_ts=None, X_img=None, X_txt=None, sample_weight=None):
        """
        Fit the tree, prune the splits with a gain lower than gamma and set the leaf values.

        Parameters
        ----------
        X : numpy.ndarray of shape (n_samples, n_features), default=None
            Tabular data.
        y : numpy.ndarray of shape (n_samples,), default=None
            Target (residuals, in XGBoostRegressor).
        X_ts, X_img, X_txt : default=None
            Other data types, see RuleTree.fit.
        sample_weight : numpy.ndarray of shape (n_samples,), default=None
            Sample weights, passed to the stumps.

        Returns
        -------
        self : RuleTreeXGBoostRegressor

        Raises
        ------
        EmptyXGBTreeException
            If the pruning removes all the splits.
        """
        y = check_target(y)  # also used below to compute the leaf values
        super().fit(X=X, y=y, X_ts=X_ts, X_img=X_img, X_txt=X_txt, sample_weight=sample_weight)
        if self._gamma_pruning(self.root).is_leaf():
            raise EmptyXGBTreeException('The value of gamma lead to an empty XGBoostTree.')

        leafs = self.apply(X=X, X_ts=X_ts, X_img=X_img, X_txt=X_txt)
        for leaf in np.unique(leafs):
            cond = leafs == leaf
            output_value = np.sum(y[cond])/(np.sum(cond) + self.lam)
            self.get_node_by_id(leaf).prediction = output_value

        return self

    def _update_node_statistics(self, node, data, y, idx):
        """
        As in RuleTreeRegressor, but the leaves predict the XGBoost output value, as in fit.

        Parameters
        ----------
        node : RuleTreeNode
            The node.
        data : dict
            Whole data set, as ``{data_type: array}``.
        y : numpy.ndarray
            Target of the whole data set.
        idx : numpy.ndarray
            Indices of the instances reaching the node.
        """
        super()._update_node_statistics(node, data, y, idx)
        if node.is_leaf() and len(idx) > 0:
            node.prediction = np.sum(y[idx]) / (len(idx) + self.lam)

    def _gamma_pruning(self, node:RuleTreeNode):
        """
        Make a leaf of every node whose split has a gain lower than gamma.

        Parameters
        ----------
        node : RuleTreeNode
            Root of the subtree.

        Returns
        -------
        node : RuleTreeNode
        """
        if node.is_leaf():
            return node
        if node.stump.gain < self.gamma:
            node.make_leaf()
            return node

        self._gamma_pruning(node.node_l)
        self._gamma_pruning(node.node_r)

        return node
