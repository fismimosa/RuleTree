"""RuleTreeRegressor: the RuleTree for regression."""

from typing import Union, Optional

import numpy as np
from sklearn.base import RegressorMixin

from RuleTree.utils.define import MODEL_TYPE_REG
from RuleTree.stumps.regression.DecisionTreeStumpRegressor import DecisionTreeStumpRegressor
from RuleTree.tree.RuleTree import RuleTree
from RuleTree.tree.RuleTreeNode import RuleTreeNode


class RuleTreeRegressor(RegressorMixin, RuleTree):
    """
    RuleTree for regression, with one target or several targets at once.

    The tree grows breadth-first. Each node predicts the mean (or another statistic, see
    ``leaf_prediction``) of the target of its instances and stores its standard deviation in
    ``prediction_probability``.

    With a target of shape (n_samples, n_targets), the tree is a multi-target (multi-output) tree:
    its splits are shared by all the targets, chosen by the average of the impurities of the
    targets (weighted by ``target_weights``), and each node predicts one value per target, as the
    multi-output trees of scikit-learn and the predictive clustering trees.

    Parameters
    ----------
    max_leaf_nodes : int, default=inf
        Maximum number of leaves.
    min_samples_split : int, default=2
        Minimum number of instances to split a node.
    max_depth : int, default=inf
        Maximum depth of the tree (the root has depth 0).
    prune_useless_leaves : bool, default=False
        Whether to merge, after fit, the sibling leaves with the same prediction.
    base_stumps : RuleTreeBaseStump or list, default=None
        Stumps used to split the nodes (see :class:`RuleTree.tree.RuleTree.RuleTree`); they must be
        regression stumps. If None, DecisionTreeStumpRegressor().
    stump_selection : {'random', 'best'}, default='random'
        How the stump of each node is chosen among base_stumps.
    random_state : int, RandomState instance or None, default=None
        Seed of the random choice of the stumps, also given to the stumps without a random_state.
    leaf_prediction : {'mean', 'median'} or callable, default='mean'
        Prediction of each node, computed on the target of its instances (for each target): the mean,
        the median, or ``f(y_node)`` that returns a number (one target) or an array of shape
        (n_targets,).
    target_weights : None, 'balanced' or array-like of shape (n_targets,), default=None
        Multi-target trees only (ignored with one target): weight of each target in the impurity of the
        splits, i.e. in the average of the squared errors of the targets. None gives every target the
        same weight, as scikit-learn: the targets with larger values count more. 'balanced' weighs each
        target by the inverse of its variance on the training set, as the predictive clustering trees,
        so that every target counts the same whatever its scale. The weights are applied by multiplying
        each target by the square root of its weight before the stumps are fitted; the predictions are
        in the original scale.

    Attributes
    ----------
    root : RuleTreeNode
        Root of the fitted tree.
    n_targets_ : int
        Number of targets (1 for a target of shape (n_samples,)).
    target_scales_ : numpy.ndarray of shape (n_targets,) or None
        Multipliers of the targets seen by the stumps (see ``target_weights``); None if not used.
    candidate_splits_ : list of tuple
        ``(probability, stump, data_type)``: the stumps usable on the training data, with the
        data type they are used on.
    cache_ : dict
        Values cached by the stumps during fit (see :meth:`get_cache`).

    See Also
    --------
    RuleTreeClassifier : the same tree for classification.
    RuleTree.stumps.regression : the stumps for regression.

    Examples
    --------
    >>> from sklearn.datasets import load_diabetes
    >>> from RuleTree import RuleTreeRegressor
    >>> X, y = load_diabetes(return_X_y=True)
    >>> model = RuleTreeRegressor(max_depth=3, random_state=0).fit(X, y)
    >>> model.predict(X[:2]).round(1)
    array([208.6,  83.4])
    """
    def __init__(self,
                 max_leaf_nodes=float('inf'),
                 min_samples_split=2,
                 max_depth=float('inf'),
                 prune_useless_leaves=False,
                 base_stumps: Union[RegressorMixin, list] = None,
                 stump_selection:str='random',
                 random_state=None,
                 leaf_prediction='mean',
                 target_weights=None,
                 ):

        """Initialize the RuleTreeRegressor; see the class documentation for the parameters."""
        if base_stumps is None:
            base_stumps = DecisionTreeStumpRegressor()

        super().__init__(max_leaf_nodes=max_leaf_nodes,
                         min_samples_split=min_samples_split,
                         max_depth=max_depth,
                         prune_useless_leaves=prune_useless_leaves,
                         base_stumps=base_stumps,
                         stump_selection=stump_selection,
                         random_state=random_state)
        self.leaf_prediction = leaf_prediction
        self.target_weights = target_weights

    def fit(self, X: np.ndarray | None = None, y: np.ndarray | None = None,
            X_ts=None, X_img=None, X_txt=None, sample_weight=None):
        """
        Fit the tree; see RuleTree.fit. The target can have shape (n_samples,) or (n_samples, n_targets).

        Returns
        -------
        self : RuleTreeRegressor
        """
        y = check_target(y)
        self.n_targets_ = 1 if y.ndim == 1 else y.shape[1]
        self.target_scales_ = self._target_scales(y)
        return super().fit(X=X, y=y, X_ts=X_ts, X_img=X_img, X_txt=X_txt, sample_weight=sample_weight)

    def partial_fit(self, X: np.ndarray | None = None, y: np.ndarray | None = None,
                    X_ts=None, X_img=None, X_txt=None, sample_weight=None):
        """
        Grow the fitted tree further; see RuleTree.partial_fit. The target weights of the first fit are
        kept (computed on y for a tree loaded with from_dict).

        Returns
        -------
        self : RuleTreeRegressor
        """
        if self.root is None:
            return self.fit(X=X, y=y, X_ts=X_ts, X_img=X_img, X_txt=X_txt, sample_weight=sample_weight)
        y = check_target(y)
        if not hasattr(self, "target_scales_"):
            self.n_targets_ = 1 if y.ndim == 1 else y.shape[1]
            self.target_scales_ = self._target_scales(y)
        return super().partial_fit(X=X, y=y, X_ts=X_ts, X_img=X_img, X_txt=X_txt, sample_weight=sample_weight)

    def _check_target(self, y):
        """The target as an array of shape (n_samples,) or (n_samples, n_targets) (see check_target)."""
        return check_target(y)

    def _target_scales(self, y: np.ndarray):
        """
        Multipliers of the targets seen by the stumps, from ``target_weights``.

        Parameters
        ----------
        y : numpy.ndarray
            Target of the training set.

        Returns
        -------
        scales : numpy.ndarray of shape (n_targets,) or None
            None with one target or without weights.

        Raises
        ------
        ValueError
            If target_weights is not None, 'balanced' or one non-negative weight per target.
        """
        if self.target_weights is None or y.ndim == 1:  # with one target, the weight changes no split
            return None
        if isinstance(self.target_weights, str):
            if self.target_weights != "balanced":
                raise ValueError(f"target_weights must be None, 'balanced' or an array, got {self.target_weights!r}.")
            std = np.std(y, axis=0)
            return np.where(std > 0, 1 / np.where(std > 0, std, 1), 1.0)

        weights = np.asarray(self.target_weights, dtype=float)
        if weights.shape != (y.shape[1],) or np.any(weights < 0):
            raise ValueError(f"target_weights must have one non-negative weight per target ({y.shape[1]}), "
                             f"got {self.target_weights!r}.")
        return np.sqrt(weights)

    def _split_target(self, y):
        """Target seen by the stumps: each target multiplied by its scale (see target_weights)."""
        scales = getattr(self, "target_scales_", None)
        return y if scales is None else y * scales

    def prepare_node(self, y: np.ndarray, idx: np.ndarray, node_id: str, node: Optional[RuleTreeNode] = None) -> RuleTreeNode:
        """
        Create a node (or update ``node``) with the prediction (see ``leaf_prediction``) and the
        standard deviation of the target and the number of the instances ``idx``; with several targets,
        one value per target.

        Parameters
        ----------
        y : numpy.ndarray
            Target of the whole data set.
        idx : numpy.ndarray
            Indices of the instances reaching the node.
        node_id : str
            Identifier of the node.
        node : RuleTreeNode, default=None
            Existing node to update instead of creating a new one.

        Returns
        -------
        node : RuleTreeNode
        """
        # also used by RuleTreeClusterRegressor, which has no leaf_prediction parameter
        prediction = leaf_value(y[idx], getattr(self, "leaf_prediction", "mean"))
        prediction_std = float(np.std(y[idx])) if y.ndim == 1 else np.std(y[idx], axis=0)

        if node is not None:
            node.prediction = prediction
            node.prediction_probability = prediction_std
            node.classes = None
            node.samples = len(y[idx])

            return node

        return RuleTreeNode(
            node_id=node_id,
            prediction=prediction,
            prediction_probability=prediction_std,
            log_odds=np.nan,
            parent=None,
            stump=None,
            node_l=None,
            node_r=None,
            samples=len(y[idx]),
            classes=None,
        )

    def _update_node_statistics(self, node, data, y, idx):
        """
        Prediction and standard deviation of the target of the instances ``idx``, as in fit, and
        statistics of the stump. A node without instances predicts nan (for each target).

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
        if len(idx) == 0:
            missing = np.nan if y.ndim == 1 else np.full(y.shape[1], np.nan)
            node.prediction, node.prediction_probability, node.samples = missing, missing, 0
            return

        self.prepare_node(y, idx, node.node_id, node=node)
        if node.stump is not None and not node.is_leaf():
            node.stump.update_statistics(data, self._split_target(y), idx=idx,
                                         context=self, node_id=node.node_id, cache=self._stump_cache())

    def _get_stumps_task(self):
        """
        Task of the stumps of the tree.

        Returns
        -------
        task : str
            MODEL_TYPE_REG.
        """
        return MODEL_TYPE_REG
        
    def local_interpretation(self, X, joint_contribution = False):
        """
        Contribution of each feature to the predictions, as in treeinterpreter: the prediction of an
        instance is the one of the root (bias) plus the contributions of the features of the splits
        along its path.

        Parameters
        ----------
        X : numpy.ndarray of shape (n_samples, n_features)
            Tabular data.
        joint_contribution : bool, default=False
            If True, the contributions are given to the sets of features along the path.

        Returns
        -------
        prediction : numpy.ndarray of shape (n_samples,)
            Prediction of the leaf of each instance.
        bias : numpy.ndarray of shape (n_samples,)
            Prediction of the root.
        contributions : numpy.ndarray of shape (n_samples, n_features) or list of dict
            Contribution of each feature (or set of features, with joint_contribution).
        """
        if getattr(self, "n_targets_", 1) > 1:
            raise ValueError("local_interpretation supports trees with a single target.")
        leaves, paths, leaf_to_path, values = super().local_interpretation(X = X,
                                                                           joint_contribution = joint_contribution)
        
        values = values.squeeze(axis=1)
        biases = np.full(X.shape[0], values[paths[0][0]])
        line_shape = X.shape[1]
        
        return super().eval_contributions(
                                        leaves=leaves,
                                        paths=paths,
                                        leaf_to_path=leaf_to_path,
                                        values=values,
                                        biases=biases,
                                        line_shape=line_shape,
                                        joint_contribution=joint_contribution
                                    )




def check_target(y) -> np.ndarray:
    """
    The target of a regression tree as an array: of shape (n_samples,) for one target (also when given
    as (n_samples, 1), as scikit-learn) or (n_samples, n_targets).

    Parameters
    ----------
    y : array-like

    Returns
    -------
    y : numpy.ndarray

    Raises
    ------
    ValueError
        If y has more than two dimensions.
    """
    y = np.asarray(y, dtype=float)
    if y.ndim == 2 and y.shape[1] == 1:
        return y.ravel()
    if y.ndim not in (1, 2):
        raise ValueError(f"The target must have shape (n_samples,) or (n_samples, n_targets), got {y.shape}.")
    return y


def leaf_value(y: np.ndarray, how='mean'):
    """
    Prediction of a node from the target of its instances.

    Parameters
    ----------
    y : numpy.ndarray of shape (n_samples,) or (n_samples, n_targets)
        Target of the instances of the node.
    how : {'mean', 'median'} or callable, default='mean'
        Statistic of each target, or a function of y.

    Returns
    -------
    prediction : float or numpy.ndarray of shape (n_targets,)

    Raises
    ------
    ValueError
        If ``how`` is an unknown string.

    Examples
    --------
    >>> import numpy as np
    >>> from RuleTree.tree.RuleTreeRegressor import leaf_value
    >>> leaf_value(np.array([1.0, 2.0, 9.0]), "median")
    2.0
    >>> leaf_value(np.array([[1.0, 10.0], [3.0, 30.0]]))
    array([ 2., 20.])
    """
    if callable(how):
        value = how(y)
    elif how == "mean":
        value = np.mean(y, axis=0)
    elif how == "median":
        value = np.median(y, axis=0)
    else:
        raise ValueError(f"leaf_prediction must be 'mean', 'median' or a callable, got {how!r}.")

    return float(value) if y.ndim == 1 else np.asarray(value, dtype=float)
