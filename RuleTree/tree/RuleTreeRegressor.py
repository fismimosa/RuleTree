"""RuleTreeRegressor: the RuleTree for regression."""

import heapq
from typing import Union, Optional

import numpy as np
from sklearn.base import RegressorMixin

from RuleTree.utils.define import MODEL_TYPE_REG
from RuleTree.stumps.regression.DecisionTreeStumpRegressor import DecisionTreeStumpRegressor
from RuleTree.tree.RuleTree import RuleTree
from RuleTree.tree.RuleTreeNode import RuleTreeNode


class RuleTreeRegressor(RegressorMixin, RuleTree):
    """
    RuleTree for regression.

    The tree grows breadth-first. Each node predicts the mean of the target of its instances and
    stores its standard deviation in ``prediction_probability``.

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
    distance_measure : str or callable, default=None
        Not used.

    Attributes
    ----------
    root : RuleTreeNode
        Root of the fitted tree.
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
                 distance_measure=None,
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
                         random_state=random_state,
                         distance_measure=distance_measure)

    def queue_push(self, node: RuleTreeNode, idx: np.ndarray):
        """
        Push a node into the queue, with priority to the shallowest nodes (breadth-first growth).

        Parameters
        ----------
        node : RuleTreeNode
            The node.
        idx : numpy.ndarray
            Indices of the instances reaching the node.
        """
        heapq.heappush(self.queue, (len(node.node_id), next(self.tiebreaker), idx, node))

    def compute_medoids(self, X: np.ndarray, y, idx: np.ndarray, **kwargs):
        """
        Not implemented for regression: does nothing.

        Parameters
        ----------
        X : numpy.ndarray
            Not used.
        y : numpy.ndarray
            Not used.
        idx : numpy.ndarray
            Not used.
        **kwargs : dict
            Not used.
        """
        pass

    def prepare_node(self, y: np.ndarray, idx: np.ndarray, node_id: str, node: Optional[RuleTreeNode] = None) -> RuleTreeNode:
        """
        Create a node (or update ``node``) with the mean (prediction) and the standard deviation of the
        target and the number of the instances ``idx``.

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
        prediction = float(np.mean(y[idx]))
        prediction_std = float(np.std(y[idx]))

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
        Mean (prediction) and standard deviation of the target of the instances ``idx``, as in fit, and
        statistics of the stump. A node without instances predicts nan.

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
            node.prediction, node.prediction_probability, node.samples = np.nan, np.nan, 0
            return

        self.prepare_node(y, idx, node.node_id, node=node)
        if node.stump is not None and not node.is_leaf():
            node.stump.update_statistics(data, y, idx=idx, context=self)

    def _get_stumps_task(self):
        """
        Task of the stumps of the tree.

        Returns
        -------
        task : str
            MODEL_TYPE_REG.
        """
        return MODEL_TYPE_REG
        
    def _get_prediction_probas(self, current_node = None, probas=None):
        """
        Predictions of the nodes in depth-first order (see RuleTree._get_prediction_probas).

        Parameters
        ----------
        current_node : RuleTreeNode, default=None
            Root of the subtree. If None, the root of the tree.
        probas : list, default=None
            List being filled.

        Returns
        -------
        probas : list of float
        """
        if probas is None:
            probas = []
            
        if current_node is None:
            current_node = self.root
        
    
        if current_node.prediction is not None:
            probas.append(current_node.prediction)
           
        if current_node.node_l:
            self._get_prediction_probas(current_node.node_l, probas)
            self._get_prediction_probas(current_node.node_r, probas)
        
        return probas
    
    
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


