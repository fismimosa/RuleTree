import heapq
import warnings
from typing import Union, Optional

import numpy as np
import sklearn
from sklearn import tree
from sklearn.base import RegressorMixin

from RuleTree.stumps.regression.DecisionTreeStumpRegressor import DecisionTreeStumpRegressor
from RuleTree.tree.RuleTree import RuleTree
from RuleTree.tree.RuleTreeNode import RuleTreeNode
from RuleTree.utils.data_utils import get_info_gain


class RuleTreeRegressor(RuleTree, RegressorMixin):
    """
    Implementation of Rule Tree for regression tasks.
    
    A Rule Tree is a decision tree variant where each node can contain different 
    types of decision functions (stumps). This flexibility allows for more 
    powerful and potentially interpretable models compared to standard 
    decision trees.
    
    Parameters
    ----------
    max_leaf_nodes : int, default=inf
        Maximum number of leaf nodes allowed in the tree.
    
    min_samples_split : int, default=2
        Minimum number of samples required to split a node.
    
    max_depth : int, default=inf
        Maximum depth of the tree.
    
    prune_useless_leaves : bool, default=False
        Whether to prune leaves that don't add information.
    
    base_stumps : RegressorMixin or list, default=None
        Base stumps to use for splitting nodes. If None, DecisionTreeStumpRegressor is used.
    
    stump_selection : str, default='random'
        Method to select stumps. Options: 'random', 'best'.
    
    random_state : int, default=None
        Random seed for reproducibility.
    
    criterion : str, default='squared_error'
        Function to measure the quality of a split.
    
    splitter : str, default='best'
        Strategy used to choose the split at each node.
    
    min_samples_leaf : int, default=1
        Minimum number of samples required to be at a leaf node.
    
    min_weight_fraction_leaf : float, default=0.0
        Minimum weighted fraction of the sum total of weights required to be at a leaf node.
    
    max_features : int, float, str, default=None
        Number of features to consider when looking for the best split.
    
    min_impurity_decrease : float, default=0.0
        A node will be split if this split induces a decrease of the impurity greater than this value.
    
    ccp_alpha : non-negative float, default=0.0
        Complexity parameter used for minimal cost-complexity pruning.
    
    monotonic_cst : array-like of int, default=None
        Monotonicity constraints for features.
    
    oblique : bool, default=False
        Whether to use oblique splits.
    
    oblique_params : dict, default={}
        Parameters for oblique splits.
    
    oblique_split_type : str, default='householder'
        Type of oblique split to use.
    
    force_oblique : bool, default=False
        Whether to force using oblique splits.
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

        if base_stumps is None:
            base_stumps = DecisionTreeStumpRegressor(max_depth=1)

        super().__init__(max_leaf_nodes=max_leaf_nodes,
                         min_samples_split=min_samples_split,
                         max_depth=max_depth,
                         prune_useless_leaves=prune_useless_leaves,
                         base_stumps=base_stumps,
                         stump_selection=stump_selection,
                         random_state=random_state,
                         distance_measure=distance_measure)

    def is_split_useless(self, X: np.ndarray, clf: tree, idx: np.ndarray):
        """
        Determine if a split is useless by checking if all samples end up in the same leaf.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.
        clf : sklearn.tree estimator
            The classifier to evaluate.
        idx : array-like
            Indices of the samples to consider.
            
        Returns
        -------
        bool
            True if the split is useless, False otherwise.
        """
        labels = clf.apply(X[idx])
        return len(np.unique(labels)) == 1

    def queue_push(self, node: RuleTreeNode, idx: np.ndarray):
        """
        Push a node to the priority queue for processing.
        
        Parameters
        ----------
        node : RuleTreeNode
            The node to push to the queue.
        idx : array-like
            Indices of the samples that reach this node.
        """
        heapq.heappush(self.queue, (len(node.node_id), next(self.tiebreaker), idx, node))

    def make_split(self, X: np.ndarray, y, idx: np.ndarray, **kwargs) -> tree:
        """
        Make a split by selecting and fitting a stump on the data.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.
        y : array-like of shape (n_samples,)
            The target values.
        idx : array-like
            Indices of the samples to consider.
        **kwargs : dict
            Additional parameters to pass to the stump's fit method.
            
        Returns
        -------
        tree
            The fitted stump.
            
        Raises
        ------
        TypeError
            If an unknown stump selection method is specified.
        """
        if self.stump_selection == 'random':
            stump = self._get_random_stump(X)
            stump.fit(X=X,
                      y=y,
                      idx=idx,
                      context=self,
                      **kwargs)
        elif self.stump_selection == 'best':
            clfs = []
            info_gains = []
            for _, stump in self._filter_types(X):
                stump = sklearn.clone(stump)
                stump.fit(X=X,
                          y=y,
                          idx=idx,
                          context=self,
                          **kwargs)

                gain = get_info_gain(stump)
                info_gains.append(gain)
                
                clfs.append(stump)

            stump = clfs[np.argmax(info_gains)]
        else:
            raise TypeError('Unknown stump selection method')

        return stump


    def compute_medoids(self, X: np.ndarray, y, idx: np.ndarray, **kwargs):
        """
        Compute the medoids of the data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.
        y : array-like of shape (n_samples,)
            The target values.
        idx : array-like
            Indices of the samples to consider.
        **kwargs : dict
            Additional parameters.

        Notes
        -----
        This method is currently a placeholder.
        """
        pass

    def prepare_node(self, y: np.ndarray, idx: np.ndarray, node_id: str, node: Optional[RuleTreeNode] = None) -> RuleTreeNode:
        """
        Prepare a node with prediction information.
        
        Parameters
        ----------
        y : array-like of shape (n_samples,)
            The target values.
        idx : array-like
            Indices of the samples that reach this node.
        node_id : str
            Identifier for the node.
            
        Returns
        -------
        RuleTreeNode
            The prepared node with prediction and other metadata.
        """
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            prediction = float(np.mean(y[idx]))
            prediction_std = float(np.std(y[idx]))

        if node is not None:
            node.prediction = prediction
            node.predict_proba = prediction_std
            node.classes = self.classes_
            node.samples = len(y[idx])

            return node

        return RuleTreeNode(
            node_id=node_id,
            prediction=prediction,
            prediction_probability=prediction_std,
            parent=None,
            stump=None,
            node_l=None,
            node_r=None,
            samples=len(y[idx]),
            classes=self.classes_,
            n_features=self.n_features,
        )

    def _get_stumps_base_class(self):
        """
        Get the base class for stumps used in the regressor.
        
        Returns
        -------
        class
            The RegressorMixin class.
        """
        return RegressorMixin
        
    def _get_prediction_probas(self, current_node = None, probas=None):
        """
        Get the prediction probabilities for all nodes in the tree.
        
        Parameters
        ----------
        current_node : RuleTreeNode, default=None
            The current node being processed. If None, starts from the root.
        probas : list, default=None
            List to store the probabilities.
            
        Returns
        -------
        list
            The prediction probabilities.
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
        Compute local interpretations of the model predictions.
        
        This method calculates feature contributions to the predictions for each sample.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples for which to compute interpretations.
        joint_contribution : bool, default=False
            Whether to compute joint contributions of features.
            
        Returns
        -------
        tuple
            The contributions of features to the predictions.
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


