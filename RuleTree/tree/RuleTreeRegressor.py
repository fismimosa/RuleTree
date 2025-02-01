import heapq
import warnings

import numpy as np
import sklearn
from sklearn import tree
from sklearn.base import RegressorMixin

from RuleTree.stumps.regression.DecisionTreeStumpRegressor import DecisionTreeStumpRegressor
from RuleTree.tree.RuleTree import RuleTree
from RuleTree.tree.RuleTreeNode import RuleTreeNode
from RuleTree.utils.data_utils import get_info_gain


class RuleTreeRegressor(RuleTree, RegressorMixin):
    def __init__(self,
                 max_leaf_nodes=float('inf'),
                 min_samples_split=2,
                 max_depth=float('inf'),
                 prune_useless_leaves=False,
                 base_stumps: RegressorMixin | list = None,
                 stump_selection:str='random',
                 random_state=None,

                 criterion='squared_error',
                 splitter='best',
                 min_samples_leaf=1,
                 min_weight_fraction_leaf=0.0,
                 max_features=None,
                 min_impurity_decrease=0.0,
                 ccp_alpha=0.0,
                 monotonic_cst=None,
                 oblique = False,
                 oblique_params = {},
                 oblique_split_type =  'householder',
                 force_oblique = False
                 ):
        if base_stumps is None:
            base_stumps = DecisionTreeStumpRegressor(
                max_depth=1,
                criterion=criterion,
                splitter=splitter,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                min_weight_fraction_leaf=min_weight_fraction_leaf,
                max_features=max_features,
                random_state=random_state,
                min_impurity_decrease=min_impurity_decrease,
                ccp_alpha=ccp_alpha,
                monotonic_cst=monotonic_cst
            )

        super().__init__(max_leaf_nodes=max_leaf_nodes,
                         min_samples_split=min_samples_split,
                         max_depth=max_depth,
                         prune_useless_leaves=prune_useless_leaves,
                         base_stumps=base_stumps,
                         stump_selection=stump_selection,
                         random_state=random_state)

        self.criterion = criterion
        self.splitter = splitter
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_features = max_features
        self.min_impurity_decrease = min_impurity_decrease
        self.ccp_alpha = ccp_alpha
        self.monotonic_cst = monotonic_cst
        self.oblique = oblique
        self.oblique_params = oblique_params
        self.oblique_split_type = oblique_split_type
        self.force_oblique = force_oblique

    def is_split_useless(self, X, clf: tree, idx: np.ndarray):
        labels = clf.apply(X[idx])
        return len(np.unique(labels)) == 1

    def queue_push(self, node: RuleTreeNode, idx: np.ndarray):
        heapq.heappush(self.queue, (len(node.node_id), next(self.tiebreaker), idx, node))

    def make_split(self, X: np.ndarray, y, idx: np.ndarray, **kwargs) -> tree:
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
        pass
        
    def prepare_node(self, y: np.ndarray, idx: np.ndarray, node_id: str) -> RuleTreeNode:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            prediction = float(np.mean(y[idx]))
            prediction_std = float(np.std(y[idx]))

        return RuleTreeNode(
            node_id=node_id,
            prediction=prediction,
            prediction_probability=prediction_std,
            parent=None,
            stump=None,
            node_l=None,
            node_r=None,
            samples=len(y[idx]),
            classes=self.classes_
        )

    def _get_stumps_base_class(self):
        return RegressorMixin
        
    def _get_prediction_probas(self, current_node = None, probas=None):
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

