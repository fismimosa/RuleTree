import heapq
import warnings

import numpy as np
import sklearn
from sklearn import tree
from sklearn.base import RegressorMixin

from ruletree.stumps.regression.DecisionTreeStumpRegressor import DecisionTreeStumpRegressor
from ruletree.tree.RuleTree import RuleTree
from ruletree.tree.RuleTreeNode import RuleTreeNode
from ruletree.utils.data_utils import get_info_gain


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
            stump.fit(X[idx], y[idx], **kwargs)
        elif self.stump_selection == 'best':
            clfs = []
            info_gains = []
            for _, stump in self._filter_types(X):
                stump = sklearn.clone(stump)
                stump.fit(X[idx], y[idx], **kwargs)

                gain = get_info_gain(stump)
                info_gains.append(gain)
                
                clfs.append(stump)

            stump = clfs[np.argmax(info_gains)]
        else:
            raise TypeError('Unknown stump selection method')

        return stump


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
