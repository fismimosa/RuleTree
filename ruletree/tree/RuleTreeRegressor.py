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
                 base_stump: RegressorMixin | list = None,
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
        if base_stump is None:
            base_stump = DecisionTreeStumpRegressor(
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
                         base_stump=base_stump,
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

    def is_split_useless(self, clf: tree, idx: np.ndarray):
        labels = clf.apply(self.X[idx])
        return len(np.unique(labels)) == 1

    def queue_push(self, node: RuleTreeNode, idx: np.ndarray):
        heapq.heappush(self.queue, (len(node.node_id), next(self.tiebreaker), idx, node))

    def make_split(self, X: np.ndarray, y, idx: np.ndarray, **kwargs) -> tree:
        if self.stump_selection == 'random':
            clf = self._get_random_stump()
            clf.fit(X[idx], y[idx], **kwargs)
        elif self.stump_selection == 'best':
            clfs = []
            info_gains = []
            for _, clf in self.base_stump:
                clf = sklearn.clone(clf)
                clf.fit(X[idx], y[idx], **kwargs)

                gain = get_info_gain(clf)
                info_gains.append(gain)

            clf = clfs[np.argmax(info_gains)]
        else:
            raise Exception('Unknown stump selection method')

        return clf

    

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
            clf=None,
            node_l=None,
            node_r=None,
            samples=len(y[idx]),
        )

    def _get_stumps_base_class(self):
        return RegressorMixin
