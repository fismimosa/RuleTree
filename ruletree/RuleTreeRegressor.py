import heapq
import warnings
from typing import Tuple

import numpy as np
from numpy import bool_
from sklearn import tree
from sklearn.base import RegressorMixin
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from ruletree.RuleTree import RuleTree
from ruletree.RuleTreeBase import RuleTreeBase
from ruletree.RuleTreeNode import RuleTreeNode
from ruletree.utils.MyDecisionTreeRegressor import MyDecisionTreeRegressor
from ruletree.utils.data_utils import calculate_mode


class RuleTreeRegressor(RuleTree, RegressorMixin):
    def __init__(self,
                 max_nbr_nodes=float('inf'),
                 min_samples_split=2,
                 max_depth=float('inf'),
                 prune_useless_leaves=False,
                 random_state=None,

                 criterion='squared_error',
                 splitter='best',
                 min_samples_leaf=1,
                 min_weight_fraction_leaf=0.0,
                 max_features=None,
                 min_impurity_decrease=0.0,
                 ccp_alpha=0.0,
                 monotonic_cst=None
                 ):
        super().__init__(max_nbr_nodes=max_nbr_nodes,
                         min_samples_split=min_samples_split,
                         max_depth=max_depth,
                         prune_useless_leaves=prune_useless_leaves,
                         random_state=random_state)

        self.criterion = criterion
        self.splitter = splitter
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_features = max_features
        self.min_impurity_decrease = min_impurity_decrease
        self.ccp_alpha = ccp_alpha
        self.monotonic_cst = monotonic_cst

    def is_split_useless(self, clf: tree, idx: np.ndarray):
        labels = clf.apply(self.X[idx])

        return len(np.unique(labels)) == 1

    def queue_push(self, node: RuleTreeNode, idx: np.ndarray):
        heapq.heappush(self.queue, (len(node.node_id), next(self.tiebreaker), idx, node))

    def make_split(self, X: np.ndarray, y, idx: np.ndarray) -> tree:
        clf = MyDecisionTreeRegressor(
            max_depth=1,
            criterion=self.criterion,
            splitter=self.splitter,
            min_samples_split=self.min_samples_split,
            min_samples_leaf = self.min_samples_leaf,
            min_weight_fraction_leaf=self.min_weight_fraction_leaf,
            max_features=self.max_features,
            random_state=self.random_state,
            min_impurity_decrease=self.min_impurity_decrease,
            ccp_alpha=self.ccp_alpha,
            monotonic_cst = self.monotonic_cst
        )

        clf.fit(X[idx], y[idx],)

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

