import heapq
import warnings
from random import random

import numpy as np
from sklearn import tree
from sklearn.base import RegressorMixin

from ruletree.RuleTree import RuleTree
from ruletree.RuleTreeNode import RuleTreeNode
from ruletree.stumps.DecisionTreeStumpRegressor import DecisionTreeStumpRegressor, MyObliqueDecisionTreeRegressor
from ruletree.utils.data_utils import get_info_gain


class RuleTreeRegressor(RuleTree, RegressorMixin):
    def __init__(self,
                 max_leaf_nodes=float('inf'),
                 min_samples_split=2,
                 max_depth=float('inf'),
                 prune_useless_leaves=False,
                 base_stump: RegressorMixin | list = None,
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
        super().__init__(max_leaf_nodes=max_leaf_nodes,
                         min_samples_split=min_samples_split,
                         max_depth=max_depth,
                         prune_useless_leaves=prune_useless_leaves,
                         base_stump=base_stump,
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
        splitter = .5 if self.splitter == 'hybrid' else self.splitter
        if type(splitter) is float:
            if random() < splitter:
                splitter = 'random'
            else:
                splitter = 'best'

        clf = DecisionTreeStumpRegressor(
            max_depth=1,
            criterion=self.criterion,
            splitter=splitter,
            min_samples_split=self.min_samples_split,
            min_samples_leaf = self.min_samples_leaf,
            min_weight_fraction_leaf=self.min_weight_fraction_leaf,
            max_features=self.max_features,
            random_state=self.random_state,
            min_impurity_decrease=self.min_impurity_decrease,
            ccp_alpha=self.ccp_alpha,
            monotonic_cst = self.monotonic_cst
        )

        clf.fit(X[idx], y[idx], **kwargs)
        
        if self.oblique:
            clf_obl = MyObliqueDecisionTreeRegressor(
                max_depth=1,
                criterion=self.criterion,
                splitter=splitter,
                min_samples_split=self.min_samples_split,
                min_samples_leaf = self.min_samples_leaf,
                min_weight_fraction_leaf=self.min_weight_fraction_leaf,
                max_features=self.max_features,
                random_state=self.random_state,
                min_impurity_decrease=self.min_impurity_decrease,
                ccp_alpha=self.ccp_alpha,
                monotonic_cst = self.monotonic_cst,
                oblique_params = self.oblique_params,
                oblique_split_type =  self.oblique_split_type
            )

            clf_obl.fit(X[idx], y[idx])
            
            if clf_obl is not None:
                gain_obl = get_info_gain(clf_obl.oblique_split.oblq_clf)
                gain_univ = get_info_gain(clf)
                
                if gain_obl > gain_univ or self.force_oblique:
                    clf = clf_obl

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
