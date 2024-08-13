import heapq
from typing import Tuple

import numpy as np
import pandas as pd
from numpy import bool_
from sklearn import tree
from sklearn.base import ClassifierMixin

from ruletree.RuleTree import RuleTree
from ruletree.RuleTreeNode import RuleTreeNode
from ruletree.utils.MyDecisionTreeClassifier import MyDecisionTreeClassifier
from ruletree.utils.data_utils import calculate_mode, get_info_gain, simplify_decode


class RuleTreeClassifier(RuleTree, ClassifierMixin):
    def __init__(self,
                 max_leaf_nodes=float('inf'),
                 min_samples_split=2,
                 max_depth=float('inf'),
                 prune_useless_leaves=False,
                 random_state=None,

                 criterion='gini',
                 splitter='best',
                 min_samples_leaf=1,
                 min_weight_fraction_leaf=0.0,
                 max_features=None,
                 min_impurity_decrease=0.0,
                 class_weight=None,
                 ccp_alpha=0.0,
                 monotonic_cst=None
                 ):
        super().__init__(max_leaf_nodes=max_leaf_nodes,
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
        self.class_weight = class_weight
        self.ccp_alpha = ccp_alpha
        self.monotonic_cst = monotonic_cst

    def is_split_useless(self, clf: tree, idx: np.ndarray):
        labels = clf.apply(self.X[idx])

        return len(np.unique(labels)) == 1

    def check_additional_halting_condition(self, curr_idx: np.ndarray):
        return len(np.unique(self.y[curr_idx])) == 1  # only 1 target

    def queue_push(self, node: RuleTreeNode, idx: np.ndarray):
        heapq.heappush(self.queue, (len(node.node_id), next(self.tiebreaker), idx, node))

    def make_split(self, X: np.ndarray, y, idx: np.ndarray, sample_weight=None, **kwargs) -> tree:
        clf = MyDecisionTreeClassifier(
            max_depth=1,
            criterion=self.criterion,
            splitter=self.splitter,
            min_samples_split=self.min_samples_split,
            min_samples_leaf = self.min_samples_leaf,
            min_weight_fraction_leaf=self.min_weight_fraction_leaf,
            max_features=self.max_features,
            random_state=self.random_state,
            min_impurity_decrease=self.min_impurity_decrease,
            class_weight=self.class_weight,
            ccp_alpha=self.ccp_alpha,
            monotonic_cst = self.monotonic_cst
        )

        clf.fit(X[idx], y[idx], sample_weight=None if sample_weight is None else sample_weight[idx])

        return clf

    def prepare_node(self, y: np.ndarray, idx: np.ndarray, node_id: str) -> RuleTreeNode:
        prediction = calculate_mode(y[idx])
        predict_proba = sum(np.where(y[idx] == prediction, 1, 0)) / len(y[idx])


        return RuleTreeNode(
            node_id=node_id,
            prediction=prediction,
            prediction_probability=predict_proba,
            parent=None,
            clf=None,
            node_l=None,
            node_r=None,
            samples=len(y[idx]),
        )
    
    def fit(self, X: np.array, y: np.array=None, sample_weight=None, **kwargs):
        super().fit(X, y, sample_weight=sample_weight, **kwargs)

    
    @classmethod
    def decode_ruletree(cls, vector, n_features_in_, n_classes_, n_outputs_, 
                        numerical_idxs=None, categorical_idxs=None, criterion=None):
        
        idx_to_node = super().decode_ruletree(vector, n_features_in_, n_classes_, n_outputs_, 
                                              numerical_idxs, categorical_idxs, criterion)
        
        for index in range(len(vector[0])):
            if vector[0][index] == -1:
                idx_to_node[index].prediction = vector[1][index]
            else:
                clf = MyDecisionTreeClassifier(
                                        criterion=criterion)
                                   
            
                if numerical_idxs is not None:
                   clf.numerical = numerical_idxs
        
                if categorical_idxs is not None:
                   clf.categorical = categorical_idxs
                            
                if isinstance(vector[1][index], str):
                    configure_cat_split(clf, vector[0][index], vector[1][index])
                else:
                    configure_non_cat_split(clf, vector, index, 
                                               n_features_in_, n_classes_, n_outputs_)
                idx_to_node[index].clf = clf
                set_node_children(idx_to_node, index, vector)
                
        rule_tree = RuleTreeClassifier()
        simplify_decode(idx_to_node[0])
        rule_tree.root = idx_to_node[0]
        return rule_tree

