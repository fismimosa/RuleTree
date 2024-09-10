import heapq
from random import random
from typing import Tuple

import numpy as np
import pandas as pd
from numpy import bool_
from sklearn import tree
from sklearn.base import ClassifierMixin

from ruletree.RuleTree import RuleTree
from ruletree.RuleTreeNode import RuleTreeNode
from ruletree.utils.DecisionTreeClassifierStump import MyDecisionTreeClassifier, MyObliqueDecisionTreeClassifier
from ruletree.utils.data_utils import calculate_mode, get_info_gain

from ruletree.utils.utils_decoding import configure_non_cat_split, configure_cat_split
from ruletree.utils.utils_decoding import set_node_children , simplify_decode



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
        self.oblique = oblique
        self.oblique_params = oblique_params
        self.oblique_split_type = oblique_split_type
        self.force_oblique = force_oblique

    def is_split_useless(self, clf: tree, idx: np.ndarray):
        labels = clf.apply(self.X[idx])

        return len(np.unique(labels)) == 1

    def check_additional_halting_condition(self, curr_idx: np.ndarray):
        return len(np.unique(self.y[curr_idx])) == 1  # only 1 target

    def queue_push(self, node: RuleTreeNode, idx: np.ndarray):
        heapq.heappush(self.queue, (len(node.node_id), next(self.tiebreaker), idx, node))

    def make_split(self, X: np.ndarray, y, idx: np.ndarray, sample_weight=None, **kwargs) -> tree:
        splitter = .5 if self.splitter == 'hybrid_tree' else self.splitter
        if type(splitter) is float:
            if random() < splitter:
                splitter = 'random'
            else:
                splitter = 'best'

        clf = MyDecisionTreeClassifier(
            max_depth=1,
            criterion=self.criterion,
            splitter=splitter,
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
        
        if self.oblique:
            clf_obl = MyObliqueDecisionTreeClassifier(
                max_depth=1,
                criterion=self.criterion,
                splitter=splitter,
                min_samples_split=self.min_samples_split,
                min_samples_leaf = self.min_samples_leaf,
                min_weight_fraction_leaf=self.min_weight_fraction_leaf,
                max_features=self.max_features,
                random_state=self.random_state,
                min_impurity_decrease=self.min_impurity_decrease,
                class_weight=self.class_weight,
                ccp_alpha=self.ccp_alpha,
                monotonic_cst = self.monotonic_cst,
                oblique_params = self.oblique_params,
                oblique_split_type =  self.oblique_split_type
            )

            clf_obl.fit(X[idx], y[idx], sample_weight=None if sample_weight is None else sample_weight[idx])
            
            if clf_obl is not None:
                gain_obl = get_info_gain(clf_obl.oblique_split.oblq_clf)
                gain_univ = get_info_gain(clf)
                
                if gain_obl > gain_univ or self.force_oblique:
                    clf = clf_obl

        return clf

    def prepare_node(self, y: np.ndarray, idx: np.ndarray, node_id: str) -> RuleTreeNode:
        prediction = calculate_mode(y[idx])
        predict_proba = np.zeros((len(self.classes_), ))
        for i, classe in enumerate(self.classes_):
            predict_proba[i] = sum(np.where(y[idx] == classe, 1, 0)) / len(y[idx])


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

    def predict_proba(self, X: np.ndarray):
        labels, leaves, proba = self._predict(X, self.root)

        return proba


    def _predict(self, X: np.ndarray, current_node: RuleTreeNode):
        if current_node.is_leaf():
            n = len(X)
            return np.array([current_node.prediction] * n), \
                np.array([current_node.node_id] * n), \
                np.zeros((len(X), len(self.classes_)), dtype=float) + current_node.prediction_probability

        else:
            labels, leaves, proba = (
                np.full(len(X), fill_value=-1,
                        dtype=object if type(current_node.prediction) is str else type(current_node.prediction)),
                np.zeros(len(X), dtype=object),
                np.ones((len(X), len(self.classes_)), dtype=float) * -1
            )

            clf = current_node.clf
            labels_clf = clf.apply(X)
            X_l, X_r = X[labels_clf == 1], X[labels_clf == 2]
            if X_l.shape[0] != 0:
                labels[labels_clf == 1], leaves[labels_clf == 1], proba[labels_clf == 1] = self._predict(X_l,
                                                                                                         current_node.node_l)
            if X_r.shape[0] != 0:
                labels[labels_clf == 2], leaves[labels_clf == 2], proba[labels_clf == 2] = self._predict(X_r,
                                                                                                         current_node.node_r)

            return labels, leaves, proba
    
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

