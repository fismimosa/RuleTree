import heapq
from abc import ABC, abstractmethod
from collections import deque
from datetime import datetime
from itertools import count

import numpy as np
from sklearn import tree

from ruletree.RuleTreeBase import RuleTreeBase
from ruletree.RuleTreeNode import RuleTreeNode


class RuleTree(RuleTreeBase, ABC):
    def __init__(self,
                 max_leaf_nodes,
                 min_samples_split,
                 max_depth,
                 prune_useless_leaves,
                 random_state,
                 ):
        self.max_leaf_nodes = float("inf") if max_leaf_nodes is None else max_leaf_nodes
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.prune_useless_leaves = prune_useless_leaves
        self.random_state = random_state

    def fit(self, X: np.array, y: np.array = None, **kwargs):
        self.X = X
        self.y = y
        self.root = None
        self.tiebreaker = count()
        self.queue = list()
        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)

        idx = np.arange(X.shape[0])

        self.root = self.prepare_node(self.y, idx, "R")

        self.queue_push(self.root, idx)

        nbr_curr_nodes = 0
        while len(self.queue) > 0 and nbr_curr_nodes + len(self.queue) < self.max_leaf_nodes:
            idx, current_node = self.queue_pop()

            if len(idx) < self.min_samples_split:
                self.make_leaf(current_node)
                nbr_curr_nodes += 1
                continue

            if nbr_curr_nodes + len(self.queue) + 1 >= self.max_leaf_nodes:
                self.make_leaf(current_node)
                nbr_curr_nodes += 1
                continue

            if self.max_depth is not None and current_node.get_depth() >= self.max_depth:
                self.make_leaf(current_node)
                nbr_curr_nodes += 1
                continue

            if self.check_additional_halting_condition(curr_idx=idx):
                self.make_leaf(current_node)
                nbr_curr_nodes += 1
                continue

            clf = self.make_split(self.X, self.y, idx=idx, **kwargs)
            labels = clf.apply(self.X[idx])

            if self.is_split_useless(clf, idx):
                self.make_leaf(current_node)
                nbr_curr_nodes += 1
                continue

            idx_l, idx_r = idx[labels == 1], idx[labels == 2]

            current_node.clf = clf
            current_node.node_l = self.prepare_node(self.y, idx_l, current_node.node_id + "l", )
            current_node.node_r = self.prepare_node(self.y, idx_r, current_node.node_id + "r", )
            current_node.node_l.parent, current_node.node_r.parent = current_node, current_node

            self.queue_push(current_node.node_l, idx_l)
            self.queue_push(current_node.node_r, idx_r)

        if self.prune_useless_leaves:
            self.root = self.root.simplify()

        self._post_fit_fix()

        return self

    def predict(self, X: np.ndarray):
        labels, leaves, proba = self._predict(X, self.root)

        return labels

    def apply(self, X: np.ndarray):
        labels, leaves, proba = self._predict(X, self.root)

        return leaves

    def predict_proba(self, X: np.ndarray):
        labels, leaves, proba = self._predict(X, self.root)
        proba_matrix = np.zeros((X.shape[0], self.n_classes_))
        for classe in self.classes_:
            proba_matrix[labels == classe, self.classes_ == classe] = proba[labels == classe]

        return proba_matrix

    def _predict(self, X: np.ndarray, current_node: RuleTreeNode):
        if current_node.is_leaf():
            n = len(X)
            return np.array([current_node.prediction] * n), \
                np.array([current_node.node_id] * n), \
                np.array([current_node.prediction_probability] * n)

        else:
            labels, leaves, proba = (
                np.full(len(X), fill_value=-1,
                        dtype=object if type(current_node.prediction) is str else type(current_node.prediction)),
                np.zeros(len(X), dtype=object),
                np.ones(len(X), dtype=float) * -1
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

    def get_rules(self):
        return self.root.get_rule()

    def make_leaf(self, node: RuleTreeNode) -> RuleTreeNode:
        return node

    def queue_pop(self):
        el = heapq.heappop(self.queue)
        return el[-2:]

    def check_additional_halting_condition(self, curr_idx: np.ndarray):
        return False

    def _post_fit_fix(self):
        return

    @abstractmethod
    def make_split(self, X: np.ndarray, y, idx: np.ndarray, **kwargs) -> tree:
        pass

    @abstractmethod
    def prepare_node(self, y: np.ndarray, idx: np.ndarray, node_id: str) -> RuleTreeNode:
        pass

    @abstractmethod
    def queue_push(self, node: RuleTreeNode, idx: np.ndarray):
        pass

    @abstractmethod
    def is_split_useless(self, clf: tree, idx: np.ndarray):
        pass

    @classmethod
    def print_rules(cls, rules: dict, columns_names: list = None, ndigits=2, indent: int = 0, ):
        names = lambda x: f"X_{x}"

        if columns_names is not None:
            names = lambda x: columns_names[x]

        indentation = "".join(["|   " for _ in range(indent)])

        if rules["is_leaf"]:
            pred = rules['prediction']

            print(f"{indentation} output: "
                  f"{pred if type(pred) in [np.str_, np.string_, str] else round(pred, ndigits=ndigits)}")
        else:
            comparison = "==" if rules['is_categorical'] else "<="
            not_comparison = "!=" if rules['is_categorical'] else ">"
            feature_idx = rules['feature_idx']
            thr = rules['threshold_idx']
            print(f"{indentation}|--- {names(feature_idx)} {comparison} "
                  f"{thr if type(thr) in [np.str_, np.string_, str] else round(thr, ndigits=ndigits)}"
                  f"\t{rules['samples']}")
            cls.print_rules(rules=rules['left_node'], columns_names=columns_names, indent=indent + 1)

            print(f"{indentation}|--- {names(feature_idx)} {not_comparison} "
                  f"{thr if type(thr) in [np.str_, np.string_, str] else round(thr, ndigits=ndigits)}")
            cls.print_rules(rules=rules['right_node'], columns_names=columns_names, indent=indent + 1)

    @classmethod
    def decode_ruletree(cls, vector, n_features_in_, n_classes_, n_outputs_, 
                        numerical_idxs=None, categorical_idxs=None, criterion = None):

        #need to check if n_classes_ is actually necessary
        
        n_classes_ = np.array([n_classes_], dtype=np.intp)
        
        idx_to_node = {index: RuleTreeNode(node_id=None, prediction=None, prediction_probability=None, parent=-1) 
                       for index in range(len(vector[0]))}
        
        idx_to_node[0].node_id = 'R'
        idx_to_node[0].parent = -1
        
        return idx_to_node
                            
    def encode_ruletree(self):
        nodes = (2 ** (self.max_depth + 1)) - 1
        vector = np.zeros((2, nodes), dtype=object)
        
        index = {'R': 0}  # root index
        parent = {}
        
        if not hasattr(self, 'root'):
            raise ValueError('This RuleTree instance must be fitted before encoding.')
        else:
            root_node = self.root
        root_node.encode_node(index, parent, vector, self)
        
        
        for node in range(vector.shape[1]):
            if vector[0][node] == -1:
                # children update
                if ((2*node + 1) < (vector.shape[1] - 1)) and ((2*node + 2) < vector.shape[1]):
                    vector[0][2*node + 1] = -1
                    vector[1][2*node + 1] = vector[1][node]
                    vector[0][2*node + 2] = -1
                    vector[1][2*node + 2] = vector[1][node]

                    parent[2*node + 1] = node
                    parent[2*node + 2] = node

                    # update the node itself
                    vector[0][node] = vector[0][parent[node]]
                    vector[1][node] = vector[1][parent[node]]
                    
        self.vector = vector
        
        return vector

