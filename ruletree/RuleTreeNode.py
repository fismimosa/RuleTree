from typing import Self

import numpy as np
from sklearn import tree

class RuleTreeNode:

    def __init__(self,
                 node_id: str,
                 prediction: int | str | float,
                 prediction_probability: np.ndarray | float,
                 parent: Self | None,
                 clf: tree = None,
                 node_l: Self = None,
                 node_r: Self = None,
                 samples: int = None,
                 **kwargs):
        self.node_id = node_id
        self.prediction = prediction
        self.prediction_probability = prediction_probability
        self.parent = parent
        self.clf = clf
        self.node_l = node_l
        self.node_r = node_r
        self.samples = samples

        for name, value in kwargs.items():
            setattr(self, name, value)

    def is_leaf(self):
        return self.node_l is None and self.node_r is None

    def make_leaf(self):
        self.node_l, self.node_r = None, None
        return self

    def simplify(self) -> Self:
        self._simplify()
        return self

    def _simplify(self):
        if self.is_leaf():
            return {self.prediction}
        else:
            all_pred = self.node_l._simplify() | self.node_r._simplify() | {self.prediction}

            if len(all_pred) == 1:
                self.make_leaf()
                return {self.prediction}
            else:
                return all_pred

    def get_possible_outputs(self) -> set:
        if self.is_leaf():
            return {self.prediction}
        else:
            return self.node_l.get_possible_outputs() | self.node_r.get_possible_outputs() | {self.prediction}

    def get_depth(self):
        return len(self.node_id) - 1

    def get_rule(self):
        if self.is_leaf():
            return {
                "node_id": self.node_id,
                "is_leaf": True,
                "prediction": self.prediction,
                "prediction_probability": self.prediction_probability,
                "samples": self.samples,
            }
        else:
            return {
                "node_id": self.node_id,
                "is_leaf": False,
                "prediction": self.prediction,
                "prediction_probability": self.prediction_probability,
                "feature_idx": self.clf.feature_original[0],
                "threshold_idx": self.clf.threshold_original[0],
                "is_categorical": self.clf.is_categorical,
                "samples": self.samples,
                "left_node": self.node_l.get_rule(),
                "right_node": self.node_r.get_rule(),
            }
            
    def encode_node(self, index, parent, vector, clf, node_index=0):
        if self.is_leaf():
            vector[0][node_index] = -1
            vector[1][node_index] = self.prediction
        else:
            feat = self.clf.feature_original[0]
            thr = self.clf.threshold_original[0]
         
            vector[0][node_index] = feat + 1
            vector[1][node_index] = thr

            node_l = self.node_l
            node_r = self.node_r

            index[node_l.node_id] = 2 * node_index + 1
            index[node_r.node_id] = 2 * node_index + 2
            
            parent[2 * index[self.node_id] + 1] = index[self.node_id]
            parent[2 * index[self.node_id] + 2] = index[self.node_id]

            node_l.encode_node(index, parent, vector, clf, 2 * node_index + 1)
            node_r.encode_node(index, parent, vector, clf, 2 * node_index + 2)
