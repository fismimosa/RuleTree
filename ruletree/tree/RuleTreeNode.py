from typing import Self

import numpy as np
from sklearn import tree

from ruletree.base.RuleTreeBaseStump import RuleTreeBaseStump
from ruletree.stumps.classification.DecisionTreeStumpClassifier import DecisionTreeStumpClassifier


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
                 balance_score : float = None,
                 **kwargs):
        self.node_id = node_id
        self.prediction = prediction  # TODO: remove (by saving it in the clf)
        self.prediction_probability = prediction_probability  # TODO: remove (by saving it in the clf)
        self.parent = parent
        self.clf = clf
        self.node_l = node_l
        self.node_r = node_r
        self.samples = samples  # TODO: remove (by saving it in the clf)
        self.balance_score = balance_score  # TODO: remove (by saving it in the clf)

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

    def _simplify(self):  # TODO: update
        if self.is_leaf():
            return {self.prediction}
        else:
            all_pred = self.node_l._simplify() | self.node_r._simplify() | {self.prediction}

            if len(all_pred) == 1:
                self.make_leaf()
                return {self.prediction}
            else:
                return all_pred

    def set_clf(self, clf:RuleTreeBaseStump):
        self.clf = clf

    def get_possible_outputs(self) -> tuple[set, set]:  # TODO: update
        if self.is_leaf():
            return {self.prediction}, set()
        else:
            leaf_l, node_l = self.node_l.get_possible_outputs()
            leaf_r, node_r = self.node_r.get_possible_outputs()

            return leaf_l | leaf_r, node_l | node_r | {self.prediction}

    def get_depth(self):
        return len(self.node_id) - 1

    def get_rule(self):
        rule = {
            "node_id": self.node_id,
            "is_leaf": self.is_leaf(),
            "prediction": self.prediction,
            "prediction_probability": self.prediction_probability,
            "samples": self.samples,
        }
    
        if not self.is_leaf():
            rule.update({
                "feature_idx": self.clf.get_feature(),
                "threshold": self.clf.get_thresholds(),
                "is_categorical": self.clf.get_is_categorical(),
                "left_node": self.node_l.get_rule(),
                "right_node": self.node_r.get_rule(),
                "coefficients": self.clf.coefficients
            })
    
        return rule
    

    def node_to_dict(self):  # TODO: update
        info_dict = {
            "node_id": self.node_id,
            "is_leaf" : self.is_leaf(),
            "prediction": self.prediction,
            "prediction_probability": self.prediction_probability,
            "parent_id": self.parent.node_id if self.parent is not None else None,
            "node_l_id": self.node_l.node_id if self.node_l is not None else None,
            "node_r_id": self.node_r.node_id if self.node_r is not None else None,
            "samples": self.samples,
            "feature_idx": self.clf.get_feature() if self.clf is not None else None,
            "threshold": self.clf.get_thresholds() if self.clf is not None else None,
            "is_categorical": self.clf.get_is_categorical() if self.clf is not None else None,
            "coefficients": self.clf.coefficients if self.clf is not None else None,
            "kwargs" : self.clf.kwargs if self.clf is not None else {}
            
           
        }
        
        return info_dict

    def dict_to_node(self, info_dict):  # TODO: update
        node = RuleTreeNode(node_id = info_dict['node_id'],
                            prediction = info_dict['prediction'],
                            prediction_probability = info_dict['prediction_probability'],
                            parent = info_dict['parent_id'],
                            samples = info_dict['samples'])
        
        if info_dict['is_leaf'] == True:
            return node
        
        node.clf = DecisionTreeStumpClassifier(**info_dict['kwargs'])
        
        node.clf.feature_original = [info_dict['feature_idx'], -2, -2]
        node.clf.threshold_original =  np.array([info_dict['threshold'], -2, -2])
        node.clf.is_categorical = info_dict['is_categorical']
        

        return node
    

        
                        
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
