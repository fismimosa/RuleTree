from typing import Self

import numpy as np
from sklearn import tree
import pygraphviz as pgv

from ruletree.base.RuleTreeBaseStump import RuleTreeBaseStump
from ruletree.stumps.classification.DecisionTreeStumpClassifier import DecisionTreeStumpClassifier


class RuleTreeNode:

    def __init__(self,
                 node_id: str,
                 prediction: int | str | float,
                 prediction_probability: np.ndarray | float,
                 classes: np.ndarray,
                 parent: Self | None,
                 stump: RuleTreeBaseStump = None,
                 node_l: Self = None,
                 node_r: Self = None,
                 samples: int = None,
                 balance_score : float = None,
                 **kwargs):
        self.node_id = node_id
        self.prediction = prediction
        self.prediction_probability = prediction_probability
        self.classes = classes
        self.parent = parent
        self.stump = stump
        self.node_l = node_l
        self.node_r = node_r
        self.samples = samples  # TODO: remove (by saving it in the stump)
        self.balance_score = balance_score  # TODO: remove (by saving it in the stump)

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

    def set_stump(self, stump:RuleTreeBaseStump):
        self.stump = stump

    def get_possible_outputs(self) -> tuple[set, set]:  # TODO: update
        if self.is_leaf():
            return {self.prediction}, set()
        else:
            leaf_l, node_l = self.node_l.get_possible_outputs()
            leaf_r, node_r = self.node_r.get_possible_outputs()

            return leaf_l | leaf_r, node_l | node_r | {self.prediction}

    def get_depth(self):
        return len(self.node_id) - 1

    def get_rule(self, column_names=None, scaler=None):
        rule = {
            "node_id": self.node_id,
            "is_leaf": self.is_leaf(),
            "prediction": self.prediction,
            "prediction_probability": self.prediction_probability,
            "prediction_classes_": self.classes,
            "left_node": self.node_l.get_rule() if self.node_l is not None else None,
            "right_node": self.node_r.get_rule() if self.node_r is not None else None,
        }
    
        if not self.is_leaf():
            rule |= self.stump.get_rule(columns_names=column_names, scaler=scaler)
    
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
            "feature_idx": self.stump.get_feature() if self.stump is not None else None,
            "threshold": self.stump.get_thresholds() if self.stump is not None else None,
            "is_categorical": self.stump.get_is_categorical() if self.stump is not None else None,
            "coefficients": self.stump.coefficients if self.stump is not None else None,
            "kwargs" : self.stump.kwargs if self.stump is not None else {}
            
           
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
        
        node.stump = DecisionTreeStumpClassifier(**info_dict['kwargs'])
        
        node.stump.feature_original = [info_dict['feature_idx'], -2, -2]
        node.stump.threshold_original =  np.array([info_dict['threshold'], -2, -2])
        node.stump.is_categorical = info_dict['is_categorical']
        

        return node
    

        
                        
    def encode_node(self, index, parent, vector, stump, node_index=0):
        if self.is_leaf():
            vector[0][node_index] = -1
            vector[1][node_index] = self.prediction
        else:
            feat = self.stump.feature_original[0]
            thr = self.stump.threshold_original[0]
         
            vector[0][node_index] = feat + 1
            vector[1][node_index] = thr

            node_l = self.node_l
            node_r = self.node_r

            index[node_l.node_id] = 2 * node_index + 1
            index[node_r.node_id] = 2 * node_index + 2
            
            parent[2 * index[self.node_id] + 1] = index[self.node_id]
            parent[2 * index[self.node_id] + 2] = index[self.node_id]

            node_l.encode_node(index, parent, vector, stump, 2 * node_index + 1)
            node_r.encode_node(index, parent, vector, stump, 2 * node_index + 2)


    def export_graphviz(self, graph=None, columns_names=None, scaler=None, float_precision=3):

        if graph is None:
            graph = pgv.AGraph(name="RuleTree")

        if self.is_leaf():
            graph.add_node(self.node_id, label=self.prediction)
            if self.parent is not None:
                graph.add_edge(self.parent.node_id, self.node_id)

            return graph

        rule = self.stump.get_rule(columns_names=columns_names, scaler=scaler, float_precision=float_precision)

        graph.add_node(self.node_id, label=rule["textual_rule"])

        if self.node_l is not None:
            graph = self.node_l.export_graphviz(graph)
            graph.add_edge(self.node_id, self.node_l.node_id, color="green")
        if self.node_r is not None:
            graph = self.node_r.export_graphviz(graph)
            graph.add_edge(self.node_id, self.node_r.node_id, color="red")

        return graph


