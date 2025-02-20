import importlib
from typing import Self

import graphviz
import numpy as np

from RuleTree.base.RuleTreeBaseStump import RuleTreeBaseStump
from RuleTree.utils.define import GRAPHVIZ_DEFAULT_NODE_SPLIT, GRAPHVIZ_DEFAULT_NODE_LEAF


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
                 **kwargs):
        self.node_id = node_id
        self.prediction = prediction
        self.prediction_probability = prediction_probability
        self.classes = classes
        self.parent = parent
        self.stump = stump
        self.node_l = node_l
        self.node_r = node_r
        self.medoids_index = None

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

    def get_rule(self, columns_names=None, scaler=None):
        rule = {
            "node_id": self.node_id,
            "is_leaf": self.is_leaf(),
            "prediction": self.prediction,
            "prediction_probability": self.prediction_probability,
            "prediction_classes_": self.classes,
            "left_node": self.node_l.get_rule(columns_names = columns_names, scaler = scaler) if self.node_l is not None else None,
            "right_node": self.node_r.get_rule(columns_names = columns_names, scaler = scaler) if self.node_r is not None else None,
        }
    
        if not self.is_leaf():
            rule |= self.stump.get_rule(columns_names=columns_names, scaler=scaler)
    
        return rule
    

    def node_to_dict(self):
        node_as_dict ={
            "node_id": self.node_id,
            "is_leaf": self.is_leaf(),
            "prediction": self.prediction,
            "prediction_probability": self.prediction_probability if isinstance(self.prediction_probability, float) else self.prediction_probability.tolist(),
            "prediction_classes_": self.classes.tolist(),
            "left_node": self.node_l.node_id if self.node_l is not None else None,
            "right_node": self.node_r.node_id if self.node_r is not None else None,
        }

        if not self.is_leaf():
            node_as_dict |= self.stump.node_to_dict()
        
        return node_as_dict

    @classmethod
    def dict_to_node(cls, info_dict, X = None):
        assert 'node_id' in info_dict
        assert 'is_leaf' in info_dict
        if not info_dict['is_leaf']:
            assert 'stump_type' in info_dict

        node = RuleTreeNode(node_id = info_dict['node_id'],
                            prediction = info_dict.get('prediction', np.nan),
                            prediction_probability = info_dict.get('prediction_probability', [np.nan]),
                            parent = None,
                            classes=info_dict.get('prediction_classes_', np.nan))
        
        if info_dict['is_leaf'] == True:
            return node

        import_path = info_dict['stump_type']
        print(import_path)
        class_c = getattr(importlib.import_module(import_path), info_dict['stump_type'].split('.')[-1])
        
        node.stump = class_c.dict_to_node(info_dict, X)
        

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
            graph = graphviz.Digraph(name="RuleTree")#pgv.AGraph(name="RuleTree")

        if self.is_leaf():
            graph.node(self.node_id, label=f"{self.prediction}", **GRAPHVIZ_DEFAULT_NODE_LEAF)

            return graph

        rule = self.stump.get_rule(columns_names=columns_names, scaler=scaler, float_precision=float_precision)

        graph.node(self.node_id, **(GRAPHVIZ_DEFAULT_NODE_SPLIT | rule["graphviz_rule"]))

        if self.node_l is not None:
            graph = self.node_l.export_graphviz(graph, columns_names, scaler, float_precision)
            graph.edge(self.node_id, self.node_l.node_id, color="#2ca02c")
        if self.node_r is not None:
            graph = self.node_r.export_graphviz(graph, columns_names, scaler, float_precision)
            graph.edge(self.node_id, self.node_r.node_id, color="#d62728")

        return graph


