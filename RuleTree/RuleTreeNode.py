from typing import Self

from sklearn import tree


class RuleTreeNode:

    def __init__(self,
                 node_id:str,
                 prediction:int|str|float,
                 prediction_probability:float,
                 parent:Self|None,
                 clf:tree=None,
                 node_l:Self=None,
                 node_r:Self=None,
                 samples:int=None,
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

    def get_possible_outputs(self)->set:
        if self.is_leaf():
            return {self.prediction}
        else:
            return self.node_l.get_possible_outputs() | self.node_r.get_possible_outputs()

    def simplify(self)->Self:
        if not self.is_leaf():
            if len(self.get_possible_outputs()) >= 2:
                self.node_r = self.node_r.simplify()
                self.node_l = self.node_l.simplify()
            else:
                self.node_r = None
                self.node_l = None

        return self

    def get_depth(self):
        return len(self.node_id)-1
