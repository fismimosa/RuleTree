from abc import ABC, abstractmethod

from sklearn.base import BaseEstimator


class RuleTreeBaseStump(BaseEstimator, ABC):
    @abstractmethod
    def get_rule(self, columns_names=None, scaler=None, float_precision:int|None=3):
        pass

    @abstractmethod
    def node_to_dict(self):
        pass

    @abstractmethod
    def dict_to_node(self, node_dict):
        pass
