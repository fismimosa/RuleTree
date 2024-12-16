from abc import ABC, abstractmethod

from sklearn.base import BaseEstimator

from ruletree.utils.define import DATA_TYPE_TABULAR


class RuleTreeBaseStump(BaseEstimator, ABC):
    @abstractmethod
    def get_rule(self, columns_names=None, scaler=None, float_precision:int|None=3):
        pass

    @abstractmethod
    def node_to_dict(self):
        pass

    @classmethod
    @abstractmethod
    def dict_to_node(self, node_dict, X):
        pass

    @staticmethod
    def supports(data_type):
        return data_type in [DATA_TYPE_TABULAR]
