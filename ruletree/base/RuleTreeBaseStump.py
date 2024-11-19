from abc import ABC, abstractmethod

from sklearn.base import BaseEstimator


class RuleTreeBaseStump(BaseEstimator, ABC):
    @abstractmethod
    def get_rule(self, columns_names=None, scaler=None, float_precision=3):
        pass

    @abstractmethod
    def node_to_dict(self, col_names):
        pass
