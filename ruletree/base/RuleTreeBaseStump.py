from abc import ABC, abstractmethod

from sklearn.base import BaseEstimator


class RuleTreeBaseStump(BaseEstimator, ABC):
    @abstractmethod
    def get_feature(self):
        pass

    @abstractmethod
    def get_thresholds(self):
        pass

    @abstractmethod
    def get_is_categorical(self):
        pass