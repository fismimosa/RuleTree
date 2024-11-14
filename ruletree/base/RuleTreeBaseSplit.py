from abc import ABC, abstractmethod


class RuleTreeBaseSplit(ABC):
    @abstractmethod
    def __init__(self, ml_task):
        self.ml_task = ml_task