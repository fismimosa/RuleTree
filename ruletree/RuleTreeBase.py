from abc import ABC, abstractmethod

import numpy as np
from sklearn import tree
from sklearn.base import BaseEstimator

from ruletree.RuleTreeNode import RuleTreeNode


class RuleTreeBase(BaseEstimator, ABC):
    @abstractmethod
    def make_split(self, X: np.ndarray, y, idx: np.ndarray) -> tree:
        pass

    @abstractmethod
    def make_leaf(self, node: RuleTreeNode) -> RuleTreeNode:
        pass

    @abstractmethod
    def prepare_node(self, y: np.ndarray, idx: np.ndarray, node_id: str) -> RuleTreeNode:
        pass

    @abstractmethod
    def queue_push(self, node: RuleTreeNode, idx: np.ndarray):
        pass

    @abstractmethod
    def queue_pop(self) -> tuple[np.ndarray, RuleTreeNode]:
        pass

    @abstractmethod
    def check_additional_halting_condition(self, curr_idx: np.ndarray) -> bool:
        pass

    @abstractmethod
    def is_split_useless(self, clf: tree, idx: np.ndarray):
        pass

    @abstractmethod
    def _post_fit_fix(self):
        pass
