import warnings

import numpy as np
from line_profiler_pycharm import profile

from RuleTree import RuleTreeRegressor
from RuleTree.stumps.classification.XGBoostStumpClassifier import XGBoostStumpClassifier
from RuleTree.stumps.regression.XGBoostStumpRegressor import XGBoostStumpRegressor
from RuleTree.tree.RuleTreeNode import RuleTreeNode


class EmptyXGBTreeException(Exception):
    pass


class RuleTreeXGBoostRegressor(RuleTreeRegressor):
    def __init__(self, gamma=0, lam=1, max_depth=6, min_cover=1, n_jobs=1, base_stumps=None, **kwargs):
        if base_stumps is None or not issubclass(base_stumps.__class__, XGBoostStumpRegressor | XGBoostStumpClassifier):
            warnings.warn('base_stumps will be set as XGBoostTreeStumpRegressor.',)
            base_stumps = XGBoostStumpRegressor(lam=lam, n_jobs=n_jobs, min_cover=min_cover)
        super().__init__(max_depth=max_depth, base_stumps=base_stumps, **kwargs)
        self.gamma = gamma
        self.lam = lam
        self.min_cover = min_cover
        self.n_jobs = n_jobs


    def fit(self, X: np.array, y: np.array = None, **kwargs):
        super().fit(X, y, **kwargs)
        if self._gamma_pruning(self.root).is_leaf():
            raise EmptyXGBTreeException('The value of gamma lead to an empty XGBoostTree.')

        leafs = self.apply(X)
        for leaf in np.unique(leafs):
            cond = leafs == leaf
            output_value = np.sum(y[cond])/(np.sum(cond) + self.lam)
            self._get_node(leaf).prediction = output_value

    def _gamma_pruning(self, node:RuleTreeNode):
        if node.is_leaf():
            return node
        if node.stump.gain < self.gamma:
            node.make_leaf()
            return node

        self._gamma_pruning(node.node_l)
        self._gamma_pruning(node.node_r)

        return node