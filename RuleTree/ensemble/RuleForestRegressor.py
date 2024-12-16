from random import random

import numpy as np
import sklearn
from sklearn.ensemble import BaggingRegressor

from RuleTree import RuleTreeRegressor
from RuleTree.base.RuleTreeBase import RuleTreeBase


class RuleForestRegressor(BaggingRegressor, RuleTreeBase):
    def __init__(self,
                 n_estimators=100,
                 criterion='squared_error',
                 max_depth=None,
                 min_samples_split=2,
                 min_samples_leaf=1,
                 min_weight_fraction_leaf=0.0,
                 min_impurity_decrease=0.0,
                 max_leaf_nodes=float("inf"),
                 ccp_alpha=0.0,
                 prune_useless_leaves=False,
                 splitter='best',
                 *,
                 max_samples=None,
                 max_features=1.0,
                 bootstrap=True,
                 oob_score=False,
                 warm_start=False,
                 custom_estimator: sklearn.base.RegressorMixin = None,
                 n_jobs=None,
                 random_state=None,
                 verbose=0):
        self.n_estimators = n_estimators
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.min_impurity_decrease = min_impurity_decrease
        self.max_leaf_nodes = max_leaf_nodes
        self.ccp_alpha = ccp_alpha
        self.prune_useless_leaves = prune_useless_leaves
        self.splitter = splitter

        self.max_samples = max_samples
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.oob_score = oob_score
        self.warm_start = warm_start
        self.custom_estimator = custom_estimator
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.verbose = verbose

    def fit(self, X:np.ndarray, y:np.ndarray, sample_weight=None):
        if self.max_features is None:
            self.max_features = X.shape[1]

        if type(self.max_features) is str:
            if self.max_features == "sqrt":
                self.max_features = int(np.sqrt(X.shape[1]))
            elif self.max_features == "log2":
                self.max_features = int(np.log2(X.shape[1]))

        base_estimator = RuleTreeRegressor if self.custom_estimator is None else self.custom_estimator
        splitter = .5 if self.splitter == 'hybrid_forest' else self.splitter
        if type(splitter) is float:
            base_estimator = RuleTreeRegressor_choosing_splitter_randomly

        super().__init__(estimator=base_estimator(criterion=self.criterion,
                                                  max_depth=self.max_depth,
                                                  min_samples_split=self.min_samples_split,
                                                  min_samples_leaf=self.min_samples_leaf,
                                                  min_weight_fraction_leaf=self.min_weight_fraction_leaf,
                                                  min_impurity_decrease=self.min_impurity_decrease,
                                                  max_leaf_nodes=self.max_leaf_nodes,
                                                  ccp_alpha=self.ccp_alpha,
                                                  prune_useless_leaves=self.prune_useless_leaves,
                                                  splitter=self.splitter
                                                  ),
                         n_estimators=self.n_estimators,
                         max_samples=X.shape[0] if self.max_samples is None else self.max_samples,
                         max_features=self.max_features,
                         bootstrap=self.bootstrap,
                         bootstrap_features=True,
                         oob_score=self.oob_score,
                         warm_start=self.warm_start,
                         n_jobs=self.n_jobs,
                         random_state=self.random_state,
                         verbose=self.verbose)

        return super().fit(X, y, sample_weight=sample_weight)

class RuleTreeRegressor_choosing_splitter_randomly(RuleTreeRegressor):
    def __init__(self, splitter, **kwargs):
        if random() < splitter:
            if random() < splitter:
                splitter = 'random'
            else:
                splitter = 'best'
        kwargs["splitter"] = splitter
        super().__init__(**kwargs)