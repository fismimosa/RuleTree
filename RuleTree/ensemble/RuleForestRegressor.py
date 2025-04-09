from random import random

import numpy as np
import sklearn
from sklearn.ensemble import BaggingRegressor

from RuleTree import RuleTreeRegressor
from RuleTree.base.RuleTreeBase import RuleTreeBase


class RuleForestRegressor(BaggingRegressor, RuleTreeBase):
    """
    A rule-based implementation of Random Forest for regression problems.

    This ensemble method combines multiple RuleTreeRegressor estimators to improve
    predictive accuracy and control overfitting. It extends scikit-learn's BaggingRegressor
    with rule-based decision trees.

    Parameters
    ----------
    n_estimators : int, default=100
        The number of trees in the forest.

    criterion : {"squared_error", "absolute_error", "friedman_mse", "poisson"}, default="squared_error"
        The function to measure the quality of a split.

    max_depth : int, default=None
        The maximum depth of the tree. If None, nodes are expanded until all leaves
        contain min_samples_split samples or until all leaves are pure.

    min_samples_split : int or float, default=2
        The minimum number of samples required to split an internal node.

    min_samples_leaf : int or float, default=1
        The minimum number of samples required to be at a leaf node.

    min_weight_fraction_leaf : float, default=0.0
        The minimum weighted fraction of the sum total of weights required to be at a leaf node.

    min_impurity_decrease : float, default=0.0
        A node will be split if this split induces a decrease of the impurity greater than or equal
        to this value.

    max_leaf_nodes : int, default=inf
        Grow trees with max_leaf_nodes in best-first fashion. If None, then unlimited number of
        leaf nodes.

    ccp_alpha : non-negative float, default=0.0
        Complexity parameter used for Minimal Cost-Complexity Pruning.

    prune_useless_leaves : bool, default=False
        Whether to prune leaves that don't improve performance.

    splitter : {"best", "random", "hybrid_forest"} or float, default="best"
        The strategy used to choose the split at each node.
        - "best": choose the best split
        - "random": choose the best random split
        - "hybrid_forest": randomly choose between "best" and "random" for each tree
        - float (between 0 and 1): probability to choose "random" over "best"

    max_samples : int or float, default=None
        The number of samples to draw from X to train each base estimator.
        - If int, then draw max_samples samples.
        - If float, then draw max_samples * X.shape[0] samples.
        - If None, then draw X.shape[0] samples.

    max_features : int, float, {"auto", "sqrt", "log2"}, default=1.0
        The number of features to consider when looking for the best split.
        - If int, then consider max_features features at each split.
        - If float, then max_features is a fraction and int(max_features * n_features) features are considered.
        - If "sqrt", then max_features=sqrt(n_features).
        - If "log2", then max_features=log2(n_features).

    bootstrap : bool, default=True
        Whether samples are drawn with replacement.

    oob_score : bool, default=False
        Whether to use out-of-bag samples to estimate the generalization score.

    warm_start : bool, default=False
        When set to True, reuse the solution of the previous call to fit and add more
        estimators to the ensemble.

    custom_estimator : sklearn.base.RegressorMixin, default=None
        A custom estimator to use instead of RuleTreeRegressor.

    n_jobs : int, default=None
        The number of jobs to run in parallel. None means 1.

    random_state : int, RandomState instance or None, default=None
        Controls the random resampling of the original dataset.

    verbose : int, default=0
        Controls the verbosity when fitting and predicting.
    """
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
        """
        Build a forest of rule-based trees from the training set (X, y).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The training input samples.

        y : array-like of shape (n_samples,)
            The target values (real numbers for regression).

        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights. If None, then samples are equally weighted.

        Returns
        -------
        self : object
            Fitted estimator.
        """
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
    """
    A specialized RuleTreeRegressor that randomly chooses between 'best' and 'random' splitter.

    This class extends RuleTreeRegressor to support probabilistic selection of the splitter
    strategy. Based on the provided probability (splitter parameter), it will randomly
    determine whether to use 'best' or 'random' splitting strategy for each tree.

    Parameters
    ----------
    splitter : float
        A number between 0 and 1 representing the probability of choosing 'random' splitter.
        Higher values increase the chance of using random splits.

    **kwargs :
        Additional parameters passed to the parent RuleTreeRegressor class.
    """
    def __init__(self, splitter, **kwargs):
        if random() < splitter:
            if random() < splitter:
                splitter = 'random'
            else:
                splitter = 'best'
        kwargs["splitter"] = splitter
        super().__init__(**kwargs)
