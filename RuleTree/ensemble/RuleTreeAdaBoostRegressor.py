from sklearn.ensemble import AdaBoostRegressor

from RuleTree import RuleTreeRegressor
from RuleTree.base.RuleTreeBase import RuleTreeBase


class RuleTreeAdaBoostRegressor(AdaBoostRegressor, RuleTreeBase):
    def __init__(self,
                 n_estimators=50,
                 min_samples_split=2,
                 prune_useless_leaves=False,
                 random_state=None,
                 criterion='squared_error',
                 splitter='best',
                 min_samples_leaf=1,
                 min_weight_fraction_leaf=0.0,
                 max_features=None,
                 min_impurity_decrease=0.0,
                 ccp_alpha=0.0,
                 monotonic_cst=None,
                 *,
                 learning_rate=1.0,
                 loss='linear'
                 ):
        self.min_samples_split = min_samples_split
        self.prune_useless_leaves = prune_useless_leaves
        self.random_state = random_state
        self.criterion = criterion
        self.splitter = splitter
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_features = max_features
        self.min_impurity_decrease = min_impurity_decrease
        self.ccp_alpha = ccp_alpha
        self.monotonic_cst = monotonic_cst
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.loss = loss

        super().__init__(
            estimator=RuleTreeRegressor(max_depth=1, #stump
                                        prune_useless_leaves=prune_useless_leaves,
                                        random_state=random_state,
                                        criterion=criterion,
                                        splitter=splitter,
                                        min_samples_leaf=min_samples_leaf,
                                        min_weight_fraction_leaf=min_weight_fraction_leaf,
                                        max_features=max_features,
                                        min_impurity_decrease=min_impurity_decrease,
                                        ccp_alpha=ccp_alpha,
                                        monotonic_cst=monotonic_cst
                                        ),
            n_estimators=n_estimators, learning_rate=learning_rate, loss=loss, random_state=random_state
        )