from sklearn.ensemble import AdaBoostClassifier

from RuleTree import RuleTreeClassifier
from RuleTree.base.RuleTreeBase import RuleTreeBase


class RuleTreeAdaBoostClassifier(AdaBoostClassifier, RuleTreeBase):
    def __init__(self,
                 n_estimators=50,
                 min_samples_split=2,
                 prune_useless_leaves=False,
                 random_state=None,
                 criterion='gini',
                 splitter='best',
                 min_samples_leaf=1,
                 min_weight_fraction_leaf=0.0,
                 max_features=None,
                 min_impurity_decrease=0.0,
                 class_weight=None,
                 ccp_alpha=0.0,
                 monotonic_cst=None,
                 *,
                 learning_rate=1.0,
                 algorithm='SAMME'
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
        self.class_weight = class_weight
        self.ccp_alpha = ccp_alpha
        self.monotonic_cst = monotonic_cst
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.algorithm = algorithm

        estimator = RuleTreeClassifier(min_samples_split=min_samples_split,
                                         max_depth=3, #stump
                                         prune_useless_leaves=prune_useless_leaves,
                                         random_state=random_state,

                                         criterion=criterion,
                                         splitter=splitter,
                                         min_samples_leaf=min_samples_leaf,
                                         min_weight_fraction_leaf=min_weight_fraction_leaf,
                                         max_features=max_features,
                                         min_impurity_decrease=min_impurity_decrease,
                                         class_weight=class_weight,
                                         ccp_alpha=ccp_alpha,
                                         monotonic_cst=monotonic_cst
                                         )

        super().__init__(
            estimator=estimator,
            n_estimators=n_estimators, learning_rate=learning_rate, algorithm=algorithm, random_state=random_state
        )