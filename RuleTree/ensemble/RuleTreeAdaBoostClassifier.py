from sklearn.ensemble import AdaBoostClassifier

from RuleTree import RuleTreeClassifier
from RuleTree.base.RuleTreeBase import RuleTreeBase


class RuleTreeAdaBoostClassifier(AdaBoostClassifier, RuleTreeBase):
    """An AdaBoost classifier that uses RuleTreeClassifier as the base estimator.
    
    This implementation combines the AdaBoost algorithm with RuleTree-based weak learners.
    It inherits from both AdaBoostClassifier and RuleTreeBase, providing boosting
    capabilities with rule-based decision trees.
    
    AdaBoost builds a strong classifier by combining multiple weak classifiers (RuleTrees)
    sequentially, where each subsequent model tries to correct errors made by 
    the previous ones by adjusting the weights of incorrectly classified examples.
    
    Parameters
    ----------
    n_estimators : int, default=50
        The maximum number of estimators at which boosting is terminated.
        In case of perfect fit, the learning procedure is stopped early.
    
    min_samples_split : int or float, default=2
        The minimum number of samples required to split an internal node.
        
    prune_useless_leaves : bool, default=False
        Whether to prune leaves that do not improve the performance.
        
    random_state : int, RandomState instance or None, default=None
        Controls the random seed given to each RuleTree estimator and the bootstrap
        sample selection for building the ensemble.
        
    criterion : {"gini", "entropy"}, default="gini"
        The function to measure the quality of a split.
        
    splitter : {"best", "random"}, default="best"
        The strategy used to choose the split at each node.
        
    min_samples_leaf : int or float, default=1
        The minimum number of samples required to be at a leaf node.
        
    min_weight_fraction_leaf : float, default=0.0
        The minimum weighted fraction of the sum total of weights required to be at a leaf node.
        
    max_features : int, float or {"auto", "sqrt", "log2"}, default=None
        The number of features to consider when looking for the best split.
        
    min_impurity_decrease : float, default=0.0
        A node will be split if this split induces a decrease of the impurity greater than or equal
        to this value.
        
    class_weight : dict, list of dict or "balanced", default=None
        Weights associated with classes.
        
    ccp_alpha : non-negative float, default=0.0
        Complexity parameter used for Minimal Cost-Complexity Pruning.
        
    monotonic_cst : array-like of int of shape (n_features), default=None
        Monotonicity constraints for each feature.
        
    learning_rate : float, default=1.0
        Learning rate shrinks the contribution of each classifier by `learning_rate`.
        
    algorithm : {'SAMME', 'SAMME.R'}, default='SAMME'
        The boosting algorithm to use.
    
    Attributes
    ----------
    estimator_ : RuleTreeClassifier
        The base estimator from which the ensemble is grown.
        
    estimators_ : list of RuleTreeClassifier
        The collection of fitted RuleTree sub-estimators.
        
    classes_ : ndarray of shape (n_classes,)
        The classes labels.
        
    n_classes_ : int
        The number of classes.
        
    See Also
    --------
    RuleTreeClassifier : Base rule tree classifier.
    sklearn.ensemble.AdaBoostClassifier : Original scikit-learn AdaBoost implementation.
    """
    
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
        """Initialize the RuleTreeAdaBoostClassifier.
        
        Parameters are the same as in the class docstring.
        """
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
