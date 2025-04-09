"""
This file contains the implementation of a custom AdaBoost Regressor using Rule Trees.
It combines the power of AdaBoost with the specific characteristics of Rule Tree Regression.
"""
from sklearn.ensemble import AdaBoostRegressor

from RuleTree import RuleTreeRegressor
from RuleTree.base.RuleTreeBase import RuleTreeBase


class RuleTreeAdaBoostRegressor(AdaBoostRegressor, RuleTreeBase):
    """
    A custom AdaBoost Regressor that uses Rule Tree Regression as its base estimator.

    This class implements the AdaBoost algorithm with Rule Tree Regressors as the base estimators.
    It uses decision stumps (Rule Trees with max_depth=1) by default as weak learners.
    The ensemble builds multiple Rule Trees sequentially, with each tree focusing more
    on instances that previous trees had higher error on by adjusting instance weights.

    The implementation inherits from both AdaBoostRegressor and RuleTreeBase, providing
    interpretability features from Rule Trees along with the improved accuracy from boosting.

    Parameters:
        n_estimators (int): The number of estimators at which boosting is terminated.
        min_samples_split (int): The minimum number of samples required to split an internal node.
        prune_useless_leaves (bool): Whether to prune leaves that do not improve the fit.
        random_state (int or None): Controls the randomness of the estimator.
        criterion (str): The function to measure the quality of a split.
        splitter (str): The strategy used to choose the split at each node.
        min_samples_leaf (int): The minimum number of samples required to be at a leaf node.
        min_weight_fraction_leaf (float): The minimum weighted fraction of the sum total of weights required to be at a leaf node.
        max_features (int, float, str or None): The number of features to consider when looking for the best split.
        min_impurity_decrease (float): A node will be split if this split induces a decrease of the impurity greater than or equal to this value.
        ccp_alpha (float): Complexity parameter used for Minimal Cost-Complexity Pruning.
        monotonic_cst (array-like or None): Monotonic constraints on features.
        learning_rate (float): Learning rate shrinks the contribution of each regressor.
        loss (str): The loss function to use when updating the weights after each boosting iteration.
            'linear', 'square', 'exponential' are supported.

    Attributes:
        estimators_ (list): List of fitted Rule Tree regressors.
        estimator_weights_ (array): Weights for each estimator in the boosted ensemble.
        estimator_errors_ (array): Regression error for each estimator in the boosted ensemble.
        feature_importances_ (array): The impurity-based feature importances.
    """
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

    def fit(self, X, y, sample_weight=None):
        """
        Build a boosted regressor from the training data.

        Parameters:
            X (array-like of shape (n_samples, n_features)): Training data.
            y (array-like of shape (n_samples,)): Target values.
            sample_weight (array-like of shape (n_samples,), default=None):
                Sample weights. If None, then samples are equally weighted.

        Returns:
            self: Returns self.
        """
        return super().fit(X, y, sample_weight)

    def predict(self, X):
        """
        Predict regression value for X.

        The predicted regression value of an input sample is computed as the weighted median
        prediction of the regressors in the ensemble.

        Parameters:
            X (array-like of shape (n_samples, n_features)): Input samples.

        Returns:
            y (ndarray of shape (n_samples,)): The predicted values.
        """
        return super().predict(X)

    def staged_predict(self, X):
        """
        Return staged predictions for X.

        The predicted regression value of an input sample is computed as the weighted median
        prediction of the regressors in the ensemble.

        This generator method yields the ensemble prediction after each iteration of
        boosting and therefore allows monitoring, such as to determine the
        prediction on test data after each boost.

        Parameters:
            X (array-like of shape (n_samples, n_features)): Input samples.

        Yields:
            y (ndarray of shape (n_samples,)): The predicted values at each iteration.
        """
        return super().staged_predict(X)

    def feature_importance(self):
        """
        Return the feature importances based on the ensemble of trees.

        The feature importances are calculated as the mean and standard deviation
        of accumulation of the importance across all trees in the ensemble.

        Returns:
            feature_importances (ndarray of shape (n_features,)): Feature importances.
        """
        return super().feature_importances_
