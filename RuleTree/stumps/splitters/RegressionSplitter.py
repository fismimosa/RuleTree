import numpy as np
from sklearn.metrics import mean_absolute_error, mean_poisson_deviance, mean_squared_error
from sklearn.tree import DecisionTreeRegressor

from RuleTree.base.RuleTreeBaseSplitter import RuleTreeBaseSplitter
from RuleTree.exceptions import NoSplitFoundWarning
from RuleTree.utils.data_utils import _get_info_gain, get_info_gain
from RuleTree.utils.define import MODEL_TYPE_REG


class RegressionSplitter(RuleTreeBaseSplitter):
    """
    Best split for regression. Numerical features are split by a scikit-learn decision tree of
    depth 1 (same parameters); categorical features are split by testing each of their values.

    Parameters
    ----------
    criterion : {'squared_error', 'absolute_error', 'poisson'} or callable, default='squared_error'
        Impurity measure. A callable takes (y_true, y_pred) and returns the impurity of the
        categorical splits; the numerical ones use the scikit-learn criterion.
    min_samples_leaf, min_weight_fraction_leaf, max_features, random_state, min_impurity_decrease
        Parameters of the scikit-learn decision tree used for the numerical features.
    """

    task = MODEL_TYPE_REG

    def __init__(self, criterion='squared_error', min_samples_leaf=1, min_weight_fraction_leaf=0.0,
                 max_features=None, random_state=None, min_impurity_decrease=0.0):
        self.criterion = criterion
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_features = max_features
        self.random_state = random_state
        self.min_impurity_decrease = min_impurity_decrease

    @classmethod
    def _get_impurity_fun(cls, imp):
        """Impurity function (y_true, y_pred) -> float of a criterion name or callable."""
        if imp == "squared_error":
            return mean_squared_error
        elif imp == "friedman_mse":
            raise NotImplementedError("not implemented")  # TODO: implement in DecisionTreeRegressor?
        elif imp == "absolute_error":
            return mean_absolute_error
        elif imp == "poisson":
            return mean_poisson_deviance
        else:
            return imp

    @classmethod
    def _impurity_fun(cls, impurity_fun, **x):
        """Impurity of y_true predicted as y_pred (0 for no instances)."""
        f = cls._get_impurity_fun(impurity_fun)
        return f(**x) if len(x["y_true"]) > 0 else 0  # TODO: check

    @classmethod
    def _split_impurity(cls, criterion, y, left):
        """Impurity of the node and of the two children, each predicted with its mean."""
        return [
            cls._impurity_fun(criterion, y_true=y, y_pred=np.ones((len(y),)) * np.mean(y)),
            cls._impurity_fun(criterion, y_true=y[left], y_pred=np.ones((len(y[left]),)) * np.mean(y[left])),
            cls._impurity_fun(criterion, y_true=y[~left], y_pred=np.ones((len(y[~left]),)) * np.mean(y[~left])),
        ]

    def fit(self, X, y, numerical, categorical, sample_weight=None, check_input=True):
        self.feature_, self.threshold_, self.is_categorical_, self.impurity_ = None, None, False, None
        best_info_gain = -float('inf')

        if len(numerical) > 0:
            tree = DecisionTreeRegressor(max_depth=1, criterion=self.criterion, min_samples_leaf=self.min_samples_leaf,
                                         min_weight_fraction_leaf=self.min_weight_fraction_leaf,
                                         max_features=self.max_features, random_state=self.random_state,
                                         min_impurity_decrease=self.min_impurity_decrease)
            tree.fit(X[:, numerical], y, sample_weight=sample_weight, check_input=check_input)

            if tree.tree_.node_count == 1:
                raise NoSplitFoundWarning(f"No split found for X {X.shape} and y {np.unique(y)}")

            self.feature_ = numerical[tree.tree_.feature[0]]
            self.threshold_ = tree.tree_.threshold[0]
            best_info_gain = get_info_gain(tree)
            self.impurity_ = self._split_impurity(self.criterion, y, X[:, self.feature_] <= self.threshold_)

        best_info_gain = self._fit_categorical(X, y, categorical, best_info_gain)

        if self.feature_ is None:
            raise NoSplitFoundWarning(f"No split found for X {X.shape} and y {np.unique(y)}")

        self.gain_ = best_info_gain

        return self

    def _fit_categorical(self, X, y, categorical, best_info_gain):
        """Test each value of each categorical feature; keep the split if better than best_info_gain."""
        len_x = len(X)
        for i in categorical:
            for value in np.unique(X[:, i]):
                X_split = X[:, i] == value
                len_left = np.sum(X_split)
                if len_left == 0 or len_left == len_x:  # degenerate split, one side would be empty
                    continue

                impurity = self._split_impurity(self.criterion, y, X_split)
                info_gain = _get_info_gain(*impurity, len_x, len_left, len_x - len_left)

                if info_gain > best_info_gain:
                    best_info_gain = info_gain
                    self.feature_ = i
                    self.threshold_ = value
                    self.is_categorical_ = True
                    self.impurity_ = impurity

        return best_info_gain

    def update_statistics(self, X, y, sample_weight=None):
        self.impurity_ = self._split_impurity(self.criterion, y, self.apply(X) == 1)
