import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_poisson_deviance
from sklearn.tree import DecisionTreeRegressor

from ruletree.base.RuleTreeBaseStump import RuleTreeBaseStump
from ruletree.stumps.classification.DecisionTreeStumpClassifier import DecisionTreeStumpClassifier

from ruletree.utils.data_utils import get_info_gain, _get_info_gain


class DecisionTreeStumpRegressor(DecisionTreeRegressor, RuleTreeBaseStump):
    def get_rule(self, columns_names=None, scaler=None, float_precision=3):
        DecisionTreeStumpClassifier.get_rule(self,
                                             columns_names=columns_names,
                                             scaler=scaler,
                                             float_precision=float_precision)

    def node_to_dict(self, col_names):
        pass

    def export_graphviz(self, graph=None, columns_names=None, scaler=None, float_precision=3):
        pass

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.is_categorical = False
        self.kwargs = kwargs
        self.unique_val_enum = None
        self.threshold_original = None
        self.feature_original = None

        if kwargs['criterion'] == "squared_error":
            self.impurity_fun = mean_squared_error
        elif kwargs['criterion'] == "friedman_mse":
            raise Exception("not implemented") # TODO: implement
        elif kwargs['criterion'] == "absolute_error":
            self.impurity_fun = mean_absolute_error
        elif kwargs['criterion'] == "poisson":
            self.impurity_fun = mean_poisson_deviance
        else:
            self.impurity_fun = kwargs['criterion']


    def __impurity_fun(self, **x):
        return self.impurity_fun(**x) if len(x["y_true"]) > 0 else 0 # TODO: check

    def get_params(self, deep=True):
        return self.kwargs

    def fit(self, X, y, **kwargs):
        dtypes = pd.DataFrame(X).infer_objects().dtypes
        self.numerical = dtypes[dtypes != np.dtype('O')].index
        self.categorical = dtypes[dtypes == np.dtype('O')].index
        best_info_gain = -float('inf')

        if len(self.numerical) > 0:
            super().fit(X[:, self.numerical], y, **kwargs)
            self.feature_original = self.tree_.feature
            self.threshold_original = self.tree_.threshold
            best_info_gain = get_info_gain(self)

        self._fit_cat(X, y, best_info_gain)



        return self

    def _fit_cat(self, X, y, best_info_gain):
        if self.max_depth > 1:
            raise Exception("not implemented") # TODO: implement?

        len_x = len(X)

        if len(self.categorical) > 0 and best_info_gain != float('inf'):
            for i in self.categorical:
                for value in np.unique(X[:, i]):
                    X_split = X[:, i:i+1] == value
                    len_left = np.sum(X_split)
                    curr_pred = np.ones((len(y), ))*np.mean(y)
                    with warnings.catch_warnings():
                        warnings.simplefilter('ignore')
                        l_pred = np.ones((len(y[X_split[:, 0]]),)) * np.mean(y[X_split[:, 0]])
                        r_pred = np.ones((len(y[~X_split[:, 0]]),)) * np.mean(y[~X_split[:, 0]])

                        info_gain = _get_info_gain(self.__impurity_fun(y_true=y, y_pred=curr_pred),
                                                   self.__impurity_fun(y_true=y[X_split[:, 0]], y_pred=l_pred),
                                                   self.__impurity_fun(y_true=y[~X_split[:, 0]], y_pred=r_pred),
                                                   len_x,
                                                   len_left,
                                                   len_x - len_left)

                    if info_gain > best_info_gain:
                        best_info_gain = info_gain
                        self.feature_original = [i, -2, -2]
                        self.threshold_original = np.array([value, -2, -2])
                        self.unique_val_enum = np.unique(X[:, i])
                        self.is_categorical = True


    def apply(self, X, check_input=False):
        if not self.is_categorical:
            return super().apply(X[:, self.numerical])
        else:
            y_pred = np.ones(X.shape[0], dtype=int) * 2
            X_feature = X[:, self.feature_original[0]]
            y_pred[X_feature == self.threshold_original[0]] = 1

            return y_pred

