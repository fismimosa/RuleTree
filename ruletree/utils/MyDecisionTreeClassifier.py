import numpy as np
import pandas as pd
import sklearn
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.tree import DecisionTreeClassifier

from ruletree.utils.data_utils import get_info_gain, _get_info_gain, gini, entropy


class MyDecisionTreeClassifier(DecisionTreeClassifier):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.is_categorical = False
        self.kwargs = kwargs
        self.unique_val_enum = None
        self.threshold_original = None
        self.feature_original = None

        if kwargs['criterion'] == "gini":
            self.impurity_fun = gini
        else:
            self.impurity_fun = entropy


    def fit(self, X, y):
        dtypes = pd.DataFrame(X).infer_objects().dtypes
        self.numerical = dtypes[dtypes != np.dtype('O')].index
        self.categorical = dtypes[dtypes == np.dtype('O')].index

        super().fit(X[:, self.numerical], y)
        self.feature_original = self.tree_.feature
        self.threshold_original = self.tree_.threshold

        best_info_gain = get_info_gain(self)

        self._fit_cat(X, y, best_info_gain)

        return self

    def _fit_cat(self, X, y, best_info_gain):
        len_x = len(X)

        if len(self.categorical) > 0 and best_info_gain != float('inf'):
            for i in self.categorical:
                for value in np.unique(X[:, i]):
                    X_split = X[:, i:i+1] == value
                    len_left = np.sum(X_split)
                    info_gain = _get_info_gain(self.impurity_fun(y),
                                               self.impurity_fun(y[X_split[:, 0]]),
                                               self.impurity_fun(y[~X_split[:, 0]]),
                                               len_x,
                                               len_left,
                                               len_x-len_left)

                    if info_gain > best_info_gain:
                        best_info_gain = info_gain
                        self.feature_original = [i, -2, -2]
                        self.threshold_original = np.array([value, -2, -2])
                        self.unique_val_enum = np.unique(X[:, i])
                        self.is_categorical = True


    def apply(self, X):
        if X.shape[0] == 0:
            print("HERE")
        if not self.is_categorical:
            return super().apply(X[:, self.numerical])
        else:
            y_pred = np.ones(X.shape[0]) * 2
            X_feature = X[:, self.feature_original[0]]
            y_pred[X_feature == self.threshold_original[0]] = 1

            return y_pred

