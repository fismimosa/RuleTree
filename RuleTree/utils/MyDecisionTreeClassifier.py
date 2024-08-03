import numpy as np
import pandas as pd
import sklearn
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.tree import DecisionTreeClassifier

from RuleTree.utils.data_utils import get_info_gain


class MyDecisionTreeClassifier(DecisionTreeClassifier):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.is_categorical = False
        self.kwargs = kwargs
        self.unique_val_enum = None
        self.threshold_original = None
        self.feature_original = None

    def fit(self, X, y):
        dtypes = pd.DataFrame(X).infer_objects().dtypes
        self.numerical = dtypes[dtypes != np.dtype('O')].index
        self.categorical = dtypes[dtypes == np.dtype('O')].index

        super().fit(X[:, self.numerical], y)
        self.feature_original = self.tree_.feature
        self.threshold_original = self.tree_.threshold

        best_info_gain = get_info_gain(self)

        if len(self.categorical) > 0 and best_info_gain != float('inf'):
            clf = DecisionTreeClassifier(**self.kwargs)
            X_onehot = np.zeros((X.shape[0], 1))
            for i in self.categorical:
                for value in np.unique(X[:, i]):
                    X_onehot[:] = 0
                    X_onehot[X[:, i] == value] = 1
                    clf.fit(X_onehot, y)
                    info_gain = get_info_gain(clf)

                    if info_gain > best_info_gain:
                        best_info_gain = info_gain
                        self.feature_original = [i, -2, -2]
                        self.threshold_original = np.array([value, -2, -2])
                        self.unique_val_enum = np.unique(X[:, i])
                        self.is_categorical = True

        return self

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

