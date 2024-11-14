import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier


from ruletree.base.RuleTreeBaseStump import RuleTreeBaseStump

from ruletree.utils.data_utils import get_info_gain, _get_info_gain, gini, entropy, _my_counts


class DecisionTreeStumpClassifier(DecisionTreeClassifier, RuleTreeBaseStump):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.is_categorical = False
        self.is_oblique = False
        self.is_pivotal = False
        
        self.kwargs = kwargs
        self.unique_val_enum = None
        
        self.threshold_original = None
        self.feature_original = None
        self.coefficients = None
        
        if 'criterion' not in kwargs or kwargs['criterion'] == "gini":
            self.impurity_fun = gini
        elif kwargs['criterion'] == "entropy":
            self.impurity_fun = entropy
        else:
            self.impurity_fun = kwargs['criterion']

    def get_params(self, deep=True):
        return self.kwargs
    
    def feature_analysis(self, X, y):
        dtypes = pd.DataFrame(X).infer_objects().dtypes
        self.numerical = dtypes[dtypes != np.dtype('O')].index
        self.categorical = dtypes[dtypes == np.dtype('O')].index

    def fit(self, X, y, sample_weight=None, check_input=True):
        self.feature_analysis(X, y)
        best_info_gain = -float('inf')

        if len(self.numerical) > 0:
            super().fit(X[:, self.numerical], y, sample_weight=sample_weight, check_input=check_input)
            self.feature_original = self.tree_.feature
            self.threshold_original = self.tree_.threshold
            best_info_gain = get_info_gain(self)
            
        self._fit_cat(X, y, best_info_gain)

        return self

    def _fit_cat(self, X, y, best_info_gain, sample_weight=None):
        if self.max_depth > 1:
            raise Exception("not implemented") # TODO: implement?

        if len(self.categorical) > 0 and best_info_gain != float('inf'):
            len_x = len(X)

            class_weight = None
            if self.class_weight == "balanced":
                class_weight = dict()
                for class_label in np.unique(y):
                    class_weight[class_label] = len_x / (len(self.classes_) * len(y[y == class_label]))


            for i in self.categorical:
                for value in np.unique(X[:, i]):
                    X_split = X[:, i:i+1] == value

                    len_left = np.sum(X_split)

                    if sample_weight is not None:
                        # TODO: check. Sample weights. If None, then samples are equally weighted. Splits that would
                        #  create child nodes with net zero or negative weight are ignored while searching for a split
                        #  in each node. Splits are also ignored if they would result in any single class carrying a
                        #  negative weight in either child node.

                        if _my_counts(y, sample_weight) - (_my_counts(y[X_split[:, 0]], sample_weight)
                                                           + _my_counts(y[~X_split[:, 0]], sample_weight)) <= 0:
                            continue

                        if sum(sample_weight[X_split[:, 0]]) < self.min_weight_fraction_leaf \
                            or sum(sample_weight[~X_split[:, 0]]) < self.min_weight_fraction_leaf:
                            continue

                        if ((_my_counts(y[X_split[:, 0]], sample_weight) <= 0).any()
                                or (_my_counts(y[~X_split[:, 0]], sample_weight) <= 0).any()):
                            continue

                        info_gain = _get_info_gain(self.impurity_fun(y, sample_weight, class_weight),
                                                   self.impurity_fun(y[X_split[:, 0]],
                                                                     sample_weight[X_split[:, 0]],
                                                                     class_weight),
                                                   self.impurity_fun(y[~X_split[:, 0]],
                                                                     sample_weight[~X_split[:, 0]],
                                                                     class_weight),
                                                   len_x,
                                                   len_left,
                                                   len_x-len_left)
                    else:
                        info_gain = _get_info_gain(self.impurity_fun(y, sample_weight, class_weight),
                                                   self.impurity_fun(y[X_split[:, 0]], None, class_weight),
                                                   self.impurity_fun(y[~X_split[:, 0]], None, class_weight),
                                                   len_x,
                                                   len_left,
                                                   len_x - len_left)

                    if info_gain > best_info_gain:
                        best_info_gain = info_gain
                        self.feature_original = [i, -2, -2]
                        self.threshold_original = np.array([value, -2, -2])
                        self.unique_val_enum = np.unique(X[:, i])
                        self.is_categorical = True


    def apply(self, X):
        if not self.is_categorical:
            y_pred = (np.ones(X.shape[0]) * 2)
            X_feature = X[:, self.feature_original[0]]
            y_pred[X_feature <= self.threshold_original[0]] = 1
            
            return y_pred
            
        else:
            y_pred = np.ones(X.shape[0]) * 2
            X_feature = X[:, self.feature_original[0]]
            y_pred[X_feature == self.threshold_original[0]] = 1

            return y_pred
        
    #def apply_fit(self, X):
    #    if not self.is_categorical:
     #       return super().apply(X[:, self.numerical])
     #   else:
     #       y_pred = np.ones(X.shape[0]) * 2
     #       X_feature = X[:, self.feature_original[0]]
     #       y_pred[X_feature == self.threshold_original[0]] = 1

     #      return y_pred
        

    def get_feature(self):
        return self.feature_original[0]

    def get_thresholds(self):
        return self.threshold_original[0]

    def get_is_categorical(self):
        return self.is_categorical
        
