from copy import copy

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

from ruletree.stumps.splitters.ObliqueHouseHolderSplit import ObliqueHouseHolderSplit
from ruletree.stumps.splitters.ObliqueBivariateSplitRegressor import ObliqueBivariateSplitRegressor
from ruletree.base.RuleTreeBaseStump import RuleTreeBaseStump

from ruletree.utils.data_utils import get_info_gain, _get_info_gain, gini, entropy, _my_counts

from ruletree.utils.define import MODEL_TYPE_CLF


class DecisionTreeStumpClassifier(DecisionTreeClassifier, RuleTreeBaseStump):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.is_categorical = False
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

    def fit(self, X, y, sample_weight=None, check_input=True):
        dtypes = pd.DataFrame(X).infer_objects().dtypes
        self.numerical = dtypes[dtypes != np.dtype('O')].index
        self.categorical = dtypes[dtypes == np.dtype('O')].index
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
            return super().apply(X[:, self.numerical])
        else:
            y_pred = np.ones(X.shape[0]) * 2
            X_feature = X[:, self.feature_original[0]]
            y_pred[X_feature == self.threshold_original[0]] = 1

            return y_pred


class MyObliqueDecisionTreeClassifier(DecisionTreeClassifier):
    def __init__(self, oblique_params = {}, oblique_split_type =  'householder', **kwargs):
        super().__init__(**kwargs)
        self.is_categorical = False
        self.kwargs = kwargs
        self.unique_val_enum = None
        self.threshold_original = None
        self.feature_original = None
        self.is_oblique = True
        self.coefficients = None
        
        self.oblique_params = oblique_params
        self.oblique_split_type = oblique_split_type
        self.oblique_split = None
        
        
        if self.oblique_split_type == 'householder':
            self.oblique_split = ObliqueHouseHolderSplit(**oblique_params, **kwargs)
           
        if self.oblique_split_type == 'bivariate':
            self.oblique_split = ObliqueBivariateSplitRegressor(**oblique_params, **kwargs)
        
       
    def fit(self, X, y, sample_weight=None, check_input=True):
        dtypes = pd.DataFrame(X).infer_objects().dtypes
        self.numerical = dtypes[dtypes != np.dtype('O')].index
        self.categorical = dtypes[dtypes == np.dtype('O')].index
        best_info_gain = -float('inf')
        
        if len(self.numerical) > 0:
            self.oblique_split.fit(X[:, self.numerical], y, sample_weight=sample_weight, check_input=check_input)
            self.feature_original = [[self.oblique_split.feats], -2, -2]
            self.coefficients = self.oblique_split.coeff
            self.threshold_original = np.array([self.oblique_split.threshold, -2, -2])
            best_info_gain = get_info_gain(self.oblique_split.oblq_clf)
            
            self.best_info_gain = best_info_gain
       
        return self
    
    def apply(self, X):
        return self.oblique_split.apply(X[:, self.numerical])
