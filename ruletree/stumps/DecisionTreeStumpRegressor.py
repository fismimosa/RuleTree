import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_poisson_deviance
from sklearn.tree import DecisionTreeRegressor

from ruletree.stumps.ObliqueHouseHolderSplit import ObliqueHouseHolderSplit
from ruletree.stumps.ObliqueBivariateSplitRegressor import ObliqueBivariateSplit

from ruletree.utils.data_utils import get_info_gain, _get_info_gain

from ruletree.utils.define import MODEL_TYPE_REG


class MyDecisionTreeRegressor(DecisionTreeRegressor, RuleTreeBaseStump):
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
        if self.tree_.max_depth > 1:
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


    def apply(self, X):
        if not self.is_categorical:
            return super().apply(X[:, self.numerical])
        else:
            y_pred = np.ones(X.shape[0], dtype=int) * 2
            X_feature = X[:, self.feature_original[0]]
            y_pred[X_feature == self.threshold_original[0]] = 1

            return y_pred
        
        
class MyObliqueDecisionTreeRegressor(DecisionTreeRegressor):
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

        
        if self.oblique_split_type == 'householder':
            self.oblique_split = ObliqueHouseHolderSplit(**oblique_params, **kwargs, model_type = MODEL_TYPE_REG)
           
        if self.oblique_split_type == 'bivariate':
            self.oblique_split = ObliqueBivariateSplit(**oblique_params, **kwargs, model_type  = MODEL_TYPE_REG)
        
    def __impurity_fun(self, **x):
        return self.impurity_fun(**x) if len(x["y_true"]) > 0 else 0 # TODO: chec
    
    def fit(self, X, y):
        dtypes = pd.DataFrame(X).infer_objects().dtypes
        self.numerical = dtypes[dtypes != np.dtype('O')].index
        self.categorical = dtypes[dtypes == np.dtype('O')].index
        best_info_gain = -float('inf')
        
        if len(self.numerical) > 0:
            self.oblique_split.fit(X[:, self.numerical], y)
            self.feature_original = [[self.oblique_split.feats], -2, -2]
            self.coefficients = self.oblique_split.coeff
            self.threshold_original = np.array([self.oblique_split.threshold, -2, -2])
            best_info_gain = get_info_gain(self.oblique_split.oblq_clf)
            
            self.best_info_gain = best_info_gain
       
        return self
    
    def apply(self, X):
        return self.oblique_split.apply(X[:, self.numerical])

