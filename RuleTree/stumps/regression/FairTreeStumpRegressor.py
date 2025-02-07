import copy
import os
import warnings

import math

os.environ["COLUMNS"] = "1"

import numpy as np
import pandas as pd
import numba
from numba import jit
from scipy.stats import wasserstein_distance
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_poisson_deviance
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.tree import DecisionTreeRegressor

from RuleTree.base.RuleTreeBaseStump import RuleTreeBaseStump
from RuleTree.stumps.classification.DecisionTreeStumpClassifier import DecisionTreeStumpClassifier
from RuleTree.stumps.regression import DecisionTreeStumpRegressor

from RuleTree.utils.data_utils import get_info_gain, _get_info_gain

warnings.filterwarnings("ignore")

class FairTreeStumpRegressor(DecisionTreeStumpRegressor):
    def __init__(self,
                 sensible_attribute:int,
                 k_anonymity:int|float,
                 l_diversity:int|float,
                 t_closeness:float,
                 strict:bool|float, #if True -> no unfair split, if False==DTRegressor, if float == penalization weight
                 **kwargs):
        super().__init__(**kwargs)
        self.is_categorical = False
        self.kwargs = kwargs
        self.unique_val_enum = None
        self.threshold_original = None
        self.feature_original = None

        self.sensible_attribute = sensible_attribute
        self.k_anonymity = k_anonymity
        self.l_diversity = l_diversity
        self.t_closeness = t_closeness
        self.strict = strict

        self.kwargs["sensible_attribute"] = sensible_attribute
        self.kwargs["k_anonymity"] = k_anonymity
        self.kwargs["l_diversity"] = l_diversity
        self.kwargs["t_closeness"] = t_closeness
        self.kwargs["strict"] = strict

    def __impurity_fun(self, **x):
        return self.impurity_fun(**x) if len(x["y_true"]) > 0 else 0  # TODO: check

    def get_params(self, deep=True):
        return self.kwargs

    def __admissible_split_fairness(self, X, X_bool):
        return adm_split(X, X_bool, self.sensible_attribute, self.k_anonymity, self.l_diversity,
                         self.t_closeness, self.strict)


    def fit(self, X, y, idx=None, context=None, sample_weight=None, check_input=True):
        if idx is None:
            idx = slice(None)
        X = X[idx]
        y = y[idx]
        len_x = len(idx)

        self.n_node_samples = np.zeros(3)
        self.n_node_samples[0] = len_x

        self.feature_analysis(X, y)
        best_info_gain = -float('inf')

        for i in range(X.shape[1]):
            for value in np.unique(X[:, i]):
                if i in self.categorical:
                    X_split = X[:, i:i + 1] == value
                else:
                    X_split = X[:, i:i + 1] <= value

                if np.sum(X_split)*np.sum(~X_split) == 0:
                    continue

                eval_split = self.__admissible_split_fairness(X, X_split)
                #print(eval_split, balance(X_split, X[self.sensible_attribute]))
                if self.strict and eval_split == -np.inf:
                    continue

                len_left = np.sum(X_split)
                curr_pred = np.ones((len(y),)) * np.mean(y)

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

                    info_gain = 1/(1 + np.exp(-info_gain))

                    if type(eval_split) != bool:
                        info_gain -= info_gain*eval_split

                if info_gain > best_info_gain:
                    best_info_gain = info_gain
                    self.feature_original = [i, -2, -2]
                    self.threshold_original = np.array([value, -2, -2])
                    self.unique_val_enum = np.unique(X[:, i])
                    self.is_categorical = i in self.categorical
                    self.fitted_ = True

        return self


    def apply(self, X, check_input=False):
        if len(self.feature_original) < 3:
            return np.ones(X.shape[0])

        if not self.is_categorical:
            y_pred = np.ones(X.shape[0], dtype=int) * 2
            X_feature = X[:, self.feature_original[0]]
            y_pred[X_feature <= self.threshold_original[0]] = 1

            return y_pred
        else:
            y_pred = np.ones(X.shape[0], dtype=int) * 2
            X_feature = X[:, self.feature_original[0]]
            y_pred[X_feature == self.threshold_original[0]] = 1

            return y_pred

    def get_rule(self, columns_names=None, scaler=None, float_precision=3):
        return DecisionTreeStumpClassifier.get_rule(self,
                                                    columns_names=columns_names,
                                                    scaler=scaler,
                                                    float_precision=float_precision)

    def node_to_dict(self):
        rule = self.get_rule(float_precision=None)

        rule["stump_type"] = self.__class__.__name__
        rule["samples"] = self.tree_.n_node_samples[0]
        rule["impurity"] = self.tree_.impurity[0]

        rule["args"] = {
                           "unique_val_enum": self.unique_val_enum,
                       } | self.kwargs

        rule["split"] = {
            "args": {}
        }

        return rule

    def dict_to_node(self, node_dict, X=None):
        self.feature_original = np.zeros(3)
        self.threshold_original = np.zeros(3)

        self.feature_original[0] = node_dict["feature_original"]
        self.threshold_original[0] = node_dict["threshold"]
        self.is_categorical = node_dict["is_categorical"]

        args = copy.deepcopy(node_dict["args"])
        self.unique_val_enum = args.pop("unique_val_enum")
        self.kwargs = args

        self.__set_impurity_fun(args["criterion"])

