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

def balance(labels:np.ndarray, prot_attr=np.ndarray):
    res = []

    for pr_attr in np.unique(prot_attr):
        r = np.sum(prot_attr == pr_attr)
        for cl_id in np.unique(labels):
            ra = np.sum((labels == cl_id) & (prot_attr == pr_attr))
            rab= r/ra if ra != 0 else 0
            rab_1 = 1/rab if rab != 0 else 1
            res.append(min(rab, rab_1))


    return min(res)

@jit(nopython=True)
def unique_with_counts(arr):
    arr_sorted = np.sort(arr)
    unique_vals = []
    counts = []

    prev = arr_sorted[0]
    count = 1

    for i in range(1, len(arr_sorted)):
        if arr_sorted[i] == prev:
            count += 1
        else:
            unique_vals.append(prev)
            counts.append(count)
            prev = arr_sorted[i]
            count = 1

    unique_vals.append(prev)
    counts.append(count)

    return np.array(unique_vals), np.array(counts)


@jit
def _earth_mover_distance(a:np.ndarray, b_idx:np.ndarray, is_categorical:bool):
    b = a[b_idx]
    if is_categorical:
        raise NotImplementedError("Earth mover distance is not implemented for categorical data")
    else:
        with numba.objmode(a_binned='float64[:]', b_binned='float64[:]'):
            bins = KBinsDiscretizer(n_bins=max(2, len(a)), strategy='uniform')
            a_binned = np.asarray(np.sum(bins.fit_transform(a.reshape(-1, 1)), axis=0))[0]/len(a)
            b_binned = np.asarray(np.sum(bins.transform(b.reshape(-1, 1)), axis=0))[0]/len(b)

        bin_list = np.array([i for i in range(a_binned.shape[0])]).astype(np.float64)
        with numba.objmode(d='float64'):
            d=wasserstein_distance(bin_list, bin_list, a_binned, b_binned) / (len(a)-1)
        return d

@jit
def adm_split(X, X_bool, sensible_attribute, k_anonymity, l_diversity, t_closeness, strict):
    X_bool = X_bool.copy().reshape(-1)

    k_left = min(unique_with_counts(X[X_bool, sensible_attribute])[1])
    l_left = len(np.unique(X[X_bool, sensible_attribute])) - 1
    k_right = min(unique_with_counts(X[~X_bool, sensible_attribute])[1])
    l_right = len(np.unique(X[~X_bool, sensible_attribute])) - 1

    if isinstance(k_anonymity, float):
        k_left /= np.sum(X_bool)
        k_right /= np.sum(~X_bool)

    t_left = np.inf
    t_right = np.inf

    # print(round(k_left, 3), round(k_right, 3), l_left, l_right, t_left, t_right, sep='\t', end='')

    if isinstance(strict, bool) and strict:
        if min(k_left, k_right) < k_anonymity \
                or min(l_left, l_right) < l_diversity:
            # print()
            return -np.inf

    for prot_attr_value in np.unique(X[X_bool, sensible_attribute]):
        for feat_idx in range(X.shape[1]):
            if feat_idx == sensible_attribute:
                continue

            t_curr = _earth_mover_distance(X[X_bool, feat_idx],
                                           X[X_bool, sensible_attribute] == prot_attr_value,
                                           False)
            if t_curr < t_left:
                t_left = t_curr

    for prot_attr_value in np.unique(X[X_bool, sensible_attribute]):
        for feat_idx in range(X.shape[1]):
            if feat_idx == sensible_attribute:
                continue

            t_curr = _earth_mover_distance(X[X_bool, feat_idx],
                                           X[X_bool, sensible_attribute] == prot_attr_value,
                                           False)
            if t_curr < t_right:
                t_right = t_curr

    # print('\t', t_left, '\t', t_right)
    if isinstance(strict, bool) and strict:
        if max(t_left, t_right) > t_closeness:
            return -np.inf

    if not isinstance(strict, bool):
        k = min(k_left, k_right)
        l = min(l_left, l_right)
        t = max(t_left, t_right)

        return ((
                        (k_anonymity - min(k, k_anonymity)) / k_anonymity  # k-anonimity tra 0 e 1, 0=ok
                        + (l_diversity - min(l, l_diversity)) / l_diversity
                        + (max(t, t_closeness) - t_closeness) / t_closeness
                ) / 3) * strict

    return .0

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

