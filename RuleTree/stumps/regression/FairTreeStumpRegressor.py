import copy
import os
import warnings

import math
from concurrent.futures.process import ProcessPoolExecutor

import psutil
from statsmodels.graphics.tukeyplot import results

from RuleTree.utils.fairness_metrics import balance_metric, max_fairness_cost, privacy_metric, privacy_metric_all

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
    """
    Fair Decision Tree Stump Regressor that enforces fairness constraints during training.
    
    This class extends DecisionTreeStumpRegressor by introducing fairness constraints during the
    learning process. It can enforce balancing, maximum fairness cost, or privacy-related
    constraints (k-anonymity, l-diversity, t-closeness) with respect to a sensitive attribute.
    
    Parameters
    ----------
    penalty : str, default=None
        Type of fairness constraint to apply. Options are:
        - "balance": Enforce statistical parity (equal representation)
        - "mfc": Maximum Fairness Cost according to ideal distribution
        - "privacy": Enforce privacy constraints (k-anonymity, l-diversity, t-closeness)
        - None: No fairness constraints (equivalent to standard DecisionTreeStumpRegressor)
        
    sensible_attribute : int, default=-1
        The index of the column in X containing the sensitive/protected attribute.
        
    penalization_weight : float, default=0.3
        Weight of the fairness penalty term. Higher values enforce stronger fairness.
        
    ideal_distribution : dict, default=None
        Required when penalty="mfc". Dictionary specifying the ideal distribution for different groups.
        
    k_anonymity : int or float, default=2
        Minimum group size for k-anonymity (used when penalty="privacy").
        
    l_diversity : int or float, default=2
        Minimum distinct values required for l-diversity (used when penalty="privacy").
        
    t_closeness : float, default=0.2
        Maximum earth mover's distance allowed for t-closeness (used when penalty="privacy").
        
    strict : bool, default=True
        If True, strictly enforces privacy constraints; if False, uses penalization instead.
        
    use_t : bool, default=True
        Whether to use t-closeness in privacy constraint calculations.
        
    n_jobs : int, default=psutil.cpu_count(logical=False)
        Number of CPU cores to use for parallel processing.
        
    **kwargs
        Additional parameters passed to the parent DecisionTreeStumpRegressor class.
    """
    def __init__(self,
                 penalty:str=None,
                 sensible_attribute:int=-1,
                 penalization_weight: float = 0.3,

                 ideal_distribution:dict=None,

                 k_anonymity:int|float=2,
                 l_diversity:int|float=2,
                 t_closeness:float=.2,
                 strict:bool=True, #if True -> no unfair split, if False==DTRegressor, if float == penalization weight
                 use_t: bool = True,

                 n_jobs:int=psutil.cpu_count(logical=False),
                 #n_jobs:int=1,

                 **kwargs):
        super().__init__(**kwargs)
        self.is_categorical = False
        self.kwargs = kwargs
        self.unique_val_enum = None
        self.threshold_original = None
        self.feature_original = None

        self.penalty = penalty
        self.sensible_attribute = sensible_attribute
        self.penalization_weight = penalization_weight

        self.ideal_distribution = ideal_distribution

        self.k_anonymity = k_anonymity
        self.l_diversity = l_diversity
        self.t_closeness = t_closeness
        self.strict = strict
        self.use_t = use_t

        self.n_jobs = n_jobs

        self.kwargs["penalty"] = penalty
        self.kwargs["sensible_attribute"] = sensible_attribute
        self.kwargs["penalization_weight"] = penalization_weight
        self.kwargs["ideal_distribution"] = ideal_distribution
        self.kwargs["k_anonymity"] = k_anonymity
        self.kwargs["l_diversity"] = l_diversity
        self.kwargs["t_closeness"] = t_closeness
        self.kwargs["strict"] = strict
        self.kwargs["use_t"] = use_t
        self.kwargs["n_jobs"] = n_jobs

    def __check_fairness_hyper(self):
        """
        Validates the fairness-related hyperparameters based on the chosen penalty type.
        
        Raises
        ------
        AssertionError
            If the fairness hyperparameters are not properly set for the chosen penalty.
        """
        if self.penalty is not None:
            assert self.sensible_attribute is not None
            assert self.penalization_weight is not None
            assert self.penalty in ["balance", "mfc", "privacy"]

            if self.penalty == "mfc":
                assert self.ideal_distribution is not None and isinstance(self.ideal_distribution, dict)
            if self.penalty == "privacy":
                assert None not in [self.k_anonymity, self.l_diversity, self.t_closeness, self.strict]

    def get_params(self, deep=True):
        """
        Get parameters for this estimator.
        
        Parameters
        ----------
        deep : bool, default=True
            If True, will return the parameters for this estimator and
            contained subobjects that are estimators.
            
        Returns
        -------
        params : dict
            Parameter names mapped to their values.
        """
        return self.kwargs

    def fit(self, X, y, idx=None, context=None, sample_weight=None, check_input=True):
        """
        Build a fair decision tree stump regressor from the training set (X, y).
        
        The method searches for the best split that optimizes both prediction performance
        and the fairness constraint specified by the penalty parameter.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The training input samples.
            
        y : array-like of shape (n_samples,)
            The target values (real numbers).
            
        idx : slice or array-like, default=None
            Indices of samples to use for training. If None, all samples are used.
            
        context : Not used, present for API consistency by convention.
        
        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights. Currently not used.
            
        check_input : bool, default=True
            Allow to bypass several input checking. Don't use this parameter
            unless you know what you do.
            
        Returns
        -------
        self : object
            Fitted estimator.
        """
        if idx is None:
            idx = slice(None)
        X = X[idx]
        y = y[idx]
        len_x = len(idx)

        self.n_node_samples = np.zeros(3)
        self.n_node_samples[0] = len_x

        self.feature_analysis(X, y)
        best_info_gain = -float('inf')
        self.feature_original = [-2]

        processes = []

        with ProcessPoolExecutor(max_workers=self.n_jobs) as executor:
            for i in range(X.shape[1]):
                if i == self.sensible_attribute:
                    continue
                sorted_arr = np.hstack([X, y.reshape(-1, 1)])
                sorted_arr = sorted_arr[:, [i, -1]][np.lexsort((X[:, -1], X[:, i]))]
                unique_sorted_array = np.unique(sorted_arr, axis=0)
                if i not in self.categorical:
                    values = [
                        (np.mean(unique_sorted_array[j - 1:j + 1, 0]))
                        for j in range(1, len(unique_sorted_array))
                        if unique_sorted_array[j - 1, 1] != unique_sorted_array[j, 1]
                           and  unique_sorted_array[j - 1, 0] != unique_sorted_array[j, 0]
                    ] # elimino tutti i threshold "duplicati" (np.unique(threshold)) + i threshold dove il primo record
                      # a destra ha la stessa "classe" (output regressore) del record a sinistra
                      # possiamo farlo anche nel caso del protected attribute? non penso

                    n_values_per_core = max(10, len(values) // self.n_jobs)


                    for start_idx in range(0, len(values), n_values_per_core):
                        if self.n_jobs == 1:
                            processes.append(_inner_loop_best_split(X, y, i,
                                                                    values[start_idx:start_idx+n_values_per_core],
                                                                    self.categorical, self.impurity_fun, self.penalty,
                                                                    self.sensible_attribute, self.penalization_weight,
                                                                    self.ideal_distribution, self.k_anonymity,
                                                                    self.l_diversity, self.t_closeness, self.strict,
                                                                    self.use_t)
                                             )
                        else:
                            processes.append(
                                executor.submit(_inner_loop_best_split, X, y, i,
                                                values[start_idx:start_idx+n_values_per_core], self.categorical,
                                                self.impurity_fun, self.penalty, self.sensible_attribute,
                                                self.penalization_weight, self.ideal_distribution, self.k_anonymity,
                                                self.l_diversity, self.t_closeness, self.strict, self.use_t)
                            )

            if self.n_jobs != 1:
                processes = [x.result() for x in processes]

        for info_gain, threshold, col_idx in processes:
            if info_gain > best_info_gain:
                best_info_gain = info_gain
                self.feature_original = [col_idx, -2, -2]
                self.threshold_original = np.array([threshold, -2, -2])
                self.unique_val_enum = np.unique(X[:, col_idx])
                self.is_categorical = col_idx in self.categorical
                self.fitted_ = True

        return self


    def apply(self, X, check_input=False):
        """
        Apply the decision tree stump to X.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.
            
        check_input : bool, default=False
            Allow to bypass several input checking. Don't use this parameter
            unless you know what you do.
            
        Returns
        -------
        X_leaves : array-like of shape (n_samples,)
            Leaf node indices (1 for left, 2 for right) for each sample.
        """
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
        """
        Get a human-readable rule representation of the tree stump.
        
        Parameters
        ----------
        columns_names : array-like, default=None
            Names of the features. If None, generic feature names are used.
            
        scaler : object, default=None
            If not None, the scaler is used to transform the threshold values
            back to the original scale.
            
        float_precision : int, default=3
            Number of decimal places to use when rounding threshold values.
            
        Returns
        -------
        rule : dict
            Dictionary representation of the learned rule.
        """
        return DecisionTreeStumpClassifier.get_rule(self,
                                                    columns_names=columns_names,
                                                    scaler=scaler,
                                                    float_precision=float_precision)

    def node_to_dict(self):
        """
        Convert the tree stump node to a dictionary representation.
        
        This is useful for serialization and model storage.
        
        Returns
        -------
        rule : dict
            Dictionary representation of the node, including split information
            and fairness parameters.
        """
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
        """
        Initialize the tree stump node from a dictionary representation.
        
        This is useful for deserialization and model loading.
        
        Parameters
        ----------
        node_dict : dict
            Dictionary representation of the node.
            
        X : array-like, default=None
            The training input samples. Not used in this implementation.
            
        Returns
        -------
        self : object
            The initialized tree stump regressor.
        """
        self.feature_original = np.zeros(3)
        self.threshold_original = np.zeros(3)

        self.feature_original[0] = node_dict["feature_original"]
        self.threshold_original[0] = node_dict["threshold"]
        self.is_categorical = node_dict["is_categorical"]

        args = copy.deepcopy(node_dict["args"])
        self.unique_val_enum = args.pop("unique_val_enum")
        self.kwargs = args

        self.__set_impurity_fun(args["criterion"])


def _check_balance(labels:np.ndarray, prot_attr:np.ndarray):
    """
    Check if a split satisfies balance (statistical parity) fairness constraint.
    
    Parameters
    ----------
    labels : np.ndarray
        Binary array indicating sample assignments (True for left node, False for right)
    prot_attr : np.ndarray
        Protected attribute values for each sample
        
    Returns
    -------
    tuple
        (is_fair, fairness_score) where is_fair is always True and
        fairness_score is the complement of the balance metric (higher is better)
    """
    return True, 1-balance_metric(labels, prot_attr)

def _check_max_fairness_cost(labels:np.ndarray, prot_attr:np.ndarray, ideal_dist:dict):
    """
    Check if a split satisfies the maximum fairness cost constraint.
    
    Parameters
    ----------
    labels : np.ndarray
        Binary array indicating sample assignments (True for left node, False for right)
    prot_attr : np.ndarray
        Protected attribute values for each sample
    ideal_dist : dict
        Dictionary specifying the ideal distribution for different groups
        
    Returns
    -------
    tuple
        (is_fair, fairness_score) where is_fair is always True and
        fairness_score is the maximum fairness cost (lower is better)
    """
    return True, max_fairness_cost(labels, prot_attr, ideal_dist)

def _check_privacy(X, X_bool, sensible_attribute, k_anonymity, l_diversity, t_closeness, strict, categorical, use_t):
    """
    Check if a split satisfies privacy constraints (k-anonymity, l-diversity, t-closeness).
    
    Parameters
    ----------
    X : np.ndarray
        Input feature matrix
    X_bool : np.ndarray
        Binary array indicating sample assignments (True for left node, False for right)
    sensible_attribute : int
        Index of the sensitive attribute column in X
    k_anonymity : int or float
        Minimum group size required for k-anonymity
    l_diversity : int or float
        Minimum distinct values required for l-diversity
    t_closeness : float
        Maximum earth mover's distance allowed for t-closeness
    strict : bool
        If True, strictly enforce privacy constraints
    categorical : set
        Set of indices of categorical features
    use_t : bool
        Whether to use t-closeness in constraint calculations
        
    Returns
    -------
    tuple
        (is_fair, fairness_score) where is_fair indicates if privacy constraints are satisfied
        and fairness_score is the normalized maximum privacy metric
    """
    can_split, k, l, t = privacy_metric(X, X_bool, sensible_attribute, k_anonymity, l_diversity, t_closeness, strict,
                                        categorical, use_t)
    if not use_t:
        t = t_closeness

    if not can_split:
        return False, np.nan
    return can_split, np.max(privacy_metric_all(k, k_anonymity, l, l_diversity, t, t_closeness))#/3

def _inner_loop_best_split(X:np.ndarray, y:np.ndarray, col_idx:int, thresholds:list, categorical:set, impurity_fun,
                           penalty:str, sensible_attribute: int, penalization_weight: float, ideal_distribution: dict,
                           k_anonymity, l_diversity, t_closeness, strict, use_t):
    """
    Find the best split for a feature column considering fairness constraints.
    
    Parameters
    ----------
    X : np.ndarray
        Input feature matrix
    y : np.ndarray
        Target values
    col_idx : int
        Index of the feature column to split on
    thresholds : list
        List of candidate threshold values to evaluate
    categorical : set
        Set of indices of categorical features
    impurity_fun : callable
        Function to calculate impurity (e.g., mse, mae)
    penalty : str
        Type of fairness constraint to apply ("balance", "mfc", or "privacy")
    sensible_attribute : int
        Index of the sensitive attribute column
    penalization_weight : float
        Weight for the fairness penalty term
    ideal_distribution : dict
        Ideal distribution when using "mfc" penalty
    k_anonymity, l_diversity, t_closeness : Various types
        Privacy constraint parameters when using "privacy" penalty
    strict : bool
        Whether to strictly enforce privacy constraints
    use_t : bool
        Whether to use t-closeness in privacy constraints
        
    Returns
    -------
    tuple
        (best_info_gain, best_threshold, col_idx) representing the best split found
    """
    len_x = len(X)

    best_info_gain = -np.inf
    best_threshold = -1
    for value in thresholds:
        if col_idx in categorical:
            X_split = X[:, col_idx:col_idx + 1] == value
        else:
            X_split = X[:, col_idx:col_idx + 1] <= value

        if np.sum(X_split) * np.sum(~X_split) == 0:
            continue

        if penalty == "balance":
            ok_to_split, penalty_value = _check_balance(X_split[:, 0], X[:, sensible_attribute])
        elif penalty == "mfc":
             ok_to_split, penalty_value = _check_max_fairness_cost(X_split[:, 0], X[:, sensible_attribute], ideal_distribution)
        else:
            ok_to_split, penalty_value = _check_privacy(X, X_split, sensible_attribute, k_anonymity, l_diversity,
                                                        t_closeness, strict, categorical, use_t)
        if not ok_to_split:
            continue

        len_left = np.sum(X_split)
        curr_pred = np.ones((len(y),)) * np.mean(y)

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            l_pred = np.ones((len(y[X_split[:, 0]]),)) * np.mean(y[X_split[:, 0]])
            r_pred = np.ones((len(y[~X_split[:, 0]]),)) * np.mean(y[~X_split[:, 0]])

            info_gain = _get_info_gain(FairTreeStumpRegressor._impurity_fun(impurity_fun, y_true=y, y_pred=curr_pred),
                                       FairTreeStumpRegressor._impurity_fun(impurity_fun, y_true=y[X_split[:, 0]], y_pred=l_pred),
                                       FairTreeStumpRegressor._impurity_fun(impurity_fun, y_true=y[~X_split[:, 0]], y_pred=r_pred),
                                       len_x,
                                       len_left,
                                       len_x - len_left)

            info_gain = 1 / (1 + np.exp(-info_gain))

            info_gain -= info_gain * (penalty_value*penalization_weight)

        if info_gain > best_info_gain:
            best_info_gain = info_gain
            best_threshold = value

    return best_info_gain, best_threshold, col_idx
