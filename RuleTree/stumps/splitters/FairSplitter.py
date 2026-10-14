from concurrent.futures.process import ProcessPoolExecutor
from typing import Union

import numpy as np
import psutil

from RuleTree.base.RuleTreeBaseSplitter import RuleTreeBaseSplitter
from RuleTree.exceptions import NoSplitFoundWarning
from RuleTree.stumps.splitters.RegressionSplitter import RegressionSplitter
from RuleTree.utils.data_utils import _get_info_gain
from RuleTree.utils.define import MODEL_TYPE_REG
from RuleTree.utils.fairness_metrics import balance_metric, max_fairness_cost, privacy_metric, privacy_metric_all


class FairSplitter(RuleTreeBaseSplitter):
    """
    Best regression split under a fairness constraint on a sensitive attribute.

    Parameters
    ----------
    penalty : str, default=None
        Type of fairness constraint to apply:
        - "balance": Enforce statistical parity (equal representation)
        - "mfc": Maximum Fairness Cost according to ideal distribution
        - "privacy": Enforce privacy constraints (k-anonymity, l-diversity, t-closeness)
    sensible_attribute : int, default=-1
        The index of the column containing the sensitive/protected attribute.
    penalization_weight : float, default=0.3
        Weight of the fairness penalty term. Higher values enforce stronger fairness.
    ideal_distribution : dict, default=None
        Required when penalty="mfc". Ideal distribution for the different groups.
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
        Number of processes used to evaluate the thresholds.
    criterion : str or callable, default='squared_error'
        Impurity measure (see RegressionSplitter).
    """

    task = MODEL_TYPE_REG

    def __init__(self,
                 penalty: str = None,
                 sensible_attribute: int = -1,
                 penalization_weight: float = 0.3,
                 ideal_distribution: dict = None,
                 k_anonymity: Union[int, float] = 2,
                 l_diversity: Union[int, float] = 2,
                 t_closeness: float = .2,
                 strict: bool = True,
                 use_t: bool = True,
                 n_jobs: int = psutil.cpu_count(logical=False),
                 criterion='squared_error'):
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
        self.criterion = criterion

    def fit(self, X, y, numerical, categorical, sample_weight=None, check_input=True):
        best_info_gain = -float('inf')
        self.feature_, self.threshold_, self.is_categorical_, self.impurity_ = None, None, False, None

        processes = []

        with ProcessPoolExecutor(max_workers=self.n_jobs) as executor:
            for i in range(X.shape[1]):
                if i == self.sensible_attribute:
                    continue
                sorted_arr = np.hstack([X, y.reshape(-1, 1)])
                sorted_arr = sorted_arr[:, [i, -1]][np.lexsort((X[:, -1], X[:, i]))]
                unique_sorted_array = np.unique(sorted_arr, axis=0)
                if i not in categorical:
                    values = [
                        (np.mean(unique_sorted_array[j - 1:j + 1, 0]))
                        for j in range(1, len(unique_sorted_array))
                        if unique_sorted_array[j - 1, 1] != unique_sorted_array[j, 1]
                           and unique_sorted_array[j - 1, 0] != unique_sorted_array[j, 0]
                    ]

                    n_values_per_core = max(10, len(values) // self.n_jobs)

                    for start_idx in range(0, len(values), n_values_per_core):
                        args = (X, y, i, values[start_idx:start_idx + n_values_per_core], categorical,
                                self.criterion, self.penalty, self.sensible_attribute, self.penalization_weight,
                                self.ideal_distribution, self.k_anonymity, self.l_diversity, self.t_closeness,
                                self.strict, self.use_t)
                        if self.n_jobs == 1:
                            processes.append(_inner_loop_best_split(*args))
                        else:
                            processes.append(executor.submit(_inner_loop_best_split, *args))

            if self.n_jobs != 1:
                processes = [x.result() for x in processes]

        for info_gain, threshold, col_idx in processes:
            if info_gain > best_info_gain:
                best_info_gain = info_gain
                self.feature_ = col_idx
                self.threshold_ = threshold
                self.is_categorical_ = col_idx in categorical

        if self.feature_ is None:
            raise NoSplitFoundWarning("No fair split found")

        self.gain_ = best_info_gain

        return self

    def update_statistics(self, X, y, sample_weight=None):
        self.impurity_ = RegressionSplitter._split_impurity(self.criterion, y, self.apply(X) == 1)


def _check_balance(labels: np.ndarray, prot_attr: np.ndarray):
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
    return True, 1 - balance_metric(labels, prot_attr)


def _check_max_fairness_cost(labels: np.ndarray, prot_attr: np.ndarray, ideal_dist: dict):
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
    return can_split, np.max(privacy_metric_all(k, k_anonymity, l, l_diversity, t, t_closeness))  # /3


def _inner_loop_best_split(X: np.ndarray, y: np.ndarray, col_idx: int, thresholds: list, categorical: set, impurity_fun,
                           penalty: str, sensible_attribute: int, penalization_weight: float, ideal_distribution: dict,
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
            ok_to_split, penalty_value = _check_max_fairness_cost(X_split[:, 0], X[:, sensible_attribute],
                                                                  ideal_distribution)
        else:
            ok_to_split, penalty_value = _check_privacy(X, X_split, sensible_attribute, k_anonymity, l_diversity,
                                                        t_closeness, strict, categorical, use_t)
        if not ok_to_split:
            continue

        len_left = np.sum(X_split)
        curr_pred = np.ones((len(y),)) * np.mean(y)

        l_pred = np.ones((len(y[X_split[:, 0]]),)) * np.mean(y[X_split[:, 0]])
        r_pred = np.ones((len(y[~X_split[:, 0]]),)) * np.mean(y[~X_split[:, 0]])

        info_gain = _get_info_gain(RegressionSplitter._impurity_fun(impurity_fun, y_true=y, y_pred=curr_pred),
                                   RegressionSplitter._impurity_fun(impurity_fun, y_true=y[X_split[:, 0]],
                                                                        y_pred=l_pred),
                                   RegressionSplitter._impurity_fun(impurity_fun, y_true=y[~X_split[:, 0]],
                                                                        y_pred=r_pred),
                                   len_x,
                                   len_left,
                                   len_x - len_left)

        info_gain = 1 / (1 + np.exp(-info_gain))

        info_gain -= info_gain * (penalty_value * penalization_weight)

        if info_gain > best_info_gain:
            best_info_gain = info_gain
            best_threshold = value

    return best_info_gain, best_threshold, col_idx