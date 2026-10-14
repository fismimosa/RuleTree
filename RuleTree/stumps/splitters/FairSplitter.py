from concurrent.futures.process import ProcessPoolExecutor
from typing import Union

import numpy as np
import psutil

from RuleTree.base.RuleTreeBaseSplitter import RuleTreeBaseSplitter
from RuleTree.exceptions import NoSplitFoundWarning
from RuleTree.stumps.splitters.RegressionSplitter import RegressionSplitter
from RuleTree.utils.define import MODEL_TYPE_REG


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
        # the thresholds are evaluated by the functions of the Fair stump module
        from RuleTree.stumps.regression.FairTreeStumpRegressor import _inner_loop_best_split

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
