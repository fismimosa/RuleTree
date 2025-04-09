import numpy as np

from RuleTree.utils.privacy_utils import compute_k_anonimity, compute_t_closeness, compute_l_diversity

"""
Fairness and privacy metrics for evaluating machine learning models.

This module provides various metrics to assess fairness with respect to protected attributes
and privacy preservation in the context of decision trees and rule-based models.
"""

def balance_metric(labels:np.ndarray, prot_attr:np.ndarray):
    """
    Computes the balance fairness metric between protected attributes and class labels.

    The balance metric measures how evenly the protected attributes are distributed across
    different class labels. A value closer to 1 indicates better balance.

    Parameters
    ----------
    labels : np.ndarray
        Array of class labels for each instance
    prot_attr : np.ndarray
        Array of protected attribute values for each instance

    Returns
    -------
    float
        Minimum balance ratio across all protected attribute and class label combinations.
        Values range from 0 to 1, with 1 indicating perfect balance.
    """
    res = []

    for pr_attr in np.unique(prot_attr):
        r = np.sum(prot_attr == pr_attr)/len(labels)
        for cl_id in np.unique(labels):
            ra = np.sum((labels == cl_id) & (prot_attr == pr_attr))/np.sum(labels == cl_id)
            rab= r/ra if ra != 0 else 0
            rab_1 = 1/rab if rab != 0 else 1
            res.append(min(rab, rab_1))


    return min(res)


def max_fairness_cost(labels:np.ndarray, prot_attr:np.ndarray, ideal_dist:dict):
    """
    Calculates the maximum fairness cost based on deviation from an ideal distribution.

    This metric measures how far the actual distribution of protected attributes within
    each class is from an ideal (fair) distribution.

    Parameters
    ----------
    labels : np.ndarray
        Array of class labels for each instance
    prot_attr : np.ndarray
        Array of protected attribute values for each instance
    ideal_dist : dict
        Dictionary mapping protected attribute values to their ideal proportions

    Returns
    -------
    float
        Maximum fairness cost across all class labels. Lower values indicate
        distributions closer to the ideal fair distribution.
    """
    sums = dict()

    n_prot_attr = len(np.unique(prot_attr))

    for pr_attr in np.unique(prot_attr):
        for cl_id in np.unique(labels):
            if cl_id not in sums:
                sums[cl_id] = .0

            pab = (np.sum((prot_attr == pr_attr) & (labels == cl_id))/np.sum(labels == cl_id))

            sums[cl_id] += (np.abs(pab - ideal_dist[pr_attr])/n_prot_attr)

    return max(sums.values())



def privacy_metric(X, X_bool, sensible_attribute, k_anonymity, l_diversity, t_closeness, strict, categorical, use_t):
    """
    Evaluates privacy preservation based on k-anonymity, l-diversity, and t-closeness.

    This function checks whether a split of the data satisfies specified privacy constraints.

    Parameters
    ----------
    X : array-like
        The dataset features
    X_bool : array-like
        Boolean array indicating the split (True/False for each instance)
    sensible_attribute : int or list
        Index or indices of the sensitive attribute(s) in X
    k_anonymity : int or float
        Minimum required k-anonymity level. If float, interpreted as a proportion
    l_diversity : int
        Minimum required l-diversity level
    t_closeness : float
        Maximum allowed t-closeness value
    strict : bool
        If True, enforces all privacy constraints strictly
    categorical : bool
        Whether the sensitive attribute is categorical
    use_t : bool
        Whether to compute and use t-closeness in the evaluation

    Returns
    -------
    tuple
        (is_valid, k, l, t) where:
        - is_valid: boolean indicating if privacy constraints are satisfied
        - k: achieved k-anonymity value
        - l: achieved l-diversity value
        - t: achieved t-closeness value or np.nan if not computed
    """
    X_bool = X_bool.copy().reshape(-1)

    k_left, k_right = compute_k_anonimity(X, X_bool, sensible_attribute)
    l_left, l_right = compute_l_diversity(X, X_bool, sensible_attribute)

    if isinstance(k_anonymity, float):
        k_left /= np.sum(X_bool)
        k_right /= np.sum(~X_bool)

    k = min(k_left, k_right)
    l = min(l_left, l_right)

    if strict and (k < k_anonymity or l < l_diversity):
        return False, k, l, np.nan

    if not use_t:
        return True, k, l, np.nan

    t_left, t_right = compute_t_closeness(X, X_bool, sensible_attribute, categorical)
    t = max(t_left, t_right)

    # print('\t', t_left, '\t', t_right)
    if strict and t > t_closeness:
        return False, k, l, t

    k = min(k_left, k_right)
    l = min(l_left, l_right)

    return True, k, l, t


def privacy_metric_all(k, k_thr, l, l_thr, t, t_thr):
    """
    Computes normalized privacy violations for k-anonymity, l-diversity, and t-closeness.

    Parameters
    ----------
    k : float
        Achieved k-anonymity value
    k_thr : float
        Threshold for k-anonymity
    l : float
        Achieved l-diversity value
    l_thr : float
        Threshold for l-diversity
    t : float
        Achieved t-closeness value
    t_thr : float
        Threshold for t-closeness

    Returns
    -------
    list
        List of normalized violation scores for [k-anonymity, l-diversity, t-closeness].
        Each score is 0 if the constraint is satisfied, otherwise it's a positive value
        representing the degree of violation.
    """
    return [
        0 if k >= k_thr else (k_thr-k)/k_thr,
        0 if l >= l_thr else (l_thr-l)/l_thr,
        0 if t <= t_thr else t,
    ]
