import numpy as np

from RuleTree.utils.privacy_utils import compute_k_anonimity, compute_t_closeness, compute_l_diversity


def balance_metric(labels:np.ndarray, prot_attr:np.ndarray):
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
    return [
        0 if k >= k_thr else (k_thr-k)/k_thr,
        0 if l >= l_thr else (l_thr-l)/l_thr,
        0 if t <= t_thr else t,
    ]