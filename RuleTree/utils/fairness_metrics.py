import numpy as np

from RuleTree.utils.privacy_utils import compute_k_anonimity, _compute_t_closeness, compute_l_diversity


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

            pab = np.abs(np.sum((prot_attr == pr_attr) & (labels == cl_id))/np.sum(labels == cl_id))

            sums[cl_id] += (pab - ideal_dist[pr_attr])/n_prot_attr

    return max(sums.values())



def privacy_metric(X, X_bool, sensible_attribute, k_anonymity, l_diversity, t_closeness, strict):
    X_bool = X_bool.copy().reshape(-1)

    k_left, k_right = compute_k_anonimity(X, X_bool, sensible_attribute)
    l_left, l_right = compute_l_diversity(X, X, sensible_attribute)

    if isinstance(k_anonymity, float):
        k_left /= np.sum(X_bool)
        k_right /= np.sum(~X_bool)

    # print(round(k_left, 3), round(k_right, 3), l_left, l_right, t_left, t_right, sep='\t', end='')

    if isinstance(strict, bool) and strict:
        if min(k_left, k_right) < k_anonymity \
                or min(l_left, l_right) < l_diversity:
            # print()
            return -np.inf

    t_left, t_right = _compute_t_closeness(X, X_bool, sensible_attribute)

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