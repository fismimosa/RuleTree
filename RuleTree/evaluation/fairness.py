"""Fairness of a partition of the instances (clusters or predicted classes) with respect to a sensitive attribute."""

import numpy as np

from RuleTree.utils.fairness_metrics import balance_metric, max_fairness_cost


def evaluate_fairness(labels, sensitive_attribute, ideal_distribution=None) -> dict:
    """
    Fairness of the groups of a partition (clusters, leaves or predicted classes) with respect to
    a sensitive attribute.

    Parameters
    ----------
    labels : array-like of shape (n_samples,)
        Group of each instance, e.g. its cluster.
    sensitive_attribute : array-like of shape (n_samples,)
        Value of the sensitive attribute of each instance.
    ideal_distribution : dict, default=None
        Value of the sensitive attribute -> ideal fraction in each group. If None, the distribution
        in the whole data set.

    Returns
    -------
    scores : dict
        ``balance``: the smallest ratio between the fraction of a value of the sensitive attribute
        in a group and in the whole data set (or its inverse), from 0 to 1, the higher the fairer;
        ``max_fairness_cost``: the largest mean absolute difference between the distribution of
        the sensitive attribute in a group and the ideal one, the lower the fairer.

    See Also
    --------
    RuleTree.stumps.regression.FairTreeStumpRegressor : the stump that builds fair trees.

    Examples
    --------
    >>> import numpy as np
    >>> from RuleTree.evaluation import evaluate_fairness
    >>> labels = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    >>> sex = np.array(["F", "M", "F", "M", "F", "F", "F", "M"])
    >>> evaluate_fairness(labels, sex)
    {'balance': 0.6666666666666666, 'max_fairness_cost': 0.125}
    """
    labels, sensitive_attribute = np.asarray(labels), np.asarray(sensitive_attribute)
    if ideal_distribution is None:
        values, counts = np.unique(sensitive_attribute, return_counts=True)
        ideal_distribution = dict(zip(values, counts / counts.sum()))

    return {
        "balance": float(balance_metric(labels, sensitive_attribute)),
        "max_fairness_cost": float(max_fairness_cost(labels, sensitive_attribute, ideal_distribution)),
    }
