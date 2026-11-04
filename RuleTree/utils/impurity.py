"""
Impurity of the nodes and gain of the splits, shared by all the splitters.

The impurities are functions of sums over the instances: the weight of each class for classification;
the weight, the weighted sum of the target and of its square (or of ``y * log(y)`` for 'poisson') for
regression. So a splitter computes them for one node by summing over its instances, or for all the
candidate splits at once with cumulative sums (see FlatMatrixSplitter).
"""

import numpy as np
from scipy.special import xlogy


def class_weights(y, classes, sample_weight=None, class_weight=None) -> np.ndarray:
    """
    Total weight of each class in y: the sum of the sample weights, times the weight of the class.

    Parameters
    ----------
    y : numpy.ndarray of shape (n_samples,)
        Classes of the instances.
    classes : array-like
        Classes whose weight is returned (also the ones without instances).
    sample_weight : numpy.ndarray of shape (n_samples,), default=None
        Weight of each instance; 1 if None.
    class_weight : dict, default=None
        Class -> weight; 1 for the classes not in it.

    Returns
    -------
    weights : numpy.ndarray of shape (n_classes,)
    """
    weights = np.ones(len(y)) if sample_weight is None else np.asarray(sample_weight, dtype=float)
    totals = np.array([weights[y == class_label].sum() for class_label in classes], dtype=float)
    if class_weight is not None:
        totals *= [class_weight.get(class_label, 1.0) for class_label in classes]
    return totals


def _probabilities(weights):
    """Probabilities of the classes from their weights (axis 0), all 0 without instances."""
    totals = weights.sum(axis=0)
    return weights / np.where(totals > 0, totals, 1)


def gini(weights) -> np.ndarray:
    """
    Gini impurity from the weights of the classes, 0 without instances.

    Parameters
    ----------
    weights : numpy.ndarray of shape (n_classes, ...)
        Weight of each class, e.g. from :func:`class_weights`.

    Returns
    -------
    impurity : numpy.ndarray of shape (...)
    """
    weights = np.asarray(weights, dtype=float)
    return np.where(weights.sum(axis=0) > 0, 1 - np.sum(_probabilities(weights) ** 2, axis=0), 0.0)


def entropy(weights) -> np.ndarray:
    """
    Entropy (in bits) from the weights of the classes, 0 without instances.

    Parameters
    ----------
    weights : numpy.ndarray of shape (n_classes, ...)
        Weight of each class.

    Returns
    -------
    impurity : numpy.ndarray of shape (...)
    """
    probabilities = _probabilities(np.asarray(weights, dtype=float))
    log_probabilities = np.log2(probabilities, out=np.zeros_like(probabilities), where=probabilities > 0)
    return 0.0 - np.sum(probabilities * log_probabilities, axis=0)  # 0.0, not -0.0, for a pure node


def misclassification(weights) -> np.ndarray:
    """
    Fraction of the weight outside the majority class, 0 without instances.

    Parameters
    ----------
    weights : numpy.ndarray of shape (n_classes, ...)
        Weight of each class.

    Returns
    -------
    impurity : numpy.ndarray of shape (...)
    """
    weights = np.asarray(weights, dtype=float)
    return np.where(weights.sum(axis=0) > 0, 1 - _probabilities(weights).max(axis=0), 0.0)


CLASSIFICATION_IMPURITIES = {"gini": gini, "entropy": entropy, "log_loss": entropy,
                             "misclassification": misclassification}


def regression_statistics(y, criterion="squared_error", sample_weight=None) -> np.ndarray:
    """
    Statistics of each instance for a regression impurity: its weight, its product with the target and
    with the square of the target (with ``y * log(y)`` for 'poisson'). Except for 'poisson', the target is
    centered, for smaller rounding errors.

    Parameters
    ----------
    y : numpy.ndarray of shape (n_samples,)
        Target.
    criterion : {'squared_error', 'poisson'}, default='squared_error'
        Impurity measure.
    sample_weight : numpy.ndarray of shape (n_samples,), default=None
        Weight of each instance; 1 if None.

    Returns
    -------
    statistics : numpy.ndarray of shape (n_samples, 3)

    Raises
    ------
    ValueError
        If the criterion is 'poisson' and the target has negative values.
    """
    weights = np.ones(len(y)) if sample_weight is None else np.asarray(sample_weight, dtype=float)
    y = np.asarray(y, dtype=float)
    if criterion == "poisson":
        if np.any(y < 0):
            raise ValueError("The criterion 'poisson' requires a target without negative values")
        return np.column_stack([weights, weights * y, weights * xlogy(y, y)])
    y = y - weights @ y / max(weights.sum(), np.finfo(float).tiny)
    return np.column_stack([weights, weights * y, weights * y ** 2])


def regression_impurity(statistics, criterion="squared_error") -> np.ndarray:
    """
    Regression impurity from the sums of the statistics of :func:`regression_statistics`, 0 without
    instances: the variance of the target ('squared_error'), or half the Poisson deviance of the target
    predicted with its mean ('poisson'), as in scikit-learn.

    Parameters
    ----------
    statistics : numpy.ndarray of shape (3, ...)
        Total weight, weighted sum of the target and weighted sum of its square (or of ``y * log(y)``).
    criterion : {'squared_error', 'friedman_mse', 'poisson'}, default='squared_error'
        Impurity measure ('friedman_mse' has the impurity of 'squared_error').

    Returns
    -------
    impurity : numpy.ndarray of shape (...)

    Raises
    ------
    ValueError
        If the criterion is unknown.
    """
    weight, y_sum, y_second = np.maximum(statistics[0], 0), statistics[1], statistics[2]
    safe_weight = np.where(weight > 0, weight, 1)
    if criterion in ("squared_error", "friedman_mse"):
        return np.maximum(y_second / safe_weight - (y_sum / safe_weight) ** 2, 0)
    elif criterion == "poisson":
        y_sum = np.maximum(y_sum, 0)
        return np.maximum((y_second - xlogy(y_sum, y_sum / safe_weight)) / safe_weight, 0)
    raise ValueError(f"Unknown criterion {criterion!r}: use 'squared_error' or 'poisson'")


def information_gain(impurity, impurity_left, impurity_right, weight_left, weight_right) -> np.ndarray:
    """
    Decrease of impurity of one or more splits.

    Parameters
    ----------
    impurity : float
        Impurity of the node.
    impurity_left, impurity_right : float or numpy.ndarray of shape (n_candidates,)
        Impurity of the children of each split.
    weight_left, weight_right : float or numpy.ndarray of shape (n_candidates,)
        Total weight of the children of each split.

    Returns
    -------
    gains : float or numpy.ndarray of shape (n_candidates,)
    """
    return impurity - (weight_left * impurity_left + weight_right * impurity_right) / (weight_left + weight_right)


def tree_gain(tree) -> float:
    """
    Decrease of impurity of the split of a fitted scikit-learn tree of depth 1; 0 if the tree has no split.

    Parameters
    ----------
    tree : DecisionTreeClassifier or DecisionTreeRegressor
        A fitted tree of depth 1.

    Returns
    -------
    gain : float
    """
    if len(tree.tree_.impurity) == 1:
        return 0
    impurity, impurity_left, impurity_right = tree.tree_.impurity
    _, weight_left, weight_right = tree.tree_.weighted_n_node_samples
    return information_gain(impurity, impurity_left, impurity_right, weight_left, weight_right)
