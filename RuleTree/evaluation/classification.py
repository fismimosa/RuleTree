"""Metrics of a classifier on a test set."""

import numpy as np
import sklearn.metrics as skm


def evaluate_classification(y_true, y_pred, y_proba=None, labels=None) -> dict:
    """
    Accuracy, F1, precision and recall of the predictions of a classifier and, with the class
    probabilities, ROC AUC and average precision.

    F1, precision and recall are given with the micro, macro and weighted averages; ``f1_score``,
    ``precision_score`` and ``recall_score`` are the ones of the positive class (the second one)
    for binary problems, and the micro averages otherwise.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        True classes.
    y_pred : array-like of shape (n_samples,)
        Predicted classes.
    y_proba : array-like of shape (n_samples, n_classes), default=None
        Predicted class probabilities (e.g. ``predict_proba``), in the order of ``labels``. If None,
        the metrics based on probabilities are not computed.
    labels : array-like of shape (n_classes,), default=None
        Classes, in the order of the columns of ``y_proba`` (e.g. ``model.classes_``). If None, the
        sorted classes of ``y_true``.

    Returns
    -------
    scores : dict
        Metric name -> value.

    See Also
    --------
    sklearn.metrics : the definitions of the metrics.

    Examples
    --------
    >>> from sklearn.datasets import load_iris
    >>> from RuleTree import RuleTreeClassifier
    >>> from RuleTree.evaluation import evaluate_classification
    >>> X, y = load_iris(return_X_y=True)
    >>> model = RuleTreeClassifier(max_depth=2, random_state=0).fit(X, y)
    >>> scores = evaluate_classification(y, model.predict(X), model.predict_proba(X), labels=model.classes_)
    >>> round(scores["accuracy"], 3), round(scores["roc_macro"], 3)
    (0.96, 0.98)
    """
    labels = np.unique(y_true) if labels is None else np.asarray(labels)
    binary = len(labels) <= 2
    single = dict(average="binary", pos_label=labels[-1]) if binary else dict(average="micro")

    scores = {
        "accuracy": skm.accuracy_score(y_true, y_pred),
        "balanced_accuracy": skm.balanced_accuracy_score(y_true, y_pred),
    }
    for name, metric in (("f1", skm.f1_score), ("precision", skm.precision_score), ("recall", skm.recall_score)):
        scores[f"{name}_score"] = metric(y_true, y_pred, zero_division=0, **single)
        for average in ("micro", "macro", "weighted"):
            scores[f"{name}_{average}"] = metric(y_true, y_pred, average=average, zero_division=0)

    if y_proba is not None:
        y_proba = np.asarray(y_proba)
        if binary:
            score_positive = y_proba[:, -1] if y_proba.ndim == 2 else y_proba
            y_positive = np.asarray(y_true) == labels[-1]
            for average in ("macro", "micro", "weighted"):
                scores[f"roc_{average}"] = skm.roc_auc_score(y_positive, score_positive, average=average)
                scores[f"average_precision_{average}"] = skm.average_precision_score(
                    y_positive, score_positive, average=average)
        else:
            y_onehot = np.asarray(y_true)[:, None] == labels[None, :]
            for average in ("macro", "micro", "weighted"):
                scores[f"roc_{average}"] = skm.roc_auc_score(y_onehot, y_proba, average=average)
                scores[f"average_precision_{average}"] = skm.average_precision_score(
                    y_onehot, y_proba, average=average)

    return scores
