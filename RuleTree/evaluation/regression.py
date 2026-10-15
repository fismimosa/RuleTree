"""Metrics of a regressor on a test set."""

import numpy as np
import sklearn.metrics as skm


def evaluate_regression(y_true, y_pred) -> dict:
    """
    Errors and explained variance of the predictions of a regressor.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        True values.
    y_pred : array-like of shape (n_samples,)
        Predicted values.

    Returns
    -------
    scores : dict
        ``r2``, ``explained_variance``, ``max_error``, ``mean_absolute_error``,
        ``mean_squared_error``, ``median_absolute_error``, ``mean_absolute_percentage_error`` and
        ``mean_squared_log_error`` (nan if some value is negative).

    See Also
    --------
    sklearn.metrics : the definitions of the metrics.

    Examples
    --------
    >>> from sklearn.datasets import load_diabetes
    >>> from RuleTree import RuleTreeRegressor
    >>> from RuleTree.evaluation import evaluate_regression
    >>> X, y = load_diabetes(return_X_y=True)
    >>> model = RuleTreeRegressor(max_depth=3, random_state=0).fit(X, y)
    >>> round(evaluate_regression(y, model.predict(X))["r2"], 3)
    0.501
    """
    non_negative = np.min(y_true) >= 0 and np.min(y_pred) >= 0
    return {
        "r2": skm.r2_score(y_true, y_pred),
        "explained_variance": skm.explained_variance_score(y_true, y_pred),
        "max_error": skm.max_error(y_true, y_pred),
        "mean_absolute_error": skm.mean_absolute_error(y_true, y_pred),
        "mean_squared_error": skm.mean_squared_error(y_true, y_pred),
        "median_absolute_error": skm.median_absolute_error(y_true, y_pred),
        "mean_absolute_percentage_error": skm.mean_absolute_percentage_error(y_true, y_pred),
        "mean_squared_log_error": skm.mean_squared_log_error(y_true, y_pred) if non_negative else np.nan,
    }
