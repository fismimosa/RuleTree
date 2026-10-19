"""Metrics of a regressor on a test set."""

import numpy as np
import sklearn.metrics as skm


def evaluate_regression(y_true, y_pred, multioutput="uniform_average") -> dict:
    """
    Errors and explained variance of the predictions of a regressor, with one or several targets.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,) or (n_samples, n_targets)
        True values.
    y_pred : array-like of shape (n_samples,) or (n_samples, n_targets)
        Predicted values.
    multioutput : {'uniform_average', 'raw_values'} or array-like of shape (n_targets,), default='uniform_average'
        With several targets, how the metric of each target is combined, as in :mod:`sklearn.metrics`:
        the mean, the value of each target (an array), or the mean weighted by the given weights.

    Returns
    -------
    scores : dict
        ``r2``, ``explained_variance``, ``max_error``, ``mean_absolute_error``,
        ``mean_squared_error``, ``median_absolute_error``, ``mean_absolute_percentage_error`` and
        ``mean_squared_log_error`` (nan if some value is negative). With several targets,
        ``max_error`` is the largest error of each target, combined as the other metrics.

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

    With two targets, the value of each of them:

    >>> import numpy as np
    >>> Y = np.column_stack([y, X[:, 2]])
    >>> model = RuleTreeRegressor(max_depth=3, random_state=0).fit(X, Y)
    >>> evaluate_regression(Y, model.predict(X), multioutput="raw_values")["r2"].round(3)
    array([0.501, 0.822])
    """
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    non_negative = np.min(y_true) >= 0 and np.min(y_pred) >= 0
    nan = np.full(y_true.shape[1], np.nan) if y_true.ndim == 2 and _is_raw(multioutput) else np.nan

    return {
        "r2": skm.r2_score(y_true, y_pred, multioutput=multioutput),
        "explained_variance": skm.explained_variance_score(y_true, y_pred, multioutput=multioutput),
        "max_error": _max_error(y_true, y_pred, multioutput),
        "mean_absolute_error": skm.mean_absolute_error(y_true, y_pred, multioutput=multioutput),
        "mean_squared_error": skm.mean_squared_error(y_true, y_pred, multioutput=multioutput),
        "median_absolute_error": skm.median_absolute_error(y_true, y_pred, multioutput=multioutput),
        "mean_absolute_percentage_error": skm.mean_absolute_percentage_error(y_true, y_pred,
                                                                             multioutput=multioutput),
        "mean_squared_log_error": skm.mean_squared_log_error(y_true, y_pred, multioutput=multioutput)
        if non_negative else nan,
    }


def _is_raw(multioutput) -> bool:
    """Whether the metrics are returned for each target."""
    return isinstance(multioutput, str) and multioutput == "raw_values"


def _max_error(y_true: np.ndarray, y_pred: np.ndarray, multioutput):
    """
    Largest absolute error: of the single target, or of each target combined as ``multioutput``
    (scikit-learn's max_error supports only one target).
    """
    if y_true.ndim == 1:
        return skm.max_error(y_true, y_pred)

    errors = np.abs(y_true - y_pred).max(axis=0)
    if _is_raw(multioutput):
        return errors
    if isinstance(multioutput, str):
        if multioutput != "uniform_average":
            raise ValueError(f"multioutput must be 'uniform_average', 'raw_values' or an array of weights, "
                             f"got {multioutput!r}.")
        return float(errors.mean())
    return float(np.average(errors, weights=multioutput))
