"""Base classes for RuleTree estimators."""
from abc import ABC

from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_consistent_length


class RuleTreeBase(BaseEstimator, ABC):
    """Base class for RuleTree estimators.

    RuleTree estimators may receive one input container for each supported data
    modality: ``X`` for generic or tabular data, ``X_ts`` for time series,
    ``X_img`` for images, and ``X_txt`` for text. The only assumption made by
    this base class is that the first dimension of every provided container is
    the number of samples. This is enough to validate that all provided inputs,
    and ``y`` when available, refer to the same set of instances.

    Subclasses should call :meth:`fit`, :meth:`predict`,
    :meth:`_validate_inputs`, or :meth:`_resolve_data_input` to reuse this
    validation before implementing their estimator-specific logic.

    Notes
    -----
    The class follows the scikit-learn estimator API by inheriting from
    :class:`sklearn.base.BaseEstimator`. The base implementation only validates
    inputs; concrete estimators are expected to implement the actual fitting and
    prediction behavior.
    """

    _DATA_INPUT_NAMES = ("X", "X_ts", "X_img", "X_txt")

    def _validate_inputs(self, X=None, y=None, X_ts=None, X_img=None, X_txt=None):
        """Validate multimodal RuleTree inputs.

        Parameters
        ----------
        X : array-like of shape (n_samples, ...), default=None
            Generic or tabular input data.
        X_ts : array-like of shape (n_samples, ...), default=None
            Time-series input data.
        X_img : array-like of shape (n_samples, ...), default=None
            Image input data.
        X_txt : array-like of shape (n_samples, ...), default=None
            Text input data.
        y : array-like of shape (n_samples,), default=None
            Target values. If provided, its length is checked against the first
            dimension of the provided input containers.

        Returns
        -------
        data_inputs : dict
            Dictionary containing only the provided input containers, keyed by
            their parameter name.

        Raises
        ------
        ValueError
            If no input container is provided, or if the provided containers and
            ``y`` do not have a consistent number of samples.
        """
        data_inputs = {
            name: value
            for name, value in zip(
                self._DATA_INPUT_NAMES,
                (X, X_ts, X_img, X_txt)
            )
            if value is not None
        }

        if not data_inputs:
            raise ValueError("At least one of X, X_ts, X_img or X_txt must be specified.")

        values_to_check = list(data_inputs.values())
        if y is not None:
            values_to_check.append(y)

        check_consistent_length(*values_to_check)
        return data_inputs

    def fit(self, X=None, y=None, X_ts=None, X_img=None, X_txt=None):
        """Validate input data before fitting.

        Parameters
        ----------
        X : array-like of shape (n_samples, ...), default=None
            Generic or tabular input data.
        X_ts : array-like of shape (n_samples, ...), default=None
            Time-series input data.
        X_img : array-like of shape (n_samples, ...), default=None
            Image input data.
        X_txt : array-like of shape (n_samples, ...), default=None
            Text input data.
        y : array-like of shape (n_samples,), default=None
            Target values.

        Returns
        -------
        self : RuleTreeBase
            The validated estimator instance.

        Raises
        ------
        ValueError
            If no input container is provided, or if the provided containers and
            ``y`` do not have a consistent number of samples.
        """
        self._validate_inputs(X=X, y=y, X_ts=X_ts, X_img=X_img, X_txt=X_txt)
        return self

    def predict(self, X=None, X_ts=None, X_img=None, X_txt=None):
        """Validate input data before prediction.

        Parameters
        ----------
        X : array-like of shape (n_samples, ...), default=None
            Generic or tabular input data.
        X_ts : array-like of shape (n_samples, ...), default=None
            Time-series input data.
        X_img : array-like of shape (n_samples, ...), default=None
            Image input data.
        X_txt : array-like of shape (n_samples, ...), default=None
            Text input data.

        Raises
        ------
        ValueError
            If no input container is provided, or if the provided containers do
            not have a consistent number of samples.
        """
        self._validate_inputs(X=X, X_ts=X_ts, X_img=X_img, X_txt=X_txt)
