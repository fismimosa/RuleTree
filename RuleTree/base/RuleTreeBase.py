"""Base classes for RuleTree estimators."""
from abc import ABC

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_consistent_length

from RuleTree.utils.define import DATA_INPUT_TYPES


class RuleTreeBase(BaseEstimator, ABC):
    """
    Base class for RuleTree estimators.

    RuleTree estimators may receive one input container for each supported data
    type: ``X`` for tabular data, ``X_ts`` for time series, ``X_img`` for images,
    and ``X_txt`` for text. The only assumption made by this base class is that
    the first dimension of every provided container is the number of samples.
    This is enough to validate that all provided inputs, and ``y`` when
    available, refer to the same set of instances.

    Internally, the inputs are handled as a single dictionary ``{data_type: array}``
    (see :data:`RuleTree.utils.define.DATA_INPUT_TYPES`), built by
    :meth:`_validate_inputs`.

    Notes
    -----
    The class follows the scikit-learn estimator API by inheriting from
    :class:`sklearn.base.BaseEstimator`. The base implementation only validates
    inputs; concrete estimators are expected to implement the actual fitting and
    prediction behavior.

    See Also
    --------
    RuleTree.tree.RuleTree.RuleTree : the base class of the trees.
    """

    def _validate_inputs(self, X=None, y=None, X_ts=None, X_img=None, X_txt=None) -> dict:
        """Validate multimodal RuleTree inputs.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features), default=None
            Tabular input data.
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
        data : dict
            Dictionary ``{data_type: numpy.ndarray}`` containing only the provided
            input containers.

        Raises
        ------
        ValueError
            If no input container is provided, or if the provided containers and
            ``y`` do not have a consistent number of samples.
        """
        inputs = {"X": X, "X_ts": X_ts, "X_img": X_img, "X_txt": X_txt}
        data = {
            DATA_INPUT_TYPES[name]: np.asarray(value)
            for name, value in inputs.items()
            if value is not None
        }

        if not data:
            raise ValueError("At least one of X, X_ts, X_img or X_txt must be specified.")

        values_to_check = list(data.values())
        if y is not None:
            values_to_check.append(y)

        check_consistent_length(*values_to_check)
        return data

    @staticmethod
    def _n_samples(data: dict) -> int:
        """Number of samples in a dictionary returned by :meth:`_validate_inputs`."""
        return len(next(iter(data.values())))

    def fit(self, X=None, y=None, X_ts=None, X_img=None, X_txt=None):
        """Validate input data before fitting.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features), default=None
            Tabular input data.
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
        X : array-like of shape (n_samples, n_features), default=None
            Tabular input data.
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
