"""
Abstract base class of the splitters: the component of a stump that finds the best split of a
feature matrix (the matrix produced by the stump's transformer, see RuleTreeBaseTransformer).
"""
from abc import ABC, abstractmethod

import numpy as np
from sklearn.base import BaseEstimator


class RuleTreeBaseSplitter(BaseEstimator, ABC):
    """
    Abstract base class of the splitters, the component of a stump that finds the best split of a
    feature matrix: a threshold on a numerical feature (``x <= threshold`` goes to the left child)
    or a value of a categorical feature (``x == value`` goes to the left child).

    A new splitter implements :meth:`fit`, which sets the attributes below; :meth:`apply`,
    :meth:`get_state` and :meth:`set_state` work for any split of this form. Splitters are
    scikit-learn estimators: their parameters are the arguments of ``__init__``.

    Attributes
    ----------
    task : str
        Kind of target: MODEL_TYPE_CLF (classification) or MODEL_TYPE_REG (regression). Class attribute.
    feature_ : int
        Column of the matrix used by the split.
    threshold_ : float or object
        Threshold (numerical split) or value (categorical split).
    is_categorical_ : bool
        Whether the split is categorical.
    gain_ : float
        Gain of the split: the stumps with the highest gain are chosen with stump_selection='best'.
    impurity_ : list of float or None
        Impurity of the node and of the two children, if computed.
    """

    # MODEL_TYPE_CLF or MODEL_TYPE_REG: the kind of target the splitter works with
    task = None

    # fitted attributes saved by get_state
    _state = ("feature_", "threshold_", "is_categorical_", "gain_", "impurity_")

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray, numerical: list, categorical: list, sample_weight=None,
            check_input=True):
        """
        Find the best split of X.

        Parameters
        ----------
        X : numpy.ndarray of shape (n_samples, n_features)
            Feature matrix of the instances in the node.
        y : numpy.ndarray of shape (n_samples,)
            Target of the instances in the node.
        numerical : list of int
            Indices of the numerical columns of X.
        categorical : list of int
            Indices of the categorical columns of X.
        sample_weight : numpy.ndarray of shape (n_samples,), default=None
            Sample weights of the instances in the node.
        check_input : bool, default=True
            Whether to check the input data.

        Returns
        -------
        self : RuleTreeBaseSplitter
            The fitted splitter.

        Raises
        ------
        NoSplitFoundWarning
            If no split is found.
        """

    def apply(self, X: np.ndarray) -> np.ndarray:
        """
        Apply the split to a feature matrix.

        Parameters
        ----------
        X : numpy.ndarray of shape (n_samples, n_features)
            Feature matrix.

        Returns
        -------
        labels : numpy.ndarray of int
            For each row, 1 if it goes to the left child, 2 otherwise.
        """
        X_feature = X[:, self.feature_]
        if self.is_categorical_:
            return np.where(X_feature == self.threshold_, 1, 2)

        return np.where(X_feature <= self.threshold_, 1, 2)

    def update_statistics(self, X: np.ndarray, y: np.ndarray, sample_weight=None):
        """
        Recompute the statistics of the split (e.g. impurity) on new data, without changing the split.
        The default implementation does nothing.

        Parameters
        ----------
        X : numpy.ndarray of shape (n_samples, n_features)
            Feature matrix of the instances in the node.
        y : numpy.ndarray of shape (n_samples,)
            Target of the instances in the node.
        sample_weight : numpy.ndarray of shape (n_samples,), default=None
            Sample weights of the instances in the node.
        """

    def get_state(self) -> dict:
        """
        Fitted state of the splitter: the attributes listed in ``_state``.

        Returns
        -------
        state : dict
            JSON-friendly dictionary, restored by :meth:`set_state`.
        """
        return {name: getattr(self, name, None) for name in self._state}

    def set_state(self, state: dict):
        """
        Restore the fitted state saved by :meth:`get_state`.

        Parameters
        ----------
        state : dict
            Dictionary created by :meth:`get_state`.

        Returns
        -------
        self : RuleTreeBaseSplitter
        """
        for name in self._state:
            setattr(self, name, state.get(name))

        return self
