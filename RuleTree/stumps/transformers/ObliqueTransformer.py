"""ObliqueTransformer: linear combinations of the numerical features, the base of the oblique transformers."""

from abc import ABC, abstractmethod

import numpy as np

from RuleTree.base.RuleTreeBaseTransformer import RuleTreeBaseTransformer, get_feature_types
from RuleTree.exceptions import NoSplitFoundWarning
from RuleTree.utils.define import DATA_TYPE_TABULAR


class ObliqueTransformer(RuleTreeBaseTransformer, ABC):
    """
    Base class of the transformers into linear combinations of the numerical features (oblique
    directions): each column of the feature matrix is ``X[:, numerical] @ coefficients_[:, j]``, so
    a threshold on it is an oblique hyperplane, e.g. ``0.8 * X_2 - 0.6 * X_3 <= 1.5``.

    The subclasses choose the directions in :meth:`fit` (``_fit_directions``). After the splitter has
    chosen a column, only its direction is kept (``select``).

    Attributes
    ----------
    numerical_ : list of int
        Numerical columns of the feature matrix (all of them).
    categorical_ : list of int
        Categorical columns of the feature matrix (none).
    num_pre_transformed_ : list of int
        Numerical columns of the data, the inputs of the combinations.
    coefficients_ : numpy.ndarray of shape (n_numerical_features, n_directions)
        Coefficients of each direction (only the chosen one after ``select``).

    See Also
    --------
    HouseholderTransformer, BivariateTransformer : the oblique transformers.
    """

    supported_data_types = (DATA_TYPE_TABULAR,)

    def fit(self, X, y, idx=None, data_type=None, cache=None, sample_weight=None):
        """
        Choose the oblique directions on the numerical features of the instances in the node.

        Parameters
        ----------
        X : numpy.ndarray
            Tabular data, all the instances.
        y : numpy.ndarray
            Target, all the instances.
        idx : numpy.ndarray, default=None
            Indices of the instances in the node. If None, all the instances.
        data_type : str, default=None
            Type of X, used as key of the cached values.
        cache : dict, default=None
            Values shared among the splits of the tree.
        sample_weight : numpy.ndarray, default=None
            Sample weights, all the instances.

        Returns
        -------
        self : ObliqueTransformer

        Raises
        ------
        NoSplitFoundWarning
            If the data have no numerical feature.
        """
        if idx is None:
            idx = np.arange(len(X))

        self.num_pre_transformed_, _ = get_feature_types(X, cache, data_type)
        if len(self.num_pre_transformed_) == 0:
            raise NoSplitFoundWarning(f"{self.__class__.__name__} needs numerical features")

        X_node = self._numerical(X[idx])
        self.coefficients_ = self._fit_directions(X_node, y[idx],
                                                  None if sample_weight is None else sample_weight[idx])
        self.numerical_, self.categorical_ = list(range(self.coefficients_.shape[1])), []

        return self

    @abstractmethod
    def _fit_directions(self, X, y, sample_weight=None) -> np.ndarray:
        """
        Oblique directions for the instances of the node.

        Parameters
        ----------
        X : numpy.ndarray of shape (n_samples, n_numerical_features)
            Numerical features of the instances in the node.
        y : numpy.ndarray of shape (n_samples,)
            Target of the instances in the node.
        sample_weight : numpy.ndarray of shape (n_samples,), default=None
            Sample weights of the instances in the node.

        Returns
        -------
        coefficients : numpy.ndarray of shape (n_numerical_features, n_directions)
        """

    def _numerical(self, X):
        """Numerical features of X, as floats."""
        return np.asarray(X)[:, self.num_pre_transformed_].astype(float)

    def transform(self, X):
        """
        Projection of the instances on the oblique directions (on the chosen one, after select).

        Parameters
        ----------
        X : numpy.ndarray
            Tabular data.

        Returns
        -------
        X_transformed : numpy.ndarray of shape (n_samples, n_directions)
        """
        return self._numerical(X) @ self.coefficients_

    def select(self, feature):
        """
        Keep only the chosen direction.

        Parameters
        ----------
        feature : int
            Column chosen by the splitter.

        Returns
        -------
        feature : int
            Always 0: the projection on the only direction kept.
        """
        self.coefficients_ = self.coefficients_[:, [feature]]
        self.numerical_, self.categorical_ = [0], []
        return 0

    def combination(self, feature: int) -> tuple[list, list]:
        """
        Features and coefficients of a direction: the column is ``sum(c * X[:, f])``.

        Parameters
        ----------
        feature : int
            Column of the feature matrix.

        Returns
        -------
        features : list of int
            Columns of the data with a non-zero coefficient.
        coefficients : list of float
            Their coefficients.
        """
        column = self.coefficients_[:, feature]
        nonzero = np.flatnonzero(np.abs(column) > 1e-12)
        return [int(self.num_pre_transformed_[i]) for i in nonzero], [float(c) for c in column[nonzero]]

    def feature_id(self, feature):
        """
        Identifier of a direction: the columns of the data it combines.

        Parameters
        ----------
        feature : int
            Column of the feature matrix.

        Returns
        -------
        feature_id : tuple of int
        """
        return tuple(self.combination(feature)[0])

    def get_state(self):
        """
        Fitted state: the numerical features and the coefficients of the directions.

        Returns
        -------
        state : dict
            JSON-friendly dictionary, restored by set_state.
        """
        return {"num_pre_transformed": [int(f) for f in self.num_pre_transformed_],
                "coefficients": self.coefficients_.tolist()}

    def set_state(self, state):
        """
        Restore the fitted state saved by get_state.

        Parameters
        ----------
        state : dict
            Dictionary created by get_state.

        Returns
        -------
        self : ObliqueTransformer
        """
        self.num_pre_transformed_ = list(state["num_pre_transformed"])
        self.coefficients_ = np.array(state["coefficients"], dtype=float)
        self.numerical_, self.categorical_ = list(range(self.coefficients_.shape[1])), []
        return self
