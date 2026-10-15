"""HouseholderTransformer: oblique directions from the Householder reflection of the principal component."""

import numpy as np
from sklearn.decomposition import PCA

from RuleTree.stumps.transformers.ObliqueTransformer import ObliqueTransformer


class HouseholderTransformer(ObliqueTransformer):
    """
    Transformer into the oblique directions given by a Householder reflection: the reflection that
    maps the principal component of the instances of the node onto the closest axis.

    The reflected axes ``H = I - 2 u u^T`` are the columns of the feature matrix: the splitter
    chooses the best threshold on one of them. The reflection vector ``u`` is restricted to the
    ``max_oblique_features`` features with the largest weights, so that each oblique direction
    combines at most that many features (plus the axis itself); the axes not involved in the
    reflection stay axis-parallel. The directions are computed on the numerical features of the
    node, as they are: scale the data beforehand if the features have different scales.

    Parameters
    ----------
    max_oblique_features : int, default=2
        Maximum number of features in the reflection vector.
    tau : float, default=1e-4
        If the principal component is closer than tau to every axis, the directions are the axes
        (no reflection).
    random_state : int, RandomState instance or None, default=None
        Seed of the randomized solver of the PCA, used on large nodes.

    Attributes
    ----------
    numerical_ : list of int
        Numerical columns of the feature matrix.
    categorical_ : list of int
        Categorical columns of the feature matrix (none).
    num_pre_transformed_ : list of int
        Numerical columns of the data.
    coefficients_ : numpy.ndarray of shape (n_numerical_features, n_directions)
        The columns of the reflection ``H`` (only the chosen one after ``select``).

    See Also
    --------
    BivariateTransformer : oblique directions on pairs of features.
    ObliqueDecisionTreeStumpClassifier, ObliqueDecisionTreeStumpRegressor : the stumps that use it.

    Examples
    --------
    >>> from sklearn.datasets import load_iris
    >>> from RuleTree.stumps.transformers import HouseholderTransformer
    >>> X, y = load_iris(return_X_y=True)
    >>> transformer = HouseholderTransformer().fit(X, y)
    >>> transformer.transform(X).shape  # one column per reflected axis
    (150, 4)
    >>> features, coefficients = transformer.combination(1)  # the reflected axis of feature 1
    >>> features, [round(c, 3) for c in coefficients]
    ([1, 2], [-0.085, 0.857])
    """

    def __init__(self, max_oblique_features=2, tau=1e-4, random_state=None):
        """Initialize the HouseholderTransformer; see the class documentation for the parameters."""
        self.max_oblique_features = max_oblique_features
        self.tau = tau
        self.random_state = random_state

    def _fit_directions(self, X, y, sample_weight=None):
        """
        Columns of the Householder reflection of the principal component of X.

        Parameters
        ----------
        X : numpy.ndarray of shape (n_samples, n_numerical_features)
            Numerical features of the instances in the node.
        y : numpy.ndarray of shape (n_samples,)
            Not used: the reflection depends only on the data.
        sample_weight : numpy.ndarray, default=None
            Not used.

        Returns
        -------
        coefficients : numpy.ndarray of shape (n_numerical_features, n_numerical_features)
        """
        n_features = X.shape[1]
        identity = np.eye(n_features)
        if n_features == 1 or len(X) < 2:
            return identity

        dominant = PCA(n_components=1, random_state=self.random_state).fit(X).components_[0]
        distances = np.sqrt(((identity - dominant) ** 2).sum(axis=1))  # from each axis
        if (distances > self.tau).sum() == 0:
            return identity  # nothing to reflect

        axis = identity[np.argmax(distances)]
        u = (axis - dominant) / np.linalg.norm(axis - dominant)

        if self.max_oblique_features < n_features:
            keep = np.argpartition(np.abs(u), -self.max_oblique_features)[-self.max_oblique_features:]
            u_kept = np.zeros(n_features)
            u_kept[keep] = u[keep]
            u = u_kept

        return identity - 2 * np.outer(u, u)
