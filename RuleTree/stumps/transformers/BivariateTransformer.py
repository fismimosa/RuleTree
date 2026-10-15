"""BivariateTransformer: oblique directions on the best pair of features."""

import itertools

import numpy as np
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from RuleTree.exceptions import NoSplitFoundWarning
from RuleTree.stumps.transformers.ObliqueTransformer import ObliqueTransformer
from RuleTree.utils.data_utils import get_info_gain
from RuleTree.utils.define import MODEL_TYPE_CLF, MODEL_TYPE_REG


class BivariateTransformer(ObliqueTransformer):
    """
    Transformer into oblique directions on a pair of features: ``cos(a) * X_i + sin(a) * X_j`` for
    ``n_orientations`` angles ``a`` between 0 and 180 degrees.

    The pair is chosen in ``fit``: for each pair of numerical features, a depth-1 decision tree of
    the task is fitted on its projections, and the pair with the highest gain is kept. The splitter
    then chooses the angle and the threshold. The projections are computed on the features as they
    are: scale the data beforehand if the features have different scales.

    Parameters
    ----------
    ml_task : {MODEL_TYPE_CLF, MODEL_TYPE_REG}, default=MODEL_TYPE_CLF
        Task of the stump, which determines the trees that choose the pair.
    n_orientations : int, default=10
        Number of angles, evenly spaced in [0, 180) degrees.
    tree_params : dict, default=None
        Parameters of the depth-1 decision trees that choose the pair (e.g. ``criterion``,
        ``random_state``).

    Attributes
    ----------
    numerical_ : list of int
        Numerical columns of the feature matrix.
    categorical_ : list of int
        Categorical columns of the feature matrix (none).
    num_pre_transformed_ : list of int
        Numerical columns of the data.
    coefficients_ : numpy.ndarray of shape (n_numerical_features, n_orientations)
        Coefficients of each direction: non-zero only on the chosen pair (only the chosen
        direction after ``select``).

    See Also
    --------
    HouseholderTransformer : oblique directions from a reflection of the principal component.
    ObliqueDecisionTreeStumpClassifier, ObliqueDecisionTreeStumpRegressor : the stumps that use it.

    Examples
    --------
    >>> from sklearn.datasets import load_iris
    >>> from RuleTree.stumps.transformers import BivariateTransformer
    >>> X, y = load_iris(return_X_y=True)
    >>> transformer = BivariateTransformer(n_orientations=4, tree_params={"random_state": 0}).fit(X, y)
    >>> transformer.transform(X).shape  # one column per angle
    (150, 4)
    >>> transformer.feature_id(1)  # the chosen pair of features
    (0, 1)
    """

    def __init__(self, ml_task=MODEL_TYPE_CLF, n_orientations=10, tree_params=None):
        """Initialize the BivariateTransformer; see the class documentation for the parameters."""
        self.ml_task = ml_task
        self.n_orientations = n_orientations
        self.tree_params = tree_params

    def _orientations(self):
        """Matrix of shape (2, n_orientations) with the cosine and sine of each angle."""
        angles = np.linspace(0, np.pi, self.n_orientations, endpoint=False)
        return np.vstack([np.cos(angles), np.sin(angles)])

    def _gain(self, X_projected, y, sample_weight=None):
        """Gain of the depth-1 decision tree of the task on the projections of a pair."""
        tree_params = (self.tree_params or {}) | {"max_depth": 1}
        tree = (DecisionTreeRegressor if self.ml_task == MODEL_TYPE_REG else DecisionTreeClassifier)(**tree_params)
        tree.fit(X_projected, y, sample_weight=sample_weight)
        return get_info_gain(tree)

    def _fit_directions(self, X, y, sample_weight=None):
        """
        Projections on the pair of features with the highest gain.

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
        coefficients : numpy.ndarray of shape (n_numerical_features, n_orientations)

        Raises
        ------
        NoSplitFoundWarning
            If there are fewer than two numerical features.
        """
        if X.shape[1] < 2:
            raise NoSplitFoundWarning(f"{self.__class__.__name__} needs at least two numerical features")

        orientations = self._orientations()
        best_gain, best_pair = -np.inf, None
        for pair in itertools.combinations(range(X.shape[1]), 2):
            gain = self._gain(X[:, list(pair)] @ orientations, y, sample_weight)
            if gain > best_gain:
                best_gain, best_pair = gain, pair

        coefficients = np.zeros((X.shape[1], orientations.shape[1]))
        coefficients[list(best_pair)] = orientations
        return coefficients
