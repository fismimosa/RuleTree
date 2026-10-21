"""ObliquePivotTransformer: oblique directions on the distances from the candidate pivots."""

import numpy as np

from RuleTree.stumps.transformers.BivariateTransformer import BivariateTransformer
from RuleTree.stumps.transformers.HouseholderTransformer import HouseholderTransformer
from RuleTree.stumps.transformers.PivotTransformer import PivotTransformer, scaler_from_state, scaler_state
from RuleTree.stumps.transformers._pivot_selection import PivotSelection
from RuleTree.utils.define import MODEL_TYPE_CLF


class ObliquePivotTransformer(PivotTransformer):
    """
    Transformer of tabular data into oblique directions on the distances from the candidate pivots,
    e.g. ``0.7 * P_12 - 0.3 * P_57`` where ``P_i`` is the distance from the training instance ``i``.

    The candidate pivots are selected as in
    :class:`~RuleTree.stumps.transformers.PivotTransformer.PivotTransformer`, but the discriminative
    ones are the instances combined by an oblique direction (``oblique_split_type``, see
    PivotSelection). Then the distances of the instances of the node from the candidates are
    transformed by an oblique transformer of the same type
    (:class:`~RuleTree.stumps.transformers.HouseholderTransformer.HouseholderTransformer` or
    :class:`~RuleTree.stumps.transformers.BivariateTransformer.BivariateTransformer`), whose
    directions are the columns of the feature matrix. After the splitter has chosen a column, only its
    direction is kept (``select``); the candidates are all kept.

    Parameters
    ----------
    ml_task : {MODEL_TYPE_CLF, MODEL_TYPE_REG}, default=MODEL_TYPE_CLF
        Task of the stump, which determines how the pivots and the directions are selected.
    distance_measure : str or callable, default='euclidean'
        Distance between the instances, computed on the standardized numerical features.
    k, reg_contrast_lambda, reg_max_thresholds
        Regression only: see PivotTransformer.
    tree_params : dict, default=None
        Parameters of the decision trees that select the pivots and the directions.
    oblique_split_type : {'householder', 'bivariate'}, default='householder'
        Type of the oblique directions.
    max_oblique_features : int, default=2
        Householder only: maximum number of pivots combined by a direction.
    tau : float, default=1e-4
        Householder only: threshold of the reflection.
    n_orientations : int, default=10
        Bivariate only: number of angles.

    Attributes
    ----------
    numerical_ : list of int
        Numerical columns of the feature matrix.
    categorical_ : list of int
        Categorical columns of the feature matrix (none).
    candidates_ : numpy.ndarray of shape (n_candidates, n_numerical_features)
        The candidate pivots, standardized.
    candidate_names_ : numpy.ndarray of shape (n_candidates,)
        Indices of the candidate pivots in the training set.
    oblique_ : HouseholderTransformer or BivariateTransformer
        The oblique transformer of the distances from the candidates.
    pivots_ : dict
        Indices in the training set of the ``descriptives``, ``discriminatives`` and ``candidates``
        pivots of the node.
    scaler_ : StandardScaler
        Scaler of the numerical features, fitted on the whole data set.

    See Also
    --------
    ObliquePivotTreeStumpClassifier : the stump that uses it.

    Examples
    --------
    >>> from sklearn.datasets import load_iris
    >>> from RuleTree.stumps.transformers import ObliquePivotTransformer
    >>> X, y = load_iris(return_X_y=True)
    >>> transformer = ObliquePivotTransformer(tree_params={"random_state": 0}).fit(X, y, context=None, node_id=None)
    >>> transformer.transform(X).shape[1] == len(transformer.candidate_names_)  # one direction per candidate
    True
    """

    def __init__(self, ml_task=MODEL_TYPE_CLF, distance_measure='euclidean', k=None, reg_contrast_lambda=.2,
                 reg_max_thresholds=None, tree_params=None, oblique_split_type='householder', max_oblique_features=2,
                 tau=1e-4, n_orientations=10):
        """Initialize the ObliquePivotTransformer; see the class documentation for the parameters."""
        super().__init__(ml_task=ml_task, distance_measure=distance_measure, k=k,
                         reg_contrast_lambda=reg_contrast_lambda, reg_max_thresholds=reg_max_thresholds,
                         tree_params=tree_params)
        self.oblique_split_type = oblique_split_type
        self.max_oblique_features = max_oblique_features
        self.tau = tau
        self.n_orientations = n_orientations

    def _selection(self):
        """A new PivotSelection whose discriminative pivots are found with ``oblique_split_type``."""
        return PivotSelection(self.ml_task, k=self.k, reg_contrast_lambda=self.reg_contrast_lambda,
                              reg_max_thresholds=self.reg_max_thresholds, discriminative=self.oblique_split_type,
                              max_oblique_features=self.max_oblique_features, tau=self.tau,
                              n_orientations=self.n_orientations, **(self.tree_params or {}))

    def _oblique(self):
        """A new, unfitted oblique transformer of the distances, of type ``oblique_split_type``."""
        tree_params = self.tree_params or {}
        if self.oblique_split_type == "bivariate":
            return BivariateTransformer(self.ml_task, n_orientations=self.n_orientations, tree_params=tree_params)
        return HouseholderTransformer(max_oblique_features=self.max_oblique_features, tau=self.tau,
                                      random_state=tree_params.get("random_state"))

    def _distances(self, X):
        """Distances of the instances from the candidate pivots."""
        return PivotTransformer.transform(self, X)

    def fit(self, X, y, idx=None, data_type=None, sample_weight=None, *, context, node_id):
        """
        Select the candidate pivots of the node (see PivotTransformer.fit), then the oblique directions on
        the distances of the instances of the node from them.

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
        sample_weight : numpy.ndarray, default=None
            Sample weights, all the instances.
        context : RuleTree or None
            Tree being built, None outside a tree.
        node_id : str or None
            Identifier of the node, e.g. ``"Rl"`` (``context.get_node_by_id(node_id)``), None outside a tree.

        Returns
        -------
        self : ObliquePivotTransformer

        Raises
        ------
        NoSplitFoundWarning
            If the data have no numerical feature, or (bivariate) there are fewer than two candidates.
        """
        if idx is None:
            idx = np.arange(len(X))
        super().fit(X, y, idx, data_type=data_type, sample_weight=sample_weight, context=context,
                    node_id=node_id)

        # on the distances of the node from the candidates: no cache of the tree
        self.oblique_ = self._oblique().fit(self._distances(X[idx]), np.asarray(y)[idx],
                                            sample_weight=None if sample_weight is None else sample_weight[idx],
                                            context=None, node_id=node_id)
        self.numerical_, self.categorical_ = self.oblique_.numerical_, []

        return self

    def transform(self, X, context=None, node_id=None):
        """
        Projection of the distances from the candidates on the oblique directions (on the chosen one,
        after select).

        Parameters
        ----------
        X : numpy.ndarray
            Tabular data.
        context : RuleTree, default=None
            The tree, None outside a tree.
        node_id : str, default=None
            Identifier of the node of the stump, None outside a tree.

        Returns
        -------
        X_transformed : numpy.ndarray of shape (n_samples, n_directions)
        """
        return self.oblique_.transform(self._distances(X))

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
        feature = self.oblique_.select(feature)
        self.numerical_, self.categorical_ = [0], []
        return feature

    def combination(self, feature):
        """
        Pivots and coefficients of a direction: the column is ``sum(c * P_i)``.

        Parameters
        ----------
        feature : int
            Column of the feature matrix.

        Returns
        -------
        pivots : list of int
            Indices in the training set of the pivots with a non-zero coefficient.
        coefficients : list of float
            Their coefficients.
        """
        candidates, coefficients = self.oblique_.combination(feature)
        return [int(self.candidate_names_[c]) for c in candidates], coefficients

    def feature_id(self, feature):
        """
        Identifier of a direction: the indices of the pivots it combines.

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
        Fitted state: the candidate pivots and their indices, the numerical features, the scaler and the
        state of the oblique transformer.

        Returns
        -------
        state : dict
            JSON-friendly dictionary, restored by set_state.
        """
        return {
            'num_pre_transformed': list(self.num_pre_transformed_),
            'cat_pre_transformed': list(self.cat_pre_transformed_),
            'candidates': np.asarray(self.candidates_).tolist(),
            'candidate_names': [int(name) for name in self.candidate_names_],
            'pivots': self.pivots_,
            'scaler': scaler_state(self.scaler_),
            'oblique': self.oblique_.get_state(),
        }

    def set_state(self, state):
        """
        Restore the fitted state saved by get_state.

        Parameters
        ----------
        state : dict
            Dictionary created by get_state.

        Returns
        -------
        self : ObliquePivotTransformer
        """
        self.num_pre_transformed_ = state['num_pre_transformed']
        self.cat_pre_transformed_ = state['cat_pre_transformed']
        self.candidates_ = np.array(state['candidates'])
        self.candidate_names_ = np.array(state['candidate_names'])
        self.pivots_ = state['pivots']
        self.scaler_ = scaler_from_state(state['scaler'])
        self.oblique_ = self._oblique().set_state(state['oblique'])
        self.numerical_, self.categorical_ = self.oblique_.numerical_, []
        return self
