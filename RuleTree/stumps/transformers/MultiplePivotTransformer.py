"""MultiplePivotTransformer: which of two pivots is closer to each instance."""

import itertools

import numpy as np

from RuleTree.exceptions import NoSplitFoundWarning
from RuleTree.stumps.transformers.PivotTransformer import PivotTransformer, scaler_from_state, scaler_state
from RuleTree.stumps.transformers.ProximityTransformer import closer_to_second
from RuleTree.stumps.transformers._pivot_selection import PivotSelection
from RuleTree.utils.define import MODEL_TYPE_CLF


class MultiplePivotTransformer(PivotTransformer):
    """
    Transformer of tabular data into pivot comparisons: for each pair of candidate pivots, whether the
    instance is closer to the second one than to the first.

    The candidate pivots are selected as in
    :class:`~RuleTree.stumps.transformers.PivotTransformer.PivotTransformer` (with
    ``discriminative``, also with oblique directions); for classification, only the pairs of pivots of
    different classes are compared. The feature matrix has one 0/1 column for each pair (see
    :func:`~RuleTree.stumps.transformers.ProximityTransformer.closer_to_second`). After the splitter
    has chosen a pair, only its two pivots are kept (``select``).

    Parameters
    ----------
    ml_task : {MODEL_TYPE_CLF, MODEL_TYPE_REG}, default=MODEL_TYPE_CLF
        Task of the stump, which determines how the pivots are selected and compared.
    distance_measure : str or callable, default='euclidean'
        Distance between the instances, computed on the standardized numerical features.
    k, reg_contrast_lambda, reg_max_thresholds
        Regression only: see PivotTransformer.
    tree_params : dict, default=None
        Parameters of the decision trees that select the discriminative pivots.
    discriminative : {'tree', 'householder', 'bivariate'}, default='tree'
        How the discriminative pivots are found (see PivotSelection).
    max_oblique_features : int, default=2
        Householder only: maximum number of discriminative pivots of each class.
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
        The candidate pivots (the two of the chosen pair after ``select``), standardized.
    candidate_names_ : numpy.ndarray of shape (n_candidates,)
        Indices of the candidate pivots in the training set.
    pairs_ : list of tuple
        Pairs of candidates of the columns of the feature matrix.
    pivots_ : dict
        Indices in the training set of the ``descriptives``, ``discriminatives`` and ``candidates``
        pivots of the node.
    scaler_ : StandardScaler
        Scaler of the numerical features, fitted on the whole data set.

    See Also
    --------
    MultiplePivotTreeStumpClassifier, MultipleObliquePivotTreeStumpClassifier : the stumps that use it.

    Examples
    --------
    >>> from sklearn.datasets import load_iris
    >>> from RuleTree.stumps.transformers import MultiplePivotTransformer
    >>> X, y = load_iris(return_X_y=True)
    >>> transformer = MultiplePivotTransformer(tree_params={"random_state": 0}).fit(X, y, context=None, node_id=None)
    >>> len(transformer.candidate_names_), len(transformer.pairs_)  # pairs of pivots of different classes
    (6, 12)
    """

    def __init__(self, ml_task=MODEL_TYPE_CLF, distance_measure='euclidean', k=None, reg_contrast_lambda=.2,
                 reg_max_thresholds=None, tree_params=None, discriminative='tree', max_oblique_features=2, tau=1e-4,
                 n_orientations=10):
        """Initialize the MultiplePivotTransformer; see the class documentation for the parameters."""
        super().__init__(ml_task=ml_task, distance_measure=distance_measure, k=k,
                         reg_contrast_lambda=reg_contrast_lambda, reg_max_thresholds=reg_max_thresholds,
                         tree_params=tree_params)
        self.discriminative = discriminative
        self.max_oblique_features = max_oblique_features
        self.tau = tau
        self.n_orientations = n_orientations

    def _selection(self):
        """A new PivotSelection with the parameters of the transformer, ``discriminative`` included."""
        return PivotSelection(self.ml_task, k=self.k, reg_contrast_lambda=self.reg_contrast_lambda,
                              reg_max_thresholds=self.reg_max_thresholds, discriminative=self.discriminative,
                              max_oblique_features=self.max_oblique_features, tau=self.tau,
                              n_orientations=self.n_orientations, **(self.tree_params or {}))

    def fit(self, X, y, idx=None, data_type=None, sample_weight=None, *, context, node_id):
        """
        Select the candidate pivots of the node (see PivotTransformer.fit) and the pairs to compare.

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
        self : MultiplePivotTransformer

        Raises
        ------
        NoSplitFoundWarning
            If the data have no numerical feature or there is no pair of pivots to compare.
        """
        super().fit(X, y, idx, data_type=data_type, sample_weight=sample_weight, context=context,
                    node_id=node_id)

        self.pairs_ = [(a, b) for a, b in itertools.combinations(range(len(self.candidate_names_)), 2)
                       if self.candidate_names_[a] != self.candidate_names_[b]
                       and (self.ml_task != MODEL_TYPE_CLF or self.candidate_labels_[a] != self.candidate_labels_[b])]
        if not self.pairs_:
            raise NoSplitFoundWarning(f"{self.__class__.__name__}: no pair of pivots to compare")
        self.numerical_, self.categorical_ = list(range(len(self.pairs_))), []

        return self

    def transform(self, X, context=None, node_id=None):
        """
        Pivot comparisons of each instance for each pair of candidates (for the chosen pair, after select).

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
        X_transformed : numpy.ndarray of shape (n_samples, n_pairs)
            1 if the instance is closer to the second pivot of the pair, 0 otherwise.
        """
        return closer_to_second(super().transform(X, context=context, node_id=node_id), self.pairs_)

    def select(self, feature):
        """
        Keep only the two pivots of the chosen pair.

        Parameters
        ----------
        feature : int
            Column chosen by the splitter.

        Returns
        -------
        feature : int
            Always 0: the comparison of the only pair kept.
        """
        pair = list(self.pairs_[feature])
        self.candidates_ = self.candidates_[pair]
        self.candidate_names_ = self.candidate_names_[pair]
        self.candidate_labels_ = self.candidate_labels_[pair]
        self.pairs_ = [(0, 1)]
        self.numerical_, self.categorical_ = [0], []
        return 0

    def feature_id(self, feature):
        """
        Identifier of a pair: the indices of its two pivots in the training set.

        Parameters
        ----------
        feature : int
            Column of the feature matrix.

        Returns
        -------
        feature_id : tuple of int
        """
        a, b = self.pairs_[feature]
        return int(self.candidate_names_[a]), int(self.candidate_names_[b])

    def get_state(self):
        """
        Fitted state: the two pivots and their indices, the candidates, the numerical features and the
        scaler.

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
            'pairs': [[int(a), int(b)] for a, b in self.pairs_],
            'pivots': self.pivots_,
            'scaler': scaler_state(self.scaler_),
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
        self : MultiplePivotTransformer
        """
        self.num_pre_transformed_ = state['num_pre_transformed']
        self.cat_pre_transformed_ = state['cat_pre_transformed']
        self.candidates_ = np.array(state['candidates'])
        self.candidate_names_ = np.array(state['candidate_names'])
        self.pairs_ = [tuple(pair) for pair in state['pairs']]
        self.pivots_ = state['pivots']
        self.scaler_ = scaler_from_state(state['scaler'])
        self.numerical_, self.categorical_ = list(range(len(self.pairs_))), []
        return self
