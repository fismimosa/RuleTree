"""PartialProximityTransformer: which of two partial pivots is closer to each instance."""

import itertools

import numpy as np

from RuleTree.exceptions import NoSplitFoundWarning
from RuleTree.stumps.transformers.PartialPivotTransformer import PartialPivotTransformer
from RuleTree.stumps.transformers.ProximityTransformer import closer_to_second


class PartialProximityTransformer(PartialPivotTransformer):
    """
    Transformer of tabular data into proximity comparisons: for each pair of partial pivots (instances
    of the node restricted to a few features), whether the instance is closer to the second one than
    to the first.

    The partial pivots are selected as in
    :class:`~RuleTree.stumps.transformers.PartialPivotTransformer.PartialPivotTransformer`; the
    feature matrix has one 0/1 column for each pair of them (see
    :func:`~RuleTree.stumps.transformers.ProximityTransformer.closer_to_second`). After the splitter
    has chosen a pair, only its two pivots are kept (``select``).

    Parameters
    ----------
    n_shapelets : int or numpy.inf, default=8
        Number of partial pivots selected at each node (numpy.inf with selection='all').
    n_ts_for_selection : int, float or numpy.inf, default=100
        Number (int) or fraction (float) of the instances of the node used to select the pivots.
    n_features_strategy : int, float, tuple or str, default=2
        Number of features of each partial pivot (see PartialPivotTransformer).
    selection : {'random', 'cluster', 'all'}, default='random'
        How the pivots are selected among the candidates.
    distance : {'euclidean', 'sqeuclidean', 'cosine', 'cityblock'} or callable, default='euclidean'
        Distance between an instance and a partial pivot, on the features of the pivot. The data are
        used as they are: scale them beforehand if the features have different scales.
    proximity_on_same_features : bool, default=True
        Whether to compare only the pivots on the same features, whose distances are comparable.
    use_combination : bool, default=True
        With the strategies based on feature importances, whether to use all the combinations of the
        important features or only the set of all of them.
    random_state : int, default=42
        Seed of the random choices.
    n_jobs : int, default=1
        Number of threads used to compute the distances (-1: all the CPUs).

    Attributes
    ----------
    numerical_ : list of int
        Numerical columns of the feature matrix.
    categorical_ : list of int
        Categorical columns of the feature matrix (none).
    st_ : TabularShapelets
        The fitted transform (only the two pivots of the chosen pair after ``select``).
    pairs_ : list of tuple
        Pairs of pivots of the columns of the feature matrix.

    See Also
    --------
    PartialProximityTreeStumpClassifier : the stump that uses it.
    ProximityTransformer : the same comparisons with shapelets, for time series.

    Examples
    --------
    >>> from sklearn.datasets import load_iris
    >>> from RuleTree.stumps.transformers import PartialProximityTransformer
    >>> X, y = load_iris(return_X_y=True)
    >>> transformer = PartialProximityTransformer(n_shapelets=4, random_state=0).fit(X, y)
    >>> transformer.transform(X).shape[1] == len(transformer.pairs_)
    True
    """

    def __init__(self,
                 n_shapelets=8,
                 n_ts_for_selection=100,  # int, inf
                 n_features_strategy=2,
                 selection='random',  # random, cluster, all
                 distance='euclidean',
                 proximity_on_same_features=True,
                 use_combination=True,
                 random_state=42, n_jobs=1):
        """Initialize the PartialProximityTransformer; see the class documentation for the parameters."""
        super().__init__(n_shapelets=n_shapelets, n_ts_for_selection=n_ts_for_selection,
                         n_features_strategy=n_features_strategy, selection=selection, distance=distance,
                         use_combination=use_combination, random_state=random_state, n_jobs=n_jobs)
        self.proximity_on_same_features = proximity_on_same_features

    def fit(self, X, y, idx=None, data_type=None, cache=None, sample_weight=None):
        """
        Select the partial pivots among the instances in the node, and the pairs to compare.

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
            Not supported: ignored with a warning.

        Returns
        -------
        self : PartialProximityTransformer

        Raises
        ------
        NoSplitFoundWarning
            If there is no pair of pivots to compare.
        """
        super().fit(X, y, idx, data_type=data_type, cache=cache, sample_weight=sample_weight)

        used = ~np.isnan(np.asarray(self.st_.shapelets, dtype=float))  # features used by each pivot
        self.pairs_ = [(i, j) for i, j in itertools.combinations(range(len(used)), 2)
                       if not self.proximity_on_same_features or np.array_equal(used[i], used[j])]
        if not self.pairs_:
            raise NoSplitFoundWarning(f"{self.__class__.__name__}: no pair of partial pivots to compare")
        self.numerical_, self.categorical_ = list(range(len(self.pairs_))), []

        return self

    def transform(self, X):
        """
        Proximity comparisons of each instance for each pair of pivots (for the chosen pair, after select).

        Parameters
        ----------
        X : numpy.ndarray
            Tabular data.

        Returns
        -------
        X_transformed : numpy.ndarray of shape (n_samples, n_pairs)
            1 if the instance is closer to the second pivot of the pair, 0 otherwise.
        """
        return closer_to_second(self.st_.transform(X), self.pairs_)

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
        self.st_._optimize_memory(np.array(self.pairs_[feature]))
        self.pairs_ = [(0, 1)]
        self.numerical_, self.categorical_ = [0], []
        return 0

    def get_state(self):
        """
        Fitted state: the partial pivots and the pairs.

        Returns
        -------
        state : dict
            JSON-friendly dictionary, restored by set_state.
        """
        return super().get_state() | {"pairs": [[int(i), int(j)] for i, j in self.pairs_]}

    def set_state(self, state):
        """
        Restore the fitted state saved by get_state.

        Parameters
        ----------
        state : dict
            Dictionary created by get_state.

        Returns
        -------
        self : PartialProximityTransformer
        """
        super().set_state(state)
        self.pairs_ = [tuple(pair) for pair in state["pairs"]]
        self.numerical_, self.categorical_ = list(range(len(self.pairs_))), []
        return self
