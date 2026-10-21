"""PartialProximityTransformer: which of two partial pivots is closer to each instance."""

import itertools

import numpy as np

from RuleTree.stumps.transformers.PartialPivotTransformer import PartialPivotTransformer
from RuleTree.stumps.transformers.ProximityTransformer import ProximityComparisons


class PartialProximityTransformer(ProximityComparisons, PartialPivotTransformer):
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
    >>> transformer = PartialProximityTransformer(n_shapelets=4, random_state=0).fit(X, y, context=None, node_id=None)
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

    def fit(self, X, y, idx=None, data_type=None, sample_weight=None, *, context, node_id):
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
        sample_weight : numpy.ndarray, default=None
            Not supported: ignored with a warning.
        context : RuleTree or None
            Tree being built, None outside a tree.
        node_id : str or None
            Identifier of the node, e.g. ``"Rl"`` (``context.get_node_by_id(node_id)``), None outside a tree.

        Returns
        -------
        self : PartialProximityTransformer

        Raises
        ------
        NoSplitFoundWarning
            If there is no pair of pivots to compare.
        """
        super().fit(X, y, idx, data_type=data_type, sample_weight=sample_weight, context=context,
                    node_id=node_id)

        used = ~np.isnan(np.asarray(self.st_.shapelets, dtype=float))  # features used by each pivot
        self._set_pairs([(i, j) for i, j in itertools.combinations(range(len(used)), 2)
                         if not self.proximity_on_same_features or np.array_equal(used[i], used[j])],
                        f"{self.__class__.__name__}: no pair of partial pivots to compare")

        return self

    def _keep_references(self, indices):
        """
        Keep only some partial pivots.

        Parameters
        ----------
        indices : list of int
            Indices of the partial pivots to keep, in order.
        """
        self.st_._optimize_memory(np.array(indices))
