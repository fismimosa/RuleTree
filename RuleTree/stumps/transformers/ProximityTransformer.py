"""ProximityTransformer: which of two shapelets is closer to each time series."""

import itertools

import numpy as np

from RuleTree.exceptions import NoSplitFoundWarning
from RuleTree.stumps.transformers.ShapeletTransformer import ShapeletTransformer


def closer_to_second(distances: np.ndarray, pairs: list) -> np.ndarray:
    """
    For each pair ``(i, j)`` of columns of ``distances``, 1 where ``distances[:, i] > distances[:, j]``
    (the instance is closer to the second one), 0 otherwise.

    A split ``column <= t`` with ``0 <= t < 1`` sends left the instances closer to the first one
    (or at the same distance from both).

    Parameters
    ----------
    distances : numpy.ndarray of shape (n_samples, n_references)
        Distance of each instance from each reference (shapelet, pivot, ...).
    pairs : list of tuple
        Pairs of columns.

    Returns
    -------
    comparisons : numpy.ndarray of shape (n_samples, n_pairs)

    Examples
    --------
    >>> import numpy as np
    >>> from RuleTree.stumps.transformers.ProximityTransformer import closer_to_second
    >>> closer_to_second(np.array([[1.0, 2.0], [3.0, 0.5]]), [(0, 1)])
    array([[0.],
           [1.]])
    """
    pairs = np.asarray(pairs, dtype=int).reshape(-1, 2)
    return (distances[:, pairs[:, 0]] > distances[:, pairs[:, 1]]).astype(float)


class ProximityTransformer(ShapeletTransformer):
    """
    Transformer of time series into proximity comparisons: for each pair of shapelets, whether the
    time series is closer to the second one than to the first.

    The shapelets are selected as in
    :class:`~RuleTree.stumps.transformers.ShapeletTransformer.ShapeletTransformer`; the feature matrix
    has one 0/1 column for each pair of them (see :func:`closer_to_second`), so that a split sends
    left the time series closer to the first shapelet of a pair. After the splitter has chosen a
    pair, only its two shapelets are kept (``select``).

    Parameters
    ----------
    n_shapelets : int, default=8
        Number of shapelets selected at each node (at least 2).
    n_shapelets_for_selection : int, numpy.inf or 'stratified', default=500
        Number of candidate subsequences, among which the shapelets are selected.
    n_ts_for_selection_per_class : int or numpy.inf, default=100
        Number of time series of each class from which the candidates are extracted.
    sliding_window : int, default=50
        Length of the shapelets.
    selection : {'random', 'mi_clf', 'mi_reg', 'cluster'}, default='mi_clf'
        How the shapelets are selected among the candidates.
    distance : {'euclidean', 'sqeuclidean', 'cosine', 'cityblock'} or callable, default='euclidean'
        Distance between a shapelet and a subsequence.
    mi_n_neighbors : int, default=100
        Number of neighbors used to estimate the mutual information.
    random_state : int, default=42
        Seed of the random choices of the candidates and of the shapelets.
    n_jobs : int, default=1
        Number of threads used to compute the distances (-1: all the CPUs).
    ml_task : {MODEL_TYPE_CLF, MODEL_TYPE_REG}, default=MODEL_TYPE_CLF
        Task of the stump: for classification, the candidates are drawn from the time series of each
        class; for regression (also with several targets), from all the time series.

    Attributes
    ----------
    numerical_ : list of int
        Numerical columns of the feature matrix.
    categorical_ : list of int
        Categorical columns of the feature matrix (none).
    st_ : Shapelets
        The fitted shapelet transform (only the two shapelets of the chosen pair after ``select``).
    pairs_ : list of tuple
        Pairs of shapelets of the columns of the feature matrix.

    See Also
    --------
    ProximityTreeStumpClassifier, ProximityTreeStumpRegressor : the stumps that use it.
    PartialProximityTransformer : the same comparisons with partial pivots, for tabular data.

    Examples
    --------
    >>> import numpy as np
    >>> from RuleTree.stumps.transformers import ProximityTransformer
    >>> rng = np.random.default_rng(0)
    >>> X_ts, y = rng.normal(size=(20, 1, 30)), rng.integers(0, 2, size=20)
    >>> transformer = ProximityTransformer(n_shapelets=4, sliding_window=5, selection="random").fit(X_ts, y)
    >>> transformer.transform(X_ts).shape  # one column for each of the 6 pairs of shapelets
    (20, 6)
    """

    def fit(self, X, y, idx=None, data_type=None, cache=None, sample_weight=None):
        """
        Select the shapelets among the subsequences of the time series in the node.

        Parameters
        ----------
        X : numpy.ndarray
            Time series, all the instances.
        y : numpy.ndarray
            Target, all the instances.
        idx : numpy.ndarray, default=None
            Indices of the instances in the node. If None, all the instances.
        data_type : str, default=None
            Type of X, used as key of the cached values.
        cache : dict, default=None
            Values shared among the splits of the tree.
        sample_weight : numpy.ndarray, default=None
            Not supported: must be None.

        Returns
        -------
        self : ProximityTransformer

        Raises
        ------
        ValueError
            If sample_weight is given.
        NoSplitFoundWarning
            If fewer than two shapelets are selected.
        """
        super().fit(X, y, idx, data_type=data_type, cache=cache, sample_weight=sample_weight)

        self.pairs_ = list(itertools.combinations(range(len(self.st_.shapelets)), 2))
        if not self.pairs_:
            raise NoSplitFoundWarning(f"{self.__class__.__name__} needs at least two shapelets")
        self.numerical_, self.categorical_ = list(range(len(self.pairs_))), []

        return self

    def transform(self, X):
        """
        Proximity comparisons of each time series for each pair of shapelets (for the chosen pair, after
        select).

        Parameters
        ----------
        X : numpy.ndarray
            Time series, of shape (n_samples, n_signals, length).

        Returns
        -------
        X_transformed : numpy.ndarray of shape (n_samples, n_pairs)
            1 if the series is closer to the second shapelet of the pair, 0 otherwise.
        """
        return closer_to_second(self.st_.transform(X), self.pairs_)

    def select(self, feature):
        """
        Keep only the two shapelets of the chosen pair.

        Parameters
        ----------
        feature : int
            Column chosen by the splitter.

        Returns
        -------
        feature : int
            Always 0: the comparison of the only pair kept.
        """
        self.st_.shapelets = self.st_.shapelets[list(self.pairs_[feature])]
        self.pairs_ = [(0, 1)]
        self.numerical_, self.categorical_ = [0], []
        return 0

    def get_state(self):
        """
        Fitted state: the shapelets and the pairs.

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
        self : ProximityTransformer
        """
        super().set_state(state)
        self.pairs_ = [tuple(pair) for pair in state["pairs"]]
        self.numerical_, self.categorical_ = list(range(len(self.pairs_))), []
        return self
