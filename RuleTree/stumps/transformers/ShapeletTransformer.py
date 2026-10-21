"""ShapeletTransformer: distances of time series from shapelets."""

import numpy as np

from RuleTree.base.RuleTreeBaseTransformer import RuleTreeBaseTransformer
from RuleTree.utils.data_utils import check_one_target
from RuleTree.utils.define import DATA_TYPE_TS, MODEL_TYPE_CLF
from RuleTree.utils.optional import import_optional


class ShapeletTransformer(RuleTreeBaseTransformer):
    """
    Transformer of time series into their distances from shapelets, subsequences selected among the
    ones of the time series in the node.

    The distance of a time series from a shapelet is the smallest distance between the shapelet and
    the subsequences of the series of the same length. After the splitter has chosen a column, only
    the chosen shapelet is kept (``select``).

    Parameters
    ----------
    n_shapelets : int, default=8
        Number of shapelets selected at each node.
    n_shapelets_for_selection : int, numpy.inf or 'stratified', default=500
        Number of candidate subsequences extracted from the time series, among which the shapelets
        are selected. 'stratified' draws them according to the class distribution.
    n_ts_for_selection_per_class : int or numpy.inf, default=100
        Number of time series of each class from which the candidates are extracted.
    sliding_window : int, default=50
        Length of the shapelets.
    selection : {'random', 'mi_clf', 'mi_reg', 'cluster'}, default='mi_clf'
        How the shapelets are selected among the candidates: at random, by mutual information with
        the target, or as the medoids of a k-medoids clustering (requires scikit-learn-extra).
    distance : {'euclidean', 'sqeuclidean', 'cosine', 'cityblock'} or callable, default='euclidean'
        Distance between a shapelet and a subsequence; the distance of a time series from a shapelet
        is the minimum over the positions of the shapelet.
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
        Categorical columns of the feature matrix.
    st_ : Shapelets
        The fitted shapelet transform, with the selected shapelets.

    See Also
    --------
    ShapeletTreeStumpClassifier, ShapeletTreeStumpRegressor : the stumps that use it.

    Examples
    --------
    >>> import numpy as np
    >>> from RuleTree.stumps.transformers import ShapeletTransformer
    >>> rng = np.random.default_rng(0)
    >>> X_ts, y = rng.normal(size=(20, 1, 30)), rng.integers(0, 2, size=20)
    >>> transformer = ShapeletTransformer(n_shapelets=3, sliding_window=5, selection="random")
    >>> transformer = transformer.fit(X_ts, y, context=None, node_id=None)
    >>> transformer.transform(X_ts).shape
    (20, 3)
    """

    supported_data_types = (DATA_TYPE_TS,)

    def __init__(self, n_shapelets=8,
                 n_shapelets_for_selection=500,  # int, inf, or 'stratified'
                 n_ts_for_selection_per_class=100,  # int, inf
                 sliding_window=50,
                 selection='mi_clf',  # random, mi_clf, mi_reg, cluster
                 distance='euclidean',
                 mi_n_neighbors=100,
                 random_state=42, n_jobs=1, ml_task=MODEL_TYPE_CLF):
        """Initialize the ShapeletTransformer; see Shapelets for the parameters."""
        self.n_shapelets = n_shapelets
        self.n_shapelets_for_selection = n_shapelets_for_selection
        self.n_ts_for_selection_per_class = n_ts_for_selection_per_class
        self.sliding_window = sliding_window
        self.selection = selection
        self.distance = distance
        self.mi_n_neighbors = mi_n_neighbors
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.ml_task = ml_task

    def _shapelets(self):
        """
        A new, unfitted Shapelets transform with the parameters of the transformer.

        Returns
        -------
        st : Shapelets

        Raises
        ------
        MissingOptionalDependencyError
            If numba is not installed.
        """
        import_optional("numba", self.__class__.__name__)
        from RuleTree.utils.shapelet_transform.Shapelets import Shapelets

        return Shapelets(n_shapelets=self.n_shapelets,
                         n_shapelets_for_selection=self.n_shapelets_for_selection,
                         n_ts_for_selection_per_class=self.n_ts_for_selection_per_class,
                         sliding_window=self.sliding_window,
                         selection=self.selection,
                         distance=self.distance,
                         mi_n_neighbors=self.mi_n_neighbors,
                         random_state=self.random_state,
                         n_jobs=self.n_jobs,
                         ml_task=self.ml_task)

    def fit(self, X, y, idx=None, data_type=None, sample_weight=None, *, context, node_id):
        """
        Select the shapelets among the subsequences of the time series in the node.

        Parameters
        ----------
        X : numpy.ndarray
            Data of the transformer's type, all the instances.
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
        self : ShapeletTransformer

        Raises
        ------
        ValueError
            If sample_weight is given, or y has several targets with a mutual information selection.
        """
        if self.selection.startswith("mi_"):  # the mutual information is computed with a single target
            check_one_target(y, f"{self.__class__.__name__}(selection={self.selection!r})")
        if sample_weight is not None:
            raise ValueError(f"sample_weight is not supported for {self.__class__.__name__}")
        if idx is None:
            idx = slice(None)

        self.st_ = self._shapelets().fit(X[idx], y[idx])
        self.numerical_, self.categorical_ = list(range(len(self.st_.shapelets))), []

        return self

    def transform(self, X, context=None, node_id=None):
        """
        Distance of each time series from each shapelet.

        Parameters
        ----------
        X : numpy.ndarray
            Time series, of shape (n_samples, 1, length).
        context : RuleTree, default=None
            The tree, None outside a tree.
        node_id : str, default=None
            Identifier of the node of the stump, None outside a tree.

        Returns
        -------
        X_transformed : numpy.ndarray of shape (n_samples, n_shapelets)
        """
        return self.st_.transform(X)

    def get_state(self):
        """
        Fitted state: the shapelets.

        Returns
        -------
        state : dict
            JSON-friendly dictionary, restored by set_state.
        """
        return {"shapelets": self.st_.shapelets.tolist()}

    def set_state(self, state):
        """
        Restore the fitted state saved by get_state.

        Parameters
        ----------
        state : dict
            Dictionary created by get_state.

        Returns
        -------
        self : ShapeletTransformer
        """
        self.st_ = self._shapelets()
        self.st_.shapelets = np.array(state["shapelets"])
        return self
