"""PartialPivotTransformer: distances from partial pivots (instances restricted to subsets of the features)."""

import warnings

import numpy as np

from RuleTree.base.RuleTreeBaseTransformer import RuleTreeBaseTransformer
from RuleTree.utils.define import DATA_TYPE_TABULAR
from RuleTree.utils.optional import import_optional


class PartialPivotTransformer(RuleTreeBaseTransformer):
    """
    Transformer of tabular data into the distances from partial pivots, instances of the node
    restricted to subsets of the features.

    The distance from a partial pivot is computed only on its features. After the splitter has chosen
    a column, only the chosen pivot is kept (``select``), and the splitter is fitted again on its
    distances (``refit_after_select``).

    Parameters
    ----------
    n_shapelets : int or numpy.inf, default=8
        Number of partial pivots selected at each node (numpy.inf with selection='all').
    n_ts_for_selection : int, float or numpy.inf, default=100
        Number (int) or fraction (float) of the instances of the node used to select the pivots.
    n_features_strategy : int, float, tuple or str, default=2
        Number of features of each partial pivot: an int, a fraction of the features (float), a range
        ``(min, max)``, or chosen from the feature importances of a random forest ('elbow', 'drop',
        'new') or iteratively with depth-1 trees ('iterative').
    selection : {'random', 'cluster', 'all'}, default='random'
        How the pivots are selected among the candidates: at random, the instances closest to the
        k-means centroids, or all of them.
    distance : {'euclidean', 'sqeuclidean', 'cosine', 'cityblock'} or callable, default='euclidean'
        Distance between an instance and a partial pivot, on the features of the pivot. The data are
        used as they are: scale them beforehand if the features have different scales.
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
        Categorical columns of the feature matrix.
    st_ : TabularShapelets
        The fitted transform, with the partial pivots.

    See Also
    --------
    PartialPivotTreeStumpClassifier : the stump that uses it.

    Examples
    --------
    >>> from sklearn.datasets import load_iris
    >>> from RuleTree.stumps.transformers import PartialPivotTransformer
    >>> X, y = load_iris(return_X_y=True)
    >>> transformer = PartialPivotTransformer(n_shapelets=4, random_state=0)
    >>> transformer.fit(X, y, context=None, node_id=None, cache=None).transform(X).shape
    (150, 4)
    """

    supported_data_types = (DATA_TYPE_TABULAR,)
    refit_after_select = True

    def __init__(self,
                 n_shapelets=8,
                 n_ts_for_selection=100,  # int, inf
                 n_features_strategy=2,
                 selection='random',  # random, cluster, all
                 distance='euclidean',
                 use_combination=True,
                 random_state=42, n_jobs=1):
        """Initialize the PartialPivotTransformer; see the class documentation for the parameters."""
        self.n_shapelets = n_shapelets
        self.n_ts_for_selection = n_ts_for_selection
        self.n_features_strategy = n_features_strategy
        self.selection = selection
        self.distance = distance
        self.use_combination = use_combination
        self.random_state = random_state
        self.n_jobs = n_jobs

    def _shapelets(self):
        """
        A new, unfitted TabularShapelets transform with the parameters of the transformer.

        Returns
        -------
        st : TabularShapelets

        Raises
        ------
        MissingOptionalDependencyError
            If numba is not installed.
        """
        import_optional("numba", self.__class__.__name__)
        from RuleTree.utils.shapelet_transform.TabularShapelets import TabularShapelets

        return TabularShapelets(n_shapelets=self.n_shapelets,
                                n_ts_for_selection=self.n_ts_for_selection,
                                n_features_strategy=self.n_features_strategy,
                                selection=self.selection,
                                distance=self.distance,
                                random_state=self.random_state,
                                use_combination=self.use_combination,
                                n_jobs=self.n_jobs)

    def fit(self, X, y, idx=None, data_type=None, sample_weight=None, *, context, node_id, cache):
        """
        Select the partial pivots among the instances in the node.

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
        cache : dict or None
            Cache of the tree (namespace -> dictionary of values), None outside a tree: the component
            decides whether and how to use it (the ones of the library use ``self._get_cache(cache)``).

        Returns
        -------
        self : PartialPivotTransformer

        Warns
        -----
        Warning
            If sample_weight is given: it is not supported and ignored.
        """
        if idx is None:
            idx = slice(None)

        if sample_weight is not None:
            warnings.warn(f"sample_weight is not supported for {self.__class__.__name__}", Warning)

        self.st_ = self._shapelets().fit(X[idx], y[idx])
        self.numerical_, self.categorical_ = list(range(len(self.st_.shapelets))), []

        return self

    def transform(self, X, context=None, node_id=None, cache=None):
        """
        Distance of each instance from each partial pivot (from the chosen one, after select).

        Parameters
        ----------
        X : numpy.ndarray
            Tabular data.
        context : RuleTree, default=None
            The tree, None outside a tree.
        node_id : str, default=None
            Identifier of the node of the stump, None outside a tree.
        cache : dict, default=None
            Cache of the tree, when called by it; None otherwise.

        Returns
        -------
        X_transformed : numpy.ndarray of shape (n_samples, n_pivots)
        """
        return self.st_.transform(X)

    def select(self, feature):
        """
        Keep only the chosen partial pivot.

        Parameters
        ----------
        feature : int
            Column chosen by the splitter.

        Returns
        -------
        feature : int
            Always 0: the distance from the only partial pivot kept.
        """
        self.st_._optimize_memory(np.array([feature]))
        self.numerical_, self.categorical_ = [0], []
        return 0

    def get_state(self):
        """
        Fitted state: the partial pivots.

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
        self : PartialPivotTransformer
        """
        self.st_ = self._shapelets()
        self.st_.shapelets = np.array(state["shapelets"])
        return self
