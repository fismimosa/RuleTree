"""ShapeletTreeStumpRegressor: the split on the distance from a shapelet, for regression of time series."""

from typing import Optional


from RuleTree.stumps.classification.ShapeletTreeStumpClassifier import ShapeletTreeStumpClassifier
from RuleTree.stumps.regression.DecisionTreeStumpRegressor import DecisionTreeStumpRegressor
from RuleTree.stumps.transformers.ShapeletTransformer import ShapeletTransformer


class ShapeletTreeStumpRegressor(DecisionTreeStumpRegressor):
    """
    Shapelet stump for regression of time series: a threshold on the distance of the time series from a
    *shapelet* (``distance <= t`` goes left).

    A shapelet is a subsequence of one of the time series of the node; the distance of a time series
    from it is the smallest distance between the shapelet and the subsequences of the series, so it
    does not depend on where the pattern occurs. At each node, ``n_shapelets`` shapelets are selected
    among candidate subsequences, and the split is the best threshold on the distance from one of
    them. In the rules, ``euclidean(TS, shp)`` is the distance from the shapelet.

    It is a :class:`~RuleTree.stumps.Stump.Stump` made of a ShapeletTransformer and a
    RegressionSplitter. The time series have shape
    (n_samples, n_signals, length) and are passed to the trees as ``X_ts``.

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
    selection : {'random', 'mi_reg', 'cluster'}, default='mi_reg'
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
    criterion : {'squared_error', 'absolute_error', 'poisson'}, default='squared_error'
        Impurity measure.
    min_samples_leaf : int or float, default=1
        Minimum number of instances in each child (a fraction of the instances of the node if
        float). Applies to the numerical features.
    min_weight_fraction_leaf : float, default=0.0
        Minimum fraction of the sample weights of the node in each child. Applies to the numerical
        features.
    max_features : int, float, {'sqrt', 'log2'} or None, default=None
        Number of numerical features considered for the split, drawn at random; None for all.
    min_impurity_decrease : float, default=0.0
        A numerical split is kept only if it decreases the weighted impurity by at least this value.

    Attributes
    ----------
    transformer_ : ShapeletTransformer
        The fitted transformer.
    splitter_ : RegressionSplitter
        The fitted splitter (feature, threshold, gain and impurity of the split).
    data_type_ : str
        Data type the stump was fitted on.
    feature_original : int
        Index of the shapelet of the split among the selected ones.
    threshold_original : float
        Threshold of the split.

    See Also
    --------
    ShapeletTreeStumpClassifier : the same stump for classification.
    RuleTree.utils.shapelet_transform.Shapelets.Shapelets : the selection of the shapelets.

    Examples
    --------
    >>> import numpy as np
    >>> from RuleTree import RuleTreeRegressor
    >>> from RuleTree.stumps.regression import ShapeletTreeStumpRegressor
    >>> rng = np.random.default_rng(0)
    >>> X_ts = rng.normal(size=(60, 1, 30))  # 60 time series of length 30
    >>> y = X_ts[:, 0, 10:20].mean(axis=1)
    >>> stump = ShapeletTreeStumpRegressor(n_shapelets=5, sliding_window=10, selection="random")
    >>> model = RuleTreeRegressor(max_depth=2, base_stumps=stump).fit(X_ts=X_ts, y=y)
    >>> model.predict(X_ts=X_ts).shape
    (60,)
    """

    def __init__(self, n_shapelets=8,
                 n_shapelets_for_selection=500,  # int, inf, or 'stratified'
                 n_ts_for_selection_per_class=100,  # int, inf
                 sliding_window=50,
                 selection='mi_reg',  # random, mi_reg, cluster
                 distance='euclidean',
                 mi_n_neighbors=100,
                 random_state=42, n_jobs=1,
                 criterion='squared_error', min_samples_leaf=1, min_weight_fraction_leaf=0.0,
                 max_features=None, min_impurity_decrease=0.0):
        """Initialize the ShapeletTreeStumpRegressor; see the class documentation for the parameters."""
        if selection not in ['random', 'mi_reg', 'cluster']:
            raise ValueError("'selection' must be 'random', 'mi_reg' or 'cluster'")

        self.n_shapelets = n_shapelets
        self.n_shapelets_for_selection = n_shapelets_for_selection
        self.n_ts_for_selection_per_class = n_ts_for_selection_per_class
        self.sliding_window = sliding_window
        self.selection = selection
        self.distance = distance
        self.mi_n_neighbors = mi_n_neighbors
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.criterion = criterion
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_features = max_features
        self.min_impurity_decrease = min_impurity_decrease

    def _splitter_params(self):
        """
        Parameters of the splitter: as in the decision tree stump, but without random_state, which is
        the one of the shapelet selection.

        Returns
        -------
        params : dict
        """
        return super()._splitter_params() | {"random_state": None}

    def _make_transformer(self):
        """
        A new, unfitted ShapeletTransformer with the parameters of the stump.

        Returns
        -------
        transformer : ShapeletTransformer
        """
        return ShapeletTransformer(n_shapelets=self.n_shapelets,
                                   n_shapelets_for_selection=self.n_shapelets_for_selection,
                                   n_ts_for_selection_per_class=self.n_ts_for_selection_per_class,
                                   sliding_window=self.sliding_window,
                                   selection=self.selection,
                                   distance=self.distance,
                                   mi_n_neighbors=self.mi_n_neighbors,
                                   random_state=self.random_state,
                                   n_jobs=self.n_jobs,
                                   ml_task=self.task)

    @property
    def st(self):
        """The fitted shapelet transform."""
        return self.transformer_.st_

    def get_rule(self, columns_names=None, scaler=None, float_precision: Optional[int] = 3):
        """
        Rule of the split, e.g. ``euclidean(TS, shp) <= 1.2``.

        Parameters
        ----------
        columns_names : list of str, default=None
            Not used.
        scaler : TransformerMixin, default=None
            Not supported: must be None.
        float_precision : int, default=3
            Number of decimals of the threshold. If None, it is not rounded.

        Returns
        -------
        rule : dict
            See RuleTreeBaseStump.get_rule.

        Raises
        ------
        ValueError
            If a scaler is given.
        """
        return ShapeletTreeStumpClassifier.get_rule(self, columns_names, scaler, float_precision)

    def get_graphviz_rule(self, columns_names=None, scaler=None, float_precision: Optional[int] = 3):
        """
        Attributes of the Graphviz node of the split, with a plot of the shapelet (see
        ShapeletTreeStumpClassifier.get_graphviz_rule).

        Parameters
        ----------
        columns_names : list of str, default=None
            Not used.
        scaler : TransformerMixin, default=None
            Not supported: must be None.
        float_precision : int, default=3
            Number of decimals of the threshold. If None, it is not rounded.

        Returns
        -------
        attributes : dict
            Attributes of the Graphviz node.
        """
        return ShapeletTreeStumpClassifier.get_graphviz_rule(self, columns_names, scaler, float_precision)
