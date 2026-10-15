"""ProximityTreeStumpClassifier: which of two shapelets is closer to the time series, for classification."""

from RuleTree.stumps.ProximityTreeStump import ProximityTreeStump
from RuleTree.stumps.classification.ShapeletTreeStumpClassifier import ShapeletTreeStumpClassifier


class ProximityTreeStumpClassifier(ProximityTreeStump, ShapeletTreeStumpClassifier):
    """
    Proximity stump for classification of time series: the time series closer to one shapelet than to
    another one go left (``euclidean(TS, green_shp) <= euclidean(TS, red_shp)``).

    At each node, ``n_shapelets`` shapelets are selected as in
    :class:`~RuleTree.stumps.classification.ShapeletTreeStumpClassifier.ShapeletTreeStumpClassifier`, and the split
    is the pair of shapelets whose comparison best splits the node. Unlike the shapelet stump, the
    rule has no threshold: it only says which shapelet the time series resembles more. In the
    Graphviz drawing, the first shapelet is green and the second one red.

    It is a :class:`~RuleTree.stumps.Stump.Stump` made of a ProximityTransformer and a
    ClassificationSplitter. The time series have shape (n_samples, n_signals, length) and are passed to the
    trees as ``X_ts``.

    Parameters
    ----------
    n_shapelets : int, default=8
        Number of shapelets selected at each node (at least 2).
    n_shapelets_for_selection : int, numpy.inf or 'stratified', default=500
        Number of candidate subsequences extracted from the time series, among which the shapelets
        are selected. 'stratified' draws them according to the class distribution.
    n_ts_for_selection_per_class : int or numpy.inf, default=100
        Number of time series of each class from which the candidates are extracted.
    sliding_window : int, default=50
        Length of the shapelets.
    selection : {'random', 'mi_clf', 'cluster'}, default='mi_clf'
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
    criterion : {'gini', 'entropy'}, default='gini'
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
    class_weight : dict, 'balanced' or None, default=None
        Weights of the classes in the impurity; 'balanced' weighs them inversely to their frequency.

    Attributes
    ----------
    transformer_ : ProximityTransformer
        The fitted transformer.
    splitter_ : ClassificationSplitter
        The fitted splitter (feature, threshold, gain and impurity of the split).
    data_type_ : str
        Data type the stump was fitted on.
    st : Shapelets
        The shapelets of the split: ``st.shapelets[0]`` (green) and ``st.shapelets[1]`` (red).

    See Also
    --------
    ProximityTreeStumpRegressor : the same stump for regression.
    ShapeletTreeStumpClassifier : the distance from a single shapelet.

    Examples
    --------
    >>> import numpy as np
    >>> from RuleTree import RuleTreeClassifier
    >>> from RuleTree.stumps.classification import ProximityTreeStumpClassifier
    >>> rng = np.random.default_rng(0)
    >>> X_ts = rng.normal(scale=0.3, size=(60, 1, 30))  # 60 time series of length 30
    >>> y = rng.integers(0, 2, size=60)
    >>> X_ts[y == 1, 0, 10:20] += 2  # a bump for the class 1
    >>> stump = ProximityTreeStumpClassifier(n_shapelets=4, sliding_window=10, selection="random")
    >>> model = RuleTreeClassifier(max_depth=1, base_stumps=stump).fit(X_ts=X_ts, y=y)
    >>> model.get_rules()["textual_rule"]
    'euclidean(TS, green_shp) <= euclidean(TS, red_shp)'
    """
