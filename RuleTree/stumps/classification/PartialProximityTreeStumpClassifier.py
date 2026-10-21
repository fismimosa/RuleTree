"""PartialProximityTreeStumpClassifier: which of two partial pivots is closer to the instance, for classification."""

from RuleTree.stumps.ProximityTreeStump import PartialProximityTreeStump
from RuleTree.stumps.classification.PartialPivotTreeStumpClassifier import PartialPivotTreeStumpClassifier


class PartialProximityTreeStumpClassifier(PartialProximityTreeStump, PartialPivotTreeStumpClassifier):
    """
    Partial proximity stump for classification of tabular data: the instances closer to one *partial
    pivot* than to another one go left (``euclidean(x, SP0(...)) <= euclidean(x, SP1(...))``).

    At each node, ``n_shapelets`` partial pivots (instances of the node restricted to a few features)
    are selected as in
    :class:`~RuleTree.stumps.classification.PartialPivotTreeStumpClassifier.PartialPivotTreeStumpClassifier`,
    and the split is the pair of pivots whose comparison best splits the node. In the rules,
    ``SP<i>(...)`` lists the features and values of the pivot.

    It is a :class:`~RuleTree.stumps.Stump.Stump` made of a PartialProximityTransformer and a
    ClassificationSplitter.

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
    proximity_on_same_features : bool, default=True
        Whether to compare only the partial pivots on the same features, whose distances are
        comparable.
    use_combination : bool, default=True
        With the strategies based on feature importances, whether to use all the combinations of the
        important features or only the set of all of them.
    random_state : int, default=42
        Seed of the random choices.
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
    transformer_ : PartialProximityTransformer
        The fitted transformer.
    splitter_ : ClassificationSplitter
        The fitted splitter (feature, threshold, gain and impurity of the split).
    data_type_ : str
        Data type the stump was fitted on.
    st : TabularShapelets
        The partial pivots of the split: ``st.shapelets[0]`` and ``st.shapelets[1]``.

    See Also
    --------
    PartialPivotTreeStumpClassifier : the distance from a single partial pivot.
    ProximityTreeStumpClassifier : the same comparison with shapelets, for time series.

    Examples
    --------
    >>> from sklearn.datasets import load_iris
    >>> from RuleTree import RuleTreeClassifier
    >>> from RuleTree.stumps.classification import PartialProximityTreeStumpClassifier
    >>> X, y = load_iris(return_X_y=True)
    >>> stump = PartialProximityTreeStumpClassifier(n_features_strategy=1, random_state=0)
    >>> stump = stump.fit(X, y, context=None, node_id=None, cache=None)
    >>> stump.get_rule(columns_names=["sl", "sw", "pl", "pw"])["textual_rule"]
    'euclidean(x, SP0(d: pw=1.8)) <= euclidean(x, SP1(d: pw=0.1))'
    >>> model = RuleTreeClassifier(max_depth=3, base_stumps=PartialProximityTreeStumpClassifier()).fit(X, y)
    """

    def __init__(self,
                 n_shapelets=8,
                 n_ts_for_selection=100,  # int, inf
                 n_features_strategy=2,
                 selection='random',  # random, cluster, all
                 distance='euclidean',
                 proximity_on_same_features=True,
                 use_combination=True,
                 random_state=42, n_jobs=1,
                 criterion='gini', min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=None,
                 min_impurity_decrease=0.0, class_weight=None):
        """Initialize the PartialProximityTreeStumpClassifier; see the class documentation for the parameters."""
        super().__init__(n_shapelets=n_shapelets, n_ts_for_selection=n_ts_for_selection,
                         n_features_strategy=n_features_strategy, selection=selection, distance=distance,
                         use_combination=use_combination, random_state=random_state, n_jobs=n_jobs,
                         criterion=criterion, min_samples_leaf=min_samples_leaf,
                         min_weight_fraction_leaf=min_weight_fraction_leaf, max_features=max_features,
                         min_impurity_decrease=min_impurity_decrease, class_weight=class_weight)
        self.proximity_on_same_features = proximity_on_same_features
