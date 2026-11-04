"""ProximityPivotTreeStumpClassifier: which of two pivots is closer to the instance, for classification."""

from RuleTree.stumps.PivotTreeStump import ProximityPivotTreeStump
from RuleTree.stumps.classification.DecisionTreeStumpClassifier import DecisionTreeStumpClassifier


class ProximityPivotTreeStumpClassifier(ProximityPivotTreeStump, DecisionTreeStumpClassifier):
    """
    Proximity pivot stump for classification of tabular data, time series and images: the instances
    closer to one *pivot* than to another one go left (``P_a <= P_b``, where ``P_i`` is the distance
    from the training instance ``i``).

    At each node, the candidate pivots are selected as in
    :class:`~RuleTree.stumps.classification.PivotTreeStumpClassifier.PivotTreeStumpClassifier` (for each
    class, its medoid and its most discriminative instance), and the split is the pair of pivots of
    different classes whose comparison best splits the node. Distances are computed on the
    standardized features (categorical ones one-hot encoded); the distance matrix of the training set is
    computed once and cached in the tree.

    It is a :class:`~RuleTree.stumps.Stump.Stump` made of a ProximityPivotTransformer and a
    ClassificationSplitter.

    Parameters
    ----------
    distance_measure : str or callable, default='euclidean'
        Distance between the instances, computed on the standardized features (categorical ones one-hot
        encoded): any metric of ``sklearn.metrics.pairwise_distances``.
    standardize : bool, default=True
        Whether the features (or the embeddings) are standardized (z-score) before the distances, so that
        each of them weighs the same.
    embedder : transformer, default=None
        Transformer of the data into vectors, e.g. ``RuleTree.utils.embeddings.ImageEmbedder`` or
        ``TextEmbedder`` (with it, also texts are supported): the distances (``distance_measure``) are
        computed on the standardized embeddings. Pretrained or stateless (see PivotTransformer). If None,
        time series and images are flattened.
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
    random_state : int, RandomState instance or None, default=None
        Seed of the choice of the features (``max_features``) and of the ties between features with
        the same gain. With None, a tree gives the stump its own ``random_state``.
    min_impurity_decrease : float, default=0.0
        A numerical split is kept only if it decreases the weighted impurity by at least this value.
    class_weight : dict, 'balanced' or None, default=None
        Weights of the classes in the impurity; 'balanced' weighs them inversely to their frequency.

    Attributes
    ----------
    transformer_ : ProximityPivotTransformer
        The fitted transformer.
    splitter_ : ClassificationSplitter
        The fitted splitter (feature, threshold, gain and impurity of the split).
    data_type_ : str
        Data type the stump was fitted on.
    feature_original : tuple of int
        Indices (in the training set) of the two pivots of the split.

    See Also
    --------
    PivotTreeStumpClassifier : a threshold on the distance from a single pivot.
    ObliqueProximityPivotTreeStumpClassifier : the same comparison, with pivots found by oblique splits.

    Examples
    --------
    >>> from sklearn.datasets import load_iris
    >>> from RuleTree import RuleTreeClassifier
    >>> from RuleTree.stumps.classification import ProximityPivotTreeStumpClassifier
    >>> X, y = load_iris(return_X_y=True)
    >>> stump = ProximityPivotTreeStumpClassifier(random_state=0).fit(X, y, context=None, node_id=None, cache=None)
    >>> stump.get_rule()["textual_rule"]  # closer to the training instance a than to b
    'P_27 <= P_76'
    >>> model = RuleTreeClassifier(max_depth=2, base_stumps=ProximityPivotTreeStumpClassifier()).fit(X, y)
    """

    def __init__(self, distance_measure='euclidean',
                 criterion='gini', min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=None,
                 random_state=None, min_impurity_decrease=0.0, class_weight=None, standardize=True, embedder=None):
        """Initialize the ProximityPivotTreeStumpClassifier; see the class documentation for the parameters."""
        self.standardize = standardize
        self.embedder = embedder
        self.distance_measure = distance_measure
        self.criterion = criterion
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_features = max_features
        self.random_state = random_state
        self.min_impurity_decrease = min_impurity_decrease
        self.class_weight = class_weight
