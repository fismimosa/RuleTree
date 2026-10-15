"""PivotTreeStumpClassifier: the split on the distance from a pivot instance, for classification."""

from RuleTree.stumps.classification.DecisionTreeStumpClassifier import DecisionTreeStumpClassifier
from RuleTree.stumps.PivotTreeStump import PivotTreeStump


class PivotTreeStumpClassifier(PivotTreeStump, DecisionTreeStumpClassifier):
    """
    Pivot stump for classification: a threshold on the distance from a *pivot*, an instance of the
    training set (``distance <= t`` goes left).

    At each node, the candidate pivots are selected among the instances of the node: for each class,
    its medoid (descriptive pivot) and the instance whose distance best separates the classes
    (discriminative pivot). The split is the best threshold on the distance from one of the
    candidates. Distances are computed on the standardized numerical features; the distance matrix
    of the training set is computed once and cached in the tree. In the rules, ``P_<i>`` is the
    distance from the training instance ``i``.

    It is a :class:`~RuleTree.stumps.Stump.Stump` made of a PivotTransformer and a
    ClassificationSplitter; it supports only tabular data.

    Parameters
    ----------
    distance_measure : str or callable, default='euclidean'
        Distance between the instances, computed on the standardized numerical features: any metric
        of ``sklearn.metrics.pairwise_distances``.
    k : int, default=None
        Regression only: number of nearest and farthest instances used to measure how well a
        candidate pivot contrasts close and distant instances. None for the square root of the
        number of instances of the node (between 5 and n - 1).
    reg_contrast_lambda : float, default=0.2
        Regression only: weight of the contrast with respect to the gain of the split when the
        descriptive pivot is selected.
    reg_max_thresholds : int, default=None
        Regression only: maximum number of thresholds tested on the distances from each candidate
        pivot (quantiles). None for ``max(256, sqrt(n))``.
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
    transformer_ : PivotTransformer
        The fitted transformer.
    splitter_ : ClassificationSplitter
        The fitted splitter (feature, threshold, gain and impurity of the split).
    data_type_ : str
        Data type the stump was fitted on.
    feature_original : int
        Index (in the training set) of the pivot of the split.
    threshold_original : float
        Threshold of the split.
    X_split_instance : numpy.ndarray
        The pivot, as standardized numerical features.

    See Also
    --------
    PivotTreeStumpRegressor : the same stump for regression.
    PivotTransformer : the selection of the pivots and the distances.
    PivotTreeStump : the methods shared by the pivot stumps, e.g. ``get_pivots``.

    Notes
    -----
    The parameters of the split (criterion, min_samples_leaf, ...) are also used by the decision trees
    that select the discriminative pivots.

    Examples
    --------
    >>> from sklearn.datasets import load_iris
    >>> from RuleTree import RuleTreeClassifier
    >>> from RuleTree.stumps.classification import PivotTreeStumpClassifier
    >>> X, y = load_iris(return_X_y=True)
    >>> stump = PivotTreeStumpClassifier(random_state=0).fit(X, y)
    >>> stump.get_rule()["textual_rule"]  # distance from the training instance 112
    'P_112 <= 3.692'
    >>> model = RuleTreeClassifier(max_depth=2, base_stumps=PivotTreeStumpClassifier(distance_measure="cosine"))
    >>> model = model.fit(X, y)
    """

    def __init__(self, distance_measure='euclidean', k=None, reg_contrast_lambda=.2, reg_max_thresholds=None,
                 criterion='gini', min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=None,
                 random_state=None, min_impurity_decrease=0.0, class_weight=None):
        """Initialize the PivotTreeStumpClassifier; see the class documentation for the parameters."""
        self.distance_measure = distance_measure
        self.k = k
        self.reg_contrast_lambda = reg_contrast_lambda
        self.reg_max_thresholds = reg_max_thresholds
        self.criterion = criterion
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_features = max_features
        self.random_state = random_state
        self.min_impurity_decrease = min_impurity_decrease
        self.class_weight = class_weight
