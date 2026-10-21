"""ObliquePivotTreeStumpClassifier: a threshold on a linear combination of distances from pivots."""

from RuleTree.stumps.ObliqueTreeStump import check_oblique_split_type
from RuleTree.stumps.PivotTreeStump import ObliquePivotTreeStump
from RuleTree.stumps.classification.DecisionTreeStumpClassifier import DecisionTreeStumpClassifier


class ObliquePivotTreeStumpClassifier(ObliquePivotTreeStump, DecisionTreeStumpClassifier):
    """
    Oblique pivot stump for classification of tabular data: a threshold on a linear combination of the
    distances from a few pivots (``0.7 * P_12 - 0.3 * P_57 <= t`` goes left, where ``P_i`` is the
    distance from the training instance ``i``).

    At each node, the candidate pivots are, for each class, its medoid and the instances combined by an
    oblique direction on the distances (``oblique_split_type``); then the distances of the instances
    of the node from the candidates are combined by oblique directions of the same type, and the split
    is the best threshold on one of them (see
    :class:`~RuleTree.stumps.transformers.ObliquePivotTransformer.ObliquePivotTransformer`). Distances
    are computed on the standardized numerical features.

    It is a :class:`~RuleTree.stumps.Stump.Stump` made of an ObliquePivotTransformer and a
    ClassificationSplitter.

    Parameters
    ----------
    distance_measure : str or callable, default='euclidean'
        Distance between the instances, computed on the standardized numerical features: any metric
        of ``sklearn.metrics.pairwise_distances``.
    oblique_split_type : {'householder', 'bivariate'}, default='householder'
        How the oblique directions are chosen (see ObliqueDecisionTreeStumpClassifier).
    max_oblique_features : int, default=2
        Householder only: maximum number of pivots combined by a direction.
    tau : float, default=1e-4
        Householder only: if the principal component is closer than tau to every axis, the
        directions are the axes.
    n_orientations : int, default=10
        Bivariate only: number of angles, evenly spaced in [0, 180) degrees.
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
    transformer_ : ObliquePivotTransformer
        The fitted transformer.
    splitter_ : ClassificationSplitter
        The fitted splitter (feature, threshold, gain and impurity of the split).
    data_type_ : str
        Data type the stump was fitted on.
    feature_original : tuple of int
        Indices (in the training set) of the pivots combined by the split.
    coefficients : list of float
        Their coefficients.

    See Also
    --------
    PivotTreeStumpClassifier : a threshold on the distance from a single pivot.
    ObliqueDecisionTreeStumpClassifier : oblique splits on the features.

    Examples
    --------
    >>> from sklearn.datasets import load_iris
    >>> from RuleTree import RuleTreeClassifier
    >>> from RuleTree.stumps.classification import ObliquePivotTreeStumpClassifier
    >>> X, y = load_iris(return_X_y=True)
    >>> stump = ObliquePivotTreeStumpClassifier(random_state=0).fit(X, y, context=None, node_id=None)
    >>> stump.get_rule()["textual_rule"]
    '-0.447 * P_22 + 0.468 * P_118 <= 0.955'
    >>> model = RuleTreeClassifier(max_depth=2, base_stumps=ObliquePivotTreeStumpClassifier("cosine")).fit(X, y)
    """

    def __init__(self, distance_measure='euclidean', oblique_split_type='householder', max_oblique_features=2,
                 tau=1e-4, n_orientations=10,
                 criterion='gini', min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=None,
                 random_state=None, min_impurity_decrease=0.0, class_weight=None):
        """Initialize the ObliquePivotTreeStumpClassifier; see the class documentation for the parameters."""
        check_oblique_split_type(oblique_split_type)

        self.distance_measure = distance_measure
        self.oblique_split_type = oblique_split_type
        self.max_oblique_features = max_oblique_features
        self.tau = tau
        self.n_orientations = n_orientations
        self.criterion = criterion
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_features = max_features
        self.random_state = random_state
        self.min_impurity_decrease = min_impurity_decrease
        self.class_weight = class_weight
