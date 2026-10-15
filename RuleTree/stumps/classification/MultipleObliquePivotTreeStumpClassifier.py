"""MultipleObliquePivotTreeStumpClassifier: which of two pivots, found by oblique splits, is closer."""

from RuleTree.stumps.ObliqueTreeStump import check_oblique_split_type
from RuleTree.stumps.PivotTreeStump import MultiplePivotTreeStump
from RuleTree.stumps.classification.DecisionTreeStumpClassifier import DecisionTreeStumpClassifier


class MultipleObliquePivotTreeStumpClassifier(MultiplePivotTreeStump, DecisionTreeStumpClassifier):
    """
    Multiple oblique pivot stump for classification of tabular data: as
    :class:`~RuleTree.stumps.classification.MultiplePivotTreeStumpClassifier.MultiplePivotTreeStumpClassifier`
    (the instances closer to one pivot than to another one go left, ``P_a <= P_b``), but the
    discriminative pivots of each class are the instances combined by an oblique direction on the
    distances (``oblique_split_type``), instead of the root split of a decision tree: up to
    ``max_oblique_features`` (Householder) or two (bivariate) for each class.

    It is a :class:`~RuleTree.stumps.Stump.Stump` made of a MultiplePivotTransformer and a
    ClassificationSplitter.

    Parameters
    ----------
    distance_measure : str or callable, default='euclidean'
        Distance between the instances, computed on the standardized numerical features: any metric
        of ``sklearn.metrics.pairwise_distances``.
    oblique_split_type : {'householder', 'bivariate'}, default='householder'
        How the oblique directions that find the discriminative pivots are chosen (see
        ObliqueDecisionTreeStumpClassifier).
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
    transformer_ : MultiplePivotTransformer
        The fitted transformer.
    splitter_ : ClassificationSplitter
        The fitted splitter (feature, threshold, gain and impurity of the split).
    data_type_ : str
        Data type the stump was fitted on.
    feature_original : tuple of int
        Indices (in the training set) of the two pivots of the split.

    See Also
    --------
    MultiplePivotTreeStumpClassifier : the discriminative pivots found by a decision tree.
    ObliquePivotTreeStumpClassifier : an oblique split on the distances from the pivots.

    Examples
    --------
    >>> from sklearn.datasets import load_iris
    >>> from RuleTree import RuleTreeClassifier
    >>> from RuleTree.stumps.classification import MultipleObliquePivotTreeStumpClassifier
    >>> X, y = load_iris(return_X_y=True)
    >>> stump = MultipleObliquePivotTreeStumpClassifier(random_state=0).fit(X, y)
    >>> stump.get_pivots()["discriminatives"]  # up to 2 for each class
    [15, 22, 83, 98, 106, 118]
    >>> model = RuleTreeClassifier(max_depth=2, base_stumps=MultipleObliquePivotTreeStumpClassifier()).fit(X, y)
    """

    def __init__(self, distance_measure='euclidean', oblique_split_type='householder', max_oblique_features=2,
                 tau=1e-4, n_orientations=10,
                 criterion='gini', min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=None,
                 random_state=None, min_impurity_decrease=0.0, class_weight=None):
        """Initialize the MultipleObliquePivotTreeStumpClassifier; see the class documentation for the parameters."""
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

    def _selection_params(self):
        """Parameters of the selection of the pivots: discriminative pivots found by oblique splits."""
        return dict(discriminative=self.oblique_split_type, max_oblique_features=self.max_oblique_features,
                    tau=self.tau, n_orientations=self.n_orientations)
