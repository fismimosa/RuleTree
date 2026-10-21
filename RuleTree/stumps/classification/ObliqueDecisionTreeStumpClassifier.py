"""ObliqueDecisionTreeStumpClassifier: a threshold on a linear combination of features, for classification."""

from RuleTree.stumps.classification.DecisionTreeStumpClassifier import DecisionTreeStumpClassifier
from RuleTree.stumps.ObliqueTreeStump import ObliqueTreeStump, check_oblique_split_type


class ObliqueDecisionTreeStumpClassifier(ObliqueTreeStump, DecisionTreeStumpClassifier):
    """
    Oblique stump for classification of tabular data: a threshold on a linear combination of a few
    numerical features (``0.8 * X_2 - 0.6 * X_3 <= t`` goes left), i.e. a hyperplane that is not
    parallel to the axes.

    The directions are chosen at each node in one of two ways (``oblique_split_type``):

    - ``'householder'``: the axes reflected by the Householder reflection that maps the principal
      component of the node onto an axis; each direction combines at most ``max_oblique_features``
      features (see :class:`~RuleTree.stumps.transformers.HouseholderTransformer.HouseholderTransformer`);
    - ``'bivariate'``: ``n_orientations`` rotations of the best pair of features (see
      :class:`~RuleTree.stumps.transformers.BivariateTransformer.BivariateTransformer`).

    The combinations are computed on the numerical features as they are: scale the data beforehand
    if the features have different scales. It is a :class:`~RuleTree.stumps.Stump.Stump` made of
    an oblique transformer and a ClassificationSplitter.

    Parameters
    ----------
    oblique_split_type : {'householder', 'bivariate'}, default='householder'
        How the oblique directions are chosen.
    max_oblique_features : int, default=2
        Householder only: maximum number of features in the reflection vector.
    tau : float, default=1e-4
        Householder only: if the principal component is closer than tau to every axis, the
        directions are the axes.
    n_orientations : int, default=10
        Bivariate only: number of angles, evenly spaced in [0, 180) degrees.
    criterion : {'gini', 'entropy'}, default='gini'
        Impurity measure.
    min_samples_leaf : int or float, default=1
        Minimum number of instances in each child (a fraction of the instances of the node if
        float).
    min_weight_fraction_leaf : float, default=0.0
        Minimum fraction of the sample weights of the node in each child.
    max_features : int, float, {'sqrt', 'log2'} or None, default=None
        Number of directions considered for the split, drawn at random; None for all.
    random_state : int, RandomState instance or None, default=None
        Seed of the PCA (Householder), of the choice of the directions (``max_features``) and of the
        ties between directions with the same gain. With None, a tree gives the stump its own
        ``random_state``.
    min_impurity_decrease : float, default=0.0
        A split is kept only if it decreases the weighted impurity by at least this value.
    class_weight : dict, 'balanced' or None, default=None
        Weights of the classes in the impurity; 'balanced' weighs them inversely to their frequency.

    Attributes
    ----------
    transformer_ : HouseholderTransformer or BivariateTransformer
        The fitted transformer.
    splitter_ : ClassificationSplitter
        The fitted splitter (feature, threshold, gain and impurity of the split).
    data_type_ : str
        Data type the stump was fitted on.
    feature_original : tuple of int
        Columns of the data combined by the split.
    coefficients : list of float
        Their coefficients.
    threshold_original : float
        Threshold of the split.

    See Also
    --------
    ObliqueDecisionTreeStumpRegressor : the same stump for regression.
    DecisionTreeStumpClassifier : the axis-parallel splits.

    Examples
    --------
    >>> from sklearn.datasets import load_iris
    >>> from RuleTree import RuleTreeClassifier
    >>> from RuleTree.stumps.classification import ObliqueDecisionTreeStumpClassifier
    >>> X, y = load_iris(return_X_y=True)
    >>> stump = ObliqueDecisionTreeStumpClassifier(random_state=0).fit(X, y, context=None, node_id=None, cache=None)
    >>> stump.get_rule(columns_names=["sl", "sw", "pl", "pw"])["textual_rule"]
    '-0.085 * sw + 0.857 * pl <= 1.85'
    >>> stump.feature_original, [round(c, 3) for c in stump.coefficients]
    ((1, 2), [-0.085, 0.857])
    >>> model = RuleTreeClassifier(max_depth=2, base_stumps=ObliqueDecisionTreeStumpClassifier("bivariate"))
    >>> model = model.fit(X, y)
    """

    def __init__(self, oblique_split_type='householder', max_oblique_features=2, tau=1e-4, n_orientations=10,
                 criterion='gini', min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=None,
                 random_state=None, min_impurity_decrease=0.0, class_weight=None):
        """Initialize the ObliqueDecisionTreeStumpClassifier; see the class documentation for the parameters."""
        check_oblique_split_type(oblique_split_type)

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
