"""ObliqueDecisionTreeStumpRegressor: a threshold on a linear combination of features, for regression."""

from RuleTree.stumps.ObliqueTreeStump import ObliqueTreeStump, check_oblique_split_type
from RuleTree.stumps.regression.DecisionTreeStumpRegressor import DecisionTreeStumpRegressor


class ObliqueDecisionTreeStumpRegressor(ObliqueTreeStump, DecisionTreeStumpRegressor):
    """
    Oblique stump for regression of tabular data: a threshold on a linear combination of a few
    numerical features (``0.8 * X_2 - 0.6 * X_3 <= t`` goes left).

    The directions are chosen as in
    :class:`~RuleTree.stumps.classification.ObliqueDecisionTreeStumpClassifier.ObliqueDecisionTreeStumpClassifier`
    (``'householder'`` or ``'bivariate'``). As a regression stump, it can also be used by the
    clustering trees. It is a :class:`~RuleTree.stumps.Stump.Stump` made of an oblique transformer
    and a RegressionSplitter.

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
    criterion : {'squared_error', 'absolute_error', 'poisson'}, default='squared_error'
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

    Attributes
    ----------
    transformer_ : HouseholderTransformer or BivariateTransformer
        The fitted transformer.
    splitter_ : RegressionSplitter
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
    ObliqueDecisionTreeStumpClassifier : the same stump for classification.
    DecisionTreeStumpRegressor : the axis-parallel splits.

    Examples
    --------
    >>> from sklearn.datasets import load_diabetes
    >>> from RuleTree import RuleTreeRegressor
    >>> from RuleTree.stumps.regression import ObliqueDecisionTreeStumpRegressor
    >>> X, y = load_diabetes(return_X_y=True)
    >>> model = RuleTreeRegressor(max_depth=2, random_state=0,
    ...                           base_stumps=ObliqueDecisionTreeStumpRegressor("bivariate")).fit(X, y)
    >>> model.get_rules()["textual_rule"]
    '0.588 * X_2 + 0.809 * X_8 <= 0.005'
    """

    def __init__(self, oblique_split_type='householder', max_oblique_features=2, tau=1e-4, n_orientations=10,
                 criterion='squared_error', min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=None,
                 random_state=None, min_impurity_decrease=0.0):
        """Initialize the ObliqueDecisionTreeStumpRegressor; see the class documentation for the parameters."""
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
