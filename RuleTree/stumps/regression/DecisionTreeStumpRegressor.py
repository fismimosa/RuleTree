"""DecisionTreeStumpRegressor: the split on one feature for regression."""

from RuleTree.stumps.DecisionTreeStump import DecisionTreeStump
from RuleTree.stumps.splitters.RegressionSplitter import RegressionSplitter


class DecisionTreeStumpRegressor(DecisionTreeStump):
    """
    Decision tree stump for regression: a threshold on a numerical feature (``x <= t``) or a value of
    a categorical feature (``x == v``), chosen to maximize the decrease of impurity.

    It works on tabular data, time series and images; time series and images are flattened into one
    feature per value. It is a :class:`~RuleTree.stumps.Stump.Stump` made of a MatrixTransformer and
    a RegressionSplitter, and the default stump of RuleTreeRegressor.

    Parameters
    ----------
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
    random_state : int, RandomState instance or None, default=None
        Seed of the choice of the features (``max_features``) and of the ties between features with
        the same gain. With None, a tree gives the stump its own ``random_state``.
    min_impurity_decrease : float, default=0.0
        A numerical split is kept only if it decreases the weighted impurity by at least this value.

    Attributes
    ----------
    transformer_ : MatrixTransformer
        The fitted transformer.
    splitter_ : RegressionSplitter
        The fitted splitter (feature, threshold, gain and impurity of the split).
    data_type_ : str
        Data type the stump was fitted on.
    feature_original : int
        Column of the data used by the split.
    threshold_original : float or object
        Threshold of a numerical split, or value of a categorical split.

    See Also
    --------
    DecisionTreeStumpClassifier : the same stump for classification.
    sklearn.tree.DecisionTreeRegressor : the tree used to split the numerical features.

    Examples
    --------
    >>> from sklearn.datasets import load_diabetes
    >>> from RuleTree.stumps.regression import DecisionTreeStumpRegressor
    >>> X, y = load_diabetes(return_X_y=True)
    >>> stump = DecisionTreeStumpRegressor(min_samples_leaf=20, random_state=0)
    >>> stump = stump.fit(X, y, context=None, node_id=None, cache=None)
    >>> stump.get_rule()["textual_rule"]
    'X_8 <= -0.004'
    """

    _splitter_class = RegressionSplitter

    def __init__(self, criterion='squared_error', min_samples_leaf=1, min_weight_fraction_leaf=0.0,
                 max_features=None, random_state=None, min_impurity_decrease=0.0):
        """Initialize the DecisionTreeStumpRegressor; see the class documentation for the parameters."""
        self.criterion = criterion
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_features = max_features
        self.random_state = random_state
        self.min_impurity_decrease = min_impurity_decrease
