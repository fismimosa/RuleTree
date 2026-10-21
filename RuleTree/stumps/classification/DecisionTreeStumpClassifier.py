"""DecisionTreeStumpClassifier: the split on one feature for classification."""

from RuleTree.stumps.DecisionTreeStump import DecisionTreeStump
from RuleTree.stumps.splitters.ClassificationSplitter import ClassificationSplitter


class DecisionTreeStumpClassifier(DecisionTreeStump):
    """
    Decision tree stump for classification: a threshold on a numerical feature (``x <= t``) or a
    value of a categorical feature (``x == v``), chosen to maximize the information gain.

    It works on tabular data, time series and images; time series and images are flattened into one
    feature per value. It is a :class:`~RuleTree.stumps.Stump.Stump` made of a MatrixTransformer and
    a ClassificationSplitter, and the default stump of RuleTreeClassifier.

    Parameters
    ----------
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
    transformer_ : MatrixTransformer
        The fitted transformer.
    splitter_ : ClassificationSplitter
        The fitted splitter (feature, threshold, gain and impurity of the split).
    data_type_ : str
        Data type the stump was fitted on.
    feature_original : int
        Column of the data used by the split.
    threshold_original : float or object
        Threshold of a numerical split, or value of a categorical split.

    See Also
    --------
    DecisionTreeStumpRegressor : the same stump for regression.
    sklearn.tree.DecisionTreeClassifier : the tree used to split the numerical features.

    Examples
    --------
    >>> from sklearn.datasets import load_iris
    >>> from RuleTree import RuleTreeClassifier
    >>> from RuleTree.stumps.classification import DecisionTreeStumpClassifier
    >>> X, y = load_iris(return_X_y=True)
    >>> stump = DecisionTreeStumpClassifier(criterion="entropy", random_state=0).fit(X, y, context=None, node_id=None)
    >>> stump.get_rule()["textual_rule"]
    'X_3 <= 0.8'
    >>> stump.apply(X[[0, 100]])  # 1: left child, 2: right child
    array([1, 2])

    As base stump of a tree:

    >>> model = RuleTreeClassifier(max_depth=2, base_stumps=DecisionTreeStumpClassifier(min_samples_leaf=5))
    >>> model = model.fit(X, y)
    """

    _splitter_class = ClassificationSplitter

    def __init__(self, criterion='gini', min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=None,
                 random_state=None, min_impurity_decrease=0.0, class_weight=None):
        """Initialize the DecisionTreeStumpClassifier; see the class documentation for the parameters."""
        self.criterion = criterion
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_features = max_features
        self.random_state = random_state
        self.min_impurity_decrease = min_impurity_decrease
        self.class_weight = class_weight

    def _splitter_params(self) -> dict:
        """
        Parameters of the ClassificationSplitter: those of DecisionTreeStump and class_weight.

        Returns
        -------
        params : dict
        """
        return super()._splitter_params() | {"class_weight": self.class_weight}
