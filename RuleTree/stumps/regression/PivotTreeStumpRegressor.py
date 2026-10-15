"""PivotTreeStumpRegressor: the split on the distance from a pivot instance, for regression."""

from RuleTree.stumps.regression.DecisionTreeStumpRegressor import DecisionTreeStumpRegressor
from RuleTree.stumps.transformers.PivotTransformer import PivotTransformer
from RuleTree.utils.define import MODEL_TYPE_REG


class PivotTreeStumpRegressor(DecisionTreeStumpRegressor):
    """
    Pivot stump for regression: a threshold on the distance from a *pivot*, an instance of the
    training set (``distance <= t`` goes left).

    At each node, the candidate pivots are selected among the instances of the node: the instance
    whose distance best separates high and low values of the target (descriptive pivot) and the one
    found by a regression tree on the distances (discriminative pivot). The split is the best
    threshold on the distance from one of the candidates. Distances are computed on the
    standardized numerical features; the distance matrix of the training set is computed once and
    cached in the tree. In the rules, ``P_<i>`` is the distance from the training instance ``i``.

    It is a :class:`~RuleTree.stumps.Stump.Stump` made of a PivotTransformer and a
    RegressionSplitter; it supports only tabular data.

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
    transformer_ : PivotTransformer
        The fitted transformer.
    splitter_ : RegressionSplitter
        The fitted splitter (feature, threshold, gain and impurity of the split).
    data_type_ : str
        Data type the stump was fitted on.
    feature_original : int
        Index (in the training set) of the pivot of the split.
    threshold_original : float
        Threshold of the split.
    pivot_split : PivotSplit
        The selection of the candidate pivots of the node.
    X_split_instance : numpy.ndarray
        The pivot, as standardized numerical features.

    See Also
    --------
    PivotTreeStumpClassifier : the same stump for classification.
    PivotTransformer : the selection of the pivots and the distances.

    Notes
    -----
    The parameters of the split (criterion, min_samples_leaf, ...) are also used by the decision trees
    that select the discriminative pivots.

    Examples
    --------
    >>> from sklearn.datasets import load_diabetes
    >>> from RuleTree.stumps.regression import PivotTreeStumpRegressor
    >>> X, y = load_diabetes(return_X_y=True)
    >>> stump = PivotTreeStumpRegressor(random_state=0).fit(X[:200], y[:200])
    >>> stump.get_rule()["textual_rule"]  # distance from the training instance 62
    'P_62 <= 4.311'
    """

    def __init__(self, distance_measure='euclidean', k=None, reg_contrast_lambda=.2, reg_max_thresholds=None,
                 criterion='squared_error', min_samples_leaf=1, min_weight_fraction_leaf=0.0,
                 max_features=None, random_state=None, min_impurity_decrease=0.0):
        """Initialize the PivotTreeStumpRegressor; see the class documentation for the parameters."""
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

    def _make_transformer(self):
        """
        A new, unfitted PivotTransformer with the parameters of the stump; its decision trees use the
        parameters of the splitter.

        Returns
        -------
        transformer : PivotTransformer
        """
        return PivotTransformer(ml_task=MODEL_TYPE_REG, distance_measure=self.distance_measure, k=self.k,
                                reg_contrast_lambda=self.reg_contrast_lambda,
                                reg_max_thresholds=self.reg_max_thresholds, tree_params=self._splitter_params())

    @property
    def pivot_split(self):
        """PivotSplit that selected the candidate pivots (see RuleTreeClassifier.get_pivots)."""
        return self.transformer_.pivot_split_

    @property
    def X_split_instance(self):
        """The pivot instance (standardized numerical features), None if not fitted."""
        return self.transformer_.candidates_[0] if self._is_fitted() else None

    def get_rule(self, columns_names=None, scaler=None, float_precision=3):
        """
        Get the rule representation of this classifier.

        This method generates a dictionary containing rule information,
        including feature indices, thresholds, and human-readable representations
        of the decision rule.

        Parameters
        ----------
        columns_names : list, optional
            List of column names for the features. Default is None.
        scaler : object, optional
            Scaling object used to transform feature values. Default is None.
        float_precision : int, optional
            Number of decimal places for floating point values in rules. Default is 3.

        Returns
        -------
        dict
            A dictionary containing rule information with keys:
            - feature_idx: Feature index in the original feature space
            - threshold: Threshold value for the decision rule
            - is_categorical: Whether the feature is categorical
            - samples: Number of samples in the node
            - feature_name: Name of the feature
            - textual_rule: Human-readable rule representation
            - blob_rule: Simplified rule representation
            - graphviz_rule: Rule representation for Graphviz visualization
            - not_textual_rule: Negated rule representation
            - not_blob_rule: Negated simplified rule representation
            - not_graphviz_rule: Negated rule representation for Graphviz visualization
        """
        if self.X_split_instance is None:
            return {}

        rule = {
            "feature_idx": self.feature_original,
            "threshold": self.threshold_original,
            "is_categorical": self.is_categorical,
        }

        feat_name = f"P_{rule['feature_idx']}"
        if columns_names is not None:
            # feat_names should not be useful for pivot tree
            # feat_name = columns_names[self.feature_original]
            feat_name = None
        rule["feature_name"] = feat_name

        if scaler is not None:
            NotImplementedError()

        comparison = "<=" if not self.is_categorical else "="
        not_comparison = ">" if not self.is_categorical else "!="
        rounded_value = str(rule["threshold"]) if float_precision is None else round(rule["threshold"], float_precision)
        if scaler is not None:
            NotImplementedError()
        rule["textual_rule"] = f"{feat_name} {comparison} {rounded_value}"
        rule["blob_rule"] = f"{feat_name} {comparison} {rounded_value}"
        rule["graphviz_rule"] = {
            "label": f"{feat_name} {'<=' if not self.is_categorical else '='} {rounded_value}",
        }

        rule["not_textual_rule"] = f"{feat_name} {not_comparison} {rounded_value}"
        rule["not_blob_rule"] = f"{feat_name} {not_comparison} {rounded_value}"
        rule["not_graphviz_rule"] = {
            "label": f"{feat_name} {'>' if not self.is_categorical else '!='} {rounded_value}"
        }

        return rule
