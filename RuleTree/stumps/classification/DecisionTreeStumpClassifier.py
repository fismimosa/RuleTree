"""DecisionTreeStumpClassifier: the split on one feature for classification."""

import numpy as np

from RuleTree.stumps.Stump import Stump
from RuleTree.stumps.splitters.ClassificationSplitter import ClassificationSplitter
from RuleTree.stumps.transformers.MatrixTransformer import MatrixTransformer


class DecisionTreeStumpClassifier(Stump):
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
    >>> stump = DecisionTreeStumpClassifier(criterion="entropy", random_state=0).fit(X, y)
    >>> stump.get_rule()["textual_rule"]
    'X_3 <= 0.8'
    >>> stump.apply(X[[0, 100]])  # 1: left child, 2: right child
    array([1, 2])

    As base stump of a tree:

    >>> model = RuleTreeClassifier(max_depth=2, base_stumps=DecisionTreeStumpClassifier(min_samples_leaf=5))
    >>> model = model.fit(X, y)
    """

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
        Parameters of the ClassificationSplitter.

        Returns
        -------
        params : dict
        """
        return dict(criterion=self.criterion, min_samples_leaf=self.min_samples_leaf,
                    min_weight_fraction_leaf=self.min_weight_fraction_leaf, max_features=self.max_features,
                    random_state=self.random_state, min_impurity_decrease=self.min_impurity_decrease,
                    class_weight=self.class_weight)

    def _make_transformer(self):
        """
        A new MatrixTransformer: the data are used as they are (time series and images flattened).

        Returns
        -------
        transformer : MatrixTransformer
        """
        return MatrixTransformer()

    def _make_splitter(self):
        """
        A new, unfitted ClassificationSplitter with the parameters of the stump.

        Returns
        -------
        splitter : ClassificationSplitter
        """
        return ClassificationSplitter(**self._splitter_params())

    def get_rule(self, columns_names=None, scaler=None, float_precision=3):
        """
        Rule of the split, e.g. ``petal length <= 2.45`` or ``color = red``.

        Parameters
        ----------
        columns_names : list of str, default=None
            Names of the features. If None, ``X_<index>``. Features named by the transformer (see
            RuleTreeBaseTransformer.feature_id) keep their name.
        scaler : TransformerMixin, default=None
            Scaler used to preprocess the data: the threshold is also given in the original scale
            (``threshold_scaled``) and shown in the rule.
        float_precision : int, default=3
            Number of decimals of the threshold. If None, it is not rounded.

        Returns
        -------
        rule : dict
            ``feature_idx``, ``threshold``, ``is_categorical``, ``feature_name``, ``threshold_scaled``
            (with a scaler), the condition of the left child (``textual_rule``, ``blob_rule``,
            ``graphviz_rule``) and its negation (``not_textual_rule``, ``not_blob_rule``,
            ``not_graphviz_rule``).
        """
        rule = {
            "feature_idx": self.feature_original,
            "threshold": self.threshold_original,
            "is_categorical": self.is_categorical,
        }

        if isinstance(rule["feature_idx"], str):  # a feature named by the transformer, e.g. "mean"
            feat_name = rule["feature_idx"]
        elif columns_names is not None:
            feat_name = columns_names[self.feature_original]
        else:
            feat_name = f"X_{rule['feature_idx']}"
        rule["feature_name"] = feat_name

        if scaler is not None:
            array = np.zeros((1, scaler.n_features_in_))
            array[0, self.feature_original] = self.threshold_original

            rule["threshold_scaled"] = scaler.inverse_transform(array)[0, self.feature_original]

        comparison = "<=" if not self.is_categorical else "="
        not_comparison = ">" if not self.is_categorical else "!="
        rounded_value = str(rule["threshold"]) if float_precision is None or self.is_categorical else (
            round(rule["threshold"], float_precision))
        if scaler is not None:
            rounded_value = str(rule["threshold_scaled"]) if float_precision is None else (
                round(rule["threshold_scaled"], float_precision))
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
