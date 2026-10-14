"""DecisionTreeStumpClassifier: the split on one feature for classification."""

import numpy as np

from RuleTree.stumps.Stump import Stump
from RuleTree.stumps.splitters.ClassificationSplitter import ClassificationSplitter
from RuleTree.stumps.transformers.MatrixTransformer import MatrixTransformer


class DecisionTreeStumpClassifier(Stump):
    """
    A decision tree stump for classification: a single split on one feature.

    Numerical features are split with a threshold (``x <= t``), categorical features on a value
    (``x == v``): the split with the highest information gain is selected. Time series and images
    are flattened into one feature per value.

    It is made of a MatrixTransformer and a ClassificationSplitter.

    Parameters
    ----------
    criterion : {'gini', 'entropy'}, default='gini'
        Impurity measure.
    min_samples_leaf, min_weight_fraction_leaf, max_features, random_state, min_impurity_decrease, class_weight
        As in sklearn.tree.DecisionTreeClassifier, for the numerical features. With random_state=None,
        ties between features with the same gain are broken at random.
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
            "feature_idx": self.feature_original[0],
            "threshold": self.threshold_original[0],
            "is_categorical": self.is_categorical,
        }

        if isinstance(rule["feature_idx"], str):  # a feature named by the transformer, e.g. "mean"
            feat_name = rule["feature_idx"]
        elif columns_names is not None:
            feat_name = columns_names[self.feature_original[0]]
        else:
            feat_name = f"X_{rule['feature_idx']}"
        rule["feature_name"] = feat_name

        if scaler is not None:
            array = np.zeros((1, scaler.n_features_in_))
            array[0, self.feature_original[0]] = self.threshold_original[0]

            rule["threshold_scaled"] = scaler.inverse_transform(array)[0, self.feature_original[0]]

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
