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
        self.criterion = criterion
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_features = max_features
        self.random_state = random_state
        self.min_impurity_decrease = min_impurity_decrease
        self.class_weight = class_weight

    def _splitter_params(self) -> dict:
        return dict(criterion=self.criterion, min_samples_leaf=self.min_samples_leaf,
                    min_weight_fraction_leaf=self.min_weight_fraction_leaf, max_features=self.max_features,
                    random_state=self.random_state, min_impurity_decrease=self.min_impurity_decrease,
                    class_weight=self.class_weight)

    def _make_transformer(self):
        return MatrixTransformer()

    def _make_splitter(self):
        return ClassificationSplitter(**self._splitter_params())

    def get_rule(self, columns_names=None, scaler=None, float_precision=3):
        """
        Extracts the rule represented by the decision tree stump in a human-readable format.

        This method generates a dictionary that describes the splitting rule, including feature index,
        threshold value, textual representation, and graphical representation options.

        Args:
            columns_names (list, optional): List of column names for feature mapping. If provided,
                                           feature names will be used instead of indices in the rule text.
            scaler (object, optional): Scaler object for inverse transformation of thresholds.
                                      Useful when working with normalized/standardized features.
            float_precision (int, optional): Number of decimal places to round threshold values to.
                                            Set to None to avoid rounding.

        Returns:
            dict: A dictionary containing the rule details with the following keys:
                - feature_idx: The index of the feature used for the split
                - threshold: The split threshold value
                - is_categorical: Whether the split is categorical or numerical
                - samples: Number of samples at the node
                - feature_name: Name of the feature (from columns_names if provided)
                - threshold_scaled: Original scale threshold (if scaler is provided)
                - textual_rule: Human-readable representation of the rule
                - blob_rule: Compact representation of the rule
                - graphviz_rule: Dictionary with graphviz options for visualization
                - not_textual_rule: Negation of the rule in text form
                - not_blob_rule: Compact representation of the negated rule
                - not_graphviz_rule: Dictionary with graphviz options for the negated rule
        """
        rule = {
            "feature_idx": self.feature_original[0],
            "threshold": self.threshold_original[0],
            "is_categorical": self.is_categorical,
        }

        feat_name = f"X_{rule['feature_idx']}"
        if columns_names is not None:
            feat_name = columns_names[self.feature_original[0]]
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
