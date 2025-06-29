import copy

import numpy as np
from sklearn.tree import DecisionTreeClassifier


from RuleTree.base.RuleTreeBaseStump import RuleTreeBaseStump
from RuleTree.exceptions import NoSplitFoundWarning

from RuleTree.utils.data_utils import get_info_gain, _get_info_gain, gini, entropy, _my_counts


class DecisionTreeStumpClassifier(DecisionTreeClassifier, RuleTreeBaseStump):
    """
    A decision tree stump classifier that extends sklearn's DecisionTreeClassifier and RuleTreeBaseStump.

    A decision tree stump is a decision tree with a maximum depth of 1 (a single split), making
    it a simple interpretable model. This implementation supports both numerical and categorical features,
    provides methods for rule extraction, and can be used as a building block in more complex ensembles.

    The class handles both numerical splits (using ≤ comparisons) and categorical splits (using = comparisons),
    and automatically selects the feature and split that maximizes information gain.

    Attributes:
        is_categorical (bool): Whether the selected split is categorical.
        kwargs (dict): Additional arguments passed to DecisionTreeClassifier.
        threshold_original (array): Split threshold values.
        feature_original (array): Feature indices used for splits.
        coefficients (array): Coefficients for oblique splits (if used).
        impurity_fun (function): Function used to calculate impurity (gini, entropy, etc.).
    """

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
        rounded_value = str(rule["threshold"]) if float_precision is None else round(rule["threshold"], float_precision)
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

    def node_to_dict(self):
        """
        Serializes the current node into a dictionary representation.

        This method captures all information necessary to reconstruct the stump,
        including feature index, threshold, metadata, and additional parameters.

        Returns:
            dict: A dictionary containing the node's attributes and metadata with the following keys:
                - All fields from the get_rule() method
                - stump_type: The fully qualified class name
                - impurity: Node impurity value
                - args: Dictionary with constructor arguments and parameters
                - split: Dictionary with split information
        """
        rule = self.get_rule(float_precision=None)

        rule["stump_type"] = self.__class__.__module__
        rule["impurity"] = self.impurity

        rule["args"] = {
            "coefficients": self.coefficients,
        } | self.kwargs

        rule["split"] = {
            "args": {}
        }

        return rule

    @classmethod
    def dict_to_node(cls, node_dict, X=None):
        """
        Deserializes a dictionary into a DecisionTreeStumpClassifier node.

        This is essentially the inverse operation of node_to_dict(), recreating
        a stump instance from a serialized dictionary representation.

        Args:
            node_dict (dict): Dictionary containing node attributes, typically created
                             with the node_to_dict() method.
            X (array-like, optional): Feature matrix for additional context.
                                     Not typically used but available for subclass implementations.

        Returns:
            DecisionTreeStumpClassifier: An instance of the class initialized with the node's attributes.

        Raises:
            AssertionError: If required fields are missing from the dictionary.
        """
        self = cls()

        assert 'feature_idx' in node_dict
        assert 'threshold' in node_dict
        assert 'is_categorical' in node_dict

        self.feature_original = np.zeros(3, dtype=int)
        self.threshold_original = np.zeros(3)

        self.feature_original[0] = node_dict["feature_idx"]
        self.threshold_original[0] = node_dict["threshold"]
        self.is_categorical = node_dict["is_categorical"]

        args = copy.deepcopy(node_dict.get("args", {}))
        self.coefficients = args.pop("coefficients", np.nan)
        self.impurity = args.pop("impurity", np.nan)
        self.kwargs = args

        return self


    def __init__(self, **kwargs):
        """
        Initializes the DecisionTreeStumpClassifier.

        This constructor sets up the decision tree stump with default values and
        interprets the provided keyword arguments for configuration.

        Args:
            **kwargs: Additional arguments for the DecisionTreeClassifier. Common parameters include:
                - criterion (str): Function to measure the quality of a split ('gini' or 'entropy')
                - min_samples_leaf (int): Minimum samples required to be at a leaf node
                - class_weight (dict or 'balanced'): Weights associated with classes
                - random_state (int): Seed for the random number generator
        """
        super().__init__(**kwargs)

        self.is_categorical = None

        self.kwargs = kwargs

        self.threshold_original = None
        self.feature_original = None
        self.coefficients = None

        if 'criterion' not in kwargs or kwargs['criterion'] == "gini":
            self.impurity_fun = gini
        elif kwargs['criterion'] == "entropy":
            self.impurity_fun = entropy
        else:
            self.impurity_fun = kwargs['criterion']

    def get_params(self, deep=True):
        """
        Returns the parameters of the classifier.

        This method overrides the parent method to ensure proper parameter retrieval
        for hyperparameter tuning and model inspection.

        Args:
            deep (bool, optional): If True, will return the parameters for this estimator and contained subobjects.
                                  Default is True.

        Returns:
            dict: A dictionary of parameters, with parameter names mapped to their values.
        """
        return self.kwargs

    def fit(self, X, y, idx=None, context=None, sample_weight=None, check_input=True):
        """
        Fits the decision tree stump to the provided data.

        This method finds the optimal single-feature split that maximizes information gain
        for both numerical and categorical features.

        Args:
            X (array-like): Feature matrix of shape (n_samples, n_features).
            y (array-like): Target vector of shape (n_samples,).
            idx (slice, optional): Indices for slicing the data. If None, all samples are used.
            context (object, optional): Additional context for fitting (not used directly).
            sample_weight (array-like, optional): Sample weights of shape (n_samples,).
                                                 If None, samples are equally weighted.
            check_input (bool, optional): Whether to check the input data. Default is True.

        Returns:
            DecisionTreeStumpClassifier: The fitted classifier (self).
        """
        if idx is None:
            idx = slice(None)
        X = X[idx]
        y = y[idx]

        len_x = len(X)

        class_weight = None
        if self.class_weight == "balanced":
            class_weight = {}
            for class_label in np.unique(y):
                class_weight[class_label] = len_x / (len(self.classes_) * len(y[y == class_label]))

        self.feature_analysis(X, y)
        best_info_gain = -float('inf')

        if len(self.numerical) > 0:
            super().fit(X[:, self.numerical], y, sample_weight=sample_weight, check_input=check_input)
            self.feature_original = [self.numerical[x] if x != -2 else x for x in self.tree_.feature]
            self.threshold_original = self.tree_.threshold
            best_info_gain = get_info_gain(self)

            # no split
            if len(self.feature_original) == 1:
                raise NoSplitFoundWarning(f"No split found for X {X.shape} and y {np.unique(y)}")

            self.impurity = [
                self.impurity_fun(y, sample_weight, class_weight),
                self.impurity_fun(y[X[:, self.feature_original[0]] <= self.threshold_original[0]], None, class_weight),
                self.impurity_fun(y[X[:, self.feature_original[0]] > self.threshold_original[0]], None, class_weight)
            ]

        self._fit_cat(X, y, best_info_gain)

        return self

    def _fit_cat(self, X, y, best_info_gain, sample_weight=None):
        """
        Fits the stump for categorical features.

        This method evaluates categorical splits and updates the model if a better split
        (higher information gain) is found compared to the best numerical split.

        Args:
            X (array-like): Feature matrix of shape (n_samples, n_features).
            y (array-like): Target vector of shape (n_samples,).
            best_info_gain (float): Best information gain observed so far from numerical splits.
            sample_weight (array-like, optional): Sample weights of shape (n_samples,).
                                                 If None, samples are equally weighted.

        """
        if len(self.categorical) > 0 and best_info_gain != float('inf'):
            len_x = len(X)

            class_weight = None
            if self.class_weight == "balanced":
                class_weight = {}
                for class_label in np.unique(y):
                    class_weight[class_label] = len_x / (len(self.classes_) * len(y[y == class_label]))


            for i in self.categorical:
                for value in np.unique(X[:, i]):
                    X_split = X[:, i:i+1] == value

                    len_left = np.sum(X_split)

                    if sample_weight is not None:
                        if _my_counts(y, sample_weight) - (_my_counts(y[X_split[:, 0]], sample_weight)
                                                           + _my_counts(y[~X_split[:, 0]], sample_weight)) <= 0:
                            continue

                        if sum(sample_weight[X_split[:, 0]]) < self.min_weight_fraction_leaf \
                            or sum(sample_weight[~X_split[:, 0]]) < self.min_weight_fraction_leaf:
                            continue

                        if ((_my_counts(y[X_split[:, 0]], sample_weight) <= 0).any()
                                or (_my_counts(y[~X_split[:, 0]], sample_weight) <= 0).any()):
                            continue

                        info_gain = _get_info_gain(self.impurity_fun(y, sample_weight, class_weight),
                                                   self.impurity_fun(y[X_split[:, 0]],
                                                                     sample_weight[X_split[:, 0]],
                                                                     class_weight),
                                                   self.impurity_fun(y[~X_split[:, 0]],
                                                                     sample_weight[~X_split[:, 0]],
                                                                     class_weight),
                                                   len_x,
                                                   len_left,
                                                   len_x-len_left)
                    else:
                        info_gain = _get_info_gain(self.impurity_fun(y, sample_weight, class_weight),
                                                   self.impurity_fun(y[X_split[:, 0]], None, class_weight),
                                                   self.impurity_fun(y[~X_split[:, 0]], None, class_weight),
                                                   len_x,
                                                   len_left,
                                                   len_x - len_left)

                    if info_gain > best_info_gain:
                        best_info_gain = info_gain
                        self.feature_original = [i, -2, -2]
                        self.threshold_original = np.array([value, -2, -2])
                        self.is_categorical = True
                        self.impurity = [
                            self.impurity_fun(y, sample_weight, class_weight),
                            self.impurity_fun(y[X_split[:, 0]], None, class_weight),
                            self.impurity_fun(y[~X_split[:, 0]], None, class_weight)
                        ]


    def apply(self, X, check_input=False):
        if len(self.feature_original) < 3:
            return np.ones(X.shape[0])

        if not self.is_categorical:
            y_pred = (np.ones(X.shape[0]) * 2)
            X_feature = X[:, self.feature_original[0]]
            y_pred[X_feature <= self.threshold_original[0]] = 1

            return y_pred

        else:
            y_pred = np.ones(X.shape[0]) * 2
            X_feature = X[:, self.feature_original[0]]
            y_pred[X_feature == self.threshold_original[0]] = 1

            return y_pred

    def update_statistics(self, X, y, idx=None, context=None, sample_weight=None, check_input=True):
        X = X[idx]
        y = y[idx]

        len_x = len(X)

        class_weight = None
        if self.class_weight == "balanced":
            class_weight = {}
            for class_label in np.unique(y):
                class_weight[class_label] = len_x / (len(self.classes_) * len(y[y == class_label]))

        self.impurity = [
            self.impurity_fun(y, sample_weight, class_weight),
            self.impurity_fun(y[X[:, self.feature_original[0]] <= self.threshold_original[0]], None, class_weight),
            self.impurity_fun(y[X[:, self.feature_original[0]] > self.threshold_original[0]], None, class_weight)
        ]
