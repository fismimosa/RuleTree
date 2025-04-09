from RuleTree.base.RuleTreeBaseStump import RuleTreeBaseStump
from RuleTree.stumps.classification.DecisionTreeStumpClassifier import DecisionTreeStumpClassifier
from RuleTree.stumps.splitters.MultiplePivotSplit import MultiplePivotSplit
from RuleTree.utils import MODEL_TYPE_CLF
from sklearn.metrics.pairwise import pairwise_distances
import copy
import numpy as np


class MultiplePivotTreeStumpClassifier(DecisionTreeStumpClassifier, RuleTreeBaseStump):
    """
    A classifier that uses multiple pivot-based splitting strategies for decision tree stumps.
    This classifier is designed for classification tasks and supports numerical and categorical features.

    The multiple pivot approach uses pairs of instances from the dataset as reference points
    (pivots) and assigns new instances to classes based on their relative distances to these pivots.
    This creates a decision boundary that is more flexible than traditional axis-parallel splits.

    This classifier inherits functionality from both DecisionTreeStumpClassifier and RuleTreeBaseStump,
    extending them to support pivot-based decision boundaries which can better capture
    complex relationships in the feature space.
    """

    def __init__(self, **kwargs):
        """
        Initialize the MultiplePivotTreeStumpClassifier.

        Args:
            **kwargs: Additional keyword arguments passed to the parent classes and the MultiplePivotSplit.
                Common parameters include:
                - criterion: The function to measure the quality of a split ('gini' or 'entropy')
                - max_features: The number of features to consider when looking for the best split
                - random_state: Controls the randomness of the estimator
                - class_weight: Weights associated with classes
        """
        super().__init__(**kwargs)
        self.multi_pivot_split = MultiplePivotSplit(ml_task=MODEL_TYPE_CLF, **kwargs)
        self.distance_measure = None
        self.split_instance = None

    def fit(self, X, y, distance_matrix, distance_measure, idx, sample_weight=None, check_input=True):
        """
        Fit the classifier to the training data using multiple pivot-based splitting.

        The method analyzes features, applies the multiple pivot splitting strategy,
        and fits the underlying decision tree stump with the transformed data.

        Args:
            X (np.ndarray): Feature matrix of shape (n_samples, n_features).
            y (np.ndarray): Target labels of shape (n_samples,).
            distance_matrix (np.ndarray): Precomputed distance matrix of shape (n_samples, n_samples).
            distance_measure (str): Distance metric to use for pivot calculation (e.g., 'euclidean', 'manhattan').
            idx (int): Index of the initial pivot instance in X.
            sample_weight (np.ndarray, optional): Sample weights of shape (n_samples,). Defaults to None.
            check_input (bool, optional): Whether to validate input data. Defaults to True.

        Returns:
            self: Fitted classifier instance.
        """
        self.feature_analysis(X, y)
        self.num_pre_transformed = self.numerical
        self.cat_pre_transformed = self.categorical

        if len(self.numerical) > 0:
            self.multi_pivot_split.fit(X[:, self.numerical], y, distance_matrix, distance_measure, idx,
                                       sample_weight=sample_weight, check_input=check_input)
            X_transform = self.multi_pivot_split.transform(X[:, self.numerical], distance_measure)
            candidate_names = self.multi_pivot_split.get_candidates_names()
            super().fit(X_transform, y, sample_weight=sample_weight, check_input=check_input)

            self.feature_original = [(candidate_names[0], candidate_names[1]), -2, -2]
            self.threshold_original = self.tree_.threshold
            self.is_pivotal = True
            self.distance_measure = distance_measure
            self.X_best_tup = self.multi_pivot_split.best_tup

        return self

    def apply(self, X):
        """
        Apply the fitted classifier to an input sample.
        
        Transforms the input data using the pivot-based approach and applies the decision rule.
        
        Args:
            X (np.ndarray): Input samples of shape (n_samples, n_features).

        Returns:
            np.ndarray: Predicted class labels for each input sample.
        """
        dist_to_p0 = pairwise_distances(X[:, self.num_pre_transformed],
                                        self.X_best_tup[0].reshape(1, -1),
                                        metric=self.distance_measure).flatten()

        dist_to_p1 = pairwise_distances(X[:, self.num_pre_transformed],
                                        self.X_best_tup[1].reshape(1, -1),
                                        metric=self.distance_measure).flatten()

        dist_binary = np.where(dist_to_p0 < dist_to_p1, 0, 1).reshape(-1, 1)

        X_transformed = dist_binary
        y_pred = (np.ones(X_transformed.shape[0]) * 2)
        X_feature = X_transformed[:, 0]
        y_pred[X_feature <= self.threshold_original[0]] = 1

        return y_pred

    def get_rule(self, columns_names=None, scaler=None, float_precision=3):
        """
        Extract a human-readable decision rule from the classifier.

        Creates a dictionary representation of the pivot-based decision rule, including
        textual and graphical representations for interpretability.

        Args:
            columns_names (list, optional): List of feature names. Defaults to None.
            scaler (object, optional): Scaler used to transform features. Defaults to None.
            float_precision (int, optional): Number of decimal places for threshold values. Defaults to 3.

        Returns:
            dict: Dictionary containing rule information with keys:
                - feature_idx: Indices of pivot instances
                - threshold: Decision threshold
                - coefficients: Coefficients of the rule
                - is_categorical: Whether the feature is categorical
                - samples: Number of samples in the node
                - feature_name: Human-readable feature name
                - textual_rule, blob_rule, graphviz_rule: Different representations of the rule
                - not_textual_rule, not_blob_rule, not_graphviz_rule: Representations of the negated rule
        """
        rule = {
            "feature_idx": self.feature_original[0],  # tuple of instances idx
            "threshold": self.threshold_original[0],  # thr
            "coefficients": self.coefficients,  # coefficients
            "is_categorical": self.is_categorical,
            "samples": self.n_node_samples[0]
        }

        feat_name = " ".join(f"P_{idx}" for idx in list(rule['feature_idx']))  # list of sitrings

        # if columns_names is not None:
        #    feat_name = "_".join(columns_names[idx] for idx in self.feature_original[0]) #check this for feat names
        rule["feature_name"] = feat_name

        # if scaler is not None:
        #    #TODO
        #    raise NotImplementedError()

        comparison = f"closer to {rule['feature_idx'][0]}" if not self.is_categorical else "="
        not_comparison = f"closer to {rule['feature_idx'][1]}" if not self.is_categorical else "!="
        rounded_value = str(rule["threshold"]) if float_precision is None else round(rule["threshold"], float_precision)

        # if scaler is not None:
        #    #TODO
        #    raise NotImplementedError()
        #    pass

        rule["textual_rule"] = f"{comparison} \t{rule['samples']}"
        rule["blob_rule"] = f"{comparison} "
        rule["graphviz_rule"] = {
            "label": f"{comparison} {rounded_value}",
        }

        rule["not_textual_rule"] = f"{not_comparison}"
        rule["not_blob_rule"] = f"{not_comparison}"
        rule["not_graphviz_rule"] = {
            "label": f"{not_comparison}"
        }

        return rule


@classmethod
def dict_to_node(cls, node_dict, X=None):
    """
    Create a classifier instance from a dictionary representation.

    This is a deserialization method that restores a MultiplePivotTreeStumpClassifier
    from a dictionary previously created by node_to_dict().

    Args:
        node_dict (dict): Dictionary representation of the classifier node.
        X (np.ndarray, optional): Reference dataset used to resolve instance indices. Defaults to None.

    Returns:
        MultiplePivotTreeStumpClassifier: Reconstructed classifier instance.
    """
    self = cls()
    self.feature_original = np.zeros(3, dtype=object)
    self.threshold_original = np.zeros(3)
    self.n_node_samples = np.zeros(3, dtype=int)

    self.feature_original[0] = node_dict["feature_idx"]
    self.threshold_original[0] = node_dict["threshold"]
    self.n_node_samples[0] = node_dict["samples"]
    self.is_categorical = node_dict["is_categorical"]

    args = copy.deepcopy(node_dict["args"])
    self.is_oblique = args.pop("is_oblique")
    self.is_pivotal = args.pop("is_pivotal")
    self.unique_val_enum = args.pop("unique_val_enum")
    self.coefficients = args.pop("coefficients")
    self.kwargs = args

    self.distance_measure = args.pop("distance_measure")
    self.num_pre_transformed = args.pop("num_pre_transformed")
    self.cat_pre_transformed = args.pop("cat_pre_transformed")

    # X acts as a reference dataset for the instance id
    if X is not None:
        self.X_best_tup = (X[int(node_dict["feature_idx"][0])],
                           X[int(node_dict["feature_idx"][1])])

    return self


def node_to_dict(self):
    """
    Convert the classifier to a dictionary representation.

    This method serializes the MultiplePivotTreeStumpClassifier into a dictionary
    that can later be used to reconstruct the classifier using dict_to_node().

    Returns:
        dict: Dictionary representation of the classifier node with keys:
            - stump_type: The class module path
            - feature_idx, threshold, samples, impurity: Node statistics
            - args: Configuration parameters
            - split: Split information
    """
    rule = self.get_rule(float_precision=None)

    rule["stump_type"] = self.__class__.__module__
    rule["samples"] = self.n_node_samples[0]
    rule["impurity"] = self.tree_.impurity[0]

    rule["args"] = {
                       "is_oblique": self.is_oblique,
                       "is_pivotal": self.is_pivotal,
                       "unique_val_enum": self.unique_val_enum,
                       "coefficients": self.coefficients,
                       "num_pre_transformed": self.num_pre_transformed,
                       "cat_pre_transformed": self.cat_pre_transformed,

                       "distance_measure": self.distance_measure  # adding this for PT

                   } | self.kwargs

    rule["split"] = {
        "args": {}
    }

    return rule


def export_graphviz(self, graph=None, columns_names=None, scaler=None, float_precision=3):
    """
    Export the classifier as a graphviz graph for visualization.

    This method is not yet implemented.

    Args:
        graph (graphviz.Digraph, optional): A graphviz Digraph object. Defaults to None.
        columns_names (list, optional): List of feature names. Defaults to None.
        scaler (object, optional): Scaler used to transform features. Defaults to None.
        float_precision (int, optional): Number of decimal places for threshold values. Defaults to 3.

    Raises:
        NotImplementedError: This method is not yet implemented.
    """
    raise NotImplementedError()
