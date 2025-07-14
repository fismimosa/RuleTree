from sklearn.preprocessing import StandardScaler

from RuleTree.base.RuleTreeBaseStump import RuleTreeBaseStump
from RuleTree.stumps.classification.DecisionTreeStumpClassifier import DecisionTreeStumpClassifier
from RuleTree.stumps.splitters.PivotSplit import PivotSplit
from RuleTree.utils import MODEL_TYPE_CLF
from sklearn.metrics.pairwise import pairwise_distances
import numpy as np
import copy
import warnings


class PivotTreeStumpClassifier(DecisionTreeStumpClassifier, RuleTreeBaseStump):
    """
    A classifier that uses a pivot-based splitting strategy for decision tree stumps.

    This classifier implements a pivot-based approach for decision tree stumps in classification tasks.
    It works by selecting a pivot instance from the dataset and calculating distances from all other
    instances to this pivot. These distances are then used to make splitting decisions.

    The pivot-based approach is particularly useful for working with non-standard data representations
    where traditional feature-based splits may not be applicable. It provides an alternative
    splitting mechanism based on distance metrics.

    This classifier supports both numerical and categorical features and inherits functionality
    from both DecisionTreeStumpClassifier and RuleTreeBaseStump.
    """

    def __init__(self, **kwargs):
        """
        Initialize the PivotTreeStumpClassifier.

        Parameters
        ----------
        **kwargs : dict
            Additional keyword arguments passed to the parent classes and the PivotSplit.
            Common parameters include criterion, max_depth, min_samples_split, etc.
        """
        super().__init__(**kwargs)
        self.pivot_split = PivotSplit(ml_task=MODEL_TYPE_CLF, **kwargs)
        self.distance_measure = kwargs.get('distance_measure', 'euclidean')
        self.split_instance = None

    def fit(self, X, y, idx=None, context=None, sample_weight=None, check_input=True):
        """
        Fit the classifier to the training data.

        This method implements the model training process by:
        1. Analyzing the features in the input data
        2. Computing pivot-based splits using the provided distance matrix
        3. Transforming the data based on distances to the pivot instance
        4. Fitting the underlying decision tree stump to the transformed data

        Parameters
        ----------
        X : numpy.ndarray
            Feature matrix of shape (n_samples, n_features).
        y : numpy.ndarray
            Target labels of shape (n_samples,).
        idx : int
            Index of the pivot instance in the dataset.
        sample_weight : numpy.ndarray, optional
            Sample weights of shape (n_samples,). Default is None.
        check_input : bool, optional
            Whether to validate input data. Default is True.

        Returns
        -------
        self : PivotTreeStumpClassifier
            Fitted classifier instance.
        """
        X = X[idx]
        y = y[idx]
        self.feature_analysis(X, y)
        self.num_pre_transformed = self.numerical
        self.cat_pre_transformed = self.categorical

        if context is None:
            raise ValueError("Context must be provided for PivotTreeStumpClassifier.")
        if not hasattr(context, "PivotTreeStumpClassifier_scaler"):
            context.PivotTreeStumpClassifier_scaler = StandardScaler().fit(X)
            context.distance_matrix = pairwise_distances(X[:, self.numerical], metric=self.distance_measure)

        self.scaler = context.PivotTreeStumpClassifier_scaler
        distance_matrix = context.distance_matrix[np.ix_(idx, idx)]
        X_scaled = context.PivotTreeStumpClassifier_scaler.transform(X[:, self.numerical])


        if len(self.numerical) > 0:
            self.pivot_split.fit(X_scaled, y, distance_matrix, self.distance_measure, idx, sample_weight=sample_weight,
                                 check_input=check_input)
            X_transform = self.pivot_split.transform(X_scaled, self.distance_measure)
            candidate_names = self.pivot_split.get_candidates_names()
            super().fit(X_transform, y, sample_weight=sample_weight, check_input=check_input)

            self.feature_original = [f'{candidate_names[self.tree_.feature[0]]}', -2, -2]
            self.threshold_original = self.tree_.threshold

            self.X_split_instance = self.pivot_split.X_candidates[self.tree_.feature[0]]

        return self

    def apply(self, X):
        """
        Apply the fitted classifier to the input data.

        This method transforms the input data using the pivot-based approach
        and then applies the classifier decision rule.

        Parameters
        ----------
        X : numpy.ndarray
            Feature matrix of shape (n_samples, n_features).

        Returns
        -------
        numpy.ndarray
            Predicted class labels for each sample in X.
        """
        X_transformed = pairwise_distances(self.scaler.transform(X[:, self.num_pre_transformed]),
                                           self.X_split_instance.reshape(1, -1),
                                           metric=self.distance_measure)

        y_pred = (np.ones(X_transformed.shape[0]) * 2)
        X_feature = X_transformed[:, 0]
        y_pred[X_feature <= self.threshold_original[0]] = 1

        return y_pred

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
        rule = {
            "feature_idx": self.feature_original[0],
            "threshold": self.threshold_original[0],
            "is_categorical": self.is_categorical,
        }

        feat_name = f"P_{rule['feature_idx']}"
        if columns_names is not None:
            # feat_names should not be useful for pivot tree
            # feat_name = columns_names[self.feature_original[0]]
            feat_name = None
        rule["feature_name"] = feat_name

        if scaler is not None:
            NotImplementedError()

        comparison = "<=" if not self.is_categorical else "="
        not_comparison = ">" if not self.is_categorical else "!="
        rounded_value = str(rule["threshold"]) if float_precision is None else round(rule["threshold"], float_precision)
        if scaler is not None:
            NotImplementedError()
        rule["textual_rule"] = f"{feat_name} {comparison} {rounded_value}\t{rule['samples']}"
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
        Convert the classifier node to a dictionary representation.

        This method serializes the classifier's properties and configuration into a
        dictionary format, which can be used for persistence or knowledge transfer.

        Returns
        -------
        dict
            A dictionary containing all essential information about the classifier node,
            including the rule, classifier type, node statistics, and configuration parameters.
        """
        rule = self.get_rule(float_precision=None)

        rule["stump_type"] = self.__class__.__module__
        rule["impurity"] = self.tree_.impurity[0]

        rule["args"] = {
                           "num_pre_transformed": self.num_pre_transformed,
                           "cat_pre_transformed": self.cat_pre_transformed,
                           "distance_measure": self.distance_measure,
                           "X_split_instance": self.X_split_instance,
                       } | self.kwargs

        rule["split"] = {
            "args": {}
        }

        return rule

    @classmethod
    def dict_to_node(cls, node_dict, X=None):
        """
        Create a classifier node from a dictionary representation.

        This class method deserializes a dictionary back into a PivotTreeStumpClassifier
        instance, restoring its state and configuration.

        Parameters
        ----------
        node_dict : dict
            Dictionary containing the serialized classifier information.
        X : numpy.ndarray, optional
            Reference dataset that contains the pivot instance. Default is None.

        Returns
        -------
        PivotTreeStumpClassifier
            A reconstructed classifier instance.
        """
        self = cls()
        self.feature_original = np.zeros(3, dtype=int)
        self.threshold_original = np.zeros(3)

        self.feature_original[0] = node_dict["feature_idx"]
        self.threshold_original[0] = node_dict["threshold"]

        args = copy.deepcopy(node_dict["args"])
        self.kwargs = args

        self.distance_measure = args.pop("distance_measure")
        self.num_pre_transformed = args.pop("num_pre_transformed")
        self.cat_pre_transformed = args.pop("cat_pre_transformed")
        self.X_split_instance = np.array(args.pop("X_split_instance"))

        return self

    def export_graphviz(self, graph=None, columns_names=None, scaler=None, float_precision=3):
        """
        Export the classifier as a Graphviz digraph representation.

        This method is intended to visualize the classifier as a graph but is
        currently not implemented for PivotTreeStumpClassifier.

        Parameters
        ----------
        graph : pydot.Dot, optional
            An existing graph to add the classifier node to. Default is None.
        columns_names : list, optional
            List of column names for the features. Default is None.
        scaler : object, optional
            Scaling object used to transform feature values. Default is None.
        float_precision : int, optional
            Number of decimal places for floating point values. Default is 3.

        Raises
        ------
        NotImplementedError
            This method is not implemented for PivotTreeStumpClassifier.
        """
        raise NotImplementedError()