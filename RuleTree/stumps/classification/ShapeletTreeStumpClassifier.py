import copy
import inspect
import io
import random
import warnings
import numpy as np
import psutil
import tempfile312
from numba import UnsupportedError

from matplotlib import pyplot as plt

from RuleTree.stumps.classification.DecisionTreeStumpClassifier import DecisionTreeStumpClassifier
from RuleTree.utils.data_utils import gini, entropy, get_info_gain
from RuleTree.utils.define import DATA_TYPE_TS
from RuleTree.utils.shapelet_transform.Shapelets import Shapelets


class ShapeletTreeStumpClassifier(DecisionTreeStumpClassifier):
    """
    A classification tree stump that uses shapelets as features for time series data.

    This class extends `DecisionTreeStumpClassifier` and incorporates shapelet-based
    feature extraction for classification tasks. It supports various selection methods
    for shapelets and allows customization of parameters such as the number of shapelets,
    sliding window size, and distance metric.

    Attributes:
        n_shapelets (int): Number of shapelets to extract.
        n_shapelets_for_selection (int): Number of shapelets to consider during selection.
        n_ts_for_selection_per_class (int): Number of time series per class for shapelet selection.
        sliding_window (int): Size of the sliding window for shapelet extraction.
        selection (str): Method for selecting shapelets ('random', 'mi_clf', 'cluster').
        distance (str): Distance metric to use ('euclidean', etc.).
        mi_n_neighbors (int): Number of neighbors for mutual information calculation.
        random_state (int): Random seed for reproducibility.
        n_jobs (int): Number of parallel jobs for computation.
    """

    def __init__(self, n_shapelets=psutil.cpu_count(logical=False)*2,
                 n_shapelets_for_selection=500, #int, inf, or 'stratified'
                 n_ts_for_selection_per_class=100, #int, inf
                 sliding_window=50,
                 selection='mi_clf', #random, mi_clf, mi_reg, cluster
                 distance='euclidean',
                 mi_n_neighbors=100,
                 random_state=42, n_jobs=1,
                 **kwargs):
        """
        Initialize the ShapeletTreeStumpClassifier.

        Args:
            n_shapelets (int): Number of shapelets to extract.
            n_shapelets_for_selection (int): Number of shapelets to consider during selection.
            n_ts_for_selection_per_class (int): Number of time series per class for shapelet selection.
            sliding_window (int): Size of the sliding window for shapelet extraction.
            selection (str): Method for selecting shapelets ('random', 'mi_clf', 'cluster').
            distance (str): Distance metric to use ('euclidean', etc.).
            mi_n_neighbors (int): Number of neighbors for mutual information calculation.
            random_state (int): Random seed for reproducibility.
            n_jobs (int): Number of parallel jobs for computation.
            **kwargs: Additional arguments for the parent class.
        """
        self.n_shapelets = n_shapelets
        self.n_shapelets_for_selection = n_shapelets_for_selection
        self.n_ts_for_selection_per_class = n_ts_for_selection_per_class
        self.sliding_window = sliding_window
        self.selection = selection
        self.distance = distance
        self.mi_n_neighbors = mi_n_neighbors
        self.random_state = random_state
        self.n_jobs = n_jobs

        if "max_depth" in kwargs and kwargs["max_depth"] > 1:
            warnings.warn("max_depth must be 1")

        kwargs["max_depth"] = 1

        if selection not in ['random', 'mi_clf', 'cluster']:
            raise ValueError("'selection' must be 'random', 'mi_clf' or 'cluster'")

        super().__init__(**kwargs)

        kwargs |= {
            "n_shapelets": n_shapelets,
            "n_shapelets_for_selection": n_shapelets_for_selection,
            "n_ts_for_selection_per_class": n_ts_for_selection_per_class,
            "sliding_window": sliding_window,
            "selection": selection,
            "distance": distance,
            "mi_n_neighbors": mi_n_neighbors,
            "random_state": random_state,
            "n_jobs": n_jobs,
        }

    def fit(self, X, y, idx=None, context=None, sample_weight=None, check_input=True):
        """
        Fit the classifier to the given data.

        Args:
            X (array-like): Input time series data.
            y (array-like): Target labels.
            idx (slice, optional): Indices to select a subset of data.
            context (optional): Contextual information (not used).
            sample_weight (optional): Sample weights (not supported).
            check_input (bool): Whether to validate input data.

        Returns:
            self: Fitted classifier.
        """
        if idx is None:
            idx = slice(None)
        X = X[idx]
        y = y[idx]

        self.y_lims = [X.min(), X.max()]

        random.seed(self.random_state)
        if sample_weight is not None:
            raise UnsupportedError(f"sample_weight is not supported for {self.__class__.__name__}")

        self.st = Shapelets(n_shapelets=self.n_shapelets,
                            n_shapelets_for_selection=self.n_shapelets_for_selection,
                            n_ts_for_selection_per_class=self.n_ts_for_selection_per_class,
                            sliding_window=self.sliding_window,
                            selection=self.selection,
                            distance=self.distance,
                            mi_n_neighbors=self.mi_n_neighbors,
                            random_state=random.randint(0, 2**32-1),
                            n_jobs=self.n_jobs
                            )

        return super().fit(self.st.fit_transform(X, y), y=y, sample_weight=sample_weight, check_input=check_input)

    def apply(self, X, check_input=False):
        """
        Apply the fitted model to the input data.

        Args:
            X (array-like): Input time series data.
            check_input (bool): Whether to validate input data.

        Returns:
            array-like: Transformed data.
        """
        self.y_lims = [min(self.y_lims[0], X.min()), min(self.y_lims[1], X.max())]

        return super().apply(self.st.transform(X), check_input=check_input)

    def supports(self, data_type):
        """
        Check if the classifier supports the given data type.

        Args:
            data_type: Data type to check.

        Returns:
            bool: True if supported, False otherwise.
        """
        return data_type in [DATA_TYPE_TS]

    def get_rule(self, columns_names=None, scaler=None, float_precision: int | None = 3):
        """
        Generate a rule representation of the tree stump.

        Args:
            columns_names (list, optional): Column names for features.
            scaler (optional): Scaler for feature values (not supported).
            float_precision (int, optional): Precision for floating-point values.

        Returns:
            dict: Rule representation.
        """
        rule = {
            "feature_idx": self.feature_original[0],
            "threshold": self.threshold_original[0],
            "is_categorical": self.is_categorical,
            "samples": self.n_node_samples[0]
        }

        rule["feature_name"] = f"Shapelet_{rule['feature_idx']}"

        if scaler is not None:
            raise UnsupportedError(f"Scaler not supported for {self.__class__.__name__}")

        comparison = "<="
        not_comparison = ">"
        rounded_value = str(rule["threshold"]) if float_precision is None else round(rule["threshold"], float_precision)

        shape = self.st.shapelets[self.feature_original[0], 0]

        with tempfile312.NamedTemporaryFile(delete_on_close=False,
                                            delete=False,
                                            suffix=".png",
                                            mode="wb") as temp_file:
            plt.figure(figsize=(2, 1))
            plt.plot([i for i in range(shape.shape[0])], shape)
            plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
            plt.ylim(*self.y_lims)
            plt.xlim(0, shape.shape[0])
            plt.gca().tick_params(axis='both', which='both', length=2, labelsize=6)
            plt.gca().spines['right'].set_color('none')
            plt.gca().spines['top'].set_color('none')
            #plt.gca().spines['bottom'].set_position('zero')
            plt.savefig(temp_file, format="png", dpi=300, bbox_inches='tight', pad_inches=0)
            plt.close()

        rule["textual_rule"] = f"{self.distance}(TS, shp) {comparison} {rounded_value}\t{rule['samples']}"
        rule["blob_rule"] = f"{self.distance}(TS, shp) {comparison} {rounded_value}"
        rule["graphviz_rule"] = {
            "image": f'{temp_file.name}',
            "imagescale": "true",
            "imagepos": "bc",
            "label": f"{self.distance}(TS, shp) \u2264 {rounded_value}",
            "labelloc": "t",
            "fixedsize": "true",
            "width": "2",
            "height": "1.33",
            "shape": "none",
            "fontsize": "10",
        }

        rule["not_textual_rule"] = f"{self.distance}(TS, shp) {not_comparison} {rounded_value}"
        rule["not_blob_rule"] = f"{self.distance}(TS, shp) {not_comparison} {rounded_value}"
        rule["not_graphviz_rule"] = {
            "image": f'{temp_file.name}',
            "imagescale": "true",
            "label": f"{self.distance}(TS, shp) {not_comparison} {rounded_value}",
            "imagepos": "bc",
            "labelloc": "t",
            "fixedsize": "true",
            "width": "2",
            "height": "1.33",
            "shape": "none",
            "fontsize": "10",
        }

        return rule

    def node_to_dict(self):
        """
        Convert the tree node to a dictionary representation.

        Returns:
            dict: Dictionary representation of the node.
        """
        rule = super().node_to_dict() | {
            'stump_type': self.__class__.__module__,
            "feature_idx": self.feature_original[0],
            "threshold": self.threshold_original[0],
            "is_categorical": self.is_categorical,
            "samples": self.n_node_samples[0]
        }

        rule["feature_name"] = f"Shapelet_{rule['feature_idx']}"

        comparison = "<="
        not_comparison = ">"
        rounded_value = rule["threshold"]

        rule["textual_rule"] = f"{self.distance}(TS, shp) {comparison} {rounded_value}\t{rule['samples']}"
        rule["blob_rule"] = f"{self.distance}(TS, shp) {comparison} {rounded_value}"
        rule["graphviz_rule"] = {
            "image": f'None',
            "imagescale": "true",
            "imagepos": "bc",
            "label": f"{self.distance}(TS, shp) {comparison} {rounded_value}",
            "labelloc": "t",
            "fixedsize": "true",
            "width": "2",
            "height": "1.33",
            "shape": "none",
            "fontsize": "10",
        }

        rule["not_textual_rule"] = f"{self.distance}(TS, shp) {not_comparison} {rounded_value}"
        rule["not_blob_rule"] = f"{self.distance}(TS, shp) {not_comparison} {rounded_value}"
        rule["not_graphviz_rule"] = {
            "image": f'{None}',
            "imagescale": "true",
            "label": f"{self.distance}(TS, shp) {not_comparison} {rounded_value}",
            "imagepos": "bc",
            "labelloc": "t",
            "fixedsize": "true",
            "width": "2",
            "height": "1.33",
            "shape": "none",
            "fontsize": "10",
        }

        # shapelet transform stuff
        rule["shapelets"] = self.st.shapelets.tolist()
        rule["n_shapelets"] = self.st.n_shapelets
        rule["n_shapelets_for_selection"] = self.st.n_shapelets_for_selection
        rule["n_ts_for_selection_per_class"] = self.st.n_ts_for_selection_per_class
        rule["sliding_window"] = self.st.sliding_window
        rule["selection"] = self.st.selection
        rule["distance"] = self.st.distance
        rule["mi_n_neighbors"] = self.st.mi_n_neighbors
        rule["random_state"] = self.st.random_state
        rule["n_jobs"] = self.st.n_jobs
        rule["y_lims"] = self.y_lims

        return rule

    @classmethod
    def dict_to_node(cls, node_dict, X=None):
        self = cls(
            n_shapelets=node_dict["n_shapelets"],
            n_shapelets_for_selection=node_dict["n_shapelets_for_selection"],
            n_ts_for_selection_per_class=node_dict["n_ts_for_selection_per_class"],
            sliding_window=node_dict["sliding_window"],
            selection=node_dict["selection"],
            distance=node_dict["distance"],
            mi_n_neighbors=node_dict["mi_n_neighbors"],
            random_state=node_dict["random_state"],
            n_jobs=node_dict["n_jobs"]
        )

        self.st = Shapelets(
            n_shapelets=node_dict["n_shapelets"],
            n_shapelets_for_selection=node_dict["n_shapelets_for_selection"],
            n_ts_for_selection_per_class=node_dict["n_ts_for_selection_per_class"],
            sliding_window=node_dict["sliding_window"],
            selection=node_dict["selection"],
            distance=node_dict["distance"],
            mi_n_neighbors=node_dict["mi_n_neighbors"],
            random_state=node_dict["random_state"],
            n_jobs=node_dict["n_jobs"]
        )

        self.st.shapelets = np.array(node_dict["shapelets"])

        self.feature_original = np.zeros(3, dtype=int)
        self.threshold_original = np.zeros(3)
        self.n_node_samples = np.zeros(3, dtype=int)

        self.feature_original[0] = node_dict["feature_idx"]
        self.threshold_original[0] = node_dict["threshold"]
        self.n_node_samples[0] = node_dict["samples"]
        self.is_categorical = node_dict["is_categorical"]

        self.y_lims = node_dict["y_lims"]

        args = copy.deepcopy(node_dict["args"])
        self.is_oblique = args.pop("is_oblique")
        self.is_pivotal = args.pop("is_pivotal")
        self.unique_val_enum = args.pop("unique_val_enum")
        self.coefficients = args.pop("coefficients")
        self.kwargs = args

        return self