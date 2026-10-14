import random
import warnings
from typing import Optional

import numpy as np
import psutil
import tempfile312
from numba import UnsupportedError

from matplotlib import pyplot as plt

from RuleTree.stumps.classification.DecisionTreeStumpClassifier import DecisionTreeStumpClassifier
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

    supported_data_types = (DATA_TYPE_TS,)

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

        self.st = Shapelets(n_shapelets=self.n_shapelets,
                            n_shapelets_for_selection=self.n_shapelets_for_selection,
                            n_ts_for_selection_per_class=self.n_ts_for_selection_per_class,
                            sliding_window=self.sliding_window,
                            selection=self.selection,
                            distance=self.distance,
                            mi_n_neighbors=self.mi_n_neighbors,
                            random_state=random_state,
                            n_jobs=self.n_jobs
                            )

        self.kwargs |= {
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

    def fit(self, data, y, idx=None, context=None, data_type=None, sample_weight=None, check_input=True):
        """
        Fit the classifier to the given data.

        Args:
            data (dict | array-like): Whole data set, as ``{data_type: array}``.
            y (array-like): Target of the whole data set.
            idx (array-like, optional): Indices of the instances to fit on. Defaults to all.
            context (optional): Tree being built (not used).
            data_type (str, optional): Data type to use. Defaults to the first supported type in data.
            sample_weight (optional): Sample weights (not supported).
            check_input (bool): Whether to validate input data.

        Returns:
            self: Fitted classifier.
        """
        X, _ = self._resolve_fit_data(data, data_type)
        if idx is None:
            idx = slice(None)
        X = X[idx]
        y = y[idx]

        random.seed(self.random_state)
        if sample_weight is not None:
            raise UnsupportedError(f"sample_weight is not supported for {self.__class__.__name__}")

        X_transform = self.st.fit_transform(X, y)
        self.numerical, self.categorical = list(range(X_transform.shape[1])), []

        return self._fit_matrix(X_transform, y, check_input=check_input)

    def _transform(self, X):
        """Distance of each time series from each shapelet."""
        return self.st.transform(X)

    def get_rule(self, columns_names=None, scaler=None, float_precision: Optional[int] = 3):
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
            "is_categorical": False,
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
            plt.plot(list(range(shape.shape[0])), shape)
            plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
            plt.xlim(0, shape.shape[0])
            plt.gca().tick_params(axis='both', which='both', length=2, labelsize=6)
            plt.gca().spines['right'].set_color('none')
            plt.gca().spines['top'].set_color('none')
            plt.savefig(temp_file, format="png", dpi=300, bbox_inches='tight', pad_inches=0)
            plt.close()

        rule["textual_rule"] = f"{self.distance}(TS, shp) {comparison} {rounded_value}"
        rule["blob_rule"] = f"{self.distance}(TS, shp) {comparison} {rounded_value}"
        rule["graphviz_rule"] = {
            "image": f'{temp_file.name}',
            "imagescale": "true",
            "imagepos": "bc",
            "label": f"{self.distance}(TS, shp) <= {rounded_value}",
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
        Convert the stump to a dictionary representation: the split, the constructor arguments
        and the selected shapelets.

        Returns:
            dict: Dictionary representation of the node.
        """
        rule = super().node_to_dict()
        if self.feature_original is not None:
            rule["shapelets"] = self.st.shapelets.tolist()

        return rule

    @classmethod
    def dict_to_node(cls, node_dict, X=None):
        """
        Create a stump from a dictionary created by node_to_dict.

        Args:
            node_dict (dict): Dictionary representation of the node.
            X (array-like, optional): Not used.

        Returns:
            The reconstructed stump.
        """
        self = super().dict_to_node(node_dict)
        if "shapelets" in node_dict:
            self.st.shapelets = np.array(node_dict["shapelets"])

        return self

