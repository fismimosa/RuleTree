import copy
import random
import warnings
from typing import Optional

import numpy as np
import psutil
import tempfile312
from numba import UnsupportedError

from matplotlib import pyplot as plt

from RuleTree.stumps.classification import ShapeletTreeStumpClassifier
from RuleTree.stumps.regression.DecisionTreeStumpRegressor import DecisionTreeStumpRegressor
from RuleTree.utils.define import DATA_TYPE_TS
from RuleTree.utils.shapelet_transform.Shapelets import Shapelets


class ShapeletTreeStumpRegressor(DecisionTreeStumpRegressor):
    """
    A regression tree stump that uses shapelets as features for time series data.

    This class extends `DecisionTreeStumpRegressor` and incorporates shapelet-based
    feature extraction for regression tasks. It supports various selection methods
    for shapelets and allows customization of parameters such as the number of shapelets,
    sliding window size, and distance metric.

    Attributes:
        n_shapelets (int): Number of shapelets to extract.
        n_shapelets_for_selection (int): Number of shapelets to consider during selection.
        n_ts_for_selection_per_class (int): Number of time series per class for shapelet selection.
        sliding_window (int): Size of the sliding window for shapelet extraction.
        selection (str): Method for selecting shapelets ('random', 'mi_reg', 'cluster').
        distance (str): Distance metric to use ('euclidean', etc.).
        mi_n_neighbors (int): Number of neighbors for mutual information calculation.
        random_state (int): Random seed for reproducibility.
        n_jobs (int): Number of parallel jobs for computation.
    """

    def __init__(self, n_shapelets=psutil.cpu_count(logical=False)*2,
                 n_shapelets_for_selection=500, #int, inf, or 'stratified'
                 n_ts_for_selection_per_class=100, #int, inf
                 sliding_window=50,
                 selection='mi_reg', #random, mi_clf, mi_reg, cluster
                 distance='euclidean',
                 mi_n_neighbors=100,
                 random_state=42, n_jobs=1,
                 **kwargs):
        """
        Initialize the ShapeletTreeStumpRegressor.

        Args:
            n_shapelets (int): Number of shapelets to extract.
            n_shapelets_for_selection (int): Number of shapelets to consider during selection.
            n_ts_for_selection_per_class (int): Number of time series per class for shapelet selection.
            sliding_window (int): Size of the sliding window for shapelet extraction.
            selection (str): Method for selecting shapelets ('random', 'mi_reg', 'cluster').
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

        if selection not in ['random', 'mi_reg', 'cluster']:
            raise ValueError("'selection' must be 'random', 'mi_reg' or 'cluster'")

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

    def fit(self, X, y, idx=None, context=None, sample_weight=None, check_input=True):
        """
        Fit the regressor to the given data.

        Args:
            X (array-like): Input time series data.
            y (array-like): Target values.
            idx (slice, optional): Indices to select a subset of data.
            context (optional): Contextual information (not used).
            sample_weight (optional): Sample weights (not supported).
            check_input (bool): Whether to validate input data.

        Returns:
            self: Fitted regressor.
        """
        if idx is None:
            idx = slice(None)
        X = X[idx]
        y = y[idx]

        random.seed(self.random_state)
        if sample_weight is not None:
            raise UnsupportedError(f"sample_weight is not supported for {self.__class__.__name__}")

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
        return super().apply(self.st.transform(X), check_input=check_input)

    def supports(self, data_type):
        """
        Check if the regressor supports the given data type.

        Args:
            data_type: Data type to check.

        Returns:
            bool: True if supported, False otherwise.
        """
        return data_type in [DATA_TYPE_TS]

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
        return ShapeletTreeStumpClassifier.get_rule(self, columns_names, scaler, float_precision)

    def node_to_dict(self):
        """
        Convert the tree node to a dictionary representation.

        Returns:
            dict: Dictionary representation of the node.
        """
        return ShapeletTreeStumpClassifier.node_to_dict(self)

    @classmethod
    def dict_to_node(cls, node_dict, X=None):
        """
        Create a tree node from a dictionary representation.

        Args:
            node_dict (dict): Dictionary representation of the node.
            X (array-like, optional): Input data (not used).

        Returns:
            ShapeletTreeStumpRegressor: Reconstructed tree node.
        """
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

        self.st.shapelets = np.array(node_dict.get("shapelets", []))

        self.feature_original = None
        if 'feature_idx' in node_dict:
            self.feature_original = np.ones(3, dtype=int) * -2
            self.feature_original[0] = node_dict.get('feature_idx', -2)

        self.threshold_original = None
        if 'threshold' in node_dict:
            self.threshold_original = np.ones(3) * -2
            self.threshold_original[0] = node_dict.get('threshold', -2)

        args = copy.deepcopy(node_dict["args"])
        self.kwargs = args

        return self

    def update_statistics(self, X, y, idx=None, context=None, sample_weight=None, check_input=True):
        super().update_statistics(self.st.transform(X), y, idx, context, sample_weight, check_input)