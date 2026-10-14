import random
import warnings
from typing import Optional

import numpy as np
import psutil
from numba import UnsupportedError


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

    supported_data_types = (DATA_TYPE_TS,)

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

    def fit(self, data, y, idx=None, context=None, data_type=None, sample_weight=None, check_input=True):
        """
        Fit the regressor to the given data.

        Args:
            data (dict | array-like): Whole data set, as ``{data_type: array}``.
            y (array-like): Target of the whole data set.
            idx (array-like, optional): Indices of the instances to fit on. Defaults to all.
            context (optional): Tree being built (not used).
            data_type (str, optional): Data type to use. Defaults to the first supported type in data.
            sample_weight (optional): Sample weights (not supported).
            check_input (bool): Whether to validate input data.

        Returns:
            self: Fitted regressor.
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
        return ShapeletTreeStumpClassifier.get_rule(self, columns_names, scaler, float_precision)

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

