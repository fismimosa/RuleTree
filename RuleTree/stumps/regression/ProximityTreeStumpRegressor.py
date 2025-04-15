import copy
import random
import warnings

import numpy as np
import psutil
import tempfile312
from matplotlib import pyplot as plt
from numba import UnsupportedError

from RuleTree.stumps.classification import ProximityTreeStumpClassifier
from RuleTree.stumps.regression import DecisionTreeStumpRegressor
from RuleTree.utils.define import DATA_TYPE_TS
from RuleTree.utils.shapelet_transform.Shapelets import Shapelets


class ProximityTreeStumpRegressor(DecisionTreeStumpRegressor):
    """A decision stump regressor for time series data using shapelet-based proximity measures.

    This regressor transforms time series data using shapelets (subsequences of time series)
    and creates regression rules based on the distances between the input time series and
    extracted shapelets. It predicts continuous target values rather than class labels.

    Parameters
    ----------
    n_shapelets : int, default=cpu_count*2
        The number of shapelets to be used for regression.
    n_shapelets_for_selection : int or str, default=500
        Number of shapelets to select or 'stratified' for stratified selection.
    n_ts_for_selection_per_class : int, default=100
        Number of time series per class to use for shapelet extraction.
    sliding_window : int, default=50
        The length of the sliding window used to extract shapelets from time series.
    selection : str, default='mi_clf'
        Method for shapelet selection. Options: 'random', 'mi_clf', 'mi_reg', 'cluster'.
    distance : str, default='euclidean'
        Distance metric used for comparing shapelets to time series.
    mi_n_neighbors : int, default=100
        Number of neighbors for mutual information calculation.
    random_state : int, default=42
        Seed for random number generation for reproducibility.
    n_jobs : int, default=1
        Number of parallel jobs to run for shapelet extraction.
    **kwargs
        Additional parameters for the base DecisionTreeStumpRegressor.
    """
    def __init__(self,
                 n_shapelets=psutil.cpu_count(logical=False) * 2,
                 n_shapelets_for_selection=500,  # int, inf, or 'stratified'
                 n_ts_for_selection_per_class=100,  # int, inf
                 sliding_window=50,
                 selection='mi_clf',  # random, mi_clf, mi_reg, cluster
                 distance='euclidean',
                 mi_n_neighbors=100,
                 random_state=42, n_jobs=1,
                 **kwargs):
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
        """Fit the ProximityTreeStumpRegressor on the training data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_timepoints)
            The time series data.
        y : array-like of shape (n_samples,)
            The target continuous values.
        idx : array-like or slice, default=None
            Indices of the samples to be used for training.
        context : object, default=None
            Additional context information (not used).
        sample_weight : array-like, default=None
            Sample weights (not supported for this regressor).
        check_input : bool, default=True
            Whether to check the input parameters.

        Returns
        -------
        self : object
            Fitted estimator.

        Raises
        ------
        UnsupportedError
            If sample_weight is provided.
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

        X_dist = self.st.fit_transform(X, y)
        actual_n_shapelets = X_dist.shape[1]
        X_bool = np.zeros((X.shape[0], actual_n_shapelets*(actual_n_shapelets-1)), dtype=bool)

        c = 0

        for i in range(actual_n_shapelets):
            for j in range(i+1, actual_n_shapelets):
                X_bool[:, c] = X_dist[:, i] <= X_dist[:, j]
                c += 1

        return super().fit(X_bool, y=y, sample_weight=sample_weight, check_input=check_input)

    def apply(self, X, check_input=False):
        """Apply the decision tree stump to X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_timepoints)
            The time series data for regression.
        check_input : bool, default=False
            Whether to check the input parameters.

        Returns
        -------
        X_leaves : array-like of shape (n_samples,)
            For each datapoint x in X, returns the leaf node it ends up in.
        """
        self.y_lims = [min(self.y_lims[0], X.min()), min(self.y_lims[1], X.max())]
        X_dist = self.st.transform(X)
        actual_n_shapelets = X_dist.shape[1]
        X_bool = np.zeros((X.shape[0], actual_n_shapelets * (actual_n_shapelets - 1)), dtype=bool)

        c = 0

        for i in range(actual_n_shapelets):
            for j in range(i + 1, actual_n_shapelets):
                X_bool[:, c] = X_dist[:, i] <= X_dist[:, j]
                c += 1


        return super().apply(X_bool, check_input=check_input)

    def supports(self, data_type):
        """Check if the regressor supports the given data type.

        Parameters
        ----------
        data_type : int
            Data type identifier.

        Returns
        -------
        bool
            True if the data type is supported, False otherwise.
        """
        return data_type in [DATA_TYPE_TS]


    def get_rule(self, columns_names=None, scaler=None, float_precision: int | None = 3):
        """Generate a human-readable rule representation of the decision stump.

        Delegates to the ProximityTreeStumpClassifier implementation.

        Parameters
        ----------
        columns_names : list, default=None
            Names of the features (not used for time series).
        scaler : object, default=None
            Scaler used for normalization (not supported).
        float_precision : int or None, default=3
            Number of decimal places to use for floating-point values.

        Returns
        -------
        dict
            Dictionary containing textual and visual representations of the rule.
        """
        return ProximityTreeStumpClassifier.get_rule(self, columns_names, scaler, float_precision)

    def node_to_dict(self):
        """Convert the decision stump to a dictionary representation.

        Delegates to the ProximityTreeStumpClassifier implementation.

        Returns
        -------
        dict
            Dictionary representing the decision stump, including shapelet information.
        """
        return ProximityTreeStumpClassifier.node_to_dict(self)

    @classmethod
    def dict_to_node(cls, node_dict, X=None):
        """Convert a dictionary representation back to a decision stump.

        This method is used for model deserialization.

        Parameters
        ----------
        node_dict : dict
            Dictionary containing the serialized decision stump.
        X : array-like, default=None
            The time series data (not used).

        Returns
        -------
        ProximityTreeStumpRegressor
            The reconstructed decision stump regressor.
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

        self.y_lims = node_dict["y_lims"]

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

        return self
