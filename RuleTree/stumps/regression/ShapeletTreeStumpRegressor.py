from typing import Optional

import psutil

from RuleTree.stumps.classification.ShapeletTreeStumpClassifier import ShapeletTreeStumpClassifier
from RuleTree.stumps.regression.DecisionTreeStumpRegressor import DecisionTreeStumpRegressor
from RuleTree.stumps.transformers.ShapeletTransformer import ShapeletTransformer


class ShapeletTreeStumpRegressor(DecisionTreeStumpRegressor):
    """
    Shapelet-based stump for regression of time series: the split compares the distance of the time
    series from a shapelet, a subsequence selected among the ones of the time series in the node.

    It is made of a ShapeletTransformer and a RegressionSplitter.

    Parameters
    ----------
    n_shapelets, n_shapelets_for_selection, n_ts_for_selection_per_class, sliding_window, selection, distance,
    mi_n_neighbors, random_state, n_jobs
        Parameters of the shapelet selection (see Shapelets).
    criterion, min_samples_leaf, min_weight_fraction_leaf, max_features, min_impurity_decrease
        As in DecisionTreeStumpRegressor.
    """

    def __init__(self, n_shapelets=psutil.cpu_count(logical=False)*2,
                 n_shapelets_for_selection=500,  # int, inf, or 'stratified'
                 n_ts_for_selection_per_class=100,  # int, inf
                 sliding_window=50,
                 selection='mi_reg',  # random, mi_reg, cluster
                 distance='euclidean',
                 mi_n_neighbors=100,
                 random_state=42, n_jobs=1,
                 criterion='squared_error', min_samples_leaf=1, min_weight_fraction_leaf=0.0,
                 max_features=None, min_impurity_decrease=0.0):
        if selection not in ['random', 'mi_reg', 'cluster']:
            raise ValueError("'selection' must be 'random', 'mi_reg' or 'cluster'")

        self.n_shapelets = n_shapelets
        self.n_shapelets_for_selection = n_shapelets_for_selection
        self.n_ts_for_selection_per_class = n_ts_for_selection_per_class
        self.sliding_window = sliding_window
        self.selection = selection
        self.distance = distance
        self.mi_n_neighbors = mi_n_neighbors
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.criterion = criterion
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_features = max_features
        self.min_impurity_decrease = min_impurity_decrease

    def _splitter_params(self):
        # random_state is the one of the shapelet selection
        return super()._splitter_params() | {"random_state": None}

    def _make_transformer(self):
        return ShapeletTransformer(n_shapelets=self.n_shapelets,
                                   n_shapelets_for_selection=self.n_shapelets_for_selection,
                                   n_ts_for_selection_per_class=self.n_ts_for_selection_per_class,
                                   sliding_window=self.sliding_window,
                                   selection=self.selection,
                                   distance=self.distance,
                                   mi_n_neighbors=self.mi_n_neighbors,
                                   random_state=self.random_state,
                                   n_jobs=self.n_jobs)

    @property
    def st(self):
        """The fitted shapelet transform."""
        return self.transformer_.st_

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
