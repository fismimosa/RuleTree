"""ShapeletTransformer: distances of time series from shapelets."""

import psutil
import numpy as np
from numba import UnsupportedError

from RuleTree.base.RuleTreeBaseTransformer import RuleTreeBaseTransformer
from RuleTree.utils.define import DATA_TYPE_TS
from RuleTree.utils.shapelet_transform.Shapelets import Shapelets


class ShapeletTransformer(RuleTreeBaseTransformer):
    """
    Distances of the time series from shapelets selected among the subsequences of the time series
    in the node (see Shapelets for the parameters).
    """

    supported_data_types = (DATA_TYPE_TS,)

    def __init__(self, n_shapelets=psutil.cpu_count(logical=False) * 2,
                 n_shapelets_for_selection=500,  # int, inf, or 'stratified'
                 n_ts_for_selection_per_class=100,  # int, inf
                 sliding_window=50,
                 selection='mi_clf',  # random, mi_clf, mi_reg, cluster
                 distance='euclidean',
                 mi_n_neighbors=100,
                 random_state=42, n_jobs=1):
        """Initialize the ShapeletTransformer; see Shapelets for the parameters."""
        self.n_shapelets = n_shapelets
        self.n_shapelets_for_selection = n_shapelets_for_selection
        self.n_ts_for_selection_per_class = n_ts_for_selection_per_class
        self.sliding_window = sliding_window
        self.selection = selection
        self.distance = distance
        self.mi_n_neighbors = mi_n_neighbors
        self.random_state = random_state
        self.n_jobs = n_jobs

    def _shapelets(self):
        """
        A new, unfitted Shapelets transform with the parameters of the transformer.

        Returns
        -------
        st : Shapelets
        """
        return Shapelets(n_shapelets=self.n_shapelets,
                         n_shapelets_for_selection=self.n_shapelets_for_selection,
                         n_ts_for_selection_per_class=self.n_ts_for_selection_per_class,
                         sliding_window=self.sliding_window,
                         selection=self.selection,
                         distance=self.distance,
                         mi_n_neighbors=self.mi_n_neighbors,
                         random_state=self.random_state,
                         n_jobs=self.n_jobs)

    def fit(self, X, y, idx=None, data_type=None, cache=None, sample_weight=None):
        """
        Select the shapelets among the subsequences of the time series in the node.

        Parameters
        ----------
        X : numpy.ndarray
            Data of the transformer's type, all the instances.
        y : numpy.ndarray
            Target, all the instances.
        idx : numpy.ndarray, default=None
            Indices of the instances in the node. If None, all the instances.
        data_type : str, default=None
            Type of X, used as key of the cached values.
        cache : dict, default=None
            Values shared among the splits of the tree.
        sample_weight : numpy.ndarray, default=None
            Sample weights, all the instances.

        Returns
        -------
        self : ShapeletTransformer

        Raises
        ------
        UnsupportedError
            If sample_weight is given.
        """
        if sample_weight is not None:
            raise UnsupportedError(f"sample_weight is not supported for {self.__class__.__name__}")
        if idx is None:
            idx = slice(None)

        self.st_ = self._shapelets().fit(X[idx], y[idx])
        self.numerical_, self.categorical_ = list(range(len(self.st_.shapelets))), []

        return self

    def transform(self, X):
        """
        Distance of each time series from each shapelet.

        Parameters
        ----------
        X : numpy.ndarray
            Time series, of shape (n_samples, 1, length).

        Returns
        -------
        X_transformed : numpy.ndarray of shape (n_samples, n_shapelets)
        """
        return self.st_.transform(X)

    def get_state(self):
        """
        Fitted state: the shapelets.

        Returns
        -------
        state : dict
            JSON-friendly dictionary, restored by set_state.
        """
        return {"shapelets": self.st_.shapelets.tolist()}

    def set_state(self, state):
        """
        Restore the fitted state saved by get_state.

        Parameters
        ----------
        state : dict
            Dictionary created by get_state.

        Returns
        -------
        self : ShapeletTransformer
        """
        self.st_ = self._shapelets()
        self.st_.shapelets = np.array(state["shapelets"])
        return self
