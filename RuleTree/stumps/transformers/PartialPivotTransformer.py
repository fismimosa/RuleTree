import warnings

import numpy as np
import psutil
from sklearn.base import clone

from RuleTree.base.RuleTreeBaseTransformer import RuleTreeBaseTransformer
from RuleTree.utils.define import DATA_TYPE_TABULAR
from RuleTree.utils.shapelet_transform.TabularShapelets import TabularShapelets


class PartialPivotTransformer(RuleTreeBaseTransformer):
    """
    Distances of the instances from partial pivots: instances of the node restricted to subsets of
    the features (see TabularShapelets for the parameters). After select, only the chosen partial
    pivot is kept.

    Parameters
    ----------
    scaler : TransformerMixin, default=None
        Scaler applied to the data before computing the distances, fitted on the whole data set.
    """

    supported_data_types = (DATA_TYPE_TABULAR,)
    refit_after_select = True

    def __init__(self,
                 n_shapelets=psutil.cpu_count(logical=False) * 2,
                 n_ts_for_selection=100,  # int, inf
                 n_features_strategy=2,
                 selection='random',  # random, cluster, all
                 distance='euclidean',
                 scaler=None,
                 use_combination=True,
                 random_state=42, n_jobs=1):
        self.n_shapelets = n_shapelets
        self.n_ts_for_selection = n_ts_for_selection
        self.n_features_strategy = n_features_strategy
        self.selection = selection
        self.distance = distance
        self.scaler = scaler
        self.use_combination = use_combination
        self.random_state = random_state
        self.n_jobs = n_jobs

    def _shapelets(self):
        return TabularShapelets(n_shapelets=self.n_shapelets,
                                n_ts_for_selection=self.n_ts_for_selection,
                                n_features_strategy=self.n_features_strategy,
                                selection=self.selection,
                                distance=self.distance,
                                random_state=self.random_state,
                                use_combination=self.use_combination,
                                n_jobs=self.n_jobs)

    def fit(self, X, y, idx=None, data_type=None, cache=None, sample_weight=None):
        self.scaler_ = clone(self.scaler).fit(X) if self.scaler is not None else None
        if idx is None:
            idx = slice(None)

        if sample_weight is not None:
            warnings.warn(f"sample_weight is not supported for {self.__class__.__name__}", Warning)

        self.st_ = self._shapelets().fit(self._scale(X[idx]), y[idx])
        self.numerical_, self.categorical_ = list(range(len(self.st_.shapelets))), []

        return self

    def _scale(self, X):
        return self.scaler_.transform(X) if self.scaler_ is not None else X

    def transform(self, X):
        return self.st_.transform(self._scale(X))

    def select(self, feature):
        self.st_._optimize_memory(np.array([feature]))
        self.numerical_, self.categorical_ = [0], []
        return 0

    def get_state(self):
        return {"shapelets": self.st_.shapelets.tolist()}

    def set_state(self, state):
        self.scaler_ = None  # not saved: a fitted scaler is not serializable
        self.st_ = self._shapelets()
        self.st_.shapelets = np.array(state["shapelets"])
        return self
