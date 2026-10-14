import numpy as np
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.preprocessing import StandardScaler

from RuleTree.base.RuleTreeBaseTransformer import RuleTreeBaseTransformer, get_feature_types
from RuleTree.exceptions import NoSplitFoundWarning
from RuleTree.stumps.splitters.PivotSplit import PivotSplit
from RuleTree.utils.define import MODEL_TYPE_CLF


class PivotTransformer(RuleTreeBaseTransformer):
    """
    Distances of the instances from candidate pivots: instances of the node that describe a class
    (medoids) or discriminate between classes, selected by PivotSplit on the numerical features.
    After select, only the chosen pivot is kept.

    Parameters
    ----------
    ml_task : str, default=MODEL_TYPE_CLF
        Task of the stump (MODEL_TYPE_CLF or MODEL_TYPE_REG), used to select the pivots.
    distance_measure : str or callable, default='euclidean'
        Distance between the (standardized) instances, as in sklearn.metrics.pairwise_distances.
    k, reg_contrast_lambda, reg_max_thresholds
        Parameters of the pivot selection for regression (see PivotSplit).
    tree_params : dict, default=None
        Parameters of the decision trees used by PivotSplit to find the discriminative pivots.
    """

    def __init__(self, ml_task=MODEL_TYPE_CLF, distance_measure='euclidean', k=None, reg_contrast_lambda=.2,
                 reg_max_thresholds=None, tree_params=None):
        self.ml_task = ml_task
        self.distance_measure = distance_measure
        self.k = k
        self.reg_contrast_lambda = reg_contrast_lambda
        self.reg_max_thresholds = reg_max_thresholds
        self.tree_params = tree_params

    def fit(self, X, y, idx=None, data_type=None, cache=None, sample_weight=None):
        if idx is None:
            idx = np.arange(len(X))
        self.num_pre_transformed_, self.cat_pre_transformed_ = get_feature_types(X, cache, data_type)

        if len(self.num_pre_transformed_) == 0:
            raise NoSplitFoundWarning(f"{self.__class__.__name__} needs numerical features")

        self.scaler_, distance_matrix = _pivot_cache(X, self.num_pre_transformed_, self.distance_measure, data_type,
                                                     cache)
        X_scaled = self.scaler_.transform(X[idx][:, self.num_pre_transformed_])

        self.pivot_split_ = PivotSplit(ml_task=self.ml_task, k=self.k, reg_contrast_lambda=self.reg_contrast_lambda,
                                       reg_max_thresholds=self.reg_max_thresholds, **(self.tree_params or {}))
        self.pivot_split_.fit(X_scaled, y[idx], distance_matrix[np.ix_(idx, idx)], self.distance_measure, idx,
                              sample_weight=None if sample_weight is None else sample_weight[idx])

        self.candidates_ = self.pivot_split_.X_candidates
        self.candidate_names_ = self.pivot_split_.get_candidates_names()
        self.numerical_, self.categorical_ = list(range(len(self.candidates_))), []

        return self

    def transform(self, X):
        return pairwise_distances(self.scaler_.transform(X[:, self.num_pre_transformed_]), self.candidates_,
                                  metric=self.distance_measure)

    def select(self, feature):
        self.candidates_ = self.candidates_[[feature]]
        self.candidate_names_ = self.candidate_names_[[feature]]
        self.numerical_, self.categorical_ = [0], []
        return 0

    def feature_id(self, feature):
        return f'{self.candidate_names_[feature]}'

    def get_state(self):
        return {
            'num_pre_transformed': list(self.num_pre_transformed_),
            'cat_pre_transformed': list(self.cat_pre_transformed_),
            'X_split_instance': self.candidates_[0],
            'pivot_name': self.feature_id(0),
            'scaler': {
                'mean_': self.scaler_.mean_.tolist(),
                'scale_': self.scaler_.scale_.tolist(),
                'var_': self.scaler_.var_.tolist(),
                'n_samples_seen_': self.scaler_.n_samples_seen_
            },
        }

    def set_state(self, state):
        self.num_pre_transformed_ = state['num_pre_transformed']
        self.cat_pre_transformed_ = state['cat_pre_transformed']
        self.candidates_ = np.array([state['X_split_instance']])
        self.candidate_names_ = np.array([int(state['pivot_name'])])

        self.scaler_ = StandardScaler()
        self.scaler_.mean_ = np.array(state['scaler']['mean_'])
        self.scaler_.scale_ = np.array(state['scaler']['scale_'])
        self.scaler_.var_ = np.array(state['scaler']['var_'])
        self.scaler_.n_samples_seen_ = state['scaler']['n_samples_seen_']

        return self


def _pivot_cache(X, numerical, distance_measure, data_type, cache):
    """
    Scaler and distance matrix of the whole data set X, stored in the cache of the stump so that
    they are computed once per data type and distance.
    """
    cache = {} if cache is None else cache
    key = ("pivot", data_type, distance_measure)
    if key not in cache:
        scaler = StandardScaler().fit(X[:, numerical])
        # TODO: la matrice delle distanze è calcolata sui dati NON standardizzati, mentre transform
        #  (e quindi lo split) calcola le distanze dai pivot sui dati standardizzati con lo scaler.
        #  I pivot candidati vengono quindi scelti con una scala e la soglia dello split con un'altra:
        #  con feature su scale molto diverse le due distanze possono non essere coerenti.
        #  Una possibile correzione è calcolare anche questa matrice su scaler.transform(X[:, numerical]),
        #  ma cambierebbe i pivot scelti e quindi i risultati degli alberi con gli stump Pivot.
        cache[key] = (scaler, pairwise_distances(X[:, numerical], metric=distance_measure))

    return cache[key]
