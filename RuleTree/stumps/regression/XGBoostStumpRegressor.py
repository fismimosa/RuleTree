import numpy
import numpy as np

from RuleTree.exceptions import NoSplitFoundWarning
from RuleTree.stumps.regression import DecisionTreeStumpRegressor
from RuleTree.utils.tree_utils import xgboost_similarity_score_regression, xgboost_cover_regression

from line_profiler_pycharm import profile

class XGBoostStumpRegressor(DecisionTreeStumpRegressor):
    def __init__(self, lam, approximation=None, n_quantiles=32, min_cover=1, n_jobs=1):
        super().__init__()
        self.lam = lam
        self.n_jobs = n_jobs
        if approximation is not None and approximation not in ['quantile']:
            raise ValueError('approximation must be None or quantile')
        self.approximation = approximation
        self.n_quantiles = n_quantiles
        self.min_cover = min_cover
        self.kwargs |= {
            'lam': self.lam,
            'approximation': self.approximation,
            'n_quantiles': self.n_quantiles,
            'n_jobs': self.n_jobs,
            'min_cover': self.min_cover,
        }

    def _get_cover_similarity_fun(self):
        return xgboost_cover_regression, xgboost_similarity_score_regression

    def get_threshold_values(self, X, feature_idx):
        unique_val = np.unique(X[:, feature_idx])
        if feature_idx in self.categorical:
            return unique_val
        else:
            if self.approximation is None or len(unique_val) <= self.n_quantiles:
                values = np.sort(unique_val)
                return (values[:-1] + values[1:]) / 2
            else:
                return numpy.quantile(X[:, feature_idx], np.linspace(0, 1, self.n_quantiles))

    def fit(self, X, y, idx=None, context=None, sample_weight=None, check_input=True):
        if idx is None:
            idx = slice(None)
        X = X[idx]
        y = y[idx]

        cover_fn, sim_fn = self._get_cover_similarity_fun()

        if hasattr(context, 'categorical'):
            self.categorical = context.categorical
            self.numerical = context.numerical
        else:
            self.feature_analysis(X, y)
            context.categorical = self.categorical
            context.numerical = self.numerical

        root_cover = cover_fn(y)
        root_similarity = sim_fn(y, root_cover, self.lam)
        best_gain, best_feature_idx, best_threshold, best_categorical = -np.inf, -1, -1, False

        for feature_idx in range(X.shape[1]):
            for threshold in self.get_threshold_values(X, feature_idx):
                if feature_idx in self.numerical:
                    cond_l = X[:, feature_idx] <= threshold
                    cond_r = ~cond_l
                else:
                    cond_l = X[:, feature_idx] == threshold
                    cond_r = ~cond_l

                cover_l = cover_fn(y[cond_l])
                cover_r = cover_fn(y[cond_r])
                if cover_l < self.min_cover or cover_r < self.min_cover:
                    continue
                similarity_l = xgboost_similarity_score_regression(y[cond_l], cover_l, self.lam)
                similarity_r = xgboost_similarity_score_regression(y[cond_r], cover_r, self.lam)

                gain = similarity_l + similarity_r - root_similarity

                if gain > best_gain:
                    best_gain = gain
                    best_feature_idx = feature_idx
                    best_threshold = threshold
                    best_categorical = feature_idx in self.categorical

        self.threshold_original = [best_threshold, 0, 0]
        self.feature_original = [best_feature_idx, 0, 0]
        self.is_categorical = best_categorical
        if best_feature_idx == -1:
            raise NoSplitFoundWarning('No split found')
        self.update_statistics(X, y, slice(None), context=context, sample_weight=sample_weight, check_input=check_input)

        return self

    def update_statistics(self, X, y, idx=None, context=None, sample_weight=None, check_input=True):
        X = X[idx]
        y = y[idx]

        if self.is_categorical:
            idx_l = X[:, self.feature_original[0]] == self.threshold_original[0]
            idx_r = X[:, self.feature_original[0]] != self.threshold_original[0]
        else:
            idx_l = X[:, self.feature_original[0]] <= self.threshold_original[0]
            idx_r = X[:, self.feature_original[0]] > self.threshold_original[0]

        cover_fn, sim_fn = self._get_cover_similarity_fun()

        if hasattr(context, 'categorical'):
            self.categorical = context.categorical
            self.numerical = context.numerical
        else:
            self.feature_analysis(X, y)
            context.categorical = self.categorical
            context.numerical = self.numerical

        root_cover = cover_fn(y)
        self.root_similarity = sim_fn(y, root_cover, self.lam)
        cover_l = cover_fn(y[idx_l])
        cover_r = cover_fn(y[idx_r])
        self.similarity_l = xgboost_similarity_score_regression(y[idx_l], cover_l, self.lam)
        self.similarity_r = xgboost_similarity_score_regression(y[idx_r], cover_r, self.lam)
        self.gain = self.similarity_l + self.similarity_r - self.root_similarity

