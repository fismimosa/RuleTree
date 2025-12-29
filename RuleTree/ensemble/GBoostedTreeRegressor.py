import copy

import numpy as np
from line_profiler_pycharm import profile
from sklearn.base import RegressorMixin
from sklearn.metrics import mean_squared_error

from RuleTree import RuleTreeRegressor
from RuleTree.base.RuleTreeBase import RuleTreeBase


class GBoostedTreeRegressor(RuleTreeBase, RegressorMixin):
    def __init__(self, base_estimator=RuleTreeRegressor(max_depth=3),
                 n_estimators=100, learning_rate=.1, loss='squared_loss', n_iter_no_change=None, tol=1e-4,):
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        if loss != 'squared_loss' and type(loss) is str:
            raise ValueError('loss must be squared_loss or callable')
        self.loss = loss
        self.n_iter_no_change = n_iter_no_change
        self.tol = tol

    def _get_base_prediction(self, y):
        return np.mean(y)

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.base_prediction_ = self._get_base_prediction(y)
        self.estimators_ = []
        residuals = y - self.base_prediction_

        patience = self.n_iter_no_change
        for _ in range(self.n_estimators):
            est = copy.deepcopy(self.base_estimator)
            est.fit(X, residuals)

            res_delta = self.learning_rate * est.predict(X)
            if patience is not None:
                if np.mean(res_delta) <= self.tol:
                    patience -= 1
                if patience == 0:
                    self.estimators_ = self.estimators_[:-self.n_iter_no_change]
                    break
            residuals -= res_delta
            self.estimators_.append(est)

        return self


    def predict(self, X: np.ndarray):
        prediction = np.ones((X.shape[0], ))*self.base_prediction_

        for est in self.estimators_:
            prediction += self.learning_rate * est.predict(X)

        return prediction