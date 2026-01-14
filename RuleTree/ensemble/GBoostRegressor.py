import copy

import numpy as np
from line_profiler_pycharm import profile
from sklearn.base import RegressorMixin
from sklearn.metrics import mean_squared_error

from RuleTree import RuleTreeRegressor
from RuleTree.base.RuleTreeBase import RuleTreeBase


class GBoostRegressor(RuleTreeBase, RegressorMixin):
    def __init__(self, base_estimator=RuleTreeRegressor(max_depth=3),
                 n_estimators=100, learning_rate=.1, loss='squared_loss', patience=5, val_size=.15,
                 random_state=42):
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        if loss != 'squared_loss' and type(loss) is str:
            raise ValueError('loss must be squared_loss or callable')
        self.loss = loss
        self.patience = patience
        self.random_state = random_state
        self.val_size = val_size

    def _get_base_prediction(self, y):
        return np.mean(y)

    def fit(self, X: np.ndarray, y: np.ndarray):
        np.random.seed(self.random_state)
        n_val = int(self.val_size * X.shape[0])
        idx = np.random.permutation(X.shape[0])
        val_idx = idx[:n_val]
        train_idx = idx[n_val:]

        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        self.base_prediction_ = self._get_base_prediction(y_train)
        self.estimators_ = []
        residuals = y_train - self.base_prediction_
        residuals_val = y_val - self.base_prediction_
        prev_residuals_val = np.inf

        wait = 0
        for _ in range(self.n_estimators):
            est = copy.deepcopy(self.base_estimator)
            est.fit(X_train, residuals)

            res_delta = self.learning_rate * est.predict(X_train)
            res_delta_val = self.learning_rate * est.predict(X_val)
            residuals -= res_delta
            residuals_val -= res_delta_val

            self.estimators_.append(est)
            if np.sum(np.abs(residuals_val)) <= prev_residuals_val:
                wait = 0
            else:
                wait += 1
                if wait >= self.patience:
                    self.estimators_ = self.estimators_[:-self.patience+1]
                    break
            prev_residuals_val = np.sum(np.abs(residuals_val))

        return self


    def predict(self, X: np.ndarray):
        prediction = np.ones((X.shape[0], ))*self.base_prediction_

        for est in self.estimators_:
            prediction += self.learning_rate * est.predict(X)

        return prediction