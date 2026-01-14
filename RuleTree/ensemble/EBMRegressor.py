import copy
from itertools import combinations

import numpy as np
from sklearn.base import RegressorMixin
from sklearn.metrics import precision_recall_fscore_support

from RuleTree import RuleTreeRegressor
from RuleTree.base.RuleTreeBase import RuleTreeBase


class EBMRegressor(RuleTreeBase, RegressorMixin):
    def __init__(self, n_iterations=100, base_estimator=RuleTreeRegressor(max_depth=6, random_state=42),
                 learning_rate=0.04, val_size=.15, patience=5, n_jobs=1, random_state=42):
        self.n_iterations = n_iterations
        self.base_estimator = base_estimator
        self.learning_rate = learning_rate
        self.n_jobs = n_jobs
        self.val_size = val_size
        self.patience = patience
        self.random_state = random_state

    def _get_base_prediction(self, y):
        return 0

    def fit(self, X, y):
        np.random.seed(self.random_state)
        n_val = int(self.val_size * X.shape[0])
        idx = np.random.permutation(X.shape[0])
        val_idx = idx[:n_val]
        train_idx = idx[n_val:]

        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        self.base_prediction_ = self._get_base_prediction(y_train)
        residuals = y_train - self.base_prediction_
        prev_residuals_val = np.inf
        residuals_val = y_val - self.base_prediction_

        pairs = list(combinations(range(X.shape[1]), 2))
        self.estimators_ = {idx: [] for idx in list(range(X.shape[1])) + pairs}

        wait = 0
        for _ in range(self.n_iterations):
            for feature_idx in self.estimators_.keys():
                est = copy.deepcopy(self.base_estimator)
                est.fit(X_train[:, feature_idx].reshape(X_train.shape[0], -1), residuals)

                res_delta = self.learning_rate * est.predict(X_train[:, feature_idx].reshape(X_train.shape[0], -1))
                residuals -= res_delta
                res_delta_val = self.learning_rate * est.predict(X_val[:, feature_idx].reshape(X_val.shape[0], -1))
                residuals_val -= res_delta_val
                self.estimators_[feature_idx].append(est)

            if np.sum(np.abs(residuals_val)) <= prev_residuals_val:
                wait = 0
            else:
                wait += 1
                if wait >= self.patience:
                    for feature_idx in self.estimators_.keys():
                        self.estimators_[feature_idx] = self.estimators_[feature_idx][:-self.patience+1]
                    break
            prev_residuals_val = np.sum(np.abs(residuals_val))

    def predict(self, X: np.ndarray):
        prediction = np.ones((X.shape[0], ))*self.base_prediction_

        for feature_idx in self.estimators_.keys():
            for est in self.estimators_[feature_idx]:
                prediction += self.learning_rate * est.predict(X[:, feature_idx].reshape(X.shape[0], -1))

        return prediction
