import copy
from concurrent.futures import ProcessPoolExecutor
from os import cpu_count

import numpy as np
from scipy.special import expit
from sklearn.base import ClassifierMixin

from RuleTree import RuleTreeRegressor
from RuleTree.base.RuleTreeBase import RuleTreeBase


class GBoostedTreeClassifier(RuleTreeBase, ClassifierMixin):
    def __init__(self, base_estimator=RuleTreeRegressor(max_depth=3), n_estimators=100, learning_rate=.1,
                 loss='squared_loss', n_iter_no_change=None, tol=1e-4, n_jobs=cpu_count()):
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        if loss != 'squared_loss' and type(loss) is str:
            raise ValueError('loss must be squared_loss or callable')
        self.loss = loss
        self.n_iter_no_change = n_iter_no_change
        self.tol = tol
        self.n_jobs = n_jobs

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.classes_ = np.unique(y).tolist()
        n_classes = len(self.classes_)

        self.base_log_odds = np.zeros(n_classes)
        self.base_prediction_ = np.zeros(n_classes)

        for i, classe in enumerate(self.classes_):
            self.base_log_odds[i] = np.log((y == classe).sum() / (y != classe).sum())
            self.base_prediction_[i] = expit(self.base_log_odds[i])

        self.estimators_ = [[] for _ in self.classes_]

        args = [
            (
                i,
                classe,
                X,
                y,
                self.base_estimator,
                self.base_prediction_[i],
                self.base_log_odds[i],
                self.learning_rate,
                self.n_estimators,
                self.n_iter_no_change,
                self.tol,
            )
            for i, classe in enumerate(self.classes_)
        ]

        with ProcessPoolExecutor(max_workers=min(self.n_jobs, len(self.classes_))) as ex:
            for i, estimators in ex.map(_train_one_class, args):
                self.estimators_[i] = estimators

        return self


    def predict(self, X: np.ndarray):
        prediction = np.ones((X.shape[0], len(self.classes_)))*self.base_log_odds

        for i, (classe, estimators_el) in enumerate(zip(self.classes_, self.estimators_)):
            for est, gamma_map in estimators_el:
                leafs = est.apply(X)
                leafs = np.vectorize(gamma_map.get)(leafs)
                prediction[:, i] += self.learning_rate * leafs

        prediction = np.argmax(expit(prediction), axis=1)
        prediction = np.vectorize(self.classes_.index)(prediction)

        return prediction

def _train_one_class(args):
    i, classe, X, y, base_estimator, base_pred, base_log_odds, lr, n_estimators, n_iter_no_change, tol = args

    n_samples = len(y)

    residuals = (y == classe).astype(float) - base_pred
    prediction = np.ones(n_samples) * base_pred
    log_odds_prediction = np.ones(n_samples) * base_log_odds

    estimators = []

    patience = n_iter_no_change
    for _ in range(n_estimators):
        est = copy.deepcopy(base_estimator)
        est.fit(X, residuals)

        leafs = est.apply(X)
        gamma_map = {}
        denom = np.sum(prediction * (1 - prediction))
        for leaf_id in np.unique(leafs):
            gamma_map[leaf_id] = residuals[leafs == leaf_id].sum() / denom

        gamma = np.vectorize(gamma_map.get)(leafs)
        res_delta = lr * gamma

        log_odds_prediction += res_delta
        new_prediction = expit(log_odds_prediction)
        if patience is not None and np.mean(prediction - new_prediction) < tol:
            if patience == 0:
                estimators = estimators[:-n_iter_no_change]
                break
            patience -= 1
        prediction = new_prediction
        residuals = (y == classe).astype(float) - prediction

        estimators.append((est, gamma_map))

    return i, estimators