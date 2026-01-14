import copy
from concurrent.futures import ProcessPoolExecutor
from os import cpu_count

import numpy as np
from scipy.special import expit
from sklearn.base import ClassifierMixin

from sklearn.model_selection import train_test_split

from RuleTree import RuleTreeRegressor
from RuleTree.base.RuleTreeBase import RuleTreeBase


class GBoostClassifier(RuleTreeBase, ClassifierMixin):
    def __init__(self, base_estimator=RuleTreeRegressor(max_depth=4), n_estimators=100, learning_rate=.1,
                 loss='squared_loss', patience=5, val_size=.15, n_jobs=cpu_count(), random_state=42):
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        if loss != 'squared_loss' and type(loss) is str:
            raise ValueError('loss must be squared_loss or callable')
        self.loss = loss
        self.patience = patience
        self.n_jobs = n_jobs
        self.val_size = val_size
        self.random_state = random_state

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
                self.val_size,
                self.patience,
                self.random_state
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
    i, classe, X, y, base_estimator, base_pred, base_log_odds, lr, n_estimators, val_size, patience, random_state = args

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=val_size,
                                                      random_state=random_state, stratify=y)

    n_samples = len(y_train)
    residuals = (y_train == classe).astype(float) - base_pred
    residuals_val = (y_val == classe).astype(float) - base_pred
    y_val_bool = (y_val == classe).astype(float)
    prediction = np.ones((n_samples, )) * base_pred
    prediction_val = np.ones((X_val.shape[0], )) * base_pred
    log_odds_prediction = np.ones((n_samples, )) * base_log_odds
    log_odds_prediction_val = np.ones((X_val.shape[0], )) * base_log_odds

    estimators = []

    wait = 0
    for _ in range(n_estimators):
        est = copy.deepcopy(base_estimator)
        est.fit(X_train, residuals)

        leafs = est.apply(X_train)
        leafs_val = est.apply(X_val)
        gamma_map = {}
        denom = np.sum(prediction * (1 - prediction))
        for leaf_id in np.unique(leafs):
            gamma_map[leaf_id] = residuals[leafs == leaf_id].sum() / denom

        gamma = np.vectorize(gamma_map.get)(leafs)
        gamma_val = np.vectorize(gamma_map.get)(leafs_val)
        res_delta = lr * gamma
        res_delta_val = lr * gamma_val

        log_odds_prediction += res_delta
        log_odds_prediction_val += res_delta_val
        new_prediction = expit(log_odds_prediction)
        new_prediction_val = expit(log_odds_prediction_val)
        if np.mean(np.abs(y_val_bool - new_prediction_val)) < np.mean(np.abs(y_val_bool - prediction_val)):
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                estimators = estimators[:-patience+1]
                break
        prediction = new_prediction
        prediction_val = new_prediction_val
        residuals = (y_train == classe).astype(float) - prediction

        estimators.append((est, gamma_map))

    return i, estimators