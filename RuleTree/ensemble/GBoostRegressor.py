"""GBoostRegressor: gradient boosting of RuleTrees for regression."""

import copy

import numpy as np
from sklearn.base import RegressorMixin

from RuleTree import RuleTreeRegressor
from RuleTree.tree.RuleTreeRegressor import check_target
from RuleTree.base.RuleTreeBase import RuleTreeBase


class GBoostRegressor(RegressorMixin, RuleTreeBase):
    """
    Gradient boosting for regression with RuleTrees as base estimators (squared loss).

    Each estimator is fitted on the residuals of the previous ones; the boosting stops early when
    the error on a validation set does not improve for ``patience`` iterations.

    Parameters
    ----------
    base_estimator : RuleTreeRegressor, default=RuleTreeRegressor(max_depth=3)
        Estimator fitted at each iteration.
    n_estimators : int, default=100
        Maximum number of estimators.
    learning_rate : float, default=0.1
        Shrinkage of the contribution of each estimator.
    loss : 'squared_loss' or callable, default='squared_loss'
        Checked but not used: the loss is always the squared loss.
    patience : int, default=5
        Number of iterations without improvement before stopping.
    val_size : float, default=0.15
        Fraction of the training set used for validation.
    random_state : int, default=42
        Seed of the split between training and validation.

    Attributes
    ----------
    base_prediction_ : float
        Initial prediction (mean of the target).
    estimators_ : list of RuleTreeRegressor
        Fitted estimators.

    See Also
    --------
    XGBoostRegressor : the boosting of the XGBoost trees.
    EBMRegressor : a boosting of trees on single features and pairs of features.

    Examples
    --------
    >>> from sklearn.datasets import load_diabetes
    >>> from RuleTree import RuleTreeRegressor
    >>> from RuleTree.ensemble.GBoostRegressor import GBoostRegressor
    >>> X, y = load_diabetes(return_X_y=True)
    >>> model = GBoostRegressor(base_estimator=RuleTreeRegressor(max_depth=2, random_state=0),
    ...                         n_estimators=20, random_state=0).fit(X, y)
    >>> len(model.estimators_), round(model.score(X, y), 3)
    (20, 0.531)
    """
    def __init__(self, base_estimator=RuleTreeRegressor(max_depth=3),
                 n_estimators=100, learning_rate=.1, loss='squared_loss', patience=5, val_size=.15,
                 random_state=42):
        """Initialize the GBoostRegressor; see the class documentation for the parameters."""
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
        """
        Initial prediction of the boosting.

        Parameters
        ----------
        y : numpy.ndarray
            Target of the training set.

        Returns
        -------
        base_prediction : float or numpy.ndarray of shape (n_targets,)
            Mean of the target (of each target).
        """
        return np.mean(y, axis=0)

    def _get_base_estimator(self):
        """
        Estimator copied at each iteration.

        Returns
        -------
        base_estimator : RuleTreeRegressor
        """
        return self.base_estimator

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Fit the boosting.

        Parameters
        ----------
        X : numpy.ndarray of shape (n_samples, n_features)
            Tabular data.
        y : numpy.ndarray of shape (n_samples,) or (n_samples, n_targets)
            Target.

        Returns
        -------
        self : GBoostRegressor
        """
        y = check_target(y)
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

        base_estimator = self._get_base_estimator()
        wait = 0
        for _ in range(self.n_estimators):
            est = copy.deepcopy(base_estimator)
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
                    # the last patience estimators did not improve the validation error
                    self.estimators_ = self.estimators_[:len(self.estimators_) - self.patience]
                    break
            prev_residuals_val = np.sum(np.abs(residuals_val))

        return self


    def predict(self, X: np.ndarray):
        """
        Predict the target: the initial prediction plus the contributions of all the estimators.

        Parameters
        ----------
        X : numpy.ndarray of shape (n_samples, n_features)
            Tabular data.

        Returns
        -------
        y_pred : numpy.ndarray of shape (n_samples,) or (n_samples, n_targets)
        """
        prediction = np.zeros((X.shape[0],) + np.shape(self.base_prediction_)) + self.base_prediction_

        for est in self.estimators_:
            prediction += self.learning_rate * est.predict(X)

        return prediction