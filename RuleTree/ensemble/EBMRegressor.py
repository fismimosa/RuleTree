"""EBMRegressor: explainable boosting machine for regression."""

from itertools import combinations

import numpy as np
from sklearn.base import RegressorMixin

from RuleTree import RuleTreeRegressor
from RuleTree.tree.RuleTreeRegressor import check_target
from RuleTree.base.RuleTreeBase import RuleTreeBase
from RuleTree.ensemble._boosting import EarlyStopping, seeded_copy, validation_split


class EBMRegressor(RegressorMixin, RuleTreeBase):
    """
    Explainable boosting machine for regression: a cyclic gradient boosting where each base
    estimator is fitted on a single feature or on a pair of features, so that the prediction is a
    sum of functions of one or two features.

    At each iteration, a copy of the base estimator is fitted on the residuals for each feature and
    each pair of features. The boosting stops early when the error on a validation set does not
    improve for ``patience`` iterations.

    Parameters
    ----------
    n_iterations : int, default=100
        Maximum number of iterations.
    base_estimator : RuleTreeRegressor, default=RuleTreeRegressor(max_depth=6, random_state=42)
        Estimator fitted on each feature and pair of features.
    learning_rate : float, default=0.04
        Shrinkage of the contribution of each estimator.
    val_size : float, default=0.15
        Fraction of the training set used for validation.
    patience : int, default=5
        Number of iterations without improvement before stopping.
    n_jobs : int, default=1
        Not used.
    random_state : int, default=42
        Seed of the split between training and validation, and of the trees whose base estimator has no
        random_state (each tree gets a different seed drawn from it).

    Attributes
    ----------
    base_prediction_ : float
        Initial prediction (0).
    estimators_ : dict
        Feature or pair of features -> list of fitted estimators.

    See Also
    --------
    GBoostRegressor : a boosting of trees on all the features.

    Examples
    --------
    >>> from sklearn.datasets import load_diabetes
    >>> from RuleTree import RuleTreeRegressor
    >>> from RuleTree.ensemble.EBMRegressor import EBMRegressor
    >>> X, y = load_diabetes(return_X_y=True)
    >>> model = EBMRegressor(base_estimator=RuleTreeRegressor(max_depth=1, random_state=0),
    ...                      n_iterations=5, learning_rate=0.1, random_state=0).fit(X, y)
    >>> len(model.estimators_)  # 10 features and 45 pairs of features
    55
    >>> len(model.estimators_[(0, 1)])  # trees on the pair of features 0 and 1
    5
    """
    def __init__(self, n_iterations=100, base_estimator=RuleTreeRegressor(max_depth=6, random_state=42),
                 learning_rate=0.04, val_size=.15, patience=5, n_jobs=1, random_state=42):
        """Initialize the EBMRegressor; see the class documentation for the parameters."""
        self.n_iterations = n_iterations
        self.base_estimator = base_estimator
        self.learning_rate = learning_rate
        self.n_jobs = n_jobs
        self.val_size = val_size
        self.patience = patience
        self.random_state = random_state

    def _get_base_prediction(self, y):
        """
        Initial prediction of the boosting.

        Parameters
        ----------
        y : numpy.ndarray
            Target of the training set.

        Returns
        -------
        base_prediction : int or numpy.ndarray of shape (n_targets,)
            0 (for each target).
        """
        return 0 if np.ndim(y) == 1 else np.zeros(np.shape(y)[1])

    def fit(self, X, y):
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
        self : EBMRegressor
        """
        y = check_target(y)
        X_train, X_val, y_train, y_val = validation_split(X, y, self.val_size, self.random_state)

        self.base_prediction_ = self._get_base_prediction(y_train)
        residuals = y_train - self.base_prediction_
        residuals_val = y_val - self.base_prediction_

        pairs = list(combinations(range(X.shape[1]), 2))
        self.estimators_ = {idx: [] for idx in list(range(X.shape[1])) + pairs}

        rng = np.random.RandomState(self.random_state)
        stopping = EarlyStopping(self.patience)
        for _ in range(self.n_iterations):
            for feature_idx in self.estimators_.keys():
                est = seeded_copy(self.base_estimator, rng)
                est.fit(X_train[:, feature_idx].reshape(X_train.shape[0], -1), residuals)

                res_delta = self.learning_rate * est.predict(X_train[:, feature_idx].reshape(X_train.shape[0], -1))
                residuals -= res_delta
                res_delta_val = self.learning_rate * est.predict(X_val[:, feature_idx].reshape(X_val.shape[0], -1))
                residuals_val -= res_delta_val
                self.estimators_[feature_idx].append(est)

            if stopping.stop(np.sum(np.abs(residuals_val))):
                for feature_idx, estimators in self.estimators_.items():
                    # the last patience estimators did not improve the validation error
                    self.estimators_[feature_idx] = estimators[:len(estimators) - self.patience]
                break

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

        for feature_idx in self.estimators_.keys():
            for est in self.estimators_[feature_idx]:
                prediction += self.learning_rate * est.predict(X[:, feature_idx].reshape(X.shape[0], -1))

        return prediction
