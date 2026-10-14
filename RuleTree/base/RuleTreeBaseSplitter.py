"""
File: RuleTreeBaseSplitter.py

Base class of the splitters: the component of a stump that finds the best split of a feature
matrix (the matrix produced by the stump's transformer, see RuleTreeBaseTransformer).
"""
from abc import ABC, abstractmethod

import numpy as np
from sklearn.base import BaseEstimator


class RuleTreeBaseSplitter(BaseEstimator, ABC):
    """
    Finds the best split of a feature matrix: a threshold on a numerical feature (``x <= threshold``
    goes left) or a value of a categorical feature (``x == value`` goes left).

    Attributes set by fit:
        feature_ (int): Column of the matrix used by the split.
        threshold_: Threshold (numerical split) or value (categorical split).
        is_categorical_ (bool): Whether the split is categorical.
        gain_ (float): Gain of the split, used to compare stumps (stump_selection='best').
        impurity_ (list | None): Impurity of the node and of the two children, if computed.
    """

    # MODEL_TYPE_CLF or MODEL_TYPE_REG: the kind of target the splitter works with
    task = None

    # fitted attributes saved by get_state
    _state = ("feature_", "threshold_", "is_categorical_", "gain_", "impurity_")

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray, numerical: list, categorical: list, sample_weight=None,
            check_input=True):
        """
        Find the best split of X.

        Args:
            X (np.ndarray): Feature matrix of the instances in the node.
            y (np.ndarray): Target of the instances in the node.
            numerical (list): Indices of the numerical columns of X.
            categorical (list): Indices of the categorical columns of X.
            sample_weight (np.ndarray, optional): Sample weights of the instances in the node.
            check_input (bool): Whether to check the input data.

        Returns:
            RuleTreeBaseSplitter: The fitted splitter.

        Raises:
            NoSplitFoundWarning: If no split is found.
        """

    def apply(self, X: np.ndarray) -> np.ndarray:
        """
        Apply the split to a feature matrix.

        Returns:
            np.ndarray: For each row, 1 if it goes to the left child, 2 otherwise.
        """
        X_feature = X[:, self.feature_]
        if self.is_categorical_:
            return np.where(X_feature == self.threshold_, 1, 2)

        return np.where(X_feature <= self.threshold_, 1, 2)

    def update_statistics(self, X: np.ndarray, y: np.ndarray, sample_weight=None):
        """Recompute the statistics of the split (e.g. impurity) on new data, without changing the split."""

    def get_state(self) -> dict:
        """Fitted state, as a JSON-friendly dictionary (see set_state)."""
        return {name: getattr(self, name, None) for name in self._state}

    def set_state(self, state: dict):
        """Restore the fitted state saved by get_state."""
        for name in self._state:
            setattr(self, name, state.get(name))

        return self
