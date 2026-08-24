import numpy as np

from RuleTree.stumps.interfaces.BaseMatrix import BaseMatrix
from RuleTree.stumps.regression import DecisionTreeStumpRegressor
from RuleTree.stumps.utils.RegressionImpurity import RegressionImpurity


class RegressorBaseMatrix(BaseMatrix, DecisionTreeStumpRegressor):
    def __init__(self, min_samples_leaf=1, random_state=42, criterion=None, batch_size=None, n_bins=None):
        super().__init__(min_samples_leaf, random_state, criterion, batch_size, n_bins)

        #if criterion == "entropy":
        self.impurity_fun = RegressionImpurity.mse # TODO: inserire altre metriche

    def _prepare_data(self, X, y, idx, context, sample_weight=None):
        data = super()._prepare_data(
            X, y, idx, context
        )

        X = data["X"]
        y = data["y"]
        n_samples = data["n_samples"]
        m = data["m"]
        batch_size = data["batch_size"]

        y = np.asarray(y, dtype=np.float32).ravel()

        return {
            "X": X,
            "y": y,
            "n_samples": n_samples,
            "m": m,
            "batch_size": batch_size
        }

    def _calculate_gain(self, sx_mask, data):
        return RegressionImpurity.calculate_gain(
            sx_mask,
            data["y"],
            self.impurity_fun
        )