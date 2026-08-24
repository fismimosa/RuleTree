import numpy as np

class RegressionImpurity:
    @staticmethod
    def mse(sum_y, sq_sum_y, n):
        n = np.maximum(n, 1)

        return (
                sq_sum_y / n -
                (sum_y / n) ** 2
        )

    @staticmethod
    def calculate_gain(sx_mask, y, impurity_fun, Idx=None):
        n_samples = y.shape[0]

        # statistiche nodo padre
        sum_total = y.sum()
        sq_sum_total = (y ** 2).sum()

        imp_parent = impurity_fun(
            sum_total,
            sq_sum_total,
            n_samples
        )

        # statistiche figlio sinistro
        sum_left = y @ sx_mask
        sq_sum_left = (y ** 2) @ sx_mask

        # statistiche figlio destro
        sum_right = sum_total - sum_left
        sq_sum_right = sq_sum_total - sq_sum_left

        # cardinalità figli
        n_left = sx_mask.sum(axis=0)
        n_right = n_samples - n_left

        imp_left = impurity_fun(
            sum_left,
            sq_sum_left,
            n_left
        )

        imp_right = impurity_fun(
            sum_right,
            sq_sum_right,
            n_right
        )

        weighted_children = (
                (n_left / n_samples) * imp_left +
                (n_right / n_samples) * imp_right
        )

        gain = imp_parent - weighted_children

        if Idx is not None:
            return gain[Idx], imp_parent, imp_left[Idx], imp_right[Idx]

        return gain, imp_parent, imp_left, imp_right