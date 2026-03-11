import numpy as np

from RuleTree.stumps.classification.lorenzo.interfaces.IImpurity import IImpurity


class Impurity(IImpurity):

    @staticmethod
    def calculate_gain(sx_mask, y_onehot, impurity_fun, Idx=None) -> tuple:
        """
        Args:
            sx_mask: maschera samples che vanno a sx
            y_onehot: rappresentazione binaria dei samples che appartengono alle classi
            impurity_fun: funzione di impurità scelta
            Idx (opzionale): voglio solo le info relative a uno split particolare

        Returns: una quadrupla formata da
            - Array di tutti gli info_gain per ciascuno split
            - Impurità del nodo padre
            - Array di tutte le impurità del figlio sx per ciascuno split
            - Array di tutte le impurità del figlio dx per ciascuno split
        """
        counts_weighted_total = y_onehot.sum(axis=0)  # somma per ogni classe pesata
        counts_weighted_left = y_onehot.T @ sx_mask  # n_classes x k
        counts_weighted_right = counts_weighted_total[:, None] - counts_weighted_left

        n_left = sx_mask.sum(axis=0)
        n_right = (~sx_mask).sum(axis=0)

        imp_parent = impurity_fun(counts_weighted_total)
        imp_left = impurity_fun(counts_weighted_left)
        imp_right = impurity_fun(counts_weighted_right)

        imp_weighted_children = (
                (n_left / y_onehot.shape[0]) * imp_left +
                (n_right / y_onehot.shape[0]) * imp_right
        )
        info_gain = imp_parent - imp_weighted_children

        if Idx is not None:
            return info_gain[Idx], imp_parent, imp_left[Idx], imp_right[Idx]

        return info_gain, imp_parent, imp_left, imp_right

    @staticmethod
    def gini(counts: np.ndarray) -> np.ndarray:
        """
        counts: matrice (n_classes x k_splits)
        return: array (k_splits)
        """
        totals = counts.sum(axis=0)
        totals = np.where(totals == 0, 1, totals)

        probs = counts / totals
        return 1.0 - np.sum(probs ** 2, axis=0)

    @staticmethod
    def entropy(counts: np.ndarray) -> np.ndarray:
        """
        counts: matrice (n_classes x k_splits)
        return: array (k_splits)
        """
        totals = counts.sum(axis=0)
        totals = np.where(totals == 0, 1, totals)

        probs = counts / totals
        return -np.sum(np.where(probs > 0, probs * np.log2(probs), 0.0), axis=0)
