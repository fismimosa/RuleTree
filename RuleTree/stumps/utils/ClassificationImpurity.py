import numpy as np

class ClassificationImpurity():

    @staticmethod
    def calculate_gain(sx_mask, y_onehot, impurity_fun, Idx=None):
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

        n_samples = y_onehot.shape[0]

        n_left = counts_weighted_left.sum(axis=0)
        n_right = n_samples - n_left

        imp_parent = impurity_fun(counts_weighted_total)
        imp_left = impurity_fun(counts_weighted_left)
        imp_right = impurity_fun(counts_weighted_right)

        imp_weighted_children = (
                (n_left / n_samples) * imp_left +
                (n_right / n_samples) * imp_right
        )
        info_gain = imp_parent - imp_weighted_children

        if Idx is not None:
            return info_gain[Idx], imp_parent, imp_left[Idx], imp_right[Idx]

        return info_gain, imp_parent, imp_left, imp_right

    @staticmethod
    def gini(counts):
        """
        counts: matrice (n_classes x k_splits)
        return: array (k_splits)
        """
        totals = counts.sum(axis=0)
        totals = np.where(totals == 0, 1, totals)

        probs = counts / totals
        return 1.0 - np.sum(probs ** 2, axis=0)

    @staticmethod
    def entropy(counts):
        totals = counts.sum(axis=0)
        probs = counts / totals

        n_splits = probs.shape[1]
        n_classes = probs.shape[0]

        result = np.zeros(n_splits)

        for j in range(n_splits):
            ent = 0.0
            for i in range(n_classes):
                p = probs[i, j]
                if p > 0.0:
                    ent -= p * np.log2(p)
            result[j] = ent
        return result
