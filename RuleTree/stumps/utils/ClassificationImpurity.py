import numpy as np

class ClassificationImpurity():

    @staticmethod
    def calculate_gain(sx_mask, y_onehot, impurity_fun, Idx=None, protected_attribute=None, ideal=None, fairness_weight=1.0):
        """
        Args:
            sx_mask: maschera samples che vanno a sx
            y_onehot: rappresentazione binaria dei samples che appartengono alle classi
            impurity_fun: funzione di impurità scelta
            Idx (opzionale): voglio solo le info relative a uno split particolare
            protected_attribute (opzionale): attributo fair
            fairness_weight (opzionale): fairness weight

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
        
        original_imp_left = imp_left.copy()
        original_imp_right = imp_right.copy()
        mfc_left = np.zeros(imp_left.shape)
        mfc_right = np.zeros(imp_left.shape)

        # Fairness
        if protected_attribute is not None and ideal is not None:
            protected_onehot = np.eye(len(ideal))[protected_attribute]

            # Conteggi del protected attribute
            protected_total = protected_onehot.sum(axis=0)
            protected_left = protected_onehot.T @ sx_mask
            protected_right = protected_total[:, None] - protected_left

            # Evita divisioni per zero
            n_left_safe = np.where(n_left == 0, 1, n_left)
            n_right_safe = np.where(n_right == 0, 1, n_right)

            protected_prob_left = protected_left / n_left_safe
            protected_prob_right = protected_right / n_right_safe

            # MFC (non lo faccio sul padre perche ideal=distribuzione del dataset quindi mfc_parent=0)
            mfc_left = np.mean(
                np.abs(protected_prob_left - ideal[:, None]),
                axis=0
            )

            mfc_right = np.mean(
                np.abs(protected_prob_right - ideal[:, None]),
                axis=0
            )

            # Gini + penalty
            imp_left += fairness_weight * mfc_left * imp_left
            imp_right += fairness_weight * mfc_right * imp_right


        imp_weighted_children = (
                (n_left / n_samples) * imp_left +
                (n_right / n_samples) * imp_right
        )
        info_gain = imp_parent - imp_weighted_children

        if False:
            # ============================================================

            # SALVATAGGIO CSV

            # ============================================================

            import pandas as pd

            import os

            csv_path = "fairness_splits.csv"

            rows = []

            for i in range(len(info_gain)):
                rows.append({
                    "split": i,
                    "n_left": n_left[i],
                    "n_right": n_right[i],
                    "fairness_weight": fairness_weight,
                    "original_imp_left": original_imp_left[i],
                    "original_imp_right": original_imp_right[i],
                    "mfc_left": mfc_left[i],
                    "mfc_right": mfc_right[i],
                    "penalty_left": fairness_weight * mfc_left[i],
                    "penalty_right": fairness_weight * mfc_right[i],
                    "fair_impurity_left": imp_left[i],
                    "fair_impurity_right": imp_right[i],
                    "information_gain": info_gain[i]
                })

            df_splits = pd.DataFrame(rows)

            # Se il file esiste aggiunge le righe,

            # altrimenti crea il file con header

            df_splits.to_csv(
                csv_path,
                mode="a",
                header=not os.path.exists(csv_path),
                index=False
            )

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
