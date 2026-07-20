import numpy as np

from RuleTree.stumps.classification import DecisionTreeStumpClassifier
from RuleTree.stumps.interfaces.BaseMatrix import BaseMatrix
from RuleTree.stumps.utils.Impurity import Impurity


class ClassifierBaseMatrix(BaseMatrix, DecisionTreeStumpClassifier):
    def __init__(self, min_samples_leaf=1, class_weight=None, random_state=42, criterion=None, batch_size=None,
                 n_bins=None):
        """
        Initializes the DecisionTreeStumpClassifier.

        This constructor sets up the decision tree stump with default values and
        interprets the provided keyword arguments for configuration.

        Args:
                - min_samples_leaf (int): Minimum samples required to be at a leaf node
                - class_weight (dict or 'balanced'): Weights associated with classes
                - random_state (int): Seed for the random number generator
                - criterion (str): Function to measure the quality of a split ('gini' or 'entropy')
        """
        super().__init__(min_samples_leaf, random_state, criterion, batch_size, n_bins)
        self.kwargs['class_weight'] = class_weight
        self.class_weight = class_weight

        self.impurity_fun = Impurity.gini  # default gini

        if criterion == "entropy":
            self.impurity_fun = Impurity.entropy

    def _build_onehot(self, y, n_samples, sample_weight=None, class_weight=None):
        classes = np.unique(y)  # Array ordinato di classi univoche
        n_classes = len(classes)

        # Per ogni elemento di y, trovo la posizione della sua classe all'interno di classes
        y_idx = np.searchsorted(classes, y)

        # One-hot encoding di y
        # Per ogni sample dico a quale classe appartiene
        y_onehot = np.zeros((n_samples, n_classes), dtype=float)  # Matrice n_samples x n_classes di zeri

        if sample_weight is None:
            y_onehot[np.arange(n_samples), y_idx] = 1.0  # Se non c'è peso, assegno peso 1
        else:
            y_onehot[np.arange(n_samples), y_idx] = sample_weight  # Altrimenti assegno peso sample_weight_i,

        # Peso le classi
        if class_weight is not None:
            class_weight_vec = np.array([class_weight[c] for c in classes])  # Trasformo dizionario in vettore
            y_onehot *= class_weight_vec

        return y_onehot

    def _prepare_data(self, X, y, idx, context, sample_weight):
        X, y, n_samples, m, batch_size = super()._prepare_data(X, y, idx, context)

        y = np.asarray(y).ravel()
        class_weight = None
        if self.class_weight == "balanced":
            class_weight = {}
            for class_label in np.unique(y):
                class_weight[class_label] = n_samples / (len(self.classes_) * len(y[y == class_label]))

        y_onehot = self._build_onehot(y, n_samples, sample_weight, class_weight)

        return X, y, y_onehot, n_samples, m, batch_size

    def fit(self, X, y, idx=None, context=None, sample_weight=None):
        X, y, y_onehot, n_samples, m, batch_size = self._prepare_data(
            X=X,
            y=y,
            idx=idx,
            context=context,
            sample_weight=sample_weight
        )

        best_gain = -np.inf  # Miglior gain in assoluto
        best_feature = None  # Miglior feature in assoluto associata al best gain
        best_threshold = None  # Miglior threshold in assoluto associata al best gain

        for start in range(0, m, batch_size):  # Incremento di batch_size
            end = min(start + batch_size, m)
            features_batch = list(range(start, end))
            active_features, active_thresholds = self._generate_splits(X, features_batch, m)
            k = len(active_features)  # Numero di split

            sx_mask = self._build_sxmask(
                X,
                active_features,
                active_thresholds,
                k,
                n_samples
            )

            info_gain, imp_parent, imp_left, imp_right = Impurity.calculate_gain(sx_mask, y_onehot, self.impurity_fun)

            local_best = np.argmax(info_gain)  # Indice del best gain
            if info_gain[local_best] > best_gain:
                best_gain = info_gain[local_best]
                best_feature = active_features[local_best]
                best_threshold = active_thresholds[local_best]

                self.impurity[0] = imp_parent
                self.impurity[1] = imp_left[local_best]
                self.impurity[2] = imp_right[local_best]

        self._finalize_fit(X, y, best_feature, best_threshold)

        return self

    def _generate_splits(self, X, features_batch, m=None):
        raise NotImplementedError

    def _build_sxmask(self, X, active_features, active_thresholds, k, n_samples):
        raise NotImplementedError