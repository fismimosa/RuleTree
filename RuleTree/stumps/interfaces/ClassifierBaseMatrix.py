import numpy as np

from RuleTree.stumps.classification import DecisionTreeStumpClassifier
from RuleTree.stumps.interfaces.BaseMatrix import BaseMatrix
from RuleTree.stumps.utils.ClassificationImpurity import ClassificationImpurity


class ClassifierBaseMatrix(BaseMatrix, DecisionTreeStumpClassifier):
    def __init__(self, min_samples_leaf=1, class_weight=None, random_state=42, criterion=None, batch_size=None,
                 n_bins=None):
        super().__init__(min_samples_leaf, random_state, criterion, batch_size, n_bins)
        self.kwargs['class_weight'] = class_weight
        self.class_weight = class_weight

        self.impurity_fun = ClassificationImpurity.gini  # default gini

        if criterion == "entropy":
            self.impurity_fun = ClassificationImpurity.entropy

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

    def _prepare_data(self, X, y, idx, context, sample_weight=None):
        data = super()._prepare_data(X, y, idx, context)

        X = data["X"]
        y = data["y"]
        n_samples = data["n_samples"]
        m = data["m"]
        batch_size = data["batch_size"]

        y = np.asarray(y).ravel()
        class_weight = None
        if self.class_weight == "balanced":
            class_weight = {}
            for class_label in np.unique(y):
                class_weight[class_label] = n_samples / (len(self.classes_) * len(y[y == class_label]))

        y_onehot = self._build_onehot(y, n_samples, sample_weight, class_weight)

        return {
            "X": X,
            "y": y,
            "y_onehot": y_onehot,
            "n_samples": n_samples,
            "m": m,
            "batch_size": batch_size
        }

    def _calculate_gain(self, sx_mask, data):
        return ClassificationImpurity.calculate_gain(
            sx_mask,
            data["y_onehot"],
            self.impurity_fun
        )