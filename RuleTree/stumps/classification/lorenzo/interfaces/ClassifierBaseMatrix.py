import numpy as np

from RuleTree.exceptions import NoSplitFoundWarning
from RuleTree.stumps.classification import DecisionTreeStumpClassifier
from RuleTree.stumps.classification.lorenzo.utils.Impurity import Impurity


class ClassifierBaseMatrix(DecisionTreeStumpClassifier):
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

        self.is_categorical = False
        self.feature = None
        self.threshold = None
        self.categorical = None
        self.numerical = None
        self.bin_edges = {}

        self.batch_size = batch_size
        self.n_bins = n_bins
        self.class_weight = class_weight
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state
        self.kwargs = {
            'class_weight': class_weight,
            'random_state': random_state,
            'criterion': criterion,
            'min_samples_leaf': min_samples_leaf,
            'batch_size': batch_size,
            'n_bins': n_bins
        }

        self.impurity = [0.0, 0.0, 0.0]  # padre, sx, dx
        self.impurity_fun = Impurity.gini  # default gini

        if criterion == "entropy":
            self.impurity_fun = Impurity.entropy

    def _apply_binning(self, X):
        X_binned = X.astype(np.int32, copy=True)

        for f in self.numerical:
            col = X[:, f]
            edges = np.quantile(col, np.linspace(0, 1, self.n_bins + 1))
            self.bin_edges[f] = edges
            X_binned[:, f] = np.digitize(col, edges[1:-1])

        return X_binned

    def _select_dataset_subset(self, X, y, idx):
        if idx is None:  # Prendo la porzione di features e colonna target che mi interessa
            idx = slice(None)

        X = X[idx]
        y = y[idx]
        return X, y

    ## Preparazione dei dati
    def _prepare_data(self, X, y, idx, context, sample_weight):
        X, y = self._select_dataset_subset(X, y, idx)

        n_samples = len(X)

        class_weight = None
        if self.class_weight == "balanced":
            class_weight = {}
            for class_label in np.unique(y):
                class_weight[class_label] = n_samples / (len(self.classes_) * len(y[y == class_label]))

        if hasattr(context, 'categorical'):
            self.categorical = context.categorical
            self.numerical = context.numerical
        else:
            self.feature_analysis(X, y)
            context.categorical = self.categorical
            context.numerical = self.numerical

        m = X.shape[1]  # Numero di features
        batch_size = self.batch_size or m  # Dimensione del batch oppure m stesso se non specificato

        ### COSTRUZIONE ONE HOT ###
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
        ### FINE COSTRUZIONE ONEHOT ###

        if self.n_bins is not None:
            X = self._apply_binning(X)

        return X, y, y_onehot, n_samples, m, batch_size

    ## Fitting base
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

    def _finalize_fit(self, X, y, best_feature, best_threshold):
        self.feature = best_feature
        self.threshold = best_threshold
        self.is_categorical = self.feature in self.categorical

        # no split
        if self.feature is None:
            raise NoSplitFoundWarning(f"No split found for X {X.shape} and y {np.unique(y)}")

    def apply(self, X):
        """
        Restituisce un array di:
                    0 se il record "i" va a sx
                    1 se il record "i" va a dx
        """

        dim_X = X.shape[0]  # Numero dei record

        if self.feature is None:
            return np.zeros(dim_X, dtype=np.int_)  # tutti a sinistra

        if self.n_bins is not None:
            X = X.copy()
            for f in self.numerical:
                edges = self.bin_edges[f]
                X[:, f] = np.digitize(X[:, f], edges[1:-1])

        y_pred = np.ones(dim_X)  # tutti a destra
        X_feature = X[:, self.feature]  # prendo i valori della feature migliore trovata

        if not self.is_categorical:
            y_pred[X_feature <= self.threshold] = 0  # a sinistra
        else:
            y_pred[X_feature == self.threshold] = 0  # a sinistra

        return y_pred + 1
