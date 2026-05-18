import numpy as np
from line_profiler import profile

from RuleTree.exceptions import NoSplitFoundWarning
from RuleTree.stumps.classification import DecisionTreeStumpClassifier
from RuleTree.stumps.classification.lorenzo.utils.Impurity import Impurity


class DecisionTreeStumpClassifierLorenzoMatrixv2(DecisionTreeStumpClassifier):
    """
    A decision tree stump classifier that extends DecisionTreeStumpClassifier.
    Version: Matrix v2

    A decision tree stump is a decision tree with a maximum depth of 1 (a single split), making
    it a simple interpretable model. This implementation supports both numerical and categorical features,
    provides methods for rule extraction, and can be used as a building block in more complex ensembles.

    The class handles both numerical splits (using ≤ comparisons) and categorical splits (using = comparisons),
    and automatically selects the feature and split that maximizes information gain.

    Attributes:
        is_categorical (bool): Whether the selected split is categorical.
        threshold (str): Split threshold values.
        feature (array): Feature indices used for splits.
        impurity_fun (function): Function used to calculate impurity (gini, entropy, etc.).
    """

    def __init__(self, min_samples_leaf=1, class_weight=None, random_state=42, criterion=None, batch_size=None):
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

        self.batch_size = batch_size
        self.class_weight = class_weight
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state
        self.kwargs = {
            'class_weight': class_weight,
            'random_state': random_state,
            'criterion': criterion,
            'min_samples_leaf': min_samples_leaf,
            'batch_size': batch_size
        }

        self.impurity = [0.0, 0.0, 0.0]  # padre, sx, dx
        self.impurity_fun = Impurity.gini  # default gini

        # Matrici
        self.A = None
        self.B = None

        if criterion == "entropy":
            self.impurity_fun = Impurity.entropy

    # @profile
    def fit(self, X, y, idx=None, context=None, sample_weight=None):
        """
        Fits the decision tree stump to the provided data.

        This method finds the optimal single-feature split that maximizes information gain
        for both numerical and categorical features.

        Args:
            X (array-like): Feature matrix of shape (n_samples, n_features).
            y (array-like): Target vector of shape (n_samples,).
            idx (slice, optional): Indices for slicing the data. If None, all samples are used.
            context (object, optional): Additional context for fitting (not used directly).
            sample_weight (array-like, optional): Sample weights of shape (n_samples,).
                                                    If None, samples are equally weighted.

        Returns:
            DecisionTreeStumpClassifierLorenzo: The fitted classifier (self).
        """

        if idx is None:  # Prendo la porzione di features e colonna target che mi interessa
            idx = slice(None)

        X = X[idx].astype(np.float32, copy=False)  # Cambiamento float -> float32
        y = y[idx]

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

        best_gain = -np.inf  # Miglior gain in assoluto
        best_feature = None  # Miglior feature in assoluto associata al best gain
        best_threshold = None  # Miglior threshold in assoluto associata al best gain

        ### INIZIO COSTRUZIONE ONEHOT ###
        classes = np.unique(y)  # Array ordinato di classi univoche
        n_classes = len(classes)

        # Per ogni elemento di y, trovo la posizione della sua classe all'interno di classes
        y_idx = np.searchsorted(classes, y)

        # One-hot encoding di y
        # Per ogni sample dico a quale classe appartiene
        y_onehot = np.zeros((n_samples, n_classes),
                            dtype=np.float32)  # Matrice n_samples x n_classes di zeri, cambiamento float -> float32

        if sample_weight is None:
            y_onehot[np.arange(n_samples), y_idx] = 1.0  # Se non c'è peso, assegno peso 1
        else:
            y_onehot[np.arange(n_samples), y_idx] = sample_weight  # Altrimenti assegno peso sample_weight_i, TODO

        # Peso le classi
        if class_weight is not None:
            class_weight_vec = np.array([class_weight[c] for c in classes])  # Trasformo dizionario in vettore
            y_onehot *= class_weight_vec
        ### FINE COSTRUZIONE ONEHOT ###

        categorical_set = set(self.categorical)

        for start in range(0, m, batch_size):
            end = min(start + batch_size, m)
            features_batch = list(range(start, end))
            active_features, active_thresholds = self.build_splits(X, features_batch)
            k = len(active_features)

            # Maschere per split numerici e categorici
            sx_mask = np.empty((n_samples, k), dtype=bool)  # Matrice n_samples x k, Cambiamento np.zeros -> np.empty

            # np.isin() controlla se ogni elemento di un array appartiene a un insieme di valori
            categorical_mask = np.fromiter( # cambiamento
                (f in categorical_set for f in active_features),
                dtype=bool,
                count=len(active_features)
            )  # cambiamento # Quali split sono categorici?
            numerical_mask = ~categorical_mask  # Faccio la negazione cosi ottengo i numerici

            # X@A <= B || X@A==B
            num_features = active_features[numerical_mask]
            num_thresholds = active_thresholds[numerical_mask]

            cat_features = active_features[categorical_mask]
            cat_thresholds = active_thresholds[categorical_mask]

            if len(num_features) > 0:
                X_num = X[:, num_features]
                sx_mask[:, numerical_mask] = X_num <= num_thresholds

            if len(cat_features) > 0:
                X_cat = X[:, cat_features]
                sx_mask[:, categorical_mask] = X_cat == cat_thresholds

            info_gain, imp_parent, imp_left, imp_right = Impurity.calculate_gain(sx_mask, y_onehot, self.impurity_fun)

            local_best = np.argmax(info_gain)  # Indice del best gain
            if info_gain[local_best] > best_gain:
                best_gain = info_gain[local_best]
                best_feature = active_features[local_best]
                best_threshold = active_thresholds[local_best]

                self.impurity[0] = imp_parent
                self.impurity[1] = imp_left[local_best]
                self.impurity[2] = imp_right[local_best]

        self.feature = best_feature
        self.threshold = best_threshold
        self.is_categorical = self.feature in self.categorical

        # no split
        if self.feature is None:
            raise NoSplitFoundWarning(f"No split found for X {X.shape} and y {np.unique(y)}")

        return self

    def apply(self, X):
        """
        Restituisce un array di:
                    0 se il record "i" va a sx
                    1 se il record "i" va a dx
        """

        dim_X = X.shape[0]  # Numero dei record

        if self.feature is None:
            return np.zeros(dim_X, dtype=np.int_)  # tutti a sinistra

        y_pred = np.ones(dim_X)  # tutti a destra
        X_feature = X[:, self.feature]  # prendo i valori della feature migliore trovata

        if not self.is_categorical:
            y_pred[X_feature <= self.threshold] = 0  # a sinistra
        else:
            y_pred[X_feature == self.threshold] = 0  # a sinistra

        return y_pred + 1

    # @profile
    def build_splits(self, X, features):
        thresholds = [np.unique(X[:, f]) for f in features]

        active_thresholds = np.concatenate(thresholds).astype(np.float32, copy=False)

        active_features = np.concatenate([
            np.full(len(thr), f, dtype=np.int32)
            for f, thr in zip(features, thresholds)
        ])

        return active_features, active_thresholds
