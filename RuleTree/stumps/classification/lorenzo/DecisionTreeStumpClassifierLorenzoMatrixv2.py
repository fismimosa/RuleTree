import numpy as np

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

    def __init__(self, min_samples_leaf=1, class_weight=None, random_state=42, criterion=None):
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

        self.class_weight = class_weight
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state
        self.kwargs = {
            'class_weight': class_weight,
            'random_state': random_state,
            'criterion': criterion,
            'min_samples_leaf': min_samples_leaf
        }

        self.impurity = [0.0, 0.0, 0.0]  # padre, sx, dx
        self.impurity_fun = Impurity.gini  # default gini

        # Matrici
        self.A = None
        self.B = None

        if criterion == "entropy":
            self.impurity_fun = Impurity.entropy

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

        X = X[idx]
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

        active_features, active_thresholds = self.build_splits(X, range(m))
        k = len(active_features)

        # Maschere per split numerici e categorici
        sx_mask = np.zeros((n_samples, k), dtype=bool)  # Matrice n_samples x k
        # Se sx_mask[i, j] = True allora il record i va a sinistra nello split j
        # Se sx_mask[i, j] = False allora il record i va a destra nello split j

        # np.isin() controlla se ogni elemento di un array appartiene a un insieme di valori
        categorical_mask = np.isin(active_features, self.categorical)  # Quali split sono categorici?
        numerical_mask = ~categorical_mask  # Faccio la negazione cosi ottengo i numerici

        # TODO capire come si puo fare la regressione con il mio metodo, quando fatto prendo FairRuleTreeRegressor ecc. e guardare come si puo implementare nel mio metodo.
        # X@A <= B || X@A==B
        sx_mask[:, numerical_mask] = X[:, active_features[numerical_mask]] <= active_thresholds[numerical_mask]
        sx_mask[:, categorical_mask] = X[:, active_features[categorical_mask]] == active_thresholds[categorical_mask]

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
            y_onehot[np.arange(n_samples), y_idx] = sample_weight  # Altrimenti assegno peso sample_weight_i, TODO

        # Peso le classi
        if class_weight is not None:
            class_weight_vec = np.array([class_weight[c] for c in classes])  # Trasformo dizionario in vettore
            y_onehot *= class_weight_vec

        info_gain, imp_parent, imp_left, imp_right = Impurity.calculate_gain(sx_mask, y_onehot, self.impurity_fun)


        best_j = np.argmax(info_gain)  # Indice del best gain
        self.feature = active_features[best_j]
        self.threshold = active_thresholds[best_j]

        self.impurity[0] = imp_parent
        self.impurity[1] = imp_left[best_j]
        self.impurity[2] = imp_right[best_j]

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

    def build_splits(self, X, features):
        features_list = []
        thresholds_list = []

        for f in features:
            thr = np.unique(X[:, f])
            features_list.append(np.full(thr.size, f))
            thresholds_list.append(thr)

        return np.concatenate(features_list), np.concatenate(thresholds_list)
