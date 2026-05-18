import numpy as np
from line_profiler import profile

from RuleTree.exceptions import NoSplitFoundWarning
from RuleTree.stumps.classification import DecisionTreeStumpClassifier
from RuleTree.stumps.classification.lorenzo.utils.Impurity import Impurity


class DecisionTreeStumpClassifierLorenzoMatrixv1(DecisionTreeStumpClassifier):
    """
    A decision tree stump classifier that extends DecisionTreeStumpClassifier.
    Version: Matrix v1

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
            'batch_size': batch_size # L'ho dovuto inserire qui sennò andando giu nell'albero diventava None
        }

        self.impurity = [0.0, 0.0, 0.0]  # padre, sx, dx
        self.impurity_fun = Impurity.gini  # default gini


        if criterion == "entropy":
            self.impurity_fun = Impurity.entropy

    @profile
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
        batch_size = self.batch_size or m # Dimensione del batch oppure m stesso se non specificato

        best_gain = -np.inf # Miglior gain in assoluto
        best_feature = None # Miglior feature in assoluto associata al best gain
        best_threshold = None # Miglior threshold in assoluto associata al best gain

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

        for start in range(0, m, batch_size): # Incremento di batch_size
            end = min(start + batch_size, m)
            features_batch = list(range(start, end))
            A, B = self.build_A_B(X, m, features_batch)  # Costruisco le matrici A e B
            k = A.shape[1]  # Numero di split

            # Per ogni colonna di A, trova l’indice della riga con il valore più grande (la feature stessa)
            # array lunghezza k
            active_features = np.argmax(A, axis=0)  # active_features[j] = feature usata dal j-esimo split

            # Estraggo il threshold di ogni split j usando la feature che attivo in A (active_features[j])
            # array lunghezza k
            active_thresholds = B[A]  # thresholds[j] = B[active_features[j],j]

            # Maschere per split numerici e categorici
            sx_mask = np.zeros((n_samples, k), dtype=bool)  # Matrice n_samples x k
            # Se sx_mask[i, j] = True allora il record i va a sinistra nello split j
            # Se sx_mask[i, j] = False allora il record i va a destra nello split j

            # np.isin() controlla se ogni elemento di un array appartiene a un insieme di valori
            categorical_mask = np.isin(active_features, self.categorical)  # Quali split sono categorici?
            numerical_mask = ~categorical_mask  # Faccio la negazione cosi ottengo i numerici

            # X@A <= B || X@A==B
            sx_mask[:, numerical_mask] = X[:, active_features[numerical_mask]] <= active_thresholds[numerical_mask]
            sx_mask[:, categorical_mask] = X[:, active_features[categorical_mask]] == active_thresholds[categorical_mask]

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

    @profile
    def build_A_B(self, X, m, numerical):
        """
        Costruisce le matrici A e B.

        A[i,j] = 1 se lo split j usa la feature i
               = 0 altrimenti
        B[i,j] = valore del threshold corrispondente
               = 0 se la feature i non è usata

        Righe = m = numero feature
        Colonne = k = \sum_i^m |unique(X[:,feature])| = numero totale split possibili
        """

        # Tutti i threshold possibili per ogni feature
        thresholds = [np.unique(X[:, f]) for f in numerical]

        # Numero totale degli split (numero colonne matr)
        k = sum(len(i) for i in thresholds)

        # Inizializzo matrici a 0
        A = np.zeros((m, k), dtype=np.bool)  # uint8 per risparmiare spazio in memoria (1 byte)
        B = np.zeros((m, k), dtype=np.float32)  # matrice B

        col = 0  # Colonna corrente

        # Riempimento matrici
        for f, t in zip(numerical, thresholds):
            n_t = len(t)
            A[f, col:col + n_t] = 1  # gli split usano la feature f
            B[f, col:col + n_t] = t

            col += n_t  # sposto il blocco colonne

        return A, B
