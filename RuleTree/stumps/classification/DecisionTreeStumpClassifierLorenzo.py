import numpy as np

from RuleTree.exceptions import NoSplitFoundWarning
from RuleTree.stumps.classification import DecisionTreeStumpClassifier

from RuleTree.utils.data_utils import gini, entropy


class DecisionTreeStumpClassifierLorenzo(DecisionTreeStumpClassifier):
    """
    A decision tree stump classifier that implements RuleTreeBaseStumpLorenzo.

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
        self.impurity_fun = gini  # default gini

        if criterion == "entropy":
            self.impurity_fun = entropy

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

        len_x = len(X)
        len_y = len(y)

        class_weight = None
        if self.class_weight == "balanced":
            class_weight = {}
            for class_label in np.unique(y):
                class_weight[class_label] = len_x / (len(self.classes_) * len(y[y == class_label]))

        if hasattr(context, 'categorical'):
            self.categorical = context.categorical
            self.numerical = context.numerical
        else:
            self.feature_analysis(X, y)
            context.categorical = self.categorical
            context.numerical = self.numerical

        best_info_gain = -np.inf  # Inizializzo gain a -infty

        # Calcolo impurità del padre una volta sola
        self.impurity[0] = self.impurity_fun(y, sample_weight, class_weight)

        for feature in range(X.shape[1]):  # Per ogni feature numerica
            for threshold in np.unique(X[:, feature]):  # Per ogni valore numerico univoco
                if feature in self.categorical:
                    sx_mask = X[:, feature] == threshold  # Maschera booleana per figlio sx
                    dx_mask = X[:, feature] != threshold  # Maschera booleana per figlio dx
                else:
                    sx_mask = X[:, feature] <= threshold  # Maschera booleana per figlio sx
                    dx_mask = X[:, feature] > threshold  # Maschera booleana per figlio dx

                # Calcolo impurità per figlio sx e dx
                self.impurity[1] = self.impurity_fun(y[sx_mask], None, class_weight)
                self.impurity[2] = self.impurity_fun(y[dx_mask], None, class_weight)

                # Sommo #features andate a sx e dx
                sx_n = np.sum(sx_mask)
                dx_n = np.sum(dx_mask)

                # Calcolo gain
                weighted_children = (sx_n / len_y) * self.impurity[1] + (dx_n / len_y) * self.impurity[2]
                current_info_gain = self.impurity[0] - weighted_children

                if best_info_gain < current_info_gain:
                    best_info_gain = current_info_gain
                    self.threshold = threshold
                    self.feature = feature
                    self.is_categorical = feature in self.categorical

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
