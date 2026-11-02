import numpy as np

from RuleTree.exceptions import NoSplitFoundWarning
from RuleTree.stumps.classification import DecisionTreeStumpClassifier

from RuleTree.utils.data_utils import gini, entropy


class DecisionTreeStumpClassifierBase(DecisionTreeStumpClassifier):
    """
    A decision tree stump classifier that extends DecisionTreeStumpClassifier.
    Version: Base

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

    def __init__(self, min_samples_leaf=1, class_weight=None, random_state=42, criterion=None, protected_attribute_idx=None, fairness_weight=1.0):
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
        self.protected_attribute_idx = protected_attribute_idx
        self.fairness_weight = fairness_weight
        self.protected_attribute = None
        self.ideal = None
        self.kwargs = {
            'class_weight': class_weight,
            'random_state': random_state,
            'criterion': criterion,
            'min_samples_leaf': min_samples_leaf,
            'protected_attribute_idx': protected_attribute_idx,
            'fairness_weight': fairness_weight
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
            DecisionTreeStumpClassifier: The fitted classifier (self).
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

        if self.protected_attribute_idx is not None:
            self.protected_attribute = X[:, self.protected_attribute_idx].astype(int)

            self.ideal = (
                    np.bincount(self.protected_attribute)
                    / len(self.protected_attribute)
            )

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
                # Calcolo impurità per figlio
                imp_left = self.impurity_fun(
                    y[sx_mask], None, class_weight
                )

                imp_right = self.impurity_fun(
                    y[dx_mask], None, class_weight
                )

                sx_n = np.sum(sx_mask)
                dx_n = np.sum(dx_mask)

                # Fairness
                if self.protected_attribute is not None and self.ideal is not None:
                    protected_left = self.protected_attribute[sx_mask]
                    protected_right = self.protected_attribute[dx_mask]

                    protected_prob_left = (
                            np.bincount(protected_left, minlength=len(self.ideal))
                            / len(protected_left)
                    )

                    protected_prob_right = (
                            np.bincount(protected_right, minlength=len(self.ideal))
                            / len(protected_right)
                    )

                    mfc_left = np.mean(
                        np.abs(protected_prob_left - self.ideal)
                    )

                    mfc_right = np.mean(
                        np.abs(protected_prob_right - self.ideal)
                    )

                    # Fairness penalty sui due figli
                    imp_left += self.fairness_weight * mfc_left * imp_left
                    imp_right += self.fairness_weight * mfc_right * imp_left

                # Gain
                weighted_children = (
                        (sx_n / len_y) * imp_left
                        + (dx_n / len_y) * imp_right
                )

                current_info_gain = (
                        self.impurity[0]
                        - weighted_children
                )

                if best_info_gain < current_info_gain:
                    best_info_gain = current_info_gain
                    self.threshold = threshold
                    self.feature = feature
                    self.is_categorical = feature in self.categorical

                    self.impurity[1] = imp_left
                    self.impurity[2] = imp_right

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