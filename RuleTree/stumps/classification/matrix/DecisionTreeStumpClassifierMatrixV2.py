import numpy as np
from RuleTree.stumps.interfaces.ClassifierBaseMatrix import ClassifierBaseMatrix


class DecisionTreeStumpClassifierMatrixV2(ClassifierBaseMatrix):
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

    def __init__(self, min_samples_leaf=1, class_weight=None, random_state=42, criterion=None, batch_size=None, n_bins=None):

        super().__init__(
            min_samples_leaf=min_samples_leaf,
            class_weight=class_weight,
            random_state=random_state,
            criterion=criterion,
            batch_size=batch_size,
            n_bins=n_bins
        )

        self.categorical_set = None
        self.unique_thresholds = None

    def _select_dataset_subset(self, X, y, idx):
        if idx is None:  # Prendo la porzione di features e colonna target che mi interessa
            idx = slice(None)

        X = X[idx].astype(np.float32, copy=False)
        y = y[idx]
        return X, y

    def _prepare_data(self, X, y, idx, context, sample_weight):
        X, y, y_onehot, n_samples, m, batch_size = super()._prepare_data(
            X=X,
            y=y,
            idx=idx,
            context=context,
            sample_weight=sample_weight
        )

        if self.categorical_set is None:
            self.categorical_set = set(self.categorical)

        self.unique_thresholds = [
            np.unique(X[:, f])
            for f in range(m)
        ]

        return X, y, y_onehot, n_samples, m, batch_size


    def _build_sxmask(self, X, active_features, active_thresholds, k, n_samples):
        # Maschere per split numerici e categorici
        sx_mask = np.empty((n_samples, k), dtype=bool)  # Matrice n_samples x k, Cambiamento np.zeros -> np.empty

        # np.isin() controlla se ogni elemento di un array appartiene a un insieme di valori
        categorical_mask = np.fromiter(  # cambiamento
            (f in self.categorical_set for f in active_features),
            dtype=bool,
            count=len(active_features)
        )  # cambiamento # Quali split sono categorici?
        numerical_mask = ~categorical_mask  # Faccio la negazione cosi ottengo i numerici

        # X@A <= B || X@A==B
        num_features = active_features[numerical_mask]
        num_thresholds = active_thresholds[numerical_mask]

        cat_features = active_features[categorical_mask]
        cat_thresholds = active_thresholds[categorical_mask]

        if num_features.size:
            sx_mask[:, numerical_mask] = X[:, num_features] <= num_thresholds

        if cat_features.size:
            sx_mask[:, categorical_mask] = X[:, cat_features] == cat_thresholds

        return sx_mask

    def _generate_splits(self, X, features_batch, m=None):
        thresholds = [
            self.unique_thresholds[f]
            for f in features_batch
        ]
        lengths = np.fromiter((len(t) for t in thresholds), dtype=np.int32)

        active_thresholds = np.concatenate(thresholds).astype(np.float32, copy=False)
        active_features = np.repeat(features_batch, lengths)

        return active_features, active_thresholds
