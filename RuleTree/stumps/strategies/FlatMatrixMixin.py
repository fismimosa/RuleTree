import numpy as np

class FlatMatrixMixin:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.categorical_set = None
        self.unique_thresholds = None

    def _select_dataset_subset(self, X, y, idx):
        if idx is None:  # Prendo la porzione di features e colonna target che mi interessa
            idx = slice(None)

        X = X[idx].astype(np.float32, copy=False)
        y = y[idx]
        return X, y

    def _prepare_data(self, X, y, idx, context, sample_weight):
        data = super()._prepare_data(
            X=X,
            y=y,
            idx=idx,
            context=context,
            sample_weight=sample_weight
        )

        X = data["X"]
        m = data["m"]

        if self.categorical_set is None:
            self.categorical_set = set(self.categorical)

        self.unique_thresholds = [
            np.unique(X[:, f])
            for f in range(m)
        ]

        return data

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
