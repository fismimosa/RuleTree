from abc import ABC

import numpy as np

from RuleTree.base.RuleTreeBaseStump import RuleTreeBaseStump
from RuleTree.exceptions import NoSplitFoundWarning


class BaseMatrix(RuleTreeBaseStump, ABC):
    def __init__(self, min_samples_leaf=1, random_state=42, criterion=None, batch_size=None, n_bins=None):
        self.is_categorical = False
        self.feature = None
        self.threshold = None
        self.categorical = None
        self.numerical = None
        self.bin_edges = {}

        self.batch_size = batch_size
        self.n_bins = n_bins

        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state
        self.kwargs = {
            'random_state': random_state,
            'criterion': criterion,
            'min_samples_leaf': min_samples_leaf,
            'batch_size': batch_size,
            'n_bins': n_bins
        }

        self.impurity = [0.0, 0.0, 0.0]  # padre, sx, dx

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

    def _prepare_data(self, X, y, idx, context):
        X, y = self._select_dataset_subset(X, y, idx)
        n_samples = len(X)

        if hasattr(context, 'categorical'):
            self.categorical = context.categorical
            self.numerical = context.numerical
        else:
            self.feature_analysis(X, y)
            context.categorical = self.categorical
            context.numerical = self.numerical

        m = X.shape[1]  # Numero di features
        batch_size = self.batch_size or m  # Dimensione del batch oppure m stesso se non specificato

        if self.n_bins is not None:
            X = self._apply_binning(X)

        return X, y, n_samples, m, batch_size

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
