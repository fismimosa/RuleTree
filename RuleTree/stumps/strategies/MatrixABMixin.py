import numpy as np

class MatrixABMixin:
    def _build_sxmask(self, X, active_features, active_thresholds, k, n_samples):
        # Maschere per split numerici e categorici
        sx_mask = np.zeros((n_samples, k), dtype=bool)  # Matrice n_samples x k
        # Se sx_mask[i, j] = True allora il record i va a sinistra nello split j
        # Se sx_mask[i, j] = False allora il record i va a destra nello split j

        # np.isin() controlla se ogni elemento di un array appartiene a un insieme di valori
        categorical_mask = np.isin(active_features, self.categorical)  # Quali split sono categorici?
        numerical_mask = ~categorical_mask  # Faccio la negazione cosi ottengo i numerici

        # X@A <= B || X@A==B
        sx_mask[:, numerical_mask] = X[:, active_features[numerical_mask]] <= active_thresholds[numerical_mask]
        sx_mask[:, categorical_mask] = X[:, active_features[categorical_mask]] == active_thresholds[
            categorical_mask]

        return sx_mask

    def _generate_splits(self, X, features_batch, m=None):
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
        thresholds = [np.unique(X[:, f]) for f in features_batch]

        # Numero totale degli split (numero colonne matr)
        k = sum(len(i) for i in thresholds)

        # Inizializzo matrici a 0
        A = np.zeros((m, k), dtype=np.bool)  # uint8 per risparmiare spazio in memoria (1 byte)
        B = np.zeros((m, k), dtype=np.float32)  # matrice B

        col = 0  # Colonna corrente

        # Riempimento matrici
        for f, t in zip(features_batch, thresholds):
            n_t = len(t)
            A[f, col:col + n_t] = 1  # gli split usano la feature f
            B[f, col:col + n_t] = t

            col += n_t  # sposto il blocco colonne

        active_features = np.argmax(A, axis=0)
        active_thresholds = B[A]

        return active_features, active_thresholds