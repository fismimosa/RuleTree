import time
import random

import numba
import numpy as np
import pandas as pd
import scipy.io.arff

from numba import jit, prange, UnsupportedError
import psutil
from sklearn.base import TransformerMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.metrics import classification_report
from sklearn.utils import resample

from RuleTree.utils.shapelet_transform.matrix_to_vector_distances import euclidean, sqeuclidean, cosine, cityblock


class Shapelets(TransformerMixin):
    __distances = {
        'euclidean': euclidean,
        'sqeuclidean': sqeuclidean,
        'cosine': cosine,
        'cityblock': cityblock,
    }
    def __init__(self,
                 n_shapelets=100,
                 n_shapelets_for_selection=np.inf, #int, inf, or 'stratified'
                 n_ts_for_selection_per_class=np.inf, #int, inf
                 sliding_window=50,
                 selection='random', #random, mi_clf, mi_reg, cluster
                 distance='euclidean',
                 mi_n_neighbors = 100,
                 random_state=42, n_jobs=1):
        super().__init__()

        self.shapelets = None
        if n_jobs == -1:
            n_jobs = psutil.cpu_count()

        if isinstance(distance, str) and distance not in self.__distances:
            raise UnsupportedError(f"Unsupported distance '{distance}'")

        if selection not in ["random", "mi_clf", "mi_reg", "cluster"]:
            raise UnsupportedError(f"Unsupported selection '{selection}'")

        self.n_shapelets = n_shapelets
        self.n_shapelets_for_selection = n_shapelets_for_selection
        self.n_ts_for_selection_per_class = n_ts_for_selection_per_class
        self.sliding_window = sliding_window
        self.selection = selection
        self.distance = distance
        self.mi_n_neighbors = mi_n_neighbors
        self.random_state = random_state
        self.n_jobs = n_jobs

        random.seed(random_state)

    def __get_distance(self):
        if isinstance(self.distance, str):
            return self.__distances[self.distance]
        return self.distance


    def fit(self, X, y=None, **fit_params):
        # X.shape = (n_records, n_signals, n_obs)
        if X.shape[1] != 1:
            raise NotImplementedError("Multivariate TS are not supported (yet).")

        candidate_shapelets = self.__fit_partition(X, y)

        if self.selection == 'random':
            self.shapelets = self.__fit_selection_random(candidate_shapelets, X, y)
        elif self.selection == 'mi_clf':
            self.shapelets = self.__fit_selection_mutual_info(candidate_shapelets, X, y, mutual_info_classif)
        elif self.selection == 'mi_reg':
            self.shapelets = self.__fit_selection_mutual_info(candidate_shapelets, X, y, mutual_info_regression)
        elif self.selection == 'cluster':
            self.shapelets = self.__fit_selection_cluster(candidate_shapelets, X, y)

        return self




    def __fit_partition(self, X, y):
        if y is None:
            y = np.zeros(X.shape[0])

        classes = np.unique(y)
        n_classes = len(classes)

        candidate_shapelets = []

        if not isinstance(self.n_shapelets_for_selection, str) and np.isinf(self.n_shapelets_for_selection):
            for ts_idx in range(X.shape[0]):
                for position_idx in range(X.shape[-1] - self.sliding_window):
                    candidate_shapelets.append(X[ts_idx, :, position_idx: position_idx + self.sliding_window])

            return np.array(candidate_shapelets)

        if self.n_shapelets_for_selection == 'stratified':
            classes, n_candidate_per_class = np.unique(
                resample(y, stratify=y, random_state=self.random_state, n_samples=int(len(y)**.5)), return_counts=True)
        n_candidate_per_class = [max(1, round(self.n_shapelets_for_selection/n_classes)) for _ in classes]


        for classe in classes:
            X_class = X[y == classe]
            for _ in n_candidate_per_class:
                ts_idx = random.randint(0, len(X_class) - 1)
                start = random.randint(0, X_class.shape[-1] - self.sliding_window)
                stop = start + self.sliding_window
                candidate_shapelets.append(np.copy(X_class[ts_idx, :, start:stop]))

        return np.array(candidate_shapelets)

    def __fit_selection_random(self, candidate_shapelets: np.ndarray, X, y):
        n_shapelets = min(self.n_shapelets, candidate_shapelets.shape[0])
        return candidate_shapelets[np.random.choice(candidate_shapelets.shape[0], size=n_shapelets, replace=False)]

    def __fit_selection_mutual_info(self, candidate_shapelets: np.ndarray, X, y, mutual_info_fun):
        if y is None:
            raise UnsupportedError("Mutual information is not suitable for unsupervised tasks.")

        idx_to_test = resample(range(X.shape[0]), stratify=y, random_state=self.random_state)

        old_n_threads = numba.get_num_threads()
        numba.set_num_threads(self.n_jobs)
        dist = _best_fit(X[idx_to_test], candidate_shapelets, self.__get_distance())
        numba.set_num_threads(old_n_threads)

        scores = mutual_info_fun(dist, y,
                                 n_jobs=self.n_jobs,
                                 n_neighbors=min(dist.shape[0], self.mi_n_neighbors),
                                 discrete_features=False)
        if len(candidate_shapelets) == self.n_shapelets:
            return candidate_shapelets
        return candidate_shapelets[np.argpartition(scores, -min(scores.shape[0], self.n_shapelets))\
            [-min(scores.shape[0], self.n_shapelets):]]

    def __fit_selection_cluster(self, candidate_shapelets, X, y):
        old_n_threads = numba.get_num_threads()
        numba.set_num_threads(self.n_jobs)
        dist_matrix = _best_fit(candidate_shapelets, candidate_shapelets, self.__get_distance())
        numba.set_num_threads(old_n_threads)

        try:
            from sklearn_extra.cluster import KMedoids
            clu = KMedoids(n_clusters=self.n_shapelets, random_state=self.random_state, metric='precomputed')
        except Exception as e:
            raise Exception(f"Please install scikit-learn-extra [{e}]")
        clu.fit(dist_matrix)

        return candidate_shapelets[clu.medoid_indices_]

    def transform(self, X, y=None, **transform_params):
        old_n_threads = numba.get_num_threads()
        numba.set_num_threads(self.n_jobs)
        dist_matrix = _best_fit(X, self.shapelets, self.__get_distance())
        numba.set_num_threads(old_n_threads)

        return dist_matrix

@jit(parallel=True)
def _best_fit(timeseries: np.ndarray, shapelets: np.ndarray, distance):
    res = np.ones((timeseries.shape[0], shapelets.shape[0]), dtype=np.float32)*np.inf
    w = shapelets.shape[-1]

    for ts_idx in prange(timeseries.shape[0]):
        ts = timeseries[ts_idx, 0, :]
        ts_sw = np.lib.stride_tricks.sliding_window_view(ts, w)
        for shapelet_idx, shapelet in enumerate(shapelets[:, 0, :]):
            distance_matrix = distance(ts_sw, shapelet)
            res[ts_idx, shapelet_idx] = np.min(distance_matrix)

    return res





if __name__ == '__main__':
    """random.seed(42)
    X = np.random.rand(10000, 1, 500).astype(np.float32)
    shapelets = np.random.rand(100, 1, 50).astype(np.float32)

    st = Shapelet(n_jobs=-1)

    numba.set_num_threads(1)
    start = time.time()
    res = _best_fit(X, shapelets, euclidean)
    print(round(time.time()-start, 6))
    print(res[:10])"""
    df_train = pd.DataFrame(scipy.io.arff.loadarff('test_dataset/CBF/CBF_TRAIN.arff')[0])
    df_test = pd.DataFrame(scipy.io.arff.loadarff('test_dataset/CBF/CBF_TEST.arff')[0])
    df_train.target = df_train.target.astype(int)
    df_test.target = df_test.target.astype(int)

    X_train = df_train.drop(columns=['target']).values.reshape((-1, 1, 128))
    X_test = df_test.drop(columns=['target']).values.reshape((-1, 1, 128))
    y_train = df_train['target'].values
    y_test = df_test['target'].values

    st = Shapelets(n_shapelets=10, selection='random', mi_n_neighbors=100, n_jobs=-1, distance='cityblock') #euclidean, sqeuclidean, cosine, cityblock

    X_train_transform = st.fit_transform(X_train, y_train)
    X_test_transform = st.transform(X_test)

    rf = RandomForestClassifier(n_estimators=100, n_jobs=-1)
    rf.fit(X_train_transform, y_train)

    y_pred = rf.predict(X_test_transform)

    print(classification_report(y_test, y_pred))
