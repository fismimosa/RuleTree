import random
from itertools import combinations

import numba
import numpy as np
import pandas as pd
import psutil
from numba import UnsupportedError, prange, jit
from sklearn.base import TransformerMixin
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.utils import resample

from RuleTree.utils.shapelet_transform.matrix_to_vector_distances import euclidean, sqeuclidean, cosine, cityblock


class TabularShapelets(TransformerMixin):
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
                 min_n_features=2,
                 max_n_features='auto', #auto-> sqrt(X.shape[1]), sqrt-> sqrt(X.shape[1]), all -> all features, int
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
        self.min_n_features = min_n_features
        self.max_n_features = max_n_features
        self.selection = selection
        self.distance = distance
        self.mi_n_neighbors = mi_n_neighbors
        self.random_state = random_state
        self.n_jobs = n_jobs

        random.seed(random_state)

    def __get_distance(self):
        """
        Get the distance function based on the distance parameter.

        Returns
        -------
        callable
            The distance function to use for computing shapelet distances.
        """
        if isinstance(self.distance, str):
            return self.__distances[self.distance]
        return self.distance

    def fit(self, X, y=None, **fit_params):
        """
        Fit the TabularShapelets transformer on training data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training tabular data.

        y : array-like of shape (n_samples,), default=None
            Target values for supervised learning.

        **fit_params : dict
            Additional parameters passed to selection methods.

        Returns
        -------
        self : TabularShapelets
            Returns self.
        """
        random.seed(self.random_state)

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
        max_n_features = -1
        if self.max_n_features == 'auto':
            max_n_features = int(np.sqrt(max_n_features))
        elif self.max_n_features == 'sqrt':
            max_n_features = int(np.sqrt(max_n_features))
        elif self.max_n_features == 'all':
            max_n_features = X.shape[1]
        else:
            max_n_features = self.max_n_features

        subsequences_index = []
        for n_features in range(self.min_n_features, max_n_features):
            subsequences_index += list(combinations([i for i in range(X.shape[1])], n_features))

        res_data = []
        for combination in subsequences_index:
            X_combination = np.ones(X.shape) * np.nan
            X_combination[:, combination] = X[:, combination]
            res_data.append(np.unique(X_combination, axis=0))

        return np.vstack(res_data)

    def __fit_selection_random(self, candidate_shapelets: np.ndarray, X, y):
        n_shapelets = min(self.n_shapelets, candidate_shapelets.shape[0])
        return candidate_shapelets[np.random.choice(candidate_shapelets.shape[0], size=n_shapelets, replace=False)]


    def __fit_selection_mutual_info(self, candidate_shapelets: np.ndarray, X, y, mutual_info_fun):
        if y is None:
            raise UnsupportedError("Mutual information is not suitable for unsupervised tasks.")

        idx_to_test = resample(range(X.shape[0]), stratify=y, random_state=self.random_state)

        old_n_threads = numba.get_num_threads()
        numba.set_num_threads(self.n_jobs)
        dist = _compute_distance(X[idx_to_test], candidate_shapelets, self.__get_distance())
        numba.set_num_threads(old_n_threads)

        labels, labels_count = np.unique(y, return_counts=True)
        if np.sum(labels_count) == len(labels):
            scores = np.zeros((len(labels), ))
        else:
            scores = mutual_info_fun(dist, y,
                                     n_jobs=1,
                                     n_neighbors=min(dist.shape[0], self.mi_n_neighbors),
                                     discrete_features=False)
        if len(candidate_shapelets) == self.n_shapelets:
            return candidate_shapelets
        return candidate_shapelets[np.argpartition(scores, -min(scores.shape[0], self.n_shapelets)) \
            [-min(scores.shape[0], self.n_shapelets):]]

    def __fit_selection_cluster(self, candidate_shapelets, X, y):
        old_n_threads = numba.get_num_threads()
        numba.set_num_threads(self.n_jobs)
        dist_matrix = _compute_distance(candidate_shapelets, candidate_shapelets, self.__get_distance())
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
        dist_matrix = _compute_distance(X, self.shapelets, self.__get_distance())
        numba.set_num_threads(old_n_threads)

        return dist_matrix


@jit(parallel=True)
def _compute_distance(X: np.ndarray, shapelets: np.ndarray, distance):
    res = np.ones((X.shape[0], shapelets.shape[0]), dtype=np.float32) * np.inf
    w = shapelets.shape[-1]

    for idx, shapelet in enumerate(shapelets):
        cols = np.where(np.isfinite(shapelet))[0]
        res[:, idx] = distance(X[:, cols], shapelet[cols])

    return res

if __name__ == '__main__':
    iris = load_iris()
    df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    df['target'] = iris.target
    df['target'] = df['target'].map({i: name for i, name in enumerate(iris.target_names)})

    X = df[iris.feature_names].values
    y = df['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    st = TabularShapelets(n_shapelets=1,
                          min_n_features=2,
                          max_n_features=4,
                          mi_n_neighbors=100,
                          n_jobs=1,
                          distance='euclidean',
                          selection='mi_clf')

    X_train_transform = st.fit_transform(X_train, y_train)
    X_test_transform = st.transform(X_test)

    print(X_train_transform.shape)

    rf = RandomForestClassifier(n_estimators=100, n_jobs=-1)
    rf.fit(X_train_transform, y_train)

    y_pred = rf.predict(X_test_transform)

    print(classification_report(y_test, y_pred))

    shapelets = pd.DataFrame(st.shapelets, columns=[df.drop(columns="target").columns])

    print(shapelets)
