import random
from itertools import combinations

import numba
import numpy as np
import pandas as pd
import psutil
from numba import UnsupportedError, prange, jit
from scipy.spatial.distance import cdist
from sklearn.base import TransformerMixin
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.tree import DecisionTreeClassifier

from RuleTree.utils.shapelet_transform.matrix_to_vector_distances import euclidean, sqeuclidean, cosine, cityblock
from tqdm.auto import tqdm

class TabularShapelets(TransformerMixin):
    __distances = {
        'euclidean': euclidean,
        'sqeuclidean': sqeuclidean,
        'cosine': cosine,
        'cityblock': cityblock,
    }

    def __init__(self,
                 n_shapelets=100,
                 n_ts_for_selection=np.inf,  #int, inf
                 n_features_strategy=2, #tuple, float, int, 'elbow', 'drop', 'new', 'all'
                 selection='random',  #random, cluster, all
                 distance='euclidean',
                 use_combination=True,
                 random_state=42, n_jobs=1):
        super().__init__()

        self.shapelets = None
        if n_jobs == -1:
            n_jobs = psutil.cpu_count()

        if isinstance(distance, str) and distance not in self.__distances:
            raise UnsupportedError(f"Unsupported distance '{distance}'")

        if selection not in ["random", "cluster", "all"]:
            raise UnsupportedError(f"Unsupported selection '{selection}'")

        if selection in ['random', 'cluster'] and np.isinf(n_shapelets):
            raise UnsupportedError('Use selection "all" if n_shapelets is np.inf')
        if selection == 'all' and not np.isinf(n_shapelets):
            raise UnsupportedError('Use selection "random" or "cluster" if n_shapelets is not np.inf')

        self.n_shapelets = n_shapelets
        self.n_ts_for_selection = n_ts_for_selection
        self.n_features_strategy = n_features_strategy
        self.selection = selection
        self.distance = distance
        self.use_combination = use_combination
        self.random_state = random_state
        self.n_jobs = n_jobs

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

        n_ts_for_selection = self.n_ts_for_selection
        if type(n_ts_for_selection) == float:
            n_ts_for_selection = max(1, int(X.shape[0] * min(1, n_ts_for_selection)))

        sub_idx = np.random.choice(X.shape[0], min(n_ts_for_selection, X.shape[0]), replace=False)
        X_sub = X[sub_idx]
        y_sub = y[sub_idx]

        self.subsequences_index = self._selectable_features(X_sub, y_sub)
        if self.selection == 'random':
            self.shapelets = self._candidate_pivots_random(X_sub, y_sub, self.subsequences_index)
        elif self.selection == 'all':
            self.shapelets = self._candidate_pivots_all(X_sub, y_sub, self.subsequences_index)
        elif self.selection == 'cluster':
            self.shapelets = self._candidate_pivots_cluster(X_sub, y_sub, self.subsequences_index)

        return self

    def _selectable_features_smart(self, X, y):
        model = RandomForestClassifier(n_estimators=1000, n_jobs=self.n_jobs, random_state=self.random_state)
        if np.unique(y).dtype == float:
            model = RandomForestRegressor(n_estimators=1000, n_jobs=self.n_jobs, random_state=self.random_state)
        model.fit(X, y)
        feature_importances_ = model.feature_importances_
        sorted_feature_importances_idx = np.argsort(-feature_importances_)

        if self.n_features_strategy == 'drop':
            max_gap = np.argmax(np.abs(np.diff(feature_importances_[sorted_feature_importances_idx])))
            best_features_idx = sorted_feature_importances_idx[:max_gap + 1]
        else:
            knee_feature_idx = self.__utils_auto_get_knee_point_value(
                feature_importances_[sorted_feature_importances_idx])
            best_features_idx = sorted_feature_importances_idx[:knee_feature_idx + 1]

        subsequences_index = dict()
        if self.use_combination:
            for n_features in range(1, len(best_features_idx)+1):
                subsequences_index[n_features] = list(combinations(best_features_idx, n_features))
        else:
            subsequences_index[len(best_features_idx)] = [best_features_idx]
        return subsequences_index

    def _old_selectable_features_smart(self, X, y):
        subsequences_index = dict()
        for n_features in tqdm(range(1, X.shape[1]+1)):
            subsequences_index[n_features] = list(combinations([i for i in range(X.shape[1])], n_features))

        model = RandomForestClassifier(n_estimators=1000, n_jobs=self.n_jobs, random_state=self.random_state)
        if np.unique(y).dtype == float:
            model = RandomForestRegressor(n_estimators=1000, n_jobs=self.n_jobs, random_state=self.random_state)
        model.fit(X, y)
        feature_importances_ = model.feature_importances_
        sorted_feature_importances_idx = np.argsort(-feature_importances_)

        if self.n_features_strategy == 'drop':
            max_gap = np.argmax(np.abs(np.diff(feature_importances_[sorted_feature_importances_idx])))
            best_features_idx = sorted_feature_importances_idx[:max_gap + 1]
        else:
            knee_feature_idx = self.__utils_auto_get_knee_point_value(
                feature_importances_[sorted_feature_importances_idx])
            best_features_idx = sorted_feature_importances_idx[:knee_feature_idx + 1]
        filtered_subsequences_index = {}

        for n_features, subsequences in subsequences_index.items():
            if n_features > len(best_features_idx):
                continue
            valid_subs = [s for s in subsequences if all(i in best_features_idx for i in s)]
            if valid_subs:
                filtered_subsequences_index[n_features] = valid_subs

        return filtered_subsequences_index

    def _selectable_features_iterative(self, X, y, min_n_features, max_n_features):
        available_features_idx = np.arange(0, X.shape[1])
        selected_feature_idxs = []
        filtered_subsequences_index = dict()

        for n_feature in range(min_n_features, max_n_features):
            features_to_test = []
            candidate_subsequences = []
            for subsequence in combinations([i for i in range(X.shape[1])], n_feature):
                if all([i in available_features_idx for i in subsequence]) and \
                    not any([i in selected_feature_idxs for i in subsequence]):
                    candidate_subsequences.append(subsequence)
                    X_combination = np.ones(X.shape) * np.nan
                    X_combination[:, subsequence] = X[:, subsequence]
                    features_to_test.append(np.unique(X_combination, axis=0))
            if len(features_to_test) == 0:
                break
            candidate_shapelets_it = np.vstack(features_to_test)
            candidate_shapelets_it_transformed = TabularShapelets._transform(X=X, y=y, n_jobs=self.n_jobs,
                                                                             shapelets=candidate_shapelets_it,
                                                                             distance=self.__get_distance())
            dt = DecisionTreeClassifier(max_depth=1)
            dt.fit(candidate_shapelets_it_transformed, y)
            selected_column = dt.tree_.feature[0]
            filtered_subsequences_index[n_feature] = [candidate_subsequences[selected_column//X.shape[0]]]
            selected_feature_idxs += list(filtered_subsequences_index[n_feature][0])

        return filtered_subsequences_index

    def _selectable_features(self, X:np.ndarray, y:np.ndarray):
        min_n_features, max_n_features = self._compute_n_features(X)


        if self.n_features_strategy in ['elbow', 'drop', 'new']:
            return self._selectable_features_smart(X, y)
        elif self.n_features_strategy == 'iterative':
            return self._selectable_features_iterative(X, y)
        else:
            subsequences_index = dict()
            for n_features in tqdm(range(min_n_features, max_n_features)):
                subsequences_index[n_features] = list(combinations([i for i in range(X.shape[1])], n_features))
            return subsequences_index


    def _candidate_pivots_all(self, X:np.ndarray, y:np.ndarray, subsequences_index:dict):
        res_data = [
            np.unique(np.where(np.isin(np.arange(X.shape[1]), combination), X, np.nan), axis=0)
            for combinations in subsequences_index.values()
            for combination in combinations
        ]

        if len(res_data) == 0:
            print()

        return np.vstack(res_data)

    def _subsample(self, X: np.ndarray, y: np.ndarray):
        if y is None:
            subsample_idx = np.random.RandomState(self.random_state).choice(
                X.shape[0],
                size=min(self.n_shapelets, X.shape[0]),
                replace=False
            )
        else:
            try:
                splitter = StratifiedShuffleSplit(n_splits=1,
                                                  test_size=min(self.n_shapelets, X.shape[0]),
                                                  random_state=self.random_state)
                subsample_idx = next(splitter.split(X, y))[1]
            except Exception as e:
                subsample_idx = np.random.RandomState(self.random_state).choice(
                    X.shape[0],
                    size=min(self.n_shapelets, X.shape[0]),
                    replace=False
                )

        return X[subsample_idx]

    def _candidate_pivots_random(self, X:np.ndarray, y:np.ndarray, subsequences_index:dict):
        res_data = []
        n_shapelet_for_stratification = []
        for n_features, combinations in subsequences_index.items():
            res_data_n_features = []
            for combination in combinations:
                X_combination = np.ones(X.shape) * np.nan
                X_combination[:, combination] = X[:, combination]
                res_data_n_features.append(self._subsample(X=np.unique(X_combination, axis=0), y=None))
            res_data_n_features = np.vstack(res_data_n_features)
            n_shapelets = min(max(self.n_shapelets, 2), res_data_n_features.shape[0])
            res_data.append(res_data_n_features[np.random.choice(res_data_n_features.shape[0], size=n_shapelets)])
            n_shapelet_for_stratification += ([n_features]*n_shapelets)

        candidate_pivots = np.vstack(res_data)
        return self._subsample(X=candidate_pivots, y=n_shapelet_for_stratification)


    def _run_kmeans_and_return_centroids(self, X):
        X_filled = np.nan_to_num(X, nan=0.0)
        km = KMeans(n_clusters=min(len(X_filled), self.n_shapelets), random_state=self.random_state)
        km.fit(X_filled)
        distances = cdist(km.cluster_centers_, X_filled, metric=self.distance)
        nearest_indices = np.argmin(distances, axis=1)

        return X[nearest_indices]

    def _candidate_pivots_cluster(self, X:np.ndarray, y:np.ndarray, subsequences_index:dict):
        res_data = []
        for combinations in subsequences_index.values():
            for combination in combinations:
                res_data += [
                    self._run_kmeans_and_return_centroids(
                        np.unique(np.where(np.isin(np.arange(X.shape[1]), combination), X, np.nan), axis=0)
                    )
                ]

        X_all = np.vstack(res_data)

        return self._run_kmeans_and_return_centroids(X=X_all)

    def _compute_n_features(self, X):
        max_n_features = X.shape[1]
        min_n_features = 1
        if type(self.n_features_strategy) is tuple and len(self.n_features_strategy) == 2:
            min_n_features = self.n_features_strategy[0]
            max_n_features = self.n_features_strategy[1]
        elif type(self.n_features_strategy) is float:
            min_n_features = int(max_n_features * self.n_features_strategy)
            max_n_features = min_n_features + 1
        elif type(self.n_features_strategy) is int:
            min_n_features = self.n_features_strategy
            max_n_features = min_n_features + 1

        assert min_n_features < max_n_features

        return min_n_features, max_n_features

    def __utils_auto_get_knee_point_value(self, values):
        y = values
        x = np.arange(0, len(y))

        index, max_d = 0, -float('infinity')
        for i in range(0, len(x)):
            c = self.__utils_auto_closest_point_on_segment(a=[x[0], y[0]], b=[x[-1], y[-1]], p=[x[i], y[i]])
            d = np.sqrt((c[0] - x[i]) ** 2 + (c[1] - y[i]) ** 2)
            if d > max_d:
                max_d, index = d, i

        return index

    def __utils_auto_closest_point_on_segment(self, a, b, p):
        sx1, sy1 = a[0], a[1]
        sx2, sy2 = b[0], b[1]
        px, py = p[0], p[1]

        x_delta, y_delta = sx2 - sx1, sy2 - sy1

        if x_delta == 0 and y_delta == 0:
            return p

        u = ((px - sx1) * x_delta + (py - sy1) * y_delta) / (x_delta * x_delta + y_delta * y_delta)
        cp_x = sx1 + u * x_delta
        cp_y = sy1 + u * y_delta

        return [cp_x, cp_y]

    def transform(self, X, y=None):
        return TabularShapelets._transform(X, y, self.n_jobs, self.shapelets, self.__get_distance())

    @classmethod
    def _transform(cls, X, y=None, n_jobs=1, shapelets=None, distance=euclidean, **transform_params):
        old_n_threads = numba.get_num_threads()
        numba.set_num_threads(n_jobs)
        dist_matrix = _compute_distance(X, shapelets, distance)
        numba.set_num_threads(old_n_threads)

        return dist_matrix

    def _optimize_memory(self, selected_shapelet_idx:np.ndarray):
        self.shapelets = self.shapelets[selected_shapelet_idx]
        return True


@jit(parallel=True)
def _compute_distance(X: np.ndarray, shapelets: np.ndarray, distance):
    res = np.ones((X.shape[0], shapelets.shape[0]), dtype=np.float32) * np.inf

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
    y = df['target'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    st = TabularShapelets(n_shapelets=5,
                          n_features_strategy='all', #tuple, float, int, 'elbow', 'drop', 'new', 'all'
                          mi_n_neighbors=100,
                          n_jobs=10,
                          distance='euclidean',
                          selection='cluster' #random, cluster, all
                          )

    X_train_transform = st.fit_transform(X_train, y_train)
    X_test_transform = st.transform(X_test)

    print(X_train_transform.shape)

    rf = RandomForestClassifier(n_estimators=100, n_jobs=-1)
    rf.fit(X_train_transform, y_train)

    y_pred = rf.predict(X_test_transform)

    print(classification_report(y_test, y_pred))

    shapelets = pd.DataFrame(st.shapelets, columns=[df.drop(columns="target").columns])

    print(shapelets)
