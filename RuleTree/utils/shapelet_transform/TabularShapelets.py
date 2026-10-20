"""
TabularShapelets: the transform of tabular data into distances from partial pivots, used by the
partial-pivot stumps.
"""

import random
from itertools import combinations

import numba
import numpy as np
import pandas as pd
import psutil
from numba import UnsupportedError, jit
from scipy.spatial.distance import cdist
from sklearn.base import TransformerMixin
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.tree import DecisionTreeClassifier

from RuleTree.utils.shapelet_transform.matrix_to_vector_distances import euclidean, sqeuclidean, cosine, cityblock

class TabularShapelets(TransformerMixin):
    """
    Transform of tabular data into distances from partial pivots: instances restricted to a subset
    of their features (the other features are nan). The distance of an instance from a partial pivot
    is computed only on the features of the pivot.

    The subsets of features (``n_features_strategy``) and then the pivots (``selection``) are chosen
    on a sample of the training instances.

    Parameters
    ----------
    n_shapelets : int or numpy.inf, default=100
        Number of partial pivots (numpy.inf with selection='all').
    n_ts_for_selection : int, float or numpy.inf, default=numpy.inf
        Number (int) or fraction (float) of the training instances used to select the pivots.
    n_features_strategy : int, float, tuple or str, default=2
        Number of features of the pivots: an int, a fraction of the features (float), a range
        ``(min, max)`` (tuple), or chosen from the feature importances of a random forest ('elbow',
        'drop', 'new') or iteratively with depth-1 trees ('iterative').
    selection : {'random', 'cluster', 'all'}, default='random'
        How the pivots are selected among the candidates: at random (stratified by number of
        features), the instances closest to k-means centroids, or all of them.
    distance : {'euclidean', 'sqeuclidean', 'cosine', 'cityblock'} or callable, default='euclidean'
        Distance between an instance and a pivot.
    use_combination : bool, default=True
        With the strategies based on feature importances, whether to use all the combinations of the
        important features or only the set of all of them.
    random_state : int, default=42
        Seed of the random choices.
    n_jobs : int, default=1
        Number of threads used to compute the distances (-1: all the CPUs).

    Attributes
    ----------
    shapelets : numpy.ndarray of shape (n_shapelets, n_features)
        The partial pivots (nan for the features not used).
    subsequences_index : dict
        Number of features -> list of the subsets of features used.

    Raises
    ------
    UnsupportedError
        For an unknown distance or selection, or an n_shapelets inconsistent with the selection.
    """
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
        """Initialize the transform; see the class documentation for the parameters."""
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
        # as in Shapelets.fit: the selection also uses numpy's global random generator
        random.seed(self.random_state)
        np.random.seed(self.random_state)

        n_ts_for_selection = self.n_ts_for_selection
        if isinstance(n_ts_for_selection, float):
            n_ts_for_selection = max(1, int(X.shape[0] * min(1, n_ts_for_selection)))

        sub_idx = np.random.choice(X.shape[0], min(n_ts_for_selection, X.shape[0]), replace=False)
        X_sub = X[sub_idx]
        y_sub = y[sub_idx]

        self.subsequences_index = self._selectable_features(X_sub, y_sub)
        if self.selection == 'random':
            self.shapelets = self._candidate_pivots_random(X_sub, self.subsequences_index)
        elif self.selection == 'all':
            self.shapelets = self._candidate_pivots_all(X_sub, self.subsequences_index)
        elif self.selection == 'cluster':
            self.shapelets = self._candidate_pivots_cluster(X_sub, self.subsequences_index)

        return self

    def _selectable_features_smart(self, X, y):
        """
        Subsets of features from the feature importances of a random forest: the most important ones
        (up to the knee of the sorted importances, or to their largest drop with 'drop').

        Parameters
        ----------
        X : numpy.ndarray of shape (n_samples, n_features)
        y : numpy.ndarray of shape (n_samples,)

        Returns
        -------
        subsequences_index : dict
            Number of features -> list of subsets of features.
        """
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

        subsequences_index = {}
        if self.use_combination:
            for n_features in range(1, len(best_features_idx)+1):
                subsequences_index[n_features] = list(combinations(best_features_idx, n_features))
        else:
            subsequences_index[len(best_features_idx)] = [best_features_idx]
        return subsequences_index

    def _selectable_features_iterative(self, X, y):
        """
        Subsets of features chosen iteratively: for each size, the subset whose partial pivots give the
        best depth-1 split, among the subsets of features not chosen yet.

        Parameters
        ----------
        X : numpy.ndarray of shape (n_samples, n_features)
        y : numpy.ndarray of shape (n_samples,)

        Returns
        -------
        subsequences_index : dict
            Number of features -> list with the chosen subset.
        """
        min_n_features, max_n_features = self._compute_n_features(X)

        available_features_idx = np.arange(0, X.shape[1])
        selected_feature_idxs = []
        filtered_subsequences_index = {}

        for n_feature in range(min_n_features, max_n_features):
            features_to_test = []
            candidate_subsequences = []
            for subsequence in combinations(list(range(X.shape[1])), n_feature):
                if all(i in available_features_idx for i in subsequence) and \
                    not any(i in selected_feature_idxs for i in subsequence):
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
            dt = DecisionTreeClassifier(max_depth=1, random_state=self.random_state)
            dt.fit(candidate_shapelets_it_transformed, y)
            selected_column = dt.tree_.feature[0]
            filtered_subsequences_index[n_feature] = [candidate_subsequences[selected_column//X.shape[0]]]
            selected_feature_idxs += list(filtered_subsequences_index[n_feature][0])

        return filtered_subsequences_index

    def _selectable_features(self, X:np.ndarray, y:np.ndarray):
        """
        Subsets of features of the pivots, according to n_features_strategy.

        Parameters
        ----------
        X : numpy.ndarray of shape (n_samples, n_features)
        y : numpy.ndarray of shape (n_samples,)

        Returns
        -------
        subsequences_index : dict
            Number of features -> list of subsets of features.
        """
        min_n_features, max_n_features = self._compute_n_features(X)


        if self.n_features_strategy in ['elbow', 'drop', 'new']:
            return self._selectable_features_smart(X, y)
        elif self.n_features_strategy == 'iterative':
            return self._selectable_features_iterative(X, y)
        else:
            subsequences_index = {}
            for n_features in range(min_n_features, max_n_features):
                subsequences_index[n_features] = list(combinations(list(range(X.shape[1])), n_features))
            return subsequences_index


    def _candidate_pivots_all(self, X: np.ndarray, subsequences_index: dict):
        """
        All the distinct partial pivots of the instances on the subsets of features.

        Parameters
        ----------
        X : numpy.ndarray of shape (n_samples, n_features)
        subsequences_index : dict
            Number of features -> list of subsets of features.

        Returns
        -------
        pivots : numpy.ndarray
        """
        res_data = [
            np.unique(np.where(np.isin(np.arange(X.shape[1]), combination), X, np.nan), axis=0)
            for feature_sets in subsequences_index.values()
            for combination in feature_sets
        ]

        if len(res_data) == 0:
            print()

        return np.vstack(res_data)

    def _subsample(self, X: np.ndarray, y: np.ndarray):
        """
        At most n_shapelets rows of X, stratified by y when possible.

        Parameters
        ----------
        X : numpy.ndarray
        y : array-like or None
            Strata.

        Returns
        -------
        X_sample : numpy.ndarray
        """
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
            except Exception:
                subsample_idx = np.random.RandomState(self.random_state).choice(
                    X.shape[0],
                    size=min(self.n_shapelets, X.shape[0]),
                    replace=False
                )

        return X[subsample_idx]

    def _candidate_pivots_random(self, X: np.ndarray, subsequences_index: dict):
        """
        Partial pivots selected at random for each subset of features, then subsampled stratifying by
        number of features.

        Parameters
        ----------
        X : numpy.ndarray of shape (n_samples, n_features)
        subsequences_index : dict
            Number of features -> list of subsets of features.

        Returns
        -------
        pivots : numpy.ndarray
        """
        res_data = []
        n_shapelet_for_stratification = []
        for n_features, feature_sets in subsequences_index.items():
            res_data_n_features = []
            for combination in feature_sets:
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
        """
        The rows of X closest to the centroids of k-means (nan replaced by 0), with at most n_shapelets
        clusters.

        Parameters
        ----------
        X : numpy.ndarray

        Returns
        -------
        pivots : numpy.ndarray
        """
        X_filled = np.nan_to_num(X, nan=0.0)
        km = KMeans(n_clusters=min(len(X_filled), self.n_shapelets), random_state=self.random_state)
        km.fit(X_filled)
        distances = cdist(km.cluster_centers_, X_filled, metric=self.distance)
        nearest_indices = np.argmin(distances, axis=1)

        return X[nearest_indices]

    def _candidate_pivots_cluster(self, X: np.ndarray, subsequences_index: dict):
        """
        Partial pivots selected with k-means for each subset of features, then again among all of them.

        Parameters
        ----------
        X : numpy.ndarray of shape (n_samples, n_features)
        subsequences_index : dict
            Number of features -> list of subsets of features.

        Returns
        -------
        pivots : numpy.ndarray
        """
        res_data = []
        for feature_sets in subsequences_index.values():
            for combination in feature_sets:
                res_data += [
                    self._run_kmeans_and_return_centroids(
                        np.unique(np.where(np.isin(np.arange(X.shape[1]), combination), X, np.nan), axis=0)
                    )
                ]

        X_all = np.vstack(res_data)

        return self._run_kmeans_and_return_centroids(X=X_all)

    def _compute_n_features(self, X):
        """
        Range of the number of features of the pivots, from n_features_strategy.

        Parameters
        ----------
        X : numpy.ndarray of shape (n_samples, n_features)

        Returns
        -------
        min_n_features : int
        max_n_features : int
            Excluded.
        """
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
        """
        Index of the knee of a decreasing curve: the point farthest from the segment between its ends.

        Parameters
        ----------
        values : numpy.ndarray

        Returns
        -------
        index : int
        """
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
        """
        Point of the line through a and b closest to p.

        Parameters
        ----------
        a, b, p : list of float
            Points (x, y).

        Returns
        -------
        point : list of float
        """
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
        """
        Distance of each instance from each partial pivot.

        Parameters
        ----------
        X : numpy.ndarray of shape (n_samples, n_features)
        y : None
            Not used.

        Returns
        -------
        distances : numpy.ndarray of shape (n_samples, n_shapelets)
        """
        return TabularShapelets._transform(X, y, self.n_jobs, self.shapelets, self.__get_distance())

    @classmethod
    def _transform(cls, X, y=None, n_jobs=1, shapelets=None, distance=euclidean, **transform_params):
        """
        Distance of each instance from each partial pivot, computed in parallel with numba.

        Parameters
        ----------
        X : numpy.ndarray of shape (n_samples, n_features)
        y : None
            Not used.
        n_jobs : int, default=1
            Number of threads.
        shapelets : numpy.ndarray
            Partial pivots.
        distance : callable, default=euclidean
            Distance function.
        **transform_params : dict
            Not used.

        Returns
        -------
        distances : numpy.ndarray of shape (n_samples, n_shapelets)
        """
        old_n_threads = numba.get_num_threads()
        numba.set_num_threads(n_jobs)
        dist_matrix = _compute_distance(X, shapelets, distance)
        numba.set_num_threads(old_n_threads)

        return dist_matrix

    def _optimize_memory(self, selected_shapelet_idx:np.ndarray):
        """
        Keep only some partial pivots.

        Parameters
        ----------
        selected_shapelet_idx : numpy.ndarray
            Indices of the pivots to keep.

        Returns
        -------
        done : bool
            Always True.
        """
        self.shapelets = self.shapelets[selected_shapelet_idx]
        return True


@jit(parallel=True)
def _compute_distance(X: np.ndarray, shapelets: np.ndarray, distance):
    """
    Distance of each row of X from each partial pivot, on the features of the pivot (numba).

    Parameters
    ----------
    X : numpy.ndarray of shape (n_samples, n_features)
    shapelets : numpy.ndarray of shape (n_shapelets, n_features)
    distance : callable
        Distance between a matrix and a vector.

    Returns
    -------
    distances : numpy.ndarray of shape (n_samples, n_shapelets)
    """
    res = np.ones((X.shape[0], shapelets.shape[0]), dtype=np.float32) * np.inf

    for idx, shapelet in enumerate(shapelets):
        cols = np.where(np.isfinite(shapelet))[0]
        res[:, idx] = distance(X[:, cols], shapelet[cols])

    return res

if __name__ == '__main__':
    iris = load_iris()
    df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    df['target'] = iris.target
    df['target'] = df['target'].map(dict(enumerate(iris.target_names)))

    X = df[iris.feature_names].values
    y = df['target'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    st = TabularShapelets(n_shapelets=5,
                          n_features_strategy='all', #tuple, float, int, 'elbow', 'drop', 'new', 'all'
                          n_jobs=10,
                          distance='euclidean',
                          selection='cluster' #random, cluster, all
                          )

    X_train_transform = st.fit_transform(X_train, y_train)
    X_test_transform = st.transform(X_test)

    print(X_train_transform.shape)

    rf = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42)
    rf.fit(X_train_transform, y_train)

    y_pred = rf.predict(X_test_transform)

    print(classification_report(y_test, y_pred))

    shapelets = pd.DataFrame(st.shapelets, columns=[df.drop(columns="target").columns])

    print(shapelets)
