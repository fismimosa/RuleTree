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
    """
    Shapelets transformer for time series classification.
    
    This class implements the shapelet transform method which transforms time series data
    into a feature vector of distances between the time series and a set of shapelets.
    Shapelets are discriminative subsequences extracted from the time series.
    
    The transformer follows scikit-learn's TransformerMixin interface (fit, transform).
    
    Parameters
    ----------
    n_shapelets : int, default=100
        Number of shapelets to be used in the transformation.
    
    n_shapelets_for_selection : int or str, default=np.inf
        Number of candidate shapelets to generate before selection.
        If 'stratified', generates candidates based on class distribution.
        If np.inf, generates all possible candidates.
    
    n_ts_for_selection_per_class : int, default=np.inf
        Number of time series to use per class for shapelet candidates.
    
    sliding_window : int, default=50
        Length of the shapelets to extract.
    
    selection : str, default='random'
        Method used to select the final shapelets from candidates:
        - 'random': random selection
        - 'mi_clf': mutual information (classification)
        - 'mi_reg': mutual information (regression)
        - 'cluster': KMedoids clustering
    
    distance : str or callable, default='euclidean'
        Distance metric to use for computing distances between shapelets and time series.
        Options: 'euclidean', 'sqeuclidean', 'cosine', 'cityblock', or custom callable.
    
    mi_n_neighbors : int, default=100
        Number of neighbors to use for mutual information calculation.
    
    random_state : int, default=42
        Random seed for reproducibility.
    
    n_jobs : int, default=1
        Number of parallel jobs. If -1, uses all available processors.
    """
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


    def fit(self, X, y=None):
        """
        Fit the Shapelets transformer on training data.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_signals, n_timepoints)
            Training time series data.
            Currently only supports n_signals=1 (univariate time series).
        
        y : array-like of shape (n_samples,), default=None
            Target values for supervised learning.
            
        Returns
        -------
        self : Shapelets
            Returns self.
            
        Raises
        ------
        NotImplementedError
            If multivariate time series are provided (n_signals > 1).
        """
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
        """
        Generate candidate shapelets from the time series data.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_signals, n_timepoints)
            Time series data.
            
        y : array-like of shape (n_samples,), default=None
            Target values.
            
        Returns
        -------
        array-like of shape (n_candidates, n_signals, shapelet_length)
            Candidate shapelets extracted from the time series data.
        """
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
        """
        Randomly select shapelets from candidates.
        
        Parameters
        ----------
        candidate_shapelets : array-like of shape (n_candidates, n_signals, shapelet_length)
            Candidate shapelets.
        
        X : array-like of shape (n_samples, n_signals, n_timepoints)
            Time series data (unused in random selection).
            
        y : array-like of shape (n_samples,)
            Target values (unused in random selection).
            
        Returns
        -------
        array-like of shape (n_shapelets, n_signals, shapelet_length)
            Randomly selected shapelets.
        """
        n_shapelets = min(self.n_shapelets, candidate_shapelets.shape[0])
        return candidate_shapelets[np.random.choice(candidate_shapelets.shape[0], size=n_shapelets, replace=False)]

    def __fit_selection_mutual_info(self, candidate_shapelets: np.ndarray, X, y, mutual_info_fun):
        """
        Select shapelets using mutual information criteria.
        
        Parameters
        ----------
        candidate_shapelets : array-like of shape (n_candidates, n_signals, shapelet_length)
            Candidate shapelets.
            
        X : array-like of shape (n_samples, n_signals, n_timepoints)
            Time series data.
            
        y : array-like of shape (n_samples,)
            Target values.
            
        mutual_info_fun : callable
            Function to compute mutual information (either mutual_info_classif or mutual_info_regression).
            
        Returns
        -------
        array-like of shape (n_shapelets, n_signals, shapelet_length)
            Shapelets selected based on mutual information.
            
        Raises
        ------
        UnsupportedError
            If y is None (unsupervised tasks).
        """
        if y is None:
            raise UnsupportedError("Mutual information is not suitable for unsupervised tasks.")

        idx_to_test = resample(range(X.shape[0]), stratify=y, random_state=self.random_state)

        old_n_threads = numba.get_num_threads()
        numba.set_num_threads(self.n_jobs)
        dist = _best_fit(X[idx_to_test], candidate_shapelets, self.__get_distance())
        numba.set_num_threads(old_n_threads)

        labels, labels_count = np.unique(y, return_counts=True)
        if np.sum(labels_count) == len(labels):
            scores = np.zeros((len(labels), ))
        else:
            scores = mutual_info_fun(dist, y,
                                     n_jobs=self.n_jobs,
                                     n_neighbors=min(dist.shape[0], self.mi_n_neighbors),
                                     discrete_features=False)
        if len(candidate_shapelets) == self.n_shapelets:
            return candidate_shapelets
        return candidate_shapelets[np.argpartition(scores, -min(scores.shape[0], self.n_shapelets))\
            [-min(scores.shape[0], self.n_shapelets):]]

    def __fit_selection_cluster(self, candidate_shapelets, X, y):
        """
        Select shapelets using KMedoids clustering.
        
        Parameters
        ----------
        candidate_shapelets : array-like of shape (n_candidates, n_signals, shapelet_length)
            Candidate shapelets.
            
        X : array-like of shape (n_samples, n_signals, n_timepoints)
            Time series data (unused in cluster selection).
            
        y : array-like of shape (n_samples,)
            Target values (unused in cluster selection).
            
        Returns
        -------
        array-like of shape (n_shapelets, n_signals, shapelet_length)
            Shapelets selected as medoids from clustering.
            
        Raises
        ------
        Exception
            If scikit-learn-extra is not installed.
        """
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

    def transform(self, X, y=None):
        """
        Transform time series data into shapelet distance features.
        
        For each time series in X, computes the minimum distance to each shapelet,
        resulting in a feature vector of length n_shapelets.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_signals, n_timepoints)
            Time series data to transform.
            
        y : array-like of shape (n_samples,), default=None
            Target values (unused in transformation).
            
        Returns
        -------
        array-like of shape (n_samples, n_shapelets)
            Transformed data, where each feature is the minimum distance between
            a time series and a shapelet.
        """
        old_n_threads = numba.get_num_threads()
        numba.set_num_threads(self.n_jobs)
        dist_matrix = _best_fit(X, self.shapelets, self.__get_distance())
        numba.set_num_threads(old_n_threads)

        return dist_matrix

@jit(parallel=True)
def _best_fit(timeseries: np.ndarray, shapelets: np.ndarray, distance):
    """
    Compute the minimum distance between time series and shapelets.
    
    For each time series and shapelet, computes the minimum distance between
    the shapelet and all subsequences of the time series.
    
    Parameters
    ----------
    timeseries : array-like of shape (n_samples, n_signals, n_timepoints)
        Time series data.
        
    shapelets : array-like of shape (n_shapelets, n_signals, shapelet_length)
        Shapelets.
        
    distance : callable
        Distance function to use.
        
    Returns
    -------
    array-like of shape (n_samples, n_shapelets)
        Matrix of minimum distances between time series and shapelets.
    """
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
    df_train = pd.DataFrame(scipy.io.arff.loadarff('test_dataset/CBF/CBF_TRAIN.arff')[0])
    df_test = pd.DataFrame(scipy.io.arff.loadarff('test_dataset/CBF/CBF_TEST.arff')[0])
    df_train.target = df_train.target.astype(int)
    df_test.target = df_test.target.astype(int)

    X_train = df_train.drop(columns=['target']).to_numpy().reshape((-1, 1, 128))
    X_test = df_test.drop(columns=['target']).to_numpy().reshape((-1, 1, 128))
    y_train = df_train['target'].values
    y_test = df_test['target'].values

    st = Shapelets(n_shapelets=10, selection='random', mi_n_neighbors=100, n_jobs=-1, distance='cityblock') #euclidean, sqeuclidean, cosine, cityblock

    X_train_transform = st.fit_transform(X_train, y_train)
    X_test_transform = st.transform(X_test)

    rf = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42)
    rf.fit(X_train_transform, y_train)

    y_pred = rf.predict(X_test_transform)

    print(classification_report(y_test, y_pred))

