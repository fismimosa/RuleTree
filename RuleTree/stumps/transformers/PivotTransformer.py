"""PivotTransformer: distances from candidate pivots, and the cache of the pivot stumps."""

import numpy as np
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.preprocessing import StandardScaler

from RuleTree.base.RuleTreeBaseTransformer import RuleTreeBaseTransformer, get_feature_types
from RuleTree.exceptions import NoSplitFoundWarning
from RuleTree.stumps.transformers._pivot_selection import PivotSelection
from RuleTree.utils.define import MODEL_TYPE_CLF


class PivotTransformer(RuleTreeBaseTransformer):
    """
    Transformer into the distances from candidate pivots, instances of the node that describe a
    class (medoids) or separate the classes.

    The candidates are selected among the instances of the node, on the standardized numerical
    features; the scaler and the distance matrix of the whole data set are computed once and cached
    in the tree. After the splitter has chosen a column, only the chosen pivot is kept (``select``).

    Parameters
    ----------
    ml_task : {MODEL_TYPE_CLF, MODEL_TYPE_REG}, default=MODEL_TYPE_CLF
        Task of the stump, which determines how the pivots are selected.
    distance_measure : str or callable, default='euclidean'
        Distance between the instances, computed on the standardized numerical features: any metric
        of ``sklearn.metrics.pairwise_distances``.
    k : int, default=None
        Regression only: number of nearest and farthest instances used to measure how well a
        candidate pivot contrasts close and distant instances. None for the square root of the
        number of instances of the node (between 5 and n - 1).
    reg_contrast_lambda : float, default=0.2
        Regression only: weight of the contrast with respect to the gain of the split when the
        descriptive pivot is selected.
    reg_max_thresholds : int, default=None
        Regression only: maximum number of thresholds tested on the distances from each candidate
        pivot (quantiles). None for ``max(256, sqrt(n))``.
    tree_params : dict, default=None
        Parameters of the decision trees that select the discriminative pivots.

    Attributes
    ----------
    numerical_ : list of int
        Numerical columns of the feature matrix.
    categorical_ : list of int
        Categorical columns of the feature matrix.
    candidates_ : numpy.ndarray of shape (n_candidates, n_numerical_features)
        The candidate pivots (only the chosen one after ``select``), standardized.
    candidate_names_ : numpy.ndarray of shape (n_candidates,)
        Indices of the candidate pivots in the training set.
    pivots_ : dict
        Indices in the training set of the ``descriptives``, ``discriminatives`` and ``candidates``
        pivots of the node (all of them, also after ``select``).
    scaler_ : StandardScaler
        Scaler of the numerical features, fitted on the whole data set.

    See Also
    --------
    PivotTreeStumpClassifier, PivotTreeStumpRegressor : the stumps that use it.

    Examples
    --------
    >>> from sklearn.datasets import load_iris
    >>> from RuleTree import RuleTreeClassifier
    >>> from RuleTree.stumps.Stump import Stump
    >>> from RuleTree.stumps.splitters import ClassificationSplitter
    >>> from RuleTree.stumps.transformers import PivotTransformer
    >>> X, y = load_iris(return_X_y=True)
    >>> transformer = PivotTransformer().fit(X, y)
    >>> transformer.transform(X[:2]).shape  # distances from 6 candidate pivots
    (2, 6)
    >>> model = RuleTreeClassifier(max_depth=2, base_stumps=Stump(PivotTransformer(), ClassificationSplitter()))
    >>> model = model.fit(X, y)
    """

    def __init__(self, ml_task=MODEL_TYPE_CLF, distance_measure='euclidean', k=None, reg_contrast_lambda=.2,
                 reg_max_thresholds=None, tree_params=None):
        """Initialize the PivotTransformer; see the class documentation for the parameters."""
        self.ml_task = ml_task
        self.distance_measure = distance_measure
        self.k = k
        self.reg_contrast_lambda = reg_contrast_lambda
        self.reg_max_thresholds = reg_max_thresholds
        self.tree_params = tree_params

    def fit(self, X, y, idx=None, data_type=None, cache=None, sample_weight=None):
        """
        Select the candidate pivots of the node (see PivotSelection), on the standardized numerical features. The
        scaler and the distance matrix of the whole data set are taken from the cache.

        Parameters
        ----------
        X : numpy.ndarray
            Data of the transformer's type, all the instances.
        y : numpy.ndarray
            Target, all the instances.
        idx : numpy.ndarray, default=None
            Indices of the instances in the node. If None, all the instances.
        data_type : str, default=None
            Type of X, used as key of the cached values.
        cache : dict, default=None
            Values shared among the splits of the tree.
        sample_weight : numpy.ndarray, default=None
            Sample weights, all the instances.

        Returns
        -------
        self : PivotTransformer

        Raises
        ------
        NoSplitFoundWarning
            If the data have no numerical feature.
        """
        if idx is None:
            idx = np.arange(len(X))
        self.num_pre_transformed_, self.cat_pre_transformed_ = get_feature_types(X, cache, data_type)

        if len(self.num_pre_transformed_) == 0:
            raise NoSplitFoundWarning(f"{self.__class__.__name__} needs numerical features")

        self.scaler_, distance_matrix = _pivot_cache(X, self.num_pre_transformed_, self.distance_measure, data_type,
                                                     cache)
        X_scaled = self.scaler_.transform(X[idx][:, self.num_pre_transformed_])

        selection = PivotSelection(self.ml_task, k=self.k, reg_contrast_lambda=self.reg_contrast_lambda,
                                   reg_max_thresholds=self.reg_max_thresholds, **(self.tree_params or {}))
        selection.fit(X_scaled, y[idx], distance_matrix[np.ix_(idx, idx)], idx,
                      sample_weight=None if sample_weight is None else sample_weight[idx])

        self.candidates_ = selection.candidates_
        self.candidate_names_ = selection.candidate_names_
        self.pivots_ = {"descriptives": selection.descriptive_names_.tolist(),
                        "discriminatives": selection.discriminative_names_.tolist(),
                        "candidates": selection.candidate_names_.tolist()}
        self.numerical_, self.categorical_ = list(range(len(self.candidates_))), []

        return self

    def transform(self, X):
        """
        Distances of the instances from the candidate pivots (from the chosen pivot, after select).

        Parameters
        ----------
        X : numpy.ndarray
            Tabular data.

        Returns
        -------
        X_transformed : numpy.ndarray of shape (n_samples, n_candidates)
        """
        return pairwise_distances(self.scaler_.transform(X[:, self.num_pre_transformed_]), self.candidates_,
                                  metric=self.distance_measure)

    def select(self, feature):
        """
        Keep only the chosen pivot.

        Parameters
        ----------
        feature : int
            Column chosen by the splitter.

        Returns
        -------
        feature : int
            Always 0: the distance from the only pivot kept.
        """
        self.candidates_ = self.candidates_[[feature]]
        self.candidate_names_ = self.candidate_names_[[feature]]
        self.numerical_, self.categorical_ = [0], []
        return 0

    def feature_id(self, feature):
        """
        Identifier of a candidate pivot: the index of the instance in the training set.

        Parameters
        ----------
        feature : int
            Column of the feature matrix.

        Returns
        -------
        feature_id : str
        """
        return f'{self.candidate_names_[feature]}'

    def get_state(self):
        """
        Fitted state: the pivot, its index, the candidate pivots, the numerical features and the scaler.

        Returns
        -------
        state : dict
            JSON-friendly dictionary, restored by set_state.
        """
        return {
            'num_pre_transformed': list(self.num_pre_transformed_),
            'cat_pre_transformed': list(self.cat_pre_transformed_),
            'X_split_instance': self.candidates_[0],
            'pivot_name': self.feature_id(0),
            'pivots': self.pivots_,
            'scaler': {
                'mean_': self.scaler_.mean_.tolist(),
                'scale_': self.scaler_.scale_.tolist(),
                'var_': self.scaler_.var_.tolist(),
                'n_samples_seen_': self.scaler_.n_samples_seen_
            },
        }

    def set_state(self, state):
        """
        Restore the fitted state saved by get_state.

        Parameters
        ----------
        state : dict
            Dictionary created by get_state.

        Returns
        -------
        self : PivotTransformer
        """
        self.num_pre_transformed_ = state['num_pre_transformed']
        self.cat_pre_transformed_ = state['cat_pre_transformed']
        self.candidates_ = np.array([state['X_split_instance']])
        self.candidate_names_ = np.array([int(state['pivot_name'])])
        self.pivots_ = state['pivots']

        self.scaler_ = StandardScaler()
        self.scaler_.mean_ = np.array(state['scaler']['mean_'])
        self.scaler_.scale_ = np.array(state['scaler']['scale_'])
        self.scaler_.var_ = np.array(state['scaler']['var_'])
        self.scaler_.n_samples_seen_ = state['scaler']['n_samples_seen_']

        return self


def _pivot_cache(X, numerical, distance_measure, data_type, cache):
    """
    Scaler and distance matrix of the whole data set X, stored in the cache of the stump so that
    they are computed once per data type and distance.
    """
    cache = {} if cache is None else cache
    key = ("pivot", data_type, distance_measure)
    if key not in cache:
        scaler = StandardScaler().fit(X[:, numerical])
        # TODO: la matrice delle distanze è calcolata sui dati NON standardizzati, mentre transform
        #  (e quindi lo split) calcola le distanze dai pivot sui dati standardizzati con lo scaler.
        #  I pivot candidati vengono quindi scelti con una scala e la soglia dello split con un'altra:
        #  con feature su scale molto diverse le due distanze possono non essere coerenti.
        #  Una possibile correzione è calcolare anche questa matrice su scaler.transform(X[:, numerical]),
        #  ma cambierebbe i pivot scelti e quindi i risultati degli alberi con gli stump Pivot.
        cache[key] = (scaler, pairwise_distances(X[:, numerical], metric=distance_measure))

    return cache[key]
