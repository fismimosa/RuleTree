"""PivotTransformer: distances from candidate pivots, and the cache of the pivot stumps."""

import hashlib

import numpy as np
import sklearn
from scipy import sparse
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.preprocessing import StandardScaler

from RuleTree.base.RuleTreeBaseStump import estimator_namespace
from RuleTree.base.RuleTreeBaseTransformer import RuleTreeBaseTransformer, get_feature_types
from RuleTree.exceptions import NoSplitFoundWarning
from RuleTree.stumps.transformers._pivot_selection import PivotSelection
from RuleTree.utils.define import DATA_TYPE_IMAGE, DATA_TYPE_TABULAR, DATA_TYPE_TEXT, DATA_TYPE_TS, MODEL_TYPE_CLF


class PivotTransformer(RuleTreeBaseTransformer):
    """
    Transformer into the distances from candidate pivots, instances of the node that describe a
    class (medoids) or separate the classes.

    Time series and images are flattened into one feature per value, so a tree can have pivot stumps
    on each of its data types (e.g. with ``base_stumps`` as tuples with the data types of each stump).
    The candidates are selected among the instances of the node. The distances are computed on the
    numerical features and on the one-hot encoding of the categorical ones, all standardized (z-score)
    if ``standardize``;
    the encoding, the scaler and the distance matrix of the whole data set are computed once and cached
    in the tree. After the splitter has chosen a column, only the chosen pivot is kept (``select``).

    Parameters
    ----------
    ml_task : {MODEL_TYPE_CLF, MODEL_TYPE_REG}, default=MODEL_TYPE_CLF
        Task of the stump, which determines how the pivots are selected.
    distance_measure : str or callable, default='euclidean'
        Distance between the instances, computed on the standardized features (numerical and one-hot
        encoded categorical): any metric of ``sklearn.metrics.pairwise_distances``.
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
    standardize : bool, default=True
        Whether the features (or the embeddings) are standardized (z-score) before the distances, so that
        each of them weighs the same.
    embedder : transformer, default=None
        Transformer of the data into vectors (e.g. ``RuleTree.utils.embeddings.ImageEmbedder``): the
        distances (``distance_measure``) are computed on the standardized embeddings; with it, also texts
        are supported. It must be pretrained or stateless: it is
        fitted without data (``fit(None)``), also when a saved tree is loaded. The embeddings of the whole
        data set are computed once and cached in the tree, and the ones of new data once per instance.
        If None, time series and images are flattened.

    Attributes
    ----------
    numerical_ : list of int
        Numerical columns of the feature matrix.
    categorical_ : list of int
        Categorical columns of the feature matrix.
    candidates_ : numpy.ndarray of shape (n_candidates, n_encoded_features)
        The candidate pivots (only the chosen one after ``select``), encoded and standardized.
    candidate_names_ : numpy.ndarray of shape (n_candidates,)
        Indices of the candidate pivots in the training set.
    candidate_labels_ : numpy.ndarray of shape (n_candidates,)
        Target of the candidate pivots (not saved by get_state).
    pivots_ : dict
        Indices in the training set of the ``descriptives``, ``discriminatives`` and ``candidates``
        pivots of the node (all of them, also after ``select``).
    categories_ : list of list
        Values of each categorical feature in the whole data set, the columns of its one-hot encoding;
        a value not seen in training is encoded as all zeros.
    scaler_ : StandardScaler or None
        Scaler of the encoded features, fitted on the whole data set; None if not ``standardize``.
    media_ : dict
        Index in the training set -> the training instance as given to ``fit`` (e.g. an image), for the
        pivots of the split, drawn by ``plot_tree(show_media=True)``. Set by ``select``.

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
    >>> transformer = PivotTransformer().fit(X, y, context=None, node_id=None, cache=None)
    >>> transformer.transform(X[:2]).shape  # distances from 6 candidate pivots
    (2, 6)
    >>> model = RuleTreeClassifier(max_depth=2, base_stumps=Stump(PivotTransformer(), ClassificationSplitter()))
    >>> model = model.fit(X, y)
    """

    def __init__(self, ml_task=MODEL_TYPE_CLF, distance_measure='euclidean', k=None, reg_contrast_lambda=.2,
                 reg_max_thresholds=None, tree_params=None, standardize=True, embedder=None):
        """Initialize the PivotTransformer; see the class documentation for the parameters."""
        self.ml_task = ml_task
        self.distance_measure = distance_measure
        self.k = k
        self.reg_contrast_lambda = reg_contrast_lambda
        self.reg_max_thresholds = reg_max_thresholds
        self.tree_params = tree_params
        self.standardize = standardize
        self.embedder = embedder

    @property
    def supported_data_types(self):
        """Tabular data, time series and images (flattened, or through the embedder); texts with an embedder."""
        flattened = (DATA_TYPE_TABULAR, DATA_TYPE_TS, DATA_TYPE_IMAGE)
        return flattened + ((DATA_TYPE_TEXT,) if self.embedder is not None else ())

    def fit(self, X, y, idx=None, data_type=None, sample_weight=None, *, context, node_id, cache):
        """
        Select the candidate pivots of the node (see PivotSelection), on the standardized features. The
        encoding, the scaler and the distance matrix of the whole data set are taken from the cache.

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
        sample_weight : numpy.ndarray, default=None
            Sample weights, all the instances.
        context : RuleTree or None
            Tree being built, None outside a tree.
        node_id : str or None
            Identifier of the node, e.g. ``"Rl"`` (``context.get_node_by_id(node_id)``), None outside a tree.
        cache : dict or None
            Cache of the tree (namespace -> dictionary of values), None outside a tree: the component
            decides whether and how to use it (the ones of the library use ``self._get_cache(cache)``).

        Returns
        -------
        self : PivotTransformer

        Raises
        ------
        NoSplitFoundWarning
            If the data have no feature.
        """
        if idx is None:
            idx = np.arange(len(X))
        original = X
        # key of the cached values: also the embedder, by class and hyperparameters
        key = data_type if self.embedder is None else f"{data_type}, {estimator_namespace(self.embedder)}"
        X = self._vectors(X, key, cache)
        self.num_pre_transformed_, self.cat_pre_transformed_ = get_feature_types(X, self._get_cache(cache), key)

        if len(self.num_pre_transformed_) + len(self.cat_pre_transformed_) == 0:
            raise NoSplitFoundWarning(f"{self.__class__.__name__} needs at least one feature")

        self.categories_, self.scaler_, distance_matrix = _pivot_cache(
            X, self.num_pre_transformed_, self.cat_pre_transformed_, self.distance_measure, self.standardize, key,
            self._get_cache(cache))
        X_scaled = self._encode(X[idx])

        selection = self._selection()
        selection.fit(X_scaled, y[idx], distance_matrix[np.ix_(idx, idx)], idx,
                      sample_weight=None if sample_weight is None else sample_weight[idx])

        self.candidates_ = selection.candidates_
        self.candidate_names_ = selection.candidate_names_
        self.candidate_labels_ = selection.candidate_labels_
        self.pivots_ = {"descriptives": selection.descriptive_names_.tolist(),
                        "discriminatives": selection.discriminative_names_.tolist(),
                        "candidates": selection.candidate_names_.tolist()}
        self.candidate_media_ = {int(name): _instance(original, name) for name in self.candidate_names_}
        self.numerical_, self.categorical_ = list(range(len(self.candidates_))), []

        return self

    def transform(self, X, context=None, node_id=None, cache=None):
        """
        Distances of the instances from the candidate pivots (from the chosen pivot, after select).

        Parameters
        ----------
        X : numpy.ndarray
            Data of the transformer's type (time series and images are flattened).
        context : RuleTree, default=None
            The tree, None outside a tree.
        node_id : str, default=None
            Identifier of the node of the stump, None outside a tree.
        cache : dict, default=None
            Cache of the tree, when called by it; None otherwise.

        Returns
        -------
        X_transformed : numpy.ndarray of shape (n_samples, n_candidates)
        """
        return pairwise_distances(self._encode(self._new_vectors(X, cache)), self.candidates_,
                                  metric=self.distance_measure)

    def _encode(self, X):
        """
        Features of vectors, standardized if ``standardize``: the numerical ones and the one-hot encoding of the
        categorical ones.

        Parameters
        ----------
        X : numpy.ndarray of shape (n_samples, n_features)
            Vectors (see :meth:`_vectors`).

        Returns
        -------
        X_encoded : numpy.ndarray of shape (n_samples, n_encoded_features)
        """
        X_encoded = _one_hot(X, self.num_pre_transformed_, self.cat_pre_transformed_, self.categories_)
        return X_encoded if self.scaler_ is None else self.scaler_.transform(X_encoded)

    def _embedder(self):
        """The embedder, fitted without data (it is pretrained or stateless)."""
        if not hasattr(self, "embedder_"):
            self.embedder_ = sklearn.clone(self.embedder).fit(None)
        return self.embedder_

    def _vectors(self, X, key, cache):
        """
        Vectors of the whole data set: the embeddings, computed once and cached in the tree, or the data
        (time series and images flattened).

        Parameters
        ----------
        X : numpy.ndarray
            Data of the transformer's type, all the instances.
        key : str
            Data type and embedder, the key of the cached embeddings.
        cache : dict or None
            Cache of the tree.

        Returns
        -------
        X_vectors : numpy.ndarray of shape (n_samples, n_features)
        """
        if self.embedder is None:
            return _flatten(X)

        cache = self._get_cache(cache)
        if ("pivot embeddings", key) not in cache:
            embeddings = _dense(self._embedder().transform(X))
            cache[("pivot embeddings", key)] = embeddings
            # the embedding of each training instance, for the instances of the nodes given to transform
            self._embeddings(cache).update(zip((_instance_key(instance) for instance in X), embeddings))
        return cache[("pivot embeddings", key)]

    def _embeddings(self, cache):
        """Instance (its key) -> embedding: in the cache of the tree if there is one, else in the transformer."""
        if cache:
            return cache.setdefault("pivot embeddings by instance", {})
        return self.__dict__.setdefault("_embeddings", {})  # not a parameter nor a fitted attribute

    def _new_vectors(self, X, cache=None):
        """
        Vectors of new data: the data (time series and images flattened), or their embeddings, computed once
        for each distinct instance (a tree applies its stumps to the instances of each node).

        Parameters
        ----------
        X : numpy.ndarray
            Data of the transformer's type.
        cache : dict, default=None
            Cache of the tree, where the embeddings of the training instances are.

        Returns
        -------
        X_vectors : numpy.ndarray of shape (n_samples, n_features)
        """
        if self.embedder is None:
            return _flatten(X)

        embeddings = self._embeddings(self._get_cache(cache) if cache is not None else None)
        keys = [_instance_key(instance) for instance in X]
        first = {}  # the first position of each instance not embedded yet
        for i, key in enumerate(keys):
            if key not in embeddings:
                first.setdefault(key, i)
        missing = list(first.values())
        if missing:
            new = _dense(self._embedder().transform(np.asarray(X)[missing]))
            embeddings.update({keys[i]: vector for i, vector in zip(missing, new)})
        return np.array([embeddings[key] for key in keys])

    def _keep_media(self, names):
        """
        Keep the original training instances of the pivots of the split (``media_``), for ``plot_tree``.

        Parameters
        ----------
        names : list of int
            Indices of the pivots of the split in the training set.
        """
        candidate_media = self.__dict__.pop("candidate_media_", {})
        self.media_ = {int(name): candidate_media[int(name)] for name in names if int(name) in candidate_media}

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
        self._keep_media(self.candidate_names_)
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
        Fitted state: the pivot, its index, the candidate pivots, the features, the categories and the scaler.

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
            'categories': categories_state(self.categories_),
            'scaler': scaler_state(self.scaler_),
            'media': media_state(self),
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
        self.categories_ = state.get('categories', [])  # saved before the categorical features: not used
        self.scaler_ = scaler_from_state(state['scaler'])
        self.media_ = media_from_state(state)
        return self

    def _selection(self):
        """
        A new PivotSelection with the parameters of the transformer.

        Returns
        -------
        selection : PivotSelection
        """
        return PivotSelection(self.ml_task, k=self.k, reg_contrast_lambda=self.reg_contrast_lambda,
                              reg_max_thresholds=self.reg_max_thresholds, **(self.tree_params or {}))


def scaler_state(scaler: StandardScaler) -> dict:
    """
    JSON-friendly state of a fitted StandardScaler, restored by scaler_from_state.

    Parameters
    ----------
    scaler : StandardScaler or None
        Fitted scaler, None without standardization.

    Returns
    -------
    state : dict or None
    """
    if scaler is None:
        return None
    return {
        'mean_': scaler.mean_.tolist(),
        'scale_': scaler.scale_.tolist(),
        'var_': scaler.var_.tolist(),
        'n_samples_seen_': scaler.n_samples_seen_
    }


def scaler_from_state(state: dict) -> StandardScaler:
    """
    Fitted StandardScaler from the state created by scaler_state.

    Parameters
    ----------
    state : dict or None
        State of the scaler.

    Returns
    -------
    scaler : StandardScaler
    """
    if state is None:  # without standardization
        return None
    scaler = StandardScaler()
    scaler.mean_ = np.array(state['mean_'])
    scaler.scale_ = np.array(state['scale_'])
    scaler.var_ = np.array(state['var_'])
    scaler.n_samples_seen_ = state['n_samples_seen_']
    return scaler


def media_state(transformer) -> dict:
    """
    JSON-friendly original training instances of the pivots of a split (``media_``), restored by
    media_from_state.
    """
    return {str(name): value if isinstance(value, str) else np.asarray(value).tolist()
            for name, value in getattr(transformer, "media_", {}).items()}


def media_from_state(state: dict) -> dict:
    """The original training instances of the pivots of a split, from a state with ``media``."""
    return {int(name): value if isinstance(value, str) else np.asarray(value)
            for name, value in state.get("media", {}).items()}  # none in the trees saved before


def _instance(X, index):
    """An instance of the data as given to fit: a string for a text, an array otherwise."""
    instance = X[int(index)]
    return str(instance) if isinstance(instance, str) else np.array(instance)


def _instance_key(instance):
    """A short key of an instance (a hash of its content), for the cache of its embedding."""
    if isinstance(instance, str):
        content = instance.encode()
    else:
        instance = np.ascontiguousarray(instance)
        content = repr((instance.shape, instance.dtype.str)).encode() + instance.tobytes()
    return hashlib.blake2b(content, digest_size=16).digest()


def _dense(X) -> np.ndarray:
    """The embeddings as a dense 2D array (e.g. from the sparse output of a vectorizer of texts)."""
    X = X.toarray() if sparse.issparse(X) else np.asarray(X, dtype=float)
    return X.reshape(len(X), -1)


def _flatten(X) -> np.ndarray:
    """Tabular data as they are; time series and images flattened into one feature per value."""
    X = np.asarray(X)
    return X if X.ndim <= 2 else X.reshape(X.shape[0], -1)


def categories_state(categories: list) -> list:
    """
    JSON-friendly values of the categorical features (Python types instead of NumPy ones).

    Parameters
    ----------
    categories : list of array-like
        Values of each categorical feature.

    Returns
    -------
    state : list of list
    """
    return [np.asarray(values, dtype=object).tolist() for values in categories]


def _one_hot(X, numerical, categorical, categories) -> np.ndarray:
    """
    Numerical features of X followed by the one-hot encoding of the categorical ones.

    Parameters
    ----------
    X : numpy.ndarray of shape (n_samples, n_features)
        Tabular data.
    numerical : list of int
        Numerical columns of X.
    categorical : list of int
        Categorical columns of X.
    categories : list of list
        Values of each categorical column; a value not in the list is encoded as all zeros. Fewer lists
        than categorical columns (a tree saved before the categorical features were used): the
        columns without values are not encoded.

    Returns
    -------
    X_encoded : numpy.ndarray of shape (n_samples, n_numerical + total number of categories)
    """
    columns = [np.asarray(X[:, numerical], dtype=float)]
    columns += [(X[:, feature, None] == np.asarray(values, dtype=object)).astype(float)
                for feature, values in zip(categorical, categories)]
    return np.hstack(columns)


def _pivot_cache(X, numerical, categorical, distance_measure, standardize, data_type, cache):
    """
    Values of the categorical features, scaler of the encoded features (None if not standardize) and
    distance matrix of the whole data set X, encoded and standardized, stored in the cache of the
    transformer so that they are computed once per data type, distance and standardization.
    """
    cache = {} if cache is None else cache
    key = ("pivot", data_type, distance_measure, standardize)
    if key not in cache:
        categories = categories_state([np.unique(X[:, feature]) for feature in categorical])
        X_encoded = _one_hot(X, numerical, categorical, categories)
        scaler = StandardScaler().fit(X_encoded) if standardize else None
        # the same scale as the distances from the pivots in transform
        X_scaled = X_encoded if scaler is None else scaler.transform(X_encoded)
        cache[key] = (categories, scaler, pairwise_distances(X_scaled, metric=distance_measure))

    return cache[key]
