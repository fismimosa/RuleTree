"""XGBoostSplitter: the best split according to the XGBoost gain."""

import numpy as np

from RuleTree.base.RuleTreeBaseSplitter import RuleTreeBaseSplitter
from RuleTree.exceptions import NoSplitFoundWarning
from RuleTree.utils.define import MODEL_TYPE_REG
from RuleTree.utils.tree_utils import xgboost_cover_regression, xgboost_similarity_score_regression


class XGBoostSplitter(RuleTreeBaseSplitter):
    """
    Splitter for regression that maximizes the XGBoost gain: the similarity scores of the two
    children minus the one of the node, where the target are the residuals of the previous trees.

    Parameters
    ----------
    lam : float
        L2 regularization of the leaf values: the similarity score of a node is
        ``sum(residuals) ** 2 / (n + lam)``.
    approximation : {None, 'quantile'}, default=None
        With 'quantile', only ``n_quantiles`` thresholds (quantiles) are tested on each numerical
        feature, instead of all the values.
    n_quantiles : int, default=32
        Number of thresholds tested with approximation='quantile'.
    min_cover : float, default=1
        Minimum cover (number of instances) of each child.
    n_jobs : int, default=1
        Not used.

    Attributes
    ----------
    feature_ : int
        Column of the feature matrix used by the split.
    threshold_ : float
        Threshold of the split.
    is_categorical_ : bool
        Whether the split is categorical.
    gain_ : float
        XGBoost gain of the split.
    impurity_ : list of float or None
        Not computed (None).
    root_similarity, similarity_l, similarity_r : float
        Similarity scores of the node and of the two children.

    See Also
    --------
    XGBoostStumpRegressor : the stump made of a MatrixTransformer and this splitter.

    Examples
    --------
    >>> from sklearn.datasets import load_diabetes
    >>> from RuleTree.stumps.splitters import XGBoostSplitter
    >>> X, y = load_diabetes(return_X_y=True)
    >>> splitter = XGBoostSplitter(lam=1).fit(X, y - y.mean(), numerical=list(range(10)), categorical=[])
    >>> splitter.feature_
    8
    """

    task = MODEL_TYPE_REG

    # also the similarity scores and the gain: gain is used by the gamma pruning of RuleTreeXGBoostRegressor
    _state = RuleTreeBaseSplitter._state + ('gain', 'root_similarity', 'similarity_l', 'similarity_r')

    def __init__(self, lam, approximation=None, n_quantiles=32, min_cover=1, n_jobs=1):
        """Initialize the XGBoostSplitter; see the class documentation for the parameters."""
        if approximation is not None and approximation not in ['quantile']:
            raise ValueError('approximation must be None or quantile')

        self.lam = lam
        self.approximation = approximation
        self.n_quantiles = n_quantiles
        self.min_cover = min_cover
        self.n_jobs = n_jobs

    def _get_cover_similarity_fun(self):
        """
        Cover and similarity functions of regression.

        Returns
        -------
        cover_fn : callable
            Residuals -> cover (number of instances).
        similarity_fn : callable
            (residuals, cover, lambda) -> similarity score.
        """
        return xgboost_cover_regression, xgboost_similarity_score_regression

    def get_threshold_values(self, X, feature_idx, categorical):
        """
        Values tested for a feature: its values (categorical), the midpoints between its consecutive values
        (numerical) or, with approximation='quantile', n_quantiles quantiles.

        Parameters
        ----------
        X : numpy.ndarray of shape (n_samples, n_features)
            Feature matrix.
        feature_idx : int
            Column of the feature.
        categorical : list of int
            Indices of the categorical columns.

        Returns
        -------
        thresholds : numpy.ndarray
        """
        unique_val = np.unique(X[:, feature_idx])
        if feature_idx in categorical:
            return unique_val
        else:
            if self.approximation is None or len(unique_val) <= self.n_quantiles:
                values = np.sort(unique_val)
                return (values[:-1] + values[1:]) / 2
            else:
                return np.quantile(X[:, feature_idx], np.linspace(0, 1, self.n_quantiles))

    def fit(self, X, y, numerical, categorical, sample_weight=None, check_input=True):
        """
        Find the split with the highest XGBoost gain among all the features and thresholds.

        Parameters
        ----------
        X : numpy.ndarray of shape (n_samples, n_features)
            Feature matrix of the instances in the node.
        y : numpy.ndarray of shape (n_samples,)
            Target of the instances in the node.
        numerical : list of int
            Indices of the numerical columns of X.
        categorical : list of int
            Indices of the categorical columns of X.
        sample_weight : numpy.ndarray of shape (n_samples,), default=None
            Sample weights of the instances in the node.
        check_input : bool, default=True
            Whether to check the input data.

        Returns
        -------
        self : XGBoostSplitter
            The fitted splitter.

        Raises
        ------
        NoSplitFoundWarning
            If no split is found.
        """
        cover_fn, sim_fn = self._get_cover_similarity_fun()

        root_cover = cover_fn(y)
        root_similarity = sim_fn(y, root_cover, self.lam)
        best_gain, best_feature_idx, best_threshold, best_categorical = -np.inf, -1, -1, False

        for feature_idx in range(X.shape[1]):
            for threshold in self.get_threshold_values(X, feature_idx, categorical):
                if feature_idx in numerical:
                    cond_l = X[:, feature_idx] <= threshold
                    cond_r = ~cond_l
                else:
                    cond_l = X[:, feature_idx] == threshold
                    cond_r = ~cond_l

                cover_l = cover_fn(y[cond_l])
                cover_r = cover_fn(y[cond_r])
                if cover_l < self.min_cover or cover_r < self.min_cover:
                    continue
                similarity_l = xgboost_similarity_score_regression(y[cond_l], cover_l, self.lam)
                similarity_r = xgboost_similarity_score_regression(y[cond_r], cover_r, self.lam)

                gain = similarity_l + similarity_r - root_similarity

                if gain > best_gain:
                    best_gain = gain
                    best_feature_idx = feature_idx
                    best_threshold = threshold
                    best_categorical = feature_idx in categorical

        if best_feature_idx == -1:
            raise NoSplitFoundWarning('No split found')

        self.feature_ = best_feature_idx
        self.threshold_ = best_threshold
        self.is_categorical_ = best_categorical
        self.impurity_ = None
        self.update_statistics(X, y, sample_weight=sample_weight)

        return self

    def update_statistics(self, X, y, sample_weight=None):
        """
        Recompute the similarity scores and the gain of the split (``gain`` and ``gain_``).

        Parameters
        ----------
        X : numpy.ndarray of shape (n_samples, n_features)
            Feature matrix of the instances in the node.
        y : numpy.ndarray of shape (n_samples,)
            Target of the instances in the node.
        sample_weight : numpy.ndarray of shape (n_samples,), default=None
            Sample weights of the instances in the node.
        """
        left = self.apply(X) == 1

        cover_fn, sim_fn = self._get_cover_similarity_fun()

        root_cover = cover_fn(y)
        self.root_similarity = sim_fn(y, root_cover, self.lam)
        cover_l = cover_fn(y[left])
        cover_r = cover_fn(y[~left])
        self.similarity_l = xgboost_similarity_score_regression(y[left], cover_l, self.lam)
        self.similarity_r = xgboost_similarity_score_regression(y[~left], cover_r, self.lam)
        self.gain = self.similarity_l + self.similarity_r - self.root_similarity
        self.gain_ = self.gain
