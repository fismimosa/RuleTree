"""XGBoostSplitter: the best split according to the XGBoost gain."""

import numpy as np

from RuleTree.base.RuleTreeBaseSplitter import RuleTreeBaseSplitter
from RuleTree.exceptions import NoSplitFoundWarning
from RuleTree.utils.data_utils import check_one_target
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
    >>> splitter = XGBoostSplitter(lam=1)
    >>> splitter = splitter.fit(X, y - y.mean(), numerical=list(range(10)), categorical=[], context=None, node_id=None)
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

    def fit(self, X, y, numerical, categorical, sample_weight=None, check_input=True, *, context,
            node_id):
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
        context : RuleTree or None
            Tree being built, None outside a tree.
        node_id : str or None
            Identifier of the node, e.g. ``"Rl"`` (``context.get_node_by_id(node_id)``), None outside a tree.

        Returns
        -------
        self : XGBoostSplitter
            The fitted splitter.

        Raises
        ------
        NoSplitFoundWarning
            If no split is found.
        ValueError
            If y has several targets.
        """
        check_one_target(y, self.__class__.__name__)
        cover_fn, sim_fn = self._get_cover_similarity_fun()
        root_similarity = sim_fn(y, cover_fn(y), self.lam)

        # The result is the one of the exhaustive search: the first candidate, in the order of the features
        # and of their thresholds, with the highest gain. The gains of all the thresholds of a numerical
        # feature are first computed at once (sorting and cumulative sums), then the exact gain of the
        # candidates close to the best one is computed as the exhaustive search does, so that the rounding
        # of the cumulative sums cannot change the result.
        candidates, approximate_gains = [], []  # (feature, threshold), in the order of the exhaustive search
        for feature_idx in range(X.shape[1]):
            thresholds = self.get_threshold_values(X, feature_idx, categorical)
            if feature_idx in numerical and (cover_fn, sim_fn) == (xgboost_cover_regression,
                                                                   xgboost_similarity_score_regression):
                gains = self._approximate_gains(X[:, feature_idx], y, thresholds, root_similarity)
            else:
                gains = [self._gain(X, y, feature_idx, threshold, numerical, root_similarity)
                         for threshold in thresholds]
            candidates += [(feature_idx, threshold) for threshold in thresholds]
            approximate_gains = np.concatenate([approximate_gains, gains])

        finite = np.isfinite(approximate_gains)
        if not finite.any():
            raise NoSplitFoundWarning('No split found')
        best_approximate = approximate_gains[finite].max()
        scale = 1 + abs(root_similarity) + np.abs(approximate_gains[finite] + root_similarity).max()
        close = finite & (approximate_gains >= best_approximate - 1e-9 * scale)

        best_gain, best_feature_idx, best_threshold = -np.inf, -1, -1
        for k in np.flatnonzero(close):
            feature_idx, threshold = candidates[k]
            gain = self._gain(X, y, feature_idx, threshold, numerical, root_similarity)
            if gain > best_gain:
                best_gain, best_feature_idx, best_threshold = gain, feature_idx, threshold
        best_categorical = best_feature_idx in categorical

        self.feature_ = best_feature_idx
        self.threshold_ = best_threshold
        self.is_categorical_ = best_categorical
        self.impurity_ = None
        self.update_statistics(X, y, sample_weight=sample_weight, context=context, node_id=node_id)

        return self

    def _gain(self, X, y, feature_idx, threshold, numerical, root_similarity):
        """
        Gain of a candidate split, as the exhaustive search computes it.

        Parameters
        ----------
        X : numpy.ndarray of shape (n_samples, n_features)
            Feature matrix of the instances in the node.
        y : numpy.ndarray of shape (n_samples,)
            Residuals of the instances in the node.
        feature_idx : int
            Column of the split.
        threshold : float or object
            Threshold (numerical feature, ``x <= threshold`` goes left) or value (``x == value``).
        numerical : list of int
            Indices of the numerical columns of X.
        root_similarity : float
            Similarity score of the node.

        Returns
        -------
        gain : float
            -inf if a child has a cover smaller than min_cover.
        """
        cover_fn, sim_fn = self._get_cover_similarity_fun()
        cond_l = X[:, feature_idx] <= threshold if feature_idx in numerical else X[:, feature_idx] == threshold
        cover_l, cover_r = cover_fn(y[cond_l]), cover_fn(y[~cond_l])
        if cover_l < self.min_cover or cover_r < self.min_cover:
            return -np.inf
        return sim_fn(y[cond_l], cover_l, self.lam) + sim_fn(y[~cond_l], cover_r, self.lam) - root_similarity

    def _approximate_gains(self, x, y, thresholds, root_similarity) -> np.ndarray:
        """
        Gains of all the thresholds of a numerical feature at once, from the cumulative sums of the
        residuals sorted by the feature (regression similarity: sum ** 2 / (cover + lam)). They can differ
        from the exact gains by the rounding of the sums.

        Parameters
        ----------
        x : numpy.ndarray of shape (n_samples,)
            Values of the feature.
        y : numpy.ndarray of shape (n_samples,)
            Residuals.
        thresholds : numpy.ndarray
            Thresholds (``x <= threshold`` goes left).
        root_similarity : float
            Similarity score of the node.

        Returns
        -------
        gains : numpy.ndarray
            -inf for the thresholds with a child smaller than min_cover.
        """
        x = np.asarray(x, dtype=float)
        order = np.argsort(x, kind="stable")
        n_left = np.searchsorted(x[order], np.asarray(thresholds, dtype=float), side="right")
        n_right = len(y) - n_left
        cumulative = np.concatenate([[0.0], np.cumsum(y[order])])
        sum_left = cumulative[n_left]
        gains = sum_left ** 2 / (n_left + self.lam) + (cumulative[-1] - sum_left) ** 2 / (n_right + self.lam) \
            - root_similarity
        gains[(n_left < self.min_cover) | (n_right < self.min_cover)] = -np.inf
        return gains

    def update_statistics(self, X, y, sample_weight=None, *, context, node_id):
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
        context : RuleTree, default=None
            The tree, None outside a tree.
        node_id : str, default=None
            Identifier of the node, e.g. ``"Rl"`` (``context.get_node_by_id(node_id)``), None outside a tree.
        """
        left = self.apply(X, context=context, node_id=node_id) == 1

        cover_fn, sim_fn = self._get_cover_similarity_fun()

        root_cover = cover_fn(y)
        self.root_similarity = sim_fn(y, root_cover, self.lam)
        cover_l = cover_fn(y[left])
        cover_r = cover_fn(y[~left])
        self.similarity_l = xgboost_similarity_score_regression(y[left], cover_l, self.lam)
        self.similarity_r = xgboost_similarity_score_regression(y[~left], cover_r, self.lam)
        self.gain = self.similarity_l + self.similarity_r - self.root_similarity
        self.gain_ = self.gain
