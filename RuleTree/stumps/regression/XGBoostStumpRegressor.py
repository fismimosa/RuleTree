"""XGBoostStumpRegressor: the split of the trees of XGBoost."""

from RuleTree.stumps.regression.DecisionTreeStumpRegressor import DecisionTreeStumpRegressor
from RuleTree.stumps.splitters.XGBoostSplitter import XGBoostSplitter


class XGBoostStumpRegressor(DecisionTreeStumpRegressor):
    """
    Stump of the XGBoost trees (regression): the threshold with the highest XGBoost gain, the
    similarity score of the two children minus the one of the node.

    It is a :class:`~RuleTree.stumps.Stump.Stump` made of a MatrixTransformer and an XGBoostSplitter,
    used by RuleTreeXGBoostRegressor, whose target are the residuals of the previous trees.

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
    transformer_ : MatrixTransformer
        The fitted transformer.
    splitter_ : XGBoostSplitter
        The fitted splitter (feature, threshold, gain and impurity of the split).
    data_type_ : str
        Data type the stump was fitted on.
    feature_original : int
        Column of the data used by the split.
    threshold_original : float
        Threshold of the split.

    See Also
    --------
    RuleTree.tree.RuleTreeXGBoostRegressor.RuleTreeXGBoostRegressor : the tree that uses this stump.
    RuleTree.ensemble.XGBoostRegressor.XGBoostRegressor : the boosting of these trees.

    Examples
    --------
    >>> from sklearn.datasets import load_diabetes
    >>> from RuleTree.stumps.regression import XGBoostStumpRegressor
    >>> X, y = load_diabetes(return_X_y=True)
    >>> stump = XGBoostStumpRegressor(lam=1).fit(X, y - y.mean())  # residuals of the mean
    >>> stump.get_rule()["textual_rule"]
    'X_8 <= -0.004'
    """

    def __init__(self, lam, approximation=None, n_quantiles=32, min_cover=1, n_jobs=1):
        """Initialize the XGBoostStumpRegressor; see XGBoostSplitter for the parameters."""
        if approximation is not None and approximation not in ['quantile']:
            raise ValueError('approximation must be None or quantile')

        self.lam = lam
        self.approximation = approximation
        self.n_quantiles = n_quantiles
        self.min_cover = min_cover
        self.n_jobs = n_jobs

    def _make_splitter(self):
        """
        A new, unfitted XGBoostSplitter with the parameters of the stump.

        Returns
        -------
        splitter : XGBoostSplitter
        """
        return XGBoostSplitter(lam=self.lam, approximation=self.approximation, n_quantiles=self.n_quantiles,
                               min_cover=self.min_cover, n_jobs=self.n_jobs)

    @property
    def gain(self):
        """XGBoost gain of the split, used by the gamma pruning of RuleTreeXGBoostRegressor."""
        return self.splitter_.gain
