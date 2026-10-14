from RuleTree.stumps.regression.DecisionTreeStumpRegressor import DecisionTreeStumpRegressor
from RuleTree.stumps.splitters.XGBoostSplitter import XGBoostSplitter


class XGBoostStumpRegressor(DecisionTreeStumpRegressor):
    """
    Stump for the trees of XGBoost (regression): the split maximizing the XGBoost gain.

    It is made of a MatrixTransformer and an XGBoostSplitter (see it for the parameters).
    """

    def __init__(self, lam, approximation=None, n_quantiles=32, min_cover=1, n_jobs=1):
        if approximation is not None and approximation not in ['quantile']:
            raise ValueError('approximation must be None or quantile')

        self.lam = lam
        self.approximation = approximation
        self.n_quantiles = n_quantiles
        self.min_cover = min_cover
        self.n_jobs = n_jobs

    def _make_splitter(self):
        return XGBoostSplitter(lam=self.lam, approximation=self.approximation, n_quantiles=self.n_quantiles,
                               min_cover=self.min_cover, n_jobs=self.n_jobs)

    @property
    def gain(self):
        """XGBoost gain of the split, used by the gamma pruning of RuleTreeXGBoostRegressor."""
        return self.splitter_.gain
