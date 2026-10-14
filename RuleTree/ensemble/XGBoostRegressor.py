from psutil import cpu_count

from RuleTree.ensemble.GBoostRegressor import GBoostRegressor
from RuleTree.stumps.regression.XGBoostStumpRegressor import XGBoostStumpRegressor
from RuleTree.tree.RuleTreeXGBoostRegressor import RuleTreeXGBoostRegressor


class XGBoostRegressor(GBoostRegressor):
    def __init__(self, base_estimator=None, learning_rate=.3, **kwargs):
        # base_estimator=None: the default RuleTreeXGBoostRegressor is built in fit, not at import time
        super().__init__(base_estimator=base_estimator, learning_rate=learning_rate, **kwargs)

    def _get_base_estimator(self):
        if self.base_estimator is not None:
            return self.base_estimator

        n_jobs = cpu_count()
        return RuleTreeXGBoostRegressor(n_jobs=n_jobs, base_stumps=XGBoostStumpRegressor(lam=1, n_jobs=n_jobs))

    def _get_base_prediction(self, y):
        return .5
