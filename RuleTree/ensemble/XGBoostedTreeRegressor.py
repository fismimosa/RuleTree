from psutil import cpu_count

from RuleTree.ensemble.GBoostedTreeRegressor import GBoostedTreeRegressor
from RuleTree.tree.RuleTreeXGBoostRegressor import RuleTreeXGBoostRegressor


class XGBoostedTreeRegressor(GBoostedTreeRegressor):
    def __init__(self, base_estimator=RuleTreeXGBoostRegressor(n_jobs=cpu_count()),
                 learning_rate=.3, **kwargs):
        super().__init__(base_estimator=base_estimator, learning_rate=learning_rate, **kwargs)

    def _get_base_prediction(self, y):
        return .5