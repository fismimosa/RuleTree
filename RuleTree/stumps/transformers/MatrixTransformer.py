import numpy as np

from RuleTree.base.RuleTreeBaseTransformer import RuleTreeBaseTransformer, get_feature_types
from RuleTree.utils.define import DATA_TYPE_IMAGE, DATA_TYPE_TABULAR, DATA_TYPE_TS


class MatrixTransformer(RuleTreeBaseTransformer):
    """
    Uses the data as feature matrix: tabular data as they are, time series and images flattened
    into one feature per value. The feature types (numerical or categorical) are computed on the
    whole data set.
    """

    supported_data_types = (DATA_TYPE_TABULAR, DATA_TYPE_TS, DATA_TYPE_IMAGE)

    def fit(self, X, y, idx=None, data_type=None, cache=None, sample_weight=None):
        self.numerical_, self.categorical_ = get_feature_types(self.transform(X), cache, data_type)
        return self

    def transform(self, X):
        X = np.asarray(X)
        return X if X.ndim <= 2 else X.reshape(X.shape[0], -1)
