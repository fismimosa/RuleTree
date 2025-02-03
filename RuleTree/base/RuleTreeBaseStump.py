from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator

from RuleTree.utils.define import DATA_TYPE_TABULAR


class RuleTreeBaseStump(BaseEstimator, ABC):
    @abstractmethod
    def get_rule(self, columns_names=None, scaler=None, float_precision:int|None=3):
        pass

    def feature_analysis(self, X, y):
        dtypes = pd.DataFrame(X).infer_objects().dtypes
        self.numerical = dtypes[dtypes != np.dtype('O')].index
        self.categorical = dtypes[dtypes == np.dtype('O')].index

    @abstractmethod
    def node_to_dict(self):
        pass

    @classmethod
    @abstractmethod
    def dict_to_node(self, node_dict, X):
        pass

    @staticmethod
    def supports(data_type):
        return data_type in [DATA_TYPE_TABULAR]
