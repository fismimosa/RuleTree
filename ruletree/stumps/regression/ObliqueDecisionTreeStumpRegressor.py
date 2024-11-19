import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_poisson_deviance
from sklearn.tree import DecisionTreeRegressor

from ruletree.base.RuleTreeBaseStump import RuleTreeBaseStump


class ObliqueDecisionTreeStumpRegressor(DecisionTreeRegressor, RuleTreeBaseStump):
    def get_rule(self, columns_names=None, scaler=None, float_precision=3):
        raise NotImplementedError()

    def node_to_dict(self, col_names):
        raise NotImplementedError()

    def export_graphviz(self, graph=None, columns_names=None, scaler=None, float_precision=3):
        raise NotImplementedError()

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.oblique_split = None
        self.is_categorical = False
        self.kwargs = kwargs
        self.unique_val_enum = None
        self.threshold_original = None
        self.feature_original = None

        if kwargs['criterion'] == "squared_error":
            self.impurity_fun = mean_squared_error
        elif kwargs['criterion'] == "friedman_mse":
            raise Exception("not implemented")  # TODO: implement
        elif kwargs['criterion'] == "absolute_error":
            self.impurity_fun = mean_absolute_error
        elif kwargs['criterion'] == "poisson":
            self.impurity_fun = mean_poisson_deviance
        else:
            self.impurity_fun = kwargs['criterion']

    def __impurity_fun(self, **x):
        return self.impurity_fun(**x) if len(x["y_true"]) > 0 else 0  # TODO: check

    def get_params(self, deep=True):
        return self.kwargs

    def fit(self, X, y, sample_weight=None, check_input=True):
        dtypes = pd.DataFrame(X).infer_objects().dtypes
        self.numerical = dtypes[dtypes != np.dtype('O')].index
        self.categorical = dtypes[dtypes == np.dtype('O')].index

        if len(self.numerical) > 0:
            self.oblique_split.fit(X[:, self.numerical], y, sample_weight=sample_weight, check_input=check_input)
            X_transform = self.oblique_split.transform(X[:, self.numerical])
            super().fit(X_transform, y, sample_weight=sample_weight, check_input=check_input)

            self.feature_original = [self.oblique_split.feats, -2, -2]
            self.coefficients = self.oblique_split.coeff
            self.threshold_original = self.tree_.threshold

        return self

    def apply(self, X):
        X_transform = self.oblique_split.transform(X[:, self.numerical])
        return super().apply(X_transform)
