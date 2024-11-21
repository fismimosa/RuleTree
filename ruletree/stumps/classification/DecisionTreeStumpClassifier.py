import copy

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier


from ruletree.base.RuleTreeBaseStump import RuleTreeBaseStump

from ruletree.utils.data_utils import get_info_gain, _get_info_gain, gini, entropy, _my_counts


class DecisionTreeStumpClassifier(DecisionTreeClassifier, RuleTreeBaseStump):

    def get_rule(self, columns_names=None, scaler=None, float_precision=3):
        rule = {
            "feature_idx": self.feature_original[0],
            "threshold": self.threshold_original[0],
            "is_categorical": self.is_categorical,
        }

        feat_name = f"X_{rule['feature_idx']}"
        if columns_names is not None:
            feat_name = columns_names[self.feature_original[0]]
        rule["feature_name"] = feat_name

        if scaler is not None:
            array = np.zeros((1, scaler.n_features_in_))
            array[0, self.feature_original[0]] = self.feature_original[0]

            rule["threshold_scaled"] = scaler.inverse_transform(array)[0, self.feature_original[0]]

        comparison = "<=" if self.is_categorical else "="
        rounded_value = str(rule["threshold"]) if float_precision is None else round(rule["threshold"], float_precision)
        rule["textual_rule"] = f"{feat_name} {comparison} {rounded_value}"
        rule["blob_rule"] = f"{feat_name} {comparison} {rounded_value}"
        rule["graphviz_rule"] = f"{feat_name} {comparison} {rounded_value}"

        return rule

    def node_to_dict(self):
        rule = self.get_rule(float_precision=None)

        rule["stump_type"] = self.__class__.__name__
        rule["samples"] = self.tree_.n_node_samples[0]
        rule["impurity"] = self.tree_.impurity[0]

        rule["args"] = {
            "is_oblique": self.is_oblique,
            "is_pivotal": self.is_pivotal,
            "unique_val_enum": self.unique_val_enum,
            "coefficients": self.coefficients,
        } | self.kwargs

        rule["split"] = {
            "args": {}
        }

        return rule

    def dict_to_node(self, node_dict):
        self.feature_original = np.zeros(3)
        self.threshold_original = np.zeros(3)

        self.feature_original[0] = node_dict["feature_original"]
        self.threshold_original[0] = node_dict["threshold"]
        self.is_categorical = node_dict["is_categorical"]

        args = copy.deepcopy(node_dict["args"])
        self.is_oblique = args.pop("is_oblique")
        self.is_pivotal = args.pop("is_pivotal")
        self.unique_val_enum = args.pop("unique_val_enum")
        self.coefficients = args.pop("coefficients")
        self.kwargs = args


    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.is_categorical = None
        self.is_oblique = False # TODO: @Alessio, ma questi ci servono qui? Non sarebbe meglio mettere tutto in
        self.is_pivotal = False #       PivotTreeStumpClassifier e poi usare l'ereditarietà da quello?

        
        self.kwargs = kwargs
        self.unique_val_enum = None
        
        self.threshold_original = None
        self.feature_original = None
        self.coefficients = None
        
        if 'criterion' not in kwargs or kwargs['criterion'] == "gini":
            self.impurity_fun = gini
        elif kwargs['criterion'] == "entropy":
            self.impurity_fun = entropy
        else:
            self.impurity_fun = kwargs['criterion']

    def get_params(self, deep=True):
        return self.kwargs
    
    def feature_analysis(self, X, y):
        dtypes = pd.DataFrame(X).infer_objects().dtypes
        self.numerical = dtypes[dtypes != np.dtype('O')].index
        self.categorical = dtypes[dtypes == np.dtype('O')].index

    def fit(self, X, y, sample_weight=None, check_input=True):
        self.feature_analysis(X, y)
        best_info_gain = -float('inf')

        if len(self.numerical) > 0:
            super().fit(X[:, self.numerical], y, sample_weight=sample_weight, check_input=check_input)
            self.feature_original = self.tree_.feature
            self.threshold_original = self.tree_.threshold
            best_info_gain = get_info_gain(self)
            
        self._fit_cat(X, y, best_info_gain)

        return self

    def _fit_cat(self, X, y, best_info_gain, sample_weight=None):
        if self.max_depth > 1:
            raise Exception("not implemented") # TODO: implement?

        if len(self.categorical) > 0 and best_info_gain != float('inf'):
            len_x = len(X)

            class_weight = None
            if self.class_weight == "balanced":
                class_weight = dict()
                for class_label in np.unique(y):
                    class_weight[class_label] = len_x / (len(self.classes_) * len(y[y == class_label]))


            for i in self.categorical:
                for value in np.unique(X[:, i]):
                    X_split = X[:, i:i+1] == value

                    len_left = np.sum(X_split)

                    if sample_weight is not None:
                        # TODO: check. Sample weights. If None, then samples are equally weighted. Splits that would
                        #  create child nodes with net zero or negative weight are ignored while searching for a split
                        #  in each node. Splits are also ignored if they would result in any single class carrying a
                        #  negative weight in either child node.

                        if _my_counts(y, sample_weight) - (_my_counts(y[X_split[:, 0]], sample_weight)
                                                           + _my_counts(y[~X_split[:, 0]], sample_weight)) <= 0:
                            continue

                        if sum(sample_weight[X_split[:, 0]]) < self.min_weight_fraction_leaf \
                            or sum(sample_weight[~X_split[:, 0]]) < self.min_weight_fraction_leaf:
                            continue

                        if ((_my_counts(y[X_split[:, 0]], sample_weight) <= 0).any()
                                or (_my_counts(y[~X_split[:, 0]], sample_weight) <= 0).any()):
                            continue

                        info_gain = _get_info_gain(self.impurity_fun(y, sample_weight, class_weight),
                                                   self.impurity_fun(y[X_split[:, 0]],
                                                                     sample_weight[X_split[:, 0]],
                                                                     class_weight),
                                                   self.impurity_fun(y[~X_split[:, 0]],
                                                                     sample_weight[~X_split[:, 0]],
                                                                     class_weight),
                                                   len_x,
                                                   len_left,
                                                   len_x-len_left)
                    else:
                        info_gain = _get_info_gain(self.impurity_fun(y, sample_weight, class_weight),
                                                   self.impurity_fun(y[X_split[:, 0]], None, class_weight),
                                                   self.impurity_fun(y[~X_split[:, 0]], None, class_weight),
                                                   len_x,
                                                   len_left,
                                                   len_x - len_left)

                    if info_gain > best_info_gain:
                        best_info_gain = info_gain
                        self.feature_original = [i, -2, -2]
                        self.threshold_original = np.array([value, -2, -2])
                        self.unique_val_enum = np.unique(X[:, i])
                        self.is_categorical = True


    def apply(self, X, check_input=False):
        if not self.is_categorical:
            y_pred = (np.ones(X.shape[0]) * 2)
            X_feature = X[:, self.feature_original[0]]
            y_pred[X_feature <= self.threshold_original[0]] = 1
            
            return y_pred
            
        else:
            y_pred = np.ones(X.shape[0]) * 2
            X_feature = X[:, self.feature_original[0]]
            y_pred[X_feature == self.threshold_original[0]] = 1

            return y_pred
        
