import random
import warnings
from typing import Optional

import numpy as np
import psutil


from RuleTree.stumps.classification.DecisionTreeStumpClassifier import DecisionTreeStumpClassifier
from RuleTree.utils.define import DATA_TYPE_TABULAR
from RuleTree.utils.shapelet_transform.TabularShapelets import TabularShapelets


class PartialPivotTreeStumpClassifier(DecisionTreeStumpClassifier):
    supported_data_types = (DATA_TYPE_TABULAR,)

    def __init__(self,
                 n_shapelets=psutil.cpu_count(logical=False)*2,
                 n_ts_for_selection=100,  #int, inf
                 n_features_strategy=2,
                 selection='random',  #random, mi_clf, mi_reg, cluster
                 distance='euclidean',
                 scaler = None,
                 use_combination=True,
                 random_state=42, n_jobs=1,
                 **kwargs):
        self.n_shapelets = n_shapelets
        self.n_ts_for_selection = n_ts_for_selection
        self.n_features_strategy = n_features_strategy
        self.selection = selection
        self.distance = distance
        self.scaler = scaler
        self.use_combination = use_combination
        self.random_state = random_state
        self.n_jobs = n_jobs

        if "max_depth" in kwargs and kwargs["max_depth"] > 1:
            warnings.warn("max_depth must be 1")

        kwargs["max_depth"] = 1

        if selection not in ["random", "cluster", "all"]:
            raise ValueError("'selection' must be 'random', 'all' or 'cluster'")

        self.st = TabularShapelets(n_shapelets=self.n_shapelets,
                                   n_ts_for_selection=self.n_ts_for_selection,
                                   n_features_strategy=self.n_features_strategy,
                                   selection=self.selection,
                                   distance=self.distance,
                                   random_state=self.random_state,
                                   use_combination=self.use_combination,
                                   n_jobs=self.n_jobs
                                   )

        super().__init__(**kwargs)

        self.kwargs |= {
            "n_shapelets": n_shapelets,
            "n_ts_for_selection": n_ts_for_selection,
            "n_features_strategy": n_features_strategy,
            "selection": selection,
            "distance": distance,
            "scaler": scaler,
            "use_combination": use_combination,
            "random_state": random_state,
            "n_jobs": n_jobs,
        }

    def fit(self, data, y, idx=None, context=None, data_type=None, sample_weight=None, check_input=True):
        X, _ = self._resolve_fit_data(data, data_type)
        if self.scaler is not None:
            self.scaler.fit(X)

        if idx is None:
            idx = slice(None)

        X = X[idx]
        y = y[idx]
        sample_weight = None if sample_weight is None else sample_weight[idx]
        if self.scaler is not None:
            X = self.scaler.transform(X)

        random.seed(self.random_state)
        if sample_weight is not None:
            warnings.warn(f"sample_weight is not supported for {self.__class__.__name__}", Warning)

        X_transform = self.st.fit_transform(X, y)
        self.numerical, self.categorical = list(range(X_transform.shape[1])), []
        self._fit_matrix(X_transform, y, sample_weight=sample_weight, check_input=check_input)
        selected_shape = self.tree_.feature[0]
        if selected_shape == -2:
            raise ValueError("No split found")

        self.st._optimize_memory(np.array([selected_shape]))
        X_transform = self.st.transform(X, y)
        self.numerical, self.categorical = list(range(X_transform.shape[1])), []
        self._fit_matrix(X_transform, y, sample_weight=sample_weight, check_input=check_input)

        return self

    def _transform(self, X):
        if self.scaler is not None:
            X = self.scaler.transform(X)

        return self.st.transform(X)

    def get_rule(self, columns_names=None, scaler=None, float_precision: Optional[int] = 3):
        rule = {
            "feature_idx": self.feature_original[0],
            "threshold": self.threshold_original[0],
            "is_categorical": False,
        }

        if columns_names is None:
            columns_names = [f"Feat. {i}" for i in range(self.st.shapelets.shape[1])]

        pivot = np.copy(self.st.shapelets[self.feature_original[0]])
        if scaler is not None:
            pivot = scaler.inverse_transform(pivot.reshape(1, -1)).flatten()
        frac_not_null = 1- np.isnan(pivot).sum() / pivot.shape[0]
        if frac_not_null <= .50:
            feature_list = [f'{name}={val}' for name, val in zip(columns_names, pivot) if not np.isnan(val)]
            str_feat = 'd: ' + (", ".join(feature_list))
        else:
            feature_list = [f'{name}={val}' for name, val in zip(columns_names, pivot) if np.isnan(val)]
            str_feat = f"d all excpet: {', '.join(feature_list)}"

        rule["feature_name"] = f"SP{rule['feature_idx']}({str_feat})"

        comparison = "<="
        not_comparison = ">"
        rounded_value = str(rule["threshold"]) if float_precision is None else round(rule["threshold"], float_precision)

        rule["textual_rule"] = f"{rule['feature_name']} {comparison} {rounded_value}"
        rule["blob_rule"] = f"{rule['feature_name']} {comparison} {rounded_value}"
        rule["graphviz_rule"] = {
            "label": f"{rule['feature_name']} {comparison} {rounded_value}",
        }

        rule["not_textual_rule"] = f"{rule['feature_name']} {not_comparison} {rounded_value}"
        rule["not_blob_rule"] = f"{rule['feature_name']} {not_comparison} {rounded_value}"
        rule["not_graphviz_rule"] = {
            "label": f"{rule['feature_name']} {not_comparison} {rounded_value}"
        }

        return rule

    def node_to_dict(self):
        """
        Convert the stump to a dictionary representation: the split, the constructor arguments
        and the selected shapelets.

        Returns:
            dict: Dictionary representation of the node.
        """
        rule = super().node_to_dict()
        if self.feature_original is not None:
            rule["shapelets"] = self.st.shapelets.tolist()

        return rule

    @classmethod
    def dict_to_node(cls, node_dict, X=None):
        """
        Create a stump from a dictionary created by node_to_dict.

        Args:
            node_dict (dict): Dictionary representation of the node.
            X (array-like, optional): Not used.

        Returns:
            The reconstructed stump.
        """
        self = super().dict_to_node(node_dict)
        if "shapelets" in node_dict:
            self.st.shapelets = np.array(node_dict["shapelets"])

        return self

