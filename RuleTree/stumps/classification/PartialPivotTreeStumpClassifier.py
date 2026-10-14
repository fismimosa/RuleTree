from typing import Optional

import numpy as np
import psutil

from RuleTree.stumps.classification.DecisionTreeStumpClassifier import DecisionTreeStumpClassifier
from RuleTree.stumps.transformers.PartialPivotTransformer import PartialPivotTransformer


class PartialPivotTreeStumpClassifier(DecisionTreeStumpClassifier):
    """
    Partial-pivot stump for classification of tabular data: the split compares the distance of the
    instances from a partial pivot, an instance of the node restricted to a subset of the features.

    It is made of a PartialPivotTransformer and a ClassificationSplitter.

    Parameters
    ----------
    n_shapelets, n_ts_for_selection, n_features_strategy, selection, distance, scaler, use_combination,
    random_state, n_jobs
        Parameters of the selection of the partial pivots (see TabularShapelets and PartialPivotTransformer).
    criterion, min_samples_leaf, min_weight_fraction_leaf, max_features, min_impurity_decrease, class_weight
        As in DecisionTreeStumpClassifier.
    """

    def __init__(self,
                 n_shapelets=psutil.cpu_count(logical=False)*2,
                 n_ts_for_selection=100,  # int, inf
                 n_features_strategy=2,
                 selection='random',  # random, cluster, all
                 distance='euclidean',
                 scaler=None,
                 use_combination=True,
                 random_state=42, n_jobs=1,
                 criterion='gini', min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=None,
                 min_impurity_decrease=0.0, class_weight=None):
        if selection not in ["random", "cluster", "all"]:
            raise ValueError("'selection' must be 'random', 'all' or 'cluster'")

        self.n_shapelets = n_shapelets
        self.n_ts_for_selection = n_ts_for_selection
        self.n_features_strategy = n_features_strategy
        self.selection = selection
        self.distance = distance
        self.scaler = scaler
        self.use_combination = use_combination
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.criterion = criterion
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_features = max_features
        self.min_impurity_decrease = min_impurity_decrease
        self.class_weight = class_weight

    def _splitter_params(self):
        # random_state is the one of the selection of the partial pivots
        return super()._splitter_params() | {"random_state": None}

    def _make_transformer(self):
        return PartialPivotTransformer(n_shapelets=self.n_shapelets,
                                       n_ts_for_selection=self.n_ts_for_selection,
                                       n_features_strategy=self.n_features_strategy,
                                       selection=self.selection,
                                       distance=self.distance,
                                       scaler=self.scaler,
                                       use_combination=self.use_combination,
                                       random_state=self.random_state,
                                       n_jobs=self.n_jobs)

    @property
    def st(self):
        """The fitted transform of the partial pivots."""
        return self.transformer_.st_

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
