from ruletree.base.RuleTreeBaseStump import RuleTreeBaseStump
from ruletree.stumps.classification.DecisionTreeStumpClassifier import DecisionTreeStumpClassifier
from ruletree.stumps.splitters.PivotSplit import PivotSplit
from ruletree.utils import MODEL_TYPE_CLF


class PivotTreeStumpClassifier(DecisionTreeStumpClassifier, RuleTreeBaseStump):
    def __init__(self, distance_matrix=None, distance_measure='euclidean', **kwargs):
        super().__init__(**kwargs)
        self.pivot_split = PivotSplit(ml_task=MODEL_TYPE_CLF, **kwargs)
        self.distance_matrix = distance_matrix
        self.distance_measure = distance_measure

    def fit(self, X, y, distance_matrix, distance_measure, idx, sample_weight=None, check_input=True):
        self.feature_analysis(X, y)
        self.num_pre_transformed = self.numerical
        self.cat_pre_transformed = self.categorical
        best_info_gain = -float('inf')

        if len(self.numerical) > 0:
            self.pivot_split.fit(X[:, self.numerical], y, distance_matrix, distance_measure, idx,
                                 sample_weight=sample_weight, check_input=check_input)
            X_transform = self.pivot_split.transform(X[:, self.numerical])
            candidate_names = self.pivot_split.get_candidates_names()
            super().fit(X_transform, y, sample_weight=sample_weight, check_input=check_input)

            self.feature_original = [f'{candidate_names[self.tree_.feature[0]]}', -2, -2]
            self.threshold_original = self.tree_.threshold
            self.is_pivotal = True

        return self

    def apply(self, X):
        X_transformed = self.pivot_split.transform(X[:, self.num_pre_transformed], self.distance_measure)
        return super().apply_sk(X_transformed)

    def get_rule(self, columns_names=None, scaler=None, float_precision=3):
        rule = {
            "feature_idx": self.feature_original[0],
            "threshold": self.threshold_original[0],
            "coefficients" : self.coefficients,
            "is_categorical": self.is_categorical,
            "samples": self.n_node_samples[0]
        }

        feat_name = f"P_{rule['feature_idx']}"
        if columns_names is not None:
            #feat_names should not be useful for pivot tree
            #feat_name = columns_names[self.feature_original[0]]
            feat_name = None
        rule["feature_name"] = feat_name

        if scaler is not None:
            NotImplementedError()

        comparison = "<=" if not self.is_categorical else "="
        not_comparison = ">" if not self.is_categorical else "!="
        rounded_value = str(rule["threshold"]) if float_precision is None else round(rule["threshold"], float_precision)
        if scaler is not None:
            rounded_value = str(rule["threshold_scaled"]) if float_precision is None else (
                round(rule["threshold_scaled"], float_precision))
        rule["textual_rule"] = f"{feat_name} {comparison} {rounded_value}\t{rule['samples']}"
        rule["blob_rule"] = f"{feat_name} {comparison} {rounded_value}"
        rule["graphviz_rule"] = {
            "label": f"{feat_name} {comparison} {rounded_value}",
        }

        rule["not_textual_rule"] = f"{feat_name} {not_comparison} {rounded_value}"
        rule["not_blob_rule"] = f"{feat_name} {not_comparison} {rounded_value}"
        rule["not_graphviz_rule"] = {
            "label": f"{feat_name} {not_comparison} {rounded_value}"
        }

        return rule

    def node_to_dict(self):
        rule = self.get_rule(float_precision=None)

        rule["stump_type"] = self.__class__.__name__
        rule["samples"] = self.n_node_samples[0]
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

    def export_graphviz(self, graph=None, columns_names=None, scaler=None, float_precision=3):
        raise NotImplementedError()
