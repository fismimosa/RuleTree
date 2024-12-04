from ruletree.base.RuleTreeBaseStump import RuleTreeBaseStump
from ruletree.stumps.classification.DecisionTreeStumpClassifier import DecisionTreeStumpClassifier
from ruletree.stumps.splitters.MultiplePivotSplit import MultiplePivotSplit
from ruletree.utils import MODEL_TYPE_CLF


class MultiplePivotTreeStumpClassifier(DecisionTreeStumpClassifier, RuleTreeBaseStump):
    def __init__(self, distance_matrix=None, distance_measure='euclidean', **kwargs):
        super().__init__(**kwargs)
        self.multi_pivot_split = MultiplePivotSplit(ml_task=MODEL_TYPE_CLF, **kwargs)
        self.distance_matrix = distance_matrix
        self.distance_measure = distance_measure

    def fit(self, X, y, distance_matrix, distance_measure, idx, sample_weight=None, check_input=True):
        self.feature_analysis(X, y)
        self.num_pre_transformed = self.numerical
        self.cat_pre_transformed = self.categorical

        if len(self.numerical) > 0:
            self.multi_pivot_split.fit(X[:, self.numerical], y, distance_matrix, distance_measure, idx,
                                       sample_weight=sample_weight, check_input=check_input)
            X_transform = self.multi_pivot_split.transform(X[:, self.numerical])
            candidate_names = self.multi_pivot_split.get_candidates_names()
            super().fit(X_transform, y, sample_weight=sample_weight, check_input=check_input)

            self.feature_original = [(candidate_names[0], candidate_names[1]), -2, -2]
            self.threshold_original = self.tree_.threshold
            self.is_pivotal = True

        return self

    def apply(self, X):
        X_transformed = self.multi_pivot_split.transform(X[:, self.num_pre_transformed], self.distance_measure)
        return super().apply_sk(X_transformed)

    def get_rule(self, columns_names=None, scaler=None, float_precision=3):
        
         rule = {
             "feature_idx": self.feature_original[0], #tuple of instances idx
             "threshold": self.threshold_original[0], #thr
             "coefficients" : self.coefficients, #coefficients
             "is_categorical": self.is_categorical,
             "samples": self.n_node_samples[0]
         }
              
         feat_name = " ".join(f"P_{idx}" for idx in list(rule['feature_idx'])) #list of sitrings
        
         if columns_names is not None:
             feat_name = "_".join(columns_names[idx] for idx in self.feature_original[0]) #check this for feat names
         rule["feature_name"] = feat_name
         
         if scaler is not None:
             #TODO
             raise NotImplementedError()
             pass
         
         comparison = f"closer to {rule['feature_idx'][0]}" if not self.is_categorical else "="
         not_comparison = f"closer to {rule['feature_idx'][1]}" if not self.is_categorical else "!="
         rounded_value = str(rule["threshold"]) if float_precision is None else round(rule["threshold"], float_precision)
         
         if scaler is not None:
             #TODO
             raise NotImplementedError()
             pass
         
         rule["textual_rule"] = f"{comparison} \t{rule['samples']}"
         rule["blob_rule"] = f"{comparison} "
         rule["graphviz_rule"] = {
             "label": f"{comparison} {rounded_value}",
         }
         
         rule["not_textual_rule"] = f"{not_comparison}"
         rule["not_blob_rule"] = f"{not_comparison}"
         rule["not_graphviz_rule"] = {
             "label": f"{not_comparison}"
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
