from RuleTree.base.RuleTreeBaseStump import RuleTreeBaseStump
from RuleTree.stumps.classification.DecisionTreeStumpClassifier import DecisionTreeStumpClassifier
from RuleTree.stumps.splitters.ObliqueBivariateSplit import ObliqueBivariateSplit
from RuleTree.stumps.splitters.ObliqueHouseHolderSplit import ObliqueHouseHolderSplit
from RuleTree.stumps.splitters.ObliquePivotSplit import ObliquePivotSplit
from RuleTree.utils import MODEL_TYPE_CLF


class ObliquePivotTreeStumpClassifier(DecisionTreeStumpClassifier, RuleTreeBaseStump):
    def __init__(self,
                 oblique_split_type='householder',
                 pca=None,
                 max_oblique_features=2,
                 tau=1e-4,
                 n_orientations=10,
                 **kwargs):
        super().__init__(**kwargs)
        super().__init__(**kwargs)

        self.distance_measure = None
        self.pca = pca
        self.max_oblique_features = max_oblique_features
        self.tau = tau
        self.n_orientations = n_orientations
        self.oblique_split_type = oblique_split_type

        self.obl_pivot_split = ObliquePivotSplit(ml_task=MODEL_TYPE_CLF, oblique_split_type=oblique_split_type, **kwargs)

        if oblique_split_type == 'householder':
            self.oblique_split = ObliqueHouseHolderSplit(ml_task=MODEL_TYPE_CLF,
                                                         pca=self.pca,
                                                         max_oblique_features=self.max_oblique_features,
                                                         tau=self.tau,
                                                         **kwargs)

        if oblique_split_type == 'bivariate':
            self.oblique_split = ObliqueBivariateSplit(ml_task=MODEL_TYPE_CLF, n_orientations=self.n_orientations, **kwargs)

    def fit(self, X, y, distance_matrix, distance_measure, idx, sample_weight=None, check_input=True):
        self.feature_analysis(X, y)
        self.num_pre_transformed = self.numerical
        self.cat_pre_transformed = self.categorical

        if len(self.numerical) > 0:
            self.obl_pivot_split.fit(X[:, self.numerical], y, distance_matrix, distance_measure, idx,
                                     sample_weight=sample_weight, check_input=check_input)
            X_transform = self.obl_pivot_split.transform(X[:, self.numerical], distance_measure)
            candidate_names = self.obl_pivot_split.get_candidates_names()

            self.oblique_split.fit(X_transform, y, sample_weight=sample_weight, check_input=check_input)
            X_transform_oblique = self.oblique_split.transform(X_transform)
            super().fit(X_transform_oblique, y, sample_weight=sample_weight, check_input=check_input)

            feats = [f'{p}' for p in candidate_names[self.oblique_split.feats]]
            self.feature_original = [feats, -2, -2]
            self.coefficients = self.oblique_split.coeff
            self.threshold_original = self.tree_.threshold
            self.is_oblique = True
            self.is_pivotal = True
            self.distance_measure = distance_measure

        return self

    def apply(self, X):
        X_transformed = self.obl_pivot_split.transform(X[:, self.num_pre_transformed], self.distance_measure)
        X_transformed_oblique = self.oblique_split.transform(X_transformed)
        return super().apply_sk(X_transformed_oblique)
    
    def get_params(self, deep=True):
        return {
            **self.kwargs,
            'oblique_split_type' : self.oblique_split_type,
            'max_oblique_features': self.max_oblique_features,
            'pca': self.pca,
            'tau': self.tau,
            'n_orientations': self.n_orientations
        }

    def get_rule(self, columns_names=None, scaler=None, float_precision=3):
        rule = {
            "feature_idx": self.feature_original[0],
            "threshold": self.threshold_original[0],
            "coefficients" : self.coefficients,
            "is_categorical": self.is_categorical,
            "samples": self.n_node_samples[0]
        }
        
        rule['coefficients'] = [
                               str(coeff) if float_precision is None else round(float(coeff), float_precision) 
                               for coeff in rule['coefficients']
                               ]
        

        feat_name = " + ".join(f"{coeff} * P_{idx}" for coeff, idx in zip(rule['coefficients'], rule['feature_idx']))
       
        #if columns_names is not None:
        #    feat_name = "_".join(columns_names[idx] for idx in self.feature_original[0]) #check this for feat names
        #rule["feature_name"] = feat_name
        
    
        #feat_name = None
        rule["feature_name"] = feat_name

        #if scaler is not None:
        #    NotImplementedError()

        comparison = "<=" if not self.is_categorical else "="
        not_comparison = ">" if not self.is_categorical else "!="
        rounded_value = str(rule["threshold"]) if float_precision is None else round(rule["threshold"], float_precision)
        #if scaler is not None:
        #    NotImplementedError()
        rule["textual_rule"] = f"{feat_name} {comparison} {rounded_value}\t{rule['samples']}"
        rule["blob_rule"] = f"{feat_name} {comparison} {rounded_value}"
        rule["graphviz_rule"] = {
            "label": f"{feat_name} {'\u2264' if not self.is_categorical else '='} {rounded_value}",
        }

        rule["not_textual_rule"] = f"{feat_name} {not_comparison} {rounded_value}"
        rule["not_blob_rule"] = f"{feat_name} {not_comparison} {rounded_value}"
        rule["not_graphviz_rule"] = {
            "label": f"{feat_name} {'>' if not self.is_categorical else '\u2260'} {rounded_value}"
        }

        return rule
    

    def node_to_dict(self):
        rule = self.get_rule(float_precision=None)

        rule["stump_type"] = self.__class__.__module__
        rule["samples"] = self.n_node_samples[0]
        rule["impurity"] = self.tree_.impurity[0]
     
      
        rule["args"] = {
            "is_oblique": self.is_oblique,
            "is_pivotal": self.is_pivotal,
            "unique_val_enum": self.unique_val_enum,
            "coefficients": self.coefficients,
            "num_pre_transformed" : self.num_pre_transformed,
            "cat_pre_transformed" : self.cat_pre_transformed,
            
            "distance_measure" : self.distance_measure #adding this for PT
            
        } | self.kwargs

        rule["split"] = {
            "args": {}
        }
        

        return rule

    def export_graphviz(self, graph=None, columns_names=None, scaler=None, float_precision=3):
        raise NotImplementedError()
