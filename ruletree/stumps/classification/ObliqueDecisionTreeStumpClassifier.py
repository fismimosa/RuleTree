from ruletree.base.RuleTreeBaseStump import RuleTreeBaseStump
from ruletree.stumps.classification.DecisionTreeStumpClassifier import DecisionTreeStumpClassifier
from ruletree.stumps.splitters.ObliqueBivariateSplit import ObliqueBivariateSplit
from ruletree.stumps.splitters.ObliqueHouseHolderSplit import ObliqueHouseHolderSplit
from ruletree.utils import MODEL_TYPE_CLF


class ObliqueDecisionTreeStumpClassifier(DecisionTreeStumpClassifier, RuleTreeBaseStump):
    def __init__(self,
                 oblique_split_type='householder',
                 pca=None,
                 max_oblique_features=2,
                 tau=1e-4,
                 n_orientations=10,
                 **kwargs):
        super().__init__(**kwargs)
        self.pca = pca
        self.max_oblique_features = max_oblique_features
        self.tau = tau
        self.n_orientations = n_orientations
        self.oblique_split_type = oblique_split_type

        if self.oblique_split_type == 'householder':
            self.oblique_split = ObliqueHouseHolderSplit(ml_task=MODEL_TYPE_CLF,
                                                         pca=self.pca,
                                                         max_oblique_features=self.max_oblique_features,
                                                         tau=self.tau,
                                                         **kwargs)

        if self.oblique_split_type == 'bivariate':
            self.oblique_split = ObliqueBivariateSplit(ml_task=MODEL_TYPE_CLF, n_orientations=self.n_orientations, **kwargs)

    def fit(self, X, y, sample_weight=None, check_input=True):
        self.feature_analysis(X, y)
        self.num_pre_transformed = self.numerical
        self.cat_pre_transformed = self.categorical
        best_info_gain = -float('inf')

        if len(self.numerical) > 0:
            self.oblique_split.fit(X[:, self.numerical], y, sample_weight=sample_weight, check_input=check_input)
            X_transform = self.oblique_split.transform(X[:, self.numerical])
            super().fit(X_transform, y, sample_weight=sample_weight, check_input=check_input)

            self.feature_original = [self.oblique_split.feats, -2, -2]
            self.coefficients = self.oblique_split.coeff
            self.threshold_original = self.tree_.threshold
            self.is_oblique = True

        return self

    def apply(self, X):
        X_transform = self.oblique_split.transform(X[:, self.num_pre_transformed])
        return super().apply_sk(X_transform) #otherwise we need to "personalize" the apply function

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
             "feature_idx": self.feature_original[0], #list of feats
             "threshold": self.threshold_original[0], #thr
             "coefficients" : self.coefficients, #coefficients
             "is_categorical": self.is_categorical,
             "samples": self.n_node_samples[0]
         }
         
         #round coefficients here
         rule['coefficients'] = [
                                str(coeff) if float_precision is None else round(float(coeff), float_precision) 
                                for coeff in rule['coefficients']
                                ]
         
         feat_name = " + ".join(f"{coeff} * X_{idx}" for coeff, idx in zip(rule['coefficients'], rule['feature_idx']))
        
         if columns_names is not None:
             feat_name = "_".join(columns_names[idx] for idx in self.feature_original[0]) #check this for feat names
         rule["feature_name"] = feat_name
         
         if scaler is not None:
             #TODO
             raise NotImplementedError()
             pass
         
         comparison = "<=" if not self.is_categorical else "="
         not_comparison = ">" if not self.is_categorical else "!="
         rounded_value = str(rule["threshold"]) if float_precision is None else round(rule["threshold"], float_precision)
         
         if scaler is not None:
             #TODO
             raise NotImplementedError()
             pass
         
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
