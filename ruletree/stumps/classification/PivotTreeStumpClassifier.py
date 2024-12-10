from ruletree.base.RuleTreeBaseStump import RuleTreeBaseStump
from ruletree.stumps.classification.DecisionTreeStumpClassifier import DecisionTreeStumpClassifier
from ruletree.stumps.splitters.PivotSplit import PivotSplit
from ruletree.utils import MODEL_TYPE_CLF
from sklearn.metrics.pairwise import pairwise_distances
import numpy as np
import copy
import warnings



class PivotTreeStumpClassifier(DecisionTreeStumpClassifier, RuleTreeBaseStump):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.pivot_split = PivotSplit(ml_task=MODEL_TYPE_CLF, **kwargs)
        self.distance_measure = None
        self.split_instance = None

    def fit(self, X, y, distance_matrix, distance_measure, idx, sample_weight=None, check_input=True):
        self.feature_analysis(X, y)
        self.num_pre_transformed = self.numerical
        self.cat_pre_transformed = self.categorical
       
        if len(self.numerical) > 0:
            self.pivot_split.fit(X[:, self.numerical], y, distance_matrix, distance_measure, idx,
                                 sample_weight=sample_weight, check_input=check_input)
            X_transform = self.pivot_split.transform(X[:, self.numerical], distance_measure)
            candidate_names = self.pivot_split.get_candidates_names()
            super().fit(X_transform, y, sample_weight=sample_weight, check_input=check_input)

            self.feature_original = [f'{candidate_names[self.tree_.feature[0]]}', -2, -2]
            self.threshold_original = self.tree_.threshold
            self.is_pivotal = True
            
            self.distance_measure = distance_measure
            self.X_split_instance = self.pivot_split.X_candidates[self.tree_.feature[0]]
        
            
        
        return self

    def apply(self, X):
        #X_transformed = self.pivot_split.transform(X[:, self.num_pre_transformed], self.distance_measure)
        #y_pred = (np.ones(X_transformed.shape[0]) * 2)
        #X_feature = X_transformed[:, self.tree_.feature[0]]
        #y_pred[X_feature <= self.tree_.threshold[0]] = 1
        
        X_transformed = pairwise_distances(X[:, self.num_pre_transformed], 
                                           self.X_split_instance.reshape(1, -1),
                                           metric=self.distance_measure)
        
        y_pred = (np.ones(X_transformed.shape[0]) * 2)
        X_feature = X_transformed[:,  0]
        y_pred[X_feature <= self.threshold_original[0]] = 1
        
        return y_pred
        
        #return super().apply_sk(X_transformed)
        
        
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
            NotImplementedError()
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
    
    @classmethod
    def dict_to_node(cls, node_dict, X = None):
        self = cls()
        self.feature_original = np.zeros(3, dtype=int)
        self.threshold_original = np.zeros(3)
        self.n_node_samples = np.zeros(3, dtype=int)



        self.feature_original[0] = node_dict["feature_idx"]
        self.threshold_original[0] = node_dict["threshold"]
        self.n_node_samples[0] = node_dict["samples"]
        self.is_categorical = node_dict["is_categorical"]

        args = copy.deepcopy(node_dict["args"])
        self.is_oblique = args.pop("is_oblique")
        self.is_pivotal = args.pop("is_pivotal")
        self.unique_val_enum = args.pop("unique_val_enum")
        self.coefficients = args.pop("coefficients")
        self.kwargs = args
        
        self.distance_measure = args.pop("distance_measure")
        self.num_pre_transformed = args.pop("num_pre_transformed")
        self.cat_pre_transformed = args.pop("cat_pre_transformed")
        
        #X acts as a reference dataset for the instance id
        if X is not None:
            self.X_split_instance = X[int(node_dict["feature_idx"])]
        
      
        
        ##
    

        return self

    def export_graphviz(self, graph=None, columns_names=None, scaler=None, float_precision=3):
        raise NotImplementedError()
