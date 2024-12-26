from RuleTree.base.RuleTreeBaseStump import RuleTreeBaseStump
from RuleTree.stumps.classification.DecisionTreeStumpClassifier import DecisionTreeStumpClassifier
from RuleTree.stumps.splitters.MultipleObliquePivotSplit import MultipleObliquePivotSplit
from RuleTree.utils import MODEL_TYPE_CLF
from sklearn.metrics.pairwise import pairwise_distances
import copy
import numpy as np


class MultipleObliquePivotTreeStumpClassifier(DecisionTreeStumpClassifier, RuleTreeBaseStump):
    def __init__(self, 
                 oblique_split_type='householder',
                 pca=None,
                 max_oblique_features=2,
                 tau=1e-4,
                 n_orientations=10,
                 **kwargs):
        super().__init__(**kwargs)
        self.distance_measure = None
        self.pca = pca
        self.max_oblique_features = max_oblique_features
        self.tau = tau
        self.n_orientations = n_orientations
        self.oblique_split_type = oblique_split_type
        
        if oblique_split_type == 'householder':
            self.multi_oblique_pivot_split = MultipleObliquePivotSplit(ml_task=MODEL_TYPE_CLF,
                                                         pca=self.pca,
                                                         max_oblique_features=self.max_oblique_features,
                                                         tau=self.tau,
                                                         **kwargs)

        if oblique_split_type == 'bivariate':
            self.multi_oblique_pivot_split = MultipleObliquePivotSplit(ml_task=MODEL_TYPE_CLF,
                                                         n_orientations=self.n_orientations,
                                                         **kwargs)
        

    def fit(self, X, y, distance_matrix, distance_measure, idx, sample_weight=None, check_input=True):
        self.feature_analysis(X, y)
        self.num_pre_transformed = self.numerical
        self.cat_pre_transformed = self.categorical

        if len(self.numerical) > 0:
            self.multi_oblique_pivot_split.fit(X[:, self.numerical], y, distance_matrix, distance_measure, idx,
                                       sample_weight=sample_weight, check_input=check_input)
            X_transform = self.multi_oblique_pivot_split.transform(X[:, self.numerical], distance_measure)
            candidate_names = self.multi_oblique_pivot_split.get_candidates_names()
            super().fit(X_transform, y, sample_weight=sample_weight, check_input=check_input)

            self.feature_original = [(candidate_names[0], candidate_names[1]), -2, -2]
            self.threshold_original = self.tree_.threshold
            self.is_pivotal = True
            self.is_oblique = True
            self.distance_measure = distance_measure
            self.X_best_tup = self.multi_oblique_pivot_split.best_tup
            

        return self

    def apply(self, X):
        #dist_to_p0 = pairwise_distances(X, self.best_tup[0].reshape(1, -1), metric=distance_measure).flatten()
        #dist_to_p1 = pairwise_distances(X, self.best_tup[1].reshape(1, -1), metric=distance_measure).flatten()
        #dist_binary = np.where(dist_to_p0 < dist_to_p1, 0, 1).reshape(-1, 1)
        
        #we need to assign self.best_tup[0], self.best_tup[1] to self.multi_pivot_split
        
        dist_to_p0 = pairwise_distances(X[:, self.num_pre_transformed], 
                                        self.X_best_tup [0].reshape(1, -1), 
                                        metric=self.distance_measure).flatten()
        
        dist_to_p1 = pairwise_distances(X[:, self.num_pre_transformed], 
                                        self.X_best_tup [1].reshape(1, -1),
                                        metric=self.distance_measure).flatten()
        
        dist_binary = np.where(dist_to_p0 < dist_to_p1, 0, 1).reshape(-1, 1)
        
        X_transformed = dist_binary
        y_pred = (np.ones(X_transformed.shape[0]) * 2)
        X_feature = X_transformed[:, 0]
        y_pred[X_feature <= self.threshold_original[0]] = 1
        
        return y_pred
        
        
        #X_transformed = self.multi_pivot_split.transform(X[:, self.num_pre_transformed], self.distance_measure)
        #y_pred = (np.ones(X_transformed.shape[0]) * 2)
        #X_feature = X_transformed[:, self.tree_.feature[0]]
        #y_pred[X_feature <= self.tree_.threshold[0]] = 1
        #return y_pred
        
        #return super().apply_sk(X_transformed)
        
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
             "feature_idx": self.feature_original[0], #tuple of instances idx
             "threshold": self.threshold_original[0], #thr
             "coefficients" : self.coefficients, #coefficients
             "is_categorical": self.is_categorical,
             "samples": self.n_node_samples[0]
         }
              
         feat_name = " ".join(f"P_{idx}" for idx in list(rule['feature_idx'])) #list of sitrings
        
         #if columns_names is not None:
         #    feat_name = "_".join(columns_names[idx] for idx in self.feature_original[0]) #check this for feat names
         rule["feature_name"] = feat_name
         
         #if scaler is not None:
         #    #TODO
         #    raise NotImplementedError()
             
         
         comparison = f"closer to {rule['feature_idx'][0]}" if not self.is_categorical else "="
         not_comparison = f"closer to {rule['feature_idx'][1]}" if not self.is_categorical else "!="
         rounded_value = str(rule["threshold"]) if float_precision is None else round(rule["threshold"], float_precision)
         
         #if scaler is not None:
         #    #TODO
         #    raise NotImplementedError()
         #    pass
         
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
        

    @classmethod
    def dict_to_node(cls, node_dict, X = None):
        self = cls()
        self.feature_original = np.zeros(3, dtype=object)
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
           self.X_best_tup = (X[int(node_dict["feature_idx"][0])], 
                              X[int(node_dict["feature_idx"][1])])
           
           
        return self
                            

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
