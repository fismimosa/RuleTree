import numpy as np
import pandas as pd
import sklearn
from sklearn.tree import DecisionTreeClassifier

from ruletree.base.RuleTreeBaseStump import RuleTreeBaseStump

from ruletree.stumps.splitters.PivotSplitClassifier import PivotSplitClassifier

from ruletree.utils.data_utils import get_info_gain, _get_info_gain, gini, entropy, _my_counts
from ruletree.utils.define import MODEL_TYPE_CLF


class PivotTreeStumpClassifier(DecisionTreeClassifier, RuleTreeBaseStump):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.is_categorical = False
        self.is_oblique = False
        self.is_pivotal = False
        self.kwargs = kwargs
        self.unique_val_enum = None
        self.threshold_original = None
        self.feature_original = None 
        self.coefficients = None
        
        
        self.pivot_split = PivotSplitClassifier(**kwargs)

    
        if 'criterion' not in kwargs or kwargs['criterion'] == "gini":
            self.impurity_fun = gini
        elif kwargs['criterion'] == "entropy":
            self.impurity_fun = entropy
        else:
            self.impurity_fun = kwargs['criterion']
            
            
    def get_params(self, deep=True):
        return self.kwargs
    
    def fit(self, X, y, distance_matrix, idx, sample_weight=None, check_input=True):
        dtypes = pd.DataFrame(X).infer_objects().dtypes
        self.numerical = dtypes[dtypes != np.dtype('O')].index
        self.categorical = dtypes[dtypes == np.dtype('O')].index
        best_info_gain = -float('inf')
        
        if len(self.numerical) > 0:
            self.pivot_split.fit(X[:, self.numerical], y, distance_matrix, idx, sample_weight=sample_weight, check_input=check_input)
            X_transform = self.pivot_split.transform(X[:, self.numerical])
            candidate_names = self.pivot_split.get_candidates_names()
            super().fit(X_transform, y)
            
            self.feature_original = [candidate_names[self.tree_.feature[0]], -2, -2]
            self.threshold_original = self.tree_.threshold
            self.is_pivotal = True
            
    
        return self
                
    def apply(self, X):
        X_transformed = self.pivot_split.transform(X[:, self.numerical])
        return super().apply(X_transformed)

    def get_feature(self):
        return self.feature_original[0]

    def get_thresholds(self):
        return self.threshold_original[0]

    def get_is_categorical(self):
        return self.is_categorical
    
    
        
