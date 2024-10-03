from abc import abstractmethod, ABC

import numpy as np
from sklearn.base import TransformerMixin
from sklearn.metrics.pairwise import pairwise_distances

from ruletree.base.RuleTreeBaseStump import RuleTreeBaseStump
from ruletree.utils.define import MODEL_TYPE_CLF, MODEL_TYPE_REG


class PivotSplit(TransformerMixin, ABC):
    def __init__(
        self,
        **kwargs
    ):
        self.kwargs = kwargs
        self.X_candidates = None
        self.is_categorical = False
        
        self.discriminative_names = None
        self.descriptive_names = None
        self.candidates_names = None
        self.is_pivotal = False
        
        
    @abstractmethod
    def get_base_model(self):
        pass
    
    def compute_descriptive(self, sub_matrix):
        row_sums = sub_matrix.sum(axis=1)
        medoid_index = np.argmin(row_sums)
        return medoid_index
    
    def compute_discriminative(self, sub_matrix, y, sample_weight=None, check_input=True):
       disc = self.get_base_model()
       disc.fit(sub_matrix, y, sample_weight=sample_weight, check_input = check_input) # Unpack kwargs here
       discriminative_id = disc.tree_.feature[0]
       return discriminative_id


    def fit(self, X, y, distance_matrix, idx, 
            sample_weight=None, check_input=True):
        
        
        sub_matrix = distance_matrix[idx][:,idx]
        local_idx = np.arange(len(y))
    
        
        local_descriptives = []
        local_discriminatives = []
        local_candidates = []
        
        
        for label in set(y):
            idx_label = np.where(y == label)[0]
            local_idx_label = local_idx[idx_label]
            sub_matrix_label = sub_matrix[:,idx_label]
            
            disc_id = self.compute_discriminative(sub_matrix_label, y, sample_weight=sample_weight, check_input=check_input)           
            disc_idx = local_idx_label[disc_id]
           
            desc_id = self.compute_descriptive(sub_matrix_label[idx_label])
            desc_idx =  local_idx_label[desc_id]
            
            
            local_discriminatives += [disc_idx]
            local_descriptives += [desc_idx]
            
        local_candidates = local_descriptives + local_discriminatives
            
          
        self.X_candidates = X[local_candidates]
        
        self.discriminative_names = idx[local_discriminatives]
        self.descriptive_names = idx[local_descriptives]
        self.candidates_names = idx[local_candidates]
        
        
    def transform(self, X):
        return pairwise_distances(X, self.X_candidates)
    
    def get_candidates_names(self):
        return self.candidates_names
    
    def get_descriptive_names(self):
        return self.descriptive_names
    
    def get_discriminative_names(self):
        return self.discriminative_names
    

    
    
        

        
        
           
    

            
            
        
        
        
      
 