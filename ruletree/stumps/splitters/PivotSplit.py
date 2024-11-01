from abc import abstractmethod, ABC

import numpy as np
from itertools import chain
from sklearn.base import TransformerMixin
from sklearn.metrics.pairwise import pairwise_distances
from ruletree.base.RuleTreeBaseStump import RuleTreeBaseStump
from ruletree.utils.define import MODEL_TYPE_CLF, MODEL_TYPE_REG
import itertools
from ruletree.utils.data_utils import get_info_gain

class PivotSplit(TransformerMixin, ABC):
    def __init__(
        self,
        **kwargs
    ):
        self.kwargs = kwargs
        self.distance_measure = None
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
       disc.fit(sub_matrix, y, sample_weight=sample_weight, check_input = check_input)
       discriminative_id = disc.tree_.feature[0]
       return discriminative_id


    def fit(self, X, y, distance_matrix, distance_measure, idx,
            sample_weight=None, check_input=True):
    
        sub_matrix = distance_matrix
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
            
            if isinstance(disc_idx, (list, np.ndarray)):
                local_discriminatives += disc_idx.flatten().tolist() if isinstance(disc_idx, np.ndarray) else list(disc_idx)
            else:
                local_discriminatives += [disc_idx]
                
            local_descriptives += [desc_idx]
            
        
        local_candidates = local_descriptives + local_discriminatives
      
          
        self.X_candidates = X[local_candidates]
        
        self.discriminative_names = idx[local_discriminatives]
        self.descriptive_names = idx[local_descriptives]
        self.candidates_names = idx[local_candidates]
        
        
    def transform(self, X, distance_measure = 'euclidean'):
        return pairwise_distances(X, self.X_candidates, metric = distance_measure)
    
    def get_candidates_names(self):
        return self.candidates_names
    
    def get_descriptive_names(self):
        return self.descriptive_names
    
    def get_discriminative_names(self):
        return self.discriminative_names
    
    
class ObliquePivotSplit(PivotSplit, ABC):
    def __init__(
        self,
        **kwargs
    ):
        super().__init__(**kwargs)    
        
    @abstractmethod
    def get_base_model(self):
        pass
    
    def compute_discriminative(self, sub_matrix, y, sample_weight=None, check_input=True):
       disc = self.get_base_model()
       disc.fit(sub_matrix, y, sample_weight=sample_weight, check_input = check_input)
       discriminative_id = disc.feats
       return (discriminative_id)


class MultiplePivotSplit(PivotSplit, ABC):
    def __init__(
        self,
        **kwargs
    ):
        super().__init__(**kwargs)    
        self.best_tup = None
        self.best_tup_name = None
        self.best_gain = -float('inf') 
        
        
    @abstractmethod
    def get_base_model(self):
        pass
    
    def find_best_tuple(self, X, y, distance_measure = 'euclidean', sample_weight=None, check_input=True):
       two_tuples = list(itertools.combinations(range(0,len(self.X_candidates)), 2))
       
       for tup in two_tuples:
           disc = self.get_base_model()
           p1, p2 = self.X_candidates[np.array(tup)]
           name_p1, name_p2 = self.candidates_names[np.array(tup)]
           
           dist_to_p0 = pairwise_distances(X, p1.reshape(1, -1), metric = distance_measure).flatten()
           dist_to_p1 = pairwise_distances(X, p2.reshape(1, -1), metric = distance_measure).flatten()
                       
           dist_binary = np.where(dist_to_p0 < dist_to_p1, 0, 1).reshape(-1,1)
           disc.fit(dist_binary,y)
           gain_disc = get_info_gain(disc)
           
           
           if gain_disc > self.best_gain:
               self.best_gain = gain_disc
               self.best_tup = self.X_candidates[np.array(tup)]
               self.best_tup_name = self.candidates_names[np.array(tup)]
    
    def fit(self, X, y, distance_matrix, distance_measure, idx, 
            sample_weight=None, check_input=True):
        
        super().fit(X, y, distance_matrix, distance_measure, idx,sample_weight=sample_weight, check_input=check_input)
        self.find_best_tuple(X, y, distance_measure = distance_measure, sample_weight=sample_weight, check_input=check_input)   
       
    def transform(self, X, distance_measure = 'euclidean'):
        dist_to_p0 = pairwise_distances(X, self.best_tup[0].reshape(1, -1), metric = distance_measure).flatten()
        dist_to_p1 = pairwise_distances(X, self.best_tup[1].reshape(1, -1), metric = distance_measure).flatten()
        dist_binary = np.where(dist_to_p0 < dist_to_p1, 0, 1).reshape(-1,1)
        return dist_binary
    
    def get_best_tup_names(self):
        return self.best_tup_name
    
    

            
            
        
        
        
      
 

    
    
        

        
        
           
    

            
            
        
        
        
      
 
