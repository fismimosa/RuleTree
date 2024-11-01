from sklearn.base import BaseEstimator
from sklearn.tree import DecisionTreeClassifier

from ruletree.stumps.splitters.PivotSplit import PivotSplit, ObliquePivotSplit
from ruletree.stumps.splitters.PivotSplit import MultiplePivotSplit
from ruletree.stumps.splitters.ObliqueBivariateSplitClassifier import ObliqueBivariateSplitClassifier
from ruletree.stumps.splitters.ObliqueHouseHolderSplitClassifier import ObliqueHouseHolderSplitClassifier


class PivotSplitClassifier(PivotSplit):
    def get_base_model(self):
        return DecisionTreeClassifier(**self.kwargs)
    
class MultiplePivotSplitClassifier(MultiplePivotSplit):
    def get_base_model(self):
        return DecisionTreeClassifier(**self.kwargs)
    
class ObliquePivotSplitClassifier(ObliquePivotSplit):
    def __init__(self, oblique_split_type = 'householder', **kwargs): 
        super().__init__(**kwargs)  
        self.oblique_split_type = oblique_split_type
        
    def get_base_model(self):
        if self.oblique_split_type == 'householder':
           return ObliqueHouseHolderSplitClassifier(**self.kwargs)
        if self.oblique_split_type == 'bivariate':
           return ObliqueBivariateSplitClassifier(**self.kwargs)
