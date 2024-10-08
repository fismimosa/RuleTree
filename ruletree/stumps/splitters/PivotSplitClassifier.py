from sklearn.base import BaseEstimator
from sklearn.tree import DecisionTreeClassifier

from ruletree.stumps.splitters.PivotSplit import PivotSplit
from ruletree.stumps.splitters.PivotSplit import MultiplePivotSplit


class PivotSplitClassifier(PivotSplit):
    def get_base_model(self):
        return DecisionTreeClassifier(**self.kwargs)
    
class MultiplePivotSplitClassifier(MultiplePivotSplit):
    def get_base_model(self):
        return DecisionTreeClassifier(**self.kwargs)