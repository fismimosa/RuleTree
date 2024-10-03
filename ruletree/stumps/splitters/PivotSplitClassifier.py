from sklearn.base import BaseEstimator
from sklearn.tree import DecisionTreeClassifier

from ruletree.stumps.splitters.PivotSplit import PivotSplit

class PivotSplitClassifier(PivotSplit):
    def get_base_model(self):
        return DecisionTreeClassifier(**self.kwargs)