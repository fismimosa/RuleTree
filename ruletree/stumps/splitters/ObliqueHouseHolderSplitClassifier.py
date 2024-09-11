from sklearn.base import BaseEstimator
from sklearn.tree import DecisionTreeClassifier

from ruletree.stumps.splitters.ObliqueBivariateSplit import ObliqueBivariateSplit
from ruletree.stumps.splitters.ObliqueHouseHolderSplit import ObliqueHouseHolderSplit


class ObliqueHouseHolderSplitClassifier(ObliqueHouseHolderSplit):
    def get_base_model(self):
        return DecisionTreeClassifier(**self.kwargs)