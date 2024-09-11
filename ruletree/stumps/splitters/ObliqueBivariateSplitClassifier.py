from sklearn.base import ClassifierMixin
from sklearn.tree import DecisionTreeClassifier

from ruletree.stumps.splitters.ObliqueBivariateSplit import ObliqueBivariateSplit
from ruletree.utils.data_utils import get_info_gain


class ObliqueBivariateSplitClassifier(ClassifierMixin, ObliqueBivariateSplit):

    def best_threshold(self, X_proj, y, sample_weight=None, check_input=True):
        # for each orientation of the current feature pair,
        # find the best threshold with a DT

        clf = DecisionTreeClassifier(**self.kwargs)

        clf.fit(X_proj, y, sample_weight=None, check_input=True)
        gain_clf = get_info_gain(clf)

        return clf, gain_clf
