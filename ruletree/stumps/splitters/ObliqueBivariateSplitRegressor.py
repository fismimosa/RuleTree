from sklearn.base import RegressorMixin
from sklearn.tree import DecisionTreeRegressor

from ruletree.stumps.splitters.ObliqueBivariateSplit import ObliqueBivariateSplit
from ruletree.utils.data_utils import get_info_gain


class ObliqueBivariateSplitRegressor(RegressorMixin, ObliqueBivariateSplit):
    
    def best_threshold(self, X_proj, y, sample_weight=None, check_input=True):
        #for each orientation of the current feature pair, 
        #find the best threshold with a DT
    
        clf = DecisionTreeRegressor(**self.kwargs)
    
        clf.fit(X_proj, y, sample_weight=None, check_input=True)
        gain_clf = get_info_gain(clf) 
        
        return clf, gain_clf