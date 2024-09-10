import numpy as np
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from ruletree.utils.define import MODEL_TYPE_CLF, MODEL_TYPE_REG
from ruletree.utils.data_utils import get_info_gain, _get_info_gain


class ObliqueBivariateSplit:
    def __init__(
        self,
        H = 10, #number of orientations to generate
        model_type='clf',
        **kwargs
    ):
        self.kwargs =  kwargs
        self.model_type = model_type
     
        self.H = H
        self.D = None #number of features
        self.W = None #orientations matrix
        self.S = None #filter features matrix        
        self.oblq_clf = None #DecisionTreeClf/Reg used to find threshold of projected features
        
        #self.best_w = None #best orientation to choose from
        #self.best_b = None #best threshold/bias
        #self.best_feats_pair = None #best feature pairs
        
        self.feats = None
        self.coeff = None
        self.threshold = None
        
        
    def generate_orientations(self, H):
        angles = np.linspace(0, np.pi, H)  # np.pi is 180 degrees
        self.W = np.array([[np.cos(theta), np.sin(theta)] for theta in angles]).T
        
    def project_features(self, X, W):
        X_proj = X @ W
        return X_proj
    
    def best_threshold(self, X_proj, y, sample_weight=None, check_input=True):
        #for each orientation of the current feature pair, 
        #find the best threshold with a DT
    
        if self.model_type == MODEL_TYPE_CLF:
            clf = DecisionTreeClassifier(**self.kwargs) 
        elif self.model_type == MODEL_TYPE_REG:
            clf = DecisionTreeRegressor(**self.kwargs)     
        else:
            raise Exception('Unknown model %s' % self.model_type)
    
        clf.fit(X_proj, y, sample_weight=None, check_input=True)
        gain_clf = get_info_gain(clf) 
        
        return clf, gain_clf
    
    def transform(self, X):
        i, j = self.feats[0], self.feats[1]
        return X[:, [i, j]]
       
    def fit(self, X, y, sample_weight=None, check_input=True):
        self.D = X.shape[1] #number of features
        self.generate_orientations(self.H)
        best_gain = -float('inf')
        
        #iterate over pairs of features
        for i in range(self.D):
            for j in range(i + 1, self.D):
                X_pair = X[:, [i, j]]
                X_proj = self.project_features(X_pair, self.W)
                
                clf, clf_gain = self.best_threshold(X_proj, y, sample_weight=None, check_input=True)
               
                if clf_gain > best_gain: 
                    self.oblq_clf = clf
                    best_gain = clf_gain
        
                    #self.best_w = self.W[:, (clf.tree_.feature)[0]]
                    #self.best_b = clf.tree_.threshold[0]
                    #self.best_feats_pair = (i,j)
                        
                    self.coeff = self.W[:, (clf.tree_.feature)[0]]
                    self.threshold = clf.tree_.threshold[0]
                    self.feats = [i,j]
                
        return self
               
      
    def predict(self, X):
        X_pair = self.transform(X)
        X_proj = self.project_features(X_pair, self.W)
        return self.oblq_clf.predict(X_proj)
        
    def apply(self, X):
        X_pair = self.transform(X)
        X_proj = self.project_features(X_pair, self.W)
        return self.oblq_clf.apply(X_proj)
    

    
    
    
    
    
    
    
    
    
    
