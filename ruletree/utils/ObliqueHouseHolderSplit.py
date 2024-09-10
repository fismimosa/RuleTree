import numpy as np
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from scipy.linalg import norm

from ruletree.utils.define import MODEL_TYPE_CLF, MODEL_TYPE_REG


class ObliqueHouseHolderSplit:
    def __init__(
        self,
        pca=None,
        max_oblique_features=2,
        tau=1e-4,
        model_type='clf',
        **kwargs
    ):
        self.kwargs = kwargs
        self.pca = pca
        self.max_oblique_features = max_oblique_features
        self.tau = tau
        self.model_type = model_type
       
        self.dominant_ev = None
        self.u_weights = None
        self.householder_matrix = None
        self.oblq_clf = None

        self.feats = None
        self.coeff = None
        self.threshold = None

    def transform(self, X):
        X_house = X.dot(self.householder_matrix)
        return X_house

    def fit(self, X, y, sample_weight=None, check_input=True):
        
        n_features = X.shape[1]

        if self.pca is None:
            self.pca = PCA(n_components=1)
            self.pca.fit(X)

        self.dominant_ev = self.pca.components_[0]
        I = np.diag(np.ones(n_features))

        diff_w_means = np.sqrt(((I - self.dominant_ev) ** 2).sum(axis=1))

        if (diff_w_means > self.tau).sum() == 0:
            print("No variance to explain.")
            return None

        idx_max_diff = np.argmax(diff_w_means)
        e_vector = np.zeros(n_features)
        e_vector[idx_max_diff] = 1.0
        self.u_weights = (e_vector - self.dominant_ev) / norm(e_vector - self.dominant_ev)

        if self.max_oblique_features < n_features:
            idx_w = np.argpartition(np.abs(self.u_weights), -self.max_oblique_features)[-self.max_oblique_features:]
            u_weights_new = np.zeros(n_features)
            u_weights_new[idx_w] = self.u_weights[idx_w]
            self.u_weights = u_weights_new

        self.householder_matrix = I - 2 * self.u_weights[:, np.newaxis].dot(self.u_weights[:, np.newaxis].T)

        X_house = self.transform(X)

        if self.model_type == MODEL_TYPE_CLF:
            self.oblq_clf = DecisionTreeClassifier(**self.kwargs) 
        elif self.model_type == MODEL_TYPE_REG:
            self.oblq_clf = DecisionTreeRegressor(**self.kwargs)     
        else:
            raise Exception('Unknown model %s' % self.model_type)
        self.oblq_clf.fit(X_house, y)
        
        self.feats = list(np.nonzero(self.u_weights)[0])
        self.coeff = list(self.u_weights[self.feats])
        self.threshold = self.oblq_clf.tree_.threshold[0]
        
        return self

    def predict(self, X):
        X_house = self.transform(X)
        return self.oblq_clf.predict(X_house)

    def apply(self, X):
        X_house = self.transform(X)
        return self.oblq_clf.apply(X_house)
