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
        min_samples_leaf=3,
        min_samples_split=5,
        tau=1e-4,
        model_type='clf',
        clf_impurity: str = 'gini',
        reg_impurity: str = 'squared_error',
        random_state=None,
    ):
        self.pca = pca
        self.max_oblique_features = max_oblique_features
        self.min_samples_leaf = min_samples_leaf
        self.min_samples_split = min_samples_split
        self.tau = tau
        self.model_type = model_type
        self.clf_impurity = clf_impurity
        self.reg_impurity = reg_impurity

        self.dominant_ev = None
        self.u_weights = None
        self.householder_matrix = None
        self.oblq_clf = None
        self.random_state = random_state

    def transform(self, X):
        X_house = X.dot(self.householder_matrix)
        return X_house

    def fit(self, X, y):

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
            self.oblq_clf = DecisionTreeClassifier(
                max_depth=1,
                criterion=self.clf_impurity,
                min_samples_leaf=self.min_samples_leaf,
                min_samples_split=self.min_samples_split,
                random_state=self.random_state,
            )
        elif self.model_type == MODEL_TYPE_REG:
            self.oblq_clf = DecisionTreeRegressor(
                max_depth=1,
                criterion=self.reg_impurity,
                min_samples_leaf=self.min_samples_leaf,
                min_samples_split=self.min_samples_split,
                random_state=self.random_state,
            )
        else:
            raise Exception('Unknown model %s' % self.model_type)
        self.oblq_clf.fit(X_house, y)

    def predict(self, X):
        X_house = self.transform(X)
        return self.oblq_clf.predict(X_house)

    def apply(self, X):
        X_house = self.transform(X)
        return self.oblq_clf.apply(X_house)
