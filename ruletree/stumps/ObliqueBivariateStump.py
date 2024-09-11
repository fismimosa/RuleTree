from abc import abstractmethod

import numpy as np

from ruletree.base.RuleTreeBaseStump import RuleTreeBaseStump

class ObliqueBivariateStump(RuleTreeBaseStump):
    def __init__(
            self,
            n_orientations=10,  # number of orientations to generate
            **kwargs
    ):
        self.kwargs = kwargs

        self.n_orientations = n_orientations
        self.n_features = None  # number of features
        self.orientations_matrix = None  # orientations matrix
        self.feature_filters_matrix = None  # filter features matrix
        self.oblq_clf = None  # DecisionTreeClf/Reg used to find threshold of projected features

        # self.best_w = None #best orientation to choose from
        # self.best_b = None #best threshold/bias
        # self.best_feats_pair = None #best feature pairs

        self.feats = None
        self.coeff = None
        self.threshold = None

    def generate_orientations(self, H):
        angles = np.linspace(0, np.pi, H)  # np.pi is 180 degrees
        self.orientations_matrix = np.array([[np.cos(theta), np.sin(theta)] for theta in angles]).T

    def project_features(self, X, W):
        X_proj = X @ W
        return X_proj

    @abstractmethod
    def best_threshold(self, X_proj, y, sample_weight=None, check_input=True):
        pass

    def transform(self, X):
        i, j = self.feats[0], self.feats[1]
        return X[:, [i, j]]

    def fit(self, X, y, sample_weight=None, check_input=True):
        self.n_features = X.shape[1]  # number of features
        self.generate_orientations(self.n_orientations)
        best_gain = -float('inf')

        # iterate over pairs of features
        for i in range(self.n_features):
            for j in range(i + 1, self.n_features):
                X_pair = X[:, [i, j]]
                X_proj = self.project_features(X_pair, self.orientations_matrix)

                clf, clf_gain = self.best_threshold(X_proj, y, sample_weight=None, check_input=True)

                if clf_gain > best_gain:
                    self.oblq_clf = clf
                    best_gain = clf_gain

                    # self.best_w = self.W[:, (clf.tree_.feature)[0]]
                    # self.best_b = clf.tree_.threshold[0]
                    # self.best_feats_pair = (i,j)

                    self.coeff = self.orientations_matrix[:, (clf.tree_.feature)[0]]
                    self.threshold = clf.tree_.threshold[0]
                    self.feats = [i, j]

        return self

    def predict(self, X):
        X_pair = self.transform(X)
        X_proj = self.project_features(X_pair, self.orientations_matrix)
        return self.oblq_clf.predict(X_proj)

    def apply(self, X):
        X_pair = self.transform(X)
        X_proj = self.project_features(X_pair, self.orientations_matrix)
        return self.oblq_clf.apply(X_proj)












