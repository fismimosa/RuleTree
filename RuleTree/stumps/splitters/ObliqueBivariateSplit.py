from abc import abstractmethod, ABC

import numpy as np
from sklearn.base import TransformerMixin
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from RuleTree.base.RuleTreeBaseSplit import RuleTreeBaseSplit
from RuleTree.utils.data_utils import get_info_gain
from RuleTree.utils.define import MODEL_TYPE_CLF, MODEL_TYPE_REG, MODEL_TYPE_CLU


class ObliqueBivariateSplit(TransformerMixin, RuleTreeBaseSplit, ABC):
    def __init__(
            self,
            ml_task,
            n_orientations=10,  # number of orientations to generate
            **kwargs
    ):
        super(RuleTreeBaseSplit, RuleTreeBaseSplit).__init__(ml_task)

        self.kwargs = kwargs
        self.ml_task = ml_task

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
        #self.threshold = None

    def generate_orientations(self, H):
        angles = np.linspace(0, np.pi, H)  # np.pi is 180 degrees
        self.orientations_matrix = np.array([[np.cos(theta), np.sin(theta)] for theta in angles]).T

    def project_features(self, X, W):
        X_proj = X @ W
        return X_proj

    def best_threshold(self, X_proj, y, sample_weight=None, check_input=True):
        if self.ml_task == MODEL_TYPE_CLF:
            return self.__best_threshold_clf(X_proj, y, sample_weight, check_input)
        elif self.ml_task == MODEL_TYPE_REG:
            return self.__best_threshold_reg(X_proj, y, sample_weight, check_input)
        elif self.ml_task == MODEL_TYPE_CLU:
            return self.__best_threshold_clu(X_proj, y, sample_weight, check_input)

    def __best_threshold_clf(self, X_proj, y, sample_weight=None, check_input=True):
        # for each orientation of the current feature pair,
        # find the best threshold with a DT

        clf = DecisionTreeClassifier(**self.kwargs)

        clf.fit(X_proj, y, sample_weight=None, check_input=True)
        gain_clf = get_info_gain(clf)

        return clf, gain_clf

    def __best_threshold_reg(self, X_proj, y, sample_weight=None, check_input=True):
        # for each orientation of the current feature pair,
        # find the best threshold with a DT

        clf = DecisionTreeRegressor(**self.kwargs)

        clf.fit(X_proj, y, sample_weight=None, check_input=True)
        gain_clf = get_info_gain(clf)

        return clf, gain_clf

    def __best_threshold_clu(self, X_proj, y, sample_weight=None, check_input=True):
        raise NotImplementedError()

    def transform(self, X):
        i, j = self.feats
        X_proj = self.project_features(X[:, [i, j]], self.orientations_matrix)
        return X_proj

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
                    self.feats = [i, j]
                    #self.threshold = clf.tree_.threshold[0]

        return self

    #def predict(self, X):
    #    X_proj = self.transform(X)
    #    return self.oblq_clf.predict(X_proj)

    #def apply(self, X):
    #    X_proj = self.transform(X)
    #    return self.oblq_clf.apply(X_proj)
