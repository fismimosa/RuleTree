from abc import abstractmethod, ABC

import numpy as np
from sklearn.base import TransformerMixin
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from RuleTree.base.RuleTreeBaseSplit import RuleTreeBaseSplit
from RuleTree.utils.data_utils import get_info_gain
from RuleTree.utils.define import MODEL_TYPE_CLF, MODEL_TYPE_REG, MODEL_TYPE_CLU


class ObliqueBivariateSplit(TransformerMixin, RuleTreeBaseSplit, ABC):
    """
    ObliqueBivariateSplit implements oblique splits using pairs of features.
    
    This approach finds optimal oblique decision boundaries by considering all pairs
    of features and multiple orientations for each pair. It selects the feature pair
    and orientation that provides the best split.
    
    Parameters
    ----------
    ml_task : str
        The machine learning task type (classification, regression, or clustering).
    n_orientations : int, default=10
        Number of orientations to consider for each feature pair.
    **kwargs : dict
        Additional parameters to pass to the base model.
    """
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

        self.feats = None
        self.coeff = None

    def generate_orientations(self, H):
        """
        Generates a matrix of orientations (angles) for projecting features.
        
        Parameters
        ----------
        H : int
            Number of orientations to generate.
        """
        angles = np.linspace(0, np.pi, H)  # np.pi is 180 degrees
        self.orientations_matrix = np.array([[np.cos(theta), np.sin(theta)] for theta in angles]).T

    def project_features(self, X, W):
        """
        Projects feature pairs onto different orientations.
        
        Parameters
        ----------
        X : array-like
            Feature matrix with two columns (a feature pair).
        W : array-like
            Matrix of orientations to project onto.
            
        Returns
        -------
        array-like
            Projected features.
        """
        X_proj = X @ W
        return X_proj

    def best_threshold(self, X_proj, y, sample_weight=None, check_input=True):
        """
        Finds the best threshold for splitting the projected features.
        
        Parameters
        ----------
        X_proj : array-like
            Projected feature matrix.
        y : array-like
            Target values.
        sample_weight : array-like, optional
            Sample weights.
        check_input : bool, default=True
            Whether to validate input.
            
        Returns
        -------
        tuple
            (best_model, gain) for the best split.
        """
        if self.ml_task == MODEL_TYPE_CLF:
            return self.__best_threshold_clf(X_proj, y, sample_weight, check_input)
        elif self.ml_task == MODEL_TYPE_REG:
            return self.__best_threshold_reg(X_proj, y, sample_weight, check_input)
        elif self.ml_task == MODEL_TYPE_CLU:
            return self.__best_threshold_clu(X_proj, y, sample_weight, check_input)

    def __best_threshold_clf(self, X_proj, y, sample_weight=None, check_input=True):
        """
        Finds the best threshold for classification tasks.
        
        Parameters
        ----------
        X_proj : array-like
            Projected feature matrix.
        y : array-like
            Target values.
        sample_weight : array-like, optional
            Sample weights.
        check_input : bool, default=True
            Whether to validate input.
            
        Returns
        -------
        tuple
            (model, information_gain) for the best split.
        """
        # for each orientation of the current feature pair,
        # find the best threshold with a DT

        clf = DecisionTreeClassifier(**self.kwargs)

        clf.fit(X_proj, y, sample_weight=None, check_input=True)
        gain_clf = get_info_gain(clf)

        return clf, gain_clf

    def __best_threshold_reg(self, X_proj, y, sample_weight=None, check_input=True):
        """
        Finds the best threshold for regression tasks.
        
        Parameters
        ----------
        X_proj : array-like
            Projected feature matrix.
        y : array-like
            Target values.
        sample_weight : array-like, optional
            Sample weights.
        check_input : bool, default=True
            Whether to validate input.
            
        Returns
        -------
        tuple
            (model, information_gain) for the best split.
        """
        # for each orientation of the current feature pair,
        # find the best threshold with a DT

        clf = DecisionTreeRegressor(**self.kwargs)

        clf.fit(X_proj, y, sample_weight=None, check_input=True)
        gain_clf = get_info_gain(clf)

        return clf, gain_clf

    def __best_threshold_clu(self, X_proj, y, sample_weight=None, check_input=True):
        """
        Finds the best threshold for clustering tasks.
        
        Parameters
        ----------
        X_proj : array-like
            Projected feature matrix.
        y : array-like
            Target values.
        sample_weight : array-like, optional
            Sample weights.
        check_input : bool, default=True
            Whether to validate input.
            
        Returns
        -------
        tuple
            Not implemented for clustering tasks.
        
        Raises
        ------
        NotImplementedError
            Always raises this exception as clustering is not implemented.
        """
        raise NotImplementedError()

    def transform(self, X):
        """
        Projects input data onto the selected orientation.
        
        Parameters
        ----------
        X : array-like
            Input feature matrix.
            
        Returns
        -------
        array-like
            Projected features.
        """
        i, j = self.feats
        X_proj = self.project_features(X[:, [i, j]], self.orientations_matrix)
        return X_proj

    def fit(self, X, y, sample_weight=None, check_input=True):
        """
        Fits the ObliqueBivariateSplit by finding the best feature pair and orientation.
        
        Parameters
        ----------
        X : array-like
            Input feature matrix.
        y : array-like
            Target values.
        sample_weight : array-like, optional
            Sample weights.
        check_input : bool, default=True
            Whether to validate input.
            
        Returns
        -------
        self
            The fitted splitter.
        """
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

                    self.coeff = self.orientations_matrix[:, (clf.tree_.feature)[0]]
                    self.feats = [i, j]

        return self
