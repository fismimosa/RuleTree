from abc import abstractmethod, ABC

import numpy as np
from sklearn.base import TransformerMixin
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from scipy.linalg import norm

from RuleTree.base.RuleTreeBaseSplit import RuleTreeBaseSplit
from RuleTree.base.RuleTreeBaseStump import RuleTreeBaseStump
from RuleTree.utils.define import MODEL_TYPE_CLF, MODEL_TYPE_REG, MODEL_TYPE_CLU


class ObliqueHouseHolderSplit(TransformerMixin, RuleTreeBaseSplit, ABC):
    """
    ObliqueHouseHolderSplit implements oblique splits using Householder transformations.
    
    This approach finds oblique decision boundaries by applying a Householder reflection
    to the feature space, which can potentially better separate classes compared to
    axis-aligned splits.
    
    Parameters
    ----------
    ml_task : str
        The machine learning task type (classification, regression, or clustering).
    pca : estimator, default=None
        PCA object to use for dimension reduction. If None, a new PCA is created.
    max_oblique_features : int, default=2
        Maximum number of features to use for oblique split.
    tau : float, default=1e-4
        Threshold for variance explanation.
    **kwargs : dict
        Additional parameters to pass to the base model.
    """
    def __init__(
        self,
        ml_task,
        pca=None,
        max_oblique_features=2,
        tau=1e-4,
        **kwargs
    ):
        super(RuleTreeBaseSplit, RuleTreeBaseSplit).__init__(ml_task)
        self.ml_task = ml_task

        self.kwargs = kwargs
        self.pca = pca
        self.max_oblique_features = max_oblique_features
        self.tau = tau

        self.dominant_ev = None
        self.u_weights = None
        self.householder_matrix = None
        self.oblq_clf = None

        self.feats = None
        self.coeff = None

    def transform(self, X):
        """
        Applies the Householder transformation to the input data.
        
        Parameters
        ----------
        X : array-like
            Input feature matrix.
            
        Returns
        -------
        array-like
            Transformed feature matrix.
        """
        if self.householder_matrix is None:
            X_house = X
        else:
            X_house = X.dot(self.householder_matrix)
        return X_house

    def fit(self, X, y, sample_weight=None, check_input=True):
        """
        Fits the ObliqueHouseHolderSplit by computing the Householder transformation.
        
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
            The fitted splitter, or None if no split is found.
        """
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

        self.feats = list(np.nonzero(self.u_weights)[0])
        self.coeff = list(self.u_weights[self.feats])
        
        return self

    def get_base_model(self):
        """
        Returns the appropriate base model based on the machine learning task.
        
        Returns
        -------
        model : estimator
            The machine learning model to use for finding splits.
        """
        if self.ml_task == MODEL_TYPE_CLF:
            return DecisionTreeClassifier(**self.kwargs)
        elif self.ml_task == MODEL_TYPE_REG:
            return DecisionTreeRegressor(**self.kwargs)
        elif self.ml_task == MODEL_TYPE_CLU:
            raise NotImplementedError()
