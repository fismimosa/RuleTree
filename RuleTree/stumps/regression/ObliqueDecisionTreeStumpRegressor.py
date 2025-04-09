import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_poisson_deviance
from sklearn.tree import DecisionTreeRegressor

from RuleTree.base.RuleTreeBaseStump import RuleTreeBaseStump


class ObliqueDecisionTreeStumpRegressor(DecisionTreeRegressor, RuleTreeBaseStump):
    """
    Oblique Decision Tree Stump Regressor for regression tasks.
    
    This class extends DecisionTreeRegressor to create decision stumps with oblique splits,
    which allows for multivariate splits using linear combinations of features rather than
    splits on a single feature.
    
    Inherits from:
        DecisionTreeRegressor: sklearn's standard decision tree for regression
        RuleTreeBaseStump: base class for rule tree stumps
    """

    def get_rule(self, columns_names=None, scaler=None, float_precision=3):
        """
        Get a textual representation of the rule defined by this stump.
        
        Parameters:
        -----------
        columns_names : list, optional
            Names of the columns/features for creating readable rules
        scaler : object, optional
            Scaler object for transforming numeric values if the data was scaled
        float_precision : int, optional (default=3)
            Number of decimal places for rounding numeric values in the rule
            
        Returns:
        --------
        dict
            Rule representation not yet implemented for this class
            
        Raises:
        -------
        NotImplementedError
            This method is not yet implemented
        """
        raise NotImplementedError()

    def node_to_dict(self, col_names):
        """
        Convert the node to a dictionary representation.
        
        Parameters:
        -----------
        col_names : list
            List of column names
            
        Returns:
        --------
        dict
            Dictionary representation of the node
            
        Raises:
        -------
        NotImplementedError
            This method is not yet implemented
        """
        raise NotImplementedError()

    def export_graphviz(self, graph=None, columns_names=None, scaler=None, float_precision=3):
        """
        Export the stump as a graphviz visualization.
        
        Parameters:
        -----------
        graph : pydot.Graph, optional
            Existing graph object to add nodes to
        columns_names : list, optional
            Names of the columns/features
        scaler : object, optional
            Scaler object for transforming numeric values
        float_precision : int, optional (default=3)
            Number of decimal places for rounding numeric values
            
        Returns:
        --------
        pydot.Graph
            Graphviz representation of the stump
            
        Raises:
        -------
        NotImplementedError
            This method is not yet implemented
        """
        raise NotImplementedError()

    def __init__(self, **kwargs):
        """
        Initialize the Oblique Decision Tree Stump Regressor.
        
        Parameters:
        -----------
        **kwargs : dict
            Parameters to be passed to sklearn's DecisionTreeRegressor
        """
        super().__init__(**kwargs)
        self.oblique_split = None
        self.is_categorical = False
        self.kwargs = kwargs
        self.unique_val_enum = None
        self.threshold_original = None
        self.feature_original = None

        if kwargs['criterion'] == "squared_error":
            self.impurity_fun = mean_squared_error
        elif kwargs['criterion'] == "friedman_mse":
            raise Exception("not implemented")  # TODO: implement
        elif kwargs['criterion'] == "absolute_error":
            self.impurity_fun = mean_absolute_error
        elif kwargs['criterion'] == "poisson":
            self.impurity_fun = mean_poisson_deviance
        else:
            self.impurity_fun = kwargs['criterion']

    def __impurity_fun(self, **x):
        """
        Calculate impurity for a node.
        
        Parameters:
        -----------
        **x : dict
            Keyword arguments containing at least y_true
            
        Returns:
        --------
        float
            Impurity value (0 if y_true is empty)
        """
        return self.impurity_fun(**x) if len(x["y_true"]) > 0 else 0  # TODO: check

    def get_params(self, deep=True):
        """
        Get the parameters of this estimator.
        
        Parameters:
        -----------
        deep : bool, default=True
            If True, will return the parameters for this estimator and
            contained sub-objects that are estimators.
            
        Returns:
        --------
        dict
            Parameter names mapped to their values
        """
        return self.kwargs

    def fit(self, X, y, idx=None, context=None, sample_weight=None, check_input=True):
        """
        Build an oblique decision tree stump regressor from the training data.
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            The training input samples
        y : array-like of shape (n_samples,)
            The target values
        idx : slice or array-like, optional
            Indices of samples to use for training
        context : object, optional
            Additional context information (not used)
        sample_weight : array-like of shape (n_samples,), optional
            Sample weights
        check_input : bool, default=True
            Whether to check the input parameters
            
        Returns:
        --------
        self : object
            Fitted estimator
        """
        if idx is None:
            idx = slice(None)
        X = X[idx]
        y = y[idx]

        dtypes = pd.DataFrame(X).infer_objects().dtypes
        self.numerical = dtypes[dtypes != np.dtype('O')].index
        self.categorical = dtypes[dtypes == np.dtype('O')].index

        if len(self.numerical) > 0:
            self.oblique_split.fit(X[:, self.numerical], y, sample_weight=sample_weight, check_input=check_input)
            X_transform = self.oblique_split.transform(X[:, self.numerical])
            super().fit(X_transform, y, sample_weight=sample_weight, check_input=check_input)

            self.feature_original = [self.oblique_split.feats, -2, -2]
            self.coefficients = self.oblique_split.coeff
            self.threshold_original = self.tree_.threshold

        return self

    def apply(self, X):
        """
        Apply the decision stump to X and return the leaf indices.
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            The input samples to apply the stump on
            
        Returns:
        --------
        X_leaves : array-like of shape (n_samples,)
            The leaf indices for each sample
        """
        X_transform = self.oblique_split.transform(X[:, self.numerical])
        return super().apply(X_transform)
