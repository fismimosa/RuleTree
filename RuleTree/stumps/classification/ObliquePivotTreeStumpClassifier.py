from RuleTree.base.RuleTreeBaseStump import RuleTreeBaseStump
from RuleTree.stumps.classification.DecisionTreeStumpClassifier import DecisionTreeStumpClassifier
from RuleTree.stumps.splitters.ObliqueBivariateSplit import ObliqueBivariateSplit
from RuleTree.stumps.splitters.ObliqueHouseHolderSplit import ObliqueHouseHolderSplit
from RuleTree.stumps.splitters.ObliquePivotSplit import ObliquePivotSplit
from RuleTree.utils import MODEL_TYPE_CLF


class ObliquePivotTreeStumpClassifier(DecisionTreeStumpClassifier, RuleTreeBaseStump):
    """
    A classifier that combines oblique splitting with pivot-based splitting for decision tree stumps.

    This classifier leverages both pivot-based and oblique (non-axis-parallel) splitting strategies
    to create more flexible decision boundaries than traditional axis-parallel splits. It first
    transforms the data using pivot-based distances, then applies oblique splitting on the transformed
    feature space. This approach can capture more complex patterns in the data.

    The classifier supports different oblique split types:
    - 'householder': Uses Householder transformations to find optimal oblique splits
    - 'bivariate': Focuses on bivariate features with multiple orientations

    This hybrid approach is particularly useful when dealing with datasets where standard
    axis-parallel splits are insufficient to capture the underlying data structure.

    Attributes:
        distance_measure: Distance metric used for pivot-based transformations
        pca: PCA object for dimensionality reduction if specified
        max_oblique_features: Maximum number of features used in oblique splits
        tau: Regularization parameter for oblique splits
        n_orientations: Number of orientations for bivariate splits
        oblique_split_type: Type of oblique split strategy ('householder' or 'bivariate')
        obl_pivot_split: Pivot split object used for initial transformation
        oblique_split: Oblique split object used for subsequent splitting
    """

    def __init__(self,
                 oblique_split_type='householder',
                 pca=None,
                 max_oblique_features=2,
                 tau=1e-4,
                 n_orientations=10,
                 **kwargs):
        """
        Initialize the ObliquePivotTreeStumpClassifier.

        Parameters:
            oblique_split_type (str): Type of oblique split strategy to use.
                Options are 'householder' or 'bivariate'. Default is 'householder'.
            pca (object, optional): PCA object for dimensionality reduction. Default is None.
            max_oblique_features (int): Maximum number of features to consider for oblique splits.
                Only applies when oblique_split_type='householder'. Default is 2.
            tau (float): Regularization parameter for oblique splits to prevent overfitting.
                Only applies when oblique_split_type='householder'. Default is 1e-4.
            n_orientations (int): Number of orientations to consider for bivariate splits.
                Only applies when oblique_split_type='bivariate'. Default is 10.
            **kwargs: Additional keyword arguments passed to the parent classes and splitters.
                These can include parameters like criterion, splitter, max_features, etc.
        """
        super().__init__(**kwargs)
        super().__init__(**kwargs)

        self.distance_measure = None
        self.pca = pca
        self.max_oblique_features = max_oblique_features
        self.tau = tau
        self.n_orientations = n_orientations
        self.oblique_split_type = oblique_split_type

        self.obl_pivot_split = ObliquePivotSplit(ml_task=MODEL_TYPE_CLF, oblique_split_type=oblique_split_type, **kwargs)

        if oblique_split_type == 'householder':
            self.oblique_split = ObliqueHouseHolderSplit(ml_task=MODEL_TYPE_CLF,
                                                         pca=self.pca,
                                                         max_oblique_features=self.max_oblique_features,
                                                         tau=self.tau,
                                                         **kwargs)

        if oblique_split_type == 'bivariate':
            self.oblique_split = ObliqueBivariateSplit(ml_task=MODEL_TYPE_CLF, n_orientations=self.n_orientations, **kwargs)

    def fit(self, X, y, distance_matrix, distance_measure, idx, sample_weight=None, check_input=True):
        """
        Fit the classifier to the training data.

        The fitting process involves:
        1. Analyzing features (numerical vs. categorical)
        2. Applying pivot-based transformation using the specified distance measure
        3. Applying oblique split on the transformed feature space
        4. Fitting a decision tree stump on the final transformed features

        Parameters:
            X (np.ndarray): Feature matrix of shape (n_samples, n_features).
            y (np.ndarray): Target labels of shape (n_samples,).
            distance_matrix (np.ndarray): Precomputed distance matrix of shape (n_samples, n_samples).
            distance_measure (str): Distance metric to use (e.g., 'euclidean', 'manhattan').
            idx (int): Index of the pivot instance in X to use as reference.
            sample_weight (np.ndarray, optional): Sample weights of shape (n_samples,).
                If None, all samples are weighted equally. Default is None.
            check_input (bool, optional): Whether to validate input data. Default is True.

        Returns:
            self: The fitted classifier instance for method chaining.

        Note:
            The method stores transformed features and split information for later use
            in prediction and rule extraction.
        """
        self.feature_analysis(X, y)
        self.num_pre_transformed = self.numerical
        self.cat_pre_transformed = self.categorical

        if len(self.numerical) > 0:
            self.obl_pivot_split.fit(X[:, self.numerical], y, distance_matrix, distance_measure, idx,
                                     sample_weight=sample_weight, check_input=check_input)
            X_transform = self.obl_pivot_split.transform(X[:, self.numerical], distance_measure)
            candidate_names = self.obl_pivot_split.get_candidates_names()

            self.oblique_split.fit(X_transform, y, sample_weight=sample_weight, check_input=check_input)
            X_transform_oblique = self.oblique_split.transform(X_transform)
            super().fit(X_transform_oblique, y, sample_weight=sample_weight, check_input=check_input)

            feats = [f'{p}' for p in candidate_names[self.oblique_split.feats]]
            self.feature_original = [feats, -2, -2]
            self.coefficients = self.oblique_split.coeff
            self.threshold_original = self.tree_.threshold
            self.is_oblique = True
            self.is_pivotal = True
            self.distance_measure = distance_measure

        return self

    def apply(self, X):
        """
        Apply the decision rule to X and return the indices of the leaves that each sample reaches.

        This method:
        1. Transforms input data using the fitted pivot-based transformation
        2. Applies the oblique transformation to the pivot-transformed data
        3. Passes the fully transformed data through the decision tree stump

        Parameters:
            X (np.ndarray): The input samples of shape (n_samples, n_features).
                Must contain the same number of features as the data used for fitting.

        Returns:
            np.ndarray: An array of shape (n_samples,) containing the leaf indices
                        (terminal node) for each sample.
        """
        X_transformed = self.obl_pivot_split.transform(X[:, self.num_pre_transformed], self.distance_measure)
        X_transformed_oblique = self.oblique_split.transform(X_transformed)
        return super().apply_sk(X_transformed_oblique)
    
    def get_params(self, deep=True):
        """
        Get parameters for this estimator.

        Parameters:
            deep (bool): If True, will return the parameters for this estimator and
                contained subobjects that are estimators. Default is True.

        Returns:
            dict: Parameter names mapped to their values.
                Includes all parameters from kwargs plus specific parameters
                of the ObliquePivotTreeStumpClassifier.
        """
        return {
            **self.kwargs,
            'oblique_split_type' : self.oblique_split_type,
            'max_oblique_features': self.max_oblique_features,
            'pca': self.pca,
            'tau': self.tau,
            'n_orientations': self.n_orientations
        }

    def get_rule(self, columns_names=None, scaler=None, float_precision=3):
        """
        Extract a human-readable rule from the fitted classifier.

        This method generates various representations of the decision rule learned by the model,
        including textual, dictionary, and graphviz-compatible formats.

        Parameters:
            columns_names (list, optional): List of column names corresponding to the features.
                If provided, feature names will be used in the rules instead of indices.
            scaler (object, optional): Scaler object used to transform the data during preprocessing.
                If provided, thresholds will be transformed back to the original scale.
            float_precision (int, optional): Number of decimal places for floating point values
                in the generated rules. If None, exact values are used. Default is 3.

        Returns:
            dict: A dictionary containing multiple representations of the rule:
                - feature_idx: Indices of the features used in the rule
                - threshold: Threshold value for the rule
                - coefficients: Coefficients of the oblique split
                - is_categorical: Whether the rule uses categorical features
                - samples: Number of samples in the node
                - feature_name: Human-readable feature name/expression
                - textual_rule: Human-readable rule with comparison operator
                - blob_rule: Simplified rule representation
                - graphviz_rule: Rule formatted for graphviz visualization
                - not_textual_rule: Negated version of the rule
                - not_blob_rule: Negated simplified rule
                - not_graphviz_rule: Negated rule for graphviz visualization
        """
        rule = {
            "feature_idx": self.feature_original[0],
            "threshold": self.threshold_original[0],
            "coefficients" : self.coefficients,
            "is_categorical": self.is_categorical,
            "samples": self.n_node_samples[0]
        }
        
        rule['coefficients'] = [
                               str(coeff) if float_precision is None else round(float(coeff), float_precision) 
                               for coeff in rule['coefficients']
                               ]
        

        feat_name = " + ".join(f"{coeff} * P_{idx}" for coeff, idx in zip(rule['coefficients'], rule['feature_idx']))
       
        #if columns_names is not None:
        #    feat_name = "_".join(columns_names[idx] for idx in self.feature_original[0]) #check this for feat names
        #rule["feature_name"] = feat_name
        
    
        #feat_name = None
        rule["feature_name"] = feat_name

        #if scaler is not None:
        #    NotImplementedError()

        comparison = "<=" if not self.is_categorical else "="
        not_comparison = ">" if not self.is_categorical else "!="
        rounded_value = str(rule["threshold"]) if float_precision is None else round(rule["threshold"], float_precision)
        #if scaler is not None:
        #    NotImplementedError()
        rule["textual_rule"] = f"{feat_name} {comparison} {rounded_value}\t{rule['samples']}"
        rule["blob_rule"] = f"{feat_name} {comparison} {rounded_value}"
        rule["graphviz_rule"] = {
            "label": f"{feat_name} {'!=' if not self.is_categorical else '='} {rounded_value}",
        }

        rule["not_textual_rule"] = f"{feat_name} {not_comparison} {rounded_value}"
        rule["not_blob_rule"] = f"{feat_name} {not_comparison} {rounded_value}"
        rule["not_graphviz_rule"] = {
            "label": f"{feat_name} {'>' if not self.is_categorical else '<='} {rounded_value}"
        }

        return rule
    

    def node_to_dict(self):
        """
        Convert the decision stump node to a dictionary representation.

        This method creates a comprehensive dictionary containing all necessary information
        about the fitted model, including the rule, model parameters, and additional metadata.
        This dictionary representation can be used for serialization, visualization,
        or further processing.

        Returns:
            dict: A dictionary containing:
                - Rule information (from get_rule())
                - Classifier type and module information
                - Sample statistics and impurity measure
                - Model arguments and parameters
                - Split configuration
        """
        rule = self.get_rule(float_precision=None)

        rule["stump_type"] = self.__class__.__module__
        rule["samples"] = self.n_node_samples[0]
        rule["impurity"] = self.tree_.impurity[0]
     
      
        rule["args"] = {
            "is_oblique": self.is_oblique,
            "is_pivotal": self.is_pivotal,
            "unique_val_enum": self.unique_val_enum,
            "coefficients": self.coefficients,
            "num_pre_transformed" : self.num_pre_transformed,
            "cat_pre_transformed" : self.cat_pre_transformed,
            
            "distance_measure" : self.distance_measure #adding this for PT
            
        } | self.kwargs

        rule["split"] = {
            "args": {}
        }
        

        return rule

    def export_graphviz(self, graph=None, columns_names=None, scaler=None, float_precision=3):
        """
        Export the decision stump to a graphviz-compatible format.

        This method is intended to generate a graphical representation of the
        decision stump but is not yet implemented.

        Parameters:
            graph: Graph object to add nodes to.
            columns_names (list, optional): List of feature names.
            scaler (object, optional): Scaler used to preprocess the data.
            float_precision (int, optional): Number of decimal places for float values.

        Raises:
            NotImplementedError: This method is not yet implemented.
        """
        raise NotImplementedError()
