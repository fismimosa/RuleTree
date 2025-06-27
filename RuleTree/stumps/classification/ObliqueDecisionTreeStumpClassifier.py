from RuleTree.base.RuleTreeBaseStump import RuleTreeBaseStump
from RuleTree.stumps.classification.DecisionTreeStumpClassifier import DecisionTreeStumpClassifier
from RuleTree.stumps.splitters.ObliqueBivariateSplit import ObliqueBivariateSplit
from RuleTree.stumps.splitters.ObliqueHouseHolderSplit import ObliqueHouseHolderSplit
from RuleTree.utils import MODEL_TYPE_CLF


class ObliqueDecisionTreeStumpClassifier(DecisionTreeStumpClassifier, RuleTreeBaseStump):
    """
    A Decision Tree Stump Classifier that uses oblique splits for classification tasks.

    Oblique splits are hyperplanes that are not parallel to the feature axes, allowing
    for more flexible decision boundaries that can capture complex relationships between features.
    This implementation supports both Householder and Bivariate splitting methods.

    Parameters
    ----------
    oblique_split_type : str, default='householder'
        Type of oblique split to use. Options are 'householder' or 'bivariate'.
    pca : object or None, default=None
        PCA instance to use for feature transformation. If None, no PCA is applied.
    max_oblique_features : int, default=2
        Maximum number of features to consider in an oblique split.
    tau : float, default=1e-4
        Threshold parameter for Householder splits.
    n_orientations : int, default=10
        Number of orientations to consider for bivariate splits.
    **kwargs : dict
        Additional parameters to pass to the parent classifier.
    """

    def __init__(self,
                 oblique_split_type='householder',
                 pca=None,
                 max_oblique_features=2,
                 tau=1e-4,
                 n_orientations=10,
                 **kwargs):
        super().__init__(**kwargs)
        self.pca = pca
        self.max_oblique_features = max_oblique_features
        self.tau = tau
        self.n_orientations = n_orientations
        self.oblique_split_type = oblique_split_type

        if self.oblique_split_type == 'householder':
            self.oblique_split = ObliqueHouseHolderSplit(ml_task=MODEL_TYPE_CLF,
                                                         pca=self.pca,
                                                         max_oblique_features=self.max_oblique_features,
                                                         tau=self.tau,
                                                         **kwargs)

        if self.oblique_split_type == 'bivariate':
            self.oblique_split = ObliqueBivariateSplit(ml_task=MODEL_TYPE_CLF, n_orientations=self.n_orientations,
                                                       **kwargs)

    def fit(self, X, y, idx=None, context=None, sample_weight=None, check_input=True):
        """
        Build an oblique decision tree stump classifier from the training set (X, y).

        The method first transforms features using the specified oblique split method and
        then fits a decision tree stump on the transformed data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The training input samples.
        y : array-like of shape (n_samples,)
            The target values (class labels).
        idx : array-like or slice, default=None
            Indices of samples to use for training. If None, all samples are used.
        context : object, default=None
            Additional context information (not used in this implementation).
        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights. If None, samples are equally weighted.
        check_input : bool, default=True
            Whether to check input consistency.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        if idx is None:
            idx = slice(None)
        X = X[idx]
        y = y[idx]

        self.feature_analysis(X, y)
        self.num_pre_transformed = self.numerical
        self.cat_pre_transformed = self.categorical
        best_info_gain = -float('inf')

        if len(self.numerical) > 0:
            self.oblique_split.fit(X[:, self.numerical], y, sample_weight=sample_weight, check_input=check_input)
            X_transform = self.oblique_split.transform(X[:, self.numerical])
            super().fit(X_transform, y, sample_weight=sample_weight, check_input=check_input)

            self.feature_original = [self.oblique_split.feats, -2, -2]
            self.coefficients = self.oblique_split.coeff
            self.threshold_original = self.tree_.threshold
            self.is_oblique = True

        return self

    def apply(self, X):
        """
        Apply the decision tree stump to X, returning leaf indices.

        This method transforms the input features using the fitted oblique split method
        and then applies the decision tree stump to the transformed features.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        X_leaves : array-like of shape (n_samples,)
            For each datapoint x in X, return the index of the leaf x ends up in.
        """
        X_transform = self.oblique_split.transform(X[:, self.num_pre_transformed])
        return super().apply_sk(X_transform)  # otherwise we need to "personalize" the apply function

    def get_params(self, deep=True):
        """
        Get parameters for this estimator.

        Parameters
        ----------
        deep : bool, default=True
            If True, will return the parameters for this estimator and
            contained subobjects that are estimators.

        Returns
        -------
        params : dict
            Parameter names mapped to their values.
        """
        return {
            **self.kwargs,
            'oblique_split_type': self.oblique_split_type,
            'max_oblique_features': self.max_oblique_features,
            'pca': self.pca,
            'tau': self.tau,
            'n_orientations': self.n_orientations
        }

    def get_rule(self, columns_names=None, scaler=None, float_precision=3):
        """
        Extract the decision rule from the stump in a human-readable format.

        This method creates different representations of the oblique decision rule
        (textual, blob, graphviz) for visualization and interpretation purposes.

        Parameters
        ----------
        columns_names : array-like, default=None
            Names of the features. If None, feature indices are used.
        scaler : object, default=None
            Scaler object used to transform features. If provided, the rule is
            expressed in terms of the original (unscaled) feature values.
        float_precision : int, default=3
            Number of decimal places to use for floating point values in the rule.
            If None, no rounding is performed.

        Returns
        -------
        rule : dict
            Dictionary containing different representations of the rule:
            - feature_idx: indices of features used in the rule
            - threshold: threshold value for the rule
            - coefficients: coefficients of the oblique split
            - is_categorical: whether the split is on categorical features
            - samples: number of samples covered by the rule
            - feature_name: string representation of the features
            - textual_rule: human-readable rule representation
            - blob_rule: compact rule representation
            - graphviz_rule: rule representation for graphviz visualization
            - not_textual_rule: negation of the rule
            - not_blob_rule: negation of the compact rule
            - not_graphviz_rule: negation of the rule for graphviz
        """
        rule = {
            "feature_idx": self.feature_original[0],  # list of feats
            "threshold": self.threshold_original[0],  # thr
            "coefficients": self.coefficients,  # coefficients
            "is_categorical": self.is_categorical,
            "samples": self.n_node_samples[0]
        }

        # round coefficients here
        rule['coefficients'] = [
            str(coeff) if float_precision is None else round(float(coeff), float_precision)
            for coeff in rule['coefficients']
        ]

        feat_name = " + ".join(f"{coeff} * X_{idx}" for coeff, idx in zip(rule['coefficients'], rule['feature_idx']))

        if columns_names is not None:
            feat_name = "_".join(columns_names[idx] for idx in self.feature_original[0])  # check this for feat names
        rule["feature_name"] = feat_name

        if scaler is not None:
            # TODO
            raise NotImplementedError()
            pass

        comparison = "<=" if not self.is_categorical else "="
        not_comparison = ">" if not self.is_categorical else "!="
        rounded_value = str(rule["threshold"]) if float_precision is None else round(rule["threshold"], float_precision)

        if scaler is not None:
            # TODO
            raise NotImplementedError()
            pass

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

    This method is useful for serialization and visualization of the model.

    Returns
    -------
    rule : dict
        Dictionary containing the rule information and additional metadata:
        - stump_type: class name of the stump
        - samples: number of samples covered by the stump
        - impurity: impurity measure at the stump
        - args: dictionary of additional arguments including oblique-specific parameters
        - split: dictionary containing split arguments
    """
    rule = self.get_rule(float_precision=None)

    rule["stump_type"] = self.__class__.__name__
    rule["samples"] = self.n_node_samples[0]
    rule["impurity"] = self.tree_.impurity[0]

    rule["args"] = {
                       "is_oblique": self.is_oblique,
                       "is_pivotal": self.is_pivotal,
                       "unique_val_enum": self.unique_val_enum,
                       "coefficients": self.coefficients,
                   } | self.kwargs

    rule["split"] = {
        "args": {}
    }

    return rule
