from sklearn.preprocessing import StandardScaler

from RuleTree.base.RuleTreeBaseStump import RuleTreeBaseStump
from RuleTree.exceptions import NoSplitFoundWarning
from RuleTree.stumps.classification.PivotTreeStumpClassifier import _split_pivot_kwargs
from RuleTree.stumps.transformers.PivotTransformer import _pivot_cache
from RuleTree.stumps.regression.DecisionTreeStumpRegressor import DecisionTreeStumpRegressor
from RuleTree.stumps.splitters.PivotSplit import PivotSplit
from RuleTree.utils import MODEL_TYPE_REG
from RuleTree.utils.define import DATA_TYPE_TABULAR
from sklearn.metrics.pairwise import pairwise_distances
import numpy as np


class PivotTreeStumpRegressor(DecisionTreeStumpRegressor, RuleTreeBaseStump):
    supported_data_types = (DATA_TYPE_TABULAR,)

    def __init__(self, **kwargs):
        pivot_kwargs, tree_kwargs = _split_pivot_kwargs(kwargs)
        super().__init__(**tree_kwargs)
        self.kwargs |= pivot_kwargs  # get_params
        self.distance_measure = pivot_kwargs.get('distance_measure', 'euclidean')
        self.pivot_split = PivotSplit(ml_task=MODEL_TYPE_REG, **tree_kwargs,
                                      **{k: v for k, v in pivot_kwargs.items() if k != 'distance_measure'})
        self.X_split_instance = None

    def fit(self, data, y, idx=None, context=None, data_type=None, sample_weight=None, check_input=True):
        """
        Fit the regressor to the training data.

        This method implements the model training process by:
        1. Analyzing the features in the input data
        2. Computing pivot-based splits using the distance matrix of the whole data set
        3. Transforming the data based on distances to the pivot instances
        4. Fitting the underlying decision tree stump to the transformed data

        The scaler and the distance matrix are computed once on the whole data set and cached in
        the tree (``context``), so that all the splits share them.

        Parameters
        ----------
        data : dict or numpy.ndarray
            Whole data set, as ``{data_type: array}``.
        y : numpy.ndarray
            Target values of the whole data set.
        idx : numpy.ndarray, optional
            Indices of the instances to fit on. Default is all.
        context : RuleTree, optional
            Tree being built: the scaler and the distance matrix are cached in it.
        data_type : str, optional
            Data type to use. Default is the first supported type in data.
        sample_weight : numpy.ndarray, optional
            Sample weights of the whole data set. Default is None.
        check_input : bool, optional
            Whether to validate input data. Default is True.

        Returns
        -------
        self : PivotTreeStumpRegressor
            Fitted regressor instance.
        """
        X, _ = self._resolve_fit_data(data, data_type)
        if idx is None:
            idx = np.arange(len(X))
        self._set_feature_types(X, context)
        self.num_pre_transformed = self.numerical
        self.cat_pre_transformed = self.categorical

        if len(self.numerical) == 0:
            raise NoSplitFoundWarning(f"{self.__class__.__name__} needs numerical features")

        self.scaler, distance_matrix = _pivot_cache(X, self.numerical, self.distance_measure, self.data_type_, self._get_cache(context))

        y = y[idx]
        sample_weight = None if sample_weight is None else sample_weight[idx]
        X_scaled = self.scaler.transform(X[idx][:, self.numerical])

        self.pivot_split.fit(X_scaled, y, distance_matrix[np.ix_(idx, idx)], self.distance_measure, idx,
                             sample_weight=sample_weight, check_input=check_input)
        X_transform = self.pivot_split.transform(X_scaled, self.distance_measure)
        candidate_names = self.pivot_split.get_candidates_names()

        self.numerical, self.categorical = list(range(X_transform.shape[1])), []
        self._fit_matrix(X_transform, y, sample_weight=sample_weight, check_input=check_input)

        self.feature_original = [f'{candidate_names[self.tree_.feature[0]]}', -2, -2]
        self.threshold_original = self.tree_.threshold
        self.X_split_instance = self.pivot_split.X_candidates[self.tree_.feature[0]]

        return self

    def _transform(self, X):
        """Distance of each instance from the pivot."""
        return pairwise_distances(self.scaler.transform(X[:, self.num_pre_transformed]),
                                  self.X_split_instance.reshape(1, -1),
                                  metric=self.distance_measure)

    def _apply_matrix(self, X):
        return np.where(X[:, 0] <= self.threshold_original[0], 1, 2)

    def get_rule(self, columns_names=None, scaler=None, float_precision=3):
        """
        Get the rule representation of this classifier.

        This method generates a dictionary containing rule information,
        including feature indices, thresholds, and human-readable representations
        of the decision rule.

        Parameters
        ----------
        columns_names : list, optional
            List of column names for the features. Default is None.
        scaler : object, optional
            Scaling object used to transform feature values. Default is None.
        float_precision : int, optional
            Number of decimal places for floating point values in rules. Default is 3.

        Returns
        -------
        dict
            A dictionary containing rule information with keys:
            - feature_idx: Feature index in the original feature space
            - threshold: Threshold value for the decision rule
            - is_categorical: Whether the feature is categorical
            - samples: Number of samples in the node
            - feature_name: Name of the feature
            - textual_rule: Human-readable rule representation
            - blob_rule: Simplified rule representation
            - graphviz_rule: Rule representation for Graphviz visualization
            - not_textual_rule: Negated rule representation
            - not_blob_rule: Negated simplified rule representation
            - not_graphviz_rule: Negated rule representation for Graphviz visualization
        """
        if self.X_split_instance is None:
            return {}

        rule = {
            "feature_idx": self.feature_original[0],
            "threshold": self.threshold_original[0],
            "is_categorical": self.is_categorical,
        }

        feat_name = f"P_{rule['feature_idx']}"
        if columns_names is not None:
            # feat_names should not be useful for pivot tree
            # feat_name = columns_names[self.feature_original[0]]
            feat_name = None
        rule["feature_name"] = feat_name

        if scaler is not None:
            NotImplementedError()

        comparison = "<=" if not self.is_categorical else "="
        not_comparison = ">" if not self.is_categorical else "!="
        rounded_value = str(rule["threshold"]) if float_precision is None else round(rule["threshold"], float_precision)
        if scaler is not None:
            NotImplementedError()
        rule["textual_rule"] = f"{feat_name} {comparison} {rounded_value}"
        rule["blob_rule"] = f"{feat_name} {comparison} {rounded_value}"
        rule["graphviz_rule"] = {
            "label": f"{feat_name} {'<=' if not self.is_categorical else '='} {rounded_value}",
        }

        rule["not_textual_rule"] = f"{feat_name} {not_comparison} {rounded_value}"
        rule["not_blob_rule"] = f"{feat_name} {not_comparison} {rounded_value}"
        rule["not_graphviz_rule"] = {
            "label": f"{feat_name} {'>' if not self.is_categorical else '!='} {rounded_value}"
        }

        return rule

    def node_to_dict(self):
        """
        Convert the stump to a dictionary representation: the split, the constructor arguments,
        the pivot instance and the scaler.

        Returns
        -------
        dict
            Dictionary representation of the node.
        """
        rule = super().node_to_dict()

        if self.X_split_instance is not None:
            rule['num_pre_transformed'] = list(self.num_pre_transformed)
            rule['cat_pre_transformed'] = list(self.cat_pre_transformed)
            rule["X_split_instance"] = self.X_split_instance
            rule['scaler'] = {
                'mean_': self.scaler.mean_.tolist(),
                'scale_': self.scaler.scale_.tolist(),
                'var_': self.scaler.var_.tolist(),
                'n_samples_seen_': self.scaler.n_samples_seen_
            }

        return rule

    @classmethod
    def dict_to_node(cls, node_dict, X=None):
        """
        Create a stump from a dictionary created by node_to_dict.

        Parameters
        ----------
        node_dict : dict
            Dictionary representation of the node.
        X : numpy.ndarray, optional
            Not used.

        Returns
        -------
        The reconstructed stump.
        """
        self = super().dict_to_node(node_dict)

        if 'X_split_instance' in node_dict:
            self.num_pre_transformed = node_dict["num_pre_transformed"]
            self.cat_pre_transformed = node_dict["cat_pre_transformed"]
            self.X_split_instance = np.array(node_dict["X_split_instance"])

            self.scaler = StandardScaler()
            self.scaler.mean_ = np.array(node_dict['scaler']['mean_'])
            self.scaler.scale_ = np.array(node_dict['scaler']['scale_'])
            self.scaler.var_ = np.array(node_dict['scaler']['var_'])
            self.scaler.n_samples_seen_ = node_dict['scaler']['n_samples_seen_']

        return self

    def export_graphviz(self, graph=None, columns_names=None, scaler=None, float_precision=3):
        """
        Export the classifier as a Graphviz digraph representation.

        This method is intended to visualize the classifier as a graph but is
        currently not implemented for PivotTreeStumpRegressor.

        Parameters
        ----------
        graph : pydot.Dot, optional
            An existing graph to add the classifier node to. Default is None.
        columns_names : list, optional
            List of column names for the features. Default is None.
        scaler : object, optional
            Scaling object used to transform feature values. Default is None.
        float_precision : int, optional
            Number of decimal places for floating point values. Default is 3.

        Raises
        ------
        NotImplementedError
            This method is not implemented for PivotTreeStumpRegressor.
        """
        raise NotImplementedError()
