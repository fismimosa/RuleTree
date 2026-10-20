"""
File: Stump.py

The stump made of a transformer (how the data are represented, see RuleTreeBaseTransformer) and
a splitter (how the split is chosen, see RuleTreeBaseSplitter).
"""
import importlib

import numpy as np
import sklearn
from sklearn.base import BaseEstimator

from RuleTree.base.RuleTreeBaseStump import RuleTreeBaseStump, rule_dict


class Stump(RuleTreeBaseStump):
    """
    Stump made of a transformer and a splitter: the transformer turns the data of one type into a
    feature matrix, the splitter finds the best split of that matrix.

    Any transformer can be combined with any splitter. The data types supported by the stump are the
    ones of the transformer, and its task (classification or regression) is the one of the splitter.
    The stumps of the library are subclasses that build the two components from their own parameters
    (``_make_transformer`` and ``_make_splitter``); see :doc:`/guides/extending` to write new ones.

    Parameters
    ----------
    transformer : RuleTreeBaseTransformer, default=None
        Transformer of the data into a feature matrix; it is cloned at each fit.
    splitter : RuleTreeBaseSplitter, default=None
        Splitter of the feature matrix; it is cloned at each fit.

    Attributes
    ----------
    transformer_ : RuleTreeBaseTransformer
        The fitted transformer.
    splitter_ : RuleTreeBaseSplitter
        The fitted splitter (feature, threshold, gain and impurity of the split).
    data_type_ : str
        Data type the stump was fitted on.
    feature_original : int or str
        Feature of the split, as identified by the transformer: a column of the data, a pivot, ...
    threshold_original : float or object
        Threshold of a numerical split, or value of a categorical split.

    Examples
    --------
    >>> from sklearn.datasets import load_diabetes
    >>> from RuleTree import RuleTreeRegressor
    >>> from RuleTree.stumps.Stump import Stump
    >>> from RuleTree.stumps.splitters import FairSplitter, RegressionSplitter
    >>> from RuleTree.stumps.transformers import MatrixTransformer, PivotTransformer
    >>> from RuleTree.utils.define import MODEL_TYPE_REG
    >>> X, y = load_diabetes(return_X_y=True)
    >>> stump = Stump(PivotTransformer(MODEL_TYPE_REG), RegressionSplitter())
    >>> stump.task, stump.supported_data_types
    ('reg', ('tabular',))
    >>> model = RuleTreeRegressor(max_depth=2, base_stumps=stump).fit(X[:100], y[:100])
    """

    def __init__(self, transformer=None, splitter=None):
        """Initialize the stump; see the class documentation for the parameters."""
        self.transformer = transformer
        self.splitter = splitter

    def _make_transformer(self):
        """A new, unfitted transformer."""
        return sklearn.clone(self.transformer)

    def _make_splitter(self):
        """A new, unfitted splitter."""
        return sklearn.clone(self.splitter)

    @property
    def supported_data_types(self):
        """Data types supported by the stump: the ones of its transformer."""
        return self._make_transformer().supported_data_types

    @property
    def task(self):
        """Task of the stump: the one of its splitter (MODEL_TYPE_CLF or MODEL_TYPE_REG)."""
        return self._make_splitter().task

    def fit(self, data, y, idx=None, context=None, data_type=None, sample_weight=None, check_input=True):
        """
        Fit the stump on the instances ``idx`` of ``data``: the transformer on the whole data set of the
        selected type (with the indices of the node and the cache of the tree), the splitter on the
        feature matrix of the node; then the transformer keeps only what is needed by the chosen feature
        (and the splitter is fitted again if the feature matrix changes).

        Parameters
        ----------
        data : dict or numpy.ndarray
            Whole data set, as ``{data_type: array}``.
        y : numpy.ndarray
            Target of the whole data set.
        idx : numpy.ndarray, default=None
            Indices of the instances reaching the node. If None, all the instances.
        context : RuleTree, default=None
            Tree being built, whose cache is shared by the stumps of the same class.
        data_type : str, default=None
            Data type to use. If None, the first supported data type available in ``data``.
        sample_weight : numpy.ndarray, default=None
            Sample weights of the whole data set.
        check_input : bool, default=True
            Whether to check the input data.

        Returns
        -------
        self : Stump
            The fitted stump.

        Raises
        ------
        NoSplitFoundWarning
            If the instances cannot be split.
        """
        X, _ = self._resolve_fit_data(data, data_type)
        if idx is None:
            idx = np.arange(len(X))
        sample_weight_idx = None if sample_weight is None else sample_weight[idx]

        transformer = self._make_transformer().fit(X, y, idx, data_type=self.data_type_,
                                                   cache=self._get_cache(context), sample_weight=sample_weight)
        splitter = self._make_splitter().fit(transformer.transform(X[idx]), y[idx], transformer.numerical_,
                                             transformer.categorical_, sample_weight_idx, check_input)

        feature = transformer.select(splitter.feature_)
        if transformer.refit_after_select:
            splitter = self._make_splitter().fit(transformer.transform(X[idx]), y[idx], transformer.numerical_,
                                                 transformer.categorical_, sample_weight_idx, check_input)
        else:
            splitter.feature_ = feature

        self.transformer_, self.splitter_ = transformer, splitter

        return self

    def _transform(self, X):
        """
        Feature matrix of X, computed by the fitted transformer.

        Parameters
        ----------
        X : numpy.ndarray
            Data of the stump's data type.

        Returns
        -------
        X_transformed : numpy.ndarray of shape (n_samples, n_features)
        """
        return self.transformer_.transform(X)

    def _apply_matrix(self, X):
        """
        Split of a feature matrix, computed by the fitted splitter.

        Parameters
        ----------
        X : numpy.ndarray of shape (n_samples, n_features)
            Feature matrix.

        Returns
        -------
        labels : numpy.ndarray of int
            1 for the rows that go to the left child, 2 otherwise.
        """
        return self.splitter_.apply(X)

    def _update_statistics_matrix(self, X, y, sample_weight=None):
        """
        Recompute the statistics of the splitter on a feature matrix.

        Parameters
        ----------
        X : numpy.ndarray of shape (n_samples, n_features)
            Feature matrix of the instances in the node.
        y : numpy.ndarray
            Target of the instances in the node.
        sample_weight : numpy.ndarray, default=None
            Sample weights of the instances in the node.
        """
        self.splitter_.update_statistics(X, y, sample_weight)

    def _is_fitted(self):
        """
        Whether the stump is fitted.

        Returns
        -------
        is_fitted : bool
        """
        return hasattr(self, "splitter_")

    @property
    def feature_original(self):
        """
        Feature of the split, as identified by the transformer: a column of the data (int), or a name
        such as a pivot. None if not fitted.
        """
        return self.transformer_.feature_id(self.splitter_.feature_) if self._is_fitted() else None

    @property
    def threshold_original(self):
        """Threshold (numerical split) or value (categorical split). None if not fitted."""
        return self.splitter_.threshold_ if self._is_fitted() else None

    @property
    def is_categorical(self):
        """Whether the split is categorical (``x == value``). None if not fitted."""
        return self.splitter_.is_categorical_ if self._is_fitted() else None

    @property
    def impurity(self):
        """Impurity of the node and of the two children, if computed by the splitter. None if not fitted."""
        return self.splitter_.impurity_ if self._is_fitted() else None

    @property
    def gain_(self):
        """Gain of the split, used by the trees with stump_selection='best'."""
        return self.splitter_.gain_

    def get_rule(self, columns_names=None, scaler=None, float_precision=3):
        """
        Rule of the split, e.g. ``petal length <= 2.45`` or ``color = red``: the feature is named
        after the transformer's feature id.

        Parameters
        ----------
        columns_names : list of str, default=None
            Names of the features. If None, ``X_<index>``. Features named by the transformer (see
            RuleTreeBaseTransformer.feature_id) keep their name.
        scaler : TransformerMixin, default=None
            Scaler used to preprocess the data: the threshold is also given in the original scale
            (``threshold_scaled``) and shown in the rule.
        float_precision : int, default=3
            Number of decimals of the threshold. If None, it is not rounded.

        Returns
        -------
        rule : dict
            ``feature_idx``, ``threshold``, ``is_categorical``, ``feature_name``, ``threshold_scaled``
            (with a scaler), the condition of the left child (``textual_rule``, ``blob_rule``,
            ``graphviz_rule``) and its negation (``not_textual_rule``, ``not_blob_rule``,
            ``not_graphviz_rule``).
        """
        rule = {
            "feature_idx": self.feature_original,
            "threshold": self.threshold_original,
            "is_categorical": self.is_categorical,
        }

        if isinstance(rule["feature_idx"], str):  # a feature named by the transformer, e.g. "mean"
            feat_name = rule["feature_idx"]
        elif columns_names is not None:
            feat_name = columns_names[self.feature_original]
        else:
            feat_name = f"X_{rule['feature_idx']}"
        rule["feature_name"] = feat_name

        if scaler is not None:
            array = np.zeros((1, scaler.n_features_in_))
            array[0, self.feature_original] = self.threshold_original

            rule["threshold_scaled"] = scaler.inverse_transform(array)[0, self.feature_original]

        comparison = "<=" if not self.is_categorical else "="
        not_comparison = ">" if not self.is_categorical else "!="
        rounded_value = str(rule["threshold"]) if float_precision is None or self.is_categorical else (
            round(rule["threshold"], float_precision))
        if scaler is not None:
            rounded_value = str(rule["threshold_scaled"]) if float_precision is None else (
                round(rule["threshold_scaled"], float_precision))
        return rule_dict(f"{feat_name} {comparison} {rounded_value}", f"{feat_name} {not_comparison} {rounded_value}",
                         **rule)

    def node_to_dict(self):
        """
        Dictionary representation of the stump: constructor arguments and, if fitted, the rule, the
        data type and the state of the transformer and of the splitter.
        """
        args = _params_to_dict(self.get_params(deep=False))
        if not self._is_fitted():
            return {"stump_type": self.__class__.__module__, "stump_class": self.__class__.__name__, "args": args}

        rule = self.get_rule(float_precision=None)
        rule["stump_type"] = self.__class__.__module__
        rule["stump_class"] = self.__class__.__name__
        rule["data_type"] = self.data_type_
        rule["impurity"] = self.impurity
        rule["args"] = args
        rule["transformer"] = self.transformer_.get_state()
        rule["splitter"] = self.splitter_.get_state()

        return rule

    @classmethod
    def dict_to_node(cls, node_dict):
        """
        Create a stump from a dictionary created by :meth:`node_to_dict`.

        Parameters
        ----------
        node_dict : dict
            Dictionary representation of the stump.

        Returns
        -------
        stump : Stump
            The stump, fitted if it was fitted when saved.
        """
        self = cls(**_params_from_dict(node_dict["args"]))
        if "splitter" in node_dict:
            self.data_type_ = node_dict["data_type"]
            self.transformer_ = self._make_transformer().set_state(node_dict["transformer"])
            self.splitter_ = self._make_splitter().set_state(node_dict["splitter"])

        return self


def _params_to_dict(params: dict) -> dict:
    """Constructor parameters as a JSON-friendly dictionary: estimators (e.g. components) by type and parameters."""
    return {name: _estimator_to_dict(value) if isinstance(value, BaseEstimator) else value
            for name, value in params.items()}


def _params_from_dict(params: dict) -> dict:
    """
    Constructor parameters from a dictionary created by _params_to_dict.

    Parameters
    ----------
    params : dict
        Saved parameters.

    Returns
    -------
    params : dict
        Parameters, with the estimators rebuilt.
    """
    return {name: _estimator_from_dict(value) if isinstance(value, dict) and "__estimator__" in value else value
            for name, value in params.items()}


def _estimator_to_dict(estimator: BaseEstimator) -> dict:
    """
    JSON-friendly representation of an estimator: its class and its parameters.

    Parameters
    ----------
    estimator : BaseEstimator
        The estimator, e.g. a transformer or a splitter.

    Returns
    -------
    dictionary : dict
        ``{"__estimator__": "module.Class", "params": {...}}``.
    """
    return {"__estimator__": f"{estimator.__class__.__module__}.{estimator.__class__.__qualname__}",
            "params": _params_to_dict(estimator.get_params(deep=False))}


def _estimator_from_dict(dictionary: dict) -> BaseEstimator:
    """
    Estimator from a dictionary created by _estimator_to_dict. The class must be importable from its
    module (e.g. not defined in a notebook or in ``__main__`` of another script).

    Parameters
    ----------
    dictionary : dict
        Representation of the estimator.

    Returns
    -------
    estimator : BaseEstimator
        A new, unfitted estimator.
    """
    module, class_name = dictionary["__estimator__"].rsplit(".", 1)
    return getattr(importlib.import_module(module), class_name)(**_params_from_dict(dictionary["params"]))
