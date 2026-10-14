"""
File: Stump.py

The stump made of a transformer (how the data are represented, see RuleTreeBaseTransformer) and
a splitter (how the split is chosen, see RuleTreeBaseSplitter).
"""
import importlib
import inspect
import warnings

import numpy as np
import sklearn
from sklearn.base import BaseEstimator

from RuleTree.base.RuleTreeBaseStump import RuleTreeBaseStump, _is_number


class Stump(RuleTreeBaseStump):
    """
    Stump made of a transformer and a splitter: the transformer turns the data of one type into a
    feature matrix, the splitter finds the best split of that matrix.

    The stumps of the library (DecisionTreeStumpClassifier, PivotTreeStumpClassifier, ...) are
    subclasses that build the two components from their own parameters (see _make_transformer and
    _make_splitter). A Stump can also be built directly, to combine any transformer and splitter, e.g.
    ``Stump(PivotTransformer(MODEL_TYPE_REG), FairSplitter(penalty="balance"))``.

    The data types supported by the stump are the ones of the transformer, and its task
    (classification or regression) is the one of the splitter.

    Parameters
    ----------
    transformer : RuleTreeBaseTransformer
    splitter : RuleTreeBaseSplitter

    Attributes
    ----------
    transformer_, splitter_ : the fitted copies of the components.
    """

    def __init__(self, transformer=None, splitter=None):
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
        return self._make_transformer().supported_data_types

    @property
    def task(self):
        return self._make_splitter().task

    def fit(self, data, y, idx=None, context=None, data_type=None, sample_weight=None, check_input=True):
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
        return self.transformer_.transform(X)

    def _apply_matrix(self, X):
        return self.splitter_.apply(X)

    def _update_statistics_matrix(self, X, y, sample_weight=None):
        self.splitter_.update_statistics(X, y, sample_weight)

    def _is_fitted(self):
        return hasattr(self, "splitter_")

    # attributes of the split, as exposed by the stumps before their composition
    @property
    def feature_original(self):
        if not self._is_fitted():
            return None
        return [self.transformer_.feature_id(self.splitter_.feature_), -2, -2]

    @property
    def threshold_original(self):
        if not self._is_fitted():
            return None
        threshold = self.splitter_.threshold_
        return np.array([threshold, -2, -2], dtype=float if _is_number(threshold) else object)

    @property
    def is_categorical(self):
        return self.splitter_.is_categorical_ if self._is_fitted() else None

    @property
    def impurity(self):
        return self.splitter_.impurity_ if self._is_fitted() else None

    @property
    def gain_(self):
        return self.splitter_.gain_

    def get_rule(self, columns_names=None, scaler=None, float_precision=3):
        """
        Rule of the split, e.g. ``X_2 <= 4.5``, with the feature named after the transformer's feature id.
        Subclasses with a specific representation of the rules override it.
        """
        from RuleTree.stumps.classification.DecisionTreeStumpClassifier import DecisionTreeStumpClassifier

        return DecisionTreeStumpClassifier.get_rule(self, columns_names=columns_names, scaler=scaler,
                                                    float_precision=float_precision)

    def node_to_dict(self):
        """
        Dictionary representation of the stump: constructor arguments and, if fitted, the rule, the
        data type and the state of the transformer and of the splitter.
        """
        args = _params_to_dict(self.get_params(deep=False))
        if not self._is_fitted():
            return {"stump_type": self.__class__.__module__, "args": args}

        rule = self.get_rule(float_precision=None)
        rule["stump_type"] = self.__class__.__module__
        rule["data_type"] = self.data_type_
        rule["impurity"] = self.impurity
        rule["args"] = args
        rule["transformer"] = self.transformer_.get_state()
        rule["splitter"] = self.splitter_.get_state()

        return rule

    @classmethod
    def dict_to_node(cls, node_dict, X=None):
        """Create a stump from a dictionary created by node_to_dict (also by versions before the composition)."""
        self = cls(**cls._init_args(_params_from_dict(node_dict.get("args", {}))))

        if "splitter" in node_dict:
            self._restore_data_type(node_dict)
            self.transformer_ = self._make_transformer().set_state(node_dict["transformer"])
            self.splitter_ = self._make_splitter().set_state(node_dict["splitter"])
        elif "feature_idx" in node_dict:  # saved before the composition: the split is at top level
            self._restore_data_type(node_dict)
            self.transformer_ = self._make_transformer().set_state(self._legacy_transformer_state(node_dict))
            self.splitter_ = self._make_splitter().set_state(node_dict | {
                "feature_": self._legacy_feature(node_dict),
                "threshold_": node_dict["threshold"],
                "is_categorical_": bool(node_dict.get("is_categorical")),
                "impurity_": node_dict.get("impurity"),
                "gain_": node_dict.get("gain"),
            })

        return self

    @classmethod
    def _init_args(cls, args: dict) -> dict:
        """
        Constructor arguments from a dictionary saved by node_to_dict. Arguments of older versions
        that the stump no longer accepts are dropped (max_depth: stumps always have depth 1).
        """
        accepted = inspect.signature(cls.__init__).parameters
        dropped = [name for name in args if name not in accepted]
        unexpected = [name for name in dropped if name != "max_depth"]
        if unexpected:
            warnings.warn(f"{cls.__name__} ignores the parameters {unexpected} of the saved stump.", UserWarning)

        return {name: value for name, value in args.items() if name in accepted}

    def _legacy_feature(self, node_dict: dict) -> int:
        """Column of the feature matrix used by a split saved before the composition."""
        return int(node_dict["feature_idx"])

    def _legacy_transformer_state(self, node_dict: dict) -> dict:
        """State of the transformer of a stump saved before the composition."""
        return node_dict


def _params_to_dict(params: dict) -> dict:
    """Constructor parameters as a JSON-friendly dictionary: estimators (e.g. components) by type and parameters."""
    return {name: _estimator_to_dict(value) if isinstance(value, BaseEstimator) else value
            for name, value in params.items()}


def _params_from_dict(params: dict) -> dict:
    return {name: _estimator_from_dict(value) if isinstance(value, dict) and "__estimator__" in value else value
            for name, value in params.items()}


def _estimator_to_dict(estimator: BaseEstimator) -> dict:
    return {"__estimator__": f"{estimator.__class__.__module__}.{estimator.__class__.__qualname__}",
            "params": _params_to_dict(estimator.get_params(deep=False))}


def _estimator_from_dict(dictionary: dict) -> BaseEstimator:
    module, class_name = dictionary["__estimator__"].rsplit(".", 1)
    return getattr(importlib.import_module(module), class_name)(**_params_from_dict(dictionary["params"]))
