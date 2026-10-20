"""Abstract base class of the stumps: the single splits that make up the nodes of a RuleTree."""

import hashlib
import inspect
import json
from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

from RuleTree.utils.define import DATA_TYPE_TABULAR


class RuleTreeBaseStump(BaseEstimator, ABC):
    """
    Abstract base class of the stumps, the splits of the nodes of a RuleTree.

    A stump divides the instances of a node into two children: ``apply`` returns 1 for the
    instances that go to the left child and 2 for the ones that go to the right child.

    Notes
    -----
    Stumps receive the whole data set as a dictionary ``data = {data_type: array}`` (see
    :data:`RuleTree.utils.define.DATA_INPUT_TYPES`) together with ``idx``, the indices of the
    instances reaching the node. A fitted stump works on a single data type, chosen by the tree
    among ``supported_data_types`` and stored in ``data_type_``. For convenience, ``data`` can
    also be a single array, interpreted as data of the stump's data type.

    The tree passed as ``context`` holds a cache that lives until the next ``fit`` (or
    ``clear_cache()``), where stumps can store anything that can be shared among splits, e.g.
    embeddings computed on the whole data set. The stumps of the same class with the same
    hyperparameters share a cache, named by :meth:`cache_namespace`. Everything needed by ``apply``
    must be stored in the stump itself.

    The tree also passes ``node_id``, the identifier of the node being split (``"R"``, ``"Rl"``,
    ...): with ``context.get_node_by_id(node_id)`` the stump reaches the node, and from it its
    ancestors and their stumps. The node is a leaf while its stump is fitted; its ancestors are
    already split.

    Subclasses that work on a 2D matrix can implement ``_transform`` (data of the stump's data
    type -> matrix) and ``_apply_matrix`` (matrix -> 1/2 labels) and reuse :meth:`apply` and
    :meth:`update_statistics`. The stumps of the library are made of a transformer and a splitter
    (see :class:`RuleTree.stumps.Stump.Stump`), which is the simplest way to write a new stump.

    Attributes
    ----------
    supported_data_types : tuple of str
        Data types the stump can be fitted on.
    task : str
        Kind of target, MODEL_TYPE_CLF (classification) or MODEL_TYPE_REG (regression).
    data_type_ : str
        Data type the stump has been fitted on.
    node_id_ : str or None
        Identifier of the node of the stump in its tree (set by ``fit`` and by RuleTreeNode.set_stump),
        None for a stump outside a tree.

    See Also
    --------
    RuleTree.stumps.Stump.Stump : a stump made of a transformer and a splitter, the simplest way to
        write a new stump (see :doc:`/guides/extending`).
    """

    supported_data_types = (DATA_TYPE_TABULAR,)

    # identifier of the node of the stump in its tree, None outside a tree
    node_id_ = None

    @abstractmethod
    def fit(self, data, y, idx=None, context=None, data_type=None, sample_weight=None, check_input=True,
            node_id=None):
        """
        Fit the stump on the instances ``idx`` of ``data``.

        Parameters
        ----------
        data : dict or numpy.ndarray
            Whole data set, as ``{data_type: array}``.
        y : numpy.ndarray
            Target of the whole data set.
        idx : numpy.ndarray, default=None
            Indices of the instances reaching the node. If None, all the instances.
        context : RuleTree, default=None
            Tree being built, used to share cached values among the splits.
        data_type : str, default=None
            Data type to use. If None, the first supported data type available in ``data``.
        sample_weight : numpy.ndarray, default=None
            Sample weights of the whole data set.
        check_input : bool, default=True
            Whether to check the input data.
        node_id : str, default=None
            Identifier of the node being split, stored in ``node_id_``. None outside a tree.

        Returns
        -------
        self : RuleTreeBaseStump
            The fitted stump.

        Raises
        ------
        NoSplitFoundWarning
            If the instances cannot be split.
        """
        raise NotImplementedError("The fit method must be implemented in subclasses.")

    @abstractmethod
    def get_rule(self, columns_names:list=None, scaler:TransformerMixin=None, float_precision: Optional[int] = 3) -> dict:
        """
        Human-readable representation of the split.

        Parameters
        ----------
        columns_names : list of str, default=None
            Names of the features, used instead of their indices.
        scaler : TransformerMixin, default=None
            Scaler used to preprocess the data: the thresholds are shown in the original scale.
        float_precision : int, default=3
            Number of decimals of the thresholds. If None, they are not rounded.

        Returns
        -------
        rule : dict
            The rule, with at least the keys ``textual_rule`` and ``not_textual_rule`` (the condition
            of the left child and its negation), ``blob_rule``, ``not_blob_rule``, ``graphviz_rule`` and
            ``not_graphviz_rule`` (attributes of the Graphviz node).
        """
        pass

    def get_graphviz_rule(self, columns_names: list = None, scaler: TransformerMixin = None,
                          float_precision: Optional[int] = 3) -> dict:
        """
        Attributes of the Graphviz node of the split (see RuleTree.export_graphviz). The default
        implementation is ``get_rule(...)["graphviz_rule"]``; stumps that draw something in the node,
        such as the shapelet stumps, override it.

        Parameters
        ----------
        columns_names : list of str, default=None
            Names of the tabular features.
        scaler : TransformerMixin, default=None
            Scaler used to preprocess the data: the thresholds are shown in the original scale.
        float_precision : int, default=3
            Number of decimals of the threshold.

        Returns
        -------
        attributes : dict
            Attributes of the Graphviz node, e.g. ``label``.
        """
        return self.get_rule(columns_names=columns_names, scaler=scaler,
                             float_precision=float_precision)["graphviz_rule"]

    @abstractmethod
    def node_to_dict(self) -> dict:
        """
        Dictionary representation of the stump, used to save the tree (see RuleTree.to_dict).

        Returns
        -------
        node_dict : dict
            JSON-friendly dictionary with the key ``stump_type`` (module of the stump class) and all
            that is needed by :meth:`dict_to_node` to rebuild the stump.
        """
        raise NotImplementedError("The node_to_dict method must be implemented in subclasses.")

    @classmethod
    @abstractmethod
    def dict_to_node(cls, node_dict: dict) -> 'RuleTreeBaseStump':
        """
        Create a stump from a dictionary created by :meth:`node_to_dict`.

        Parameters
        ----------
        node_dict : dict
            Dictionary representation of the stump.

        Returns
        -------
        stump : RuleTreeBaseStump
            The stump, fitted if it was fitted when saved.
        """
        raise NotImplementedError("The dict_to_node method must be implemented in subclasses.")

    @property
    def task(self) -> str:
        """
        Kind of target of the stump: MODEL_TYPE_CLF (classification) or MODEL_TYPE_REG (regression).
        A tree accepts only stumps of its task (see RuleTree._get_stumps_task).
        """
        raise NotImplementedError(f"{self.__class__.__name__} must define its task.")

    def supports(self, data_type:str) -> bool:
        """
        Whether the stump can be fitted on a data type.

        Parameters
        ----------
        data_type : str
            A data type, e.g. ``"tabular"`` (see :data:`RuleTree.utils.define.DATA_INPUT_TYPES`).

        Returns
        -------
        supported : bool
            True if ``data_type`` is in ``supported_data_types``.
        """
        return data_type in self.supported_data_types

    def apply(self, data, idx=None) -> np.ndarray:
        """
        Apply the split to the instances ``idx`` of ``data``.

        Parameters
        ----------
        data : dict or numpy.ndarray
            Data set, as ``{data_type: array}`` or a single array of the stump's data type.
        idx : numpy.ndarray, default=None
            Indices of the instances to split. If None, all the instances.

        Returns
        -------
        labels : numpy.ndarray of int
            For each instance in ``idx``, 1 if it goes to the left child, 2 otherwise.
        """
        X = self._get_data(data)
        if idx is not None:
            X = X[idx]

        return self._apply_matrix(self._transform(X))

    def update_statistics(self, data, y, idx=None, context=None, sample_weight=None, check_input=True):
        """
        Recompute the statistics of the split (e.g. impurity) on the instances ``idx`` of ``data``,
        without changing the split.

        Parameters
        ----------
        data : dict or numpy.ndarray
            Whole data set, as ``{data_type: array}``.
        y : numpy.ndarray
            Target of the whole data set.
        idx : numpy.ndarray, default=None
            Indices of the instances reaching the node. If None, all the instances.
        context : RuleTree, default=None
            Not used, kept for symmetry with :meth:`fit`.
        sample_weight : numpy.ndarray, default=None
            Sample weights of the whole data set.
        check_input : bool, default=True
            Not used, kept for symmetry with :meth:`fit`.
        """
        if idx is None:
            idx = slice(None)
        X = self._transform(self._get_data(data)[idx])

        self._update_statistics_matrix(X, y[idx], None if sample_weight is None else sample_weight[idx])

    def _transform(self, X: np.ndarray) -> np.ndarray:
        """
        Turn data of the stump's data type into the matrix the split is applied on.

        Parameters
        ----------
        X : numpy.ndarray
            Data of the stump's data type.

        Returns
        -------
        X_transformed : numpy.ndarray of shape (n_samples, n_features)
            Feature matrix. The default implementation returns X.
        """
        return X

    def _apply_matrix(self, X: np.ndarray) -> np.ndarray:
        """
        Apply the split to the matrix returned by :meth:`_transform`.

        Parameters
        ----------
        X : numpy.ndarray of shape (n_samples, n_features)
            Feature matrix.

        Returns
        -------
        labels : numpy.ndarray of int
            1 for the rows that go to the left child, 2 otherwise.
        """
        raise NotImplementedError(f"{self.__class__.__name__} must implement apply or _apply_matrix.")

    def _update_statistics_matrix(self, X: np.ndarray, y: np.ndarray, sample_weight=None):
        """
        Recompute the split statistics on the matrix returned by :meth:`_transform`.

        Parameters
        ----------
        X : numpy.ndarray of shape (n_samples, n_features)
            Feature matrix of the instances in the node.
        y : numpy.ndarray
            Target of the instances in the node.
        sample_weight : numpy.ndarray, default=None
            Sample weights of the instances in the node.
        """

    def _resolve_fit_data(self, data, data_type=None) -> tuple[np.ndarray, str]:
        """
        Select the data to fit on and store its type in ``data_type_``.

        Parameters
        ----------
        data : dict or numpy.ndarray
            Data set, as ``{data_type: array}`` or a single array.
        data_type : str, default=None
            Data type to use. If None, the first supported data type available in ``data``.

        Returns
        -------
        X : numpy.ndarray
            Data of the selected type (all the instances).
        data_type : str
            The selected data type.

        Raises
        ------
        ValueError
            If the data type is not supported or not available in ``data``.
        """
        if not isinstance(data, dict):
            data = {data_type if data_type is not None else self.supported_data_types[0]: data}

        if data_type is None:
            available = [t for t in self.supported_data_types if t in data]
            if not available:
                raise ValueError(f"{self.__class__.__name__} supports {list(self.supported_data_types)} data, "
                                 f"got {list(data)}.")
            data_type = available[0]

        if not self.supports(data_type):
            raise ValueError(f"{self.__class__.__name__} does not support {data_type} data.")
        if data_type not in data:
            raise ValueError(f"No {data_type} data provided.")

        self.data_type_ = data_type
        return np.asarray(data[data_type]), data_type

    def _get_data(self, data) -> np.ndarray:
        """
        Data of the type the stump has been fitted on.

        Parameters
        ----------
        data : dict or numpy.ndarray
            Data set, as ``{data_type: array}`` or a single array of the stump's data type.

        Returns
        -------
        X : numpy.ndarray
            Data of the type ``data_type_``.
        """
        return np.asarray(data[self.data_type_]) if isinstance(data, dict) else np.asarray(data)

    def cache_namespace(self) -> str:
        """
        Name of the cache of the stump in the tree (see RuleTree.get_cache): the class name and a hash
        of the hyperparameters, also of the components (e.g. the transformer of a Stump). Stumps of
        the same class with different hyperparameters do not share cached values.

        The tree gives its ``random_state`` to the stumps without one, so the namespace of a stump
        used by a tree is the one of its copy in ``candidate_splits_``.

        Returns
        -------
        namespace : str
            E.g. ``"PivotTreeStumpClassifier-1a2b3c4d"``.

        Examples
        --------
        >>> from RuleTree.stumps.classification import DecisionTreeStumpClassifier
        >>> a, b = DecisionTreeStumpClassifier(), DecisionTreeStumpClassifier(criterion="entropy")
        >>> a.cache_namespace().startswith("DecisionTreeStumpClassifier-")
        True
        >>> a.cache_namespace() == DecisionTreeStumpClassifier().cache_namespace() != b.cache_namespace()
        True
        """
        params = {name: f"<{type(value).__module__}.{type(value).__qualname__}>"
                  if isinstance(value, BaseEstimator) else value
                  for name, value in self.get_params(deep=True).items()}
        digest = hashlib.md5(json.dumps(params, sort_keys=True, default=repr).encode()).hexdigest()
        return f"{self.__class__.__name__}-{digest[:8]}"

    def _get_cache(self, context) -> dict:
        """
        Cache of the stump in the tree ``context``, named by :meth:`cache_namespace`.

        Parameters
        ----------
        context : RuleTree or None
            The tree being built.

        Returns
        -------
        cache : dict
            The cache of the stump; without a context, an empty dictionary that is not kept.
        """
        return context.get_cache(self.cache_namespace()) if context is not None else {}


def rule_dict(condition: str, negation: str, **attributes) -> dict:
    """
    Dictionary of a rule as returned by the ``get_rule`` of the stumps: the condition of the left child
    and its negation, as text (``textual_rule``, ``blob_rule``) and as Graphviz label (``graphviz_rule``),
    after the given attributes of the split.

    Parameters
    ----------
    condition : str
        Condition of the left child, e.g. ``"X_3 <= 0.8"``.
    negation : str
        Its negation, the condition of the right child, e.g. ``"X_3 > 0.8"``.
    **attributes : dict
        Other keys of the rule, e.g. ``feature_idx``, ``threshold``, ``is_categorical``,
        ``feature_name``.

    Returns
    -------
    rule : dict

    Examples
    --------
    >>> from RuleTree.base.RuleTreeBaseStump import rule_dict
    >>> rule = rule_dict("X_3 <= 0.8", "X_3 > 0.8", feature_idx=3, threshold=0.8)
    >>> rule["textual_rule"], rule["not_graphviz_rule"]
    ('X_3 <= 0.8', {'label': 'X_3 > 0.8'})
    """
    return attributes | {
        "textual_rule": condition,
        "blob_rule": condition,
        "graphviz_rule": {"label": condition},
        "not_textual_rule": negation,
        "not_blob_rule": negation,
        "not_graphviz_rule": {"label": negation},
    }


def numerical_rule(feature_idx, feature_name: str, threshold, float_precision=3, **attributes) -> dict:
    """
    Dictionary of a numerical rule, ``feature_name <= threshold`` (see rule_dict).

    Parameters
    ----------
    feature_idx : object
        Identifier of the feature, stored in ``feature_idx``.
    feature_name : str
        Text of the feature in the rule, e.g. ``"X_3"`` or ``"P_12"``.
    threshold : float
        Threshold of the split.
    float_precision : int, default=3
        Number of decimals of the threshold in the text; None for no rounding.
    **attributes : dict
        Other keys of the rule.

    Returns
    -------
    rule : dict

    Examples
    --------
    >>> from RuleTree.base.RuleTreeBaseStump import numerical_rule
    >>> numerical_rule(112, "P_112", 3.69187, float_precision=2)["not_textual_rule"]
    'P_112 > 3.69'
    """
    value = str(threshold) if float_precision is None else round(threshold, float_precision)
    return rule_dict(f"{feature_name} <= {value}", f"{feature_name} > {value}", feature_idx=feature_idx,
                     threshold=threshold, is_categorical=False, feature_name=feature_name, **attributes)


def tree_arguments(method, context=None, node_id=None) -> dict:
    """
    The arguments ``context`` (the tree) and ``node_id`` (the node being split) for a ``fit`` method,
    if it declares them (or accepts ``**kwargs``): components written without them keep working.

    Parameters
    ----------
    method : callable
        The method that may receive them, e.g. ``transformer.fit``.
    context : RuleTree, default=None
        The tree.
    node_id : str, default=None
        Identifier of the node.

    Returns
    -------
    arguments : dict
        The keyword arguments to pass.

    Examples
    --------
    >>> from RuleTree.base.RuleTreeBaseStump import tree_arguments
    >>> def fit(X, y, node_id=None):
    ...     pass
    >>> tree_arguments(fit, context="tree", node_id="Rl")
    {'node_id': 'Rl'}
    """
    parameters = inspect.signature(method).parameters
    any_keyword = any(parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in parameters.values())
    return {name: value for name, value in (("context", context), ("node_id", node_id))
            if any_keyword or name in parameters}
