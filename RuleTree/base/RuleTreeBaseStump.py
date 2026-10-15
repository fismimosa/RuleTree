"""Abstract base class of the stumps: the single splits that make up the nodes of a RuleTree."""

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
    embeddings computed on the whole data set. Each kind of stump has its own cache, named after
    its class: see :meth:`_get_cache`. Everything needed by ``apply`` must be stored in the stump
    itself.

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

    See Also
    --------
    RuleTree.stumps.Stump.Stump : a stump made of a transformer and a splitter, the simplest way to
        write a new stump (see :doc:`/guides/extending`).
    """

    supported_data_types = (DATA_TYPE_TABULAR,)

    @abstractmethod
    def fit(self, data, y, idx=None, context=None, data_type=None, sample_weight=None, check_input=True):
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

    def _get_cache(self, context) -> dict:
        """
        Cache shared by the stumps of this class in the tree ``context`` (namespace: the class name).

        Parameters
        ----------
        context : RuleTree or None
            The tree being built.

        Returns
        -------
        cache : dict
            The cache of the stump's class; without a context, an empty dictionary that is not kept.
        """
        return context.get_cache(self.__class__.__name__) if context is not None else {}


def _is_number(value) -> bool:
    """
    Whether a value is a number (bool excluded), e.g. a numerical threshold.

    Parameters
    ----------
    value : object
        Any value.

    Returns
    -------
    is_number : bool
    """
    return isinstance(value, (int, float, np.number)) and not isinstance(value, bool)
