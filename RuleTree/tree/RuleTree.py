"""
Base class of the RuleTree models: a decision tree built node by node, where each node is split
by a stump chosen among the base stumps.
"""

import heapq
import importlib
import json
import numbers
import random
import tempfile
import traceback
import warnings
from abc import ABC, abstractmethod
from collections.abc import Sequence
from itertools import count
from typing import Optional

import numpy as np
import sklearn
from sklearn.base import ClassifierMixin
from sklearn.exceptions import NotFittedError
from sklearn.utils import check_random_state

from RuleTree.base.RuleTreeBase import RuleTreeBase
from RuleTree.exceptions import NoSplitFoundWarning
from RuleTree.tree.RuleTreeNode import RuleTreeNode, stump_class
from RuleTree.base.RuleTreeBaseStump import RuleTreeBaseStump
from RuleTree.utils.data_utils import json_numpy_encoder


class RuleTree(RuleTreeBase, ABC):
    """
    Base class of the RuleTree models.

    A RuleTree is a binary decision tree built best-first: the nodes are split one at a time, in the
    order given by ``queue_push``, until a stopping criterion is met. Each split is a *stump* (see
    :class:`RuleTree.base.RuleTreeBaseStump.RuleTreeBaseStump`) chosen among ``base_stumps``, so the
    same tree can mix different kinds of splits (thresholds on features, distances from pivots,
    shapelets, ...) and different data types (tabular data, time series, images).

    The subclasses define the task: :class:`RuleTreeClassifier`, :class:`RuleTreeRegressor`,
    :class:`RuleTreeCluster` and its variants.

    Parameters
    ----------
    max_leaf_nodes : int, default=None
        Maximum number of leaves. If None, unlimited.
    min_samples_split : int
        Minimum number of instances to split a node.
    max_depth : int
        Maximum depth of the tree (the root has depth 0).
    prune_useless_leaves : bool
        Whether to merge, after fit, the sibling leaves with the same prediction.
    base_stumps : RuleTreeBaseStump or list
        Stumps used to split the nodes. One of:

        - a stump;
        - a list of stumps, chosen with equal probability;
        - a list of tuples ``(probability, stump)`` or ``(probability, stump, data_types)``, where the
          probabilities sum to 1 and ``data_types`` (a data type or a list of data types, see
          :data:`RuleTree.utils.define.DATA_INPUT_TYPES`) restricts the data the stump is used on.
          By default, a stump is used on every data type it supports. The probability of a stump is
          split equally among its data types.
    stump_selection : {'random', 'best'}
        How the stump of each node is chosen: at random according to the probabilities, or the one
        with the highest gain among all the stumps and data types.
    random_state : int, RandomState instance or None
        Seed of the random choice of the stumps, also given to the stumps whose random_state is None
        (so that they do not break ties at random).
    exceptions : {'raise', 'ignore'}, default='raise'
        What to do when a stump raises an error: raise it, or make the node a leaf and warn.

    Attributes
    ----------
    root : RuleTreeNode
        Root of the fitted tree.
    candidate_splits_ : list of tuple
        Pairs (stump, data type) usable on the training data, with their probability:
        ``(probability, stump, data_type)``.
    cache_ : dict
        Values cached by the stumps during fit (see :meth:`get_cache`).

    See Also
    --------
    RuleTreeClassifier, RuleTreeRegressor, RuleTreeCluster : the trees for each task.
    RuleTree.base.RuleTreeBaseStump.RuleTreeBaseStump : the interface of the stumps.
    """
    def __init__(self,
                 max_leaf_nodes,
                 min_samples_split,
                 max_depth,
                 prune_useless_leaves,
                 base_stumps,
                 stump_selection, # ['random', 'best']
                 random_state,
                 exceptions='raise'
                 ):
        """Initialize the RuleTree; see the class documentation for the parameters."""
        self.tiebreaker = count()
        self.root:Optional[RuleTreeNode] = None
        self.queue = []

        self.max_leaf_nodes = float("inf") if max_leaf_nodes is None else max_leaf_nodes
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.prune_useless_leaves = prune_useless_leaves

        self.base_stumps = base_stumps
        self.stump_selection = stump_selection

        self.random_state = random_state
        if exceptions not in ['raise', 'ignore']:
            raise ValueError("exception must me either raise or ignore")
        self.exceptions = exceptions

    def _parse_base_stumps(self) -> list[tuple[float, RuleTreeBaseStump, Optional[tuple]]]:
        """
        Normalize ``base_stumps`` into a list of ``(probability, stump, data_types)``.

        Returns
        -------
        base_stumps : list of tuple
            ``data_types`` is a tuple of data types, or None when the stump can be used on every data
            type it supports.

        Raises
        ------
        TypeError
            If an entry is not a stump for the task of the tree (see _get_stumps_task).
        ValueError
            If an entry is malformed, a stump does not support the requested data types, or the
            probabilities do not sum to 1.
        """
        task = self._get_stumps_task()
        entries = self.base_stumps if isinstance(self.base_stumps, list) else [self.base_stumps]
        if len(entries) == 0:
            raise ValueError("base_stumps is empty.")

        parsed = []
        for entry in entries:
            if isinstance(entry, RuleTreeBaseStump):
                p, stump, data_types = 1 / len(entries), entry, None
            elif isinstance(entry, Sequence) and len(entry) in (2, 3):
                p, stump = entry[0], entry[1]
                data_types = entry[2] if len(entry) == 3 else None
            else:
                raise ValueError(f"Invalid base_stumps entry {entry!r}: expected a stump, (probability, stump) "
                                 f"or (probability, stump, data_types).")

            if not isinstance(stump, RuleTreeBaseStump) or stump.task != task:
                raise TypeError(f"{stump!r} is not a stump for {task} (task {getattr(stump, 'task', None)}).")

            if data_types is not None:
                data_types = (data_types,) if isinstance(data_types, str) else tuple(data_types)
                unsupported = [t for t in data_types if not stump.supports(t)]
                if unsupported:
                    raise ValueError(f"{stump.__class__.__name__} does not support {unsupported} data, "
                                     f"only {list(stump.supported_data_types)}.")

            parsed.append((p, stump, data_types))

        if not np.isclose(sum(p for p, _, _ in parsed), 1.):
            raise ValueError("The probabilities in base_stumps must sum to 1.")

        return parsed

    def _set_candidate_splits(self, data: dict):
        """
        Compute the pairs (stump, data type) that can split ``data``, with their selection probability,
        and store them in ``candidate_splits_``. The stumps receive the seed of the tree (see _seeded).

        Parameters
        ----------
        data : dict
            Whole data set, as ``{data_type: array}``.

        Raises
        ------
        ValueError
            If no stump supports the provided data.

        Warns
        -----
        UserWarning
            For each provided data type that no stump uses.
        """
        candidates = []
        seed = self._seed if hasattr(self, "_seed") else self._stumps_seed()
        for p, stump, data_types in self._parse_base_stumps():
            stump = self._seeded(stump, seed)
            usable = [t for t in (data_types or stump.supported_data_types) if t in data]
            candidates += [(p / len(usable), stump, t) for t in usable]

        if not candidates:
            raise ValueError(f"None of the base stumps supports the provided data ({', '.join(data)}).")

        for data_type in data.keys() - {t for _, _, t in candidates}:
            warnings.warn(f"No base stump uses the {data_type} data: it will be ignored.", UserWarning)

        total = sum(p for p, _, _ in candidates)
        self.candidate_splits_ = [(p / total, stump, t) for p, stump, t in candidates]

    def _stumps_seed(self):
        """
        Seed of the random choices of the tree and of the stumps without a random_state.

        Returns
        -------
        seed : int or None
            The tree's random_state if it is None or an int; an int drawn from it if it is a RandomState.
        """
        if self.random_state is None or isinstance(self.random_state, numbers.Integral):
            return self.random_state
        return int(check_random_state(self.random_state).randint(np.iinfo(np.int32).max))

    @staticmethod
    def _seeded(stump: RuleTreeBaseStump, seed) -> RuleTreeBaseStump:
        """
        The stump, or a copy of it where every random_state left to None (also of its components) is
        set to ``seed``: with the tree's random_state, the stumps do not break ties at random.

        Parameters
        ----------
        stump : RuleTreeBaseStump
            A base stump.
        seed : int or None
            Seed of the tree. If None, the stump is returned as it is.

        Returns
        -------
        stump : RuleTreeBaseStump
        """
        if seed is None:
            return stump

        unset = {name: seed for name, value in stump.get_params(deep=True).items()
                 if (name == "random_state" or name.endswith("__random_state")) and value is None}
        return sklearn.clone(stump).set_params(**unset) if unset else stump

    def _get_random_split(self) -> tuple[RuleTreeBaseStump, str]:
        """
        Draw a pair (stump, data type) from ``candidate_splits_`` according to their probabilities.

        Returns
        -------
        stump : RuleTreeBaseStump
            An unfitted copy of the stump.
        data_type : str
            The data type to use.
        """
        _, stump, data_type = self._random.choices(self.candidate_splits_,
                                                    weights=[p for p, _, _ in self.candidate_splits_])[0]
        return sklearn.clone(stump), data_type

    def get_cache(self, namespace: str) -> dict:
        """
        Dictionary where the stumps store values shared among the splits, e.g. embeddings of the whole
        data set. The cache is cleared at every :meth:`fit`, kept by :meth:`partial_fit`, and can be
        cleared manually with :meth:`clear_cache`.

        Parameters
        ----------
        namespace : str
            Name of the cache. Each transformer uses its ``cache_namespace()``: its class name and a hash
            of its hyperparameters, e.g. ``get_cache(transformer.cache_namespace())`` (the same for
            the stumps that implement ``fit`` without a transformer).

        Returns
        -------
        cache : dict
            The cache of the namespace (created if missing).
        """
        return self._stump_cache().setdefault(namespace, {})

    def _stump_cache(self) -> dict:
        """
        Cache of the tree passed to the stumps and to their components (created if missing): a
        dictionary namespace -> dictionary of values (see :meth:`get_cache`).

        Returns
        -------
        cache : dict
        """
        if not hasattr(self, "cache_"):
            self.cache_ = {}

        return self.cache_

    def clear_cache(self):
        """Remove all the values cached by the stumps (see :meth:`get_cache`)."""
        self.cache_ = {}

    def _set_classes(self, y):
        """
        Set ``classes_`` and ``n_classes_``, the classes of the target: only for classification trees.

        Parameters
        ----------
        y : numpy.ndarray
            Target.
        """
        if isinstance(self, ClassifierMixin):
            self.classes_ = np.unique(y)
            self.n_classes_ = len(self.classes_)

    def fit(self, X:np.ndarray|None=None, y:np.ndarray|None=None,
            X_ts:np.ndarray|None=None, X_img:np.ndarray|None=None, X_txt:np.ndarray|None=None, sample_weight=None):
        """
        Fit the tree on the provided data, starting from an empty tree.

        At least one of X, X_ts, X_img and X_txt must be given; all the inputs refer to the same
        instances. Each argument is a data type (see :data:`RuleTree.utils.define.DATA_INPUT_TYPES`):
        the stumps are used on the data types they support.

        Parameters
        ----------
        X : numpy.ndarray of shape (n_samples, n_features), default=None
            Tabular data.
        y : numpy.ndarray of shape (n_samples,), default=None
            Target (None for clustering).
        X_ts : numpy.ndarray of shape (n_samples, n_signals, length), default=None
            Time series.
        X_img : numpy.ndarray of shape (n_samples, ...), default=None
            Images.
        X_txt : array-like of shape (n_samples,), default=None
            Texts.
        sample_weight : numpy.ndarray of shape (n_samples,), default=None
            Sample weights, passed to the stumps.

        Returns
        -------
        self : RuleTree
            The fitted tree.

        Raises
        ------
        ValueError
            If no input is given, the inputs have different numbers of instances, or no stump supports
            the provided data.
        """
        y = self._check_target(y)
        data = self._validate_inputs(X=X, y=y, X_ts=X_ts, X_img=X_img, X_txt=X_txt)

        self.clear_cache()
        self.tiebreaker = count()
        self.queue = []
        self._seed = self._stumps_seed()
        self._random = random.Random(self._seed)
        self._set_classes(y)
        self._set_candidate_splits(data)

        idx = np.arange(self._n_samples(data))
        self.root = self.prepare_node(y=y, idx=idx, node_id="R")
        self.queue_push(self.root, idx)

        self._grow(data, y, sample_weight)

        return self

    def partial_fit(self, X:np.ndarray|None=None, y:np.ndarray|None=None,
                    X_ts:np.ndarray|None=None, X_img:np.ndarray|None=None, X_txt:np.ndarray|None=None,
                    sample_weight=None):
        """
        Continue growing the fitted tree: the current leaves are split further using the provided data,
        within the limits of max_depth and max_leaf_nodes (e.g. after increasing them with set_params).
        The existing splits are not changed. If the tree is not fitted, this is :meth:`fit`.

        The cache of the stumps (see :meth:`get_cache`) is kept: call :meth:`clear_cache` first if the
        data differ from those used to fill it.

        Parameters
        ----------
        X : numpy.ndarray of shape (n_samples, n_features), default=None
            Tabular data.
        y : numpy.ndarray of shape (n_samples,), default=None
            Target (None for clustering).
        X_ts : numpy.ndarray of shape (n_samples, n_signals, length), default=None
            Time series.
        X_img : numpy.ndarray of shape (n_samples, ...), default=None
            Images.
        X_txt : array-like of shape (n_samples,), default=None
            Texts.
        sample_weight : numpy.ndarray of shape (n_samples,), default=None
            Sample weights, passed to the stumps.

        Returns
        -------
        self : RuleTree
            The fitted tree.
        """
        if self.root is None:
            return self.fit(X=X, y=y, X_ts=X_ts, X_img=X_img, X_txt=X_txt, sample_weight=sample_weight)

        y = self._check_target(y)
        data = self._validate_inputs(X=X, y=y, X_ts=X_ts, X_img=X_img, X_txt=X_txt)

        if not hasattr(self, "_seed"):
            self._seed = self._stumps_seed()
        if not hasattr(self, "_random"):
            self._random = random.Random(self._seed)
        if not hasattr(self, "classes_"):
            self._set_classes(y)
        self._set_candidate_splits(data)

        self.queue = []
        self._push_leaves(self.root, data, np.arange(self._n_samples(data)))

        self._grow(data, y, sample_weight)

        return self

    def _push_leaves(self, node: RuleTreeNode, data: dict, idx: np.ndarray):
        """
        Push the leaves of the subtree rooted at ``node`` into the queue, with the instances reaching them.

        Parameters
        ----------
        node : RuleTreeNode
            Root of the subtree.
        data : dict
            Whole data set, as ``{data_type: array}``.
        idx : numpy.ndarray
            Indices of the instances reaching ``node``.
        """
        if node.is_leaf():
            self.queue_push(node, idx)
            return

        labels = node.stump.apply(data, idx, context=self, node_id=node.node_id, cache=self._stump_cache())
        self._push_leaves(node.node_l, data, idx[labels == 1])
        self._push_leaves(node.node_r, data, idx[labels == 2])

    def _grow(self, data: dict, y: np.ndarray|None, sample_weight=None):
        """
        Split the nodes in the queue until a stopping criterion is met, then prune the useless leaves
        (if requested) and call _post_fit_fix.

        Parameters
        ----------
        data : dict
            Whole data set, as ``{data_type: array}``.
        y : numpy.ndarray or None
            Target of the whole data set.
        sample_weight : numpy.ndarray, default=None
            Sample weights of the whole data set.
        """
        nbr_curr_nodes = 0
        y_split = self._split_target(y)  # the target seen by the stumps; the nodes use y
        while len(self.queue) > 0 and nbr_curr_nodes + len(self.queue) < self.max_leaf_nodes:
            idx, current_node = self.queue_pop()

            if (len(idx) < self.min_samples_split
                    or nbr_curr_nodes + len(self.queue) + 1 >= self.max_leaf_nodes
                    or (self.max_depth is not None and current_node.get_depth() >= self.max_depth)
                    or self.check_additional_halting_condition(y=y, curr_idx=idx)):
                self.make_leaf(current_node)
                nbr_curr_nodes += 1
                continue

            try:
                clf = self.make_split(data, y_split, idx, sample_weight=sample_weight,
                                      node_id=current_node.node_id)
                labels = clf.apply(data, idx, context=self, node_id=current_node.node_id, cache=self._stump_cache())
            except NoSplitFoundWarning:
                self.make_leaf(node=current_node)
                nbr_curr_nodes += 1
                continue
            except (ValueError, AttributeError, IndexError) as e:
                if self.exceptions == 'raise':
                    raise e
                self.make_leaf(node=current_node)
                nbr_curr_nodes += 1
                warning = RuntimeWarning(*e.args)
                warning.with_traceback(e.__traceback__)
                warnings.warn(warning)
                continue

            if self.is_split_useless(data, idx, labels):
                self.make_leaf(current_node)
                nbr_curr_nodes += 1
                continue

            self._set_balance_score(current_node, clf, data)

            idx_l, idx_r = idx[labels == 1], idx[labels == 2]

            current_node.set_stump(clf)
            current_node.node_l = self.prepare_node(y, idx_l, current_node.node_id + "l", )
            current_node.node_r = self.prepare_node(y, idx_r, current_node.node_id + "r", )
            current_node.node_l.parent, current_node.node_r.parent = current_node, current_node

            self.queue_push(current_node.node_l, idx_l)
            self.queue_push(current_node.node_r, idx_r)

        self.queue = []

        if self.prune_useless_leaves:
            self.root = self.root.simplify()

        self._post_fit_fix()

    def _check_target(self, y):
        """
        The target as an array (None for clustering); called by fit, partial_fit and update_statistics.
        The regression trees also turn a target of shape (n_samples, 1) into (n_samples,), so that they
        predict a vector.

        Parameters
        ----------
        y : array-like or None

        Returns
        -------
        y : numpy.ndarray or None
        """
        return None if y is None else np.asarray(y)

    def _split_target(self, y):
        """
        Target given to the stumps to fit the splits. The default implementation is y itself; the
        regression tree weighs its targets (see RuleTreeRegressor's target_weights).

        Parameters
        ----------
        y : numpy.ndarray or None
            Target of the whole data set.

        Returns
        -------
        y_split : numpy.ndarray or None
        """
        return y

    def _set_balance_score(self, node: RuleTreeNode, stump: RuleTreeBaseStump, data: dict):
        """
        Set ``balance_score`` of a node: the fraction of the whole data set in the smaller side of its
        split (see RuleTreeClassifier.get_balanced_stumps).

        Parameters
        ----------
        node : RuleTreeNode
            The node being split.
        stump : RuleTreeBaseStump
            The stump of the node.
        data : dict
            Whole data set, as ``{data_type: array}``.
        """
        labels = stump.apply(data, context=self, node_id=node.node_id, cache=self._stump_cache())
        _, counts = np.unique(labels, return_counts=True)
        node.balance_score_global = counts.min() / counts.sum()
        node.balance_score = node.balance_score_global

    def make_split(self, data: dict, y: np.ndarray|None, idx: np.ndarray, sample_weight=None,
                   node_id: Optional[str] = None) -> RuleTreeBaseStump:
        """
        Fit the split of a node, choosing among ``candidate_splits_`` according to ``stump_selection``:
        a random pair (stump, data type), or the pair with the highest gain (``gain_``).

        Parameters
        ----------
        data : dict
            Whole data set, as ``{data_type: array}``.
        y : numpy.ndarray or None
            Target of the whole data set.
        idx : numpy.ndarray
            Indices of the instances in the node.
        sample_weight : numpy.ndarray, default=None
            Sample weights of the whole data set.
        node_id : str, default=None
            Identifier of the node, passed to the stumps.

        Returns
        -------
        stump : RuleTreeBaseStump
            The fitted stump.

        Raises
        ------
        NoSplitFoundWarning
            If no stump can split the node.
        """
        if self.stump_selection == 'random':
            stump, data_type = self._get_random_split()
            stump.fit(data, y, idx=idx, data_type=data_type, sample_weight=sample_weight,
                      context=self, node_id=node_id, cache=self._stump_cache())

            return stump

        elif self.stump_selection == 'best':
            clfs = []
            info_gains = []
            for _, stump, data_type in self.candidate_splits_:
                try:
                    stump = sklearn.clone(stump)
                    stump.fit(data, y, idx=idx, data_type=data_type, sample_weight=sample_weight,
                              context=self, node_id=node_id, cache=self._stump_cache())

                    info_gains.append(stump.gain_)
                    clfs.append(stump)
                except NoSplitFoundWarning:
                    continue
                except Exception as e:
                    if self.exceptions == 'raise':
                        raise e
                    traceback.print_exc()
                    continue

            if len(clfs) == 0:
                raise NoSplitFoundWarning('No splits found')

            return clfs[np.argmax(info_gains)]

        else:
            raise TypeError('Unknown stump selection method')

    def is_split_useless(self, data: dict, idx: np.ndarray, labels: np.ndarray) -> bool:
        """
        Whether a split is useless, i.e. all the instances go to the same child.

        Parameters
        ----------
        data : dict
            Whole data set, as ``{data_type: array}``.
        idx : numpy.ndarray
            Indices of the instances in the node.
        labels : numpy.ndarray
            Result of the split on idx (1 left, 2 right).

        Returns
        -------
        useless : bool
        """
        return len(np.unique(labels)) == 1

    def _update_node_statistics(self, node: RuleTreeNode, data: dict, y: np.ndarray, idx: np.ndarray):
        """
        Statistics of a node and of its stump on the instances ``idx`` (see :meth:`update_statistics`).
        By default the ones of classification (majority class, class probabilities, log-odds): see
        RuleTreeNode.update_statistics.

        Parameters
        ----------
        node : RuleTreeNode
            The node.
        data : dict
            Whole data set, as ``{data_type: array}``.
        y : numpy.ndarray
            Target of the whole data set.
        idx : numpy.ndarray
            Indices of the instances reaching the node.
        """
        node.update_statistics(data, y, idx, self, cache=self._stump_cache())

    def predict(self, X: np.ndarray = None, X_ts=None, X_img=None, X_txt=None):
        """
        Predict the output of each instance: the prediction of the leaf it reaches.

        Parameters
        ----------
        X : numpy.ndarray of shape (n_samples, n_features), default=None
            Tabular data.
        X_ts : numpy.ndarray of shape (n_samples, n_signals, length), default=None
            Time series.
        X_img : numpy.ndarray of shape (n_samples, ...), default=None
            Images.
        X_txt : array-like of shape (n_samples,), default=None
            Texts.

        Returns
        -------
        y_pred : numpy.ndarray of shape (n_samples,)
            Predicted classes, values or cluster labels.
        """
        labels, _, _ = self.root.predict(self._validate_inputs(X=X, X_ts=X_ts, X_img=X_img, X_txt=X_txt),
                                         context=self, cache=getattr(self, "cache_", None))

        return labels

    def apply(self, X: np.ndarray|None = None, X_ts=None, X_img=None, X_txt=None):
        """
        Leaf reached by each instance.

        Parameters
        ----------
        X : numpy.ndarray of shape (n_samples, n_features), default=None
            Tabular data.
        X_ts : numpy.ndarray of shape (n_samples, n_signals, length), default=None
            Time series.
        X_img : numpy.ndarray of shape (n_samples, ...), default=None
            Images.
        X_txt : array-like of shape (n_samples,), default=None
            Texts.

        Returns
        -------
        leaves : numpy.ndarray of shape (n_samples,)
            Identifier (node_id) of the leaf reached by each instance.
        """
        _, leaves, _ = self.root.predict(self._validate_inputs(X=X, X_ts=X_ts, X_img=X_img, X_txt=X_txt),
                                         context=self, cache=getattr(self, "cache_", None))

        return leaves

    def update_statistics(self, X: np.ndarray|None = None, y: np.ndarray|None = None,
                          X_ts=None, X_img=None, X_txt=None):
        """
        Recompute the statistics of the nodes (predictions, probabilities, number of instances, ...) and
        of their stumps on new data, without changing the splits. For classification, ``classes_`` become
        the classes of y (e.g. to re-label the tree with a new target). A node reached by no instance
        predicts nan.

        Parameters
        ----------
        X : numpy.ndarray of shape (n_samples, n_features), default=None
            Tabular data.
        y : numpy.ndarray of shape (n_samples,), default=None
            Target (None for clustering).
        X_ts : numpy.ndarray of shape (n_samples, n_signals, length), default=None
            Time series.
        X_img : numpy.ndarray of shape (n_samples, ...), default=None
            Images.
        X_txt : array-like of shape (n_samples,), default=None
            Texts.
        """
        y = self._check_target(y)
        data = self._validate_inputs(X=X, y=y, X_ts=X_ts, X_img=X_img, X_txt=X_txt)
        self._set_classes(y)

        self._update_statistics(data, y, self.root, np.arange(self._n_samples(data)))

    def _update_statistics(self, data: dict, y: np.ndarray, node: RuleTreeNode, idx: np.ndarray):
        """
        Recompute the statistics of the subtree rooted at ``node`` (see :meth:`update_statistics`).

        Parameters
        ----------
        data : dict
            Whole data set, as ``{data_type: array}``.
        y : numpy.ndarray
            Target of the whole data set.
        node : RuleTreeNode
            Root of the subtree.
        idx : numpy.ndarray
            Indices of the instances reaching ``node``.
        """
        if node is None:
            return

        node.classes = getattr(self, "classes_", None)

        self._update_node_statistics(node, data, y, idx)
        if node.is_leaf() or node.stump is None:
            return

        labels = node.stump.apply(data, idx, context=self, node_id=node.node_id, cache=self._stump_cache())
        idx_l = idx[labels == 1]
        idx_r = idx[labels == 2]

        self._update_statistics(data, y, node.node_l, idx_l)
        self._update_statistics(data, y, node.node_r, idx_r)

    def get_rules(self, columns_names = None):
        """
        Rules of the tree, as a nested dictionary.

        Parameters
        ----------
        columns_names : list of str, default=None
            Names of the tabular features, used in the rules instead of ``X_<index>``.

        Returns
        -------
        rules : dict
            Rule of the root: the node statistics (``node_id``, ``is_leaf``, ``prediction``, ...), the
            rule of the split (``textual_rule``, ``not_textual_rule``, ...) and the rules of the children
            in ``left_node`` and ``right_node`` (None for a leaf). See :meth:`print_rules`.
        """
        return self.root.get_rule(columns_names = columns_names)

    def make_leaf(self, node: RuleTreeNode) -> RuleTreeNode:
        """
        Hook called when a node becomes a leaf. The default implementation does nothing.

        Parameters
        ----------
        node : RuleTreeNode
            The new leaf.

        Returns
        -------
        node : RuleTreeNode
        """
        return node

    def queue_pop(self):
        """
        Pop the next node to split from the queue.

        Returns
        -------
        idx : numpy.ndarray
            Indices of the instances reaching the node.
        node : RuleTreeNode
        """
        el = heapq.heappop(self.queue)
        return el[-2:]

    def check_additional_halting_condition(self, y, curr_idx: np.ndarray):
        """
        Additional condition to stop splitting a node, defined by the subclasses (e.g. a pure node for
        classification). The default implementation never stops.

        Parameters
        ----------
        y : numpy.ndarray or None
            Target of the whole data set.
        curr_idx : numpy.ndarray
            Indices of the instances in the node.

        Returns
        -------
        halt : bool
        """
        return False

    def _post_fit_fix(self):
        """
        Hook called at the end of fit and partial_fit, e.g. to relabel the clusters. The default
        implementation does nothing.
        """

    @abstractmethod
    def prepare_node(self, y: np.ndarray|None, idx: np.ndarray, node_id: str, node: Optional[RuleTreeNode] = None) -> RuleTreeNode:
        """
        Create a node (or update ``node``) with the statistics of the instances ``idx``: prediction,
        probabilities, number of instances.

        Parameters
        ----------
        y : numpy.ndarray or None
            Target of the whole data set.
        idx : numpy.ndarray
            Indices of the instances reaching the node.
        node_id : str
            Identifier of the node: its path from the root, e.g. ``"Rlr"``.
        node : RuleTreeNode, default=None
            Existing node to update instead of creating a new one.

        Returns
        -------
        node : RuleTreeNode
        """
        pass

    def queue_push(self, node: RuleTreeNode, idx: np.ndarray):
        """
        Push a node to split into the queue: the priority defines the order in which the tree grows. By
        default, the shallowest nodes first (breadth-first growth).

        Parameters
        ----------
        node : RuleTreeNode
            The node.
        idx : numpy.ndarray
            Indices of the instances reaching the node.
        """
        heapq.heappush(self.queue, (len(node.node_id), next(self.tiebreaker), idx, node))

    @abstractmethod
    def _get_stumps_task(self) -> str:
        """
        Task of the stumps of the tree: the tree accepts only base stumps with this task.

        Returns
        -------
        task : str
            MODEL_TYPE_CLF or MODEL_TYPE_REG.
        """

    def _get_tree_paths(self, current_node = None):
        """
        Paths from each leaf to the root.

        Parameters
        ----------
        current_node : RuleTreeNode
            Root of the subtree.

        Returns
        -------
        paths : list of list of str
            For each leaf, the node ids from the leaf to ``current_node``.
        """
        if current_node is None:
            current_node = self.root

        if current_node.is_leaf():
            return [[current_node.node_id]]

        paths = self._get_tree_paths(current_node.node_l) + self._get_tree_paths(current_node.node_r)
        for path in paths:
            path.append(current_node.node_id)

        return paths
    
    def _node_dict(self, current_node=None, d=None, i=0):
        """
        Index of each node in depth-first order ('R': 0, 'Rl': 1, ...).

        Parameters
        ----------
        current_node : RuleTreeNode, default=None
            Root of the subtree. If None, the root of the tree.
        d : dict, default=None
            Dictionary being filled.
        i : int, default=0
            Next index.

        Returns
        -------
        d : dict
            Node id -> index.
        i : int
            Next free index.
        """
        if d is None:
            d = {}
            
        if current_node is None:
            current_node = self.root
    
        # Assign an index to the current node
        if current_node.node_id not in d:
            d[current_node.node_id] = i
            i += 1
    
        if not current_node.is_leaf():
            i = self._node_dict(current_node.node_l, d, i)[-1]
            i = self._node_dict(current_node.node_r, d, i)[-1]
    
        return d, i

    def _get_prediction_probas(self, current_node=None, probas=None):
        """
        Values of the nodes with a prediction in depth-first order (see _node_value), used by
        local_interpretation.

        Parameters
        ----------
        current_node : RuleTreeNode, default=None
            Root of the subtree. If None, the root of the tree.
        probas : list, default=None
            List being filled.

        Returns
        -------
        probas : list
        """
        if probas is None:
            probas = []

        if current_node is None:
            current_node = self.root

        if current_node.prediction is not None:
            probas.append(self._node_value(current_node))

        if current_node.node_l:
            self._get_prediction_probas(current_node.node_l, probas)
            self._get_prediction_probas(current_node.node_r, probas)

        return probas

    def _node_value(self, node: RuleTreeNode):
        """
        Value of a node in _get_prediction_probas: by default its prediction.

        Parameters
        ----------
        node : RuleTreeNode
            The node.

        Returns
        -------
        value : object
        """
        return node.prediction

    def _tree_value(self):
        """
        Values of the nodes in depth-first order (see _get_prediction_probas), as an array.

        Returns
        -------
        values : numpy.ndarray of shape (n_nodes, 1, n_values)
        """
        probas = self._get_prediction_probas(self.root)
        if isinstance(probas[0], (list, np.ndarray)):
            return np.array(probas).reshape(-1, 1, len(probas[0]))
        else:
            return np.array(probas).reshape(-1, 1, 1)
            
    def _tree_feature(self, current_node = None, feats=None):
        """
        Feature of each node in depth-first order (-2 for the leaves), used by local_interpretation.

        Parameters
        ----------
        current_node : RuleTreeNode, default=None
            Root of the subtree. If None, the root of the tree.
        feats : list, default=None
            List being filled.

        Returns
        -------
        feats : list
        """
        if feats is None:
            feats = []
            
        if current_node is None:
            current_node = self.root
        
        if current_node.is_leaf():
            feats.append(-2)
        else:
            feats.append(current_node.stump.feature_original)
            self._tree_feature(current_node.node_l, feats = feats)
            self._tree_feature(current_node.node_r, feats = feats)
        
        return feats
        
        
    def local_interpretation(self, X, joint_contribution = False):
        """
        Data needed to compute the contribution of each feature to the predictions (see the
        implementations of RuleTreeClassifier and RuleTreeRegressor, which return the contributions).

        Parameters
        ----------
        X : numpy.ndarray of shape (n_samples, n_features)
            Tabular data.
        joint_contribution : bool, default=False
            Not used here.

        Returns
        -------
        leaves : numpy.ndarray
            Index of the leaf reached by each instance.
        paths : list of list of int
            Path from the root to each leaf.
        leaf_to_path : dict
            Leaf index -> its path.
        values : numpy.ndarray
            Values of the nodes (see _tree_value).
        """
        node_dict = self._node_dict()[0] #-> Turn 'R': 0, 'Rl' : 1, 'Rr' :2 and so on

        leaves = np.array([node_dict[x] for x in self.apply(X)])
        paths = [[node_dict[x] for x in path] for path in self._get_tree_paths(self.root)]
        
        for path in paths:
            path.reverse()
        
        leaf_to_path = {}
        
        for path in paths:
            leaf_to_path[path[-1]] = path
            
        values = self._tree_value().squeeze(axis = 1)
        
        return leaves, paths, leaf_to_path, values
        
    
    def eval_contributions(self, 
                           leaves, 
                           paths, 
                           leaf_to_path, 
                           values, 
                           biases,
                           line_shape,
                           joint_contribution = False):
        """
        Contributions of the features to the predictions, as in treeinterpreter: the prediction of an
        instance is the value of the root (bias) plus the change of value along its path, attributed to
        the feature of each split.

        Parameters
        ----------
        leaves : numpy.ndarray
            Index of the leaf reached by each instance.
        paths : list of list of int
            Path from the root to each leaf.
        leaf_to_path : dict
            Leaf index -> its path.
        values : numpy.ndarray
            Values of the nodes.
        biases : numpy.ndarray
            Value of the root, for each instance.
        line_shape : int or tuple
            Shape of the contributions of one instance.
        joint_contribution : bool, default=False
            If True, the contributions are given to the sets of features along the path.

        Returns
        -------
        direct_prediction : numpy.ndarray
            Value of the leaf of each instance.
        biases : numpy.ndarray
            Value of the root.
        contributions : numpy.ndarray or list of dict
            Contribution of each feature (or set of features, with joint_contribution) to each prediction.
        """
        direct_prediction = values[leaves]
        values_list = list(values)
        feature_index = list(self._tree_feature())
        
        contributions = []
        if joint_contribution:
            for row, leaf in enumerate(leaves):
                path = leaf_to_path[leaf]
                
                
                path_features = set()
                contributions.append({})
                for i in range(len(path) - 1):
                    path_features.add(feature_index[path[i]])
                    contrib = values_list[path[i+1]] - \
                             values_list[path[i]]
                    contributions[row][tuple(sorted(path_features))] = \
                        contributions[row].get(tuple(sorted(path_features)), 0) + contrib
            return direct_prediction, biases, contributions
        
        else:
            unique_contributions = {}
            for leaf in np.unique(leaves):
                path = leaf_to_path[leaf]
                contribs = np.zeros(line_shape)
                for i in range(len(path) - 1):
                    contribs[feature_index[path[i]]] += values_list[path[i + 1]] - values_list[path[i]]
                unique_contributions[leaf] = contribs

            return direct_prediction, biases, np.array([unique_contributions[leaf] for leaf in leaves])


    
    
    
    @classmethod
    def print_rules(cls, rules: dict=None, digits=2, indent: int = 0, ):
        """
        Print rules as an indented list of conditions and outputs.

        Parameters
        ----------
        rules : dict
            Rules returned by :meth:`get_rules`.
        digits : int, default=2
            Number of decimals of the numerical outputs.
        indent : int, default=0
            Indentation level of the root.
        """
      
        indentation = "".join(["|   " for _ in range(indent)])
        
        if rules["is_leaf"]:
            pred = rules['prediction']
            print(f"{indentation} output: "
                  f"{pred if type(pred) in [np.str_, np.bytes_, str] else np.round(pred, digits)}")
            
        else:
            print(f"{indentation}|--- {rules['textual_rule']}")
            cls.print_rules(rules=rules['left_node'], digits=digits, indent=indent + 1)
            print(f"{indentation}|--- {rules['not_textual_rule']}")
            cls.print_rules(rules=rules['right_node'], digits=digits, indent=indent + 1)

    def to_dict(self, filename=None):
        """
        Dictionary representation of the fitted tree, from which :meth:`from_dict` rebuilds it.

        Parameters
        ----------
        filename : str or path-like, default=None
            If given, the dictionary is also saved to this file as JSON.

        Returns
        -------
        dictionary : dict
            JSON-friendly dictionary with the class of the tree, its parameters (``args``, base stumps
            included), the classes for classification trees and the nodes. The values may be NumPy
            scalars: to write it with ``json``, use ``cls=RuleTree.utils.data_utils.json_numpy_encoder``
            (as done with ``filename``). See :doc:`/guides/saving`.

        Raises
        ------
        sklearn.exceptions.NotFittedError
            If the tree is not fitted.
        """
        if self.root is None:
            raise NotFittedError(f"This {self.__class__.__name__} is not fitted: call fit before to_dict.")

        node_list = [self.root]

        # constructor parameters of the concrete class, so that from_dict can call its __init__
        args = {k: v for k, v in self.get_params(deep=False).items() if k != "base_stumps"}
        args["base_stumps"] = [
            (p, stump.node_to_dict(), data_types) for p, stump, data_types in self._parse_base_stumps()
        ]

        dictionary = {
            "tree_type": self.__class__.__module__,
            "tree_class": self.__class__.__name__,
            "args": args,
            "nodes": [],
        }
        if hasattr(self, "classes_"):
            dictionary["classes_"] = np.array(self.classes_).tolist()
            dictionary["n_classes_"] = self.n_classes_

        while len(node_list) > 0:
            node = node_list.pop()
            dictionary["nodes"].append(node.node_to_dict())
            if not node.is_leaf():
                node_list += [node.node_l, node.node_r]

        if filename is not None:
            with open(filename, 'w') as f:
                json.dump(dictionary, f, cls=json_numpy_encoder)

        return dictionary

    @classmethod
    def from_dict(cls, filename):
        """
        Rebuild a tree from a dictionary created by :meth:`to_dict`.

        Parameters
        ----------
        filename : dict, str or path-like
            The dictionary, or the JSON file where it is saved.

        Returns
        -------
        tree : RuleTree
            The tree, of the class it had when saved.
        """
        if not isinstance(filename, dict):
            with open(filename, 'r') as f:
                dictionary = json.load(f)
        else:
            dictionary = filename

        tree_class = getattr(importlib.import_module(dictionary['tree_type']), dictionary['tree_class'])
        args = dictionary['args'] | {'base_stumps': [
            (p, stump_class(stump_dict).dict_to_node(stump_dict), data_types)
            for p, stump_dict, data_types in dictionary['args']['base_stumps']
        ]}

        tree = tree_class(**args)
        if 'classes_' in dictionary:
            tree.classes_ = np.array(dictionary['classes_'])
            tree.n_classes_ = dictionary['n_classes_']

        nodes = {node["node_id"]: RuleTreeNode.dict_to_node(node) for node in dictionary['nodes']}

        for node_instance, node_info in zip(nodes.values(), dictionary['nodes']):
            if not node_info["is_leaf"]:
                node_instance.node_l = nodes[node_info["left_node"]]
                node_instance.node_r = nodes[node_info["right_node"]]
                nodes[node_info["left_node"]].parent = node_instance
                nodes[node_info["right_node"]].parent = node_instance

        tree.root = nodes['R']

        return tree

    def export_graphviz(self, columns_names=None, scaler=None, float_precision=3, filename:Optional[str]=None):
        """
        Graphviz representation of the tree (requires the graphviz package).

        Parameters
        ----------
        columns_names : list of str, default=None
            Names of the tabular features.
        scaler : TransformerMixin, default=None
            Scaler used to preprocess the data: the thresholds are shown in the original scale.
        float_precision : int, default=3
            Number of decimals of the thresholds.
        filename : str, default=None
            If given, the graph is rendered to this file; otherwise it is rendered to a temporary file
            and opened (both require the Graphviz binaries).

        Returns
        -------
        graph : graphviz.Digraph or None
            The graph, if filename is None.
        """
        dot = self.root.export_graphviz(columns_names=columns_names, scaler=scaler, float_precision=float_precision)
        if filename is None:
            dot.render(directory=tempfile.mkdtemp(), view=True)
            return dot
        else:
            dot.render(filename=filename)

    def plot_tree(self, ax=None, *, columns_names=None, class_names=None, scaler=None, float_precision=3,
                  max_depth=None, node_ids=True, samples=True, filled=True, fontsize=None, colors=None):
        """
        Draw the tree with matplotlib, as ``sklearn.tree.plot_tree`` (requires matplotlib).

        Each internal node shows its id, its rule (for the shapelet and proximity stumps, also a plot
        of the shapelets) and its number of training instances; each leaf its prediction. The left
        child of a node (arrow "true") gets the instances for which the rule is true. Unlike
        ``export_graphviz``, it needs neither the Graphviz programs nor files: the drawing is a
        matplotlib figure, shown in notebooks and saved with ``savefig``.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, default=None
            Axes to draw on. If None, a new figure, sized on the number of leaves and the depth.
        columns_names : list of str, default=None
            Names of the tabular features.
        class_names : list of str, default=None
            Names of the classes, in the order of ``classes_`` (classification trees).
        scaler : TransformerMixin, default=None
            Scaler used to preprocess the data: the thresholds are shown in the original scale.
        float_precision : int, default=3
            Number of decimals of the thresholds and of the predicted values.
        max_depth : int, default=None
            Maximum depth drawn: the subtrees below are shown as ``(...)``. If None, the whole tree.
        node_ids : bool, default=True
            Whether to show the id of each node.
        samples : bool, default=True
            Whether to show the number of training instances of each node.
        filled : bool, default=True
            Whether to color the leaves by class, cluster or predicted value.
        fontsize : float, default=None
            Size of the text. If None, the largest size with which the boxes do not overlap.
        colors : dict, default=None
            Colors replacing the ones of ``RuleTree.utils.plotting.DEFAULT_COLORS`` (e.g. for a dark
            background): ``text``, ``split``, ``split_edge``, ``leaf``, ``leaf_edge``, ``edge``,
            ``classes`` (a list) and ``values`` (a colormap).

        Returns
        -------
        ax : matplotlib.axes.Axes
            The axes of the drawing. The box of each node is a text of ``ax.texts`` whose gid is the
            id of the node.

        Examples
        --------
        >>> import matplotlib
        >>> matplotlib.use("Agg")
        >>> from sklearn.datasets import load_iris
        >>> from RuleTree import RuleTreeClassifier
        >>> iris = load_iris()
        >>> model = RuleTreeClassifier(max_depth=2, random_state=0).fit(iris.data, iris.target)
        >>> ax = model.plot_tree(columns_names=iris.feature_names, class_names=list(iris.target_names))
        >>> [text.get_text().split("\\n")[1] for text in ax.texts if text.get_gid() in ("R", "Rl")]
        ['petal width (cm) <= 0.8', 'class = setosa']
        """
        return self.root.plot_tree(ax, columns_names=columns_names, class_names=class_names, scaler=scaler,
                                   float_precision=float_precision, max_depth=max_depth, node_ids=node_ids,
                                   samples=samples, filled=filled, fontsize=fontsize, colors=colors)

    def get_predicates(self):
        """
        Internal nodes (the ones with a split) in depth-first order.

        Returns
        -------
        predicates : dict
            Node id -> RuleTreeNode.
        """
        return self.root.get_predicates()

    def get_node_by_id(self, node_id):
        """
        Node with the given identifier.

        Parameters
        ----------
        node_id : str
            Identifier of the node: its path from the root, e.g. ``"Rlr"``.

        Returns
        -------
        node : RuleTreeNode or None
            The node, or None if not in the tree.
        """
        return self.root.get_node_by_id(node_id)

    def get_leaf_nodes(self):
        """
        Leaves of the tree.

        Returns
        -------
        leaves : dict
            Node id -> RuleTreeNode.
        """
        return self.root.get_leaf_nodes()
