"""
Module containing the RuleTree class.

The RuleTree class implements a decision tree structure with additional functionality
for handling custom stumps and local interpretation.
"""

import heapq
import importlib
import json
import numbers
import random
import traceback
import warnings
from abc import ABC, abstractmethod
from collections.abc import Sequence
from itertools import count
from typing import Optional

import numpy as np
import sklearn
from sklearn.base import ClassifierMixin
from sklearn.utils import check_random_state
from sklearn.metrics import pairwise_distances
from tempfile312 import TemporaryDirectory

from RuleTree.base.RuleTreeBase import RuleTreeBase
from RuleTree.exceptions import NoSplitFoundWarning
from RuleTree.tree.RuleTreeNode import RuleTreeNode
from RuleTree.base.RuleTreeBaseStump import RuleTreeBaseStump
from RuleTree.utils.data_utils import json_numpy_encoder


class RuleTree(RuleTreeBase, ABC):
    """
    Base class for RuleTree models.

    This class provides the core implementation of a RuleTree, including methods for
    fitting, predicting, and exporting the tree structure. It supports custom stumps
    and advanced features like local interpretation.

    Attributes:
        max_leaf_nodes (int): Maximum number of leaf nodes in the tree.
        min_samples_split (int): Minimum number of samples required to split a node.
        max_depth (int): Maximum depth of the tree.
        prune_useless_leaves (bool): Whether to prune leaves that do not contribute.
        base_stumps (list): List of base stumps used for splitting.
        stump_selection (str): Method for selecting stumps ('random' or 'best').
        random_state (int): Random seed for reproducibility.
    """
    def __init__(self,
                 max_leaf_nodes,
                 min_samples_split,
                 max_depth,
                 prune_useless_leaves,
                 base_stumps,
                 stump_selection, # ['random', 'best']
                 random_state,
                 distance_measure=None,
                 exceptions='raise'
                 ):
        """
        Initialize the RuleTree.

        Args:
            max_leaf_nodes (int): Maximum number of leaf nodes.
            min_samples_split (int): Minimum samples required to split a node.
            max_depth (int): Maximum depth of the tree.
            prune_useless_leaves (bool): Whether to prune useless leaves.
            base_stumps: Stumps used to split the nodes. One of:
                - a stump;
                - a list of stumps, chosen with equal probability;
                - a list of tuples ``(probability, stump)`` or ``(probability, stump, data_types)``, where
                  the probabilities sum to 1 and ``data_types`` (a data type or a list of data types, see
                  :data:`RuleTree.utils.define.DATA_INPUT_TYPES`) restricts the data the stump is used on.
                  By default, a stump is used on every data type it supports. The probability of a stump
                  is split equally among its data types.
            stump_selection (str): Stump selection strategy ('random' or 'best').
            random_state (int): Random seed for reproducibility: used to choose the stumps (stump_selection=
                'random') and given to the stumps whose random_state is None.
        """
        self.tiebreaker = count()
        self.root:Optional[RuleTreeNode] = None
        self.queue = []

        self.max_leaf_nodes = float("inf") if max_leaf_nodes is None else max_leaf_nodes
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.prune_useless_leaves = prune_useless_leaves
        self.distance_measure = distance_measure

        self.base_stumps = base_stumps
        self.stump_selection = stump_selection

        self.random_state = random_state
        if exceptions not in ['raise', 'ignore']:
            raise ValueError("exception must me either raise or ignore")
        self.exceptions = exceptions

    def _parse_base_stumps(self) -> list[tuple[float, RuleTreeBaseStump, Optional[tuple]]]:
        """
        Normalize ``base_stumps`` into a list of ``(probability, stump, data_types)``.

        ``data_types`` is a tuple of data types, or None when the stump can be used on every data type
        it supports.

        Raises:
            TypeError: If an entry is not a stump for the task of the tree (see _get_stumps_task).
            ValueError: If an entry is malformed, a stump does not support the requested data types, or
                the probabilities do not sum to 1.
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
        and store them in ``candidate_splits_``.

        Raises:
            ValueError: If no stump supports the provided data.

        Warns:
            UserWarning: For each provided data type that no stump uses.
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
        Seed of the random choices of the tree and of the stumps without a random_state, from the tree's
        random_state: an int is used as is, from a RandomState an int is drawn.
        """
        if self.random_state is None or isinstance(self.random_state, numbers.Integral):
            return self.random_state
        return int(check_random_state(self.random_state).randint(np.iinfo(np.int32).max))

    @staticmethod
    def _seeded(stump: RuleTreeBaseStump, seed) -> RuleTreeBaseStump:
        """
        The stump, or a copy of it where every random_state left to None (also of its components)
        is set to seed: with the tree's random_state, the stumps do not break ties at random.
        """
        if seed is None:
            return stump

        unset = {name: seed for name, value in stump.get_params(deep=True).items()
                 if (name == "random_state" or name.endswith("__random_state")) and value is None}
        return sklearn.clone(stump).set_params(**unset) if unset else stump

    def _get_random_split(self) -> tuple[RuleTreeBaseStump, str]:
        """
        Draw a pair (stump, data type) from ``candidate_splits_`` according to their probabilities.

        Returns:
            tuple[RuleTreeBaseStump, str]: An unfitted copy of the stump and the data type to use.
        """
        _, stump, data_type = self._random.choices(self.candidate_splits_,
                                                    weights=[p for p, _, _ in self.candidate_splits_])[0]
        return sklearn.clone(stump), data_type

    def get_cache(self, namespace: str) -> dict:
        """
        Dictionary where the stumps can store values shared among the splits, e.g. embeddings of the
        whole data set. The cache is cleared at every :meth:`fit`, kept by :meth:`partial_fit`, and can
        be cleared manually with :meth:`clear_cache`.

        Args:
            namespace (str): Name of the cache. Stumps use their class name (see
                ``RuleTreeBaseStump._get_cache``), e.g. ``get_cache("PivotTreeStumpClassifier")``.

        Returns:
            dict: The cache of the namespace.
        """
        if not hasattr(self, "cache_"):
            self.cache_ = {}

        return self.cache_.setdefault(namespace, {})

    def clear_cache(self):
        """Remove all the values cached by the stumps (see :meth:`get_cache`)."""
        self.cache_ = {}

    def _set_classes(self, y):
        """Set classes_ and n_classes_, the classes of the target: only for classification trees."""
        if isinstance(self, ClassifierMixin):
            self.classes_ = np.unique(y)
            self.n_classes_ = len(self.classes_)

    def fit(self, X:np.ndarray|None=None, y:np.ndarray|None=None,
            X_ts:np.ndarray|None=None, X_img:np.ndarray|None=None, X_txt:np.ndarray|None=None, sample_weight=None):
        """
        Fit the RuleTree to the provided data, starting from an empty tree.

        Args:
            X (np.ndarray): Tabular data.
            y (np.ndarray): Target.
            X_ts (np.ndarray): Time series.
            X_img (np.ndarray): Images.
            X_txt (np.ndarray): Texts.
            sample_weight (np.ndarray): Sample weights.

        Returns:
            RuleTree: The fitted RuleTree instance.
        """
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
        Continue growing the fitted tree: the current leaves are split further using the provided
        data, within the limits of max_depth and max_leaf_nodes (e.g. after increasing them with
        set_params). The existing splits are not changed. If the tree is not fitted, this is :meth:`fit`.

        The cache of the stumps (see :meth:`get_cache`) is kept: call :meth:`clear_cache` first if the
        data differ from those used to fill it.

        Args:
            X (np.ndarray): Tabular data.
            y (np.ndarray): Target.
            X_ts (np.ndarray): Time series.
            X_img (np.ndarray): Images.
            X_txt (np.ndarray): Texts.
            sample_weight (np.ndarray): Sample weights.

        Returns:
            RuleTree: The fitted RuleTree instance.
        """
        if self.root is None:
            return self.fit(X=X, y=y, X_ts=X_ts, X_img=X_img, X_txt=X_txt, sample_weight=sample_weight)

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
        """Push the leaves of the subtree rooted at node into the queue, with the instances reaching them."""
        if node.is_leaf():
            self.queue_push(node, idx)
            return

        labels = node.stump.apply(data, idx)
        self._push_leaves(node.node_l, data, idx[labels == 1])
        self._push_leaves(node.node_r, data, idx[labels == 2])

    def _grow(self, data: dict, y: np.ndarray|None, sample_weight=None):
        """
        Split the nodes in the queue until a stopping condition is met.

        Args:
            data (dict): Whole data set, as ``{data_type: array}``.
            y (np.ndarray): Target of the whole data set.
            sample_weight (np.ndarray): Sample weights of the whole data set.
        """
        nbr_curr_nodes = 0
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
                clf = self.make_split(data, y, idx, sample_weight=sample_weight)
                labels = clf.apply(data, idx)
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

    @staticmethod
    def _set_balance_score(node: RuleTreeNode, stump: RuleTreeBaseStump, data: dict):
        """Fraction of the whole data set in the smaller side of the split (see get_balanced_stumps)."""
        _, counts = np.unique(stump.apply(data), return_counts=True)
        node.balance_score_global = counts.min() / counts.sum()
        node.balance_score = node.balance_score_global

    def make_split(self, data: dict, y: np.ndarray|None, idx: np.ndarray, sample_weight=None) -> RuleTreeBaseStump:
        """
        Fit a split on the instances idx, choosing among ``candidate_splits_`` according to
        ``stump_selection``: a random pair (stump, data type), or the pair with the best information gain.

        Args:
            data (dict): Whole data set, as ``{data_type: array}``.
            y (np.ndarray): Target of the whole data set.
            idx (np.ndarray): Indices of the instances in the node.
            sample_weight (np.ndarray): Sample weights of the whole data set.

        Returns:
            RuleTreeBaseStump: The fitted stump.

        Raises:
            NoSplitFoundWarning: If no stump can split the node.
        """
        if self.stump_selection == 'random':
            stump, data_type = self._get_random_split()
            stump.fit(data, y, idx=idx, context=self, data_type=data_type, sample_weight=sample_weight)

            return stump

        elif self.stump_selection == 'best':
            clfs = []
            info_gains = []
            for _, stump, data_type in self.candidate_splits_:
                try:
                    stump = sklearn.clone(stump)
                    stump.fit(data, y, idx=idx, context=self, data_type=data_type, sample_weight=sample_weight)

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
        Check if a split is useless, i.e. if all the instances go to the same child.

        Args:
            data (dict): Whole data set, as ``{data_type: array}``.
            idx (np.ndarray): Indices of the instances in the node.
            labels (np.ndarray): Result of the split on idx (1 left, 2 right).

        Returns:
            bool: Whether the split is useless.
        """
        return len(np.unique(labels)) == 1

    def _update_node_statistics(self, node: RuleTreeNode, data: dict, y: np.ndarray, idx: np.ndarray):
        """
        Statistics of a node and of its stump on the instances idx. By default the ones of
        classification (majority class, class probabilities, log-odds): see RuleTreeNode.update_statistics.
        """
        node.update_statistics(data, y, idx, self)

    def predict(self, X: np.ndarray = None, X_ts=None, X_img=None, X_txt=None):
        """
        Predict the output for the input data.

        Args:
            X (np.ndarray): Tabular data.
            X_ts (np.ndarray): Time series.
            X_img (np.ndarray): Images.
            X_txt (np.ndarray): Texts.

        Returns:
            np.ndarray: Predicted outputs.
        """
        labels, _, _ = self.root.predict(self._validate_inputs(X=X, X_ts=X_ts, X_img=X_img, X_txt=X_txt))

        return labels

    def apply(self, X: np.ndarray|None = None, X_ts=None, X_img=None, X_txt=None):
        """
        Apply the RuleTree to the input data.

        Args:
            X (np.ndarray): Tabular data.
            X_ts (np.ndarray): Time series.
            X_img (np.ndarray): Images.
            X_txt (np.ndarray): Texts.

        Returns:
            np.ndarray: Leaf ids for each sample.
        """
        _, leaves, _ = self.root.predict(self._validate_inputs(X=X, X_ts=X_ts, X_img=X_img, X_txt=X_txt))

        return leaves

    def update_statistics(self, X: np.ndarray|None = None, y: np.ndarray|None = None,
                          X_ts=None, X_img=None, X_txt=None):
        """
        Recompute the statistics of the nodes (predictions, probabilities, number of samples, ...) and
        of their stumps on new data, without changing the splits.

        Args:
            X (np.ndarray): Tabular data.
            y (np.ndarray): Target.
            X_ts (np.ndarray): Time series.
            X_img (np.ndarray): Images.
            X_txt (np.ndarray): Texts.
        """
        data = self._validate_inputs(X=X, y=y, X_ts=X_ts, X_img=X_img, X_txt=X_txt)
        self._set_classes(y)

        self._update_statistics(data, y, self.root, np.arange(self._n_samples(data)))

    def _update_statistics(self, data: dict, y: np.ndarray, node: RuleTreeNode, idx: np.ndarray):
        if node is None:
            return

        node.classes = getattr(self, "classes_", None)

        self._update_node_statistics(node, data, y, idx)
        if node.is_leaf() or node.stump is None:
            return

        labels = node.stump.apply(data, idx)
        idx_l = idx[labels == 1]
        idx_r = idx[labels == 2]

        self._update_statistics(data, y, node.node_l, idx_l)
        self._update_statistics(data, y, node.node_r, idx_r)

    def compute_medoids(self, X: np.ndarray, y, idx: np.ndarray, **kwargs):
        """
        Compute medoids for the given data.

        Args:
            X (np.ndarray): Feature matrix.
            y (np.ndarray): Target labels.
            idx (np.ndarray): Indices of the samples to consider.
            **kwargs: Additional arguments for medoid computation.

        Returns:
            list: Indices of the computed medoids.
        """
        if self.distance_measure is not None:
            medoids = []
            sub_matrix = None
            for label in set(y[idx]):
                idx_local_label = np.where(y[idx] == label)[0]
                idx_label = idx[idx_local_label]
                X_class_points = X[idx_label]

                if hasattr(self, 'distance_matrix'):
                    sub_matrix = self.distance_matrix[idx_label][:, idx_label]
                else:
                    sub_matrix = pairwise_distances(X_class_points, metric=self.distance_measure)
                total_distances = sub_matrix.sum(axis=1)
                medoid_index = idx_label[total_distances.argmin()]
                medoids += [medoid_index]

            return medoids
        else:
            return None

    def get_rules(self, columns_names = None):
        """
        Get the rules of the RuleTree.

        Args:
            columns_names (list): Column names for the features.

        Returns:
            dict: Rules of the tree.
        """
        return self.root.get_rule(columns_names = columns_names)

    def make_leaf(self, node: RuleTreeNode) -> RuleTreeNode:
        """
        Convert a node into a leaf.

        Args:
            node (RuleTreeNode): Node to be converted.

        Returns:
            RuleTreeNode: Leaf node.
        """
        return node

    def queue_pop(self):
        """
        Pop an element from the queue.

        Returns:
            tuple: Index and current node.
        """
        el = heapq.heappop(self.queue)
        return el[-2:]

    def check_additional_halting_condition(self, y, curr_idx: np.ndarray):
        """
        Check additional halting conditions for splitting.

        Args:
            y: Target labels.
            curr_idx (np.ndarray): Current indices.

        Returns:
            bool: Whether to halt splitting.
        """
        return False

    def _post_fit_fix(self):
        """
        Perform post-fit adjustments to the RuleTree.
        """

    @abstractmethod
    def prepare_node(self, y: np.ndarray|None, idx: np.ndarray, node_id: str, node: Optional[RuleTreeNode] = None) -> RuleTreeNode:
        """
        Prepare a node in the RuleTree.

        Args:
            y (np.ndarray): Target labels.
            idx (np.ndarray): Indices of the samples.
            node_id (str): Node identifier.

        Returns:
            RuleTreeNode: Prepared node.
        """
        pass

    @abstractmethod
    def queue_push(self, node: RuleTreeNode, idx: np.ndarray):
        """
        Push a node into the queue.

        Args:
            node (RuleTreeNode): Node to be pushed.
            idx (np.ndarray): Indices of the samples.
        """
        pass

    @abstractmethod
    def _get_stumps_task(self) -> str:
        """
        Task of the stumps of the tree (the task of their splitter).

        Returns:
            str: MODEL_TYPE_CLF or MODEL_TYPE_REG.
        """

    def _get_tree_paths(self, current_node = None):
        """
        Get all paths in the RuleTree.

        Args:
            current_node (RuleTreeNode): Current node in the tree.

        Returns:
            list: List of paths.
        """
        if current_node.is_leaf():
            return [[current_node.node_id]]
        
        if current_node is None:
            current_node = self.root
                

        if current_node.node_l:  
            left_paths = self._get_tree_paths(current_node.node_l)
            right_paths = self._get_tree_paths(current_node.node_r)
            
            for path in left_paths:
                path.append(current_node.node_id) 
                
            for path in right_paths:
                path.append(current_node.node_id)
                
            paths = left_paths + right_paths
            
        return paths
    
    def _node_dict(self, current_node=None, d=None, i=0):
        """
        Create a dictionary mapping node IDs to indices.

        Args:
            current_node (RuleTreeNode): Current node in the tree.
            d (dict): Dictionary to store mappings.
            i (int): Current index.

        Returns:
            tuple: Dictionary and current index.
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

    @abstractmethod
    def _get_prediction_probas(self, current_node, probas=None):
        """
        Get prediction probabilities for the RuleTree.

        Args:
            current_node (RuleTreeNode): Current node in the tree.
            probas (list): List to store probabilities.

        Returns:
            list: Prediction probabilities.
        """
        pass

    def _tree_value(self):
        """
        Get the values of the RuleTree.

        Returns:
            np.ndarray: Tree values.
        """
        probas = self._get_prediction_probas(self.root)
        if isinstance(probas[0], (list, np.ndarray)):
            return np.array(probas).reshape(-1, 1, len(probas[0]))
        else:
            return np.array(probas).reshape(-1, 1, 1)
            
    def _tree_feature(self, current_node = None, feats=None):
        """
        Get the features used in the RuleTree.

        Args:
            current_node (RuleTreeNode): Current node in the tree.
            feats (list): List to store features.

        Returns:
            list: Features used in the tree.
        """
        if feats is None:
            feats = []
            
        if current_node is None:
            current_node = self.root
        
        if current_node.is_leaf():
            feats.append(-2)
        else:
            feats.append(current_node.stump.feature_original[0])
            self._tree_feature(current_node.node_l, feats = feats)
            self._tree_feature(current_node.node_r, feats = feats)
        
        return feats
        
        
    def local_interpretation(self, X, joint_contribution = False):
        """
        Perform local interpretation for the input data.

        Args:
            X (np.ndarray): Feature matrix.
            joint_contribution (bool): Whether to compute joint contributions.

        Returns:
            tuple: Direct predictions, biases, and contributions.
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
        Evaluate contributions for the RuleTree.

        Args:
            leaves (np.ndarray): Leaf indices.
            paths (list): List of paths.
            leaf_to_path (dict): Mapping of leaves to paths.
            values (np.ndarray): Tree values.
            biases (np.ndarray): Biases.
            line_shape (int): Shape of the feature matrix.
            joint_contribution (bool): Whether to compute joint contributions.

        Returns:
            tuple: Direct predictions, biases, and contributions.
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
                    #path_features.sort()
                    contributions[row][tuple(sorted(path_features))] = \
                        contributions[row].get(tuple(sorted(path_features)), 0) + contrib
            return direct_prediction, biases, contributions
        
        else:
            unique_leaves = np.unique(leaves)
            unique_contributions = {}
            
            for row, leaf in enumerate(unique_leaves):
                for path in paths:
                    if leaf == path[-1]:
                        break
                
                contribs = np.zeros(line_shape)
                for i in range(len(path) - 1):
                    
                    contrib = values_list[path[i+1]] - \
                             values_list[path[i]]
                    contribs[feature_index[path[i]]] += contrib
                unique_contributions[leaf] = contribs
                
            for row, leaf in enumerate(leaves):
                contributions.append(unique_contributions[leaf])

            return direct_prediction, biases, np.array(contributions)


    
    
    
    @classmethod
    def print_rules(cls, rules: dict=None, digits=2, indent: int = 0, ):
        """
        Print the rules of the RuleTree.

        Args:
            rules (dict): Rules of the tree.
            digits (int): Number of digits for rounding.
            indent (int): Indentation level.
        """
      
        indentation = "".join(["|   " for _ in range(indent)])
        
        if rules["is_leaf"]:
            pred = rules['prediction']
            print(f"{indentation} output: "
                  f"{pred if type(pred) in [np.str_, np.bytes_, str] else round(pred, ndigits=digits)}")
            
        else:
            print(f"{indentation}|--- {rules['textual_rule']}")
            cls.print_rules(rules=rules['left_node'], indent=indent + 1)
            print(f"{indentation}|--- {rules['not_textual_rule']}")
            cls.print_rules(rules=rules['right_node'], indent=indent + 1)

    def to_dict(self, filename=None):
        """
        Convert the RuleTree to a dictionary representation.

        Args:
            filename (str): File to save the dictionary.

        Returns:
            dict: Dictionary representation of the tree.
        """
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
        Create a RuleTree from a dictionary representation.

        Args:
            filename (str): File containing the dictionary.

        Returns:
            RuleTree: RuleTree instance.
        """
        if not isinstance(filename, dict):
            with open(filename, 'r') as f:
                dictionary = json.load(f)
        else:
            dictionary = filename

        assert 'tree_type' in dictionary

        # older dictionaries have no "tree_class": the class is named as its module
        class_name = dictionary.get('tree_class', dictionary['tree_type'].split('.')[-1])
        class_c = getattr(importlib.import_module(dictionary['tree_type']), class_name)

        # base_stumps entries: (p, stump_dict) or, from newer dictionaries, (p, stump_dict, data_types)
        args = dict(dictionary.get('args', {}))
        args['base_stumps'] = []
        for p, stump_dict, *data_types in dictionary['args']['base_stumps']:
            stump_class = getattr(importlib.import_module(stump_dict['stump_type']), stump_dict['stump_type'].split('.')[-1])
            args['base_stumps'].append((p, stump_class.dict_to_node(stump_dict), *data_types))

        tree = class_c(**args)
        # older dictionaries have classes_ also for the regression and clustering trees
        if isinstance(tree, ClassifierMixin) and dictionary.get('classes_') is not None:
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
        Export the RuleTree to a Graphviz representation.

        Args:
            columns_names (list): Column names for the features.
            scaler: Scaler for the features.
            float_precision (int): Number of digits for rounding.
            filename (str): File to save the Graphviz representation.

        Returns:
            graphviz.Digraph: Graphviz representation of the tree.
        """
        dot = self.root.export_graphviz(columns_names=columns_names, scaler=scaler, float_precision=float_precision)
        if filename is None:
            dot.render(directory=TemporaryDirectory(delete=False).name, view=True)
            return dot
        else:
            dot.render(filename=filename)

    def get_predicates(self):
        """
        Get the predicates (conditions) for the stump in the tree in depth-first order.

        Returns:
            dict: dictionary of predicates used in the tree in the form node_id: RuleTreeNode
        """
        return self.root.get_predicates()

    def get_node_by_id(self, node_id):
        """
        Get a node by its ID.

        Args:
            node_id (str): Node identifier.

        Returns:
            RuleTreeNode: RuleTreeNode instance.
        """
        return self.root.get_node_by_id(node_id)

    def get_leaf_nodes(self):
        """
        Get all leaf nodes in the RuleTree.

        Returns:
            list: List of leaf nodes.
        """
        return self.root.get_leaf_nodes()
