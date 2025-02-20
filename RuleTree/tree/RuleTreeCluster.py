import heapq

import numpy as np
import pandas as pd
from sklearn import tree
from sklearn.base import ClusterMixin, ClassifierMixin, RegressorMixin
from sklearn.metrics import r2_score

from RuleTree import RuleTreeRegressor, RuleTreeClassifier
from RuleTree.tree.RuleTree import RuleTree
from RuleTree.tree.RuleTreeNode import RuleTreeNode
from RuleTree.utils import bic, light_famd
from RuleTree.stumps.regression.DecisionTreeStumpRegressor import DecisionTreeStumpRegressor


class RuleTreeCluster(RuleTree, ClusterMixin):
    def __init__(self,
                 n_components: int = 2,
                 clus_impurity: str = 'r2',
                 bic_eps: float = .0,
                 max_leaf_nodes=float('inf'),
                 min_samples_split=2,
                 max_depth=float('inf'),
                 prune_useless_leaves=False,
                 base_stumps: RegressorMixin | list = None,
                 random_state=None,

                 criterion='squared_error',
                 splitter='best',
                 min_samples_leaf=1,
                 min_weight_fraction_leaf=0.0,
                 max_features=None,
                 min_impurity_decrease=0.0,
                 ccp_alpha=0.0,
                 monotonic_cst=None
                 ):
        if base_stumps is None:
            base_stumps = DecisionTreeStumpRegressor(
                max_depth=1,
                criterion=criterion,
                splitter=splitter,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                min_weight_fraction_leaf=min_weight_fraction_leaf,
                max_features=max_features,
                random_state=random_state,
                min_impurity_decrease=min_impurity_decrease,
                ccp_alpha=ccp_alpha,
                monotonic_cst=monotonic_cst
            )

        super().__init__(max_leaf_nodes=max_leaf_nodes,
                         min_samples_split=min_samples_split,
                         max_depth=max_depth,
                         prune_useless_leaves=prune_useless_leaves,
                         base_stumps=base_stumps,
                         stump_selection='random',
                         random_state=random_state)

        self.n_components = n_components
        self.clus_impurity = clus_impurity
        self.bic_eps = bic_eps
        self.criterion = criterion
        self.splitter = splitter
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_features = max_features
        self.min_impurity_decrease = min_impurity_decrease
        self.ccp_alpha = ccp_alpha
        self.monotonic_cst = monotonic_cst

        if self.clus_impurity not in ['bic', 'r2']:
            raise Exception('Unknown clustering impurity measure %s' % self.clus_impurity)

    def is_split_useless(self, X, clf: tree, idx: np.ndarray):
        labels = clf.apply(X[idx])

        if len(np.unique(labels)) == 1:
            return True

        # CHECK BIC DECREASE
        bic_parent = bic(X[idx], [0] * len(idx))
        bic_children = bic(X[idx], (np.array(labels) - 1).tolist())

        return bic_parent < bic_children - self.bic_eps * np.abs(bic_parent)

    def queue_push(self, node: RuleTreeNode, idx: np.ndarray):
        heapq.heappush(self.queue, (-len(idx), next(self.tiebreaker), idx, node))

    def make_split(self, X: np.ndarray, y, idx: np.ndarray, **kwargs) -> tree:
        if len(X.shape) != 2:
            raise TypeError(f'Unsupported data type for shape {X.shape}')

        n_components_split = min(self.n_components, len(idx))

        dtypes = pd.DataFrame(X).infer_objects().dtypes
        numerical = dtypes[dtypes != np.dtype('O')].index
        categorical = dtypes[dtypes == np.dtype('O')].index

        if len(categorical) == 0:  # all continuous
            principal_transform = light_famd.PCA(n_components=n_components_split, random_state=self.random_state)
        elif len(numerical) == 0:  # all categorical
            principal_transform = light_famd.MCA(n_components=n_components_split, random_state=self.random_state)
        else:  # mixed
            principal_transform = light_famd.FAMD(n_components=n_components_split, random_state=self.random_state)

        y_pca = principal_transform.fit_transform(X[idx])
        y_pca_all = np.zeros((X.shape[0], y_pca.shape[1]))
        y_pca_all[idx] = y_pca

        best_clf = None
        best_score = float('inf')
        for i in range(n_components_split):
            clf = self._get_random_stump(y_pca)

            clf.fit(
                X=X,
                y=y_pca_all[:, i],
                idx=idx,
                context=self
            )
            if self.clus_impurity == 'r2':
                score = -1 * r2_score(clf.apply(X[idx]), y_pca[:, i])
            else:
                labels_i = clf.apply(X[idx]).astype(int)
                score = bic(X[idx], (np.array(labels_i) - 1).tolist())

            if score < best_score:
                best_score = score
                best_clf = clf

        return best_clf
        
    def compute_medoids(self, X: np.ndarray, y, idx: np.ndarray, **kwargs):
        pass
        
    def prepare_node(self, y: np.ndarray, idx: np.ndarray, node_id: str) -> RuleTreeNode:
        return RuleTreeNode(
            node_id=node_id,
            prediction=node_id,
            prediction_probability=-1,
            classes=np.array(['NA']),
            parent=None,
            stump=None,
            node_l=None,
            node_r=None,
            samples=len(idx),
        )

    def _post_fit_fix(self):
        possible_labels, inner_nodes = self.root.get_possible_outputs()
        all_outputs = list(possible_labels) + list(inner_nodes)
        if type(next(iter(all_outputs))) is str and not hasattr(self, 'label_encoder'):
            self.label_encoder = {k: all_outputs.index(k) for k in set(all_outputs)}
            self.__labels_obj_to_int(self.root)

    def __labels_obj_to_int(self, node: RuleTreeNode):
        node.prediction = self.label_encoder[node.prediction]

        if node.is_leaf():
            return

        self.__labels_obj_to_int(node.node_l)
        self.__labels_obj_to_int(node.node_r)

    def _get_stumps_base_class(self):
        return RegressorMixin

    def _get_prediction_probas(self, current_node = None, probas=None):
        raise NotImplementedError()


class RuleTreeClusterClassifier(RuleTreeCluster, ClassifierMixin):
    def prepare_node(self, y: np.ndarray, idx: np.ndarray, node_id: str) -> RuleTreeNode:
        return RuleTreeClassifier.prepare_node(self, y, idx, node_id)

    def _post_fit_fix(self):
        return

    def _predict(self, X: np.ndarray, current_node: RuleTreeNode):
        return RuleTreeClassifier._predict(self, X, current_node)


class RuleTreeClusterRegressor(RuleTreeCluster, RegressorMixin):
    def prepare_node(self, y: np.ndarray, idx: np.ndarray, node_id: str) -> RuleTreeNode:
        return RuleTreeRegressor.prepare_node(self, y, idx, node_id)

    def _post_fit_fix(self):
        return

    def _predict(self, X: np.ndarray, current_node: RuleTreeNode):
        return RuleTreeRegressor._predict(self, X, current_node)
