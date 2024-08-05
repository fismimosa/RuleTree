import heapq
from typing import Tuple

import numpy as np
import pandas as pd
from numpy import bool_
from sklearn import tree
from sklearn.base import ClusterMixin
from sklearn.metrics import r2_score
from sklearn.preprocessing import LabelEncoder

from ruletree import light_famd
from ruletree.RuleTree import RuleTree
from ruletree.RuleTreeNode import RuleTreeNode
from ruletree.utils import bic
from ruletree.utils.MyDecisionTreeClassifier import MyDecisionTreeClassifier
from ruletree.utils.MyDecisionTreeRegressor import MyDecisionTreeRegressor
from ruletree.utils.data_utils import calculate_mode, get_info_gain


class RuleTreeCluster(RuleTree, ClusterMixin):
    def __init__(self,
                 n_components:int=2,
                 clus_impurity:str='bic',
                 bic_eps:float=.0,
                 max_nbr_nodes=float('inf'),
                 min_samples_split=2,
                 max_depth=float('inf'),
                 prune_useless_leaves=False,
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
        super().__init__(max_nbr_nodes=max_nbr_nodes,
                         min_samples_split=min_samples_split,
                         max_depth=max_depth,
                         prune_useless_leaves=prune_useless_leaves,
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

    def is_split_useless(self, clf: tree, idx: np.ndarray):
        labels = clf.apply(self.X[idx])

        if len(np.unique(labels)) == 1:
            return True

        # CHECK BIC DECREASE
        bic_parent = bic(self.X[idx], [0] * len(idx))
        bic_children = bic(self.X[idx], (np.array(labels) - 1).tolist())

        return bic_parent < bic_children - self.bic_eps * np.abs(bic_parent)

    def queue_push(self, node: RuleTreeNode, idx: np.ndarray):
        heapq.heappush(self.queue, (-len(idx), next(self.tiebreaker), idx, node))

    def make_split(self, X: np.ndarray, y, idx: np.ndarray) -> tree:
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


        best_clf = None
        best_score = float('inf')
        for i in range(n_components_split):
            clf = MyDecisionTreeRegressor(
                max_depth=1,
                criterion=self.criterion,
                splitter=self.splitter,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                min_weight_fraction_leaf=self.min_weight_fraction_leaf,
                max_features=self.max_features,
                random_state=self.random_state,
                min_impurity_decrease=self.min_impurity_decrease,
                ccp_alpha=self.ccp_alpha,
                monotonic_cst=self.monotonic_cst
            )

            clf.fit(X[idx], y_pca[:, i])
            if self.clus_impurity == 'r2':
                score = -1 * r2_score(clf.predict(X[idx]), y_pca[:, i])
            else:
                labels_i = clf.apply(X[idx]).astype(int)
                score = bic(X[idx], (np.array(labels_i) - 1).tolist())

            if score < best_score:
                best_score = score
                best_clf = clf

        return best_clf


    def prepare_node(self, y: np.ndarray, idx: np.ndarray, node_id: str) -> RuleTreeNode:
        return RuleTreeNode(
            node_id=node_id,
            prediction=node_id,
            prediction_probability=-1,
            parent=None,
            clf=None,
            node_l=None,
            node_r=None,
            samples=len(idx),
        )

    def _post_fit_fix(self):
        possible_labels = np.array(list(self.root.get_possible_outputs()))
        if np.issubdtype(possible_labels.dtype, np.str_) and not hasattr(self, 'label_encoder'):
            self.label_encoder = LabelEncoder().fit(possible_labels)
            self.__labels_obj_to_int(self.root, dict(zip(self.label_encoder.classes_,
                                                         self.label_encoder.transform(self.label_encoder.classes_))))

    def __labels_obj_to_int(self, node:RuleTreeNode, map:dict):
        node.prediction = map[node.prediction]

        if node.is_leaf():
            return

        self.__labels_obj_to_int(node.node_l, map)
        self.__labels_obj_to_int(node.node_r, map)





