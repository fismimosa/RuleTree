import datetime
from typing import Union

import numpy as np
from scipy.spatial.distance import cdist
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.tree import DecisionTreeClassifier

from RuleTree.RuleTree import RuleTree
from RuleTree.RuleTreeNode import RuleTreeNode
from RuleTree.utils import MODEL_TYPE_CLF
from RuleTree.utils.data_utils import prepare_data, preprocessing


class RuleTreeClassifier(RuleTree):
    def __init__(self,
                 impurity_measure: str = "gini",

                 max_depth: int = 4, max_nbr_nodes: int = 32, min_samples_leaf: int = 3, min_samples_split: int = 5,
                 max_nbr_values: Union[int, float] = np.inf, max_nbr_values_cat: Union[int, float] = 20,
                 allow_oblique_splits: bool = False, force_oblique_splits: bool = False, max_oblique_features: int = 2,
                 prune_useless_leaves: bool = True,

                 n_components: int = 2, bic_eps: float = 0.0,

                 feature_names: list = None, one_hot_encode_cat: bool = True, categorical_indices: list = None,
                 numerical_indices: list = None, numerical_scaler: Union[StandardScaler, MinMaxScaler] = None,

                 precision: int = 2, cat_precision: int = 2, exclude_split_feature_from_reduction: bool = False,

                 random_state: int = None, n_jobs: int = 1, verbose=False,

                 **kwargs
                 ):
        super().__init__(max_depth=max_depth, max_nbr_nodes=max_nbr_nodes, min_samples_leaf=min_samples_leaf,
                         min_samples_split=min_samples_split, max_nbr_values=max_nbr_values,
                         max_nbr_values_cat=max_nbr_values_cat, allow_oblique_splits=allow_oblique_splits,
                         force_oblique_splits=force_oblique_splits, max_oblique_features=max_oblique_features,
                         prune_useless_leaves=prune_useless_leaves, n_components=n_components, bic_eps=bic_eps,
                         feature_names=feature_names, one_hot_encode_cat=one_hot_encode_cat,
                         categorical_indices=categorical_indices, numerical_indices=numerical_indices,
                         numerical_scaler=numerical_scaler, precision=precision, cat_precision=cat_precision,
                         exclude_split_feature_from_reduction=exclude_split_feature_from_reduction,
                         random_state=random_state, n_jobs=n_jobs, verbose=verbose, **kwargs)

        self.impurity = impurity_measure

    def _make_leaf(self, node: RuleTreeNode):
        super()._make_leaf(node=node)
        self.labels_[node.idx] = self.labels_[node.idx].astype(int)

    def _make_split(self, idx_iter):
        clf = DecisionTreeClassifier(
            max_depth=1,
            criterion=self.impurity,
            min_samples_leaf=self.min_samples_leaf,
            min_samples_split=self.min_samples_split,
            random_state=self.random_state,
            **self.kwargs
        )

        return super()._make_supervised_split(idx_iter=idx_iter, clf=clf)

    def fit(self, X, y=None):
        assert X is not None
        assert y is not None

        super().fit(X=X, y=y)

        if self.verbose:
            print(datetime.datetime.now(), 'Model type: %s.', MODEL_TYPE_CLF)

        n_idx = X.shape[0]
        idx = np.arange(n_idx)

        self.labels_ = -1 * np.ones(n_idx).astype(int)

        node_id = 0
        majority_class = RuleTree.calculate_mode(self._y)
        proba = RuleTree.calculate_proba(self._y, self.nbr_classes_)

        root_node = RuleTreeNode(idx, node_id, majority_class, proba=proba, parent_id=-1)
        self._node_dict[root_node.node_id] = root_node

        self._queue.append((idx, 0, root_node))

        if self.feature_names_r is None:
            self.feature_names_r = np.array(['X_%s' % i for i in range(X.shape[1])])
        else:
            self.feature_names_r = np.array(self.feature_names_r)

        self._Xr = X
        res = prepare_data(self._X, self.max_nbr_values, self.max_nbr_values_cat, self.feature_names_r,
                           self.one_hot_encode_cat, self.categorical_indices, self.numerical_indices,
                           self.numerical_scaler)

        self._X = res[0]
        (self.feature_values_r, self.is_cat_feat_r, self.feature_values, self.is_cat_feat, self.data_encoder,
         self.feature_names) = res[1]

        self.maps = res[2]

        if self.feature_names is None:
            self.feature_names = self.feature_names_r

        self.con_indexes = np.where(~self.is_cat_feat)[0]
        self.cat_indexes = np.where(self.is_cat_feat)[0]

        self.con_indexes_r = np.where(~self.is_cat_feat_r)[0]
        self.cat_indexes_r = np.where(self.is_cat_feat_r)[0]

        if self.verbose:
            print(datetime.datetime.now(), 'Continuous features real %s.' % self.con_indexes_r)
            print(datetime.datetime.now(), 'Continuous features onehot %s.' % self.con_indexes)
            print(datetime.datetime.now(), 'Categorical features real %s.' % self.cat_indexes_r)
            print(datetime.datetime.now(), 'Categorical features onehot %s.' % self.cat_indexes)

        if self.verbose:
            print(datetime.datetime.now(), 'Training started')

        nbr_curr_leaves = 0
        iter_id = 0
        while len(self._queue) > 0 and nbr_curr_leaves + len(self._queue) <= self.max_nbr_nodes:
            if self.verbose:
                print(datetime.datetime.now(), 'Iteration: %s, current nbr leaves: %s.' % (iter_id, nbr_curr_leaves))
            iter_id += 1

            (idx_iter, node_depth, node) = self._queue.pop(0)

            if len(np.unique(self._y[idx_iter])) == 1:
                self._make_leaf(node)
                nbr_curr_leaves += 1
                if self.verbose:
                    print(datetime.datetime.now(), 'Classification, node with unique target.')
                continue

            nbr_samples = len(idx_iter)
            if super()._halting_conditions(node, nbr_samples, nbr_curr_leaves, node_depth):
                nbr_curr_leaves += 1
                continue

            clf, is_oblique = self._make_split(idx_iter)
            idx_leaves = clf.apply(self._X[idx_iter]) # check

            if len(np.unique(idx_leaves)) == 1:
                self._make_leaf(node)
                nbr_curr_leaves += 1
                if self.verbose:
                    print(datetime.datetime.now(), 'Split useless in classification.')
                continue

            idx_l, idx_r = np.where(idx_leaves == 1)[0], np.where(idx_leaves == 2)[0]

            idx_all_l = idx_iter[idx_l]
            idx_all_r = idx_iter[idx_r]

            label_l = RuleTree.calculate_mode(self._y[idx_all_l])
            label_r = RuleTree.calculate_mode(self._y[idx_all_r])
            proba_l = RuleTree.calculate_proba(self._y[idx_all_l], self.nbr_classes_)
            proba_r = RuleTree.calculate_proba(self._y[idx_all_r], self.nbr_classes_)

            node_id += 1
            if not is_oblique:
                impurity_l = clf.tree_.impurity[1]
            else:
                impurity_l = clf.oblq_clf.tree_.impurity[1]
            node_l = RuleTreeNode(idx=idx_all_l, node_id=node_id, label=label_l,
                                  parent_id=node.node_id, impurity=impurity_l, proba=proba_l)

            node_id += 1
            if not is_oblique:
                impurity_r = clf.tree_.impurity[2]
            else:
                impurity_r = clf.oblq_clf.tree_.impurity[2]
            node_r = RuleTreeNode(idx=idx_all_r, node_id=node_id, label=label_r,
                                  parent_id=node.node_id, impurity=impurity_r, proba=proba_r)

            node.clf = clf
            node.node_l = node_l
            node.node_r = node_r
            if not is_oblique:
                node.impurity = clf.tree_.impurity[0]
            else:
                node.impurity = clf.oblq_clf.tree_.impurity[0]
            node.is_oblique = is_oblique

            self._tree_structure[node.node_id] = (node_l.node_id, node_r.node_id)
            self._node_dict[node_l.node_id] = node_l
            self._node_dict[node_r.node_id] = node_r

            self._queue.append((idx_all_l, node_depth + 1, node_l))
            self._queue.append((idx_all_r, node_depth + 1, node_r))

        if self.verbose:
            print(datetime.datetime.now(),
                  'Training ended in %s iterations, with %s leaves.' % (iter_id, nbr_curr_leaves))

        if self.prune_useless_leaves:
            if self.verbose:
                print(datetime.datetime.now(), 'Pruning of useless leaves.')
            self._prune_useless_leaves()

        self.root_ = root_node

        if self.class_encoder_ is not None:
            self._y = self.class_encoder_.inverse_transform(self._y.reshape(-1, 1))[:, 0]
        self.rules_to_tree_print_ = self._get_rules_to_print_tree()
        self.rules_ = self._calculate_rules()
        self.rules_ = self._compact_rules()
        self.rules_s_ = self._rules2str()
        self.medoid_dict_ = self._calculate_all_medoids()

        self.task_medoid_dict_ = self._calculate_task_medoids()

        if self.verbose:
            print(datetime.datetime.now(), 'RULE TREE - END.\n')

        return self

    def predict(self, X, get_leaf=False, get_rule=False):
        return super().predict(X, get_leaf=get_leaf, get_rule=get_rule)

    def _predict_adjust_labels(self, labels):
        if self.class_encoder_ is not None:
            labels = self.class_encoder_.inverse_transform(labels.reshape(-1, 1))[:, 0]

        return labels

    def _predict(self, X, idx, node):
        labels, leaves, proba = super()._predict(X=X, idx=idx, node=node)

        return labels, leaves, proba

    def transform(self, X, metric='euclidean', include_descriptive_pivot=True, include_target_pivot=False,
                  only_leaf_pivot=False):
        P = list()
        if include_descriptive_pivot:
            if only_leaf_pivot:
                for k in self.rules_:
                    P.append(self.medoid_dict_[k])
            else:
                for k in self.medoid_dict_:
                    P.append(self.medoid_dict_[k])

        if include_target_pivot:
            if only_leaf_pivot:
                for k in self.rules_:
                    for l in self.task_medoid_dict_[k]:
                        P.append(self.task_medoid_dict_[k][l])

            else:
                for k in self.task_medoid_dict_:
                    for l in self.task_medoid_dict_[k]:
                        P.append(self.task_medoid_dict_[k][l])

        if len(P) == 0:
            return None

        P = np.array(P)

        X = preprocessing(X, self.feature_names_r, self.is_cat_feat, self.data_encoder, self.numerical_scaler)
        P = preprocessing(P, self.feature_names_r, self.is_cat_feat, self.data_encoder, self.numerical_scaler)

        dist = cdist(X.astype(float), P.astype(float), metric=metric)

        return dist

    def _get_labels_from_node(self, node):
        return self.class_encoder_.inverse_transform(np.array([node.label]).reshape(-1, 1))[0, 0]

    def _rules2str_text(self, cond):
        cons_txt = cond[0]
        return '%s' % cons_txt

    def _calculate_task_medoids(self):
        if self.verbose:
            print(datetime.datetime.now(), 'Calculate class medoids.')
        return RuleTreeClassifier.calculate_task_medoids(node_dict=self._node_dict, X=self._X, y=self._y,
                                                         Xr=self._Xr)

    @staticmethod
    def calculate_task_medoids(node_dict, X, y, Xr):
        class_medoid_dict = dict()
        for node_id in node_dict:
            node = node_dict[node_id]
            node.task_medoid = dict()
            for l in np.unique(y[node.idx]):
                node.task_medoid[l] = RuleTree.calculate_medoid(X[node.idx][y[node.idx] == l],
                                                                Xr[node.idx][y[node.idx] == l])
            class_medoid_dict[node_id] = node.task_medoid
        return class_medoid_dict

    def _print_tree_text(self):
        return "class"
