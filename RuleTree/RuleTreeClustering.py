import datetime
import heapq
from typing import Union

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.tree import DecisionTreeRegressor

from RuleTree import light_famd
from RuleTree.RuleTree import RuleTree
from RuleTree.RuleTreeClassifier import RuleTreeClassifier
from RuleTree.RuleTreeNode import RuleTreeNode
from RuleTree.RuleTreeRegressor import RuleTreeRegressor
from RuleTree.utils import ObliqueHouseHolderSplit, MODEL_TYPE_REG, MODEL_TYPE_CLU
from RuleTree.utils.bic_estimator import bic


class RuleTreeClustering(RuleTree):
    def __init__(self,
                 impurity_measure: str = "r2",
                 clu_for_clf: bool = False,
                 clu_for_reg: bool = False,

                 max_depth: int = 4,
                 max_nbr_nodes: int = 32,
                 min_samples_leaf: int = 3,
                 min_samples_split: int = 5,
                 max_nbr_values: Union[int, float] = np.inf,
                 max_nbr_values_cat: Union[int, float] = 20,
                 allow_oblique_splits: bool = False,
                 force_oblique_splits: bool = False,
                 max_oblique_features: int = 2,
                 prune_useless_leaves: bool = True,

                 n_components: int = 2,
                 bic_eps: float = 0.0,

                 feature_names: list = None,
                 one_hot_encode_cat: bool = True,
                 categorical_indices: list = None,
                 numerical_indices: list = None,
                 numerical_scaler: Union[StandardScaler, MinMaxScaler] = None,

                 precision: int = 2,
                 cat_precision: int = 2,
                 exclude_split_feature_from_reduction: bool = False,

                 random_state: int = None,
                 n_jobs: int = 1,
                 verbose=False,

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
                         random_state=random_state, n_jobs=n_jobs, verbose=verbose, kwargs=kwargs)

        self._typed_X = None
        self.impurity = impurity_measure
        self.clu_for_clf = clu_for_clf
        self.clu_for_reg = clu_for_reg

    def _make_leaf(self, node: RuleTreeNode):
        super()._make_leaf(node=node)
        if self.clu_for_clf:
            self.labels_[node.idx] = self.labels_[node.idx].astype(int)

    def _make_split(self, idx_iter):
        if self.exclude_split_feature_from_reduction:
            if self.verbose:
                print(datetime.datetime.now(), 'Exclude split feature from reduction.')
            return self._make_unsupervised_split_iter(idx_iter)

        nbr_samples = len(idx_iter)
        n_components_split = min(self.n_components, nbr_samples)

        if n_components_split >= self._typed_X.loc[idx_iter].shape[1]:
            raise ValueError(
                f"n_components_split ({n_components_split}) should be less than X.shape[1] ({self._typed_X.loc[idx_iter].shape[1]})")

        if len(self.cat_indexes_r) == 0:  # all continous
            principal_transform = light_famd.PCA(n_components=n_components_split, random_state=self.random_state)
        elif len(self.con_indexes_r) == 0:  # all caregorical
            principal_transform = light_famd.MCA(n_components=n_components_split, random_state=self.random_state)
        else:  # mixed
            principal_transform = light_famd.FAMD(n_components=n_components_split, random_state=self.random_state)

        y_pca = principal_transform.fit_transform(self._typed_X.loc[idx_iter])

        clf_list = list()
        eval_list = list()
        for i in range(n_components_split):
            clf = None

            if not self.force_oblique_splits:
                clf = DecisionTreeRegressor(
                    max_depth=1,
                    min_samples_leaf=self.min_samples_leaf,
                    min_samples_split=self.min_samples_split,
                    random_state=self.random_state,
                )

            if self.allow_oblique_splits and i > 0:
                clf = ObliqueHouseHolderSplit(
                    pca=principal_transform,
                    max_oblique_features=self.max_oblique_features,
                    min_samples_leaf=self.min_samples_leaf,
                    min_samples_split=self.min_samples_split,
                    model_type=MODEL_TYPE_REG,
                    random_state=self.random_state,
                )

            if clf is not None:

                clf.fit(self._X[idx_iter], y_pca[:, i])
                if self.impurity == 'r2':
                    eval_val = -1 * r2_score(clf.predict(self._X[idx_iter]), y_pca[:, i])
                elif self.impurity == 'bic':
                    labels_i = clf.apply(self._X[idx_iter])
                    eval_val = bic(self._X[idx_iter], (np.array(labels_i) - 1).tolist())
                else:
                    raise Exception('Unknown clustering impurity measure %s' % self.impurity)

                clf_list.append(clf)
                eval_list.append(eval_val)

        idx_min = np.argmin(eval_list)
        is_oblique = self.force_oblique_splits or self.allow_oblique_splits and idx_min > 0 and idx_min % 2 == 0
        clf = clf_list[idx_min]

        return clf, is_oblique

    def _make_unsupervised_split_iter(self, idx_iter):
        nbr_samples = len(idx_iter)
        n_components_split = min(self.n_components, nbr_samples)

        if n_components_split >= self._typed_X.loc[idx_iter].shape[1]:
            raise ValueError(f"n_components_split ({n_components_split}) should be less than X.shape[1] "
                             f"({self._typed_X.loc[idx_iter].shape[1]})")

        features_idx = np.arange(self._Xr.shape[1])
        clf_list_all = list()
        eval_list_all = list()

        for feature in features_idx:

            con_indexes_r_feature = [f for f in self.con_indexes_r if f != feature]
            cat_indexes_r_feature = [f for f in self.cat_indexes_r if f != feature]

            if len(cat_indexes_r_feature) == 0:  # all continous
                principal_transform = light_famd.PCA(n_components=n_components_split, random_state=self.random_state)
            elif len(con_indexes_r_feature) == 0:  # all caregorical
                principal_transform = light_famd.MCA(n_components=n_components_split, random_state=self.random_state)
            else:  # mixed
                principal_transform = light_famd.FAMD(n_components=n_components_split, random_state=self.random_state)

            y_pca = principal_transform.fit_transform(self._typed_X.loc[idx_iter, features_idx != feature])

            clf_list = list()
            eval_list = list()
            for i in range(n_components_split):

                clf = DecisionTreeRegressor(
                    max_depth=1,
                    min_samples_leaf=self.min_samples_leaf,
                    min_samples_split=self.min_samples_split,
                    random_state=self.random_state,
                )

                # metto tutto a zero eccetto quello in esame per non avere problemi di dimensionalita
                if self.maps is not None:
                    features_to_use = self.maps['map_original_onehot_idx'][feature]
                else:
                    features_to_use = feature
                data_for_training = np.zeros(self._X[idx_iter].shape)
                data_for_training[:, features_to_use] = self._X[idx_iter][:, features_to_use]

                clf.fit(data_for_training, y_pca[:, i])
                if self.impurity == 'r2':
                    eval_val = -1 * r2_score(clf.predict(self._X[idx_iter]), y_pca[:, i])
                elif self.impurity == 'bic':
                    labels_i = clf.apply(self._X[idx_iter])
                    eval_val = bic(self._X[idx_iter], (np.array(labels_i) - 1).tolist())
                else:
                    raise Exception('Unknown clustering impurity measure %s' % self.impurity)

                clf_list.append(clf)
                eval_list.append(eval_val)

            idx_min = np.argmin(eval_list)
            eval_list_all.append(eval_list[idx_min])
            clf_list_all.append(clf_list[idx_min])

        idx_min = np.argmin(eval_list_all)
        is_oblique = False
        clf = clf_list_all[idx_min]

        return clf, is_oblique

    def fit(self, X, y=None):
        super().fit(X=X, y=y)

        if self.verbose:
            print(datetime.datetime.now(), 'Model type: %s.', MODEL_TYPE_CLU)
            print(datetime.datetime.now(), 'Clustering for classification: %s.' % self.clu_for_clf)
            print(datetime.datetime.now(), 'Clustering for regression: %s.' % self.clu_for_reg)

        n_idx = X.shape[0]
        idx = np.arange(n_idx)

        if self.clu_for_reg:
            self.labels_ = (-1.0 * np.ones(n_idx)).astype(float)

        node_id = 0
        proba = None
        if self._y is None:
            majority_class = node_id
        else:
            if len(np.unique(self._y)) >= self.max_nbr_values_cat:  # infer is numerical
                majority_class = np.mean(self._y)
            else:  # infer it is categorical
                majority_class = RuleTree.calculate_mode(self._y)
                proba = RuleTree.calculate_proba(y, self.nbr_classes_)

        root_node = RuleTreeNode(idx, node_id, majority_class, proba=proba, parent_id=-1)
        self._node_dict[root_node.node_id] = root_node

        heapq.heappush(self._queue, (-len(idx), (next(self.tiebreaker), idx, 0, root_node)))

        if self.feature_names_r is None:
            self.feature_names_r = np.array(['X_%s' % i for i in range(X.shape[1])])
        else:
            self.feature_names_r = np.array(self.feature_names_r)

        self._Xr = X
        res = RuleTree.prepare_data(self._X, self.max_nbr_values, self.max_nbr_values_cat,
                                    self.feature_names_r, self.one_hot_encode_cat, self.categorical_indices,
                                    self.numerical_indices, self.numerical_scaler)
        self._X = res[0]
        self.feature_values_r, self.is_cat_feat_r, self.feature_values, self.is_cat_feat, self.data_encoder, self.feature_names = \
            res[1]
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

        self._typed_X = pd.DataFrame(data=self._Xr)
        for index in self.con_indexes_r:
            self._typed_X[index] = self._typed_X[index].astype(float)
        for index in self.cat_indexes_r:
            self._typed_X[index] = self._typed_X[index].astype(str)
        self._typed_X.columns = self._typed_X.columns.astype(str)

        if len(self.cat_indexes_r) == 0:
            if self.verbose:
                print(datetime.datetime.now(), 'All continuous features, use PCA.')
        elif len(self.con_indexes_r) == 0:
            if self.verbose:
                print(datetime.datetime.now(), 'All categorical features, use MCA.')
        else:
            if self.verbose:
                print(datetime.datetime.now(), 'All categorical features, use FAMD.')

        if self.verbose:
            print(datetime.datetime.now(), 'Training started')

        nbr_curr_nodes = 0
        iter_id = 0
        while len(self._queue) > 0 and nbr_curr_nodes + len(self._queue) <= self.max_nbr_nodes:
            if self.verbose:
                print(datetime.datetime.now(), 'Iteration: %s, current nbr leaves: %s.' % (iter_id, nbr_curr_nodes))
            iter_id += 1

            _, (_, idx_iter, node_depth, node) = heapq.heappop(self._queue)

            nbr_samples = len(idx_iter)
            if super()._halting_conditions(node, nbr_samples, nbr_curr_nodes, node_depth):
                nbr_curr_nodes += 1
                continue

            clf, is_oblique = self._make_split(idx_iter)
            labels = clf.apply(self._X[idx_iter])
            y_pred = clf.predict(self._X[idx_iter])

            if len(np.unique(labels)) == 1 or len(np.unique(y_pred)) == 1:
                self._make_leaf(node)
                nbr_curr_nodes += 1
                if self.verbose:
                    print(datetime.datetime.now(), 'Split useless in clustering.')
                continue

            bic_parent = bic(self._X[idx_iter], [0] * nbr_samples)
            bic_children = bic(self._X[idx_iter], (np.array(labels) - 1).tolist())

            if bic_parent < bic_children - self.bic_eps * np.abs(bic_parent):
                self._make_leaf(node)
                nbr_curr_nodes += 1
                if self.verbose:
                    print(datetime.datetime.now(), 'No advantage in split w.r.t. BIC.')
                continue

            idx_l, idx_r = np.where(labels == 1)[0], np.where(labels == 2)[0]

            idx_all_l = idx_iter[idx_l]
            idx_all_r = idx_iter[idx_r]
            proba_l = None
            proba_r = None

            if self._y is None:
                label_l = node_id + 1
                label_r = node_id + 2

            else:
                if self.y_is_numerical:
                    label_l = np.mean(self._y[idx_all_l])
                    label_r = np.mean(self._y[idx_all_r])
                else:
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

            bic_l = bic(self._X[idx_iter[idx_l]], [0] * len(idx_l))
            bic_r = bic(self._X[idx_iter[idx_r]], [0] * len(idx_r))
            heapq.heappush(self._queue, (-len(idx_all_l) + 0.00001 * bic_l,
                                          (next(self.tiebreaker), idx_all_l, node_depth + 1, node_l)))
            heapq.heappush(self._queue, (-len(idx_all_r) + 0.00001 * bic_r,
                                          (next(self.tiebreaker), idx_all_r, node_depth + 1, node_r)))

            if self.verbose:
                print(datetime.datetime.now(),
                      'Training ended in %s iterations, with %s leaves.' % (iter_id, nbr_curr_nodes))

            self.root_ = root_node
            self.label_encoder_ = LabelEncoder()

            if self.clu_for_clf:
                if self.verbose:
                    print(datetime.datetime.now(), 'Normalize labels id.')
                self.labels_ = self.label_encoder_.fit_transform(self.labels_)

            if self.class_encoder_ is not None:
                self._y = self.class_encoder_.inverse_transform(self._y)
            self.rules_to_tree_print_ = self._get_rules_to_print_tree()
            self.rules_ = self._calculate_rules()
            self.rules_ = self._compact_rules()
            self.rules_s_ = self._rules2str()
            self.medoid_dict_ = self._calculate_all_medoids()

            self.task_medoid_dict_ = self._calculate_task_medoids()

            if self.verbose:
                print(datetime.datetime.now(), 'RULE TREE - END.\n')

    def predict(self, X, get_leaf=False, get_rule=False):
        super().predict(X, get_leaf=False, get_rule=False)

    def _predict_adjust_labels(self, labels):
        if self.clu_for_clf:
            labels = self.label_encoder_.transform(labels)
            if self.class_encoder_ is not None:
                labels = self.class_encoder_.inverse_transform(labels)

        #if self.clu_for_clf:
        #    if self.class_encoder_ is not None:
        #        labels = self.class_encoder_.inverse_transform(labels)
        if not self.clu_for_clf:
            labels = self.label_encoder_.transform(labels)

        return labels

    def _predict(self, X, idx, node):
        labels, leaves, proba = super()._predict(X=X, idx=idx, node=node)

        if self.clu_for_reg:
            return labels.astype(float), leaves, proba
        else:
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
                if self.clu_for_clf:
                    for k in self.rules_:
                        for l in self.task_medoid_dict_[k]:
                            P.append(self.task_medoid_dict_[k][l])

                if self.clu_for_reg:
                    for k in self.rules_:
                        P.append(self.task_medoid_dict_[k])
            else:
                if self.clu_for_clf:
                    for k in self.task_medoid_dict_:
                        for l in self.task_medoid_dict_[k]:
                            P.append(self.task_medoid_dict_[k][l])

                if self.clu_for_reg:
                    for k in self.task_medoid_dict_:
                        P.append(self.task_medoid_dict_[k])

        if len(P) == 0:
            return None

        P = np.array(P)

        X = RuleTree.preprocessing(X, self.feature_names_r, self.is_cat_feat, self.data_encoder, self.numerical_scaler)
        P = RuleTree.preprocessing(P, self.feature_names_r, self.is_cat_feat, self.data_encoder, self.numerical_scaler)

        dist = cdist(X.astype(float), P.astype(float), metric=metric)

        return dist

    def _get_labels(self, node):
        if self.clu_for_clf:
            label = self.label_encoder_.transform([node.label])[0]
            if self.class_encoder_ is not None:
                label = self.class_encoder_.inverse_transform([label])[0]
        else:
            label = node.label

        return label


    def _rules2str_text(self, cond):
        if self.clu_for_reg:
            cons_txt = np.round(cond[0], self.precision)
        else:
            # if self.class_encoder_ is not None:
            #    cons_txt = self.class_encoder_.inverse_transform([cond[0]])[0]
            # else:
            cons_txt = cond[0]  # TODO: check con rick
        return '%s' % cons_txt

    def _calculate_task_medoids(self):
        if self.clu_for_clf:
            if self.verbose:
                print(datetime.datetime.now(), 'Calculate class medoids.')

            return RuleTreeClassifier.calculate_task_medoids(node_dict=self._node_dict, X=self._X, y=self._y,
                                                             Xr=self._Xr)

        elif self.clu_for_reg:
            if self.verbose:
                print(datetime.datetime.now(), 'Calculate regression medoids.')

            return RuleTreeRegressor.calculate_task_medoids(node_dict=self._node_dict, X=self._X, y=self._y,
                                                            Xr=self._Xr)
