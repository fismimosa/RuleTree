import datetime
import random
from abc import ABC, abstractmethod
from itertools import count
from typing import Union
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder

from RuleTree import RuleTreeNode
from RuleTree.utils import MODEL_TYPE_CLF, MODEL_TYPE_REG, ObliqueHouseHolderSplit


class RuleTree(ABC):
    @staticmethod
    def calculate_medoid(X, Xr):
        median = np.median(X, axis=0)
        diff_with_median = np.sum(np.abs(median - X), axis=1)
        idx_min = np.argmin(diff_with_median)
        return Xr[idx_min].tolist()

    @staticmethod
    def calculate_mode(x):
        vals, counts = np.unique(x, return_counts=True)
        idx = np.argmax(counts)
        return vals[idx]

    @staticmethod
    def calculate_proba(y, nbr_classes):
        vals, counts = np.unique(y, return_counts=True)
        proba = (counts / len(y)).astype(float)
        proba_ret = np.zeros(nbr_classes)
        for i, v in enumerate(vals):
            proba_ret[v] = proba[i]
        return proba_ret

    def __init__(self,
                 max_depth: int = 4,
                 min_samples_leaf: int = 3,
                 max_nbr_nodes: int = 32,
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
        self.medoid_dict_ = None
        self.task_medoid_dict_ = None
        self.rules_to_tree_print_ = None
        self.maps = None
        self.data_encoder = None
        self.rules_ = None
        self.rules_s_ = None
        self.class_encoder_ = None
        self.nbr_classes_ = None
        self.y_is_numerical = None
        self._X = None
        self._Xr = None
        self._typed_X = None
        self._y = None
        self.labels_ = None
        self.root_ = None
        self.label_encoder_ = None

        self.is_cat_feat = None
        self.feature_values = None
        self.cat_indexes = None
        self.con_indexes = None
        self.feature_names = None

        self.is_cat_feat_r = None
        self.feature_values_r = None
        self.cat_indexes_r = None
        self.con_indexes_r = None

        self._queue = list()
        self._tree_structure = dict()
        self._node_dict = dict()
        self._leaf_rule = dict()
        self.tiebreaker = count()  # counter for the priority queue. Used in case of the same -len(idx)

        self.max_depth = max_depth
        self.max_nbr_nodes = max_nbr_nodes
        self.min_samples_leaf = min_samples_leaf
        self.min_samples_split = min_samples_split
        self.max_nbr_values = max_nbr_values
        self.max_nbr_values_cat = max_nbr_values_cat
        self.allow_oblique_splits = allow_oblique_splits
        self.force_oblique_splits = force_oblique_splits
        self.max_nbr_values = max_nbr_values
        self.max_nbr_values_cat = max_nbr_values_cat
        self.allow_oblique_splits = allow_oblique_splits
        self.force_oblique_splits = force_oblique_splits
        self.max_oblique_features = max_oblique_features
        self.prune_useless_leaves = prune_useless_leaves

        self.n_components = n_components
        self.bic_eps = bic_eps

        self.feature_names_r = feature_names
        self.one_hot_encode_cat = one_hot_encode_cat
        self.categorical_indices = categorical_indices
        self.numerical_indices = numerical_indices
        self.numerical_scaler = numerical_scaler

        self.precision = precision
        self.cat_precision = cat_precision
        self.exclude_split_feature_from_reduction = exclude_split_feature_from_reduction

        self.random_state = random_state
        self.n_jobs = n_jobs
        self.verbose = verbose

        self.kwargs = kwargs

        random.seed(random_state)

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)

        if self.categorical_indices is not None and self.numerical_indices is not None:
            if len(set(self.categorical_indices) & set(self.numerical_indices)) > 0:
                raise Exception('A feature cannot be categorical and numerical.')

        return self

    @abstractmethod
    def _make_leaf(self, node: RuleTreeNode):
        nbr_samples = len(node.idx)
        leaf_labels = np.array([node.label] * nbr_samples)
        node.samples = nbr_samples
        node.support = nbr_samples / len(self._X)
        node.is_leaf = True
        self.labels_[node.idx] = leaf_labels

    @abstractmethod
    def _make_split(self, idx_iter):
        pass

    def _make_supervised_split(self, idx_iter, clf):

        nbr_samples = len(idx_iter)

        clf.fit(self._X[idx_iter], self._y[idx_iter])
        labels = clf.apply(self._X[idx_iter])

        is_oblique = False
        if self.allow_oblique_splits:
            olq_clf = ObliqueHouseHolderSplit(
                max_oblique_features=self.max_oblique_features,
                min_samples_leaf=self.min_samples_leaf,
                min_samples_split=self.min_samples_split,
                random_state=self.random_state,
            )

            olq_clf.fit(self._X[idx_iter], self._y[idx_iter])
            labels_ob = olq_clf.apply(self._X[idx_iter])

            vals, counts = np.unique(labels, return_counts=True)
            if len(vals) == 1:
                impurity_ap = np.inf
            else:
                impurity_l = clf.tree_.impurity[1]
                impurity_r = clf.tree_.impurity[2]
                impurity_ap = counts[0] / nbr_samples * impurity_l + counts[1] / nbr_samples * impurity_r

            vals, counts = np.unique(labels_ob, return_counts=True)
            if len(vals) == 1:
                impurity_ob = np.inf
            else:
                impurity_l_ob = olq_clf.oblq_clf.tree_.impurity[1]
                impurity_r_ob = olq_clf.oblq_clf.tree_.impurity[2]
                impurity_ob = counts[0] / nbr_samples * impurity_l_ob + counts[1] / nbr_samples * impurity_r_ob

            if self.force_oblique_splits or impurity_ob < impurity_ap:
                clf = olq_clf
                is_oblique = True

        return clf, is_oblique

    @abstractmethod
    def fit(self, X, y=None):
        if self.verbose:
            print(datetime.datetime.now(), 'RULE TREE - BEGIN.')

        if self.max_nbr_nodes is None:
            self.max_nbr_nodes = len(self._X)

        # self.processPoolExecutor = ProcessPoolExecutor(self.n_jobs, initializer=init_pool, initargs=(__X,))

        self._X = X
        self._y = y

        if self._y is not None:
            if len(np.unique(self._y)) >= self.max_nbr_values_cat:  # infer is numerical
                self.y_is_numerical = True
            else:  # infer y is categorical
                self.y_is_numerical = False
                self.nbr_classes_ = len(np.unique(self._y))
                if isinstance(np.unique(self._y)[0], str):
                    self.class_encoder_ = LabelEncoder()
                    self.class_encoder_.fit(self._y)
                    self._y = self.class_encoder_.transform(self._y)

    @abstractmethod
    def predict(self, X, get_leaf=False, get_rule=False):
        X = RuleTree.preprocessing(X, self.feature_names_r, self.is_cat_feat, self.data_encoder, self.numerical_scaler)

        idx = np.arange(X.shape[0])
        labels, leaves, proba = self._predict(X, idx, self.root_)

        labels = self._predict_adjust_labels(labels)

        if get_leaf and get_rule:
            rules = list()
            for leaf_id in leaves:
                if leaf_id not in self.rules_s_:
                    continue
                rules.append(self.rules_s_[leaf_id])
            rules = np.array(rules)
            return labels, leaves, rules

        if get_rule:
            rules = list()
            for leaf_id in leaves:
                rules.append(self.rules_s_[leaf_id])
            rules = np.array(rules)
            return labels, rules

        if get_leaf:
            return labels, leaves

        return labels

    @abstractmethod
    def _predict_adjust_labels(self, labels):
        pass

    def predict_proba(self, X):
        X = RuleTree.preprocessing(X, self.feature_names_r, self.is_cat_feat, self.data_encoder, self.numerical_scaler)

        idx = np.arange(X.shape[0])
        _, _, proba = self._predict(X, idx, self.root_)

        return proba

    @abstractmethod
    def _predict(self, X, idx, node):
        idx_iter = idx

        if node.is_leaf:
            nbr_records = len(idx_iter)
            labels = np.array([node.label] * nbr_records)
            leaves = np.array([node.node_id] * nbr_records).astype(int)
            proba = np.repeat(np.reshape(node.proba, (1, -1)), nbr_records, axis=0)

            return labels, leaves, proba

        else:

            clf = node.clf
            labels = clf.apply(X[idx_iter])
            leaves = np.zeros(len(labels)).astype(int)
            proba = np.zeros((len(labels), self.nbr_classes_))

            idx_l, idx_r = np.where(labels == 1)[0], np.where(labels == 2)[0]
            idx_all_l = idx_iter[idx_l]
            idx_all_r = idx_iter[idx_r]

            if len(idx_all_l) > 0:
                labels_l, leaves_l, proba_l = self._predict(X, idx_all_l, node.node_l)
                labels[idx_l] = labels_l
                leaves[idx_l] = leaves_l
                proba[idx_l] = proba_l

            if len(idx_all_r) > 0:
                labels_r, leaves_r, proba_r = self._predict(X, idx_all_r, node.node_r)
                labels[idx_r] = labels_r
                leaves[idx_r] = leaves_r
                proba[idx_r] = proba_r

            return labels, leaves, proba

    @abstractmethod
    def transform(self, X, metric='euclidean', include_descriptive_pivot=True, include_target_pivot=False,
                  only_leaf_pivot=False):
        pass

    def _halting_conditions(self, node, nbr_samples, nbr_curr_nodes, node_depth):
        if nbr_samples < self.min_samples_split:
            self._make_leaf(node)

            if self.verbose:
                print(datetime.datetime.now(), 'Insufficient number of sample to split.')
            return True

        if nbr_curr_nodes + len(self._queue) + 1 >= self.max_nbr_nodes:
            self._make_leaf(node)
            if self.verbose:
                print(datetime.datetime.now(), 'Exceeded maximum number of nodes.')
            return True

        if self.max_depth is not None and node_depth >= self.max_depth:
            self._make_leaf(node)
            if self.verbose:
                print(datetime.datetime.now(), 'Exceeded maximum depth')
            return True

        return False

    def get_axes2d(self, eps=1, X=None):
        idx = np.arange(self._X.shape[0])

        if X is None:
            X = self._X

        return self._get_axes2d(idx, self.root_, eps, X)

    def _get_axes2d(self, idx, node: RuleTreeNode, eps, X):
        idx_iter = idx

        axes2d = list()

        if node.is_leaf:
            return []

        else:
            clf = node.clf
            labels = clf.apply(self._X[idx_iter])

            idx_l, idx_r = np.where(labels == 1)[0], np.where(labels == 2)[0]
            idx_all_l = idx_iter[idx_l]
            idx_all_r = idx_iter[idx_r]

            x_min, x_max = X[idx_iter][:, 0].min(), X[idx_iter][:, 0].max()
            y_min, y_max = X[idx_iter][:, 1].min(), X[idx_iter][:, 1].max()

            if not node.is_oblique:
                feat = clf.tree_.feature[0]
                thr = clf.tree_.threshold[0]

                if feat == 0:
                    axes = [[thr, thr], [y_min - eps, y_max + eps]]
                else:
                    axes = [[x_min - eps, x_max + eps], [thr, thr]]
            else:
                def line_fun(x):
                    f = clf.oblq_clf.tree_.feature[0]
                    b = clf.oblq_clf.tree_.threshold[0]
                    m = clf.householder_matrix[:, f][0] / clf.householder_matrix[:, f][1]
                    y = b / clf.householder_matrix[:, f][1] - m * x
                    return y

                axes = [
                    [x_min - eps, x_max + eps],
                    [line_fun(x_min - eps), line_fun(x_max + eps)],
                ]

            axes2d.append(axes)

            axes2d += self._get_axes2d(idx_all_l, node.node_l, eps, X)
            axes2d += self._get_axes2d(idx_all_r, node.node_r, eps, X)

            return axes2d

    def _get_rules_to_print_tree(self):
        if self.verbose:
            print(datetime.datetime.now(), 'Retrive rules to print tree.')
        idx = np.arange(self._X.shape[0])
        return self.__get_rules_to_print_tree(idx, self.root_, 0)

    def __inverse_scale_thr(self, feat, thr, cat, oblique):
        if self.numerical_scaler is None or cat or oblique:
            return thr
        fake_x = np.zeros(self._X.shape[1])
        fake_x[feat] = thr
        X_to_invert = np.array([fake_x])
        X_to_invert[:, ~self.is_cat_feat] = self.numerical_scaler.inverse_transform(X_to_invert[:, ~self.is_cat_feat])
        scaled_thr = X_to_invert[0, feat]
        return scaled_thr

    def __inverse_scale_thr_obl(self, feat_list, coef, thr):
        if self.numerical_scaler is None:
            return thr

        if isinstance(self.numerical_scaler, StandardScaler):
            mean_list = self.numerical_scaler.mean_[feat_list]
            std_list = self.numerical_scaler.scale_[feat_list]
            new_thr = thr * np.prod(std_list) + np.sum(coef * mean_list)
            return new_thr
        elif isinstance(self.numerical_scaler, MinMaxScaler):
            min_list = self.numerical_scaler.data_min_[feat_list]
            max_list = self.numerical_scaler.data_max_[feat_list]
            den = max_list - min_list
            new_thr = thr * np.prod(den) + np.sum(coef * min_list)
            return new_thr
        else:
            raise Exception('Unsupported oblique inverse scaler transform for %s' % type(self.numerical_scaler))

    def _prune_useless_leaves(self):
        while True:
            tree_pruned = False
            nodes_to_remove = list()
            for node_id in self._tree_structure:
                node_l, node_r = self._tree_structure[node_id]
                if node_l not in self._node_dict or node_r not in self._node_dict:
                    continue
                if self._node_dict[node_l].is_leaf and self._node_dict[node_r].is_leaf and \
                        self._node_dict[node_l].label == self._node_dict[node_r].label:
                    self._make_leaf(self._node_dict[node_id])
                    del self._node_dict[node_l]
                    del self._node_dict[node_r]
                    self._node_dict[node_id].node_l = None
                    self._node_dict[node_id].node_r = None
                    nodes_to_remove.append(node_id)
                    tree_pruned = True

            if not tree_pruned:
                break

            for node_id in nodes_to_remove:
                del self._tree_structure[node_id]

    def _is_verified_x_r(self, x, rule, count_cond=False, relative_count=False):
        verified = True
        count_verified = 0
        for cond in rule[:-1]:
            feat, symb, thr, cat, is_oblique, coef = cond

            if not is_oblique:
                if cat:  # categorical
                    # thr = self.feature_names[feat].split('_')[-1]
                    if symb == '==':
                        verified = verified and x[feat] > thr
                    else:  # symb = '!='
                        verified = verified and x[feat] <= thr

                else:  # continuous
                    if symb == '<=':
                        verified = verified and x[feat] <= thr
                    else:  # symb = '>'
                        verified = verified and x[feat] > thr
            else:  # is oblique
                if symb == '<=':
                    verified = verified and np.sum(x[feat] * coef) <= thr
                else:  # symb = '>'
                    verified = verified and np.sum(x[feat] * coef) > thr

            if verified:
                count_verified += 1

            if not verified and not count_cond:
                return False

        if count_cond:

            if relative_count:
                return count_verified / len(rule[:-1])

            return count_verified

        return verified

    def _is_verified_r_(self, X, rule, count_cond=False, relative_count=False):
        verified_list = list()
        for x in X:
            # print(x)
            verified_list.append(self._is_verified_x_r(x, rule, count_cond, relative_count))

        verified_list = np.array(verified_list)
        return verified_list

    def are_rules_verified(self, X, count_cond=False, relative_count=False):
        X = RuleTree.preprocessing(X, self.feature_names_r, self.is_cat_feat, self.data_encoder, self.numerical_scaler)

        verified_list = list()
        for leaf, rule in self.rules_.items():
            verified_list.append(self._is_verified_r_(X, rule, count_cond, relative_count))

        verified_list = np.array(verified_list).T
        return verified_list

    def __get_rules_to_print_tree(self, idx_iter, node: RuleTreeNode, cur_depth):
        rules = list()

        if node.is_leaf:
            label = self.__get_labels(node=node)
            leaf = (False, label, node.samples, node.support, node.node_id, cur_depth)
            rules.append(leaf)
            return rules

        else:
            clf = node.clf
            labels = clf.apply(self._X[idx_iter])

            idx_l, idx_r = np.where(labels == 1)[0], np.where(labels == 2)[0]
            idx_all_l = idx_iter[idx_l]
            idx_all_r = idx_iter[idx_r]

            if not node.is_oblique:
                feat = clf.tree_.feature[0]
                cat = feat in self.cat_indexes
                thr = self.__inverse_scale_thr(feat, clf.tree_.threshold[0], cat, node.is_oblique)
                rule = (True, [feat], [1.0], thr, cat, cur_depth)
            else:
                pca_feat = clf.oblq_clf.tree_.feature[0]
                thr = clf.oblq_clf.tree_.threshold[0]
                feat_list = np.where(clf.u_weights != 0)[0].tolist()
                coef = clf.householder_matrix[:, pca_feat][feat_list].tolist()
                thr = self.__inverse_scale_thr_obl(feat_list, coef, thr)
                rule = (True, feat_list, coef, thr, False, cur_depth)

            rules.append(rule)
            rules += self.__get_rules_to_print_tree(idx_all_l, node.node_l, cur_depth + 1)
            rules += self.__get_rules_to_print_tree(idx_all_r, node.node_r, cur_depth + 1)
            return rules

    @abstractmethod
    def _get_labels(self, node):
        pass

    def _calculate_rules(self):
        if self.verbose:
            print(datetime.datetime.now(), 'Calculate rules.')

        for node_id in self._node_dict:
            node = self._node_dict[node_id]
            if not node.is_leaf:
                continue

            label = self.__get_labels(node.label)

            rule = [(label,)]

            past_node_id = node_id
            next_node_id = node.parent_id
            if next_node_id == -1:
                break
            next_node = self._node_dict[next_node_id]

            while True:

                clf = next_node.clf
                if not next_node.is_oblique:
                    feat = clf.tree_.feature[0]
                    thr = clf.tree_.threshold[0]
                    cat = feat in self.cat_indexes
                    coef = None
                else:
                    pca_feat = clf.oblq_clf.tree_.feature[0]
                    thr = clf.oblq_clf.tree_.threshold[0]
                    feat = np.where(clf.u_weights != 0)[0].tolist()
                    coef = clf.householder_matrix[:, pca_feat][feat].tolist()
                    cat = False

                if next_node.node_l.node_id == past_node_id:
                    symb = '<=' if not cat else '!='
                else:
                    symb = '>' if not cat else '=='

                cond = (feat, symb, thr, cat, next_node.is_oblique, coef)

                rule.insert(0, cond)
                past_node_id = next_node_id
                next_node_id = next_node.parent_id
                self._leaf_rule[node_id] = rule

                if next_node_id == -1:
                    break
                next_node = self._node_dict[next_node_id]

        return self._leaf_rule

    def _calculate_all_medoids(self):
        if self.verbose:
            print(datetime.datetime.now(), 'Calculate medoids.')

        medoid_dict = dict()
        for node_id in self._node_dict:
            node = self._node_dict[node_id]
            node.medoid = RuleTree.calculate_medoid(self._X[node.idx], self._Xr[node.idx])
            medoid_dict[node_id] = node.medoid
        return medoid_dict

    def _rules2str(self):
        if self.verbose:
            print(datetime.datetime.now(), 'Turn rules into strings.')

        self.rules_s_ = dict()

        for leaf_id in self.rules_:
            rule = self.rules_[leaf_id]
            self.rules_s_[leaf_id] = list()
            for cond in rule:
                if len(cond) == 1:
                    cond_s = self._rules2str_text(cond=cond)

                else:
                    feat, symb, thr, cat, is_oblique, coef = cond

                    if not is_oblique:
                        feat_s = "%s" % self.feature_names[feat]
                    else:
                        thr = self.__inverse_scale_thr_obl(feat, coef, thr)
                        feat_s = [
                            "%s %s"
                            % (np.round(coef[i], self.precision), self.feature_names[feat[i]])
                            for i in range(len(feat))
                        ]
                        feat_s = " + ".join(feat_s)

                    if not cat:
                        thr = self.__inverse_scale_thr(feat, thr, cat, is_oblique)
                        cond_s = "%s %s %s" % (feat_s, symb, np.round(thr, self.precision))
                    else:
                        if self.one_hot_encode_cat:
                            feat_vals = feat_s.split('_')
                            thr = feat_vals[-1]
                            feat_s = '_'.join(feat_vals[:-1])
                            cond_s = "%s %s %s" % (feat_s, symb, thr)
                        else:
                            cond_s = "%s %s %s" % (feat_s, symb, np.round(thr, self.cat_precision))
                self.rules_s_[leaf_id].append(cond_s)
            antecedent = ' & '.join(self.rules_s_[leaf_id][:-1])
            consequent = ' --> %s' % self.rules_s_[leaf_id][-1]
            self.rules_s_[leaf_id] = antecedent + consequent
        return self.rules_s_

    @abstractmethod
    def _rules2str_text(self, cond):
        pass

    def _compact_rules(self):
        if self.verbose:
            print(datetime.datetime.now(), 'Compact rules.')

        compact_rules = dict()
        for leaf_id in self.rules_:
            rule = self.rules_[leaf_id]
            compact_rules[leaf_id] = list()
            rule_dict = dict()
            rule_ob_split = list()
            for cond in rule:
                if len(cond) > 1:
                    feat, symb, thr, cat, is_oblique, coef = cond
                    if is_oblique:
                        rule_ob_split.append(cond)
                        continue
                    if (feat, symb) not in rule_dict:
                        rule_dict[(feat, symb)] = list()
                    rule_dict[(feat, symb)].append(cond)

            for k in rule_dict:
                feat, symb = k
                cond_list = rule_dict[(feat, symb)]
                if len(cond_list) == 1:
                    cond = cond_list[0]
                    compact_rules[leaf_id].append(cond)
                else:
                    thr_list = [cond[2] for cond in cond_list]
                    cond = list(rule_dict[k][0])
                    if symb == '<=':
                        cond[2] = min(thr_list)
                    elif symb == '>':
                        cond[2] = max(thr_list)

                    compact_rules[leaf_id].append(tuple(cond))

            for cond in rule_ob_split:
                compact_rules[leaf_id].append(cond)

            compact_rules[leaf_id].append(rule[-1])

        return compact_rules

    @abstractmethod
    def _calculate_task_medoids(self):
        pass


    @staticmethod
    def print_tree(self, precision=2, cat_precision=2):
        rules = self.rules_to_tree_print_

        s_rules = ""
        for i, rule in enumerate(rules):
            is_rule = rule[0]
            depth = rule[-1]
            # ident = "  " * depth
            ident = "|  " * depth
            if is_rule:
                _, feat_list, coef_list, thr, cat, _ = rule
                if len(feat_list) == 1:
                    feat_s = "%s" % self.feature_names[feat_list[0]]
                else:
                    feat_s = [
                        "%s %s" % (np.round(coef_list[i], precision), self.feature_names[feat_list[i]])
                        for i in range(len(feat_list))]
                    feat_s = " + ".join(feat_s)
                if not cat:
                    cond_s = "%s <= %s" % (feat_s, np.round(thr, precision))
                else:
                    if self.one_hot_encode_cat:
                        feat_vals = feat_s.split('_')
                        thr = feat_vals[-1]
                        feat_s = '_'.join(feat_vals[:-1])
                        cond_s = "%s != %s" % (feat_s, thr)
                    else:
                        cond_s = "%s <= %s" % (feat_s, np.round(thr, cat_precision))
                # s = "%s|-+ if %s:" % (ident, cond_s)
                if i > 0:
                    # s = "%s-+ if %s:" % (ident[:-1], cond_s)
                    s = "%s+--+ if %s:" % (ident[:-3], cond_s)
                else:
                    # s = "%s+ if %s:" % (ident, cond_s)
                    s = "%sif %s:" % (ident, cond_s)
            else:
                _, label, samples, support, node_id, _ = rule
                support = np.round(support, precision)
                if self.model_type == MODEL_TYPE_REG or self.clu_for_reg:
                    text = 'value'
                elif self.model_type == MODEL_TYPE_CLF or self.clu_for_clf:
                    text = 'class'
                else:
                    text = 'cluster'
                label_txt = np.round(label, precision
                                     ) if self.model_type == MODEL_TYPE_REG or self.clu_for_reg else label
                # s = "%s|--> %s: %s (%s, %s)" % (ident, text, label_txt, samples, support)
                # s = "%s--> %s: %s (%s, %s)" % (ident[:-1], text, label_txt, samples, support)
                s = "%s+--> %s: %s (%s, %s)" % (ident[:-3], text, label_txt, samples, support)
            s_rules += "%s\n" % s

        return s_rules
