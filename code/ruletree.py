import heapq
import random
import datetime
import numpy as np
import pandas as pd
from typing import Union
from itertools import count

import statistics as st
from scipy.linalg import norm
from scipy.spatial.distance import cdist

import category_encoders as ce

from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.metrics import r2_score

import light_famd
from bic_estimator import bic

MODEL_TYPE_CLF = 'clf'
MODEL_TYPE_REG = 'reg'
MODEL_TYPE_CLU = 'clu'


# supervised (reg e clf) e unsupervised

def prepare_data(X_original, max_nbr_values, max_nbr_values_cat, feature_names_original, one_hot_encode_cat,
                 categorical_indices, numerical_indices, numerical_scaler):

    if categorical_indices is not None and numerical_indices is not None:
        if len(categorical_indices) + len(numerical_indices) != X_original.shape[1]:
            raise Exception('Provided indices are different from dataset size.')

    if categorical_indices is None and numerical_indices is not None:
        categorical_indices = [i for i in range(X_original.shape[1]) if i not in numerical_indices]

    X = np.copy(X_original)
    if not one_hot_encode_cat and categorical_indices is None:
        X = X.astype(float)

    n_features = X.shape[1]

    if categorical_indices:
        for feature in range(n_features):
            if feature not in categorical_indices:
                X[:,feature] = X[:,feature].astype(float)

    feature_values = dict()
    is_categorical_feature = np.full_like(np.zeros(n_features, dtype=bool), False)

    if categorical_indices is None:
        for feature in range(n_features):
            values = np.unique(X[:, feature])
            vals = None
            if len(values) > max_nbr_values:   # this reduces the number of values for continuous attributes
                _, vals = np.histogram(values, bins=max_nbr_values)
                values = [(vals[i] + vals[i + 1]) / 2 for i in range(len(vals) - 1)]
            feature_values[feature] = values

            if len(values) <= max_nbr_values_cat:       # this identifies categorical attributes
                is_categorical_feature[feature] = True

                if vals is not None:
                    for original_val_idx in range(X.shape[0]):
                        for min_val, max_val, binned_val in zip(vals[:-1], vals[1:], values):
                            original_val = X[original_val_idx, feature]
                            if min_val < original_val < max_val:
                                X[original_val_idx, feature] = binned_val
                                break

    else:
        for feature in range(n_features):
            values = np.unique(X[:, feature])
            if len(values) > max_nbr_values:   # this reduces the number of values for continuous attributes
                _, vals = np.histogram(values, bins=max_nbr_values)
                values = [(vals[i] + vals[i + 1]) / 2 for i in range(len(vals) - 1)]
            feature_values[feature] = values

            if feature in categorical_indices:
                is_categorical_feature[feature] = True

    is_categorical_feature_r = np.copy(is_categorical_feature)
    feature_values_r = {k: feature_values[k] for k in feature_values}

    cols = feature_names_original[np.where(is_categorical_feature_r)[0]]
    encoder = None
    feature_names = None
    maps = None
    if len(cols) > 0 and one_hot_encode_cat:
        encoder = ce.OneHotEncoder(cols=cols, use_cat_names=True)
        df = encoder.fit_transform(pd.DataFrame(data=X, columns=feature_names_original))
        X = df.values
        feature_names = df.columns.tolist()
        map_original_onehot = dict()
        map_onehot_original = dict()
        map_original_onehot_idx = dict()
        map_onehot_original_idx = dict()
        for i, c1 in enumerate(feature_names_original):
            map_original_onehot[c1] = list()
            map_original_onehot_idx[i] = list()
            for j, c2 in enumerate(feature_names):
                if c2.startswith(c1):
                    map_original_onehot[c1].append(c2)
                    map_original_onehot_idx[i].append(j)
                    map_onehot_original[c2] = c1
                    map_onehot_original_idx[j] = i

        maps = {
            'map_original_onehot': map_original_onehot,
            'map_onehot_original': map_onehot_original,
            'map_original_onehot_idx': map_original_onehot_idx,
            'map_onehot_original_idx': map_onehot_original_idx
        }

        feature_values = dict()
        n_features = X.shape[1]
        is_categorical_feature = np.full_like(np.zeros(n_features, dtype=bool), False)
        for feature in range(n_features):
            values = np.unique(X[:, feature])
            feature_values[feature] = values

            if len(values) <= max_nbr_values_cat:  # this identifies categorical attributes
                is_categorical_feature[feature] = True

    # print(is_categorical_feature)

    if numerical_scaler is not None and np.sum(~is_categorical_feature) > 0:
        X[:, ~is_categorical_feature] = numerical_scaler.fit_transform(X[:, ~is_categorical_feature])

    features = (feature_values_r, is_categorical_feature_r,
                feature_values, is_categorical_feature,
                encoder, feature_names)

    return X, features, maps


def preprocessing(X, feature_names_r, is_cat_feat, data_encoder=None, numerical_scaler=None):
    if data_encoder is not None:
        df = pd.DataFrame(data=X, columns=feature_names_r)
        X = data_encoder.transform(df).values

    if numerical_scaler is not None and np.sum(~is_cat_feat) > 0:
        X[:, ~is_cat_feat] = numerical_scaler.transform(X[:, ~is_cat_feat])

    return X


def inverse_preprocessing(X, is_cat_feat, data_encoder=None, numerical_scaler=None):
    if numerical_scaler is not None and np.sum(~is_cat_feat) > 0:
        X[:, ~is_cat_feat] = numerical_scaler.inverse_transform(X[:, ~is_cat_feat])

    if data_encoder is not None:
        X = data_encoder.inverse_transform(X).values

    return X


def calculate_medoid(X, Xr):
    median = np.median(X, axis=0)
    diff_with_median = np.sum(np.abs(median - X), axis=1)
    idx_min = np.argmin(diff_with_median)
    return Xr[idx_min].tolist()


def calculate_mode(x):
    vals, counts = np.unique(x, return_counts=True)
    idx = np.argmax(counts)
    return vals[idx]

# def calculate_mode(x, get_proba=False):
#     vals, counts = np.unique(x, return_counts=True)
#     idx = np.argmax(counts)
#     if not get_proba:
#         return vals[idx]
#     proba = (counts / len(x)).astype(float)
#     return vals[idx], proba


def calculate_proba(y, nbr_classes):
    vals, counts = np.unique(y, return_counts=True)
    proba = (counts / len(y)).astype(float)
    proba_ret = np.zeros(nbr_classes)
    for i, v in enumerate(vals):
        proba_ret[v] = proba[i]
    return proba_ret


class ObliqueHouseHolderSplit:
    def __init__(
        self,
        pca=None,
        max_oblique_features=2,
        min_samples_leaf=3,
        min_samples_split=5,
        tau=1e-4,
        model_type='clf',
        clf_impurity: str = 'gini',
        reg_impurity: str = 'squared_error',
        random_state=None,
    ):
        self.pca = pca
        self.max_oblique_features = max_oblique_features
        self.min_samples_leaf = min_samples_leaf
        self.min_samples_split = min_samples_split
        self.tau = tau
        self.model_type = model_type
        self.clf_impurity = clf_impurity
        self.reg_impurity = reg_impurity

        self.dominant_ev = None
        self.u_weights = None
        self.householder_matrix = None
        self.oblq_clf = None
        self.random_state = random_state

    def transform(self, X):
        X_house = X.dot(self.householder_matrix)
        return X_house

    def fit(self, X, y):

        n_features = X.shape[1]

        if self.pca is None:
            self.pca = PCA(n_components=1)
            self.pca.fit(X)

        self.dominant_ev = self.pca.components_[0]
        I = np.diag(np.ones(n_features))

        diff_w_means = np.sqrt(((I - self.dominant_ev) ** 2).sum(axis=1))

        if (diff_w_means > self.tau).sum() == 0:
            print("No variance to explain.")
            return None

        idx_max_diff = np.argmax(diff_w_means)
        e_vector = np.zeros(n_features)
        e_vector[idx_max_diff] = 1.0
        self.u_weights = (e_vector - self.dominant_ev) / norm(e_vector - self.dominant_ev)

        if self.max_oblique_features < n_features:
            idx_w = np.argpartition(np.abs(self.u_weights), -self.max_oblique_features)[-self.max_oblique_features:]
            u_weights_new = np.zeros(n_features)
            u_weights_new[idx_w] = self.u_weights[idx_w]
            self.u_weights = u_weights_new

        self.householder_matrix = I - 2 * self.u_weights[:, np.newaxis].dot(self.u_weights[:, np.newaxis].T)

        X_house = self.transform(X)

        if self.model_type == MODEL_TYPE_CLF:
            self.oblq_clf = DecisionTreeClassifier(
                max_depth=1,
                criterion=self.clf_impurity,
                min_samples_leaf=self.min_samples_leaf,
                min_samples_split=self.min_samples_split,
                random_state=self.random_state,
            )
        elif self.model_type == MODEL_TYPE_REG:
            self.oblq_clf = DecisionTreeRegressor(
                max_depth=1,
                criterion=self.reg_impurity,
                min_samples_leaf=self.min_samples_leaf,
                min_samples_split=self.min_samples_split,
                random_state=self.random_state,
            )
        else:
            raise Exception('Unknown model %s' % self.model_type)
        self.oblq_clf.fit(X_house, y)

    def predict(self, X):
        X_house = self.transform(X)
        return self.oblq_clf.predict(X_house)

    def apply(self, X):
        X_house = self.transform(X)
        return self.oblq_clf.apply(X_house)


class RuleTreeNode:

    def __init__(self, idx, node_id, label, parent_id, is_leaf=False, clf=None, node_l=None, node_r=None,
                 samples=None, support=None, impurity=None, is_oblique=None, proba=None):
        self.idx = idx
        self.node_id = node_id
        self.label = label
        self.is_leaf = is_leaf
        self.clf = clf
        self.node_l = node_l
        self.node_r = node_r
        self.samples = samples
        self.support = support
        self.impurity = impurity
        self.is_oblique = is_oblique
        self.parent_id = parent_id
        self.medoid = None
        self.class_medoid = None
        self.proba = proba


class RuleTree:

    def __init__(
            self,
            model_type: str = 'clf',

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
            clus_impurity: str = 'r2',
            clf_impurity: str = 'gini',
            reg_impurity: str = 'squared_error',

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
            verbose=False
    ):
        self.model_type = model_type

        self.max_depth = max_depth
        self.max_nbr_nodes = max_nbr_nodes
        self.min_samples_leaf = min_samples_leaf
        self.min_samples_split = min_samples_split
        self.max_nbr_values = max_nbr_values
        self.max_nbr_values_cat = max_nbr_values_cat
        self.allow_oblique_splits = allow_oblique_splits
        self.force_oblique_splits = force_oblique_splits
        self.max_oblique_features = max_oblique_features
        self.prune_useless_leaves = prune_useless_leaves

        self.n_components = n_components
        self.bic_eps = bic_eps
        self.clus_impurity = clus_impurity
        self.clf_impurity = clf_impurity
        self.reg_impurity = reg_impurity

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
        # self.processPoolExecutor = None

        random.seed(self.random_state)

        self.__X = None
        self.__Xr = None
        self.__typed_X = None
        self.__y = None
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

        self.__queue = list()
        self.__tree_structure = dict()
        self.__node_dict = dict()
        self.__leaf_rule = dict()
        self.rules_to_tree_print_ = None
        self.rules_ = None
        self.rules_s_ = None

        self.clu_for_clf = False
        self.clu_for_reg = False
        self.data_encoder = None

        self.medoid_dict_ = None
        self.class_medoid_dict_ = None
        self.regr_medoid_dict_ = None

        self.maps = None

        if categorical_indices is not None and numerical_indices is not None:
            if len(set(categorical_indices) & set(numerical_indices)) > 0:
                raise Exception('A feature cannot be categorical and numerical.')

        self.categorical_indices = categorical_indices
        self.numerical_indices = numerical_indices

        self.nbr_classes_ = None
        self.class_encoder_ = None

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)

        if self.categorical_indices is not None and self.numerical_indices is not None:
            if len(set(self.categorical_indices) & set(self.numerical_indices)) > 0:
                raise Exception('A feature cannot be categorical and numerical.')

        return self

    def _make_leaf(self, node: RuleTreeNode):
        nbr_samples = len(node.idx)
        leaf_labels = np.array([node.label] * nbr_samples)
        if self.model_type == MODEL_TYPE_CLF or self.clu_for_clf:
            leaf_labels = leaf_labels.astype(int)
        node.samples = nbr_samples
        node.support = nbr_samples / len(self.__X)
        node.is_leaf = True
        self.labels_[node.idx] = leaf_labels

    def _make_supervised_split(self, idx_iter):

        nbr_samples = len(idx_iter)

        if self.model_type == MODEL_TYPE_CLF:
            clf = DecisionTreeClassifier(
                max_depth=1,
                criterion=self.clf_impurity,
                min_samples_leaf=self.min_samples_leaf,
                min_samples_split=self.min_samples_split,
                random_state=self.random_state,
            )
        elif self.model_type == MODEL_TYPE_REG:
            clf = DecisionTreeRegressor(
                max_depth=1,
                criterion=self.reg_impurity,
                min_samples_leaf=self.min_samples_leaf,
                min_samples_split=self.min_samples_split,
                random_state=self.random_state,
            )
        else:
            raise Exception('Unknown model %s' % self.model_type)

        clf.fit(self.__X[idx_iter], self.__y[idx_iter])
        labels = clf.apply(self.__X[idx_iter])

        is_oblique = False
        if self.allow_oblique_splits:
            olq_clf = ObliqueHouseHolderSplit(
                max_oblique_features=self.max_oblique_features,
                min_samples_leaf=self.min_samples_leaf,
                min_samples_split=self.min_samples_split,
                random_state=self.random_state,
            )

            olq_clf.fit(self.__X[idx_iter], self.__y[idx_iter])
            labels_ob = olq_clf.apply(self.__X[idx_iter])

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

    def _make_unsupervised_split(self, idx_iter):

        if self.exclude_split_feature_from_reduction:
            if self.verbose:
                print(datetime.datetime.now(), 'Exclude split feature from reduction.')
            return self._make_unsupervised_split_iter(idx_iter)

        nbr_samples = len(idx_iter)
        n_components_split = min(self.n_components, nbr_samples)

        if len(self.cat_indexes_r) == 0:  # all continous
            principal_transform = light_famd.PCA(n_components=n_components_split, random_state=self.random_state)
        elif len(self.con_indexes_r) == 0:  # all caregorical
            principal_transform = light_famd.MCA(n_components=n_components_split, random_state=self.random_state)
        else:                               # mixed
            principal_transform = light_famd.FAMD(n_components=n_components_split, random_state=self.random_state)

        y_pca = principal_transform.fit_transform(self.__typed_X.loc[idx_iter])

        clf_list = list()
        eval_list = list()
        for i in range(n_components_split):

            if not self.force_oblique_splits:
                clf = DecisionTreeRegressor(
                    max_depth=1,
                    min_samples_leaf=self.min_samples_leaf,
                    min_samples_split=self.min_samples_split,
                    random_state=self.random_state,
                )
                clf.fit(self.__X[idx_iter], y_pca[:, i])
                if self.clus_impurity == 'r2':
                    eval_val = -1 * r2_score(clf.predict(self.__X[idx_iter]), y_pca[:, i])
                elif self.clus_impurity == 'bic':
                    labels_i = clf.apply(self.__X[idx_iter])
                    eval_val = bic(self.__X[idx_iter], (np.array(labels_i) - 1).tolist())
                else:
                    raise Exception('Unknown clustering impurity measure %s' % self.clus_impurity)

                clf_list.append(clf)
                eval_list.append(eval_val)

            if self.allow_oblique_splits and i > 0:
                olq_clf = ObliqueHouseHolderSplit(
                    pca=principal_transform,
                    max_oblique_features=self.max_oblique_features,
                    min_samples_leaf=self.min_samples_leaf,
                    min_samples_split=self.min_samples_split,
                    model_type=MODEL_TYPE_REG,
                    random_state=self.random_state,
                )

                olq_clf.fit(self.__X[idx_iter], y_pca[:, i])

                if self.clus_impurity == 'r2':
                    eval_ob = -1 * r2_score(olq_clf.predict(self.__X[idx_iter]), y_pca[:, i])
                elif self.clus_impurity == 'bic':
                    labels_i = olq_clf.apply(self.__X[idx_iter])
                    eval_ob = bic(self.__X[idx_iter], (np.array(labels_i) - 1).tolist())
                else:
                    raise Exception('Unknown clustering impurity measure %s' % self.clus_impurity)

                clf_list.append(olq_clf)
                eval_list.append(eval_ob)

        idx_min = np.argmin(eval_list)
        is_oblique = self.force_oblique_splits or self.allow_oblique_splits and idx_min > 0 and idx_min % 2 == 0
        clf = clf_list[idx_min]

        return clf, is_oblique

    def _make_unsupervised_split_iter(self, idx_iter):

        nbr_samples = len(idx_iter)
        n_components_split = min(self.n_components, nbr_samples)

        features_idx = np.arange(self.__Xr.shape[1])
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

            y_pca = principal_transform.fit_transform(self.__typed_X.loc[idx_iter, features_idx != feature])

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
                data_for_training = np.zeros(self.__X[idx_iter].shape)
                data_for_training[:, features_to_use] = self.__X[idx_iter][:, features_to_use]

                clf.fit(data_for_training, y_pca[:, i])
                if self.clus_impurity == 'r2':
                    eval_val = -1 * r2_score(clf.predict(self.__X[idx_iter]), y_pca[:, i])
                elif self.clus_impurity == 'bic':
                    labels_i = clf.apply(self.__X[idx_iter])
                    eval_val = bic(self.__X[idx_iter], (np.array(labels_i) - 1).tolist())
                else:
                    raise Exception('Unknown clustering impurity measure %s' % self.clus_impurity)

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
        if self.verbose:
            print(datetime.datetime.now(), 'RULE TREE - BEGIN.')

        # self.processPoolExecutor = ProcessPoolExecutor(self.n_jobs, initializer=init_pool, initargs=(__X,))
        tiebreaker = count()  # counter for the priority queue. Used in case of the same -len(idx)

        self.__X = X
        self.__y = y

        if self.__y is None:
            self.model_type = MODEL_TYPE_CLU

        y_is_numerical = None
        if self.__y is not None:
            if len(np.unique(self.__y)) >= self.max_nbr_values_cat:  # infer is numerical
                y_is_numerical = True
            else:                                                    # infer y is categorical
                y_is_numerical = False
                self.nbr_classes_ = len(np.unique(self.__y))
                if isinstance(np.unique(self.__y)[0], str):
                    self.class_encoder_ = LabelEncoder()
                    self.class_encoder_.fit(self.__y)
                    self.__y = self.class_encoder_.transform(self.__y)

            if self.model_type == MODEL_TYPE_CLU:
                if y_is_numerical:
                    self.clu_for_reg = True
                else:
                    self.clu_for_clf = True
                    self.nbr_classes_ = len(np.unique(self.__y))

        if self.verbose:
            print(datetime.datetime.now(), 'Model type: %s.' % self.model_type)
            print(datetime.datetime.now(), 'Clustering for classification: %s.' % self.clu_for_clf)
            print(datetime.datetime.now(), 'Clustering for regression: %s.' % self.clu_for_reg)

        n_idx = X.shape[0]
        idx = np.arange(n_idx)

        self.labels_ = -1 * np.ones(n_idx).astype(int)
        if self.model_type == MODEL_TYPE_REG or self.clu_for_reg:
            self.labels_ = (-1.0 * np.ones(n_idx)).astype(float)

        node_id = 0
        proba = None
        if self.__y is None:
            majority_class = node_id
        elif self.model_type == MODEL_TYPE_CLF:
            majority_class = calculate_mode(self.__y)
            proba = calculate_proba(self.__y, self.nbr_classes_)   # TODO se classi non intere fare mapping
        elif self.model_type == MODEL_TYPE_REG:
            majority_class = np.mean(self.__y)
        else:
            if len(np.unique(self.__y)) >= self.max_nbr_values_cat:  # infer is numerical
                majority_class = np.mean(self.__y)
            else:  # infer it is categorical
                majority_class = calculate_mode(self.__y)
                proba = calculate_proba(y, self.nbr_classes_)

        root_node = RuleTreeNode(idx, node_id, majority_class, proba=proba, parent_id=-1)
        self.__node_dict[root_node.node_id] = root_node

        if self.model_type == MODEL_TYPE_CLU:
            heapq.heappush(self.__queue, (-len(idx), (next(tiebreaker), idx, 0, root_node)))
        else:
            self.__queue.append((idx, 0, root_node))

        if self.feature_names_r is None:
            self.feature_names_r = np.array(['X_%s' % i for i in range(X.shape[1])])
        else:
            self.feature_names_r = np.array(self.feature_names_r)

        self.__Xr = X
        res = prepare_data(self.__X, self.max_nbr_values, self.max_nbr_values_cat,
                           self.feature_names_r, self.one_hot_encode_cat, self.categorical_indices,
                           self.numerical_indices, self.numerical_scaler)
        self.__X = res[0]
        self.feature_values_r, self.is_cat_feat_r, self.feature_values, self.is_cat_feat, self.data_encoder, self.feature_names = res[1]
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

        if self.model_type == MODEL_TYPE_CLU:
            self.__typed_X = pd.DataFrame(data=self.__Xr)
            for index in self.con_indexes_r:
                self.__typed_X[index] = self.__typed_X[index].astype(float)
            for index in self.cat_indexes_r:
                self.__typed_X[index] = self.__typed_X[index].astype(str)
            self.__typed_X.columns = self.__typed_X.columns.astype(str)

            if len(self.cat_indexes_r) == 0:  # all continous
                if self.verbose:
                    print(datetime.datetime.now(), 'All continuous features, use PCA.')
            elif len(self.con_indexes_r) == 0:  # all caregorical
                if self.verbose:
                    print(datetime.datetime.now(), 'All categorical features, use MCA.')
            else:  # mixed
                if self.verbose:
                    print(datetime.datetime.now(), 'All categorical features, use FAMD.')

        if self.max_nbr_nodes is None:
            self.max_nbr_nodes = len(self.__X)

        if self.verbose:
            print(datetime.datetime.now(), 'Training started')

        nbr_curr_nodes = 0
        iter_id = 0
        while len(self.__queue) > 0 and nbr_curr_nodes + len(self.__queue) <= self.max_nbr_nodes:
            if self.verbose:
                print(datetime.datetime.now(), 'Iteration: %s, current nbr leaves: %s.' % (iter_id, nbr_curr_nodes))
            iter_id += 1

            if self.model_type == MODEL_TYPE_CLU:
                _, (_, idx_iter, node_depth, node) = heapq.heappop(self.__queue)
            else:
                (idx_iter, node_depth, node) = self.__queue.pop(0)

            if self.model_type == MODEL_TYPE_CLF and len(np.unique(self.__y[idx_iter])) == 1:
                self._make_leaf(node)
                nbr_curr_nodes += 1
                if self.verbose:
                    print(datetime.datetime.now(), 'Classification, node with unique target.')
                continue

            nbr_samples = len(idx_iter)

            if nbr_samples < self.min_samples_split:
                self._make_leaf(node)
                nbr_curr_nodes += 1
                if self.verbose:
                    print(datetime.datetime.now(), 'Insufficient number of sample to split.')
                continue

            if nbr_curr_nodes + len(self.__queue) + 1 >= self.max_nbr_nodes:
                self._make_leaf(node)
                nbr_curr_nodes += 1
                if self.verbose:
                    print(datetime.datetime.now(), 'Exceeded maximum number of nodes.')
                continue

            if node_depth >= self.max_depth:
                self._make_leaf(node)
                nbr_curr_nodes += 1
                if self.verbose:
                    print(datetime.datetime.now(), 'Exceeded maximum depth')
                continue

            # if nbr_curr_nodes + len(self.__queue) + 1 >= self.max_nbr_nodes \
            #         or nbr_samples < self.min_samples_split \
            #         or node_depth >= self.max_depth:
            #     self._make_leaf(node)
            #     nbr_curr_nodes += 1
            #     continue

            if self.model_type == MODEL_TYPE_CLU:
                clf, is_oblique = self._make_unsupervised_split(idx_iter)
            else:
                clf, is_oblique = self._make_supervised_split(idx_iter)

            labels = clf.apply(self.__X[idx_iter])
            y_pred = clf.predict(self.__X[idx_iter])

            # se tengo questa stop condition qui per la classificazione vado a scartare
            # automaticamente tutti gli split che in entrambe le foglie non sono a maggioranza
            # quindi per un dataset molto sbilanciato non faccio nemmeno il 1o split
            # if len(np.unique(labels)) == 1 or len(np.unique(y_pred)) == 1:
            #     self._make_leaf(node)
            #     nbr_curr_nodes += 1
            #     continue

            if self.model_type == MODEL_TYPE_CLU:
                if len(np.unique(labels)) == 1 or len(np.unique(y_pred)) == 1:
                    self._make_leaf(node)
                    nbr_curr_nodes += 1
                    if self.verbose:
                        print(datetime.datetime.now(), 'Split useless in clustering.')
                    continue

            elif self.model_type == MODEL_TYPE_CLF or self.model_type == MODEL_TYPE_REG:
                if len(np.unique(labels)) == 1:
                    self._make_leaf(node)
                    nbr_curr_nodes += 1
                    if self.verbose:
                        print(datetime.datetime.now(), 'Split useless in classification or regression.')
                    continue

            if self.model_type == MODEL_TYPE_CLU:
                bic_parent = bic(self.__X[idx_iter], [0] * nbr_samples)
                bic_children = bic(self.__X[idx_iter], (np.array(labels) - 1).tolist())

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

            if self.__y is None:
                label_l = node_id + 1
                label_r = node_id + 2
            elif self.model_type == MODEL_TYPE_CLF:
                label_l = calculate_mode(self.__y[idx_all_l])
                label_r = calculate_mode(self.__y[idx_all_r])
                proba_l = calculate_proba(self.__y[idx_all_l], self.nbr_classes_)
                proba_r = calculate_proba(self.__y[idx_all_r], self.nbr_classes_)
            elif self.model_type == MODEL_TYPE_REG:
                label_l = np.mean(self.__y[idx_all_l])
                label_r = np.mean(self.__y[idx_all_r])
            else:
                if y_is_numerical:
                    label_l = np.mean(self.__y[idx_all_l])
                    label_r = np.mean(self.__y[idx_all_r])
                else:
                    label_l = calculate_mode(self.__y[idx_all_l])
                    label_r = calculate_mode(self.__y[idx_all_r])
                    proba_l = calculate_proba(self.__y[idx_all_l], self.nbr_classes_)
                    proba_r = calculate_proba(self.__y[idx_all_r], self.nbr_classes_)

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

            self.__tree_structure[node.node_id] = (node_l.node_id, node_r.node_id)
            self.__node_dict[node_l.node_id] = node_l
            self.__node_dict[node_r.node_id] = node_r

            if self.model_type == MODEL_TYPE_CLU:
                bic_l = bic(self.__X[idx_iter[idx_l]], [0] * len(idx_l))
                bic_r = bic(self.__X[idx_iter[idx_r]], [0] * len(idx_r))
                heapq.heappush(self.__queue, (-len(idx_all_l) + 0.00001 * bic_l,
                                              (next(tiebreaker), idx_all_l, node_depth + 1, node_l)))
                heapq.heappush(self.__queue, (-len(idx_all_r) + 0.00001 * bic_r,
                                              (next(tiebreaker), idx_all_r, node_depth + 1, node_r)))
            else:
                self.__queue.append((idx_all_l, node_depth + 1, node_l))
                self.__queue.append((idx_all_r, node_depth + 1, node_r))

        if self.verbose:
            print(datetime.datetime.now(), 'Training ended in %s iterations, with %s leaves.' % (iter_id, nbr_curr_nodes))

        if (self.model_type == MODEL_TYPE_CLF or self.model_type == MODEL_TYPE_REG) and self.prune_useless_leaves:
            if self.verbose:
                print(datetime.datetime.now(), 'Pruning of useless leaves.')
            self._prune_useless_leaves()

        self.root_ = root_node
        self.label_encoder_ = LabelEncoder()

        if self.model_type == MODEL_TYPE_CLF or self.clu_for_clf:
            if self.verbose:
                print(datetime.datetime.now(), 'Normalize labels id.')
            self.labels_ = self.label_encoder_.fit_transform(self.labels_)

        if self.class_encoder_ is not None:
            self.__y = self.class_encoder_.inverse_transform(self.__y)
        self.rules_to_tree_print_ = self._get_rules_to_print_tree()
        self.rules_ = self._calculate_rules()
        self.rules_ = self._compact_rules()
        self.rules_s_ = self._rules2str()
        self.medoid_dict_ = self._calculate_medoids()
        if self.model_type == MODEL_TYPE_CLF or self.clu_for_clf:
            self.class_medoid_dict_ = self._calculate_class_medoids()
        elif self.model_type == MODEL_TYPE_REG or self.clu_for_reg:
            self.regr_medoid_dict_ = self._calculate_regr_medoids()

        if self.verbose:
            print(datetime.datetime.now(), 'RULE TREE - END.\n')

    def _preprocessing(self, X):
        return preprocessing(X, self.feature_names_r, self.is_cat_feat, self.data_encoder, self.numerical_scaler)
        # if self.data_encoder is not None:
        #     df = pd.DataFrame(data=X, columns=self.feature_names_r)
        #     X = self.data_encoder.transform(df).values
        #
        # if self.numerical_scaler is not None and np.sum(~self.is_cat_feat) > 0:
        #     X[:, ~self.is_cat_feat] = self.numerical_scaler.transform(X[:, ~self.is_cat_feat])
        #
        # return X

    def _inverse_preprocessing(self, X):
        return inverse_preprocessing(X, self.is_cat_feat, self.data_encoder, self.numerical_scaler)
        # if self.numerical_scaler is not None and np.sum(~self.is_cat_feat) > 0:
        #     X[:, ~self.is_cat_feat] = self.numerical_scaler.inverse_transform(X[:, ~self.is_cat_feat])
        #
        # if self.data_encoder is not None:
        #     X = self.data_encoder.inverse_transform(X).values
        #
        # return X

    def predict(self, X, get_leaf=False, get_rule=False):

        # if self.data_encoder is not None:
        #     df = pd.DataFrame(data=X, columns=self.feature_names_r)
        #     X = self.data_encoder.transform(df).values
        #
        # if self.numerical_scaler is not None and np.sum(~self.is_cat_feat) > 0:
        #     X[:, ~self.is_cat_feat] = self.numerical_scaler.fit_transform(X[:, ~self.is_cat_feat])

        X = self._preprocessing(X)

        idx = np.arange(X.shape[0])
        labels, leaves, proba = self._predict(X, idx, self.root_)

        # if self.numerical_scaler is not None and np.sum(~self.is_cat_feat) > 0:
        #     X[:, ~self.is_cat_feat] = self.numerical_scaler.inverse_transform(X[:, ~self.is_cat_feat])
        #
        # if self.data_encoder is not None:
        #     X = self.data_encoder.inverse_transform(X).values

        X = self._inverse_preprocessing(X)

        if self.model_type == MODEL_TYPE_CLF or self.clu_for_clf:
            labels = self.label_encoder_.transform(labels)
            if self.class_encoder_ is not None:
                labels = self.class_encoder_.inverse_transform(labels)

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

    def predict_proba(self, X):
        X = self._preprocessing(X)

        idx = np.arange(X.shape[0])
        _, _, proba = self._predict(X, idx, self.root_)

        X = self._inverse_preprocessing(X)

        return proba

    def _predict(self, X, idx, node):
        idx_iter = idx

        if node.is_leaf:
            nbr_records = len(idx_iter)
            labels = np.array([node.label] * nbr_records)
            leaves = np.array([node.node_id] * nbr_records).astype(int)
            proba = np.repeat(np.reshape(node.proba, (1,-1)), nbr_records, axis=0)

            if self.model_type == MODEL_TYPE_REG or self.clu_for_reg:
                labels = labels.astype(float)

            return labels, leaves, proba

        else:

            clf = node.clf
            labels = clf.apply(X[idx_iter])
            if self.model_type == MODEL_TYPE_REG or self.clu_for_reg:
                labels = labels.astype(float)
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

    def get_axes2d(self, eps=1, X=None):
        idx = np.arange(self.__X.shape[0])

        if X is None:
            X = self.__X

        return self._get_axes2d(idx, self.root_, eps, X)

    def _get_axes2d(self, idx, node: RuleTreeNode, eps, X):
        idx_iter = idx

        axes2d = list()

        if node.is_leaf:
            return []

        else:
            clf = node.clf
            labels = clf.apply(self.__X[idx_iter])

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
        idx = np.arange(self.__X.shape[0])
        return self.__get_rules_to_print_tree(idx, self.root_, 0)

    def __inverse_scale_thr(self, feat, thr, cat, oblique):
        if self.numerical_scaler is None or cat or oblique:
            return thr
        fake_x = np.zeros(self.__X.shape[1])
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
            for node_id in self.__tree_structure:
                node_l, node_r = self.__tree_structure[node_id]
                if self.__node_dict[node_l].is_leaf and self.__node_dict[node_r].is_leaf and \
                        self.__node_dict[node_l].label == self.__node_dict[node_r].label:
                    self._make_leaf(self.__node_dict[node_id])
                    del self.__node_dict[node_l]
                    del self.__node_dict[node_r]
                    self.__node_dict[node_id].node_l = None
                    self.__node_dict[node_id].node_r = None
                    nodes_to_remove.append(node_id)
                    tree_pruned = True

            if not tree_pruned:
                break

            for node_id in nodes_to_remove:
                del self.__tree_structure[node_id]

    def __get_rules_to_print_tree(self, idx_iter, node: RuleTreeNode, cur_depth):
        rules = list()

        if node.is_leaf:
            label = node.label
            if self.model_type == MODEL_TYPE_CLF or self.clu_for_clf:
                label = self.label_encoder_.transform([node.label])[0]
                if self.class_encoder_ is not None:
                    label = self.class_encoder_.inverse_transform([label])[0]
            leaf = (False, label, node.samples, node.support, node.node_id, cur_depth)

            rules.append(leaf)
            return rules

        else:
            clf = node.clf
            labels = clf.apply(self.__X[idx_iter])

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

    def _calculate_rules(self):
        if self.verbose:
            print(datetime.datetime.now(), 'Calculate rules.')

        for node_id in self.__node_dict:
            node = self.__node_dict[node_id]
            if not node.is_leaf:
                continue

            label = node.label
            if self.model_type == MODEL_TYPE_CLF or self.clu_for_clf:
                label = self.label_encoder_.transform([node.label])[0]

            rule = [(label,)]

            past_node_id = node_id
            next_node_id = node.parent_id
            if next_node_id == -1:
                break
            next_node = self.__node_dict[next_node_id]

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
                self.__leaf_rule[node_id] = rule

                if next_node_id == -1:
                    break
                next_node = self.__node_dict[next_node_id]

        return self.__leaf_rule

    def get_rules(self, as_text=False):
        if as_text:
            return self.rules_s_

        return self.rules_

    def _calculate_medoids(self):
        if self.verbose:
            print(datetime.datetime.now(), 'Calculate medoids.')

        medoid_dict = dict()
        for node_id in self.__node_dict:
            node = self.__node_dict[node_id]
            node.medoid = calculate_medoid(self.__X[node.idx], self.__Xr[node.idx])
            medoid_dict[node_id] = node.medoid
        return medoid_dict

    def get_medoids(self):
        return self.medoid_dict_

    def _calculate_class_medoids(self):
        if self.verbose:
            print(datetime.datetime.now(), 'Calculate class medoids.')

        class_medoid_dict = dict()
        for node_id in self.__node_dict:
            node = self.__node_dict[node_id]
            node.class_medoid = dict()
            for l in np.unique(self.__y[node.idx]):
                node.class_medoid[l] = calculate_medoid(self.__X[node.idx][self.__y[node.idx] == l],
                                                        self.__Xr[node.idx][self.__y[node.idx] == l])
            class_medoid_dict[node_id] = node.class_medoid
        return class_medoid_dict

    def get_class_medoid(self):
        return self.class_medoid_dict_

    def _calculate_regr_medoids(self):
        if self.verbose:
            print(datetime.datetime.now(), 'Calculate regression medoids.')

        regr_medoid_dict = dict()
        for node_id in self.__node_dict:
            node = self.__node_dict[node_id]
            median_y = np.median(self.__y[node.idx])
            diff_with_median_y = self.__y[node.idx] - median_y
            idx_regr_medoid = np.argmin(diff_with_median_y)
            node.regr_medoid = self.__Xr[idx_regr_medoid]
            regr_medoid_dict[node_id] = node.regr_medoid
        return regr_medoid_dict

    def get_regr_medoids(self):
        return self.regr_medoid_dict_

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

    def _rules2str(self):
        if self.verbose:
            print(datetime.datetime.now(), 'Turn rules into strings.')

        self.rules_s_ = dict()

        for leaf_id in self.rules_:
            rule = self.rules_[leaf_id]
            self.rules_s_[leaf_id] = list()
            for cond in rule:
                if len(cond) == 1:
                    # cons_txt = np.round(cond[0], self.precision
                    #                     ) if self.model_type == MODEL_TYPE_REG or self.clu_for_reg else cond[0]
                    # cond_s = ('%s' % cons_txt)

                    if self.model_type == MODEL_TYPE_REG or self.clu_for_reg:
                        cons_txt = np.round(cond[0], self.precision)
                    else:
                        if self.class_encoder_ is not None:
                            cons_txt = self.class_encoder_.inverse_transform([cond[0]])[0]
                        else:
                            cons_txt = cond[0]
                    cond_s = ('%s' % cons_txt)

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

        # if self.data_encoder is not None:
        #     df = pd.DataFrame(data=X, columns=self.feature_names_r)
        #     X = self.data_encoder.transform(df).values
        #
        # if self.numerical_scaler is not None and np.sum(~self.is_cat_feat):
        #     X[:, ~self.is_cat_feat] = self.numerical_scaler.fit_transform(X[:, ~self.is_cat_feat])

        X = self._preprocessing(X)

        verified_list = list()
        for leaf, rule in self.rules_.items():
            verified_list.append(self._is_verified_r_(X, rule, count_cond, relative_count))

        # if self.numerical_scaler is not None and np.sum(~self.is_cat_feat) > 0:
        #     X[:, ~self.is_cat_feat] = self.numerical_scaler.inverse_transform(X[:, ~self.is_cat_feat])
        #
        # if self.data_encoder is not None:
        #     X = self.data_encoder.inverse_transform(X).values

        X = self._inverse_preprocessing(X)

        verified_list = np.array(verified_list).T
        return verified_list

    def transform(self, X, metric='euclidean',
                  include_descriptive_pivot=True,
                  include_target_pivot=False,
                  only_leaf_pivot=False,
                  ):

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

                if self.model_type == MODEL_TYPE_CLF or self.clu_for_clf:
                    for k in self.rules_:
                        for l in self.class_medoid_dict_[k]:
                            P.append(self.class_medoid_dict_[k][l])

                if self.model_type == MODEL_TYPE_REG or self.clu_for_reg:
                    for k in self.rules_:
                        P.append(self.regr_medoid_dict_[k])
            else:
                if self.model_type == MODEL_TYPE_CLF or self.clu_for_clf:
                    for k in self.class_medoid_dict_:
                        for l in self.class_medoid_dict_[k]:
                            P.append(self.class_medoid_dict_[k][l])

                if self.model_type == MODEL_TYPE_REG or self.clu_for_reg:
                    for k in self.regr_medoid_dict_:
                        P.append(self.regr_medoid_dict_[k])

        if len(P) == 0:
            return None

        P = np.array(P)

        X = self._preprocessing(X)
        P = self._preprocessing(P)

        dist = cdist(X.astype(float), P.astype(float), metric=metric)
        X = self._inverse_preprocessing(X)

        return dist


from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import export_text
from sklearn.metrics import accuracy_score, normalized_mutual_info_score, mean_absolute_percentage_error


def main():
    iris = load_iris()
    X = iris.data
    y = iris.target

    # df = pd.read_csv('../datasets/classification/iris.csv', skipinitialspace=True)
    # features = [c for c in df.columns if c != 'class']
    # X = df[features].values
    # y = df['class']

    # idx = [100,101,102,103,104,105,106,107,108,109,110,111,112,113,114,115,116,117,118,
    #        51,52,53,54,55,56,57,58,59,60,61,62,63]
    # X = X[idx][:,[0,1]]
    # y = y[idx]
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.05, random_state=0)
    # print(np.unique(y_train, return_counts=True))

    max_nbr_values_cat = 4

    # df = pd.DataFrame(data=[
    # #     # ['F', 2, 'Milk', 'Y'],
    # #     # ['M', 2, 'Milk', 'N'],
    # #     # ['F', 2, 'Milk', 'Y'],
    # #     # ['F', 3, 'Dark', 'Y'],
    # #     # ['M', 8, 'Milk', 'N'],
    # #     # ['M', 3, 'No', 'Y'],
    # #     # ['M', 10, 'Dark', 'N'],
    # #     # ['F', 2, 'No', 'N'],
    # #     # ['M', 1, 'No', 'Y']],
    #     ['F', 10, 'Milk', 1],
    #     ['M', 5, 'Dark', 0],
    #     ['F', 2, 'Milk', 1],
    #     ['F', 3, 'Dark', 1],
    #     ['M', 8, 'Milk', 0],
    #     ['M', 3, 'No', 1],
    #     ['M', 10, 'Dark', 0],
    #     ['F', 2, 'No', 0],
    #     ['M', 1, 'No', 1]
    # ],
    #     columns=['Sex', 'Lies', 'Cookies', 'Present']
    # )
    # X_train = df[['Sex', 'Lies', 'Cookies']].values
    # X_test = df[['Sex', 'Lies', 'Cookies']].values
    # y_train = df['Present'].values
    # y_test = df['Present'].values

    from sklearn.preprocessing import MinMaxScaler

    rt = RuleTree(model_type='clf',
                  # max_depth=4,
                  # min_samples_leaf=1,
                  # min_samples_split=2,
                  max_nbr_nodes=7,
                  # clf_impurity='entropy',
                  # feature_names=['Sex', 'Lies', 'Cookies'],
                  # feature_names=iris.feature_names,
                  # allow_oblique_splits=True,
                  # force_oblique_splits=True,
                  # max_nbr_values=4,
                  # max_nbr_nodes=32,
                  prune_useless_leaves=True,
                  # max_nbr_nodes=5,
                  bic_eps=0.5,
                  # max_nbr_values=20,
                  max_nbr_values_cat=max_nbr_values_cat,
                  one_hot_encode_cat=True,
                  clus_impurity='r2',
                  # categorical_indices=[4],
                  # numerical_indices=[1],
                  numerical_scaler=StandardScaler(),
                  # exclude_split_feature_from_reduction=True,
                  random_state=0,
                  verbose=True
                  )

    y_train_o = y_train[::]
    y_test_o = y_test[::]
    # y_train = np.log(X_train[:, 0] * X_train[:, 1])
    # y_test = np.log(X_test[:, 0] * X_test[:, 1])
    # X_train = pd.concat([
    #     pd.DataFrame(data=X_train[:,:]),
    #     pd.DataFrame(data=y_train_o.reshape(-1, 1).astype(str))],
    #     axis=1).values
    # X_test = pd.concat([
    #     pd.DataFrame(data=X_test[:, :]),
    #     pd.DataFrame(data=y_test_o.reshape(-1, 1).astype(str))],
    #     axis=1).values
    print(rt.bic_eps)
    rt.set_params(**{'bic_eps': 0.25})
    print(rt.bic_eps)
    rt.fit(X_train, y_train)
    # rt.fit(X_train)
    print(rt.predict_proba(X_test))
    print(rt.predict(X_test))

    # print(rt.feature_values[4])

    # print(rt.root_.node_r.node_r.__dict__)
    # print(y_train[rt.root_.node_r.node_r.idx], st.mode(y_train[rt.root_.node_r.node_r.idx]))
    print('RuleTree')
    print(rt.print_tree())
    print('')

    # for r in rt.rules_s_:
    #     print(rt.rules_s_[r])
    # print('')

    y_pred, leaves, rules = rt.predict(X_test, get_leaf=True, get_rule=True)
    print(y_pred, '<<<<<<<')
    print(y_test)
    print(leaves)
    print(rt.rules_s_)
    print(rt.predict_proba(X_test), 'PPPPPPP')
    # print(X_test[2])

    if len(np.unique(y_train)) >= max_nbr_values_cat:  # infer is numerical
        print('R2', r2_score(y_test, y_pred))
        print('MAPE', mean_absolute_percentage_error(y_test, y_pred))
    else:  # infer it is categorical
        print('Accuracy', accuracy_score(y_test, y_pred))
    print('NMI', normalized_mutual_info_score(y_test, y_pred))
    print('')
    for r in rules:
        print(r)
    print('')

    print(rt.are_rules_verified(X_test, count_cond=False, relative_count=False))
    print('')

    for k in rt.medoid_dict_:
        print(k, rt.medoid_dict_[k])

    print('')

    for k in rt.rules_s_:
        print(k, rt.rules_s_[k], rt.medoid_dict_[k])

    print('')

    if len(np.unique(y_train)) >= max_nbr_values_cat:  # infer is numerical
        if rt.regr_medoid_dict_ is not None:
            for k in rt.regr_medoid_dict_:
                print(k, rt.regr_medoid_dict_[k])

            print('')

            for k in rt.rules_s_:
                print(k, rt.rules_s_[k], rt.regr_medoid_dict_[k])

            print('')
    else:
        if rt.class_medoid_dict_ is not None:
            for k in rt.class_medoid_dict_:
                print(k, rt.class_medoid_dict_[k])

            print('')

            for k in rt.rules_s_:
                print(k, rt.rules_s_[k], rt.class_medoid_dict_[k])

            print('')

    dist = rt.transform(X_train, metric='euclidean', include_descriptive_pivot=True, include_target_pivot=True, only_leaf_pivot=True)
    dist_test = rt.transform(X_test, metric='euclidean', include_descriptive_pivot=True, include_target_pivot=True,
                             only_leaf_pivot=True)
    # print(dist)
    # print(dist.shape)
    # print(y_test)

    rtp = RuleTree(model_type='clf', max_depth=4)
    rtp.fit(dist, y_train)
    print(rtp.print_tree())

    print('Accuracy', accuracy_score(y_test, rtp.predict(dist_test)))

    # print(X_test[2][[1,2]])
    # print(rt.numerical_scaler.transform(X_test)[2][[1,2]])
    # # print(rt.numerical_scaler.mean_)
    # # print(rt.numerical_scaler.scale_)
    # # print(rt.rules_[1])
    # # print(rt.rules_[1][0][0], rt.rules_[1][0][5])
    #
    # for r in rt.rules_s_:
    #     print(r, rt.rules_s_[r])

    # print('')
    # print(rt.numerical_scaler.transform(X_test)[2][rt.rules_[1][0][0]], 'record normalized')
    # print(rt.rules_[1][0][5], 'coef originali')
    # print(rt.numerical_scaler.transform(X_test)[2][rt.rules_[1][0][0]] * rt.rules_[1][0][5], 'record norm * coef originali')
    # print(np.sum(rt.numerical_scaler.transform(X_test)[2][rt.rules_[1][0][0]] * rt.rules_[1][0][5]), 'sum')

    # print('')
    # print(X_test[2][rt.rules_[1][0][0]], 'record originale')
    # fake_x = np.zeros(X_test.shape[1])
    # fake_x[rt.rules_[1][0][0]] = rt.rules_[1][0][5]
    # X_to_invert = np.array([fake_x])
    # X_to_invert = rt.numerical_scaler.transform(X_to_invert)[0, rt.rules_[1][0][0]]
    # print(X_to_invert, 'coef normalized')
    # print(X_test[2][rt.rules_[1][0][0]] * X_to_invert, 'record originale * coef normalized')
    # print(np.sum(X_test[2][rt.rules_[1][0][0]] * X_to_invert), 'sum')

    # TODO fare benchmarking
    # add medoide discriminativo??? direidi no
    # TODO dopo benchmakring sfruttare in notebook clustering per class 2
    #  per a) clustering iniziale e b) modello predittivo su coppie di cluster

    # dataset classification: iris, adult, compas, ionosphere, titanic, german, diabetes, home, wine, bank
    # dataset regression:
    # parkinson: https://archive.ics.uci.edu/dataset/189/parkinsons+telemonitoring
    # abalone: https://archive.ics.uci.edu/dataset/1/abalone
    # wine
    # auction verification: https://archive.ics.uci.edu/dataset/713/auction+verification
    # intrusion: https://archive.ics.uci.edu/dataset/715/lt+fs+id+intrusion+detection+in+wsns
    # metamaterial: https://archive.ics.uci.edu/dataset/692/2d+elastodynamic+metamaterials
    # liver disorder: https://archive.ics.uci.edu/dataset/60/liver+disorders
    # boston: https://www.kaggle.com/datasets/fedesoriano/the-boston-houseprice-data
    # carprice: https://www.kaggle.com/datasets/hellbuoy/car-price-prediction
    # medical cost: https://www.kaggle.com/datasets/mirichoi0218/insurance
    # student performance: https://www.kaggle.com/datasets/nikhil7280/student-performance-multiple-linear-regression
    # dataset clustering puri, quelli di partree
    # dataset clustering per classifcazione e regeresisone, quelli sopra

    # competior: clf e reg, dt sklearn e knn
    # clustering: kmeans, kmeans+dt
    # pivot: random, kmeans/kmedoids, kmeans/kmedoids su classi + knn o dt


if __name__ == "__main__":
    main()
