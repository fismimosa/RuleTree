import json
import math

import numpy as np
import pandas as pd
import category_encoders as ce
from sklearn.base import ClassifierMixin, RegressorMixin
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor


def _iterative_mean(iter, current_mean, x):
    """
    Iteratively calculates mean using
    http://www.heikohoffmann.de/htmlthesis/node134.html
    :param iter: non-negative integer, iteration
    :param current_mean: numpy array, current value of mean
    :param x: numpy array, new value to be added to mean
    :return: numpy array, updated mean
    """
    return current_mean + ((x - current_mean) / (iter + 1))


def preprocessing(X, feature_names_r, is_cat_feat, data_encoder=None, numerical_scaler=None):
    X = np.copy(X)
    if data_encoder is not None:
        df = pd.DataFrame(data=X, columns=feature_names_r)
        X = data_encoder.transform(df).values

    if numerical_scaler is not None and np.sum(~is_cat_feat) > 0:
        X[:, ~is_cat_feat] = numerical_scaler.transform(X[:, ~is_cat_feat])

    return X


def inverse_preprocessing(X, is_cat_feat, data_encoder=None, numerical_scaler=None):
    X = np.copy(X)
    if numerical_scaler is not None and np.sum(~is_cat_feat) > 0:
        X[:, ~is_cat_feat] = numerical_scaler.inverse_transform(X[:, ~is_cat_feat])

    if data_encoder is not None:
        X = data_encoder.inverse_transform(X).values

    return X


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
                X[:, feature] = X[:, feature].astype(float)

    feature_values = dict()
    is_categorical_feature = np.full_like(np.zeros(n_features, dtype=bool), False)

    if categorical_indices is None:
        for feature in range(n_features):
            values = np.unique(X[:, feature])
            vals = None
            if len(values) > max_nbr_values:  # this reduces the number of values for continuous attributes
                _, vals = np.histogram(values, bins=max_nbr_values)
                values = [(vals[i] + vals[i + 1]) / 2 for i in range(len(vals) - 1)]
            feature_values[feature] = values

            if len(values) <= max_nbr_values_cat:  # this identifies categorical attributes
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
            if len(values) > max_nbr_values:  # this reduces the number of values for continuous attributes
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
        df = encoder.fit_transform(
            pd.DataFrame(data=X, columns=feature_names_original))  #TODO: serve passare da pandas??
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

def calculate_mode(x: np.ndarray):
    vals, counts = np.unique(x, return_counts=True)
    idx = np.argmax(counts)
    return vals[idx]

def get_info_gain(clf: DecisionTreeClassifier | DecisionTreeRegressor):
    if len(clf.tree_.impurity) == 1:#no_split
        return 0 # TODO: check
    imp_parent, imp_child_l, imp_child_r = clf.tree_.impurity
    n_parent, n_child_l, n_child_r = clf.tree_.weighted_n_node_samples  ##n_node_samples 
    return _get_info_gain(imp_parent, imp_child_l, imp_child_r, n_parent, n_child_l, n_child_r)

def _get_info_gain(imp_parent, imp_child_l, imp_child_r, n_parent, n_child_l, n_child_r):
    gain_split = imp_parent - imp_child_l * (n_child_l / n_parent) - imp_child_r * (n_child_r / n_parent)
    return gain_split

def get_gain_ratio(clf: DecisionTreeClassifier | DecisionTreeRegressor):
    if len(clf.tree_.impurity) == 1:#no_split
        return 0 # TODO: check
    imp_parent, imp_child_l, imp_child_r = clf.tree_.impurity
    n_parent, n_child_l, n_child_r = clf.tree_.weighted_n_node_samples  #n_node_samples
    gain_split = imp_parent - imp_child_l * (n_child_l / n_parent) - imp_child_r * (n_child_r / n_parent)
    split_info = (n_child_l / n_parent)*math.log2(n_child_l / n_parent) +\
                 (n_child_r / n_parent)*math.log2(n_child_r / n_parent)
    split_info *= -1

    return gain_split/split_info

def _my_counts(x, sample_weight:np.ndarray=None, class_weight:dict=None):
    if sample_weight is None and class_weight is None:
        _, counts = np.unique(x, return_counts=True)

    else:
        sample_weight = np.ones(x.shape[0]) if sample_weight is None else sample_weight
        class_weight = dict([(y_, 1.) for y_ in np.unique(x)]) if class_weight is None else class_weight

        counts = np.zeros((len(class_weight), ))
        for i, (c_k, c_w) in enumerate(class_weight.items()):
            counts[i] = np.sum(sample_weight[x == c_k]) * c_w

    return counts

def gini(x, sample_weight:np.ndarray=None, class_weight:dict=None):
    counts = _my_counts(x, sample_weight=sample_weight, class_weight=class_weight)

    total = sum(counts)

    return 1 - np.sum((counts/total)**2)

def entropy(x, sample_weight:np.ndarray=None, class_weight:dict=None):
    counts = _my_counts(x, sample_weight=sample_weight, class_weight=class_weight)

    p_j = counts/sum(counts)
    p_j_log = np.log2(p_j)

    return - p_j @ p_j_log

def select_stumps(node, p=0.2, selected_stumps=None):
    if selected_stumps is None:
        selected_stumps = []

    if node.stump is not None and node.balance_score > p:
        selected_stumps.append(node.stump)
        
    if node.node_l is not None:
        select_stumps(node.node_l, p, selected_stumps)
        select_stumps(node.node_r, p, selected_stumps)
    
    return selected_stumps


class json_NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(json_NumpyEncoder, self).default(obj)
