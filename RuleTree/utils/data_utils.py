"""
Utility functions for data preprocessing, transformation, and metrics calculation
for the RuleTree library. This module contains helper functions for handling 
categorical and numerical features, calculating information gain metrics, and 
performing data encoding/decoding operations.
"""

import json
import math
from typing import Union

import numpy as np
import pandas as pd
import category_encoders as ce
from sklearn.base import ClassifierMixin, RegressorMixin
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor


def _iterative_mean(iter, current_mean, x):
    """
    Iteratively calculates mean using Welford's online algorithm.
    
    This method allows for efficient computation of the mean by updating it
    iteratively as new data points arrive, without storing all previous values.
    
    Parameters
    ----------
    iter : int
        Non-negative integer representing the current iteration
    current_mean : numpy.ndarray
        Current value of the mean
    x : numpy.ndarray
        New value to be added to the mean
        
    Returns
    -------
    numpy.ndarray
        Updated mean value
        
    References
    ----------
    http://www.heikohoffmann.de/htmlthesis/node134.html
    """
    return current_mean + ((x - current_mean) / (iter + 1))


def preprocessing(X, feature_names_r, is_cat_feat, data_encoder=None, numerical_scaler=None):
    """
    Preprocess data by applying encoding and scaling transformations.
    
    Parameters
    ----------
    X : numpy.ndarray
        Input data matrix to be preprocessed
    feature_names_r : list
        Names of features in the original feature space
    is_cat_feat : numpy.ndarray
        Boolean array indicating which features are categorical
    data_encoder : category_encoders object, optional
        Encoder for transforming categorical features
    numerical_scaler : sklearn.preprocessing scaler, optional
        Scaler for normalizing numerical features
        
    Returns
    -------
    numpy.ndarray
        Preprocessed data
    """
    X = np.copy(X)
    if data_encoder is not None:
        df = pd.DataFrame(data=X, columns=feature_names_r)
        X = data_encoder.transform(df).values

    if numerical_scaler is not None and np.sum(~is_cat_feat) > 0:
        X[:, ~is_cat_feat] = numerical_scaler.transform(X[:, ~is_cat_feat])

    return X


def inverse_preprocessing(X, is_cat_feat, data_encoder=None, numerical_scaler=None):
    """
    Inverse the preprocessing operation to restore original data format.
    
    Parameters
    ----------
    X : numpy.ndarray
        Input data matrix in preprocessed form
    is_cat_feat : numpy.ndarray
        Boolean array indicating which features are categorical
    data_encoder : category_encoders object, optional
        Encoder used in the preprocessing step
    numerical_scaler : sklearn.preprocessing scaler, optional
        Scaler used in the preprocessing step
        
    Returns
    -------
    numpy.ndarray
        Data in original feature space
    """
    X = np.copy(X)
    if numerical_scaler is not None and np.sum(~is_cat_feat) > 0:
        X[:, ~is_cat_feat] = numerical_scaler.inverse_transform(X[:, ~is_cat_feat])

    if data_encoder is not None:
        X = data_encoder.inverse_transform(X).values

    return X


def prepare_data(X_original, max_nbr_values, max_nbr_values_cat, feature_names_original, one_hot_encode_cat,
                 categorical_indices, numerical_indices, numerical_scaler):
    """
    Prepare data for RuleTree by handling categorical features and applying preprocessing.
    
    This function identifies categorical features, optionally one-hot encodes them,
    scales numerical features, and creates mapping dictionaries for feature transformation.
    
    Parameters
    ----------
    X_original : numpy.ndarray
        Original input data matrix
    max_nbr_values : int
        Maximum number of values for binning continuous features
    max_nbr_values_cat : int
        Maximum number of unique values for a feature to be considered categorical
    feature_names_original : list
        Original feature names
    one_hot_encode_cat : bool
        Whether to one-hot encode categorical features
    categorical_indices : list, optional
        Indices of features known to be categorical
    numerical_indices : list, optional
        Indices of features known to be numerical
    numerical_scaler : sklearn.preprocessing scaler, optional
        Scaler to apply to numerical features
        
    Returns
    -------
    tuple
        (X, features, maps)
        - X: preprocessed data matrix
        - features: tuple containing feature information
        - maps: dictionary with mappings between original and one-hot encoded features
    
    Raises
    ------
    Exception
        If provided categorical and numerical indices don't match the dataset dimensions
    """
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

    feature_values = {}
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
        map_original_onehot = {}
        map_onehot_original = {}
        map_original_onehot_idx = {}
        map_onehot_original_idx = {}
        for i, c1 in enumerate(feature_names_original):
            map_original_onehot[c1] = []
            map_original_onehot_idx[i] = []
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

        feature_values = {}
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
    """
    Calculate the mode (most frequent value) of an array.
    
    Parameters
    ----------
    x : numpy.ndarray
        Input array
        
    Returns
    -------
    object
        Most frequent value in the input array
    """
    vals, counts = np.unique(x, return_counts=True)
    idx = np.argmax(counts)
    return vals[idx]

def get_info_gain(clf: Union[DecisionTreeClassifier, DecisionTreeRegressor]):
    """
    Calculate information gain from a decision tree split.
    
    Parameters
    ----------
    clf : DecisionTreeClassifier or DecisionTreeRegressor
        Fitted decision tree classifier or regressor
        
    Returns
    -------
    float
        Information gain achieved by the split
    """
    try:
        if len(clf.tree_.impurity) == 1:#no_split
            return 0
    except Exception:
        pass
    imp_parent, imp_child_l, imp_child_r = clf.tree_.impurity
    n_parent, n_child_l, n_child_r = clf.tree_.weighted_n_node_samples  ##n_node_samples 
    return _get_info_gain(imp_parent, imp_child_l, imp_child_r, n_parent, n_child_l, n_child_r)

def _get_info_gain(imp_parent, imp_child_l, imp_child_r, n_parent, n_child_l, n_child_r):
    """
    Helper function to calculate information gain from impurity values.
    
    Parameters
    ----------
    imp_parent : float
        Impurity of the parent node
    imp_child_l : float
        Impurity of the left child node
    imp_child_r : float
        Impurity of the right child node
    n_parent : int
        Number of samples in the parent node
    n_child_l : int
        Number of samples in the left child node
    n_child_r : int
        Number of samples in the right child node
        
    Returns
    -------
    float
        Information gain achieved by the split
    """
    gain_split = imp_parent - imp_child_l * (n_child_l / n_parent) - imp_child_r * (n_child_r / n_parent)
    return gain_split

def get_gain_ratio(clf: Union[DecisionTreeClassifier, DecisionTreeRegressor]):
    """
    Calculate the gain ratio from a decision tree split.
    
    Gain ratio is information gain divided by the split information,
    which normalizes the information gain to account for the number of splits.
    
    Parameters
    ----------
    clf : DecisionTreeClassifier or DecisionTreeRegressor
        Fitted decision tree classifier or regressor
        
    Returns
    -------
    float
        Gain ratio achieved by the split
    """
    if len(clf.tree_.impurity) == 1:#no_split
        return 0
    imp_parent, imp_child_l, imp_child_r = clf.tree_.impurity
    n_parent, n_child_l, n_child_r = clf.tree_.weighted_n_node_samples  #n_node_samples
    gain_split = imp_parent - imp_child_l * (n_child_l / n_parent) - imp_child_r * (n_child_r / n_parent)
    split_info = (n_child_l / n_parent)*math.log2(n_child_l / n_parent) +\
                 (n_child_r / n_parent)*math.log2(n_child_r / n_parent)
    split_info *= -1

    return gain_split/split_info

def _my_counts(x, sample_weight:np.ndarray=None, class_weight:dict=None):
    """
    Calculate weighted counts for each unique value in an array.
    
    Parameters
    ----------
    x : numpy.ndarray
        Input array
    sample_weight : numpy.ndarray, optional
        Sample weights to apply
    class_weight : dict, optional
        Class weights to apply
        
    Returns
    -------
    numpy.ndarray
        Weighted counts for each unique value
    """
    if sample_weight is None and class_weight is None:
        _, counts = np.unique(x, return_counts=True)

    else:
        sample_weight = np.ones(x.shape[0]) if sample_weight is None else sample_weight
        class_weight = dict.fromkeys(np.unique(x), 1) if class_weight is None else class_weight

        counts = np.zeros((len(class_weight), ))
        for i, (c_k, c_w) in enumerate(class_weight.items()):
            counts[i] = np.sum(sample_weight[x == c_k]) * c_w

    return counts

def gini(x, sample_weight:np.ndarray=None, class_weight:dict=None):
    """
    Calculate the Gini impurity of an array.
    
    Gini impurity measures the probability of incorrectly classifying a randomly chosen
    element if it were randomly labeled according to the distribution of classes.
    
    Parameters
    ----------
    x : numpy.ndarray
        Input array
    sample_weight : numpy.ndarray, optional
        Sample weights to apply
    class_weight : dict, optional
        Class weights to apply
        
    Returns
    -------
    float
        Gini impurity value
    """
    counts = _my_counts(x, sample_weight=sample_weight, class_weight=class_weight)

    total = sum(counts)

    return 1 - np.sum((counts/total)**2)

def entropy(x, sample_weight:np.ndarray=None, class_weight:dict=None):
    """
    Calculate the entropy of an array.
    
    Entropy measures the average level of "information" or "uncertainty" inherent in
    the possible outcomes of a random variable.
    
    Parameters
    ----------
    x : numpy.ndarray
        Input array
    sample_weight : numpy.ndarray, optional
        Sample weights to apply
    class_weight : dict, optional
        Class weights to apply
        
    Returns
    -------
    float
        Entropy value
    """
    counts = _my_counts(x, sample_weight=sample_weight, class_weight=class_weight)

    p_j = counts/sum(counts)
    p_j_log = np.log2(p_j)

    return - p_j @ p_j_log

def select_stumps(node, p=0.2, selected_stumps=None):
    """
    Select decision stumps from a tree based on balance score threshold.
    
    This function recursively traverses the tree and collects stumps that have
    a balance score above the specified threshold.
    
    Parameters
    ----------
    node : Node
        Current node in the tree
    p : float, default=0.2
        Balance score threshold
    selected_stumps : list, optional
        Accumulator for selected stumps during recursion
        
    Returns
    -------
    list
        Selected stumps that meet the balance score threshold
    """
    if selected_stumps is None:
        selected_stumps = []

    if node.stump is not None and node.balance_score > p:
        selected_stumps.append(node.stump)
        
    if node.node_l is not None:
        select_stumps(node.node_l, p, selected_stumps)
        select_stumps(node.node_r, p, selected_stumps)
    
    return selected_stumps


class json_numpy_encoder(json.JSONEncoder):
    """
    JSON encoder that handles NumPy data types.
    
    This class extends the standard JSON encoder to properly serialize
    NumPy integers, floats, and arrays to their Python equivalents.
    
    Methods
    -------
    default(obj)
        Override method to handle NumPy types
    """
    def default(self, obj):
        """
        Convert NumPy types to standard Python types for JSON serialization.
        
        Parameters
        ----------
        obj : object
            Object to be serialized
            
        Returns
        -------
        object
            Python equivalent of NumPy type
            
        Raises
        ------
        TypeError
            If object is not a NumPy type and cannot be serialized by the parent class
        """
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(json_numpy_encoder, self).default(obj)
