"""Helpers on data and targets: mode, log-odds, impurities, information gain, JSON encoding of NumPy values."""

import json
from typing import Union

import numpy as np
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor


def calculate_mode(x: np.ndarray) -> Union[int, float, str]:
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

def log_odds(class_counts: np.ndarray, n_samples: int) -> np.ndarray:
    """
    Compute the log-odds log(n_c / (n - n_c)) of each class.

    Pure cases are handled explicitly instead of dividing by zero: a class with no samples
    gets -inf, a class with all the samples gets +inf, and an empty node gets nan.

    Parameters
    ----------
    class_counts : numpy.ndarray
        Number of samples of each class
    n_samples : int
        Total number of samples

    Returns
    -------
    numpy.ndarray
        Log-odds of each class
    """
    class_counts = np.asarray(class_counts, dtype=float)
    other_counts = n_samples - class_counts

    result = np.full(class_counts.shape, np.nan)
    result[(class_counts == 0) & (other_counts > 0)] = -np.inf
    result[(class_counts > 0) & (other_counts == 0)] = np.inf
    both = (class_counts > 0) & (other_counts > 0)
    result[both] = np.log(class_counts[both] / other_counts[both])

    return result

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
    p_j = p_j[p_j > 0]  # 0 * log(0) = 0: classes without instances (e.g. with class_weight)
    p_j_log = np.log2(p_j)

    return - p_j @ p_j_log

class json_numpy_encoder(json.JSONEncoder):
    """
    JSON encoder that handles NumPy data types.

    This class extends the standard JSON encoder to properly serialize
    NumPy integers, floats, and arrays to their Python equivalents.

    Examples
    --------
    >>> import json
    >>> import numpy as np
    >>> from RuleTree.utils.data_utils import json_numpy_encoder
    >>> json.dumps({"a": np.arange(3), "b": np.float32(0.5)}, cls=json_numpy_encoder)
    '{"a": [0, 1, 2], "b": 0.5}'
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


def check_one_target(y, owner: str):
    """
    Check that a target has a single column, for the components that do not support several targets.

    Parameters
    ----------
    y : numpy.ndarray or None
        Target.
    owner : str
        What needs a single target, used in the error message.

    Raises
    ------
    ValueError
        If y has shape (n_samples, n_targets) with more than one target.

    Examples
    --------
    >>> import numpy as np
    >>> from RuleTree.utils.data_utils import check_one_target
    >>> check_one_target(np.zeros((3, 2)), "XGBoostSplitter")
    Traceback (most recent call last):
    ...
    ValueError: XGBoostSplitter supports a single target, got y of shape (3, 2).
    """
    if y is not None and np.ndim(y) > 1 and np.shape(y)[1] > 1:
        raise ValueError(f"{owner} supports a single target, got y of shape {np.shape(y)}.")
