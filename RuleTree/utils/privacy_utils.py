import numpy as np
from scipy.stats import wasserstein_distance
from sklearn.preprocessing import KBinsDiscretizer


def _earth_mover_distance(a:np.ndarray, b_idx:np.ndarray, is_categorical:bool):
    """
    Calculate the Earth Mover's Distance between two distributions.
    
    The Earth Mover's Distance (also known as Wasserstein distance) measures
    the minimum "work" required to transform one distribution into another.
    
    Parameters
    ----------
    a : np.ndarray
        Array containing all data points
    b_idx : np.ndarray
        Boolean mask indicating which points in 'a' belong to second distribution
    is_categorical : bool
        Whether the data is categorical or continuous
        
    Returns
    -------
    float
        The Earth Mover's Distance between distributions
    """
    b = a[b_idx]
    len_a = len(a)
    len_b = len(b)

    if is_categorical:
        unique_values, a_binned = np.unique(a, return_counts=True)
        a_binned = a_binned.astype(np.float64)
        a_binned /= len_a
        b_binned = np.asarray([np.sum(b == x) for x in unique_values])/len_b
    else:
        bins = KBinsDiscretizer(n_bins=max(3, int(len_a**.5)), strategy='uniform')
        a_binned = np.sum(bins.fit_transform(a.reshape(-1, 1)), axis=0)[0]/len_a
        b_binned = np.sum(bins.transform(b.reshape(-1, 1)), axis=0)[0]/len_b

    bin_list = np.arange(0, a_binned.shape[0], 1)
    d=wasserstein_distance(bin_list, bin_list, a_binned, b_binned)# / (len_a-1)
    return d

def compute_k_anonimity(X, X_bool, sensible_attribute):
    """
    Compute k-anonymity values for left and right partitions of data.
    
    K-anonymity ensures that each group contains at least k individuals with the same 
    sensitive attribute value, making it difficult to identify a specific individual.
    
    Parameters
    ----------
    X : np.ndarray
        Feature matrix
    X_bool : np.ndarray
        Boolean array indicating the split (True for left partition, False for right)
    sensible_attribute : int
        Index of the sensitive attribute column
        
    Returns
    -------
    tuple
        (k_left, k_right) representing the k-anonymity values for both partitions
    """
    X_bool = X_bool.copy().reshape(-1)
    k_left = min(np.unique(X[X_bool, sensible_attribute], return_counts=True)[1])
    k_right = min(np.unique(X[~X_bool, sensible_attribute], return_counts=True)[1])

    return k_left, k_right

def compute_l_diversity(X, X_bool, sensible_attribute):
    """
    Compute l-diversity values for left and right partitions of data.
    
    L-diversity ensures that sensitive attribute values are well-represented in each group,
    making it difficult to infer an individual's sensitive attribute.
    
    Parameters
    ----------
    X : np.ndarray
        Feature matrix
    X_bool : np.ndarray
        Boolean array indicating the split (True for left partition, False for right)
    sensible_attribute : int
        Index of the sensitive attribute column
        
    Returns
    -------
    tuple
        (l_left, l_right) representing the l-diversity values for both partitions
    """
    X_bool = X_bool.copy().reshape(-1)
    l_left = len(np.unique(X[X_bool, sensible_attribute])) - 1
    l_right = len(np.unique(X[~X_bool, sensible_attribute])) - 1

    return l_left, l_right

def _compute_t_closeness(X, sensible_attribute, categorical):
    """
    Compute t-closeness for a given dataset partition.
    
    T-closeness measures the distance between the distribution of a sensitive attribute 
    in a group and the distribution of the attribute in the entire dataset.
    
    Parameters
    ----------
    X : np.ndarray
        Feature matrix for a partition
    sensible_attribute : int
        Index of the sensitive attribute column
    categorical : list or set
        Indices of categorical features
        
    Returns
    -------
    float
        The minimum t-closeness value across all features
    """
    t = np.inf
    for prot_attr_value in np.unique(X[:, sensible_attribute]):
        for feat_idx in range(X.shape[1]):
            if feat_idx == sensible_attribute:
                continue

            t_curr = _earth_mover_distance(X[:, feat_idx],
                                           X[:, sensible_attribute] == prot_attr_value,
                                           feat_idx in categorical)
            if t_curr < t:
                t = t_curr

    return t

def compute_t_closeness(X, X_bool, sensible_attribute, categorical):
    """
    Compute t-closeness values for left and right partitions of data.
    
    Parameters
    ----------
    X : np.ndarray
        Feature matrix
    X_bool : np.ndarray
        Boolean array indicating the split (True for left partition, False for right)
    sensible_attribute : int
        Index of the sensitive attribute column
    categorical : list or set
        Indices of categorical features
        
    Returns
    -------
    tuple
        (t_left, t_right) representing the t-closeness values for both partitions
    """
    X_bool = X_bool.copy().reshape(-1)

    t_left = _compute_t_closeness(X[X_bool], sensible_attribute, categorical)
    t_right = _compute_t_closeness(X[~X_bool], sensible_attribute, categorical)

    return t_left, t_right
