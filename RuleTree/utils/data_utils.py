"""Helpers on data and targets: mode, log-odds, check of the target, JSON encoding of NumPy values."""

import json
from typing import Union

import numpy as np


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
