"""
Matrix to vector distance calculations for shapelet transforms.

This module provides optimized functions for computing various distance metrics
between a matrix of vectors and a single vector. These functions are critical for
efficient shapelet distance calculations in the RuleTree framework.
"""

import numpy as np
from numba import jit
from scipy.spatial import distance


@jit
def euclidean(matrix, vector):
    """
    Calculate the Euclidean distance between each row in a matrix and a vector.
    
    Parameters
    ----------
    matrix : numpy.ndarray
        A 2D numpy array where each row is a vector
    vector : numpy.ndarray
        A 1D numpy array to compute distances against
        
    Returns
    -------
    numpy.ndarray
        A 1D array containing the Euclidean distance between each row of the matrix and the vector
    """
    return np.sqrt(np.sum(np.power(matrix - vector, 2), axis=1))


@jit
def sqeuclidean(matrix, vector):
    """
    Calculate the squared Euclidean distance between each row in a matrix and a vector.
    
    Parameters
    ----------
    matrix : numpy.ndarray
        A 2D numpy array where each row is a vector
    vector : numpy.ndarray
        A 1D numpy array to compute distances against
        
    Returns
    -------
    numpy.ndarray
        A 1D array containing the squared Euclidean distance between each row of the matrix and the vector
    """
    return np.sum(np.power(matrix - vector, 2), axis=1)


@jit
def cityblock(matrix, vector):
    """
    Calculate the Manhattan (cityblock) distance between each row in a matrix and a vector.
    
    Parameters
    ----------
    matrix : numpy.ndarray
        A 2D numpy array where each row is a vector
    vector : numpy.ndarray
        A 1D numpy array to compute distances against
        
    Returns
    -------
    numpy.ndarray
        A 1D array containing the Manhattan distance between each row of the matrix and the vector
    """
    return np.sum(np.abs(matrix - vector), axis=1)


@jit(nopython=True)
def cosine(matrix, vector):
    """
    Calculate the cosine distance between each row in a matrix and a vector.
    
    Note: This function handles potential numerical issues by returning 0 for any
    NaN values that might occur during calculation.
    
    Parameters
    ----------
    matrix : numpy.ndarray
        A 2D numpy array where each row is a vector
    vector : numpy.ndarray
        A 1D numpy array to compute distances against
        
    Returns
    -------
    numpy.ndarray
        A 1D array containing the cosine distance between each row of the matrix and the vector
    """
    res = np.zeros(matrix.shape[0])

    for i in matrix.shape[0]:
        try:
            res[i] = 1 - np.dot(matrix[i], vector) / np.linalg.norm(matrix[i]) * np.linalg.norm(vector)
        except:
            res[i] = np.nan
            #warnings.warn("Er")

    return np.nan_to_num(res)


if __name__ == "__main__":
    matrix = np.array([
        [0, 1, 0],
        [0, 0, 0],
    ], dtype=np.float64)

    vector = np.array([0, -1, 0], dtype=np.float64)

    print("Euclidean:", euclidean(matrix, vector), [distance.euclidean(el, vector) for el in matrix])
    print("SqEuclidean:", sqeuclidean(matrix, vector), [distance.sqeuclidean(el, vector) for el in matrix])
    print("Cosine:", cosine(matrix, vector), [distance.cosine(el, vector) for el in matrix])
    print("CityBlock:", cityblock(matrix, vector), [distance.cityblock(el, vector) for el in matrix])
