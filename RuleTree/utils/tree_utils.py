"""The similarity and cover functions of XGBoost (regression)."""

import numpy as np



def xgboost_similarity_score_regression(residuals:np.ndarray, cover:int, _lambda:int=0) -> float:
    """
    XGBoost similarity score of a node, for regression.

    Parameters
    ----------
    residuals : numpy.ndarray
        Residuals of the instances in the node.
    cover : int
        Cover of the node (see xgboost_cover_regression).
    _lambda : float, default=0
        L2 regularization.

    Returns
    -------
    similarity : float
        ``sum(residuals) ** 2 / (cover + lambda)``.
    """
    return np.square(np.sum(residuals))/(_lambda+cover)

def xgboost_cover_regression(residuals:np.ndarray) -> float:
    """
    XGBoost cover of a node, for regression: the number of instances.

    Parameters
    ----------
    residuals : numpy.ndarray
        Residuals of the instances in the node.

    Returns
    -------
    cover : int
    """
    return len(residuals)
