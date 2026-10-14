"""Correlation helpers of light_famd."""

import numpy as np
import pandas as pd
from scipy.stats import pearsonr


def _pearsonr(A,B,threshold=0.05):
    """
    Pearson correlation between the columns of A and of B, set to nan where not significant.

    Parameters
    ----------
    A, B : numpy.ndarray
    threshold : float, default=0.05
        Significance level.

    Returns
    -------
    correlation : numpy.ndarray
    """
    co_,p_value=pearsonr(A,B)
    return co_ if p_value <=threshold else  np.nan
    
    
    
def  df_correlation(DF_C,DF_INDEX):
    """
    Correlation between the columns of two DataFrames.

    Parameters
    ----------
    DF_C : pandas.DataFrame
        Principal coordinates.
    DF_INDEX : pandas.DataFrame
        Original columns.

    Returns
    -------
    correlation : pandas.DataFrame
    """
    if not isinstance(DF_C,(pd.DataFrame,pd.SparseDataFrame)):
        DF_C=pd.DataFrame(DF_C)
    if not isinstance(DF_INDEX,(pd.DataFrame,pd.SparseDataFrame)):
        DF_INDEX=pd.DataFrame(DF_INDEX)		
    _data={}
    for col in DF_C:
        if isinstance(DF_C[col],pd.SparseSeries) :
            _c= DF_C[col].values.to_dense()
        else:
            _c= DF_C[col].values
        _embed_dict={}
        for index in  DF_INDEX:
            if isinstance(DF_INDEX[index],pd.SparseSeries) :
                _i= DF_INDEX[index].values.to_dense()
            else:
                _i= DF_INDEX[index].values
            _embed_dict[index]=_pearsonr(_c,_i)
        _data[col]=_embed_dict
            
    return pd.DataFrame(_data )    
