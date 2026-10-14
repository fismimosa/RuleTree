"""This module contains a custom one-hot encoder. It inherits from sklearn's
OneHotEncoder and returns a pandas.SparseDataFrame with appropriate column
names and index values.
"""


import numpy as np
import pandas as pd
from sklearn import preprocessing


class OneHotEncoder(preprocessing.OneHotEncoder):

    """One-hot encoder returning a DataFrame with a column per category."""
    def __init__(self):
        """Initialize the encoder."""
        super().__init__(sparse_output=True, dtype=np.uint8,handle_unknown='ignore')

    def fit(self, X, y=None):

        """
        Fit the encoder.

        Parameters
        ----------
        X : pandas.DataFrame
        y : None
            Not used.

        Returns
        -------
        self : OneHotEncoder
        """
        self = super().fit(X)
        self.column_names_ = self.get_feature_names_out(X.columns if hasattr( X,'columns') else None)

        return self

    def transform(self, X):
        """
        One-hot encoding of X.

        Parameters
        ----------
        X : pandas.DataFrame

        Returns
        -------
        X_encoded : pandas.DataFrame
        """
        return pd.DataFrame.sparse.from_spmatrix(
            data=super().transform(X),
            columns=self.column_names_,
            index=X.index if isinstance(X, pd.DataFrame) else None,
        )