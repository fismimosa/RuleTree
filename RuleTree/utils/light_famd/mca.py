"""Multiple Correspondence Analysis (MCA)"""

import numpy as np
from sklearn import utils

from . import ca
from . import one_hot


class MCA(ca.CA):

    """
    Multiple Correspondence Analysis of categorical data: a CA of their one-hot encoding. Used by
    RuleTreeCluster for categorical data.
    """
    def fit(self, X, y=None):
        """
        Fit the MCA.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Categorical data.
        y : None
            Not used.

        Returns
        -------
        self : MCA
        """
        if self.check_input:
            utils.check_array(X, dtype=[str, np.number])
            
        n_initial_columns = X.shape[1]

        # One-hot encode the data
        self.one_hot_ = one_hot.OneHotEncoder().fit(X)
        
        _X_t=  self.one_hot_.transform(X) 
        
        _0_freq_serie= (_X_t == 0).sum(axis=0)/ len(_X_t)
        
        self._usecols=_0_freq_serie[_0_freq_serie < 0.99].index
        #print('MCA PROCESS ELIMINATED {0}  COLUMNS SINCE THEIR MISS_RATES >= 99%'.format( _X_t.shape[1] - len(self._usecols) ))
        
        n_new_columns = len(self._usecols)
        self.total_inertia_ = (n_new_columns - n_initial_columns) / n_initial_columns
        # Apply CA to the indicator matrix
        super().fit(_X_t.loc[:,self._usecols])

        return self

    def _transform(self, X):
        """
        Row principal coordinates of one-hot encoded data.

        Parameters
        ----------
        X : array-like

        Returns
        -------
        coordinates : numpy.ndarray of shape (n_samples, n_components)
        """
        return super()._transform(self.one_hot_.transform(X).loc[:,self._usecols])



    def transform(self, X):
        """Computes the row principal coordinates of a dataset."""
        utils.validation.check_is_fitted(self, 'singular_values_')
        if self.check_input:
            utils.check_array(X, dtype=[str, np.number])
        return self._transform(X)

