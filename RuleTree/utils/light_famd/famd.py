"""Factor Analysis of Mixed Data (FAMD)"""
import numpy as np
import pandas as pd

from . import mfa

# dtypes of the categorical columns: since pandas 3 the strings have their own dtype, 'str', which
# select_dtypes still includes in 'object' only with a deprecation warning (and pandas 2 rejects 'str')
CATEGORICAL_DTYPES = ['object', 'category'] + (['str'] if int(pd.__version__.split('.')[0]) >= 3 else [])


class FAMD(mfa.MFA):

    """
    Factor Analysis of Mixed Data: an MFA with a group of numerical features and a group of
    categorical features. Used by RuleTreeCluster for mixed data.

    Parameters
    ----------
    n_components, n_iter, copy, check_input, random_state, engine
        See PCA.
    """
    def __init__(self, n_components=2, n_iter=2, copy=False, check_input=True, random_state=None,
                 engine='auto'):
        """Initialize the FAMD; see the class documentation for the parameters."""
        super().__init__(
            groups=None,
            normalize=True,
            n_components=n_components,
            n_iter=n_iter,
            copy=copy,
            check_input=check_input,
            random_state=random_state,
            engine=engine
        )

    def fit(self, X, y=None):

        """
        Fit the FAMD, grouping the columns into numerical and categorical.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        y : None
            Not used.

        Returns
        -------
        self : FAMD
        """
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)

        # Separate numerical columns from categorical columns
        num_cols = X.select_dtypes(np.number).columns
        cat_cols = X.select_dtypes(include=CATEGORICAL_DTYPES).columns

        # Make one group per variable type
        self.groups = {}
        if len(num_cols):
            self.groups['Numerical'] = num_cols
        else:
            raise ValueError('FAMD works with categorical and numerical data but you only have categorical data; you '
                             'should consider using MCA')
        if len(cat_cols):
            self.groups['Categorical'] = cat_cols
        else:
            raise ValueError('FAMD works with categorical and numerical data but you only have numerical data; you '
                             'should consider using PCA')

        return super().fit(X)
    
    def fit_transform(self,X,y=None):
        """
        Fit the FAMD and return the row principal coordinates.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        y : None
            Not used.

        Returns
        -------
        coordinates : numpy.ndarray of shape (n_samples, n_components)
        """
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X).infer_objects()
            X.columns = [str(x) for x in X.columns]

        # Separate numerical columns from categorical columns
        num_cols = X.select_dtypes(np.number).columns
        cat_cols = X.select_dtypes(include=CATEGORICAL_DTYPES).columns

        # Make one group per variable type
        self.groups = {}
        if len(num_cols):
            self.groups['Numerical'] = num_cols
        else:
            raise ValueError("FAMD works with categorical and numerical data but " +
                             "you only have categorical data; you should consider using MCA")
        if len(cat_cols):
            self.groups['Categorical'] = cat_cols
        else:
            raise ValueError("FAMD works with categorical and numerical data but " +
                             "you only have numerical data; you should consider using PCA")
        return super().fit_transform(X)
        

  







