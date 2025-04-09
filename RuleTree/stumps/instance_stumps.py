from RuleTree.stumps.classification.DecisionTreeStumpClassifier import DecisionTreeStumpClassifier
from RuleTree.stumps.classification.ObliqueDecisionTreeStumpClassifier import ObliqueDecisionTreeStumpClassifier
from RuleTree.stumps.classification.PivotTreeStumpClassifier import PivotTreeStumpClassifier
from RuleTree.stumps.classification.MultiplePivotTreeStumpClassifier import MultiplePivotTreeStumpClassifier
from RuleTree.stumps.classification.MultipleObliquePivotTreeStumpClassifier import MultipleObliquePivotTreeStumpClassifier
#from RuleTree.stumps.classification.ProximityTreeStumpClassifier import ProximityTreeStumpClassifier

from RuleTree.stumps.classification.ObliquePivotTreeStumpClassifier import ObliquePivotTreeStumpClassifier

from RuleTree.stumps.regression.DecisionTreeStumpRegressor import DecisionTreeStumpRegressor


def dt_stump_reg_call(random_state = 42):
    """
    Creates and returns an instance of DecisionTreeStumpRegressor.
    
    This function initializes a decision tree stump regressor with default parameters
    optimized for regression tasks.
    
    Parameters
    ----------
    random_state : int, default=42
        Random seed for reproducibility.
        
    Returns
    -------
    DecisionTreeStumpRegressor
        A configured decision tree stump regressor instance.
    """
    dt_stump = DecisionTreeStumpRegressor(
                        max_depth=1,
                        criterion='squared_error',
                        splitter='best',
                        min_samples_split=2,
                        min_samples_leaf = 1,
                        min_weight_fraction_leaf=0.0,
                        max_features=None,
                        random_state=random_state,
                        min_impurity_decrease=0.0,                    
                        ccp_alpha=0.0,
                        monotonic_cst = None)
    return dt_stump


def dt_stump_call(random_state = 42):
    """
    Creates and returns an instance of DecisionTreeStumpClassifier.
    
    This function initializes a decision tree stump classifier with default parameters
    optimized for classification tasks.
    
    Parameters
    ----------
    random_state : int, default=42
        Random seed for reproducibility.
        
    Returns
    -------
    DecisionTreeStumpClassifier
        A configured decision tree stump classifier instance.
    """
    dt_stump = DecisionTreeStumpClassifier(
                        max_depth=1,
                        criterion='gini',
                        splitter='best',
                        min_samples_split=2,
                        min_samples_leaf = 1,
                        min_weight_fraction_leaf=0.0,
                        max_features=None,
                        random_state=random_state,
                        min_impurity_decrease=0.0,
                        class_weight=None,
                        ccp_alpha=0.0,
                        monotonic_cst = None)
    return dt_stump


def obl_stump_call(
    random_state=42,
    oblique_split_type='householder',
    pca=None,
    max_oblique_features=2,
    tau=1e-4,
    n_orientations=10
):
    """
    Creates and returns an instance of ObliqueDecisionTreeStumpClassifier.
    
    This function initializes an oblique decision tree stump classifier that uses
    oblique splits instead of axis-aligned splits.
    
    Parameters
    ----------
    random_state : int, default=42
        Random seed for reproducibility.
    oblique_split_type : str, default='householder'
        Type of oblique split to use. Options include 'householder', etc.
    pca : bool or None, default=None
        Whether to use PCA for dimensionality reduction.
    max_oblique_features : int, default=2
        Maximum number of features to consider in oblique splits.
    tau : float, default=1e-4
        Tolerance parameter.
    n_orientations : int, default=10
        Number of orientations to consider for splits.
        
    Returns
    -------
    ObliqueDecisionTreeStumpClassifier
        A configured oblique decision tree stump classifier instance.
    """
    obl_stump = ObliqueDecisionTreeStumpClassifier(
        max_depth=1,
        criterion='gini',
        splitter='best',
        min_samples_split=2,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0.0,
        max_features=None,
        random_state=random_state,
        min_impurity_decrease=0.0,
        class_weight=None,
        ccp_alpha=0.0,
        monotonic_cst=None,
        oblique_split_type=oblique_split_type,
        pca=pca,
        max_oblique_features=max_oblique_features,
        tau=tau,
        n_orientations=n_orientations
    )
    return obl_stump


def obl_pt_stump_call(
    random_state=42,
    oblique_split_type='householder',
    pca=None,
    max_oblique_features=2,
    tau=1e-4,
    n_orientations=10
):
    """
    Creates and returns an instance of ObliquePivotTreeStumpClassifier.
    
    This function initializes an oblique pivot tree stump classifier that combines
    pivot-based splitting with oblique splits.
    
    Parameters
    ----------
    random_state : int, default=42
        Random seed for reproducibility.
    oblique_split_type : str, default='householder'
        Type of oblique split to use.
    pca : bool or None, default=None
        Whether to use PCA for dimensionality reduction.
    max_oblique_features : int, default=2
        Maximum number of features to consider in oblique splits.
    tau : float, default=1e-4
        Tolerance parameter.
    n_orientations : int, default=10
        Number of orientations to consider for splits.
        
    Returns
    -------
    ObliquePivotTreeStumpClassifier
        A configured oblique pivot tree stump classifier instance.
    """
    obl_pt_stump = ObliquePivotTreeStumpClassifier(
        max_depth=1,
        criterion='gini',
        splitter='best',
        min_samples_split=2,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0.0,
        max_features=None,
        random_state=random_state,
        min_impurity_decrease=0.0,
        class_weight=None,
        ccp_alpha=0.0,
        monotonic_cst=None,
        oblique_split_type=oblique_split_type,
        pca=pca,
        max_oblique_features=max_oblique_features,
        tau=tau,
        n_orientations=n_orientations
    )
    return obl_pt_stump

def multi_obl_pt_stump_call(
    random_state=42,
    oblique_split_type='householder',
    pca=None,
    max_oblique_features=2,
    tau=1e-4,
    n_orientations=10
):
    """
    Creates and returns an instance of MultipleObliquePivotTreeStumpClassifier.
    
    This function initializes a multiple oblique pivot tree stump classifier that
    considers multiple pivot points with oblique splits for improved classification.
    
    Parameters
    ----------
    random_state : int, default=42
        Random seed for reproducibility.
    oblique_split_type : str, default='householder'
        Type of oblique split to use.
    pca : bool or None, default=None
        Whether to use PCA for dimensionality reduction.
    max_oblique_features : int, default=2
        Maximum number of features to consider in oblique splits.
    tau : float, default=1e-4
        Tolerance parameter.
    n_orientations : int, default=10
        Number of orientations to consider for splits.
        
    Returns
    -------
    MultipleObliquePivotTreeStumpClassifier
        A configured multiple oblique pivot tree stump classifier instance.
    """
    multi_obl_pt_stump = MultipleObliquePivotTreeStumpClassifier(
        max_depth=1,
        criterion='gini',
        splitter='best',
        min_samples_split=2,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0.0,
        max_features=None,
        random_state=random_state,
        min_impurity_decrease=0.0,
        class_weight=None,
        ccp_alpha=0.0,
        monotonic_cst=None,
        oblique_split_type=oblique_split_type,
        pca=pca,
        max_oblique_features=max_oblique_features,
        tau=tau,
        n_orientations=n_orientations
    )
    return multi_obl_pt_stump




def pt_stump_call(random_state = 42):
    """
    Creates and returns an instance of PivotTreeStumpClassifier.
    
    This function initializes a pivot tree stump classifier which uses pivot points
    for splitting rather than traditional axis-aligned splits.
    
    Parameters
    ----------
    random_state : int, default=42
        Random seed for reproducibility.
        
    Returns
    -------
    PivotTreeStumpClassifier
        A configured pivot tree stump classifier instance.
    """
    pt_stump = PivotTreeStumpClassifier(
                        max_depth=1,
                        criterion='gini',
                        splitter='best',
                        min_samples_split=2,
                        min_samples_leaf = 1,
                        min_weight_fraction_leaf=0.0,
                        max_features=None,
                        random_state=random_state,
                        min_impurity_decrease=0.0,
                        class_weight=None,
                        ccp_alpha=0.0,
                        monotonic_cst = None,
                        )
    return pt_stump


def multi_pt_stump_call(random_state = 42):
    """
    Creates and returns an instance of MultiplePivotTreeStumpClassifier.
    
    This function initializes a multiple pivot tree stump classifier that considers
    multiple pivot points for improved classification accuracy.
    
    Parameters
    ----------
    random_state : int, default=42
        Random seed for reproducibility.
        
    Returns
    -------
    MultiplePivotTreeStumpClassifier
        A configured multiple pivot tree stump classifier instance.
    """
    multi_pt_stump = MultiplePivotTreeStumpClassifier(
                        max_depth=1,
                        criterion='gini',
                        splitter='best',
                        min_samples_split=2,
                        min_samples_leaf = 1,
                        min_weight_fraction_leaf=0.0,
                        max_features=None,
                        random_state=random_state,
                        min_impurity_decrease=0.0,
                        class_weight=None,
                        ccp_alpha=0.0,
                        monotonic_cst = None)
    return multi_pt_stump

