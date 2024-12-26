from RuleTree.stumps.classification.DecisionTreeStumpClassifier import DecisionTreeStumpClassifier
from RuleTree.stumps.classification.ObliqueDecisionTreeStumpClassifier import ObliqueDecisionTreeStumpClassifier
from RuleTree.stumps.classification.PivotTreeStumpClassifier import PivotTreeStumpClassifier
from RuleTree.stumps.classification.MultiplePivotTreeStumpClassifier import MultiplePivotTreeStumpClassifier
from RuleTree.stumps.classification.MultipleObliquePivotTreeStumpClassifier import MultipleObliquePivotTreeStumpClassifier
#from RuleTree.stumps.classification.ProximityTreeStumpClassifier import ProximityTreeStumpClassifier

from RuleTree.stumps.classification.ObliquePivotTreeStumpClassifier import ObliquePivotTreeStumpClassifier

from RuleTree.stumps.regression.DecisionTreeStumpRegressor import DecisionTreeStumpRegressor


def dt_stump_reg_call(random_state = 42):
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
    Creates and returns an instance of ObliquePivotTreeStumpClassifier.
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


