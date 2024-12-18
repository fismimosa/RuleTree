from ruletree.stumps.classification.DecisionTreeStumpClassifier import DecisionTreeStumpClassifier
from ruletree.stumps.classification.ObliqueDecisionTreeStumpClassifier import ObliqueDecisionTreeStumpClassifier
from ruletree.stumps.classification.PivotTreeStumpClassifier import PivotTreeStumpClassifier
from ruletree.stumps.classification.MultiplePivotTreeStumpClassifier import MultiplePivotTreeStumpClassifier
#from ruletree.stumps.classification.ProximityTreeStumpClassifier import ProximityTreeStumpClassifier


from ruletree.stumps.regression.DecisionTreeStumpRegressor import DecisionTreeStumpRegressor


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


def obl_stump_call(random_state = 42):
    obl_stump = ObliqueDecisionTreeStumpClassifier(
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
    return obl_stump


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


