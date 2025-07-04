from RuleTree.tree.RuleTree import RuleTree
import copy
from RuleTree.encoding.dict_utils import make_complete_rule_tree
import numpy as np


def random_forest_to_list_of_dicts(rf):
    list_of_dicts = list()
    for i, estimator in enumerate(rf.estimators_):
        list_of_dicts.append(estimator.to_dict(None))
    return list_of_dicts


def list_of_dicts_to_random_forest(list_of_dicts, original_rf):
    # original_rf_list_of_dicts = random_forest_to_list_of_dicts(original_rf)
    rf_ = copy.deepcopy(original_rf)
    rf_.estimators_ = list()
    for i, d in enumerate(list_of_dicts):
        # # this is needed to ensure that the args are not lost
        # if d.get("args") is None:
        #     d["args"] = original_rf_list_of_dicts[i]['args']
        # if d.get("classes_") is None:
        #     d["classes_"] = original_rf_list_of_dicts[i]['classes_']
        # if d.get("n_classes_") is None:
        #     d["n_classes_"] = original_rf_list_of_dicts[i]['n_classes_']
        rf_.estimators_.append(RuleTree.from_dict(d))
    return rf_


def complete_forest(rf):
    depth = np.max([estimator.max_depth for estimator in rf.estimators_])
    list_of_dicts = random_forest_to_list_of_dicts(rf)
    complete_list = list()
    for estimator_dict in list_of_dicts:
        complete_list.append(make_complete_rule_tree(estimator_dict, max_depth=depth))
    return list_of_dicts_to_random_forest(complete_list, rf)


def simplify_forest(rf):
    rf_ = copy.deepcopy(rf)
    for i, estimator in enumerate(rf_.estimators_):
        rf_.estimators_[i].root = estimator.root.simplify()
    return rf_


