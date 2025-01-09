import numpy as np
import sklearn.metrics as skm
from sklearn import tree
from sklearn.tree import _tree

from RuleTree import RuleTreeClassifier, RuleTreeRegressor
from RuleTree.tree.RuleTreeNode import RuleTreeNode


def evaluate_clf(y_test, y_pred, y_pred_proba):
    class_values = np.unique(y_test)
    binary = len(class_values) <= 2
    res = {
        'accuracy': skm.accuracy_score(y_test, y_pred),
        'balanced_accuracy': skm.balanced_accuracy_score(y_test, y_pred),
        'f1_score': skm.f1_score(y_test, y_pred, average='binary', pos_label=class_values[1])
        if binary else skm.f1_score(y_test, y_pred, average='micro'),
        'f1_micro': skm.f1_score(y_test, y_pred, average='micro'),
        'f1_macro': skm.f1_score(y_test, y_pred, average='macro'),
        'f1_weighted': skm.f1_score(y_test, y_pred, average='weighted'),
        'precision_score': skm.precision_score(y_test, y_pred, average='binary', pos_label=class_values[1])
        if binary else skm.precision_score(y_test, y_pred, average='micro'),
        'precision_micro': skm.precision_score(y_test, y_pred, average='micro'),
        'precision_macro': skm.precision_score(y_test, y_pred, average='macro'),
        'precision_weighted': skm.precision_score(y_test, y_pred, average='weighted'),
        'recall_score': skm.recall_score(y_test, y_pred, average='binary', pos_label=class_values[1])
        if binary else skm.recall_score(y_test, y_pred, average='micro'),
        'recall_micro': skm.recall_score(y_test, y_pred, average='micro'),
        'recall_macro': skm.recall_score(y_test, y_pred, average='macro'),
        'recall_weighted': skm.recall_score(y_test, y_pred, average='weighted'),

    }

    if y_pred_proba is not None:
        res.update(
            {
                'roc_macro': skm.roc_auc_score(y_test, y_pred_proba[:, 1],
                                               average='macro') if binary else skm.roc_auc_score(
                    y_test, y_pred_proba, average='macro', multi_class='ovr'),
                'roc_micro': skm.roc_auc_score(y_test, y_pred_proba[:, 1],
                                               average='micro') if binary else skm.roc_auc_score(
                    y_test, y_pred_proba, average='micro', multi_class='ovr'),
                'roc_weighted': skm.roc_auc_score(y_test, y_pred_proba[:, 1], average='weighted')
                if binary else skm.roc_auc_score(y_test, y_pred_proba, average='weighted', multi_class='ovr'),
                'average_precision_macro': skm.average_precision_score(y_test, y_pred_proba[:, 1], average='macro',
                                                                       pos_label=class_values[1])
                if binary else skm.average_precision_score(y_test, y_pred_proba, average='macro'),
                'average_precision_micro': skm.average_precision_score(y_test, y_pred_proba[:, 1], average='micro',
                                                                       pos_label=class_values[1])
                if binary else skm.average_precision_score(y_test, y_pred_proba, average='micro'),
                'average_precision_weighted': skm.average_precision_score(y_test, y_pred_proba[:, 1],
                                                                          average='weighted',
                                                                          pos_label=class_values[1])
                if binary else skm.average_precision_score(y_test, y_pred_proba, average='weighted'),
            }
        )

    return res


def evaluate_reg(y_test, y_pred):
    res = {
        'explained_variance': skm.explained_variance_score(y_test, y_pred),
        'max_error': skm.max_error(y_test, y_pred),
        'mean_absolute_error': skm.mean_absolute_error(y_test, y_pred),
        'mean_squared_error': skm.mean_squared_error(y_test, y_pred),
        'mean_squared_log_error': skm.mean_squared_log_error(y_test, y_pred),
        'median_absolute_error': skm.median_absolute_error(y_test, y_pred),
        'r2': skm.r2_score(y_test, y_pred),
        'mean_absolute_percentage_error': skm.mean_absolute_percentage_error(y_test, y_pred),
    }
    return res


def evaluate_clu_sup(y_test, y_pred):
    res = {
        'adjusted_mutual_info': skm.adjusted_mutual_info_score(y_test, y_pred),
        'adjusted_rand': skm.adjusted_rand_score(y_test, y_pred),
        'completeness': skm.completeness_score(y_test, y_pred),
        'fowlkes_mallows': skm.fowlkes_mallows_score(y_test, y_pred),
        'homogeneity': skm.homogeneity_score(y_test, y_pred),
        'mutual_info': skm.mutual_info_score(y_test, y_pred),
        'normalized_mutual_info': skm.normalized_mutual_info_score(y_test, y_pred),
        'rand_score': skm.rand_score(y_test, y_pred),
        'v_measure': skm.v_measure_score(y_test, y_pred)
    }
    return res


def evaluate_expl(model):
    if model.__class__ == RuleTreeClassifier or model.__class__ == RuleTreeRegressor:
        res = _evaluate_expl_ruletree(model.root)
    elif issubclass(model.__class__, tree.BaseDecisionTree):
        res = _evaluate_expl_decisiontree(model)
    else:
        return {}

    return {
        "n_leaf": res[0],
        "n_nodes": res[1],
        "max_depth": res[2],
    }


def _evaluate_expl_ruletree(root_node: RuleTreeNode):
    if root_node.is_leaf():
        return 1, 0, 1

    (n_leaf_l, n_nodes_l, max_depth_l) = _evaluate_expl_ruletree(root_node.node_l)

    (n_leaf_r, n_nodes_r, max_depth_r) = _evaluate_expl_ruletree(root_node.node_r)

    return (n_leaf_l + n_leaf_r,
            n_nodes_l + n_nodes_r + 1,
            max_depth_l + 1 if max_depth_l > max_depth_r else max_depth_r + 1)


def _evaluate_expl_ruletree_depth(tree_, node):
    if tree_.feature[node] != _tree.TREE_UNDEFINED:
        depth_l = _evaluate_expl_ruletree_depth(tree_, tree_.children_left[node])
        depth_r = _evaluate_expl_ruletree_depth(tree_, tree_.children_right[node])
        return depth_l + 1 if depth_l > depth_r else depth_r + 1
    else:
        return 1


def _evaluate_expl_decisiontree(clf: tree.BaseDecisionTree):
    return clf.tree_.n_leaves, clf.tree_.node_count - clf.tree_.max_depth, _evaluate_expl_ruletree_depth(clf.tree_, 0)


def evaluate_clu_unsup(y_pred, X, dist):
    if len(np.unique(y_pred)) == 1:
        return {
            'silhouette_score': np.nan,
            'calinski_harabasz': np.nan,
            'davies_bouldin': np.nan
        }

    return {
        'silhouette_score': skm.silhouette_score(dist, y_pred),
        'calinski_harabasz': skm.calinski_harabasz_score(X, y_pred),
        'davies_bouldin': skm.davies_bouldin_score(X, y_pred)
    }
