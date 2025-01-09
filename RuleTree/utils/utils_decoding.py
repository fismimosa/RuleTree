import numpy as np
from sklearn.tree._tree import Tree


def configure_cat_split(clf, feature_index, threshold_value):
    clf.is_categorical = True
    clf.feature_original = [feature_index - 1, -2, -2]
    clf.threshold_original = np.array([threshold_value, -2, -2])
    return clf
    
def configure_non_cat_split(clf, vector, index, n_features_in_, n_classes_, n_outputs_):
    inner_tree = create_inner_tree(n_features_in_, n_classes_, n_outputs_, vector, index)
    assign_tree_properties(clf, inner_tree)
    return clf


def create_inner_tree(n_features_in_, n_classes_, n_outputs_, vector, index):
    n_classes_ = np.array([n_classes_], dtype=np.intp)
    inner_tree = Tree(n_features_in_, n_classes_, n_outputs_)
    state = inner_tree.__getstate__()
    
    node_count = 3  # Assuming max_depth 1 implies 3 nodes
    state['node_count'] = node_count
    state['nodes'] = np.zeros((node_count,), 
                              dtype=[('left_child', '<i8'),
                                     ('right_child', '<i8'), 
                                     ('feature', '<i8'), 
                                     ('threshold', '<f8'), 
                                     ('impurity', '<f8'), 
                                     ('n_node_samples', '<i8'), 
                                     ('weighted_n_node_samples', '<f8'),
                                     ('missing_go_to_left', 'u1')])
    state['values'] = np.zeros((node_count, n_outputs_, n_classes_[0]))
    
    state['nodes'][0] = (1, 2, vector[0][index] - 1, vector[1][index], 0.0, 1, 1.0, 0)  # root node with two children
    state['nodes'][1] = (-1, -1, -2, -2.0, 0.0, 1, 1.0, 0)  
    state['nodes'][2] = (-1, -1, -2, -2.0, 0.0, 1, 1.0, 0)  
    inner_tree.__setstate__(state)
    
    return inner_tree

def assign_tree_properties(clf, inner_tree):
    clf.tree_ = inner_tree
    clf.n_outputs_ = inner_tree.n_outputs
    clf.n_classes_ = inner_tree.n_classes[0]
    clf.classes_ = np.arange(inner_tree.n_classes[0])
    clf.n_features_in_ = inner_tree.n_features
    clf.max_features_ = inner_tree.n_features
    clf.feature_original = clf.tree_.feature
    clf.threshold_original = clf.tree_.threshold
    
def set_node_children(idx_to_node, index, vector):
    left_child_idx = 2 * index + 1
    right_child_idx = 2 * index + 2
    
    if left_child_idx < len(vector[0]):
        idx_to_node[left_child_idx].parent = idx_to_node[index]
        idx_to_node[left_child_idx].node_id = idx_to_node[index].node_id + 'l'

    if right_child_idx < len(vector[0]):
        idx_to_node[right_child_idx].parent = idx_to_node[index]
        idx_to_node[right_child_idx].node_id = idx_to_node[index].node_id + 'r'

    idx_to_node[index].node_l = idx_to_node[left_child_idx]
    idx_to_node[index].node_r = idx_to_node[right_child_idx]


def simplify_decode(node):
     if node.is_leaf():
         return {node.prediction}
     else:
         all_pred = simplify_decode(node.node_l) | simplify_decode(node.node_r)
         if len(all_pred) == 1:
             node.prediction = node.node_l.prediction
             node.make_leaf()
             return {node.prediction}
         else:
             return all_pred
    
    
    
