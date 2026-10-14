"""Functions on the dictionaries created by RuleTree.to_dict."""

import copy
import json


def get_tree_depth(d:dict):
    """
    Depth of a saved tree.

    Parameters
    ----------
    d : dict
        Dictionary created by RuleTree.to_dict.

    Returns
    -------
    depth : int
    """
    depth = 0
    for dict_node in d['nodes']:
        depth = max(depth, len(dict_node['node_id']))

    return depth -1

def make_complete_rule_tree(filename, max_depth=None, verbose=False):
    """
    Complete a saved tree up to a depth: every leaf shallower than max_depth is replaced by a copy of
    the split of its parent, with two copies of the leaf as children, so that all the leaves have
    depth max_depth. The predictions do not change.

    Parameters
    ----------
    filename : dict, str or path-like
        Dictionary created by RuleTree.to_dict, or the JSON file where it is saved (overwritten).
    max_depth : int, default=None
        Depth of the complete tree. If None, the largest between the max_depth of the tree and its depth.
    verbose : bool, default=False
        Whether to print the number of leaves to complete.

    Returns
    -------
    d : dict
        The completed dictionary.
    """
    if not isinstance(filename, dict):
        with open(filename, 'r') as f:
            d = json.load(f)
    else:
        d = filename

    if max_depth is None:
        max_depth = max(d['args'].get('max_depth', 0), get_tree_depth(d))

    nodes = {el['node_id']: el for el in d['nodes']}
    leaves = [k for k, v in nodes.items() if v['is_leaf'] and len(k) - 1 < max_depth]

    if verbose:
        print('Leaves to complete:', len(leaves))

    while leaves:
        node_id = leaves.pop()
        left_id = node_id + 'l'
        right_id = node_id + 'r'
        parent_id = node_id[:-1]

        node = nodes[node_id]
        new_node = copy.deepcopy(nodes[parent_id])
        new_node['node_id'] = node_id
        new_node['is_leaf'] = False

        left_leaf = copy.deepcopy(node)
        right_leaf = copy.deepcopy(node)

        left_leaf['node_id'] = left_id
        right_leaf['node_id'] = right_id

        new_node['left_node'] = left_id
        new_node['right_node'] = right_id

        nodes[node_id] = new_node
        nodes[left_id] = left_leaf
        nodes[right_id] = right_leaf

        if len(left_id) - 1 < max_depth:
            leaves.append(left_id)
        if len(right_id) - 1 < max_depth:
            leaves.append(right_id)

    d['nodes'] = list(nodes.values())

    if not isinstance(filename, dict):
        with open(filename, 'w') as f:
            json.dump(d, f, indent=4)

    return d