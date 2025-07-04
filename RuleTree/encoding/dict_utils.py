import copy
import json


def get_tree_depth(d:dict):
    depth = 0
    for dict_node in d['nodes']:
        depth = max(depth, len(dict_node['node_id']))

    return depth -1


def make_complete_rule_tree(filename, max_depth=None, verbose=False):
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
