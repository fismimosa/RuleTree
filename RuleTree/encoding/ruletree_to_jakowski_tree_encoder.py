import numpy as np
import math
from RuleTree.encoding.jakowski_utils import is_leaf_index, get_deepest_descendants, get_parent_index


def get_max_depth_from_json(json:dict):
    # dato il dizionario ti ritorna la profondità massima (dei NODI! +1 per le foglie)
    return max([len(x['node_id']) for x in json['nodes']])-1


def get_col(n):
    # data la posizione nell'array ritorna la sequenza di lettere che identifica il nodo
    if n == 0:
        return "R"
    label = ""
    while n > 0:
        label = ("l" if n % 2 == 1 else "r") + label
        n = (n - 1) // 2
    return "R" + label


def jakowski_incomplete_to_jakowski_complete(enc):
    # grows the incomplete jakowski matrix to a complete jakowski matrix
    enc_ = enc.copy()
    for i in range(enc.shape[1] // 2):  # for each internal node
        # enc_[0, i] == -1 -> the node is a leaf in the incomplete tree
        # not is_leaf_index(i, enc_.shape[1]) -> the node is not a leaf in the complete tree
        # not enc_[1, i] == -1 -> the node is a missing node in the incomplete tree
        if enc_[0, i] == -1 and not is_leaf_index(i, enc_.shape[1]) and not enc_[1, i] == -1:
            # paste the class in the deepest descendants, i.e., the real leaves in the complete tree
            enc_[0, get_deepest_descendants(i, math.floor(math.log2(enc_.shape[1])))] = -1
            enc_[1, get_deepest_descendants(i, math.floor(math.log2(enc_.shape[1])))] = enc_[1, i]
    for i in range(enc_.shape[1] // 2):  # for each internal node
        # enc_[0, i] == -1 -> the node is a leaf or a missing node in the incomplete tree
        if enc_[0, i] == -1:
            # paste the feature and the threshold of the parent node
            parent_idx = get_parent_index(i)
            enc_[0, i] = enc_[0, parent_idx]
            enc_[1, i] = enc_[1, parent_idx]
    return enc_


def json_to_jakowski_incomplete(json):
    # converts the ruletree json to a jakowski matrix corresponding to a (possibly) incomplete tree
    #  as in the classical implementation, the feature_idx and the threshold are shifted by 1
    matrix = np.ones((2, 2**(get_max_depth_from_json(json)+1)-1))*np.nan

    nodes = {x['node_id']: x for x in json['nodes']}

    for pos in range(matrix.shape[1]):
        k = get_col(pos)
        if k not in nodes:
            #skip
            matrix[:, pos] = -1
            continue

        nodo = nodes[k]
        if nodo['is_leaf']:
            matrix[0, pos] = -1
            matrix[1, pos] = nodo['prediction'] + 1
        else:
            matrix[0, pos] = nodo['feature_idx'] + 1
            matrix[1, pos] = nodo['threshold']

    return matrix


def ruletree_to_jakowski(json):
    # converts the ruletree json to a jakowski matrix corresponding to a complete tree
    return jakowski_incomplete_to_jakowski_complete(json_to_jakowski_incomplete(json))


def jakowski_to_ruletree(enc, prune=False):
    # TODO: the output is not pruned, yet the behaviour of the ruletree should be the same
    es_tree = {
        'tree_type': 'RuleTree.tree.RuleTreeClassifier',
        'nodes': generate_node_list(deshift_jakowski_encoding(enc)),
    }
    if not prune:
        return es_tree
    else:
        raise NotImplementedError("Pruning is not implemented yet")


def deshift_jakowski_encoding(enc):
    # deshifts the feature_idx and the threshold of the jakowski matrix
    enc_ = enc.copy()
    enc_[0, :enc_.shape[1]//2] -= 1
    enc_[1, enc_.shape[1]//2:] -= 1
    return enc_


def generate_node_list(enc):
    nodes = []
    for pos, (feat, thr) in enumerate(enc.T):
        if feat != thr and feat != -1: #not leaf
            nodes.append({
                'node_id': get_col(pos),
                'stump_type': 'RuleTree.stumps.classification.DecisionTreeStumpClassifier',
                'feature_idx': feat,
                'threshold': thr,
                'is_leaf': False,
                'left_node': get_col(pos)+'l',
                'right_node': get_col(pos)+'r',
                'is_categorical': False,
            })
        else:
            nodes.append({
                'node_id': get_col(pos),
                'prediction': thr,
                'stump_type': '',
                'is_leaf': True,
            })
    return nodes
