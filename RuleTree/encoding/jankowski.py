import numpy as np


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


def list_of_dicts_to_jankowski(list_of_dicts):
    enc = list()
    for d in list_of_dicts:
        enc.append(dict_to_jankowski(d))
    return np.array(enc)  # assumes all encodings have the same shape


def jankowski_to_list_of_dicts(enc, original_list_of_dicts):
    list_of_dicts = list()
    for i in range(enc.shape[0]):
        d = jankowski_to_dict(enc[i], original_list_of_dicts[i])
        list_of_dicts.append(d)
    return list_of_dicts


def dict_to_jankowski(d):
    # converts the ruletree json to a jakowski matrix corresponding to a (possibly) incomplete tree
    #  as in the classical implementation, the feature_idx and the threshold are shifted by 1
    matrix = np.ones((2, 2 ** (get_max_depth_from_json(d) + 1) - 1)) * np.nan

    nodes = {x['node_id']: x for x in d['nodes']}

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


def jankowski_to_dict(enc, original_dict):
    es_tree = {
        'tree_type': original_dict['tree_type'],
        'nodes': generate_node_list(deshift_jakowski_encoding(enc), original_dict["n_classes_"]),
        'args': original_dict['args'],
        'classes_': original_dict['classes_'],
        'n_classes_': original_dict['n_classes_'],
    }
    return es_tree


def deshift_jakowski_encoding(enc):
    # deshifts the feature_idx and the threshold of the jakowski matrix
    enc_ = enc.copy()
    enc_[0, :enc_.shape[1]//2] -= 1
    enc_[1, enc_.shape[1]//2:] -= 1
    return enc_


def generate_node_list(enc, n_classes):
    nodes = []
    for pos, (feat, thr) in enumerate(enc.T):
        if feat != thr and feat != -1:  # not leaf
            nodes.append({
                'node_id': get_col(pos),
                'stump_type': 'RuleTree.stumps.classification.DecisionTreeStumpClassifier',
                'feature_idx': feat,
                'threshold': thr,
                'is_leaf': False,
                'left_node': get_col(pos) + 'l',
                'right_node': get_col(pos) + 'r',
                'is_categorical': False,
                "prediction_probability": [np.nan] * n_classes,
            })
        else:
            nodes.append({
                'node_id': get_col(pos),
                'prediction': thr,
                "prediction_probability": np.eye(n_classes)[int(thr)].tolist(),
                'stump_type': '',
                'is_leaf': True,
            })
    return nodes
