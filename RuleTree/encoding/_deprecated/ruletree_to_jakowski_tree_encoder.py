import numpy as np
import math
from RuleTree.encoding._deprecated.jakowski_utils import is_leaf_index, get_deepest_descendants, get_parent_index, max_node_count
from RuleTree.encoding.jankowski import dict_to_jankowski


def jakowski_incomplete_to_jakowski_complete(enc, depth=None):
    # grows the incomplete jakowski matrix to a complete jakowski matrix
    if depth is None:
        enc_ = enc.copy()
    else:
        enc_ = np.full((2, max_node_count(depth)), -1.0)
        enc_[:, :enc.shape[1]] = enc
    for i in range(enc_.shape[1] // 2):  # for each internal node
        # enc_[0, i] == -1 -> the node is a leaf in the incomplete tree
        # not is_leaf_index(i, enc_.shape[1]) -> the node is not a leaf in the complete tree
        # not enc_[1, i] == -1 -> the node is not a missing node in the incomplete tree
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


def ruletree_to_jakowski(json, depth=None):
    # converts the ruletree json to a jakowski matrix corresponding to a complete tree
    return jakowski_incomplete_to_jakowski_complete(dict_to_jankowski(json), depth=depth)


