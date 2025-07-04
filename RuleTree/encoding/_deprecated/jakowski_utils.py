import math


def get_deepest_descendants(i, depth):
    # Returns the indexes of the deepest descendants of a node, given its index and the depth of the tree,
    # assuming the tree is complete
    d_i = math.floor(math.log2(i + 1))
    if d_i == depth:
        return []
    d = depth - d_i
    left_desc = i * (2 ** d) + (2 ** d - 1)
    right_desc = i * (2 ** d) + (2 ** (d + 1) - 2)

    return list(range(left_desc, right_desc + 1))


def get_parent_index(index):
    # Returns the parent node index of a node, given its index
    return (index - 1) // 2


def get_children_index(index):
    # Returns the children node indexes of a node, given its index
    return 2 * index + 1, 2 * index + 2


def is_leaf_index(index, n_nodes):
    # Returns whether a node is a leaf node, given its index and the number of nodes
    return 2 * index + 1 >= n_nodes


def max_node_count(max_depth):
    # Returns the maximum number of nodes a tree can have, given its maximum depth
    nodes = 0
    for i in range(0, max_depth + 1):
        nodes += pow(2, i)
    return nodes


