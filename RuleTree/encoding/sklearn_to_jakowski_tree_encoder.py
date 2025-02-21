import copy

from sklearn.tree._tree import Tree, TREE_LEAF, TREE_UNDEFINED
from sklearn.tree import DecisionTreeClassifier
import numpy as np
from RuleTree.encoding.jakowski_utils import (
    max_node_count,
)


class JakowskiEncoder:
    def __init__(self, n_features, n_classes):
        self.n_features_in_ = n_features
        self.n_classes_ = n_classes
        self.classes_ = np.arange(n_classes)
        self.n_outputs_ = 1

    def encode(self, clf, depth):
        decisions = clf.tree_.value.argmax(axis=2).flatten().tolist()

        vector = np.zeros((2, max_node_count(depth)))

        index = {}  # {node : node index in the vector}
        index[0] = 0  # root

        parent = {}  # {node : parent_node}

        for node in range(clf.tree_.node_count):
            if is_leaf(clf.tree_, node):
                vector[0][index[node]] = -1
                vector[1][index[node]] = decisions[node] + 1  # class label
            else:
                vector[0][index[node]] = clf.tree_.feature[node] + 1  # feature [1, m]
                vector[1][index[node]] = clf.tree_.threshold[node]  # threshold

                index[clf.tree_.children_left[node]] = 2 * index[node] + 1
                index[clf.tree_.children_right[node]] = 2 * index[node] + 2

                parent[2 * index[node] + 1] = index[node]
                parent[2 * index[node] + 2] = index[node]

        for node in range(vector.shape[1]):
            if vector[0][node] == -1:
                # Aggiorno i figli
                if ((2 * node + 1) < (vector.shape[1] - 1)) and (
                    (2 * node + 2) < vector.shape[1]
                ):
                    vector[0][2 * node + 1] = -1
                    vector[1][2 * node + 1] = vector[1][node]
                    vector[0][2 * node + 2] = -1
                    vector[1][2 * node + 2] = vector[1][node]

                    parent[2 * node + 1] = node
                    parent[2 * node + 2] = node

                    # Aggiorno il nodo stesso
                    vector[0][node] = vector[0][parent[node]]
                    vector[1][node] = vector[1][parent[node]]

        return vector

    def decode(self, vector, prune=True):
        node_count = sum((vector[0] != 0) * 1)  # count the 'real' nodes in the vector
        n_classes_ = np.array([self.n_classes_], dtype=np.intp)

        inner_tree = Tree(
            self.n_features_in_, n_classes_, self.n_outputs_
        )  # underlying Tree object
        state = inner_tree.__getstate__()

        state["node_count"] = node_count
        state["nodes"] = np.zeros(
            (node_count,),
            dtype=[
                ("left_child", "<i8"),
                ("right_child", "<i8"),
                ("feature", "<i8"),
                ("threshold", "<f8"),
                ("impurity", "<f8"),
                ("n_node_samples", "<i8"),
                ("weighted_n_node_samples", "<f8"),
                ("missing_go_to_left", "u1"),
            ],
        )
        state["values"] = np.zeros((node_count, self.n_outputs_, self.n_classes_))

        internal_nodes = []  # array of internal nodes (index of the node in the tree)
        current_id = 0

        stack = []
        stack.append(0)  # root

        tree = {}  # {vector column index : index of the node in the tree}
        children = (
            {}
        )  # {vector column index: [0-L/1-R, index of the parent node in the tree]}
        for x in range(0, vector.shape[1]):
            tree[x] = -1
            children[x] = [-1, -1]

        while stack:
            index = stack.pop()  # pop the last item
            tree[index] = current_id

            # Update parent node (if child is not the root of the tree)
            # Save the tree index of the child in the parent node
            if index != 0:
                state["nodes"][children[index][1]][children[index][0]] = current_id

            # If node is a leaf
            if vector[0][index] == -1:
                state["nodes"][current_id][0] = -1  # children_left
                state["nodes"][current_id][1] = -1  # children_right
                state["nodes"][current_id][2] = -2  # feature
                state["nodes"][current_id][3] = -2.0  # threshold
                state["nodes"][current_id][5] = 1  # n_node_samples
                state["nodes"][current_id][6] = 1.0  # weighted_n_node_samples
                state["values"][current_id][0][
                    int(vector[1][index] - 1)
                ] = 1.0  # values

            # If node is an internal node
            else:
                internal_nodes.append(current_id)

                state["nodes"][current_id][2] = vector[0][index] - 1  # feature [0, m-1]
                state["nodes"][current_id][3] = vector[1][index]  # threshold

                # children right
                children[2 * index + 2][0] = 1
                children[2 * index + 2][1] = current_id
                stack.append(2 * index + 2)

                # children left
                children[2 * index + 1][0] = 0
                children[2 * index + 1][1] = current_id
                stack.append(2 * index + 1)

            current_id += 1

        # Update the number of samples belonging to each internal node
        for node in internal_nodes[::-1]:
            state["nodes"][node][5] = (
                state["nodes"][state["nodes"][node][0]][5]
                + state["nodes"][state["nodes"][node][1]][5]
            )  # n_node_samples
            state["nodes"][node][6] = (
                state["nodes"][state["nodes"][node][0]][6]
                + state["nodes"][state["nodes"][node][1]][6]
            )  # weighted_n_node_samples
            state["values"][node][0] = (
                state["values"][state["nodes"][node][0]][0]
                + state["values"][state["nodes"][node][1]][0]
            )  # values
            state["nodes"][node][4] = 1 - sum(
                (state["values"][node][0] / sum(state["values"][node][0])) ** 2
            )  # impurity

        # FIXME: This is needed in order for the predict proba to be equal before and
        #  after decoding. I hope this does not break anything else.
        state["values"] = np.eye(state["values"].shape[-1])[state["values"].argmax(axis=2)]

        inner_tree.__setstate__(state)

        state = inner_tree.__getstate__()
        state["max_depth"] = get_depth(inner_tree)  # update

        inner_tree.__setstate__(state)

        clf = DecisionTreeClassifier()
        clf.tree_ = inner_tree  # tree_
        clf.n_outputs_ = inner_tree.n_outputs  # n_outputs_
        clf.n_classes_ = inner_tree.n_classes[0]  # n_classes_
        clf.classes_ = np.arange(inner_tree.n_classes[0])  # classes_
        clf.n_features_in_ = inner_tree.n_features  # n_features_in
        clf.max_features_ = inner_tree.n_features  # max_features_

        if prune:
            prune_duplicate_leaves(clf)

        return clf

    def is_tree(self, vector):
        # Checks if self.vector matches a tree.

        # Check that the tree has at least 3 nodes (1 internal_node and 2 leaf_node)
        if vector.shape[1] < 3:
            return False

        # Check that the number of leaves is equal to "n_leaves = n_internal_nodes + 1"
        n_internal_nodes = len(np.where(vector[0] > 0)[0].tolist())
        n_leaves = vector[0].flatten().tolist().count(-1)

        # To avoid trees with one node (leaf node)
        if n_internal_nodes < 1:
            return False

        if n_leaves != (n_internal_nodes + 1):
            return False

        # Check the values of the features in the first line.
        # The features must be in the range [1, n_features +1].
        features = set(
            int(x) for x in [vector[0][x] for x in np.where(vector[0] != -1)][0]
        )
        if min(features) < 1 or max(features) > (self.n_features_in_ + 1):
            return False

        classes = set(
            int(x) for x in [vector[1][x] for x in np.where(vector[0] == -1)][0]
        )
        if min(classes) <= 0 or max(classes) > self.n_classes_:
            return False

        # Check that each inner node has 2 leaves
        last_internal_node = np.where(vector[0] > 0)[0].tolist()[-1]

        if (last_internal_node * 2 + 2) != (vector.shape[1] - 1):
            return False

        return True

    def pad(self, vector, dim, constant_values=0):
        if dim > vector.shape[1]:
            diff = dim - vector.shape[1]
            return np.pad(
                array=vector,
                pad_width=((0, 0), (0, diff)),
                mode="constant",
                constant_values=constant_values,
            )

    def unpad(self, vector):
        index = vector.shape[1]
        for i in range(index - 1, 0, -1):
            if vector[0][i] != 0 and vector[1][i] != 0:
                index = i + 1
                break
        return vector[:, :index]

    def complete_tree(self, clf, X):
        # This procedure assign the instances of a set X in each node of the DT.
        # See https://ieeexplore.ieee.org/abstract/document/8242347

        internal_nodes = np.where(clf.tree_.feature > 0)[
            0
        ].tolist()  # list of IDs of internal nodes
        internal_nodes.sort(reverse=True)

        leaves = clf.apply(
            X
        )  # List of index of the leaf that each sample is predicted as.
        classes = clf.predict(X)  # List of predict class for X.

        state = clf.tree_.__getstate__()

        # Set to zero 'n_node_samples', 'weighted_n_node_samples' and state['values'] for each node
        for node in range(clf.tree_.node_count):
            state["nodes"][node][5] = 0
            state["nodes"][node][6] = 0
            state["values"][node][0] = np.zeros(
                clf.n_classes_,
            )

        # Update 'n_node_samples', 'weighted_n_node_samples' and state['values'] for each leaf
        for i in range(len(leaves)):
            state["values"][leaves[i]][0][classes[i]] += 1
            state["nodes"][leaves[i]][5] += 1
            state["nodes"][leaves[i]][6] += 1

        # Update 'n_node_samples', 'weighted_n_node_samples', state['values'] and 'impurity' for each internal node
        for node in internal_nodes:
            state["nodes"][node][5] = (
                state["nodes"][state["nodes"][node][0]][5]
                + state["nodes"][state["nodes"][node][1]][5]
            )  # n_node_samples
            state["nodes"][node][6] = (
                state["nodes"][state["nodes"][node][0]][6]
                + state["nodes"][state["nodes"][node][1]][6]
            )  # weighted_n_node_samples
            state["values"][node][0] = (
                state["values"][state["nodes"][node][0]][0]
                + state["values"][state["nodes"][node][1]][0]
            )  # values
            state["nodes"][node][4] = 1 - sum(
                (state["values"][node][0] / sum(state["values"][node][0])) ** 2
            )

        clf.tree_.__setstate__(state)


def refine_prune_index(clf):
    # Corrects the prune index function by changing the state of the decision tree
    children_left = [x for x in clf.tree_.children_left if x != -1]
    children_right = [x for x in clf.tree_.children_right if x != -1]

    nodes = children_left + children_right
    nodes.append(0)  # root
    nodes.sort()

    n_nodes = len(nodes)  # number of nodes in the tree

    state = clf.tree_.__getstate__()

    state["node_count"] = n_nodes
    state["max_depth"] = get_depth(clf.tree_)
    new_nodes = np.zeros(
        (n_nodes,),
        dtype=[
            ("left_child", "<i8"),
            ("right_child", "<i8"),
            ("feature", "<i8"),
            ("threshold", "<f8"),
            ("impurity", "<f8"),
            ("n_node_samples", "<i8"),
            ("weighted_n_node_samples", "<f8"),
            ("missing_go_to_left", "u1"),
        ],
    )
    new_values = np.zeros((n_nodes, clf.n_outputs_, clf.n_classes_))

    i = 0  # index in the new state
    index = {}  # {node index in the old state : node index in the new state}
    index[0] = 0  # root
    parent = {}  # {node index in the old state : parent index in the new state}
    for node in nodes:
        index[node] = i
        new_nodes[index[node]] = state["nodes"][node]
        new_values[index[node]] = state["values"][node]

        if state["nodes"][node][0] != -1:  # not a leaf
            parent[state["nodes"][node][0]] = (index[node], 0)  # left child
            parent[state["nodes"][node][1]] = (index[node], 1)  # right child

        if node != 0:  # non root node
            new_nodes[parent[node][0]][parent[node][1]] = index[
                node
            ]  # update the parent

        i += 1

    state["nodes"] = copy.deepcopy(new_nodes)
    state["values"] = copy.deepcopy(new_values)

    clf.tree_.__setstate__(state)


def is_leaf(inner_tree, index):
    # Check whether node is leaf node
    return (
        inner_tree.children_left[index] == TREE_LEAF
        and inner_tree.children_right[index] == TREE_LEAF
    )


def prune_index(inner_tree, decisions, index=0):
    # Start pruning from the bottom - if we start from the top, we might miss
    # nodes that become leaves during pruning.
    # Do not use this directly - use prune_duplicate_leaves instead.
    if not is_leaf(inner_tree, inner_tree.children_left[index]):
        prune_index(inner_tree, decisions, inner_tree.children_left[index])
    if not is_leaf(inner_tree, inner_tree.children_right[index]):
        prune_index(inner_tree, decisions, inner_tree.children_right[index])

    children_left = len([x for x in inner_tree.children_left if x != -1])
    children_right = len([x for x in inner_tree.children_right if x != -1])

    n_nodes = children_left + children_right + 1

    if n_nodes > 3:
        # Prune children if both children are leaves now and make the same decision:
        if (
            is_leaf(inner_tree, inner_tree.children_left[index])
            and is_leaf(inner_tree, inner_tree.children_right[index])
            and (decisions[index] == decisions[inner_tree.children_left[index]])
            and (decisions[index] == decisions[inner_tree.children_right[index]])
        ):
            # turn node into a leaf by "unlinking" its children
            inner_tree.children_left[index] = TREE_LEAF
            inner_tree.children_right[index] = TREE_LEAF
            inner_tree.threshold[index] = TREE_UNDEFINED
            inner_tree.feature[index] = TREE_UNDEFINED
            # print("Pruned {}".format(index))


def prune_duplicate_leaves(clf):
    if clf.tree_.node_count > 3:
        # Remove leaves if both
        decisions = (
            clf.tree_.value.argmax(axis=2).flatten().tolist()
        )  # Decision for each node
        prune_index(clf.tree_, decisions)
        refine_prune_index(clf)


def leaves_count(clf):
    # Returns the number of leaves of the tree.

    children_left = clf.tree_.children_left
    children_right = clf.tree_.children_right

    n_leaves = 0

    stack = [(0, 0)]  # start with the root node id (0) and its depth (0)
    while len(stack) > 0:
        node_id, depth = stack.pop()

        # If the left and right child of a node is not the same we have a split
        is_split_node = children_left[node_id] != children_right[node_id]

        # If a split node, append left and right children and depth to `stack`
        # so we can loop through them
        if is_split_node:
            stack.append((children_left[node_id], depth + 1))
            stack.append((children_right[node_id], depth + 1))
        else:
            n_leaves += 1

    return n_leaves


def get_depth(inner_tree, node=0, depth=0, verbose=False):
    if verbose:
        print("Node: {} - depth: {}".format(node, depth))

    if (
        inner_tree.children_left[node] != TREE_LEAF
        and inner_tree.children_right[node] != TREE_LEAF
    ):
        return max(
            get_depth(
                inner_tree, inner_tree.children_left[node], depth + 1, verbose=verbose
            ),
            get_depth(
                inner_tree, inner_tree.children_right[node], depth + 1, verbose=verbose
            ),
        )

    return depth
