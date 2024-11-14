def ruletree_to_array(node, max_depth=2):
    max_nodes = 2 ** (max_depth + 1) - 1
    node_array = np.empty(max_nodes, dtype=object)
    queue = [(node, 1)] 
    
    while queue:
        current_node, index = queue.pop(0)
        node_dict = current_node.node_to_dict()
        node_dict['position'] = index
        node_array[index - 1] = node_dict
        
        if current_node.node_l is not None and 2 * index <= max_nodes:
            queue.append((current_node.node_l, 2 * index))
        if current_node.node_r is not None and 2 * index + 1 <= max_nodes:
            queue.append((current_node.node_r, 2 * index + 1))
    
    return node_array

def build_subtree(index):
    # Base case: if index is out of bounds, return None
    if index >= len(node_array) or node_array[index] is None:
        return None
    
    # Create the current node
    node = RuleTreeNode.dict_to_node(info_dict=node_array[index])

    left_index = 2 * index + 1
    right_index = 2 * index + 2
    
    node.node_l = build_subtree(left_index)  
    node.node_r = build_subtree(right_index)  
    
    if node.node_l:
        node.node_l.parent = node
    if node.node_r:
        node.node_r.parent = node
    
    return node

def array_to_ruletree(node_array, max_depth=2, n_classes_=2):
    ruletree = RuleTreeClassifier(max_depth=max_depth)
    ruletree.classes_ = [i for i in range(n_classes_)]
    ruletree.root = build_subtree(0)
    ruletree.root.parent = -1
    
    return ruletree
