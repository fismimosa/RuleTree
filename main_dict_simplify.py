import copy
import json

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

from RuleTree import RuleTreeClassifier


def get_tree_depth(d:dict):
    depth = 0
    for dict_node in d['nodes']:
        depth = max(depth, len(dict_node['node_id']))

    return depth -1

def make_complete_rule_tree(filename, max_depth=None):
    with open(filename, 'r') as f:
        d = json.load(f)

    if max_depth is None:
        max_depth = max(d['args'].get('max_depth', 0), get_tree_depth(d))

    nodes = {el['node_id']: el for el in d['nodes']}
    leaves = [k for k, v in nodes.items() if v['is_leaf'] and len(k) - 1 < max_depth]

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

    with open(filename, 'w') as f:
        json.dump(d, f, indent=4)





def test_clf_iris():
    df = pd.read_csv("datasets/CLF/iris.csv")
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    clf_rule = RuleTreeClassifier(max_depth=5, random_state=42)

    clf_rule.fit(X_train, y_train)

    f1_rule = f1_score(y_test, clf_rule.predict(X_test), average='weighted')

    print(f"F1: {f1_rule}")

    clf_rule.print_rules(clf_rule.get_rules(columns_names=df.columns))

    clf_rule.export_graphviz(filename="test")
    clf_rule.to_dict("test_dict.json")
    make_complete_rule_tree("test_dict.json")

    clf_rule = RuleTreeClassifier.from_dict("test_dict.json")
    clf_rule.print_rules(clf_rule.get_rules(columns_names=df.columns))
    f1_rule = f1_score(y_test, clf_rule.predict(X_test), average='weighted')
    print(f"F1 from dict: {f1_rule}")
    clf_rule.export_graphviz(filename="test2")

    clf_rule.root = clf_rule.root.simplify()
    clf_rule.export_graphviz(filename="test3")

if __name__ == "__main__":
    test_clf_iris()
