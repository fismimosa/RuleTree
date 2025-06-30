import json

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

from RuleTree import RuleTreeClassifier

def get_depth(d:dict):
    depth = 0
    for dict_node in d['nodes']:
        depth = max(depth, len(dict_node['node_id']))

    return depth -1

def make_complete_rule_tree(filename):
    with open(filename, 'r') as f:
        d = json.load(f)

    print(d['args']['max_depth'], get_depth(d))



def test_clf_iris():
    df = pd.read_csv("datasets/CLF/iris.csv")
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    clf_rule = RuleTreeClassifier(max_depth=3, random_state=42)

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

if __name__ == "__main__":
    test_clf_iris()
