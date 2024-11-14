import pandas as pd
from graphviz import Source
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

from ruletree import RuleTreeClassifier


def main():
    df = pd.read_csv("datasets/CLF/iris.csv")
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    clf_rule = RuleTreeClassifier(max_depth=5)

    clf_rule.fit(X_train, y_train)

    f1_rule = f1_score(y_test, clf_rule.predict(X_test), average='weighted')

    print(f"F1: {f1_rule}")

    print()
    print(clf_rule.print_rules(clf_rule.get_rules()))

    s = Source(clf_rule.root.export_graphviz().to_string())
    s.view()

    #clf_rule.root.export_graphviz().write_png("output.png")

if __name__ == '__main__':
    main()