import pandas as pd
from sklearn.metrics import classification_report, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from RuleTree import RuleTreeClassifier, RuleTreeRegressor
from RuleTree.stumps.classification.PivotTreeStumpClassifier import PivotTreeStumpClassifier
from RuleTree.stumps.regression.PivotTreeStumpRegressor import PivotTreeStumpRegressor


if __name__ == "__main__":
    random_state = 42

    print("=== CLASSIFICATION (datasets/CLF/iris.csv) ===")
    df_clf = pd.read_csv("datasets/CLF/iris.csv")
    X_clf = df_clf.iloc[:, :-1].values
    y_clf = df_clf.iloc[:, -1].values

    X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(
        X_clf,
        y_clf,
        test_size=0.3,
        random_state=random_state,
        stratify=y_clf,
    )

    clf_std = RuleTreeClassifier(
        max_depth=4,
        stump_selection="best",
        random_state=random_state,
    )
    clf_pivot = RuleTreeClassifier(
        max_depth=4,
        stump_selection="best",
        random_state=random_state,
        base_stumps=PivotTreeStumpClassifier(max_depth=1, random_state=random_state),
    )

    clf_std.fit(X_train_clf, y_train_clf)
    clf_pivot.fit(X_train_clf, y_train_clf)

    y_pred_std = clf_std.predict(X_test_clf)
    y_pred_pivot = clf_pivot.predict(X_test_clf)

    print("\n[RuleTreeClassifier - standard]")
    print(classification_report(y_test_clf, y_pred_std))

    print("[RuleTreeClassifier - pivot]")
    print(classification_report(y_test_clf, y_pred_pivot))

    print("\n=== REGRESSION (datasets/REG/boston.csv) ===")
    df_reg = pd.read_csv("datasets/REG/boston.csv")
    X_reg = df_reg.iloc[:, :-1].values
    y_reg = MinMaxScaler().fit_transform(df_reg.iloc[:, -1].values.reshape(-1, 1))

    X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
        X_reg,
        y_reg,
        test_size=0.3,
        random_state=random_state,
    )

    reg_std = RuleTreeRegressor(
        max_depth=4,
        stump_selection="best",
        random_state=random_state,
    )
    reg_pivot = RuleTreeRegressor(
        max_depth=4,
        stump_selection="best",
        random_state=random_state,
        base_stumps=PivotTreeStumpRegressor(max_depth=1, random_state=random_state),
    )

    reg_std.fit(X_train_reg, y_train_reg)
    reg_pivot.fit(X_train_reg, y_train_reg)

    y_pred_std = reg_std.predict(X_test_reg)
    y_pred_pivot = reg_pivot.predict(X_test_reg)

    print("\n[RuleTreeRegressor - standard]")
    print(f"MSE: {mean_squared_error(y_test_reg, y_pred_std):.4f}")
    print(f"R2 : {r2_score(y_test_reg, y_pred_std):.4f}")

    print("\n[RuleTreeRegressor - pivot]")
    print(f"MSE: {mean_squared_error(y_test_reg, y_pred_pivot):.4f}")
    print(f"R2 : {r2_score(y_test_reg, y_pred_pivot):.4f}")
