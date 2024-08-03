import os
from glob import glob
from time import time

import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from progress_table import ProgressTable

from RuleTree.RuleTreeClassifier import RuleTreeClassifier

def test_clf():
    datasets = glob("datasets/CLF/*.csv")
    table = ProgressTable(pbar_embedded=False, pbar_show_progress=True, pbar_show_percents=True,
                          pbar_show_throughput=False, num_decimal_places=3)

    for dataset in datasets:
        dataset_name = dataset.split("/")[-1][:-4]
        table["dataset"] = dataset_name

        try:
            df = pd.read_csv(dataset)
            ct = ColumnTransformer([("cat", OneHotEncoder(), make_column_selector(dtype_include="object"))],
                                   remainder='passthrough', verbose_feature_names_out=False, sparse_threshold=0, n_jobs=12)
            df_onehot = pd.DataFrame(ct.fit_transform(df.iloc[:, :-1]), columns=ct.get_feature_names_out())

            X = df.iloc[:, :-1].values
            y = df.iloc[:, -1].values
            X_onehot = df_onehot.to_numpy()

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
            X_train_onehot, X_test_onehot, _, _ = train_test_split(X_onehot, y, test_size=0.3, random_state=42,
                                                                   stratify=y)

            clf_rule = RuleTreeClassifier()
            clf_sklearn = DecisionTreeClassifier()

            start_rule = time()
            clf_rule.fit(X_train, y_train)
            end_rule = time()


            start_sklearn = time()
            clf_sklearn.fit(X_train_onehot, y_train)
            end_sklearn = time()

            f1_rule = f1_score(y_test, clf_rule.predict(X_test), average='weighted')
            f1_sklearn = f1_score(y_test, clf_sklearn.predict(X_test_onehot), average='weighted')

            table["f1_rule"] = f1_rule
            table["f1_sklearn"] = f1_sklearn
            table["time_rule"] = end_rule - start_rule
            table["time_sklearn"] = end_sklearn - start_sklearn
        except ValueError as e:
            table["error"] = str(e)

        table.next_row()








if __name__ == "__main__":
    test_clf()
