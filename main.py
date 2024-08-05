import os
import warnings
from glob import glob
from time import time

import pandas as pd
import numpy as np
from sklearn import tree
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score, r2_score, silhouette_score, rand_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from progress_table import ProgressTable

from ruletree import RuleTree
from ruletree import RuleTreeClassifier
from ruletree import RuleTreeCluster
from ruletree import RuleTreeRegressor
from benchmark.config import dataset_target_clu


def test_clf(max_depth=4):
    datasets = glob("datasets/CLF/*.csv")
    table = ProgressTable(pbar_embedded=False, pbar_show_progress=True, pbar_show_percents=True,
                          pbar_show_throughput=False, num_decimal_places=3)

    for dataset in datasets:
        dataset_name = dataset.split("/")[-1][:-4]
        table["dataset"] = dataset_name

        try:
            df = pd.read_csv(dataset)
            ct = ColumnTransformer([("cat", OneHotEncoder(), make_column_selector(dtype_include="object"))],
                                   remainder='passthrough', verbose_feature_names_out=False, sparse_threshold=0, n_jobs=1)

            clf_rule = RuleTreeClassifier(max_depth=max_depth, prune_useless_leaves=True)
            clf_sklearn = DecisionTreeClassifier(max_depth=max_depth)

            start_rule = time()
            X = df.iloc[:, :-1].values
            y = df.iloc[:, -1].values

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

            clf_rule.fit(X_train, y_train)
            end_rule = time()

            start_sklearn = time()
            df_onehot = pd.DataFrame(ct.fit_transform(df.iloc[:, :-1]), columns=ct.get_feature_names_out())
            X_onehot = df_onehot.to_numpy()
            X_train_onehot, X_test_onehot, _, _ = train_test_split(X_onehot, y, test_size=0.3, random_state=42,
                                                                   stratify=y)

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


def test_reg(max_depth=4):
    dataset_target_reg = {
        'abalone': 'Rings',
        'auction': 'verification.time',
        'boston': 'MEDV',
        'carprice': 'price',
        'drinks': 'drinks',
        'insurance': 'charges',
        'intrusion': 'Number of Barriers',
        'metamaterial': 'BandGapWidth',
        'parkinsons_updrs': 'motor_UPDRS',
        'parkinsons_updrs_total': 'total_UPDRS',
        'students': 'Performance Index',
    }

    #datasets = ["datasets/REG/drinks.csv"]
    datasets = glob("datasets/REG/*.csv")
    table = ProgressTable(pbar_embedded=False, pbar_show_progress=True, pbar_show_percents=True,
                          pbar_show_throughput=False, num_decimal_places=3)

    for dataset in datasets:
        dataset_name = dataset.split("/")[-1][:-4]
        table["dataset"] = dataset_name

        df = pd.read_csv(dataset)
        ct = ColumnTransformer([("cat", OneHotEncoder(), make_column_selector(dtype_include="object"))],
                               remainder='passthrough', verbose_feature_names_out=False, sparse_threshold=0, n_jobs=1)

        clf_rule = RuleTreeRegressor(max_depth=max_depth)
        clf_sklearn = DecisionTreeRegressor(max_depth=max_depth)

        start_rule = time()
        X = df.drop(columns=dataset_target_reg[dataset_name]).to_numpy()
        y = df[dataset_target_reg[dataset_name]].values

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        clf_rule.fit(X_train, y_train)
        end_rule = time()


        start_sklearn = time()
        df_onehot = pd.DataFrame(ct.fit_transform(df.drop(columns=dataset_target_reg[dataset_name])),
                                 columns=ct.get_feature_names_out())
        X_onehot = df_onehot.to_numpy()
        X_train_onehot, X_test_onehot, _, _ = train_test_split(X_onehot, y, test_size=0.3, random_state=42)

        clf_sklearn.fit(X_train_onehot, y_train)
        end_sklearn = time()

        y_pred_rule = clf_rule.predict(X_test)
        y_pred_sklearn = clf_sklearn.predict(X_test_onehot)

        r2_rule = r2_score(y_test, y_pred_rule)
        r2_sklearn = r2_score(y_test, y_pred_sklearn)

        table["r2_rule"] = r2_rule
        table["r2_sklearn"] = r2_sklearn
        table["time_rule"] = end_rule - start_rule
        table["time_sklearn"] = end_sklearn - start_sklearn

        """text_representation = tree.export_text(clf_sklearn, feature_names=ct.get_feature_names_out())
        print(text_representation)

        ruletree.print_rules(clf_rule.get_rules(), columns_names=df.columns)"""

        table.next_row()



def test_clu(max_depth=4):
    datasets = glob("datasets/CLU/*.csv")
    table = ProgressTable(pbar_embedded=False, pbar_show_progress=True, pbar_show_percents=True,
                          pbar_show_throughput=False, num_decimal_places=3)

    for dataset in datasets:
        dataset_name = dataset.split("/")[-1][:-4]
        table["dataset"] = dataset_name

        try:
            df = pd.read_csv(dataset)
            ct = ColumnTransformer([("cat", OneHotEncoder(), make_column_selector(dtype_include="object"))],
                                   remainder='passthrough', verbose_feature_names_out=False, sparse_threshold=0, n_jobs=1)

            clf_rule = RuleTreeCluster(max_depth=max_depth, prune_useless_leaves=True)
            clf_sklearn = KMeans(n_clusters=2**max_depth)

            df_onehot = pd.DataFrame(ct.fit_transform(df.iloc[:, :-1]), columns=ct.get_feature_names_out())
            X_onehot = df_onehot.to_numpy()

            start_rule = time()
            X = df.drop(columns=[dataset_target_clu[dataset_name]]).values
            y = df[dataset_target_clu[dataset_name]].values


            clf_rule.fit(X_onehot)
            end_rule = time()

            start_sklearn = time()
            clf_sklearn.fit(X_onehot)
            end_sklearn = time()

            rand_rule = rand_score(y, clf_rule.predict(X_onehot))
            rand_sklearn = rand_score(y, clf_sklearn.predict(X_onehot))
            sil_rule = silhouette_score(X_onehot, clf_rule.predict(X_onehot))
            sil_sklearn = silhouette_score(X_onehot, clf_sklearn.predict(X_onehot))

            table["rand_rule"] = rand_rule
            table["rand_sklearn"] = rand_sklearn
            table["sil_rule"] = sil_rule
            table["sil_sklearn"] = sil_sklearn
            table["time_rule"] = end_rule - start_rule
            table["time_sklearn"] = end_sklearn - start_sklearn
        except ValueError as e:
            table["error"] = str(e)
            raise e

        table.next_row()

def test_clf_iris():
    df = pd.read_csv("datasets/CLF/iris.csv")
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    clf_rule = RuleTreeClassifier(max_depth=10, prune_useless_leaves=True)

    clf_rule.fit(X_train, y_train)

    f1_rule = f1_score(y_test, clf_rule.predict(X_test), average='weighted')

    RuleTree.print_rules(clf_rule.get_rules(), columns_names=df.columns)

    print(f"F1: {f1_rule}")





if __name__ == "__main__":
    #test_clf()
    #test_reg()
    #test_clu()

    test_clf_iris()
