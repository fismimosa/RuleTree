import os
import pickle
import warnings
from glob import glob
from time import time

import pandas as pd
import numpy as np
from sklearn import tree
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, AdaBoostClassifier, AdaBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score, r2_score, silhouette_score, rand_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from progress_table import ProgressTable

from competitors.kmeanstree import KMeansTree
from ruletree import RuleTreeClassifier, RuleTreeClusterClassifier
from ruletree import RuleTreeCluster
from ruletree import RuleTreeRegressor
from benchmark.config import dataset_target_clu
from ruletree import RuleForestClassifier
from ruletree import RuleForestRegressor
from ruletree import RuleTreeAdaBoostClassifier
from ruletree import RuleTreeAdaBoostRegressor
from ruletree.stumps.classification.DecisionTreeStumpClassifier import DecisionTreeStumpClassifier


def test_clf(max_depth=4):
    datasets = glob("datasets/CLF/*.csv")
    table = ProgressTable(pbar_embedded=False, pbar_show_progress=True, pbar_show_percents=True,
                          pbar_show_throughput=False, num_decimal_places=3)

    for dataset in datasets:
        dataset_name = dataset.split("/")[-1].split("\\")[-1][:-4]
        table["dataset"] = dataset_name

        try:
            df = pd.read_csv(dataset)
            ct = ColumnTransformer([("cat", OneHotEncoder(), make_column_selector(dtype_include="object"))],
                                   remainder='passthrough', verbose_feature_names_out=False, sparse_threshold=0,
                                   n_jobs=1)

            clf_rule = RuleTreeClassifier(max_depth=max_depth)
            clf_sklearn = DecisionTreeClassifier(max_depth=max_depth)
            clf_forest_rule = RuleForestClassifier(max_depth=max_depth, n_estimators=100, n_jobs=1)
            clf_forest_sklearn = RandomForestClassifier(max_depth=max_depth, n_estimators=100, n_jobs=1)
            clf_adaboost_rule = RuleTreeAdaBoostClassifier(n_estimators=100)
            clf_adaboost_sklearn = AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=1), n_estimators=100)

            X = df.iloc[:, :-1].values
            y = df.iloc[:, -1].values
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
            df_onehot = pd.DataFrame(ct.fit_transform(df.iloc[:, :-1]), columns=ct.get_feature_names_out())
            X_onehot = df_onehot.to_numpy()
            X_train_onehot, X_test_onehot, _, _ = train_test_split(X_onehot, y, test_size=0.3, random_state=42,
                                                                   stratify=y)

            res = dict()
            for model, model_name in zip([  #clf_rule,
                #clf_sklearn,
                #clf_forest_rule,
                #clf_forest_sklearn,
                clf_adaboost_rule,
                clf_adaboost_sklearn
            ],
                    [  #"rule",
                        #"dt",
                        #"forest_rule",
                        #"forest_dt",
                        "adaboost_rule",
                        "adaboost_sklearn"
                    ]):
                start = time()
                if "rule" not in model_name:
                    model.fit(X_train_onehot, y_train)
                else:
                    model.fit(X_train, y_train)
                stop = time()

                pickle.dump(clf_rule, open("clf_rule.pkl", "wb"))
                clf_rule = pickle.load(open("clf_rule.pkl", "rb"))
                os.remove("clf_rule.pkl")

                if "rule" not in model_name:
                    f1 = f1_score(y_test, model.predict(X_test_onehot), average='weighted')
                    #print(classification_report(y_test, model.predict(X_test_onehot)))
                else:
                    f1 = f1_score(y_test, model.predict(X_test), average='weighted')
                    #print(classification_report(y_test, model.predict(X_test)))

                res[f"{model_name}_time"] = stop - start
                res[f"{model_name}_f1"] = f1

            table.update_from_dict(res)
        except Exception as e:
            table["error"] = str(e)
            #raise e

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
        dataset_name = dataset.split("/")[-1].split("\\")[-1][:-4]
        table["dataset"] = dataset_name

        try:
            df = pd.read_csv(dataset)
            ct = ColumnTransformer([("cat", OneHotEncoder(), make_column_selector(dtype_include="object"))],
                                   remainder='passthrough', verbose_feature_names_out=False, sparse_threshold=0,
                                   n_jobs=1)

            clf_rule = RuleTreeRegressor(max_depth=max_depth)
            clf_forest_rule = RuleForestRegressor(max_depth=max_depth, n_estimators=100, n_jobs=1)
            clf_sklearn = DecisionTreeRegressor(max_depth=max_depth)
            clf_forest_sklearn = RandomForestRegressor(max_depth=max_depth, n_estimators=100, n_jobs=1)
            clf_adaboost_rule = RuleTreeAdaBoostRegressor(n_estimators=100)
            clf_adaboost_sklearn = AdaBoostRegressor(estimator=DecisionTreeRegressor(max_depth=1), n_estimators=100)

            X = df.drop(columns=dataset_target_reg[dataset_name]).to_numpy()
            y = df[dataset_target_reg[dataset_name]].values
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
            df_onehot = pd.DataFrame(ct.fit_transform(df.drop(columns=dataset_target_reg[dataset_name])),
                                     columns=ct.get_feature_names_out())
            X_onehot = df_onehot.to_numpy()
            X_train_onehot, X_test_onehot, _, _ = train_test_split(X_onehot, y, test_size=0.3, random_state=42)

            res = dict()
            for model, model_name in zip([clf_rule,
                                          clf_sklearn,
                                          clf_forest_rule,
                                          clf_forest_sklearn,
                                          clf_adaboost_rule,
                                          clf_adaboost_sklearn
                                          ],
                                         ["rule",
                                          "dt",
                                          "forest_rule",
                                          "forest_dt",
                                          "adaboost_rule",
                                          "adaboost_sklearn"
                                          ]):
                start = time()
                if "rule" not in model_name:
                    model.fit(X_train_onehot, y_train)
                else:
                    model.fit(X_train, y_train)
                stop = time()

                pickle.dump(clf_rule, open("clf_rule.pkl", "wb"))
                clf_rule = pickle.load(open("clf_rule.pkl", "rb"))
                os.remove("clf_rule.pkl")

                if "rule" not in model_name:
                    r2 = r2_score(y_test, model.predict(X_test_onehot))
                else:
                    r2 = r2_score(y_test, model.predict(X_test))

                res[f"{model_name}_time"] = stop - start
                res[f"{model_name}_r2"] = r2

            table.update_from_dict(res)
        except Exception as e:
            #table["error"] = str(e)
            raise e

        table.next_row()


def test_clu(max_depth=4):
    datasets = glob("datasets/CLU/*.csv")
    table = ProgressTable(pbar_embedded=False, pbar_show_progress=True, pbar_show_percents=True,
                          pbar_show_throughput=False, num_decimal_places=3)

    for dataset in datasets:
        dataset_name = dataset.split("/")[-1].split("\\")[-1][:-4]
        table["dataset"] = dataset_name

        try:
            df = pd.read_csv(dataset)
            ct = ColumnTransformer([("cat", OneHotEncoder(), make_column_selector(dtype_include="object"))],
                                   remainder='passthrough', verbose_feature_names_out=False, sparse_threshold=0,
                                   n_jobs=1)

            clf_rule = RuleTreeCluster(max_depth=max_depth, prune_useless_leaves=True)
            clf_sklearn = KMeans(n_clusters=2 ** max_depth)

            df_onehot = pd.DataFrame(ct.fit_transform(df.iloc[:, :-1]), columns=ct.get_feature_names_out())
            X_onehot = df_onehot.to_numpy()

            start_rule = time()
            X = df.drop(columns=[dataset_target_clu[dataset_name]]).values
            y = df[dataset_target_clu[dataset_name]].values

            clf_rule.fit(X_onehot)
            end_rule = time()

            pickle.dump(clf_rule, open("clf_rule.pkl", "wb"))
            clf_rule = pickle.load(open("clf_rule.pkl", "rb"))
            os.remove("clf_rule.pkl")

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
        except Exception as e:
            table["error"] = str(e)

        table.next_row()


def test_clf_iris():
    df = pd.read_csv("datasets/CLF/iris.csv")
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    clf_rule = RuleTreeClassifier(max_depth=5)

    clf_rule.fit(X_train, y_train)

    f1_rule = f1_score(y_test, clf_rule.predict(X_test), average='weighted')

    pickle.dump(clf_rule, open("clf_rule.pkl", "wb"))
    clf_rule_load = pickle.load(open("clf_rule.pkl", "rb"))

    f1_rule_load = f1_score(y_test, clf_rule_load.predict(X_test), average='weighted')

    print(f"F1: {f1_rule}")
    print(f"F1: {f1_rule_load}")


def test_clc_iris():
    df = pd.read_csv("datasets/CLF/home.csv")
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    clf_rule = RuleTreeClusterClassifier()
    clf_comp = KMeansTree()

    clf_rule.fit(X_train, y_train)
    clf_comp.fit(X_train, y_train)

    f1_rule = f1_score(y_test, clf_rule.predict(X_test), average='weighted')
    f1_comp = f1_score(y_test, clf_rule.predict(X_test), average='weighted')

    print(f"F1: {f1_rule}")
    print(f"F1: {f1_comp}")


if __name__ == "__main__":
    test_clf()
    test_reg()
    #test_clu()

    #test_clf_iris()
    #test_clc_iris()
