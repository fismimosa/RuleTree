from glob import glob
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from tqdm.auto import tqdm
from sklearn.preprocessing import LabelEncoder

from glob import glob

from ruletree import RuleTree


def diff():
    enc = LabelEncoder()
    X_train = np.load("code/X_train.npy", allow_pickle=True)
    X_test = np.load("code/X_test.npy", allow_pickle=True)
    y_train = np.load("code/y_train.npy", allow_pickle=True)
    y_test = np.load("code/y_test.npy", allow_pickle=True)
    #y_train = enc.fit_transform(np.load("code/y_train.npy", allow_pickle=True))
    #y_test = enc.transform(np.load("code/y_test.npy", allow_pickle=True))

    RT = RuleTree(min_samples_leaf=0.01, max_depth=2, min_samples_split=0.01, prune_useless_leaves=True)
    DT = DecisionTreeClassifier(min_samples_leaf=0.01, max_depth=2, min_samples_split=0.01)

    RT.fit(X_train, y_train)
    DT.fit(X_train, y_train)

    print("RULE TREE:")
    print(RT.print_tree())
    print(classification_report(y_test, RT.predict(X_test)))

    print("\r\nDECISION TREE:")
    print(classification_report(y_test, DT.predict(X_test)))
    print(tree.export_text(DT))



def CLF():
    eval_measures_clf = [
        'accuracy',
        'average_precision_macro', 'average_precision_micro', 'average_precision_weighted',
        'balanced_accuracy',
        'f1_macro', 'f1_micro', 'f1_score', 'f1_weighted',
        'fit_time',
        'precision_macro', 'precision_micro', 'precision_score', 'precision_weighted',
        'predict_time',
        'recall_macro', 'recall_micro', 'recall_score', 'recall_weighted',
        'roc_macro', 'roc_micro', 'roc_weighted',
    ]
    rt_params = [
        'max_depth',
        'max_nbr_nodes',
        'min_samples_split',
        'min_samples_leaf',
        'allow_oblique_splits',
        'force_oblique_splits',
        'max_oblique_features',
        'prune_useless_leaves',
        'n_components',
        'bic_eps',
        'clus_impurity',
        'clf_impurity',
        'reg_impurity',
        'feature_names',
        'exclude_split_feature_from_reduction',
        'precision',
        'cat_precision',
        'n_jobs',
        'random_state',
    ]
    dt_params = [
        'criterion',
        'splitter',
        'max_depth',
        'min_samples_split',
        'min_samples_leaf',
        'min_weight_fraction_leaf',
        'max_features',
        'max_leaf_nodes',
        'min_impurity_decrease',
        'class_weight',
        'ccp_alpha',
        'random_state',
    ]
    columns = [
        'dataset',
        'expid1',
        'expid2',
        'expid3',
        'method',
        'model_type',
        'numerical_indices',
        'numerical_scaler',
        'one_hot_encode_cat',
        'repeated_holdout_id',
        'task'
    ]
    common_params = list(set(dt_params) & set(rt_params) - {'random_state'})

    for file in sorted(glob("results/CLF_results_*.csv")):
        print(file, end=" ")
        skip = False
        for to_exclude in ["adult", "compas", "bank", "fico", "german_credit", "titanic"]:
            if to_exclude in file:
                print("SKIP")
                skip = True
        if skip:
            continue

        df = pd.read_csv(file, low_memory=False)
        df = df[df.allow_oblique_splits.isin(['-1', 'False']) & df.force_oblique_splits.isin(['-1', 'False'])]

        dfg = df[['method', 'expid2'] + common_params + eval_measures_clf].groupby(
            ['method', 'expid2']).mean().reset_index()
        dfs = dfg[['method'] + common_params + eval_measures_clf].sort_values(
            ['method'] + common_params + ['accuracy'], ascending=[False, True, True, True, False])
        dfs = dfs[['method'] + common_params + eval_measures_clf].drop_duplicates(
            ['method'] + common_params, keep='first')

        df_rt = dfs[dfs['method'] == 'RT'].drop(['method'], axis=1)
        df_dt = dfs[dfs['method'] == 'DT'].drop(['method'], axis=1)

        dfm = pd.merge(df_rt, df_dt, how='outer', left_on=common_params, right_on=common_params)
        df_delta = df_rt.copy(deep=True)
        gap_c = len(common_params)
        gap_m = len(eval_measures_clf)
        for i in range(len(dfm)):
            for j in range(len(eval_measures_clf)):
                delta = dfm.iloc[i, gap_c + j] - dfm.iloc[i, gap_c + gap_m + j]
                df_delta.iloc[i, gap_c + j] = delta

        delta_diff_zero_list = list()
        for c in eval_measures_clf:
            if df_delta[c].sum() != 0:
                delta_diff_zero_list.append(df_delta.loc[df_delta[c] != 0][common_params + eval_measures_clf])

        df_check = pd.concat(delta_diff_zero_list)

        if df_check.mean()['accuracy'] > 0 and df_check.mean()['f1_macro'] > 0:
            print('OK')
        else:
            print(df_check.mean()['accuracy'], df_check.mean()['f1_macro'])

        df_check.to_csv(file.replace("_results_", "_delta_"), index=None)

if __name__ == "__main__":
    #diff()
    CLF()