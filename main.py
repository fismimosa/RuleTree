import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris
from sklearn.metrics import r2_score, mean_absolute_percentage_error, accuracy_score, normalized_mutual_info_score, \
    silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

from RuleTree.RuleTree import RuleTree
from RuleTree import RuleTreeClassifier, RuleTreeRegressor, RuleTreeClustering


def test_CLF():
    iris = load_iris()
    X = iris.data
    y = np.array(list(map(lambda x: iris.target_names[x], iris.target)))

    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=0)

    max_nbr_values_cat = 4
    rt = RuleTreeClassifier(
        # max_depth=4,
        # min_samples_leaf=1,
        # min_samples_split=2,
        max_nbr_nodes=7,
        # clf_impurity='entropy',
        # feature_names=['Sex', 'Lies', 'Cookies'],
        # feature_names=iris.feature_names,
        # allow_oblique_splits=True,
        # force_oblique_splits=True,
        # max_nbr_values=4,
        # max_nbr_nodes=32,
        prune_useless_leaves=True,
        # max_nbr_nodes=5,
        bic_eps=0.5,
        # max_nbr_values=20,
        max_nbr_values_cat=max_nbr_values_cat,
        one_hot_encode_cat=True,
        # categorical_indices=[4],
        # numerical_indices=[1],
        numerical_scaler=StandardScaler(),
        # exclude_split_feature_from_reduction=True,
        random_state=0,
        verbose=True
    )

    print(rt.bic_eps)
    rt.set_params(**{'bic_eps': 0.25})
    print(rt.bic_eps)
    rt.fit(X_train, y_train)
    # rt.fit(X_train)
    print(rt.predict(X_test))
    print(rt.predict(X_test))

    print('RuleTree')
    print(rt.print_tree())
    print('')

    y_pred, leaves, rules = rt.predict(X_test, get_leaf=True, get_rule=True)
    print(y_pred, '<<<<<<<')
    print(y_test)
    print(leaves)
    print(rt.rules_s_)
    print(rt.predict_proba(X_test), 'PPPPPPP')
    # print(X_test[2])

    if len(np.unique(y_train)) >= max_nbr_values_cat:  # infer is numerical
        print('R2', r2_score(y_test, y_pred))
        print('MAPE', mean_absolute_percentage_error(y_test, y_pred))
    else:  # infer it is categorical
        print('Accuracy', accuracy_score(y_test, y_pred))
    print('NMI', normalized_mutual_info_score(y_test, y_pred))
    print('')
    for r in rules:
        print(r)
    print('')

    print(rt.are_rules_verified(X_test, count_cond=False, relative_count=False))
    print('')

    for k in rt.medoid_dict_:
        print(k, rt.medoid_dict_[k])

    print('')

    for k in rt.rules_s_:
        print(k, rt.rules_s_[k], rt.medoid_dict_[k])

    print('')

    if len(np.unique(y_train)) >= max_nbr_values_cat:  # infer is numerical
        if rt.task_medoid_dict_ is not None:
            for k in rt.task_medoid_dict_:
                print(k, rt.task_medoid_dict_[k])

            print('')

            for k in rt.rules_s_:
                print(k, rt.rules_s_[k], rt.task_medoid_dict_[k])

            print('')
    else:
        if rt.task_medoid_dict_ is not None:
            for k in rt.task_medoid_dict_:
                print(k, rt.task_medoid_dict_[k])

            print('')

            for k in rt.rules_s_:
                print(k, rt.rules_s_[k], rt.task_medoid_dict_[k])

            print('')

    dist = rt.transform(X_train, metric='euclidean', include_descriptive_pivot=True, include_target_pivot=True,
                        only_leaf_pivot=True)
    dist_test = rt.transform(X_test, metric='euclidean', include_descriptive_pivot=True, include_target_pivot=True,
                             only_leaf_pivot=True)

    rtp = RuleTreeClassifier(max_depth=4)
    rtp.fit(dist, y_train)
    print(RuleTree.print_tree(rtp))

    print('Accuracy', accuracy_score(y_test, rtp.predict(dist_test)))

def test_REG():
    iris = load_iris()
    X = iris.data[:, :-1]
    y = iris.data[:, -1]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    max_nbr_values_cat = 4
    rt = RuleTreeRegressor(
        # max_depth=4,
        # min_samples_leaf=1,
        # min_samples_split=2,
        max_nbr_nodes=7,
        # clf_impurity='entropy',
        # feature_names=['Sex', 'Lies', 'Cookies'],
        # feature_names=iris.feature_names,
        # allow_oblique_splits=True,
        # force_oblique_splits=True,
        # max_nbr_values=4,
        # max_nbr_nodes=32,
        prune_useless_leaves=True,
        # max_nbr_nodes=5,
        bic_eps=0.5,
        # max_nbr_values=20,
        max_nbr_values_cat=max_nbr_values_cat,
        one_hot_encode_cat=True,
        # categorical_indices=[4],
        # numerical_indices=[1],
        numerical_scaler=StandardScaler(),
        # exclude_split_feature_from_reduction=True,
        random_state=0,
        verbose=True
    )

    print(rt.bic_eps)
    rt.set_params(**{'bic_eps': 0.25})
    print(rt.bic_eps)
    rt.fit(X_train, y_train)
    # rt.fit(X_train)
    print(rt.predict(X_test))
    print(rt.predict(X_test))

    print('RuleTree')
    print(rt.print_tree())
    print('')

    y_pred, leaves, rules = rt.predict(X_test, get_leaf=True, get_rule=True)
    print(y_pred, '<<<<<<<')
    print(y_test)
    print(leaves)
    print(rt.rules_s_)
    print(rt.predict_proba(X_test), 'PPPPPPP')
    # print(X_test[2])

    if len(np.unique(y_train)) >= max_nbr_values_cat:  # infer is numerical
        print('R2', r2_score(y_test, y_pred))
        print('MAPE', mean_absolute_percentage_error(y_test, y_pred))
    else:  # infer it is categorical
        print('Accuracy', accuracy_score(y_test, y_pred))
    print('NMI', normalized_mutual_info_score(y_test, y_pred))
    print('')
    for r in rules:
        print(r)
    print('')

    print(rt.are_rules_verified(X_test, count_cond=False, relative_count=False))
    print('')

    for k in rt.medoid_dict_:
        print(k, rt.medoid_dict_[k])

    print('')

    for k in rt.rules_s_:
        print(k, rt.rules_s_[k], rt.medoid_dict_[k])

    print('')

    if len(np.unique(y_train)) >= max_nbr_values_cat:  # infer is numerical
        if rt.task_medoid_dict_ is not None:
            for k in rt.task_medoid_dict_:
                print(k, rt.task_medoid_dict_[k])

            print('')

            for k in rt.rules_s_:
                print(k, rt.rules_s_[k], rt.task_medoid_dict_[k])

            print('')
    else:
        if rt.task_medoid_dict_ is not None:
            for k in rt.task_medoid_dict_:
                print(k, rt.task_medoid_dict_[k])

            print('')

            for k in rt.rules_s_:
                print(k, rt.rules_s_[k], rt.task_medoid_dict_[k])

            print('')


    dtr = DecisionTreeRegressor().fit(X_train, y_train)
    print('R2', r2_score(y_test, dtr.predict(X_test)))
    print('MAPE', mean_absolute_percentage_error(y_test, dtr.predict(X_test)))

def test_CLU():
    iris = load_iris()
    X = iris.data

    max_nbr_values_cat = 4
    rt = RuleTreeClustering(
        # max_depth=4,
        # min_samples_leaf=1,
        # min_samples_split=2,
        max_nbr_nodes=7,
        # clf_impurity='entropy',
        # feature_names=['Sex', 'Lies', 'Cookies'],
        # feature_names=iris.feature_names,
        # allow_oblique_splits=True,
        # force_oblique_splits=True,
        # max_nbr_values=4,
        # max_nbr_nodes=32,
        prune_useless_leaves=True,
        # max_nbr_nodes=5,
        bic_eps=0.5,
        # max_nbr_values=20,
        max_nbr_values_cat=max_nbr_values_cat,
        one_hot_encode_cat=True,
        # categorical_indices=[4],
        # numerical_indices=[1],
        numerical_scaler=StandardScaler(),
        # exclude_split_feature_from_reduction=True,
        random_state=0,
        verbose=True
    )

    print(rt.bic_eps)
    rt.set_params(**{'bic_eps': 0.25})
    print(rt.bic_eps)
    rt.fit(X)
    # rt.fit(X_train)

    print('RuleTree')
    print(rt.print_tree())
    print('')

    y_pred, leaves, rules = rt.predict(X, get_leaf=True, get_rule=True)
    # print(X_test[2])

    print(silhouette_score(X, y_pred))

    kmeans = KMeans(n_clusters=3, random_state=0)
    kmeans.fit(X)
    print(silhouette_score(X, kmeans.predict(X)))

if __name__ == '__main__':
    #test_CLF()
    #test_REG()
    test_CLU()

    # TODO fare benchmarking
    # add medoide discriminativo??? direidi no
    # TODO dopo benchmakring sfruttare in notebook clustering per class 2
    #  per a) clustering iniziale e b) modello predittivo su coppie di cluster