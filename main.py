import numpy as np
from sklearn.datasets import load_iris
from sklearn.metrics import r2_score, mean_absolute_percentage_error, accuracy_score, normalized_mutual_info_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from RuleTree.RuleTree import RuleTree
from RuleTree import RuleTreeClassifier

if __name__ == '__main__':
    iris = load_iris()
    X = iris.data
    y = iris.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.05, random_state=0)

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
                  clus_impurity='r2',
                  # categorical_indices=[4],
                  # numerical_indices=[1],
                  numerical_scaler=StandardScaler(),
                  # exclude_split_feature_from_reduction=True,
                  random_state=0,
                  verbose=True
                  )

    y_train_o = y_train[::]
    y_test_o = y_test[::]
    # y_train = np.log(X_train[:, 0] * X_train[:, 1])
    # y_test = np.log(X_test[:, 0] * X_test[:, 1])
    # X_train = pd.concat([
    #     pd.DataFrame(data=X_train[:,:]),
    #     pd.DataFrame(data=y_train_o.reshape(-1, 1).astype(str))],
    #     axis=1).values
    # X_test = pd.concat([
    #     pd.DataFrame(data=X_test[:, :]),
    #     pd.DataFrame(data=y_test_o.reshape(-1, 1).astype(str))],
    #     axis=1).values
    print(rt.bic_eps)
    rt.set_params(**{'bic_eps': 0.25})
    print(rt.bic_eps)
    rt.fit(X_train, y_train)
    # rt.fit(X_train)
    print(rt.predict_proba(X_test))
    print(rt.predict(X_test))

    # print(rt.feature_values[4])

    # print(rt.root_.node_r.node_r.__dict__)
    # print(y_train[rt.root_.node_r.node_r.idx], st.mode(y_train[rt.root_.node_r.node_r.idx]))
    print('RuleTree')
    print(rt.print_tree())
    print('')

    # for r in rt.rules_s_:
    #     print(rt.rules_s_[r])
    # print('')

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

    dist = rt.transform(X_train, metric='euclidean', include_descriptive_pivot=True, include_target_pivot=True, only_leaf_pivot=True)
    dist_test = rt.transform(X_test, metric='euclidean', include_descriptive_pivot=True, include_target_pivot=True,
                             only_leaf_pivot=True)
    # print(dist)
    # print(dist.shape)
    # print(y_test)

    rtp = RuleTree(model_type='clf', max_depth=4)
    rtp.fit(dist, y_train)
    print(RuleTree.print_tree(rtp))

    print('Accuracy', accuracy_score(y_test, rtp.predict(dist_test)))

    # print(X_test[2][[1,2]])
    # print(rt.numerical_scaler.transform(X_test)[2][[1,2]])
    # # print(rt.numerical_scaler.mean_)
    # # print(rt.numerical_scaler.scale_)
    # # print(rt.rules_[1])
    # # print(rt.rules_[1][0][0], rt.rules_[1][0][5])
    #
    # for r in rt.rules_s_:
    #     print(r, rt.rules_s_[r])

    # print('')
    # print(rt.numerical_scaler.transform(X_test)[2][rt.rules_[1][0][0]], 'record normalized')
    # print(rt.rules_[1][0][5], 'coef originali')
    # print(rt.numerical_scaler.transform(X_test)[2][rt.rules_[1][0][0]] * rt.rules_[1][0][5], 'record norm * coef originali')
    # print(np.sum(rt.numerical_scaler.transform(X_test)[2][rt.rules_[1][0][0]] * rt.rules_[1][0][5]), 'sum')

    # print('')
    # print(X_test[2][rt.rules_[1][0][0]], 'record originale')
    # fake_x = np.zeros(X_test.shape[1])
    # fake_x[rt.rules_[1][0][0]] = rt.rules_[1][0][5]
    # X_to_invert = np.array([fake_x])
    # X_to_invert = rt.numerical_scaler.transform(X_to_invert)[0, rt.rules_[1][0][0]]
    # print(X_to_invert, 'coef normalized')
    # print(X_test[2][rt.rules_[1][0][0]] * X_to_invert, 'record originale * coef normalized')
    # print(np.sum(X_test[2][rt.rules_[1][0][0]] * X_to_invert), 'sum')

    # TODO fare benchmarking
    # add medoide discriminativo??? direidi no
    # TODO dopo benchmakring sfruttare in notebook clustering per class 2
    #  per a) clustering iniziale e b) modello predittivo su coppie di cluster

    # dataset classification: iris, adult, compas, ionosphere, titanic, german, diabetes, home, wine, bank
    # dataset regression:
    # parkinson: https://archive.ics.uci.edu/dataset/189/parkinsons+telemonitoring
    # abalone: https://archive.ics.uci.edu/dataset/1/abalone
    # wine
    # auction verification: https://archive.ics.uci.edu/dataset/713/auction+verification
    # intrusion: https://archive.ics.uci.edu/dataset/715/lt+fs+id+intrusion+detection+in+wsns
    # metamaterial: https://archive.ics.uci.edu/dataset/692/2d+elastodynamic+metamaterials
    # liver disorder: https://archive.ics.uci.edu/dataset/60/liver+disorders
    # boston: https://www.kaggle.com/datasets/fedesoriano/the-boston-houseprice-data
    # carprice: https://www.kaggle.com/datasets/hellbuoy/car-price-prediction
    # medical cost: https://www.kaggle.com/datasets/mirichoi0218/insurance
    # student performance: https://www.kaggle.com/datasets/nikhil7280/student-performance-multiple-linear-regression
    # dataset clustering puri, quelli di partree
    # dataset clustering per classifcazione e regeresisone, quelli sopra

    # competior: clf e reg, dt sklearn e knn
    # clustering: kmeans, kmeans+dt
    # pivot: random, kmeans/kmedoids, kmeans/kmedoids su classi + knn o dt