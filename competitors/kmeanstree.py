
import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from threadpoolctl import threadpool_limits


def calculate_mode(x):
    vals, counts = np.unique(x, return_counts=True)
    idx = np.argmax(counts)
    return vals[idx]


class KMeansTree:

    def __init__(self,
                 labels_as_tree_leaves: bool = True,
                 max_nbr_values_cat: int = 4,

                 n_clusters: int = 8,
                 min_samples_leaf: int = 3,
                 min_samples_split: int = 5,
                 clf_impurity: str = 'gini',

                 init: str = 'k-means++',
                 n_init: int = 10,
                 max_iter: int = 300,
                 tol: float = 0.0001,
                 copy_x: bool = True,

                 max_depth: int = None,
                 algorithm: str = 'lloyd',
                 splitter: str = 'best',
                 min_weight_fraction_leaf: int = 0.0,
                 max_features: int = None,
                 max_leaf_nodes: int = None,
                 min_impurity_decrease: float = 0.0,
                 class_weight: dict = None,
                 ccp_alpha: float = 0.0,

                 n_jobs = 1,

                 verbose: int = 0,
                 random_state: int = None,
                 ):

        self.accuracy_ = -1
        self.labels_as_tree_leaves = labels_as_tree_leaves
        self.max_nbr_values_cat = max_nbr_values_cat

        self.n_jobs = n_jobs

        self.kmeans = KMeans(n_clusters=n_clusters,
                             init=init,
                             n_init=n_init,
                             max_iter=max_iter,
                             tol=tol,
                             verbose=verbose,
                             random_state=random_state,
                             copy_x=copy_x,
                             algorithm=algorithm)

        if max_depth is None:
            max_depth = np.round(np.log2(n_clusters)).astype(int)

        self.dt = DecisionTreeClassifier(criterion=clf_impurity,
                                         splitter=splitter,
                                         max_depth=max_depth,
                                         min_samples_split=min_samples_split,
                                         min_samples_leaf=min_samples_leaf,
                                         min_weight_fraction_leaf=min_weight_fraction_leaf,
                                         max_features=max_features,
                                         random_state=random_state,
                                         max_leaf_nodes=max_leaf_nodes,
                                         min_impurity_decrease=min_impurity_decrease,
                                         class_weight=class_weight,
                                         ccp_alpha=ccp_alpha,
                                         )

        self.labels_ = None
        self.clu_for_reg = None
        self.clu_for_clf = None
        self.leaf_val = None

    def fit(self, X, y=None):

        if y is not None:
            if len(np.unique(y)) >= self.max_nbr_values_cat:  # infer is numerical
                self.clu_for_reg = True
            else:  # infer y is categorical
                self.clu_for_clf = True

        #with threadpool_limits(limits=self.n_jobs):
        self.kmeans.fit(X)
        self.dt.fit(X, self.kmeans.labels_)
        self.accuracy_ = self.dt.score(X, self.kmeans.labels_)
        if self.labels_as_tree_leaves:
            self.labels_ = self.dt.apply(X)
        else:
            self.labels_ = self.dt.predict(X)

        if self.clu_for_clf or self.clu_for_reg:
            leaves_id = self.dt.apply(X)
            self.leaf_val = dict()
            for l in np.unique(leaves_id):
                pred = y[leaves_id == l]
                if len(pred) > 0:
                    if self.clu_for_clf:
                        self.leaf_val[l] = calculate_mode(pred)
                    else:
                        self.leaf_val[l] = np.mean(pred)

    def predict(self, X):
        #with threadpool_limits(limits=self.n_jobs):

        if self.clu_for_clf or self.clu_for_reg:
            leaves_id = self.dt.apply(X)
            pred = [self.leaf_val[l] for l in leaves_id]
            return pred

        if self.labels_as_tree_leaves:
            return self.dt.apply(X)
        else:
            return self.dt.predict(X)

    # TODO predict proba


from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import export_text
from sklearn.metrics import accuracy_score, normalized_mutual_info_score, mean_absolute_percentage_error, r2_score

from sklearn.preprocessing import StandardScaler
from RuleTree.utils.data_utils import prepare_data, preprocessing


def main():
    iris = load_iris()
    X = iris.data
    y = iris.target
    # idx = [100,101,102,103,104,105,106,107,108,109,110,111,112,113,114,115,116,117,118,
    #        51,52,53,54,55,56,57,58,59,60,61,62,63]
    # X = X[idx][:,[0,1]]
    # y = y[idx]
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.05, random_state=0)
    # print(np.unique(y_train, return_counts=True))

    max_nbr_values_cat = 4
    max_nbr_values = np.inf
    one_hot_encode_cat = True
    categorical_indices = None
    numerical_indices = None
    numerical_scaler = StandardScaler()

    kdt = KMeansTree(
        labels_as_tree_leaves=True,
        max_nbr_values_cat=max_nbr_values_cat,
        n_clusters=6,
        # min_samples_leaf=3,
        # min_samples_split=5,
        max_depth=None
    )
    y_train_o = y_train[::]
    y_test_o = y_test[::]
    y_train = np.log(X_train[:, 0] * X_train[:, 1])
    y_test = np.log(X_test[:, 0] * X_test[:, 1])
    X_train = pd.concat([
        pd.DataFrame(data=X_train[:,:]),
        pd.DataFrame(data=y_train_o.reshape(-1, 1).astype(str))],
        axis=1).values
    X_test = pd.concat([
        pd.DataFrame(data=X_test[:, :]),
        pd.DataFrame(data=y_test_o.reshape(-1, 1).astype(str))],
        axis=1).values

    feature_names_r = np.array(['X_%s' % i for i in range(X_train.shape[1])])

    res = prepare_data(X_train, max_nbr_values, max_nbr_values_cat,
                       feature_names_r, one_hot_encode_cat, categorical_indices,
                       numerical_indices, numerical_scaler)
    X_train = res[0]
    print(X_train.shape, 'AAAAA')
    feature_values_r, is_cat_feat_r, feature_values, is_cat_feat, data_encoder, feature_names = res[1]
    maps = res[2]
    kdt.fit(X_train, y_train)
    # kdt.fit(X_train)

    print('KMeansTree')
    print(export_text(kdt.dt))
    print('')

    X_test = preprocessing(X_test, feature_names_r, is_cat_feat, data_encoder, numerical_scaler)

    y_pred = kdt.predict(X_test)
    print(y_pred)

    if len(np.unique(y_train)) >= max_nbr_values_cat:  # infer is numerical
        print('R2', r2_score(y_test, y_pred))
        print('MAPE', mean_absolute_percentage_error(y_test, y_pred))
    else:  # infer it is categorical
        print('Accuracy', accuracy_score(y_test, y_pred))
    print('NMI', normalized_mutual_info_score(y_test, y_pred))
    print('')


if __name__ == "__main__":
    main()