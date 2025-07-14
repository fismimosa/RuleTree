import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

from RuleTree import RuleTreeClassifier
from sklearn.ensemble import BaggingClassifier
from RuleTree.utils.dict_utils import make_complete_rule_tree, get_tree_depth
import numpy as np


def get_mask(node: dict):
    mask = np.zeros((1, node['n_features']))

    if 'DecisionTreeStumpClassifier' in node['stump_type']:
        mask[0, node['feature_idx']] = 1
    if 'PivotTreeStumpClassifier' in node['stump_type']:
        mask[:, :] = 1
    if 'PartialPivotTreeStumpClassifier' in node['stump_type']:
        mask = (~np.isnan(np.array(node['shapelets']))).astype(float)

    return mask


def create_matrix(nodes: dict, max_depth: int, curr_node_idx='R', idx=0, prediction_classes_=None, thr_matrix=None,
                  mask=None, pred_matrix=None):
    if curr_node_idx not in nodes:
        return thr_matrix, mask, pred_matrix

    n_features = nodes[curr_node_idx]['n_features']
    if prediction_classes_ is None:
        prediction_classes_ = nodes[curr_node_idx]['prediction_classes_']
    print(curr_node_idx, n_features, prediction_classes_, nodes[curr_node_idx]['is_leaf'],
          nodes[curr_node_idx]['prediction_probability'], sep='\t')
    predict_proba_dict = dict(
        zip(nodes[curr_node_idx]['prediction_classes_'], nodes[curr_node_idx]['prediction_probability']))

    if thr_matrix is None:
        thr_matrix = -np.ones((2 ** max_depth - 1, 1))
        mask = np.zeros((2 ** max_depth - 1, n_features))
        pred_matrix = np.zeros(((2 ** max_depth - 1, len(prediction_classes_))))

    print(idx)

    if not nodes[curr_node_idx]['is_leaf']:
        thr_matrix[idx] = nodes[curr_node_idx]['threshold']
        mask[idx] = get_mask(nodes[curr_node_idx])

    for i, el in enumerate(prediction_classes_):
        pred_matrix[idx, i] = predict_proba_dict[el]

    thr_matrix, mask, pred_matrix = create_matrix(nodes, max_depth, curr_node_idx + 'l', 2 * idx + 1,
                                                  prediction_classes_, thr_matrix, mask, pred_matrix)
    thr_matrix, mask, pred_matrix = create_matrix(nodes, max_depth, curr_node_idx + 'r', 2 * idx + 2,
                                                  prediction_classes_, thr_matrix, mask, pred_matrix)

    return thr_matrix, mask, pred_matrix




if __name__ == "__main__":
    from sklearn.preprocessing import MinMaxScaler

    from RuleTree.stumps.classification import PartialPivotTreeStumpClassifier, DecisionTreeStumpClassifier, \
        PivotTreeStumpClassifier
    from sklearn import datasets

    iris = datasets.load_iris()
    iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    iris_df['target'] = iris.target

    minmax_scaler = MinMaxScaler()
    X = minmax_scaler.fit_transform(iris_df.drop('target', axis=1).values)
    y = iris_df['target'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    rf = BaggingClassifier(
        RuleTreeClassifier(
            max_depth=5,
            base_stumps=[
                PartialPivotTreeStumpClassifier(n_shapelets=np.inf, n_features_strategy='all', selection='all',
                                                n_jobs=10, random_state=42),
                # DecisionTreeStumpClassifier(max_depth=1, random_state=42),
                # PivotTreeStumpClassifier(max_depth=1, random_state=42),
            ],
            random_state=42),
        n_estimators=3, random_state=42, n_jobs=1
    )

    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    print("F1 Score:", f1_score(y_test, y_pred, average='macro'))

    rf.estimators_[0].to_dict("dict")
    dizionario = make_complete_rule_tree("dict")

    nodes = {el['node_id']: el for el in dizionario['nodes']}

    create_matrix(nodes=nodes, max_depth=get_tree_depth(dizionario) + 1)

