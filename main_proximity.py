import pandas as pd
import scipy
import numpy as np
from graphviz import Source
from sklearn.metrics import classification_report

from ruletree import RuleTreeClassifier
from ruletree.stumps.classification.ProximityTreeStumpClassifier import ProximityTreeStumpClassifier
from ruletree.stumps.classification.ShapeletTreeStumpClassifier import ShapeletTreeStumpClassifier

if __name__ == "__main__":
    df_train = pd.DataFrame(scipy.io.arff.loadarff('ruletree/utils/shapelet_transform/test_dataset/CBF/CBF_TRAIN.arff')[0])
    df_test = pd.DataFrame(scipy.io.arff.loadarff('ruletree/utils/shapelet_transform/test_dataset/CBF/CBF_TEST.arff')[0])
    df_train.target = df_train.target.astype(int)
    df_test.target = df_test.target.astype(int)

    X_train = df_train.drop(columns=['target']).values.reshape((-1, 1, 128))
    X_test = df_test.drop(columns=['target']).values.reshape((-1, 1, 128))
    y_train = df_train['target'].values
    y_test = df_test['target'].values

    dt = RuleTreeClassifier(
        max_depth=5,
        base_stumps=[
            ProximityTreeStumpClassifier(selection='cluster'),
            ShapeletTreeStumpClassifier(selection='mi_clf'),
        ]
    )

    dt.fit(X_train, y_train)
    y_pred = dt.predict(X_test)
    print(classification_report(y_test, y_pred))
    dt.export_graphviz()


    dt.to_dict("dizionario_shapelets.json")
    dt2 = dt.from_dict("dizionario_shapelets.json")

    y_pred = dt2.predict(X_test)
    print(classification_report(y_test, y_pred))
    dt2.root.simplify()
    dt2.export_graphviz()