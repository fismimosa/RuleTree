import pandas as pd
from sklearn.datasets import load_iris
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

from RuleTree import RuleTreeClassifier
from RuleTree.stumps.classification import DecisionTreeStumpClassifier
from RuleTree.stumps.classification.PartialPivotTreeStumpClassifier import PartialPivotTreeStumpClassifier
from RuleTree.stumps.classification.PartialProximityTreeStumpClassifier import PartialProximityTreeStumpClassifier

##previous stumps

#oblique decision tree
from RuleTree.stumps.classification.ObliqueDecisionTreeStumpClassifier import ObliqueDecisionTreeStumpClassifier
#pivot tree
from RuleTree.stumps.classification.PivotTreeStumpClassifier import PivotTreeStumpClassifier
#oblique pivot tree
from RuleTree.stumps.classification.ObliquePivotTreeStumpClassifier import ObliquePivotTreeStumpClassifier

#proximity pivot tree
from RuleTree.stumps.classification.MultiplePivotTreeStumpClassifier import MultiplePivotTreeStumpClassifier
#oblique proximity pivot tree
from RuleTree.stumps.classification.MultipleObliquePivotTreeStumpClassifier import MultipleObliquePivotTreeStumpClassifier




if __name__ == "__main__":
    for random_state in range(10):
        stumps = [
            PartialPivotTreeStumpClassifier(n_shapelets=10, max_n_features='all', n_jobs=10, random_state=random_state,
                                            selection='random'),
            PartialProximityTreeStumpClassifier(n_shapelets=10, max_n_features='all', n_jobs=10, random_state=random_state,
                                                selection='random'),
            DecisionTreeStumpClassifier(max_depth=1, random_state=random_state),
            ObliqueDecisionTreeStumpClassifier(max_depth=1, random_state=random_state,
                                               oblique_split_type='householder',
                                               pca=None,
                                               max_oblique_features=2,
                                               tau=1e-4),
            PivotTreeStumpClassifier(max_depth=1, random_state=random_state),
            MultiplePivotTreeStumpClassifier(max_depth=1, random_state=random_state),
            ObliquePivotTreeStumpClassifier(max_depth=1, random_state=random_state,
                                            oblique_split_type='householder',
                                            pca=None,
                                            max_oblique_features=2,
                                            tau=1e-4),
            MultipleObliquePivotTreeStumpClassifier(max_depth=1, random_state=random_state,
                                                    oblique_split_type='householder',
                                                    pca=None,
                                                    max_oblique_features=2,
                                                    tau=1e-4),
        ]

        df = pd.read_csv("datasets/CLF/iris.csv")
        X = df.iloc[:, :-1].values
        y = df.iloc[:, -1].values
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

        rt = RuleTreeClassifier(
            base_stumps=stumps,
            max_depth=5,
            stump_selection='best',
            random_state=42,
            distance_measure='euclidean' #added here metric for case-based splits
        )
        rt.fit(X_train, y_train)

        y_pred = rt.predict(X_test)

        print(classification_report(y_test, y_pred))
