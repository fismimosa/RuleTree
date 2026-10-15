Evaluating the models
=====================

:mod:`RuleTree.evaluation` computes the usual metrics of a model in one call, each as a dictionary
``{metric name: value}``, so that the results of several models can be collected in a table. The
metrics are the ones of :mod:`sklearn.metrics`.


Classification and regression
-----------------------------

``evaluate_classification`` takes the true and the predicted classes and, optionally, the class
probabilities (with their order, ``classes_``), for ROC AUC and average precision:

.. code-block:: python

   from sklearn.datasets import load_diabetes, load_iris
   from sklearn.model_selection import train_test_split

   from RuleTree import RuleTreeClassifier, RuleTreeRegressor
   from RuleTree.evaluation import evaluate_classification, evaluate_regression

   X, y = load_iris(return_X_y=True)
   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)
   model = RuleTreeClassifier(max_depth=3, random_state=0).fit(X_train, y_train)

   scores = evaluate_classification(y_test, model.predict(X_test), model.predict_proba(X_test),
                                    labels=model.classes_)
   print(sorted(scores))
   print({name: round(scores[name], 3) for name in ["accuracy", "f1_macro", "roc_macro"]})

.. code-block:: text
   :caption: Output

   ['accuracy', 'average_precision_macro', 'average_precision_micro', 'average_precision_weighted', 'balanced_accuracy', 'f1_macro', 'f1_micro', 'f1_score', 'f1_weighted', 'precision_macro', 'precision_micro', 'precision_score', 'precision_weighted', 'recall_macro', 'recall_micro', 'recall_score', 'recall_weighted', 'roc_macro', 'roc_micro', 'roc_weighted']
   {'accuracy': 0.956, 'f1_macro': 0.955, 'roc_macro': 1.0}

For regression, ``evaluate_regression`` gives the errors and the explained variance:

.. code-block:: python

   X_reg, y_reg = load_diabetes(return_X_y=True)
   X_reg_train, X_reg_test, y_reg_train, y_reg_test = train_test_split(X_reg, y_reg, test_size=0.3, random_state=0)
   regressor = RuleTreeRegressor(max_depth=3, random_state=0).fit(X_reg_train, y_reg_train)

   scores = evaluate_regression(y_reg_test, regressor.predict(X_reg_test))
   print({name: round(scores[name], 3) for name in ["r2", "mean_absolute_error"]})

.. code-block:: text
   :caption: Output

   {'r2': 0.188, 'mean_absolute_error': 50.408}


Clustering
----------

``evaluate_clustering`` compares the clusters with known classes, ``evaluate_clustering_internal``
measures their quality from the data alone (silhouette, Calinski-Harabasz, Davies-Bouldin):

.. code-block:: python

   from sklearn.datasets import make_blobs

   from RuleTree import RuleTreeCluster
   from RuleTree.evaluation import evaluate_clustering, evaluate_clustering_internal

   X_blobs, blobs = make_blobs(n_samples=300, centers=3, cluster_std=0.5, random_state=0)
   labels = RuleTreeCluster(max_depth=3, random_state=0).fit_predict(X_blobs)

   print("adjusted Rand index:", round(evaluate_clustering(blobs, labels)["adjusted_rand"], 3))
   print("silhouette:", round(evaluate_clustering_internal(X_blobs, labels)["silhouette_score"], 3))

.. code-block:: text
   :caption: Output

   adjusted Rand index: 0.95
   silhouette: 0.592


Size of the trees
-----------------

``evaluate_interpretability`` measures how long the rules of a tree are: the number of leaves
(rules), of internal nodes (splits) and the maximum depth. It also works with scikit-learn trees,
for comparisons:

.. code-block:: python

   from sklearn.tree import DecisionTreeClassifier

   from RuleTree.evaluation import evaluate_interpretability

   print(evaluate_interpretability(model))
   print(evaluate_interpretability(DecisionTreeClassifier(random_state=0).fit(X_train, y_train)))

.. code-block:: text
   :caption: Output

   {'n_leaves': 5, 'n_internal_nodes': 4, 'max_depth': 3}
   {'n_leaves': 8, 'n_internal_nodes': 7, 'max_depth': 4}


Fairness
--------

``evaluate_fairness`` measures how the values of a sensitive attribute are distributed among the
groups of a partition, e.g. the clusters or the leaves of a tree (see the fair stumps in
:doc:`stumps`): ``balance`` is 1 when each group has the distribution of the whole data set (and
0 when a group lacks a value, e.g. a group of one instance), ``max_fairness_cost`` is 0 when each
group has the ideal distribution (by default, the one of the whole data set). Here the groups are
the three blobs, with a sensitive attribute drawn at random, and with one that depends on the
position of the instances:

.. code-block:: python

   import numpy as np

   from RuleTree.evaluation import evaluate_fairness

   random_attribute = np.random.default_rng(0).integers(0, 2, size=len(blobs))
   position_attribute = (X_blobs[:, 0] > np.median(X_blobs[:, 0])).astype(int)

   print({name: round(value, 3) for name, value in evaluate_fairness(blobs, random_attribute).items()})
   print({name: round(value, 3) for name, value in evaluate_fairness(blobs, position_attribute).items()})

.. code-block:: text
   :caption: Output

   {'balance': 0.913, 'max_fairness_cost': 0.043}
   {'balance': 0.0, 'max_fairness_cost': 0.5}
