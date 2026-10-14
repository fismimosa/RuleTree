Getting started
===============

RuleTree builds interpretable decision trees one node at a time. Each node is split by a *stump*:
a single split, such as a threshold on a feature (``petal length <= 2.45``), the distance from an
instance of the training set, or the distance of a time series from a shapelet. The same tree can
mix different kinds of stumps and different types of data.

The trees follow the scikit-learn API: ``fit``, ``predict``, ``predict_proba``, ``get_params``,
``set_params``, and they work with the scikit-learn tools for model selection.


Installation
------------

RuleTree requires Python 3.10 or later. Install it from the root of the repository:

.. code-block:: bash

   pip install .

Optional features have their own dependencies:

.. code-block:: bash

   pip install ".[graphviz]"   # drawing the trees (also requires the Graphviz binaries)
   pip install ".[kmedoids]"   # shapelets selected with k-medoids
   pip install ".[test]"       # running the tests


Classification
--------------

:class:`~RuleTree.tree.RuleTreeClassifier.RuleTreeClassifier` grows the tree breadth-first until
the nodes are pure or a limit is reached (``max_depth``, ``max_leaf_nodes``,
``min_samples_split``). Each leaf predicts the majority class of its training instances.

.. code-block:: python

   from sklearn.datasets import load_iris
   from sklearn.model_selection import train_test_split

   from RuleTree import RuleTreeClassifier

   iris = load_iris()
   X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.3,
                                                       random_state=42, stratify=iris.target)

   model = RuleTreeClassifier(max_depth=3, random_state=42)
   model.fit(X_train, y_train)

   print("accuracy:", round(model.score(X_test, y_test), 3))
   print(model.predict(X_test[:5]))
   print(model.predict_proba(X_test[:2]).round(2))

.. code-block:: text
   :caption: Output

   accuracy: 0.933
   [2 1 2 1 2]
   [[0.   0.03 0.97]
    [0.   1.   0.  ]]

``random_state`` makes the tree reproducible: when two features split a node equally well, the
choice between them is not random.

The rules of the tree can be printed with the names of the features:

.. code-block:: python

   model.print_rules(model.get_rules(columns_names=iris.feature_names))

.. code-block:: text
   :caption: Output

   |--- petal length (cm) <= 2.45
   |    output: 0
   |--- petal length (cm) > 2.45
   |   |--- petal width (cm) <= 1.55
   |   |   |--- petal length (cm) <= 4.95
   |   |   |    output: 1
   |   |   |--- petal length (cm) > 4.95
   |   |   |    output: 2
   |   |--- petal width (cm) > 1.55
   |   |   |--- petal length (cm) <= 4.85
   |   |   |    output: 1
   |   |   |--- petal length (cm) > 4.85
   |   |   |    output: 2

See :doc:`inspecting` for the other ways to read a tree.


Regression
----------

:class:`~RuleTree.tree.RuleTreeRegressor.RuleTreeRegressor` works in the same way; each leaf
predicts the mean of the target of its training instances.

.. code-block:: python

   from sklearn.datasets import load_diabetes
   from sklearn.metrics import r2_score

   from RuleTree import RuleTreeRegressor

   X, y = load_diabetes(return_X_y=True)

   regressor = RuleTreeRegressor(max_depth=3, random_state=42).fit(X, y)

   print("R2 on the training set:", round(r2_score(y, regressor.predict(X)), 3))
   print("leaves:", len(regressor.get_leaf_nodes()))

.. code-block:: text
   :caption: Output

   R2 on the training set: 0.501
   leaves: 8


Clustering
----------

:class:`~RuleTree.tree.RuleTreeCluster.RuleTreeCluster` finds clusters described by rules, without
a target. To split a node, it projects its instances on their principal components and splits
the component that is best separated; a node is split only if the split improves the Bayesian
Information Criterion of the clustering (``bic_eps`` makes this stricter).

.. code-block:: python

   from sklearn.datasets import make_blobs
   from sklearn.metrics import adjusted_rand_score

   from RuleTree import RuleTreeCluster

   X, groups = make_blobs(n_samples=300, centers=3, cluster_std=0.5, random_state=0)

   clustering = RuleTreeCluster(max_depth=3, random_state=0)
   labels = clustering.fit_predict(X)

   print("clusters:", sorted(set(labels.tolist())))
   print("agreement with the true groups (ARI):", round(adjusted_rand_score(groups, labels), 2))

.. code-block:: text
   :caption: Output

   clusters: [4, 5, 6, 7, 8]
   agreement with the true groups (ARI): 0.95

The labels identify the leaves, numbered in breadth-first order (the root is 0). Here the tree
splits two of the three groups once more, so there are five clusters; the agreement with the
groups stays high because each cluster lies within a single group.
:class:`~RuleTree.tree.RuleTreeCluster.RuleTreeClusterClassifier` and
:class:`~RuleTree.tree.RuleTreeCluster.RuleTreeClusterRegressor` build the tree in the same way,
without looking at the target, and then predict the majority class or the mean target of each
cluster.


Working with scikit-learn
-------------------------

The trees are scikit-learn estimators, so they can be used in pipelines, cross-validation and
hyper-parameter searches:

.. code-block:: python

   from sklearn.model_selection import GridSearchCV, cross_val_score

   scores = cross_val_score(RuleTreeClassifier(max_depth=3, random_state=0), iris.data, iris.target, cv=5)
   print("cross-validated accuracy:", round(scores.mean(), 3))

   search = GridSearchCV(RuleTreeClassifier(random_state=0), {"max_depth": [1, 2, 3, 4]}, cv=5)
   search.fit(iris.data, iris.target)
   print("best max_depth:", search.best_params_["max_depth"])

.. code-block:: text
   :caption: Output

   cross-validated accuracy: 0.973
   best max_depth: 3


Next steps
----------

- :doc:`stumps`: the available stumps and how to combine them.
- :doc:`multimodal`: trees on time series and images, alone or together with tabular data.
- :doc:`inspecting`: rules, Graphviz drawings, leaves and feature contributions.
- :doc:`lifecycle`: saving and loading trees, growing them further, updating their statistics.
- :doc:`ensembles`: gradient boosting and explainable boosting with RuleTrees.
- :doc:`extending`: writing new stumps.
