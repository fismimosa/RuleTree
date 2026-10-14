RuleTree
========

RuleTree builds interpretable decision trees one node at a time. Each node is split by a *stump*:
a threshold on a feature, the distance from an instance of the training set (a *pivot*), the
distance of a time series from a *shapelet*, a fair split, ... The same tree can mix different
kinds of stumps and different types of data (tabular data, time series, images), and its rules
can be read, drawn and saved.

The trees follow the scikit-learn API and support classification
(:class:`~RuleTree.tree.RuleTreeClassifier.RuleTreeClassifier`), regression
(:class:`~RuleTree.tree.RuleTreeRegressor.RuleTreeRegressor`) and clustering
(:class:`~RuleTree.tree.RuleTreeCluster.RuleTreeCluster`); they can also be combined by boosting.

.. code-block:: python

   from sklearn.datasets import load_iris

   from RuleTree import RuleTreeClassifier

   X, y = load_iris(return_X_y=True)
   model = RuleTreeClassifier(max_depth=2, random_state=0).fit(X, y)
   model.print_rules(model.get_rules(columns_names=["sepal length", "sepal width",
                                                     "petal length", "petal width"]))

.. code-block:: text
   :caption: Output

   |--- petal width <= 0.8
   |    output: 0
   |--- petal width > 0.8
   |   |--- petal width <= 1.75
   |   |    output: 1
   |   |--- petal width > 1.75
   |   |    output: 2


.. toctree::
   :maxdepth: 2
   :caption: User guide

   guides/getting_started
   guides/stumps
   guides/multimodal
   guides/inspecting
   guides/lifecycle
   guides/ensembles
   guides/extending

.. toctree::
   :maxdepth: 2
   :caption: API reference

   RuleTree.tree
   RuleTree.stumps
   RuleTree.base
   RuleTree.ensemble
   RuleTree.utils
