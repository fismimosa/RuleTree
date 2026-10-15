Choosing the splits: stumps
===========================

A *stump* is the split of a node: it sends each instance of the node to the left or to the right
child. The ``base_stumps`` parameter of a tree lists the stumps it can use; at each node, the tree
fits one of them (or all of them, and keeps the best) on the instances of the node.


Available stumps
----------------

.. list-table::
   :header-rows: 1
   :widths: 30 15 20 35

   * - Stump
     - Task
     - Data types
     - Split
   * - :class:`~RuleTree.stumps.classification.DecisionTreeStumpClassifier.DecisionTreeStumpClassifier`,
       :class:`~RuleTree.stumps.regression.DecisionTreeStumpRegressor.DecisionTreeStumpRegressor`
     - classification, regression
     - tabular, time series, images
     - threshold on a feature (``x <= t``) or value of a categorical feature (``x = v``); time
       series and images are flattened into one feature per value
   * - :class:`~RuleTree.stumps.classification.PivotTreeStumpClassifier.PivotTreeStumpClassifier`,
       :class:`~RuleTree.stumps.regression.PivotTreeStumpRegressor.PivotTreeStumpRegressor`
     - classification, regression
     - tabular
     - distance from a *pivot*, an instance of the node that represents a class or separates the
       classes
   * - :class:`~RuleTree.stumps.classification.PartialPivotTreeStumpClassifier.PartialPivotTreeStumpClassifier`
     - classification
     - tabular
     - distance from a *partial pivot*, an instance restricted to a few features
   * - :class:`~RuleTree.stumps.classification.ShapeletTreeStumpClassifier.ShapeletTreeStumpClassifier`,
       :class:`~RuleTree.stumps.regression.ShapeletTreeStumpRegressor.ShapeletTreeStumpRegressor`
     - classification, regression
     - time series
     - distance from a *shapelet*, a subsequence of a time series
   * - :class:`~RuleTree.stumps.regression.FairTreeStumpRegressor.FairTreeStumpRegressor`
     - regression (and clustering)
     - tabular
     - threshold on a feature, under a fairness constraint on a sensitive attribute
   * - :class:`~RuleTree.stumps.regression.XGBoostStumpRegressor.XGBoostStumpRegressor`
     - regression
     - tabular, time series, images
     - threshold maximizing the XGBoost gain (see :doc:`ensembles`)

A tree accepts only the stumps of its task: classification stumps for
:class:`~RuleTree.tree.RuleTreeClassifier.RuleTreeClassifier`, regression stumps for
:class:`~RuleTree.tree.RuleTreeRegressor.RuleTreeRegressor` and for the clustering trees, whose
stumps predict the principal components of the data. By default, the trees use the decision tree
stump of their task.


Parameters of the stumps
------------------------

The stumps are scikit-learn estimators: their parameters are set in the constructor and shown by
``get_params``. The decision tree stumps take the parameters of a scikit-learn decision tree that
make sense for a single split:

.. code-block:: python

   from sklearn.datasets import load_breast_cancer

   from RuleTree import RuleTreeClassifier
   from RuleTree.stumps.classification import DecisionTreeStumpClassifier

   data = load_breast_cancer()
   X, y = data.data, data.target

   stump = DecisionTreeStumpClassifier(criterion="entropy", min_samples_leaf=20, class_weight="balanced")
   model = RuleTreeClassifier(max_depth=3, base_stumps=stump, random_state=0).fit(X, y)

   print(stump.get_params())
   print("leaves:", len(model.get_leaf_nodes()))

.. code-block:: text
   :caption: Output

   {'class_weight': 'balanced', 'criterion': 'entropy', 'max_features': None, 'min_impurity_decrease': 0.0, 'min_samples_leaf': 20, 'min_weight_fraction_leaf': 0.0, 'random_state': None}
   leaves: 7

The stump passed to the tree is a template: the tree fits a copy of it at each node, so ``stump``
itself stays unfitted.


Combining stumps
----------------

``base_stumps`` can be:

- a stump;
- a list of stumps;
- a list of tuples ``(probability, stump)``, with probabilities summing to 1;
- a list of tuples ``(probability, stump, data_types)``, to use a stump only on some data types
  (see :doc:`multimodal`).

With ``stump_selection="random"`` (the default), each node is split by a stump drawn according
to the probabilities (equal, for a list of stumps). With ``stump_selection="best"``, all the
stumps are fitted and the one with the highest gain is kept:

.. code-block:: python

   from collections import Counter

   from RuleTree.stumps.classification import PivotTreeStumpClassifier

   model = RuleTreeClassifier(max_depth=3, stump_selection="best", random_state=0,
                              base_stumps=[DecisionTreeStumpClassifier(), PivotTreeStumpClassifier()])
   model.fit(X, y)

   splits = Counter(type(node.stump).__name__ for node in model.get_predicates().values())
   print(sorted(splits.items()))

.. code-block:: text
   :caption: Output

   [('DecisionTreeStumpClassifier', 6), ('PivotTreeStumpClassifier', 1)]

A random choice gives more diverse trees, e.g. for ensembles; ``random_state`` makes the choice
reproducible:

.. code-block:: python

   model = RuleTreeClassifier(max_depth=3, random_state=0,
                              base_stumps=[(0.7, DecisionTreeStumpClassifier()), (0.3, PivotTreeStumpClassifier())])
   model.fit(X, y)

   splits = Counter(type(node.stump).__name__ for node in model.get_predicates().values())
   print(sorted(splits.items()))

.. code-block:: text
   :caption: Output

   [('DecisionTreeStumpClassifier', 4), ('PivotTreeStumpClassifier', 3)]

The ``random_state`` of the tree is also given to the stumps whose ``random_state`` is None, so
that they do not break ties between equally good features at random.


Pivot stumps
------------

A pivot stump splits on the distance from an instance of the training set, the *pivot*: the
instances closer than a threshold go left. The candidate pivots of a node are, for each class,
its medoid (descriptive pivot) and the instance whose distance best separates the classes
(discriminative pivot). Distances are computed on the standardized numerical features, with any
metric of ``sklearn.metrics.pairwise_distances``:

.. code-block:: python

   pivot_tree = RuleTreeClassifier(max_depth=2, random_state=0,
                                   base_stumps=PivotTreeStumpClassifier(distance_measure="cosine"))
   pivot_tree.fit(X, y)

   pivot_tree.print_rules(pivot_tree.get_rules())
   print("pivot of the root:", pivot_tree.root.stump.feature_original)

.. code-block:: text
   :caption: Output

   |--- P_461 <= 1.052
   |   |--- P_368 <= 1.011
   |   |    output: 0
   |   |--- P_368 > 1.011
   |   |    output: 0
   |--- P_461 > 1.052
   |   |--- P_199 <= 0.546
   |   |    output: 0
   |   |--- P_199 > 0.546
   |   |    output: 1
   pivot of the root: 461

In the rules, ``P_<i>`` is the distance from the training instance ``i``. The distance matrix of
the training set is computed once and cached in the tree (see :doc:`lifecycle`).

``get_pivots`` returns the pivots of a node: the descriptive and discriminative ones, all the
candidates and the one used by the split. For the whole tree:

.. code-block:: python

   for node_id, node in pivot_tree.get_predicates().items():
       print(node_id, node.stump.get_pivots())

.. code-block:: text
   :caption: Output

   R {'descriptives': [167, 249], 'discriminatives': [461, 140], 'candidates': [167, 249, 461, 140], 'used': 461}
   Rl {'descriptives': [167, 421], 'discriminatives': [368, 191], 'candidates': [167, 421, 368, 191], 'used': 368}
   Rr {'descriptives': [40, 415], 'discriminatives': [199, 71], 'candidates': [40, 415, 199, 71], 'used': 199}


Fair splits
-----------

:class:`~RuleTree.stumps.regression.FairTreeStumpRegressor.FairTreeStumpRegressor` chooses the
split under a constraint on a sensitive attribute, a column of the data that the splits never
use:

- ``penalty="balance"``: the groups of the sensitive attribute should be represented in each
  child as in the node;
- ``penalty="mfc"``: the distribution of the groups should be close to ``ideal_distribution``;
- ``penalty="privacy"``: the children should satisfy k-anonymity, l-diversity and t-closeness.

As a regression stump, it can be used by the clustering trees to find fair clusters. Here the
sensitive attribute (column 2) is strongly related to the clusters:

.. code-block:: python

   import numpy as np
   from sklearn.datasets import make_blobs

   from RuleTree import RuleTreeCluster
   from RuleTree.stumps.regression import FairTreeStumpRegressor
   from RuleTree.utils.fairness_metrics import balance_metric

   points, groups = make_blobs(n_samples=300, centers=4, cluster_std=0.8, random_state=1)
   rng = np.random.default_rng(0)
   sensitive = np.where(rng.random(300) < 0.8, groups % 2, 1 - groups % 2)
   data = np.column_stack([points, sensitive]).astype(float)

   for name, stump in [("standard", None),
                       ("fair", FairTreeStumpRegressor(penalty="balance", sensible_attribute=2, n_jobs=1))]:
       clustering = RuleTreeCluster(max_depth=2, random_state=0, base_stumps=stump)
       labels = clustering.fit_predict(data)
       print(f"{name}: {len(set(labels.tolist()))} clusters, balance {balance_metric(labels, sensitive):.2f}")

.. code-block:: text
   :caption: Output

   standard: 4 clusters, balance 0.00
   fair: 4 clusters, balance 0.44

The balance is 1 when every cluster has the same proportion of each group as the whole data set.


Other stumps
------------

- :class:`~RuleTree.stumps.classification.PartialPivotTreeStumpClassifier.PartialPivotTreeStumpClassifier`
  restricts the pivots to a few features, so that the rules are shorter.
- The shapelet stumps work on time series: see :doc:`multimodal`.
- :class:`~RuleTree.stumps.regression.XGBoostStumpRegressor.XGBoostStumpRegressor` is the stump of
  the XGBoost trees: see :doc:`ensembles`.
- New stumps can be written by combining a *transformer* and a *splitter*: see :doc:`extending`.
