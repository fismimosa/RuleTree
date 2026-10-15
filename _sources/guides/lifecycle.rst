Saving, growing and updating trees
==================================

This guide covers what happens to a tree after ``fit``: saving and loading it, growing it further,
recomputing its statistics on new data, and the cache shared by its stumps.

.. code-block:: python

   import json

   from sklearn.datasets import load_iris

   from RuleTree import RuleTreeClassifier
   from RuleTree.tree.RuleTree import RuleTree

   X, y = load_iris(return_X_y=True)
   model = RuleTreeClassifier(max_depth=2, random_state=0).fit(X, y)


Saving and loading
------------------

``to_dict`` turns a fitted tree into a JSON-friendly dictionary (and saves it to a file, if a name
is given); ``RuleTree.from_dict`` rebuilds the tree, of the class it had:

.. code-block:: python

   model.to_dict(filename="iris_tree.json")

   loaded = RuleTree.from_dict("iris_tree.json")
   print(type(loaded).__name__, (loaded.predict(X) == model.predict(X)).all())

.. code-block:: text
   :caption: Output

   RuleTreeClassifier True

The dictionary has the parameters of the tree (``args``), the classes (for classification) and
the nodes, each with its statistics and the state of its stump:

.. code-block:: python

   dictionary = model.to_dict()
   print(sorted(dictionary))
   print(dictionary["nodes"][0]["textual_rule"], "-", dictionary["nodes"][0]["stump_class"])

.. code-block:: text
   :caption: Output

   ['args', 'classes_', 'n_classes_', 'nodes', 'tree_class', 'tree_type']
   X_3 <= 0.800000011920929 - DecisionTreeStumpClassifier

The thresholds are saved without rounding (scikit-learn computes them in single precision).
:doc:`saving` describes the dictionary in detail: what is saved for the tree, the nodes and each
stump, how ``from_dict`` rebuilds the tree, what is not saved, and what a new stump needs to be
saved.


Refitting and growing a tree
----------------------------

``fit`` always starts from an empty tree. ``partial_fit`` instead grows the leaves of a fitted
tree further, without changing the existing splits: for example, after increasing ``max_depth``
or ``max_leaf_nodes``:

.. code-block:: python

   print("leaves before:", len(model.get_leaf_nodes()))

   model.set_params(max_depth=4).partial_fit(X, y)

   print("leaves after:", len(model.get_leaf_nodes()))
   print("same tree as fitting from scratch:",
         (model.predict(X) == RuleTreeClassifier(max_depth=4, random_state=0).fit(X, y).predict(X)).all())

.. code-block:: text
   :caption: Output

   leaves before: 3
   leaves after: 8
   same tree as fitting from scratch: True

``partial_fit`` can also receive new data: the leaves are split using the instances that reach
them.


Updating the statistics
-----------------------

``update_statistics`` recomputes the statistics of every node (predictions, class probabilities,
number of instances, impurity of the splits) on new data, keeping the splits. For example, a tree
trained on a data set can be re-labelled with a different target on the same instances:

.. code-block:: python

   model = RuleTreeClassifier(max_depth=2, random_state=0).fit(X, y)
   is_virginica = (y == 2).astype(int)

   model.update_statistics(X, is_virginica)

   print("classes:", model.classes_)
   print("leaf predictions:", {leaf_id: int(leaf.prediction) for leaf_id, leaf in model.get_leaf_nodes().items()})

.. code-block:: text
   :caption: Output

   classes: [0 1]
   leaf predictions: {'Rl': 0, 'Rrl': 0, 'Rrr': 1}

For regression, the nodes get the mean and the standard deviation of the new target; for
clustering, only the number of instances (the clusters keep their labels). A leaf that no
instance reaches predicts ``nan``.


The cache of the stumps
-----------------------

Some stumps compute values on the whole training set that every split can reuse: the types of
the features, or the distance matrix of the pivot stumps. They store them in the cache of the
tree. Each stump has its own namespace, ``cache_namespace()``: its class name and a hash of its
hyperparameters, so that only the stumps of the same class with the same hyperparameters share
cached values (the tree gives its ``random_state`` to the stumps, so the stumps it uses are the
copies in ``candidate_splits_``):

.. code-block:: python

   from RuleTree.stumps.classification import PivotTreeStumpClassifier

   model = RuleTreeClassifier(max_depth=2, base_stumps=PivotTreeStumpClassifier(), random_state=0).fit(X, y)

   stump = model.candidate_splits_[0][1]
   print(list(model.cache_) == [stump.cache_namespace()], stump.cache_namespace().split("-")[0])
   print(sorted(key[0] for key in model.get_cache(stump.cache_namespace())))

.. code-block:: text
   :caption: Output

   True PivotTreeStumpClassifier
   ['feature_types', 'pivot']

The cache is emptied by ``fit`` and kept by ``partial_fit``, so that growing a tree does not
recompute it. When ``partial_fit`` receives data different from the ones of ``fit``, empty it
first, since it refers to the training instances:

.. code-block:: python

   model.clear_cache()
   print(model.cache_)

.. code-block:: text
   :caption: Output

   {}

The cache is not saved by ``to_dict``.
