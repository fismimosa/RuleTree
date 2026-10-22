Growing and updating trees
==========================

This guide covers what can change in a tree after ``fit``: growing it further, recomputing its
statistics on new data, and the cache shared by its stumps. Saving a tree and loading it back is
described in :doc:`saving`.

.. code-block:: python

   from sklearn.datasets import load_iris

   from RuleTree import RuleTreeClassifier

   X, y = load_iris(return_X_y=True)
   model = RuleTreeClassifier(max_depth=2, random_state=0).fit(X, y)


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
the features, or the distance matrix of the pivot stumps. The tree passes its cache (``cache_``)
to every stump, transformer and splitter, and each decides whether and how to use it: the
transformers of the library store the values under their own namespace, ``cache_namespace()``
(the class name and a hash of the hyperparameters), so that only the transformers of the same
class with the same hyperparameters share cached values, also when they belong to different
stumps:

.. code-block:: python

   from RuleTree.stumps.classification import PivotTreeStumpClassifier

   model = RuleTreeClassifier(max_depth=2, base_stumps=PivotTreeStumpClassifier(), random_state=0).fit(X, y)

   transformer = model.root.stump.transformer_
   print(list(model.cache_) == [transformer.cache_namespace()], transformer.cache_namespace().split("-")[0])
   print(sorted(key[0] for key in model.get_cache(transformer.cache_namespace())))

.. code-block:: text
   :caption: Output

   True PivotTransformer
   ['feature_types', 'pivot']

The cache is emptied by ``fit`` and kept by ``partial_fit``, so that growing a tree does not
recompute it.

.. warning::

   The cache refers to the training instances: when ``partial_fit`` receives data different from
   the ones of ``fit``, empty it first with ``clear_cache``:

.. code-block:: python

   model.clear_cache()
   print(model.cache_)

.. code-block:: text
   :caption: Output

   {}

The cache is not saved by ``to_dict``.
