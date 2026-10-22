How RuleTree works
==================

This page describes the few objects that every RuleTree is made of, and how they work together.
The other guides build on it; the linked terms are also explained in the :doc:`/glossary`.


Trees, nodes and stumps
-----------------------

A RuleTree is a binary tree of *nodes*. Each internal node holds a :term:`stump`: a single split,
fitted on the instances that reach the node, that sends each instance to the left child (when its
rule is true) or to the right child. Each leaf holds a prediction. The rule of an instance is the
conjunction of the conditions on its path from the root to its leaf.

What makes RuleTree different from a classic decision tree is that the stump is chosen node by
node: the same tree can split the root with a threshold on a feature, a child with the distance
from an instance of the training set (a :term:`pivot`), and a grandchild with the distance of a
time series from a :term:`shapelet`.

.. figure:: /_static/figures/tree_of_stumps.svg
   :alt: Left: a tree whose root is split by a decision tree stump on tabular data, its right child
         by a pivot stump and a grandchild by a shapelet stump on time series. Right: the four steps
         that split a node.
   :width: 100%

   Left: a tree with three kinds of stumps on two types of data. Right: how the tree splits each
   node.

The tree is grown breadth-first. To split a node, it:

1. lists the :term:`candidate splits`: each stump of ``base_stumps`` on each type of data it
   supports among the ones given to ``fit``;
2. chooses among them: with ``stump_selection="random"`` (the default) it draws one, with
   ``"best"`` it fits all of them and keeps the one with the highest gain;
3. fits a copy of the chosen stump on the instances of the node;
4. sends the instances to the two children, which are split in turn.

It stops when the nodes are pure or a limit is reached (``max_depth``, ``max_leaf_nodes``,
``min_samples_split``).


Inside a stump
--------------

Most stumps are the composition of two parts (see :doc:`extending`):

- a :term:`transformer` turns the data into a feature matrix: the columns of a table, the distances
  from some pivots, the distances from some shapelets, ...;
- a :term:`splitter` finds the best split of that matrix: ``feature <= threshold`` for a numerical
  feature, ``feature = value`` for a categorical one.

So a pivot stump is a threshold on a feature, the feature being the distance from a pivot. The
fitted parts are attributes of the stump of each node:

.. code-block:: python

   from sklearn.datasets import load_iris

   from RuleTree import RuleTreeClassifier

   X, y = load_iris(return_X_y=True)
   model = RuleTreeClassifier(max_depth=2, random_state=0).fit(X, y)

   stump = model.root.stump
   print(type(stump).__name__, stump.data_type_, stump.node_id_)
   print(type(stump.transformer_).__name__, type(stump.splitter_).__name__)
   print(stump.get_rule()["textual_rule"])

.. code-block:: text
   :caption: Output

   DecisionTreeStumpClassifier tabular R
   MatrixTransformer ClassificationSplitter
   X_3 <= 0.8

The stumps given to the tree are templates: the tree fits a copy of them at each node, and leaves
them unfitted.


Data types
----------

The data are given to ``fit`` and ``predict`` by keyword, one argument per :term:`data type`:
``X`` (tabular data), ``X_ts`` (time series), ``X_img`` (images) and ``X_txt`` (texts). They must
describe the same instances, in the same order. Each stump declares the types it supports, and the
tree uses it only on those. See :doc:`multimodal`.


Tasks
-----

The task of a tree sets what its leaves predict and which stumps it accepts:

.. list-table::
   :header-rows: 1
   :widths: 30 35 35

   * - Tree
     - Leaves predict
     - Stumps
   * - :class:`~RuleTree.RuleTreeClassifier`
     - the majority class (and the class probabilities)
     - classification stumps
   * - :class:`~RuleTree.RuleTreeRegressor`
     - the mean (or median) of the target, one value per target
     - regression stumps
   * - :class:`~RuleTree.RuleTreeCluster`
     - a cluster label (the id of the leaf)
     - regression stumps, fitted on the principal components of the data

:class:`~RuleTree.RuleTreeClusterClassifier` and :class:`~RuleTree.RuleTreeClusterRegressor` find
the clusters as :class:`~RuleTree.RuleTreeCluster` does, then predict the majority class or the
mean target of each cluster.


Node ids
--------

Each node is identified by its path from the root: ``"R"`` is the root, ``"Rl"`` and ``"Rr"`` its
left and right children, ``"Rrl"`` the left child of ``"Rr"``, and so on. The ids are used by
``get_node_by_id``, returned by ``apply`` (the leaf of each instance) and saved by ``to_dict``.


After fit
---------

A fitted tree can be:

- read: its rules, a drawing, its nodes, the contributions of the features (:doc:`inspecting`);
- grown further with ``partial_fit``, or given new statistics with ``update_statistics``
  (:doc:`lifecycle`);
- saved to JSON and loaded back (:doc:`saving`);
- combined with other trees by boosting (:doc:`ensembles`);
- evaluated, also against other models (:doc:`evaluating`).
