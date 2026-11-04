Reading the tree
================

A RuleTree is meant to be read. This guide shows how to get its rules, draw it, walk its nodes
and explain its predictions. The examples use a classifier on the iris data set:

.. code-block:: python

   from sklearn.datasets import load_iris

   from RuleTree import RuleTreeClassifier

   iris = load_iris()
   names = ["sepal length", "sepal width", "petal length", "petal width"]
   model = RuleTreeClassifier(max_depth=3, random_state=0).fit(iris.data, iris.target)


Rules
-----

``get_rules`` returns the rules as a nested dictionary, and ``print_rules`` prints them:

.. code-block:: python

   rules = model.get_rules(columns_names=names)
   model.print_rules(rules)

.. code-block:: text
   :caption: Output

   |--- petal width <= 0.8
   |    output: 0
   |--- petal width > 0.8
       |--- petal width <= 1.75
       |   |--- petal length <= 4.95
       |   |    output: 1
       |   |--- petal length > 4.95
       |        output: 2
       |--- petal width > 1.75
            output: 2

Each node of the dictionary has its statistics and, for an internal node, the condition of the
left child and its negation; the children are in ``left_node`` and ``right_node``:

.. code-block:: python

   print(rules["textual_rule"], "|", rules["not_textual_rule"])
   print(rules["prediction"], rules["prediction_probability"].round(2))
   print(rules["left_node"]["is_leaf"], rules["left_node"]["prediction"])

.. code-block:: text
   :caption: Output

   petal width <= 0.8 | petal width > 0.8
   0 [0.33 0.33 0.33]
   True 0

.. tip::

   The thresholds are in the scale of the training data. If the data were standardized, pass the
   scaler to ``get_rule`` of the stumps (or to ``export_graphviz``) to see them in the original
   scale.


Drawing the tree
----------------

``plot_tree`` draws the tree with matplotlib, as ``sklearn.tree.plot_tree`` does for the
scikit-learn trees (it requires the ``plot`` extra). It returns the matplotlib axes of the
drawing, which a notebook shows and ``savefig`` saves as PNG, SVG or PDF:

.. code-block:: python

   ax = model.plot_tree(columns_names=names, class_names=list(iris.target_names))
   ax.figure.savefig("iris_tree.svg")

   # the box of each node is a text whose gid is the id of the node
   print([text.get_text().split("\n")[1] for text in ax.texts if text.get_gid() in ("R", "Rl")])

.. code-block:: text
   :caption: Output

   ['petal width <= 0.8', 'class = setosa']

.. themed-figure:: /_static/figures/plot_tree.svg
   :alt: The tree of depth 3 on iris drawn by plot_tree: each internal node with its id, rule and
         number of instances, each leaf with its class, colored by class.
   :width: 100%

   ``model.plot_tree(columns_names=names, class_names=list(iris.target_names))``. The arrow "true"
   leads to the child that gets the instances for which the rule is true.

Each internal node shows its id, its rule and its number of training instances; each leaf its
prediction: the class, the value (one per target) or the cluster. The drawing works with every
stump, since each stump says what its node shows (``get_plot_rule``): the rule, for the shapelet
and proximity stumps also the plot of their shapelets (see :doc:`multimodal`), and for the pivot
stumps their pivots (see `Drawing the instances of the splits`_). Its options:

- ``ax``: the axes to draw on, e.g. a subplot of a larger figure. By default, a new figure sized on
  the number of leaves and on the depth;
- ``max_depth``: the depth drawn; the subtrees below are shown as ``(...)``, as for large trees;
- ``node_ids`` and ``samples``: whether to show the ids and the number of instances;
- ``filled``: whether to color the leaves by class, cluster or predicted value;
- ``fontsize``: the size of the text; by default, the largest with which the boxes do not
  overlap;
- ``wrap``: the maximum number of characters per line. Long rules are split before their
  comparison operator, then between words (``mean fractal dimension <= 0.062`` becomes
  ``mean fractal`` / ``dimension`` / ``<= 0.062``), so that the boxes of trees with long feature
  names are narrower and the text larger;
- ``colors``: the colors of the boxes, of the arrows and of the classes, e.g. for a dark
  background (the figures of this documentation are drawn so);
- ``scaler`` and ``float_precision``: as for ``get_rule``.

To draw a subtree, the function ``RuleTree.utils.plotting.plot_tree`` takes its root, e.g.
``plot_tree(model.get_node_by_id("Rr"))``, and the same options.

Drawing the instances of the splits
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Some splits are defined by series, images or texts: the shapelet stumps by subsequences of the
training series, always drawn (see :doc:`multimodal`), and the pivot stumps by training instances,
the pivots: the rule ``P_447 <= P_611`` of a proximity pivot stump compares the distances from the
training instances 447 and 611. The pivot stumps on time series, images and texts keep their pivots
as given to ``fit``, and ``plot_tree(show_media=True)`` draws them: below the rule of their node,
or, when the split compares two of them (the proximity stumps), on the arrows of the two children,
the first one on the arrow of the instances closer to it. By default (``show_media=False``) only the
rules are drawn. Here a tree is trained on images of digits:

.. code-block:: python

   import numpy as np
   from sklearn.datasets import load_digits

   from RuleTree.stumps.classification import ProximityPivotTreeStumpClassifier

   digits = load_digits()
   four = np.isin(digits.target, [0, 1, 4, 7])
   images, y_digits = digits.images[four], digits.target[four]

   pivot_tree = RuleTreeClassifier(max_depth=2, random_state=0, base_stumps=ProximityPivotTreeStumpClassifier())
   pivot_tree.fit(X_img=images, y=y_digits)
   ax = pivot_tree.plot_tree(show_media=True)

   print(pivot_tree.root.stump.get_rule()["textual_rule"])
   print(sorted(inset.get_gid() for inset in ax.child_axes))  # the axes of each drawn instance

.. code-block:: text
   :caption: Output

   P_447 <= P_611
   ['media-R-false', 'media-R-true', 'media-Rl-false', 'media-Rl-true', 'media-Rr-false', 'media-Rr-true']

.. themed-figure:: /_static/figures/plot_tree_media.svg
   :alt: A proximity pivot tree of depth 2 on four digits: each internal node compares two images
         of digits, drawn on the arrows of its two children.
   :width: 80%

   ``pivot_tree.plot_tree(show_media=True)``: each node sends an image to the child whose pivot it
   is closer to.

The pivots are saved with the tree, so a loaded tree draws them too. The pivots of tabular data are
not drawn.

Graphviz
~~~~~~~~

``export_graphviz`` returns the tree as a Graphviz graph instead (it requires the ``graphviz``
extra), with the same boxes as ``plot_tree``: the id, rule and number of instances of each node, the
leaves colored by prediction, the arrows ``true`` and ``false``, and the media of the splits (the
shapelets; the pivots with ``show_media=True``), drawn into PNG files of a temporary directory. It
takes the same options, except the ones of the matplotlib layout (``ax``, ``fontsize``). Its DOT
source can be read or saved; rendering it to an image also requires the Graphviz programs:

.. code-block:: python

   graph = model.export_graphviz(columns_names=names)
   print(next(line.strip() for line in graph.source.splitlines() if line.strip().startswith("R ")))

.. code-block:: text
   :caption: Output

   R [label=<R<BR/>petal width &lt;= 0.8<BR/>samples = 150> color="#9aa5b1ff" fillcolor="#f4f6f8ff"]

.. code-block:: text

   model.export_graphviz(columns_names=names, filename="iris_tree")   # writes iris_tree.pdf

Nodes
-----

Each node has an id that is its path from the root: ``"R"`` is the root, ``"Rl"`` and ``"Rr"``
its left and right children, ``"Rrl"`` the left child of ``"Rr"``, and so on.

.. themed-figure:: /_static/figures/node_ids.svg
   :alt: A tree of depth 2 on iris with the nodes R, Rl, Rr, Rrl and Rrr, the rule of each internal
         node and the class and number of instances of each node.
   :width: 80%

   The ids of the nodes of a tree of depth 2 on iris. The left child of a node gets the instances
   for which its rule is true.

.. code-block:: python

   print("internal nodes:", list(model.get_predicates()))
   print("leaves:", list(model.get_leaf_nodes()))

   node = model.get_node_by_id("Rr")
   print(node.get_depth(), node.samples, node.prediction, node.is_leaf())
   print(node.stump.get_rule(columns_names=names)["textual_rule"])

.. code-block:: text
   :caption: Output

   internal nodes: ['R', 'Rr', 'Rrl']
   leaves: ['Rl', 'Rrll', 'Rrlr', 'Rrr']
   1 100 1 False
   petal width <= 1.75

The stump of an internal node also exposes its split: ``feature_original`` (the feature) and
``threshold_original`` (the threshold, or the value of a categorical feature).


Leaves of the instances
-----------------------

``apply`` returns the leaf reached by each instance, e.g. to group the instances by rule:

.. code-block:: python

   import numpy as np

   leaves = model.apply(iris.data)
   for leaf, count in zip(*np.unique(leaves, return_counts=True)):
       print(leaf, count)

.. code-block:: text
   :caption: Output

   Rl 50
   Rrll 48
   Rrlr 6
   Rrr 46


Contributions of the features
-----------------------------

``local_interpretation`` decomposes each prediction, as in treeinterpreter: the class
probabilities of an instance are the ones of the root (the bias) plus a contribution of each
feature, the change of probabilities along the splits on that feature in the path of the
instance.

.. code-block:: python

   prediction, bias, contributions = model.local_interpretation(iris.data)

   instance = 100
   print("prediction:", prediction[instance].round(2))
   print("bias:", bias[instance].round(2))
   for name, contribution in zip(names, contributions[instance]):
       print(f"{name}: {contribution.round(2)}")

.. code-block:: text
   :caption: Output

   prediction: [0.   0.02 0.98]
   bias: [0.33 0.33 0.33]
   sepal length: [0. 0. 0.]
   sepal width: [0. 0. 0.]
   petal length: [0. 0. 0.]
   petal width: [-0.33 -0.31  0.64]

The prediction is the bias plus the sum of the contributions. ``local_interpretation`` works for
the classification and regression trees with the decision tree stumps on tabular data.
