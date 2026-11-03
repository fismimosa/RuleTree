Splits with a custom gain
=========================

The stumps of the library choose the split that decreases the impurity most, and they find it with a
scikit-learn tree, which cannot be told to prefer some splits over others.
:class:`~RuleTree.stumps.splitters.FlatMatrixClassificationSplitter` and
:class:`~RuleTree.stumps.splitters.FlatMatrixRegressionSplitter` evaluate all the candidate splits of
a node together, so that a function can change their gain before the best one is chosen: to penalize
some features, to require a minimum share of each class in the children, to make the children fair
with respect to a sensitive attribute, and so on.

They come from the matrix formulation of the split search by Lorenzo Libardi (*Flat Matrix*), in
which each candidate is a pair (feature, threshold) and all the candidates are evaluated with array
operations. When the gain is not changed, use
:class:`~RuleTree.stumps.splitters.ClassificationSplitter` and
:class:`~RuleTree.stumps.splitters.RegressionSplitter` instead: they choose a split with the same
gain, about twice as fast and with less memory.

.. code-block:: python

   from sklearn.datasets import load_iris

   from RuleTree import RuleTreeClassifier
   from RuleTree.stumps.Stump import Stump
   from RuleTree.stumps.splitters import FlatMatrixClassificationSplitter
   from RuleTree.stumps.transformers import MatrixTransformer

   X, y = load_iris(return_X_y=True)

   stump = Stump(MatrixTransformer(), FlatMatrixClassificationSplitter(criterion="gini"))
   model = RuleTreeClassifier(max_depth=2, random_state=0, base_stumps=stump).fit(X, y)

   for node_id in model.get_predicates():
       print(node_id, model.get_node_by_id(node_id).stump.get_rule()["textual_rule"])

.. code-block:: text
   :caption: Output

   R X_2 <= 2.45
   Rr X_3 <= 1.75


The candidate splits
--------------------

Each candidate is described by three flat arrays with one value per candidate: ``features`` (the
column), ``thresholds`` and ``is_categorical``. They are:

- for each numerical feature, the thresholds halfway between its consecutive distinct values
  (``x <= threshold`` goes left); missing values (``nan``) go right;
- for each categorical feature, each of its values (``x == value`` goes left).

The candidates with fewer than ``min_samples_leaf`` instances on a side are discarded, as well as the
degenerate ones that would leave a child empty. If the node is pure, it is not split.


The impurity of all the candidates
----------------------------------

The impurity of a child depends only on sums over its instances, so it is computed for all the
candidates without building a child at a time. Each instance gets a vector of *statistics*:

.. list-table::
   :header-rows: 1
   :widths: 20 35 45

   * - Criterion
     - Statistics of an instance
     - Impurity of a child, from the sums of the statistics
   * - ``gini``
     - its weight in the column of its class
     - :math:`1 - \sum_c p_c^2`, with :math:`p_c` the fraction of the weight of class :math:`c`
   * - ``entropy``, ``log_loss``
     - as above
     - :math:`-\sum_c p_c \log_2 p_c`
   * - ``misclassification``
     - as above
     - :math:`1 - \max_c p_c`
   * - ``squared_error``
     - :math:`w`, :math:`w y`, :math:`w y^2`
     - :math:`\frac{\sum w y^2}{\sum w} - \left(\frac{\sum w y}{\sum w}\right)^2`
   * - ``poisson``
     - :math:`w`, :math:`w y`, :math:`w y \log y`
     - :math:`\frac{\sum w y \log y - \sum w y \, \log \bar{y}}{\sum w}`, with :math:`\bar{y}` the
       weighted mean (half the Poisson deviance)

The weight :math:`w` is the sample weight, times the class weight in classification. On a numerical
feature, the instances are sorted by value and the cumulative sums of their statistics give the left
child of every threshold at once; the right child is the node minus the left child. On a categorical
feature, the statistics are summed by value. For example, with four instances of a regression:

.. list-table::
   :header-rows: 1

   * - ``x``
     - ``y``
     - Threshold
     - :math:`\sum w` left
     - :math:`\sum w y` left
     - :math:`\sum w y^2` left
   * - 1
     - 2
     - 1.5
     - 1
     - 2
     - 4
   * - 2
     - 4
     - 2.5
     - 2
     - 6
     - 20
   * - 3
     - 10
     - 3.5
     - 3
     - 16
     - 120
   * - 4
     - 12
     -
     -
     -
     -

With the threshold 2.5 the left child has variance :math:`20/2 - (6/2)^2 = 1`, and the right child,
from the totals 4, 28 and 264, :math:`244/2 - (22/2)^2 = 1`. The cost of a node is a sort and a
cumulative sum per feature: :math:`O(m \, n \log n)` time, and memory that grows with the number of
instances and of candidates, not with their product.


Choosing the split
------------------

The gain of the candidates is computed by ``gain_function`` (by default
:func:`~RuleTree.stumps.splitters.information_gain`, the decrease of impurity), called once per
node with all the candidates that are left:

.. code-block:: text

   gain_function(impurity, impurity_left, impurity_right, weight_left, weight_right,
                 X, y, features, thresholds, is_categorical, sample_weight=None) -> gains

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Argument
     - Content
   * - ``impurity``
     - impurity of the node
   * - ``impurity_left``, ``impurity_right``
     - impurity of the children of each candidate
   * - ``weight_left``, ``weight_right``
     - total weight of the children of each candidate
   * - ``X``, ``y``, ``sample_weight``
     - data of the node; ``X`` is the matrix of the transformer of the stump
   * - ``features``, ``thresholds``, ``is_categorical``
     - the conditions of the candidates

It returns one gain per candidate; ``-inf`` or ``nan`` discards a candidate, and if all are
discarded the node becomes a leaf. The split with the highest gain is chosen; among equal gains, the
first one (numerical before categorical, in the order of the columns). ``gain_`` is the value
returned by the function, while ``impurity_`` stays the impurity of the node and of the children.

This function makes the splits on the last two features (the petals, the most informative ones) twice
as costly, e.g. because they are expensive to measure; it needs only the arguments it receives:

.. code-block:: python

   import numpy as np

   from RuleTree.stumps.splitters import information_gain

   feature_cost = np.array([1.0, 1.0, 2.0, 2.0])


   def cheap_features_gain(impurity, impurity_left, impurity_right, weight_left, weight_right, X, y,
                           features, thresholds, is_categorical, sample_weight=None):
       gains = information_gain(impurity, impurity_left, impurity_right, weight_left, weight_right)
       return gains / feature_cost[features]


   stump = Stump(MatrixTransformer(), FlatMatrixClassificationSplitter(gain_function=cheap_features_gain))
   model = RuleTreeClassifier(max_depth=3, random_state=0, base_stumps=stump).fit(X, y)

   for node_id in model.get_predicates():
       print(node_id, model.get_node_by_id(node_id).stump.get_rule()["textual_rule"])
   print("accuracy:", round(model.score(X, y), 3))

.. code-block:: text
   :caption: Output

   R X_0 <= 5.45
   Rl X_1 <= 2.8
   Rll X_0 <= 4.7
   Rlr X_2 <= 3.2
   Rr X_3 <= 1.75
   Rrl X_1 <= 3.45
   Rrr X_2 <= 4.85
   accuracy: 0.96


Using the instances of the children
-----------------------------------

When the gain depends on which instances go to each child, and not only on the impurity,
:func:`~RuleTree.stumps.splitters.candidate_mask` builds the boolean matrix of shape
(n_samples, n_candidates) with the instances that go left. This one discards the candidates that
leave a child with less than 10% of the instances of a class of the node:

.. code-block:: python

   from RuleTree.stumps.splitters import candidate_mask


   def all_classes_gain(impurity, impurity_left, impurity_right, weight_left, weight_right, X, y,
                        features, thresholds, is_categorical, sample_weight=None):
       left = candidate_mask(X, features, thresholds, is_categorical)
       gains = information_gain(impurity, impurity_left, impurity_right, weight_left, weight_right)
       for label in np.unique(y):
           share_left = left[y == label].mean(axis=0)
           gains[(share_left < 0.1) | (share_left > 0.9)] = -np.inf
       return gains


   stump = Stump(MatrixTransformer(), FlatMatrixClassificationSplitter(gain_function=all_classes_gain))
   model = RuleTreeClassifier(max_depth=1, random_state=0, base_stumps=stump).fit(X, y)

   print(model.get_rules()["textual_rule"])

.. code-block:: text
   :caption: Output

   X_1 <= 3.15

The best split for the gain alone, ``X_2 <= 2.45``, sends all the setosa to the left child and none of
the other classes, so it is discarded.

The mask takes one byte per instance and candidate, and on a continuous feature there are about as
many candidates as instances: the memory grows with the square of the instances of the node. The
function can call ``candidate_mask`` on subsets of the candidates (e.g.
``candidate_mask(X, features[block], thresholds[block], is_categorical[block])``) to process them in
blocks.


Saving the trees
----------------

The function is a parameter of the splitter, so :meth:`~RuleTree.tree.RuleTree.RuleTree.to_dict`
saves it by name and :meth:`~RuleTree.tree.RuleTree.RuleTree.from_dict` imports it again: it must be
defined at the top of a module that can be imported when the tree is loaded. A lambda, or a function
defined inside another one, raises a ``ValueError``; a function defined in a notebook or in the
script being run can be loaded only by the same process. See :doc:`saving`.


Limits
------

- The criteria are the ones in the table above: the impurity cannot be replaced, only the gain;
  ``absolute_error`` is not available, since the median of a child is not a sum.
- ``X`` is the matrix of the transformer: with a transformer other than
  :class:`~RuleTree.stumps.transformers.MatrixTransformer` (pivots, projections) it may not contain
  the original columns, e.g. a sensitive attribute. A value of the whole data set, such as the
  distribution of an attribute, is computed before the fit and given to the function, e.g. as a
  module-level variable.
- Among splits with the same gain the first one is always chosen, while the scikit-learn splitters
  choose at random with ``random_state``: the trees can differ even with the same gains.
- There is no ``max_features``: all the features are evaluated at every node.

.. seealso::

   :doc:`extending`, to write splitters and stumps; :doc:`stumps`, for the stumps of the library.
