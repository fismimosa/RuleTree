Glossary
========

.. glossary::
   :sorted:

   stump
      The split of an internal node: it sends each instance of the node to the left child (its rule
      is true) or to the right child. Most stumps are a :term:`transformer` followed by a
      :term:`splitter`. See :doc:`guides/stumps`.

   base stumps
      The stumps a tree can use, given by its ``base_stumps`` parameter: a stump, a list of stumps,
      or a list of tuples with the probability of each stump and, optionally, the data types it is
      used on. See :doc:`guides/stumps`.

   transformer
      The part of a stump that turns the data of one type into a feature matrix, e.g. the columns of
      a table or the distances from some pivots. See :doc:`guides/extending`.

   splitter
      The part of a stump that finds the best split of the feature matrix of the transformer: a
      threshold on a numerical feature or a value of a categorical one. See :doc:`guides/extending`.

   candidate splits
      The pairs (stump, data type) among which a tree chooses the split of a node: each base stump
      on each data type it supports among the ones given to ``fit``. They are in
      ``candidate_splits_``.

   stump selection
      How a tree chooses among the candidate splits of a node (``stump_selection``): ``"random"``
      draws one according to their probabilities, ``"best"`` fits all of them and keeps the one
      with the highest gain.

   gain
      How much a split decreases the impurity of a node (e.g. the entropy, or the squared error),
      weighted by the size of the children. The stumps choose the split with the highest gain.

   pivot
      An instance of the training set used as a reference: a pivot stump splits on the distance
      from it (``P_i <= t``, where ``i`` is the index of the instance). The candidate pivots of a
      node are, for each class, a *descriptive* pivot (its medoid) and a *discriminative* pivot
      (the instance whose distance best separates the classes).

   partial pivot
      A pivot restricted to a few features, so that the rule involves fewer features.

   shapelet
      A subsequence of a training time series. The distance of a series from a shapelet is its
      smallest distance over all the positions, so a shapelet finds a pattern wherever it occurs.

   proximity split
      A split that compares two references (two pivots, partial pivots or shapelets): the instances
      closer to the first one than to the second one go left, with no threshold.

   oblique split
      A threshold on a linear combination of a few features, e.g.
      ``0.8 * X_2 - 0.6 * X_3 <= t``: a hyperplane that is not parallel to the axes.

   data type
      The kind of data of a ``fit`` argument: tabular data (``X``), time series (``X_ts``), images
      (``X_img``) and texts (``X_txt``). See :doc:`guides/multimodal`.

   node id
      The path of a node from the root: ``"R"`` is the root, ``"Rl"`` its left child, ``"Rlr"`` the
      right child of ``"Rl"``, and so on.

   cache
      Values computed once on the training set and shared by the stumps of a tree, such as the
      distance matrix of the pivot stumps (``cache_``). See :doc:`guides/lifecycle`.

   extra
      A group of optional dependencies, installed with ``pip install "RuleTree[name]"``: e.g.
      ``shapelets`` for the shapelet, proximity and partial pivot stumps, ``graphviz`` for the
      drawings. See :doc:`guides/getting_started`.
