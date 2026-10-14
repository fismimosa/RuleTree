Tabular data, time series and images
====================================

A RuleTree can be trained on several types of data at once, as long as they describe the same
instances. Each type has its own argument of ``fit``, ``predict``, ``predict_proba`` and
``apply``:

.. list-table::
   :header-rows: 1

   * - Argument
     - Data type
     - Shape
   * - ``X``
     - ``"tabular"``
     - (n_samples, n_features)
   * - ``X_ts``
     - ``"ts"``
     - (n_samples, n_signals, length)
   * - ``X_img``
     - ``"image"``
     - (n_samples, ...)
   * - ``X_txt``
     - ``"text"``
     - (n_samples,)

The type is given by the argument, not by the shape of the array: a 3D array passed as ``X`` is
tabular data. Each stump declares the data types it supports (``supported_data_types``), and the
tree uses each stump on the types it supports among the ones given to ``fit``.


Time series
-----------

The examples use synthetic time series: the series of class 1 have a bump, at a different
position in each series.

.. code-block:: python

   import numpy as np

   rng = np.random.default_rng(0)
   y = rng.integers(0, 2, size=200)
   X_ts = rng.normal(scale=0.4, size=(200, 1, 50))
   starts = rng.integers(5, 35, size=200)
   for i in np.where(y == 1)[0]:
       X_ts[i, 0, starts[i]:starts[i] + 10] += 1.0

The decision tree stump flattens the time series into one feature per time step, so each split is
a threshold on the value at a given time:

.. code-block:: python

   from RuleTree import RuleTreeClassifier

   model = RuleTreeClassifier(max_depth=2, random_state=0).fit(X_ts=X_ts, y=y)

   print("accuracy:", round((model.predict(X_ts=X_ts) == y).mean(), 3))
   print(model.get_rules()["textual_rule"])

.. code-block:: text
   :caption: Output

   accuracy: 0.81
   X_19 <= 0.556

``score`` follows the scikit-learn signature ``score(X, y)``, so it works only with tabular
data: with other data types, the predictions are compared with the target explicitly, as above.

Since the bump moves, no single time step separates the classes well. The shapelet stumps split
on the distance from a *shapelet*, a subsequence of one of the training series: the distance of a
series from a shapelet is its smallest distance over all the positions, so it does not depend on
where the bump is:

.. code-block:: python

   from RuleTree.stumps.classification import DecisionTreeStumpClassifier, ShapeletTreeStumpClassifier

   shapelets = ShapeletTreeStumpClassifier(n_shapelets=10, sliding_window=10, selection="random")
   model = RuleTreeClassifier(max_depth=2, base_stumps=shapelets, random_state=0).fit(X_ts=X_ts, y=y)

   print("accuracy:", round((model.predict(X_ts=X_ts) == y).mean(), 3))
   print(model.root.stump.get_rule(float_precision=2)["textual_rule"])

.. code-block:: text
   :caption: Output

   accuracy: 0.96
   euclidean(TS, shp) <= 1.12

The Graphviz drawing of a shapelet tree shows the plot of the shapelet in each node (see
:doc:`inspecting`).


Several data types together
---------------------------

When several types are given, the candidate splits of each node are the pairs (stump, data type)
supported by the stumps. Here the tabular features describe the instances only partially, and the
time series are needed to separate the classes:

.. code-block:: python

   X = np.column_stack([rng.normal(size=200), y + rng.normal(scale=1.5, size=200)])

   model = RuleTreeClassifier(max_depth=3, stump_selection="best", random_state=0,
                              base_stumps=[DecisionTreeStumpClassifier(), shapelets])
   model.fit(X=X, X_ts=X_ts, y=y)

   for probability, stump, data_type in model.candidate_splits_:
       print(f"{type(stump).__name__} on {data_type} data, probability {probability:.2f}")
   print("accuracy:", round((model.predict(X=X, X_ts=X_ts) == y).mean(), 3))

.. code-block:: text
   :caption: Output

   DecisionTreeStumpClassifier on tabular data, probability 0.25
   DecisionTreeStumpClassifier on ts data, probability 0.25
   ShapeletTreeStumpClassifier on ts data, probability 0.50
   accuracy: 0.915

The decision tree stump supports both types, so it is a candidate on both; the probabilities are
used with ``stump_selection="random"``: the probability of each stump is split equally among its
data types. To use a stump only on some types, give them as third element of the tuple:

.. code-block:: python

   model = RuleTreeClassifier(max_depth=3, random_state=0,
                              base_stumps=[(0.5, DecisionTreeStumpClassifier(), "tabular"),
                                           (0.5, shapelets)])
   model.fit(X=X, X_ts=X_ts, y=y)

   for node_id, node in model.get_predicates().items():
       print(node_id, type(node.stump).__name__, node.stump.data_type_)

.. code-block:: text
   :caption: Output

   R ShapeletTreeStumpClassifier ts
   Rl ShapeletTreeStumpClassifier ts
   Rr DecisionTreeStumpClassifier tabular
   Rrl DecisionTreeStumpClassifier tabular
   Rrr ShapeletTreeStumpClassifier ts

The data type used by a fitted stump is stored in ``data_type_``. ``predict`` must receive the same
types given to ``fit``.

The clustering trees support only tabular data.


Unsupported data
----------------

If none of the stumps supports the data given to ``fit``, the tree raises an error; if some data
type is not used by any stump, it is ignored with a warning:

.. code-block:: python

   import warnings

   try:
       RuleTreeClassifier(base_stumps=shapelets).fit(X=X, y=y)
   except ValueError as error:
       print("error:", error)

   texts = np.array(["a text"] * 200)
   with warnings.catch_warnings(record=True) as caught:
       warnings.simplefilter("always")
       RuleTreeClassifier(max_depth=1).fit(X=X, X_txt=texts, y=y)
   print("warning:", caught[0].message)

.. code-block:: text
   :caption: Output

   error: None of the base stumps supports the provided data (tabular).
   warning: No base stump uses the text data: it will be ignored.

No stump of the library supports texts yet: they can be supported by a new stump with a
transformer that turns texts into features (see :doc:`extending`).
