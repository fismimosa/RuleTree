Saving and loading trees
========================

A fitted tree is saved with :meth:`~RuleTree.tree.RuleTree.RuleTree.to_dict`, which turns it into a
dictionary of plain values (numbers, strings, lists) that can be written as JSON, and loaded with
:meth:`~RuleTree.tree.RuleTree.RuleTree.from_dict`, which rebuilds a tree of the same class that
makes the same predictions. This guide describes what is saved, how the tree is rebuilt, and what
a new stump needs to be saved.

.. code-block:: python

   from sklearn.datasets import load_iris

   from RuleTree import RuleTreeClassifier
   from RuleTree.tree.RuleTree import RuleTree

   X, y = load_iris(return_X_y=True)
   model = RuleTreeClassifier(max_depth=2, random_state=0).fit(X, y)

   model.to_dict(filename="tree.json")
   loaded = RuleTree.from_dict("tree.json")

   print(type(loaded).__name__, (loaded.predict(X) == model.predict(X)).all())

.. code-block:: text
   :caption: Output

   RuleTreeClassifier True

``from_dict`` is called on :class:`~RuleTree.tree.RuleTree.RuleTree`, the base class: the class of
the tree is read from the dictionary (calling it on the concrete class, e.g.
``RuleTreeClassifier.from_dict``, gives the same result).


Files and dictionaries
----------------------

``to_dict()`` returns the dictionary; ``to_dict(filename=...)`` also writes it to a JSON file.
``from_dict`` accepts either a dictionary or the name of a file. The dictionary can contain NumPy
numbers (e.g. the thresholds of the splits), which the ``json`` module does not write by itself:
to write it in another way (a string, a database, a compressed file), use the encoder of the
library, which turns them into Python numbers and lists:

.. code-block:: python

   import json

   from RuleTree.utils.data_utils import json_numpy_encoder

   text = json.dumps(model.to_dict(), cls=json_numpy_encoder)
   loaded = RuleTree.from_dict(json.loads(text))

   print((loaded.predict(X) == model.predict(X)).all())

.. code-block:: text
   :caption: Output

   True

The numbers are saved without rounding, so the loaded tree splits exactly as the original one.
The JSON may contain ``Infinity`` (e.g. the default ``max_depth`` and ``max_leaf_nodes``) and
``NaN`` (e.g. the features not used by a partial pivot, or the prediction of a node that no
instance reaches after ``update_statistics``): they are read by the ``json`` module of Python,
but they are not part of the JSON standard, and other languages may need an option to accept them.


What the dictionary contains
----------------------------

The dictionary has the class of the tree, its parameters, the classes of the target (only for
classification trees) and the nodes:

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Key
     - Content
   * - ``tree_type``, ``tree_class``
     - module and name of the class of the tree, e.g. ``RuleTree.tree.RuleTreeClassifier`` and
       ``RuleTreeClassifier``
   * - ``args``
     - the parameters of the constructor of the tree (``get_params``), with the base stumps saved
       as dictionaries
   * - ``classes_``, ``n_classes_``
     - the classes of the target (classification trees only)
   * - ``nodes``
     - one dictionary for each node of the tree

Parameters and base stumps
~~~~~~~~~~~~~~~~~~~~~~~~~~

``args["base_stumps"]`` is the list of the base stumps in the form
``[probability, stump, data_types]``, whatever the form given to the tree (see :doc:`stumps`):
the probabilities are explicit, and ``data_types`` is None when the stump is used on all the data
types it supports. Each stump, not fitted, is saved as its module (``stump_type``), its class
(``stump_class``) and its parameters (``args``). The parameters that are estimators, such as the
transformer and the splitter of a :class:`~RuleTree.stumps.Stump.Stump`, are saved as their class
(``__estimator__``) and their own parameters:

.. code-block:: python

   from RuleTree.stumps.Stump import Stump
   from RuleTree.stumps.splitters import ClassificationSplitter
   from RuleTree.stumps.transformers import MatrixTransformer

   composed = RuleTreeClassifier(max_depth=2, random_state=0,
                                 base_stumps=Stump(MatrixTransformer(), ClassificationSplitter(criterion="entropy")))
   dictionary = composed.fit(X, y).to_dict()

   print(dictionary["tree_type"], dictionary["tree_class"])
   print({name: value for name, value in dictionary["args"].items() if name != "base_stumps"})
   print(json.dumps(dictionary["args"]["base_stumps"], indent=1))

.. code-block:: text
   :caption: Output

   RuleTree.tree.RuleTreeClassifier RuleTreeClassifier
   {'max_depth': 2, 'max_leaf_nodes': inf, 'min_samples_split': 2, 'prune_useless_leaves': False, 'random_state': 0, 'stump_selection': 'random'}
   [
    [
     1.0,
     {
      "stump_type": "RuleTree.stumps.Stump",
      "stump_class": "Stump",
      "args": {
       "splitter": {
        "__estimator__": "RuleTree.stumps.splitters.ClassificationSplitter.ClassificationSplitter",
        "params": {
         "class_weight": null,
         "criterion": "entropy",
         "max_features": null,
         "min_impurity_decrease": 0.0,
         "min_samples_leaf": 1,
         "min_weight_fraction_leaf": 0.0,
         "random_state": null
        }
       },
       "transformer": {
        "__estimator__": "RuleTree.stumps.transformers.MatrixTransformer.MatrixTransformer",
        "params": {}
       }
      }
     },
     null
    ]
   ]

Nodes
~~~~~

``nodes`` is a flat list: each node refers to its children by their identifier (``left_node`` and
``right_node``, None for a leaf), so the order of the list does not matter. Every node has:

- ``node_id``: its path from the root (``"R"``, ``"Rl"``, ``"Rlr"``, ...);
- ``is_leaf``;
- ``prediction``, ``prediction_probability`` and ``log_odds``: the class (or value, or cluster) it
  predicts, the class probabilities (the standard deviation of the target, for regression) and
  their log-odds;
- ``prediction_classes_``: the classes of ``prediction_probability``;
- ``samples``: the number of training instances that reach it;
- for the internal nodes, ``balance_score`` and ``balance_score_global``: the fraction of the
  training set in the smaller side of the split (see ``get_balanced_stumps``).

.. code-block:: python

   for node in dictionary["nodes"]:
       print(node["node_id"], node["is_leaf"], node["samples"], node["prediction"],
             node["left_node"], node["right_node"])

.. code-block:: text
   :caption: Output

   R False 150 0 Rl Rr
   Rr False 100 1 Rrl Rrr
   Rrr True 46 2 None None
   Rrl True 54 1 None None
   Rl True 50 0 None None

The internal nodes also contain their fitted stump. Besides the module, the class and the
parameters, as in ``base_stumps``, the dictionary of a fitted stump has:

- ``data_type``: the type of data the stump was fitted on (see :doc:`multimodal`);
- ``transformer`` and ``splitter``: the fitted state of the two components, needed to apply the
  split (see `What the stumps save`_);
- ``impurity``: the impurity of the node and of the two children;
- the rule of the split, as returned by ``get_rule`` without rounding (``textual_rule``,
  ``threshold``, ``feature_idx``, ...). The rule is saved to be read without loading the tree:
  ``from_dict`` does not use it.

The stump has no identifier of its own in the dictionary: when the tree is loaded, it gets the
``node_id`` of its node back in ``node_id_`` (see :doc:`extending`).

.. code-block:: python

   root = dictionary["nodes"][0]

   print(root["stump_class"], root["data_type"], root["textual_rule"])
   print(json.dumps({"transformer": root["transformer"], "splitter": root["splitter"]},
                    cls=json_numpy_encoder))

.. code-block:: text
   :caption: Output

   Stump tabular X_3 <= 0.800000011920929
   {"transformer": {}, "splitter": {"feature_": 3, "threshold_": 0.800000011920929, "is_categorical_": false, "gain_": 0.9182958340544897, "impurity_": [1.584962500721156, 0.0, 1.0]}}


How a tree is rebuilt
---------------------

``from_dict``:

1. imports the module ``tree_type`` and takes the class ``tree_class`` from it;
2. rebuilds the base stumps: for each of them, it imports its class (``stump_type`` and
   ``stump_class``) and calls its ``dict_to_node``, which creates the stump from ``args``;
3. creates the tree with the saved parameters, ``tree_class(**args)``, and sets ``classes_``;
4. rebuilds each node with :meth:`~RuleTree.tree.RuleTreeNode.RuleTreeNode.dict_to_node`, and
   the stump of each internal node with the ``dict_to_node`` of its class. For a
   :class:`~RuleTree.stumps.Stump.Stump`, this creates the stump from its parameters, then new
   components (``_make_transformer`` and ``_make_splitter``) whose fitted state is restored with
   ``set_state``;
5. links the nodes through ``left_node`` and ``right_node``, and takes ``"R"`` as the root.

.. figure:: /_static/figures/saved_tree.svg
   :alt: The flat list of the saved nodes, each with its id and the ids of its children, and the tree
         rebuilt from it.
   :width: 100%

   The nodes are saved as a flat list, in any order; the tree is rebuilt from the ids of the
   children.

What is not saved
~~~~~~~~~~~~~~~~~

The dictionary contains what the tree needs to predict and to be described, not what was used
only to build it:

- the training data: the pivot and shapelet stumps save only their pivots and shapelets (see
  below), and the distance matrix of the pivot stumps is not saved;
- the cache of the stumps (``cache_``, see :doc:`lifecycle`) and the candidate splits
  (``candidate_splits_``), which are computed again at the next ``fit`` or ``partial_fit``;
- the state of the random generator of the tree.

So a loaded tree predicts, describes its rules (``get_rules``, ``export_graphviz``,
``get_pivots``, ...) and recomputes its statistics (``update_statistics``) as the original one,
and it can be grown further with ``partial_fit``:

.. code-block:: python

   print(hasattr(loaded, "cache_"), hasattr(loaded, "candidate_splits_"))

   loaded.set_params(max_depth=3).partial_fit(X, y)
   print("leaves after partial_fit:", len(loaded.get_leaf_nodes()))

.. code-block:: text
   :caption: Output

   False False
   leaves after partial_fit: 5

Since the random generator is not saved, a loaded tree that chooses its stumps at random among
several ones (``stump_selection="random"``) can choose differently, in ``partial_fit``, from the
original tree.


What the stumps save
--------------------

Each stump saves only the fitted state needed to apply its split:

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Stumps
     - Fitted state
   * - decision tree, XGBoost and fair stumps
     - the feature, the threshold (or the value of a categorical split), the gain and the impurity
       (the XGBoost splitter also its similarity scores)
   * - pivot stumps
     - the pivot (an instance of the training set, on the standardized numerical features), its
       index, the pivots of the node and the scaler of the numerical features
   * - multiple pivot stumps
     - the two pivots of the split and their indices, the pivots of the node, the scaler
   * - oblique pivot stumps
     - all the candidate pivots of the node, the scaler and the coefficients of the combination
   * - oblique stumps
     - the numerical features and the coefficients of the combination
   * - shapelet and proximity stumps
     - the shapelets (subsequences of the training series): all the selected ones for the shapelet
       stumps, the two of the split for the proximity stumps
   * - partial pivot and partial proximity stumps
     - the partial pivots (values of some features of training instances)

.. warning::

   The pivots and the shapelets are pieces of the training data. With sensitive data, a saved
   tree should be treated as the data it was trained on:

.. code-block:: python

   import numpy as np

   from RuleTree.stumps.classification import PivotTreeStumpClassifier

   pivot_tree = RuleTreeClassifier(max_depth=1, random_state=0, base_stumps=PivotTreeStumpClassifier())
   state = pivot_tree.fit(X, y).to_dict()["nodes"][0]["transformer"]

   print("pivot:", state["pivot_name"], np.round(state["X_split_instance"], 2).tolist())

.. code-block:: text
   :caption: Output

   pivot: 112 [1.16, -0.13, 0.99, 1.19]

The pivot is saved standardized, but the scaler (``state["scaler"]``) is saved too, so its
original values can be recovered.


Saving new stumps
-----------------

A stump written as a :class:`~RuleTree.stumps.Stump.Stump` (see :doc:`extending`) is saved with
no extra code, provided that:

- its parameters, and the ones of its components, are numbers, strings, lists, dictionaries or
  estimators with such parameters;
- the transformer saves its fitted state with ``get_state``, as a dictionary of plain values,
  and restores it with ``set_state``. A transformer without fitted state does not need them;
- the splitter keeps its fitted state in the attributes listed in ``_state``
  (``feature_``, ``threshold_``, ``is_categorical_``, ``gain_``, ``impurity_``). A splitter with
  other fitted attributes adds them to ``_state``, as
  :class:`~RuleTree.stumps.splitters.XGBoostSplitter` does;
- the classes are importable from their module. The saved tree records the module of each class,
  so the classes defined in a script run as ``__main__`` or in a notebook can be loaded only in
  that same script or notebook.

A stump that derives directly from
:class:`~RuleTree.base.RuleTreeBaseStump.RuleTreeBaseStump` implements ``node_to_dict``, which
returns a dictionary with at least ``stump_type`` (the module) and ``stump_class`` (the class),
and the class method ``dict_to_node``, which rebuilds the stump from it.

The quickest check of a new stump is a round trip through JSON, which also catches the values
that JSON cannot represent:

.. code-block:: python

   from RuleTree.stumps.classification import ObliqueDecisionTreeStumpClassifier

   stump = ObliqueDecisionTreeStumpClassifier(random_state=0).fit(X, y, context=None, node_id=None, cache=None)
   copy = type(stump).dict_to_node(json.loads(json.dumps(stump.node_to_dict(), cls=json_numpy_encoder)))

   print((copy.apply(X) == stump.apply(X)).all(), copy.get_rule()["textual_rule"])

.. code-block:: text
   :caption: Output

   True -0.085 * X_1 + 0.857 * X_2 <= 1.85

``tests/test_stumps.py`` runs this check on every stump of the library: adding a new stump to its
list ``STUMPS`` tests its saving.


Limits
------

The ensembles (:doc:`ensembles`) have no ``to_dict``. They, like the trees, can be saved with
``pickle`` or ``joblib``, which save the whole Python object.

.. warning::

   A pickle file must come from a trusted source, and is loaded only with compatible versions of
   RuleTree and of its dependencies.

.. code-block:: python

   import pickle

   from sklearn.datasets import load_diabetes

   from RuleTree import RuleTreeRegressor
   from RuleTree.ensemble.GBoostRegressor import GBoostRegressor

   X_reg, y_reg = load_diabetes(return_X_y=True)
   boosting = GBoostRegressor(base_estimator=RuleTreeRegressor(max_depth=2, random_state=0),
                              n_estimators=5, random_state=0).fit(X_reg, y_reg)

   with open("boosting.pkl", "wb") as f:
       pickle.dump(boosting, f)
   with open("boosting.pkl", "rb") as f:
       loaded_boosting = pickle.load(f)

   print(np.allclose(loaded_boosting.predict(X_reg), boosting.predict(X_reg)))

.. code-block:: text
   :caption: Output

   True

Moreover:

- the dictionary has no version number: a tree saved by a different version of RuleTree may not
  load, or may load with different defaults for the parameters added in between;
- an unfitted tree cannot be saved: ``to_dict`` raises ``NotFittedError``.
