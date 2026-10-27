Ensembles
=========

The :mod:`RuleTree.ensemble` module combines RuleTrees by boosting. The ensembles take tabular data
(``fit(X, y)``, ``predict(X)``) and stop early when the error on a validation set, taken from the
training set (``val_size``), does not improve for ``patience`` iterations.

The ``random_state`` of an ensemble sets the validation set and makes the ensemble reproducible:
each tree whose base estimator has no ``random_state`` gets a different seed drawn from it, while a
``random_state`` given to the base estimator is kept for all the trees.

.. code-block:: python

   from sklearn.datasets import load_diabetes
   from sklearn.metrics import r2_score
   from sklearn.model_selection import train_test_split

   from RuleTree import RuleTreeRegressor

   X, y = load_diabetes(return_X_y=True)
   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

   single_tree = RuleTreeRegressor(max_depth=3, random_state=0).fit(X_train, y_train)
   print("single tree, R2 on the test set:", round(r2_score(y_test, single_tree.predict(X_test)), 3))

.. code-block:: text
   :caption: Output

   single tree, R2 on the test set: 0.188


Gradient boosting
-----------------

:class:`~RuleTree.ensemble.GBoostRegressor.GBoostRegressor` fits each tree on the residuals of the
previous ones (squared loss). The base estimator can be any RuleTreeRegressor, with any stumps:

.. themed-figure:: /_static/figures/boosting.svg
   :alt: F0, the mean of the target, followed by tree 1, tree 2 and tree n, each fitted on the
         residuals of the previous prediction and added with the learning rate lr.
   :width: 100%

   Each tree is fitted on the residuals of the sum of the previous ones, and added with the
   learning rate ``learning_rate``.

.. code-block:: python

   from RuleTree.ensemble.GBoostRegressor import GBoostRegressor

   boosting = GBoostRegressor(base_estimator=RuleTreeRegressor(max_depth=2, random_state=0),
                              n_estimators=50, learning_rate=0.1, random_state=0)
   boosting.fit(X_train, y_train)

   print("trees:", len(boosting.estimators_))
   print("R2 on the test set:", round(r2_score(y_test, boosting.predict(X_test)), 3))

.. code-block:: text
   :caption: Output

   trees: 45
   R2 on the test set: 0.339

:class:`~RuleTree.ensemble.GBoostClassifier.GBoostClassifier` fits, for each class, a boosting of
regression trees on the log-odds of the class (one-vs-rest, logistic loss); the classes are fitted
in parallel processes (``n_jobs``):

.. code-block:: python

   from sklearn.datasets import load_breast_cancer

   from RuleTree.ensemble.GBoostClassifier import GBoostClassifier

   X_bc, y_bc = load_breast_cancer(return_X_y=True)
   X_bc_train, X_bc_test, y_bc_train, y_bc_test = train_test_split(X_bc, y_bc, test_size=0.3, random_state=0)

   classifier = GBoostClassifier(base_estimator=RuleTreeRegressor(max_depth=2, random_state=0),
                                 n_estimators=20, n_jobs=1, random_state=0)
   classifier.fit(X_bc_train, y_bc_train)

   print("accuracy on the test set:", round((classifier.predict(X_bc_test) == y_bc_test).mean(), 3))

.. code-block:: text
   :caption: Output

   accuracy on the test set: 0.947

.. warning::

   ``GBoostClassifier.predict`` returns the right classes only when they are
   ``0, ..., n_classes - 1``: encode other labels first, e.g. with
   ``sklearn.preprocessing.LabelEncoder``.


XGBoost
-------

:class:`~RuleTree.ensemble.XGBoostRegressor.XGBoostRegressor` is a gradient boosting with the
trees of XGBoost (:class:`~RuleTree.tree.RuleTreeXGBoostRegressor.RuleTreeXGBoostRegressor`): the
splits maximize the XGBoost gain, the splits with a gain lower than ``gamma`` are pruned, and each
leaf predicts its regularized value ``sum(residuals) / (n + lam)``.

.. code-block:: python

   from RuleTree.ensemble.XGBoostRegressor import XGBoostRegressor
   from RuleTree.stumps.regression import XGBoostStumpRegressor
   from RuleTree.tree.RuleTreeXGBoostRegressor import RuleTreeXGBoostRegressor

   xgboost = XGBoostRegressor(base_estimator=RuleTreeXGBoostRegressor(
       max_depth=2, lam=1, base_stumps=XGBoostStumpRegressor(lam=1)), n_estimators=20)
   xgboost.fit(X_train, y_train)

   print("R2 on the test set:", round(r2_score(y_test, xgboost.predict(X_test)), 3))

.. code-block:: text
   :caption: Output

   R2 on the test set: 0.332

Without ``base_estimator``, the trees use as many processes as CPUs.


Explainable boosting
--------------------

:class:`~RuleTree.ensemble.EBMRegressor.EBMRegressor` is an explainable boosting machine: each tree
is fitted on a single feature or on a pair of features, so the prediction is a sum of functions of
one or two features, which can be plotted and inspected.

.. code-block:: python

   from RuleTree.ensemble.EBMRegressor import EBMRegressor

   ebm = EBMRegressor(base_estimator=RuleTreeRegressor(max_depth=1, random_state=0),
                      n_iterations=20, learning_rate=0.1, random_state=0)
   ebm.fit(X_train, y_train)

   print("terms:", len(ebm.estimators_), "- e.g.", list(ebm.estimators_)[:3], list(ebm.estimators_)[-1])
   print("R2 on the test set:", round(r2_score(y_test, ebm.predict(X_test)), 3))

.. code-block:: text
   :caption: Output

   terms: 55 - e.g. [0, 1, 2] (8, 9)
   R2 on the test set: 0.344

The terms are the features (their index) and the pairs of features; ``estimators_[term]`` is the
list of trees of the term.
