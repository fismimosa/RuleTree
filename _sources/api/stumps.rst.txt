Stumps
======

The splits of the nodes, given to a tree with ``base_stumps``. A tree accepts only the stumps of
its task: classification stumps for :class:`~RuleTree.RuleTreeClassifier`, regression stumps for
:class:`~RuleTree.RuleTreeRegressor` and the clustering trees. See :doc:`/guides/stumps`.

Classification
--------------

.. currentmodule:: RuleTree.stumps.classification

.. autosummary::
   :toctree: generated/
   :nosignatures:

   DecisionTreeStumpClassifier
   ObliqueDecisionTreeStumpClassifier
   PivotTreeStumpClassifier
   MultiplePivotTreeStumpClassifier
   ObliquePivotTreeStumpClassifier
   MultipleObliquePivotTreeStumpClassifier
   PartialPivotTreeStumpClassifier
   PartialProximityTreeStumpClassifier
   ShapeletTreeStumpClassifier
   ProximityTreeStumpClassifier

Regression
----------

.. currentmodule:: RuleTree.stumps.regression

.. autosummary::
   :toctree: generated/
   :nosignatures:

   DecisionTreeStumpRegressor
   ObliqueDecisionTreeStumpRegressor
   PivotTreeStumpRegressor
   ShapeletTreeStumpRegressor
   ProximityTreeStumpRegressor
   FairTreeStumpRegressor
   XGBoostStumpRegressor

Building blocks
---------------

:class:`~RuleTree.stumps.Stump.Stump` composes a transformer and a splitter (see
:doc:`components`); the other classes hold what the classification and regression versions of a
stump have in common. See :doc:`/guides/extending`.

.. currentmodule:: RuleTree.stumps

.. autosummary::
   :toctree: generated/
   :nosignatures:

   ~Stump.Stump
   ~DecisionTreeStump.DecisionTreeStump
   ~ObliqueTreeStump.ObliqueTreeStump
   ~PivotTreeStump.PivotTreeStump
   ~PivotTreeStump.MultiplePivotTreeStump
   ~PivotTreeStump.ObliquePivotTreeStump
   ~ShapeletTreeStump.ShapeletTreeStump
   ~ProximityTreeStump.ProximityTreeStump
   ~ProximityTreeStump.PartialProximityTreeStump
