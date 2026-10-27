Transformers and splitters
==========================

A :class:`~RuleTree.stumps.Stump.Stump` is a *transformer*, which turns the data into a feature
matrix, followed by a *splitter*, which finds the best split of that matrix. Any transformer can be
combined with any splitter. See :doc:`/guides/extending`.

Transformers
------------

.. currentmodule:: RuleTree.stumps.transformers

.. autosummary::
   :toctree: generated/
   :nosignatures:

   MatrixTransformer
   PivotTransformer
   PartialPivotTransformer
   MultiplePivotTransformer
   ObliquePivotTransformer
   ObliqueTransformer
   HouseholderTransformer
   BivariateTransformer
   ShapeletTransformer
   ProximityTransformer
   PartialProximityTransformer

Splitters
---------

.. currentmodule:: RuleTree.stumps.splitters

.. autosummary::
   :toctree: generated/
   :nosignatures:

   ClassificationSplitter
   RegressionSplitter
   XGBoostSplitter
   FairSplitter
   ~DecisionTreeSplitter.DecisionTreeSplitter
