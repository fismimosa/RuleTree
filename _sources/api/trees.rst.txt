Trees
=====

The estimators of RuleTree, with the scikit-learn API. See :doc:`/guides/getting_started`.

.. currentmodule:: RuleTree

Classification and regression
-----------------------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   RuleTreeClassifier
   RuleTreeRegressor

Clustering
----------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   RuleTreeCluster
   RuleTreeClusterClassifier
   RuleTreeClusterRegressor

Trees of the ensembles
----------------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   ~tree.RuleTreeXGBoostRegressor.RuleTreeXGBoostRegressor

Base class and nodes
--------------------

:class:`~RuleTree.tree.RuleTree.RuleTree` has the methods shared by all the trees (rules,
drawing, saving, growing); each node of a fitted tree is a
:class:`~RuleTree.tree.RuleTreeNode.RuleTreeNode`. See :doc:`/guides/inspecting`.

.. autosummary::
   :toctree: generated/
   :nosignatures:

   ~tree.RuleTree.RuleTree
   ~tree.RuleTreeNode.RuleTreeNode
