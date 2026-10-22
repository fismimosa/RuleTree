Utilities
=========

Exceptions
----------

.. currentmodule:: RuleTree.exceptions

.. autosummary::
   :toctree: generated/
   :nosignatures:

   MissingOptionalDependencyError
   NoSplitFoundWarning

Optional dependencies
---------------------

.. currentmodule:: RuleTree.utils.optional

.. autosummary::
   :toctree: generated/
   :nosignatures:

   import_optional

Constants
---------

.. automodule:: RuleTree.utils.define
   :no-members:

Saving
------

.. currentmodule:: RuleTree.utils.data_utils

.. autosummary::
   :toctree: generated/
   :nosignatures:

   json_numpy_encoder

Fairness and privacy
--------------------

.. currentmodule:: RuleTree.utils.fairness_metrics

.. autosummary::
   :toctree: generated/
   :nosignatures:

   balance_metric
   max_fairness_cost
   privacy_metric

Clustering
----------

.. currentmodule:: RuleTree.utils.bic_estimator

.. autosummary::
   :toctree: generated/
   :nosignatures:

   bic

Shapelets
---------

The shapelet transforms used by the shapelet and proximity stumps (they need the ``shapelets``
extra).

.. currentmodule:: RuleTree.utils.shapelet_transform

.. autosummary::
   :toctree: generated/
   :nosignatures:

   ~Shapelets.Shapelets
   ~TabularShapelets.TabularShapelets

Factor analysis
---------------

The principal component analyses used by the clustering trees, from the ``light_famd`` package.

.. currentmodule:: RuleTree.utils.light_famd

.. autosummary::
   :toctree: generated/
   :nosignatures:

   PCA
   CA
   MCA
   MFA
   FAMD
