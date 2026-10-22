API reference
=============

The classes and functions of RuleTree, grouped by what they are for. The user guide explains how
to use them together; each page here lists the parameters, attributes and methods of one object.

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Page
     - Content
   * - :doc:`trees`
     - the trees: classification, regression, clustering, and their nodes
   * - :doc:`stumps`
     - the stumps of classification and regression, ready to be given to a tree
   * - :doc:`components`
     - the transformers and the splitters that stumps are made of
   * - :doc:`base`
     - the base classes to write new trees, stumps, transformers and splitters
   * - :doc:`ensemble`
     - boosting of RuleTrees
   * - :doc:`evaluation`
     - metrics of predictions, clusters, size of the trees and fairness
   * - :doc:`utils`
     - exceptions, constants and helper functions

.. toctree::
   :hidden:

   trees
   stumps
   components
   base
   ensemble
   evaluation
   utils
