API reference
=============

The classes and functions of RuleTree, grouped by what they are for. The user guide explains how
to use them together; each page here lists the parameters, attributes and methods of one object.

.. grid:: 1 2 3 3
   :gutter: 3

   .. grid-item-card:: :octicon:`git-branch;1.2em` Trees
      :link: trees
      :link-type: doc

      The trees: classification, regression, clustering, and their nodes.

   .. grid-item-card:: :octicon:`workflow;1.2em` Stumps
      :link: stumps
      :link-type: doc

      The stumps of classification and regression, ready to be given to a tree.

   .. grid-item-card:: :octicon:`tools;1.2em` Transformers and splitters
      :link: components
      :link-type: doc

      The two parts that stumps are made of.

   .. grid-item-card:: :octicon:`package;1.2em` Base classes
      :link: base
      :link-type: doc

      The interfaces to write new trees, stumps, transformers and splitters.

   .. grid-item-card:: :octicon:`stack;1.2em` Ensembles
      :link: ensemble
      :link-type: doc

      Boosting of RuleTrees.

   .. grid-item-card:: :octicon:`graph;1.2em` Evaluation
      :link: evaluation
      :link-type: doc

      Metrics of predictions, clusters, size of the trees and fairness.

   .. grid-item-card:: :octicon:`gear;1.2em` Utilities
      :link: utils
      :link-type: doc

      Exceptions, constants and helper functions.

.. toctree::
   :hidden:

   trees
   stumps
   components
   base
   ensemble
   evaluation
   utils
