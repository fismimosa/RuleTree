User guide
==========

The guides explain how to use RuleTree, from the objects a tree is made of to writing new stumps.
Their examples run as they are, and print what is shown under them. To install RuleTree and train
a first tree, start from :doc:`getting_started`.

.. grid:: 1 2 3 3
   :gutter: 3

   .. grid-item-card:: :octicon:`light-bulb;1.2em` How RuleTree works
      :link: concepts
      :link-type: doc

      Trees, nodes, stumps, transformers and splitters.

   .. grid-item-card:: :octicon:`git-branch;1.2em` Choosing the splits
      :link: stumps
      :link-type: doc

      The available stumps, which one to use and how to combine them.

   .. grid-item-card:: :octicon:`stack;1.2em` Several data types
      :link: multimodal
      :link-type: doc

      Tabular data, time series and images, alone or together.

   .. grid-item-card:: :octicon:`eye;1.2em` Reading the tree
      :link: inspecting
      :link-type: doc

      Rules, drawings, nodes, leaves and contributions of the features.

   .. grid-item-card:: :octicon:`sync;1.2em` Growing and updating
      :link: lifecycle
      :link-type: doc

      ``partial_fit``, ``update_statistics`` and the cache of the stumps.

   .. grid-item-card:: :octicon:`download;1.2em` Saving and loading
      :link: saving
      :link-type: doc

      What ``to_dict`` saves and how ``from_dict`` rebuilds a tree.

   .. grid-item-card:: :octicon:`versions;1.2em` Ensembles
      :link: ensembles
      :link-type: doc

      Gradient boosting, XGBoost and explainable boosting of RuleTrees.

   .. grid-item-card:: :octicon:`graph;1.2em` Evaluating the models
      :link: evaluating
      :link-type: doc

      Metrics of predictions, clusters, size of the trees and fairness.

   .. grid-item-card:: :octicon:`tools;1.2em` Extending the library
      :link: extending
      :link-type: doc

      New transformers, splitters and stumps.

.. toctree::
   :hidden:
   :maxdepth: 2

   concepts
   stumps
   multimodal
   inspecting
   lifecycle
   saving
   ensembles
   evaluating
   extending
