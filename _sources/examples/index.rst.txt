Examples
========

Notebooks that train a tree end to end, from loading the data to reading the rules. They are the
notebooks of the `Examples <https://github.com/fismimosa/RuleTree/tree/main/Examples>`_ folder of
the repository, run again at every build of this documentation.

.. grid:: 1 2 2 2
   :gutter: 3

   .. grid-item-card:: Classification
      :link: ruletreeclassifier_example
      :link-type: doc

      A RuleTreeClassifier on the breast cancer data: rules, predictions, probabilities and the
      contributions of the features.

   .. grid-item-card:: Regression
      :link: ruletreeregressor_example
      :link-type: doc

      A RuleTreeRegressor on the diabetes data: rules, predictions and the contributions of the
      features.

   .. grid-item-card:: Clustering
      :link: ruletreecluster_example
      :link-type: doc

      A RuleTreeCluster on the breast cancer data, compared with the classes.

   .. grid-item-card:: Pivot trees
      :link: pivottreeclassifier_example
      :link-type: doc

      A classifier whose splits are distances from instances of the training set.

   .. grid-item-card:: Classification of a DataFrame
      :link: tabular_classification
      :link-type: doc

      A tree trained on the columns of a pandas DataFrame, with classes as strings, and its
      Graphviz source.

.. toctree::
   :hidden:
   :glob:

   *
