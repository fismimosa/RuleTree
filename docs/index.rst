:html_theme.sidebar_secondary.remove: true

RuleTree
========

.. rst-class:: hero-lead

   Interpretable decision trees built one node at a time, where each node can be split in a
   different way.

RuleTree splits each node with a *stump*: a threshold on a feature or on a combination of a few
features, the distance from an instance of the training set (a *pivot*), the distance of a time
series from a *shapelet*, which of two pivots or shapelets is closer, a fair split, ... The same
tree can mix different kinds of stumps and different types of data (tabular data, time series,
images), and its rules can be read, drawn and saved.

.. grid:: 1 2 4 4
   :gutter: 3
   :class-container: feature-grid

   .. grid-item-card:: :octicon:`git-branch;1.2em` Many kinds of splits
      :link: guides/stumps
      :link-type: doc

      Thresholds, oblique splits, pivots, shapelets, fair splits, mixed in one tree.

   .. grid-item-card:: :octicon:`stack;1.2em` Several data types
      :link: guides/multimodal
      :link-type: doc

      Tabular data, time series and images, together in the same tree.

   .. grid-item-card:: :octicon:`eye;1.2em` Rules you can read
      :link: guides/inspecting
      :link-type: doc

      Rules as text or drawings, node ids, contributions of the features.

   .. grid-item-card:: :octicon:`plug;1.2em` scikit-learn API
      :link: guides/getting_started
      :link-type: doc

      Classification, regression and clustering, in pipelines and grid searches.


Quick example
-------------

.. tab-set::

   .. tab-item:: Classification

      .. code-block:: python

         from sklearn.datasets import load_iris

         from RuleTree import RuleTreeClassifier

         X, y = load_iris(return_X_y=True)
         model = RuleTreeClassifier(max_depth=2, random_state=0).fit(X, y)
         model.print_rules(model.get_rules(columns_names=["sepal length", "sepal width",
                                                           "petal length", "petal width"]))

      .. code-block:: text
         :caption: Output

         |--- petal width <= 0.8
         |    output: 0
         |--- petal width > 0.8
         |   |--- petal width <= 1.75
         |   |    output: 1
         |   |--- petal width > 1.75
         |   |    output: 2

   .. tab-item:: Regression

      .. code-block:: python

         from sklearn.datasets import load_diabetes

         from RuleTree import RuleTreeRegressor

         data = load_diabetes()
         model = RuleTreeRegressor(max_depth=2, random_state=0).fit(data.data, data.target)
         model.print_rules(model.get_rules(columns_names=data.feature_names))

      .. code-block:: text
         :caption: Output

         |--- s5 <= -0.004
         |   |--- bmi <= 0.006
         |   |    output: 96.31
         |   |--- bmi > 0.006
         |   |    output: 159.74
         |--- s5 > -0.004
         |   |--- bmi <= 0.015
         |   |    output: 162.68
         |   |--- bmi > 0.015
         |   |    output: 225.88

   .. tab-item:: Clustering

      .. code-block:: python

         from sklearn.datasets import make_blobs

         from RuleTree import RuleTreeCluster

         X, _ = make_blobs(n_samples=300, centers=3, cluster_std=0.5, random_state=0)
         model = RuleTreeCluster(max_depth=2, random_state=0).fit(X)
         model.print_rules(model.get_rules(columns_names=["x", "y"]))

      .. code-block:: text
         :caption: Output

         |--- y <= 1.772
         |   |--- x <= -0.131
         |   |    output: 3
         |   |--- x > -0.131
         |   |    output: 4
         |--- y > 1.772
         |   |--- x <= -0.411
         |   |    output: 5
         |   |--- x > -0.411
         |   |    output: 6

Install it with ``pip install RuleTree`` (see :doc:`guides/getting_started` for the optional
features).


Where to go next
----------------

.. grid:: 1 2 3 3
   :gutter: 3

   .. grid-item-card:: :octicon:`rocket;1.2em` Getting started
      :link: guides/getting_started
      :link-type: doc

      Install RuleTree and train a first tree for each task.

   .. grid-item-card:: :octicon:`light-bulb;1.2em` How RuleTree works
      :link: guides/concepts
      :link-type: doc

      Trees, nodes, stumps and their components, in one page.

   .. grid-item-card:: :octicon:`book;1.2em` User guide
      :link: guides/index
      :link-type: doc

      Stumps, data types, rules, saving, ensembles, evaluation, extensions.

   .. grid-item-card:: :octicon:`file-code;1.2em` Examples
      :link: examples/index
      :link-type: doc

      Notebooks that train a tree end to end.

   .. grid-item-card:: :octicon:`code-square;1.2em` API reference
      :link: api/index
      :link-type: doc

      Every class and function, grouped by topic.

   .. grid-item-card:: :octicon:`mortar-board;1.2em` Citing RuleTree
      :link: citing
      :link-type: doc

      BibTeX entries of the library and of its papers.


.. toctree::
   :hidden:
   :maxdepth: 2

   Getting started <guides/getting_started>
   User guide <guides/index>
   Examples <examples/index>
   API reference <api/index>
   Glossary <glossary>
   Citing <citing>
