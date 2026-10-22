# RuleTree

Interpretable decision trees whose nodes can be split in different ways (a threshold on a feature,
the distance from an example, the distance of a time series from a pattern, ...), on tabular data,
time series and images. Documentation: https://fismimosa.github.io/RuleTree/

- Classification, regression (also with several targets) and clustering trees with the scikit-learn API.
- Stumps for tabular data, time series and images, usable together in the same tree.
- Rules as text or drawings (matplotlib or Graphviz), feature contributions of each prediction.
- Trees saved to and loaded from JSON, grown further with `partial_fit`.
- Gradient boosting, XGBoost and explainable boosting machines of RuleTrees.
- Metrics to evaluate the models: predictions, clusters, size of the trees and fairness.
- New stumps written by combining a *transformer* (the features) and a *splitter* (the split).

## Installation

RuleTree requires Python 3.10 or later:

```bash
pip install RuleTree                # or, from the root of the repository: pip install .
pip install "RuleTree[shapelets]"   # optional: shapelet, proximity and partial pivot stumps (numba)
pip install "RuleTree[plot]"        # optional: drawing the trees with matplotlib (plot_tree)
pip install "RuleTree[graphviz]"    # optional: Graphviz drawings (also requires the Graphviz binaries)
pip install "RuleTree[all]"         # all the optional features
```

## Example

```python
from sklearn.datasets import load_iris

from RuleTree import RuleTreeClassifier

X, y = load_iris(return_X_y=True)
model = RuleTreeClassifier(max_depth=2, random_state=0).fit(X, y)
model.print_rules(model.get_rules(columns_names=["sepal length", "sepal width",
                                                  "petal length", "petal width"]))
```

```text
|--- petal width <= 0.8
|    output: 0
|--- petal width > 0.8
|   |--- petal width <= 1.75
|   |    output: 1
|   |--- petal width > 1.75
|   |    output: 2
```

Trees with other stumps and other types of data:

```python
from RuleTree.stumps.classification import DecisionTreeStumpClassifier, ShapeletTreeStumpClassifier

# X: tabular data, X_ts: time series of shape (n_samples, n_signals, length)
model = RuleTreeClassifier(max_depth=3, stump_selection="best",
                           base_stumps=[DecisionTreeStumpClassifier(), ShapeletTreeStumpClassifier()])
model.fit(X=X, X_ts=X_ts, y=y)
```

## Documentation

The user guide is in [`docs/guides`](docs/guides):

- [Getting started](docs/guides/getting_started.rst): classification, regression, clustering.
- [How RuleTree works](docs/guides/concepts.rst): trees, nodes, stumps and their components.
- [Stumps](docs/guides/stumps.rst): the available stumps, which one to use and how to combine them.
- [Tabular data, time series and images](docs/guides/multimodal.rst).
- [Reading the tree](docs/guides/inspecting.rst): rules, drawings, nodes, feature contributions.
- [Growing and updating trees](docs/guides/lifecycle.rst): `partial_fit`, `update_statistics` and the cache of the stumps.
- [Saving and loading trees](docs/guides/saving.rst): what `to_dict` saves and how `from_dict` rebuilds a tree.
- [Ensembles](docs/guides/ensembles.rst): boosting of RuleTrees.
- [Evaluating the models](docs/guides/evaluating.rst): metrics of predictions, clusters, tree size and fairness; all but the tree size need only the output of a model, so they evaluate any model.
- [Extending the library](docs/guides/extending.rst): writing new stumps, transformers and splitters.

The HTML documentation, with the API reference, is built with Sphinx:

```bash
pip install -r requirements.txt -r requirements.docs.txt
make -C docs html   # docs/_build/html/index.html
```

The figures of the guides (`docs/_static/figures`, each in a light and a dark version) are drawn
from fitted models by `python docs/make_figures.py` (it needs matplotlib): run it again when a
change affects them.

More examples are in the notebooks of [`Examples`](Examples), which the HTML documentation runs
again at every build. To cite RuleTree, see [`docs/citing.rst`](docs/citing.rst).

## Tests

```bash
pip install ".[test]"
pytest
```

The tests include the docstring examples and the code of the guides, whose printed output is
compared with the one in the guides.
