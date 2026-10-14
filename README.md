# RuleTree

RuleTree builds interpretable decision trees one node at a time. Each node is split by a *stump*:
a threshold on a feature, the distance from an instance of the training set (a *pivot*), the
distance of a time series from a *shapelet*, a fair split, ... The same tree can mix different
kinds of stumps and different types of data (tabular data, time series, images), and its rules can
be read, drawn and saved.

- Classification, regression and clustering trees with the scikit-learn API.
- Stumps for tabular data, time series and images, usable together in the same tree.
- Rules as text or Graphviz drawings, feature contributions of each prediction.
- Trees saved to and loaded from JSON, grown further with `partial_fit`.
- Gradient boosting, XGBoost and explainable boosting machines of RuleTrees.
- New stumps written by combining a *transformer* (the features) and a *splitter* (the split).

## Installation

RuleTree requires Python 3.10 or later. From the root of the repository:

```bash
pip install .
pip install ".[graphviz]"   # optional: drawing the trees (also requires the Graphviz binaries)
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
- [Stumps](docs/guides/stumps.rst): the available stumps and how to combine them.
- [Tabular data, time series and images](docs/guides/multimodal.rst).
- [Reading the tree](docs/guides/inspecting.rst): rules, drawings, nodes, feature contributions.
- [Saving, growing and updating trees](docs/guides/lifecycle.rst).
- [Ensembles](docs/guides/ensembles.rst): boosting of RuleTrees.
- [Extending the library](docs/guides/extending.rst): writing new stumps, transformers and splitters.

The HTML documentation, with the API reference, is built with Sphinx:

```bash
pip install -r requirements.docs.txt
make -C docs html   # docs/_build/html/index.html
```

More examples are in the notebooks of [`Examples`](Examples).

## Tests

```bash
pip install ".[test]"
pytest
```

The tests include the docstring examples and the code of the guides, whose printed output is
compared with the one in the guides.
