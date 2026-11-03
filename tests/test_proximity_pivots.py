import json

import numpy as np
import pytest
from sklearn.datasets import load_wine
from sklearn.metrics import pairwise_distances

from RuleTree import RuleTreeClassifier
from RuleTree.stumps.classification import (ObliquePivotTreeStumpClassifier, ObliqueProximityPivotTreeStumpClassifier,
                                            PivotTreeStumpClassifier, ProximityPivotTreeStumpClassifier)
from RuleTree.tree.RuleTree import RuleTree
from RuleTree.utils.data_utils import json_numpy_encoder

STUMPS = [
    ProximityPivotTreeStumpClassifier,
    ObliquePivotTreeStumpClassifier,
    lambda **kwargs: ObliquePivotTreeStumpClassifier(oblique_split_type="bivariate", **kwargs),
    ObliqueProximityPivotTreeStumpClassifier,
    lambda **kwargs: ObliqueProximityPivotTreeStumpClassifier(oblique_split_type="bivariate", **kwargs),
]
IDS = ["ProximityPivot", "ObliquePivot-householder", "ObliquePivot-bivariate", "ObliqueProximityPivot-householder",
       "ObliqueProximityPivot-bivariate"]


def json_round_trip(dictionary):
    return json.loads(json.dumps(dictionary, cls=json_numpy_encoder))


def distances_from(stump, X, pivots):
    """Distances from training instances, as computed by the stump: standardized numerical features."""
    X_scaled = stump.transformer_.scaler_.transform(X)
    return pairwise_distances(X_scaled, X_scaled[list(pivots)], metric=stump.distance_measure)


@pytest.mark.parametrize("stump_class", [ProximityPivotTreeStumpClassifier, ObliqueProximityPivotTreeStumpClassifier])
def test_proximity_pivot_rule_is_the_split(stump_class, iris):
    """P_a <= P_b: the instances closer to the training instance a than to b go left."""
    X, y = iris
    stump = stump_class(random_state=0).fit(X, y, context=None, node_id=None, cache=None)

    a, b = stump.feature_original
    distances = distances_from(stump, X, (a, b))

    assert stump.get_rule()["textual_rule"] == f"P_{a} <= P_{b}"
    assert np.array_equal(stump.apply(X), np.where(distances[:, 0] <= distances[:, 1], 1, 2))
    assert y[a] != y[b]  # pivots of different classes


@pytest.mark.parametrize("oblique_split_type", ["householder", "bivariate"])
def test_oblique_pivot_rule_is_the_split(oblique_split_type, iris):
    X, y = iris
    stump = ObliquePivotTreeStumpClassifier(oblique_split_type=oblique_split_type, random_state=0)
    stump = stump.fit(X, y, context=None, node_id=None, cache=None)

    rule = stump.get_rule(float_precision=None)
    left = distances_from(stump, X, rule["feature_idx"]) @ np.array(rule["coefficients"]) <= rule["threshold"]

    assert np.array_equal(stump.apply(X), np.where(left, 1, 2))
    assert all(f"P_{pivot}" in rule["textual_rule"] for pivot in rule["feature_idx"])


@pytest.mark.parametrize("make_stump", STUMPS, ids=IDS)
def test_pivots_of_the_split(make_stump, iris):
    X, y = iris
    model = RuleTreeClassifier(max_depth=2, base_stumps=make_stump(), random_state=0).fit(X, y)
    loaded = RuleTree.from_dict(json_round_trip(model.to_dict()))

    assert model.score(X, y) > 0.8
    assert np.array_equal(loaded.predict(X), model.predict(X))
    for node_id, node in model.get_predicates().items():
        pivots = node.stump.get_pivots()
        assert set(pivots["used"]) <= set(pivots["candidates"])
        assert pivots["candidates"] == pivots["descriptives"] + pivots["discriminatives"]
        assert loaded.get_node_by_id(node_id).stump.get_pivots() == pivots
        assert loaded.get_node_by_id(node_id).stump.get_rule() == node.stump.get_rule()


def test_invalid_oblique_split_type():
    for stump_class in (ObliquePivotTreeStumpClassifier, ObliqueProximityPivotTreeStumpClassifier):
        with pytest.raises(ValueError, match="oblique_split_type"):
            stump_class(oblique_split_type="diagonal")


def test_pivot_stumps_together():
    """All the pivot stumps in the same tree, chosen by gain: on wine, more than one kind is chosen."""
    X, y = load_wine(return_X_y=True)
    model = RuleTreeClassifier(max_depth=3, stump_selection="best", random_state=0,
                               base_stumps=[PivotTreeStumpClassifier(), ProximityPivotTreeStumpClassifier(),
                                            ObliquePivotTreeStumpClassifier()])

    model.fit(X, y)

    assert model.score(X, y) >= 0.9
    assert len({type(node.stump) for node in model.get_predicates().values()}) > 1
    loaded = RuleTree.from_dict(json_round_trip(model.to_dict()))
    assert np.array_equal(loaded.predict(X), model.predict(X))


@pytest.mark.parametrize("old, new", [("MultiplePivotTreeStumpClassifier", ProximityPivotTreeStumpClassifier),
                                      ("MultipleObliquePivotTreeStumpClassifier",
                                       ObliqueProximityPivotTreeStumpClassifier)])
def test_old_names(old, new, iris):
    """The names before the renaming still work, with a warning, also for the trees saved with them."""
    import RuleTree.stumps.classification as classification

    with pytest.warns(FutureWarning, match=new.__name__):
        assert getattr(classification, old) is new

    X, y = iris
    model = RuleTreeClassifier(max_depth=2, random_state=0, base_stumps=new()).fit(X, y)
    saved = json.dumps(model.to_dict(), cls=json_numpy_encoder).replace(new.__name__, old)  # as saved before
    with pytest.warns(FutureWarning):
        loaded = RuleTree.from_dict(json.loads(saved))
    assert np.array_equal(loaded.predict(X), model.predict(X))


def test_old_transformer_name():
    import RuleTree.stumps.transformers as transformers
    from RuleTree.stumps.transformers import ProximityPivotTransformer

    with pytest.warns(FutureWarning, match="ProximityPivotTransformer"):
        assert transformers.MultiplePivotTransformer is ProximityPivotTransformer


def test_old_transformer_in_a_saved_stump(iris):
    from RuleTree.stumps.Stump import Stump
    from RuleTree.stumps.splitters import ClassificationSplitter
    from RuleTree.stumps.transformers import ProximityPivotTransformer

    X, y = iris
    model = RuleTreeClassifier(max_depth=2, random_state=0,
                               base_stumps=Stump(ProximityPivotTransformer(), ClassificationSplitter())).fit(X, y)
    saved = json.dumps(model.to_dict(), cls=json_numpy_encoder).replace("ProximityPivotTransformer",
                                                                         "MultiplePivotTransformer")
    with pytest.warns(FutureWarning, match="ProximityPivotTransformer"):
        loaded = RuleTree.from_dict(json.loads(saved))
    assert np.array_equal(loaded.predict(X), model.predict(X))


@pytest.mark.parametrize("stump", [PivotTreeStumpClassifier, ProximityPivotTreeStumpClassifier,
                                   ObliquePivotTreeStumpClassifier, ObliqueProximityPivotTreeStumpClassifier])
def test_pivot_stumps_on_images_and_time_series(stump):
    """Images and time series are flattened: the same tree as on the flattened data, saved and loaded."""
    from sklearn.datasets import load_digits

    digits = load_digits()
    images, y = digits.images[:300], digits.target[:300] % 3
    series = images.reshape(len(images), 1, -1)
    flat = RuleTreeClassifier(max_depth=2, random_state=0, base_stumps=stump()).fit(images.reshape(len(y), -1), y)

    for argument, data in [("X_img", images), ("X_ts", series)]:
        model = RuleTreeClassifier(max_depth=2, random_state=0, base_stumps=stump()).fit(y=y, **{argument: data})
        loaded = RuleTree.from_dict(json_round_trip(model.to_dict()))

        assert np.array_equal(model.predict(**{argument: data}), flat.predict(images.reshape(len(y), -1)))
        assert np.array_equal(loaded.predict(**{argument: data}), model.predict(**{argument: data}))


class CountingEmbedder:
    """A stateless embedder (the mean of each row and of its halves) that counts the instances it transforms."""

    calls = []

    def get_params(self, deep=True):
        return {}

    def set_params(self, **params):
        return self

    def fit(self, X=None, y=None):
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float).reshape(len(X), -1)
        CountingEmbedder.calls.append(len(X))
        half = X.shape[1] // 2
        return np.column_stack([X.mean(axis=1), X[:, :half].mean(axis=1), X[:, half:].mean(axis=1)])


@pytest.mark.parametrize("stump", [PivotTreeStumpClassifier, ProximityPivotTreeStumpClassifier,
                                   ObliquePivotTreeStumpClassifier])
def test_embeddings_computed_once_per_instance(stump):
    """In fit, the whole data set once (not once per node); in predict, only the instances not seen yet."""
    from sklearn.datasets import load_digits

    images, y = load_digits(return_X_y=True)[0][:300].reshape(300, 8, 8), load_digits(return_X_y=True)[1][:300] % 2
    CountingEmbedder.calls.clear()
    model = RuleTreeClassifier(max_depth=3, random_state=0, base_stumps=stump(embedder=CountingEmbedder()))

    model.fit(X_img=images, y=y)
    assert CountingEmbedder.calls == [len(images)] and len(model.get_predicates()) > 1

    model.predict(X_img=images)  # the training instances: already embedded
    model.predict(X_img=images[:10] + 1)  # ten new instances, embedded once also if they cross several nodes
    assert CountingEmbedder.calls == [len(images), 10]


def test_texts_need_an_embedder():
    assert "text" not in PivotTreeStumpClassifier().supported_data_types
    assert "text" in PivotTreeStumpClassifier(embedder=CountingEmbedder()).supported_data_types


def test_image_embedder_inside_the_stump():
    """A pretrained network as embedder, saved by its parameters and fitted again when the tree is loaded."""
    pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    from sklearn.datasets import load_digits

    from RuleTree.utils.embeddings import ImageEmbedder

    images, y = load_digits(return_X_y=True)[0][:60].reshape(60, 8, 8) / 16, load_digits(return_X_y=True)[1][:60] % 2
    embedder = ImageEmbedder(weights=None, image_size=32, device="cpu")  # random weights: nothing downloaded
    model = RuleTreeClassifier(max_depth=1, random_state=0, base_stumps=PivotTreeStumpClassifier(embedder=embedder))
    model.fit(X_img=images, y=y)

    loaded = RuleTree.from_dict(json_round_trip(model.to_dict()))

    assert loaded.root.stump.embedder.get_params() == embedder.get_params()
    assert np.array_equal(loaded.root.stump.transformer_.media_[loaded.root.stump.get_pivots()["used"]],
                          images[model.root.stump.get_pivots()["used"]])
