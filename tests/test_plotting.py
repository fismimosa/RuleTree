"""plot_tree: the drawing of a tree with matplotlib, for every kind of tree and stump."""
import json

import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import to_hex  # noqa: E402

from RuleTree import (RuleTreeClassifier, RuleTreeCluster, RuleTreeClusterClassifier,  # noqa: E402
                      RuleTreeClusterRegressor, RuleTreeRegressor)
from RuleTree.stumps.classification import (ObliquePivotTreeStumpClassifier, PivotTreeStumpClassifier,  # noqa: E402
                                            ProximityPivotTreeStumpClassifier, ProximityTreeStumpClassifier,
                                            ShapeletTreeStumpClassifier)
from RuleTree.tree.RuleTree import RuleTree  # noqa: E402
from RuleTree.utils.data_utils import json_numpy_encoder  # noqa: E402
from RuleTree.utils.plotting import DEFAULT_COLORS, PLACEHOLDER  # noqa: E402

from .test_stumps import STUMPS, time_series  # noqa: E402,F401 (fixture of the stumps)


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


def boxes(ax):
    """The text of each box, by gid (the id of the node, or the placeholder of a cut subtree)."""
    result = {}
    for text in ax.texts:
        if text.get_gid() is not None:
            result.setdefault(text.get_gid(), []).append(text.get_text())
    return result


def lines(ax, node_id):
    return boxes(ax)[node_id][0].strip("\n").split("\n")


@pytest.fixture
def classifier(iris):
    X, y = iris
    return RuleTreeClassifier(max_depth=3, random_state=0).fit(X, y)


def test_one_box_per_node_with_rule_and_samples(classifier):
    ax = classifier.plot_tree()

    nodes = classifier.get_predicates() | classifier.get_leaf_nodes()
    assert set(boxes(ax)) == set(nodes)
    for node_id, node in classifier.get_predicates().items():
        assert lines(ax, node_id) == [node_id, node.stump.get_rule()["textual_rule"], f"samples = {node.samples}"]
    for node_id, node in classifier.get_leaf_nodes().items():
        assert lines(ax, node_id) == [node_id, f"class = {node.prediction}", f"samples = {node.samples}"]


def test_one_arrow_per_child_with_true_and_false(classifier):
    ax = classifier.plot_tree()

    arrows = [text for text in ax.texts if text.get_gid() is None and text.arrow_patch is not None]
    labels = [text.get_text() for text in ax.texts if text.get_gid() is None and text.get_text()]
    n_children = 2 * len(classifier.get_predicates())
    assert len(arrows) == n_children
    assert sorted(labels) == sorted(["true", "false"] * (n_children // 2))


def test_names_of_features_and_classes(iris):
    X, y = iris
    names = ["sepal length", "sepal width", "petal length", "petal width"]
    model = RuleTreeClassifier(max_depth=1, random_state=0).fit(X, np.array(["setosa", "other"])[(y > 0).astype(int)])

    ax = model.plot_tree(columns_names=names, class_names=["Other", "Setosa"])

    assert lines(ax, "R")[1] == model.root.stump.get_rule(columns_names=names)["textual_rule"]
    assert lines(ax, "R")[1].startswith("petal")
    assert {lines(ax, "Rl")[1], lines(ax, "Rr")[1]} == {"class = Setosa", "class = Other"}


def test_options_hide_ids_and_samples(classifier):
    ax = classifier.plot_tree(node_ids=False, samples=False)

    assert lines(ax, "R") == [classifier.root.stump.get_rule()["textual_rule"]]
    assert lines(ax, "Rl") == [f"class = {classifier.root.node_l.prediction}"]


def test_max_depth_cuts_the_subtrees(diabetes):
    X, y = diabetes
    model = RuleTreeRegressor(max_depth=4, random_state=0).fit(X, y)

    ax = model.plot_tree(max_depth=1)

    drawn = boxes(ax)
    assert set(drawn) - {PLACEHOLDER} == {"R", "Rl", "Rr"}
    assert len(drawn[PLACEHOLDER]) == 4  # the children of Rl and Rr


def test_regression_values_rounded(diabetes):
    X, y = diabetes
    model = RuleTreeRegressor(max_depth=1, random_state=0).fit(X, y)

    ax = model.plot_tree(float_precision=1)

    assert lines(ax, "Rl")[1] == f"value = {round(float(model.root.node_l.prediction), 1)}"


def test_several_targets(diabetes):
    X, y = diabetes
    model = RuleTreeRegressor(max_depth=1, random_state=0).fit(X, np.column_stack([y, y / 100]))

    ax = model.plot_tree(float_precision=2)

    assert lines(ax, "Rl")[1] == f"value = {np.round(model.root.node_l.prediction, 2).tolist()}"


@pytest.mark.parametrize("make_model, prefix", [
    (lambda: RuleTreeCluster(max_depth=2, random_state=0), "cluster = "),
    (lambda: RuleTreeClusterClassifier(max_depth=2, random_state=0), "class = "),
    (lambda: RuleTreeClusterRegressor(max_depth=2, random_state=0), "value = "),
])
def test_clustering_trees(make_model, prefix, blobs):
    X, groups = blobs
    model = make_model()
    model.fit(X) if type(model) is RuleTreeCluster else model.fit(X, groups)

    ax = model.plot_tree()

    for node_id in model.get_leaf_nodes():
        assert lines(ax, node_id)[1].startswith(prefix)


def test_leaves_colored_by_class(classifier):
    colors = {"classes": ["#111111", "#222222", "#333333"]}

    ax = classifier.plot_tree(colors=colors)

    for text in ax.texts:
        node = classifier.get_leaf_nodes().get(text.get_gid())
        if node is not None:
            expected = colors["classes"][list(node.classes).index(node.prediction)]
            assert to_hex(text.get_bbox_patch().get_edgecolor()) == expected


def test_not_filled_and_custom_colors(classifier):
    ax = classifier.plot_tree(filled=False, colors={"text": "#123456", "leaf_edge": "#abcdef"})

    for text in ax.texts:
        if text.get_gid() in classifier.get_leaf_nodes():
            assert to_hex(text.get_color()) == "#123456"
            assert to_hex(text.get_bbox_patch().get_edgecolor()) == "#abcdef"
            assert to_hex(text.get_bbox_patch().get_facecolor()) == DEFAULT_COLORS["leaf"]


def test_fontsize_fixed_or_fitted(classifier):
    ax = classifier.plot_tree(fontsize=7)
    assert {text.get_fontsize() for text in ax.texts if text.get_gid() is not None} == {7}

    ax = classifier.plot_tree()
    sizes = {text.get_fontsize() for text in ax.texts if text.get_gid() is not None}
    assert len(sizes) == 1 and 0 < sizes.pop() <= 14


def test_boxes_do_not_overlap(diabetes):
    X, y = diabetes
    model = RuleTreeRegressor(max_depth=4, random_state=0).fit(X, y)

    ax = model.plot_tree()
    renderer = ax.figure.canvas.get_renderer()

    extents = [text.get_bbox_patch().get_window_extent(renderer) for text in ax.texts if text.get_gid() is not None]
    for i, first in enumerate(extents):
        for second in extents[i + 1:]:
            assert not first.overlaps(second)


def test_draws_on_given_axes_and_subtree(classifier):
    figure, ax = plt.subplots()

    result = classifier.get_node_by_id("Rr").plot_tree(ax)

    assert result is ax and len(figure.axes) == 1
    assert set(boxes(ax)) == {node_id for node_id in classifier.get_predicates() | classifier.get_leaf_nodes()
                              if node_id.startswith("Rr")}


def test_saved_and_loaded_tree_draws_the_same(classifier):
    loaded = RuleTree.from_dict(json.loads(json.dumps(classifier.to_dict(), cls=json_numpy_encoder)))

    assert boxes(loaded.plot_tree()) == boxes(classifier.plot_tree())


def test_saves_to_file(classifier, tmp_path):
    classifier.plot_tree().figure.savefig(tmp_path / "tree.svg")

    assert (tmp_path / "tree.svg").read_text().startswith("<?xml")


@pytest.mark.parametrize("make_stump, data_type, dataset, continuous", STUMPS,
                         ids=[f"{s[0]().__class__.__name__}-{s[1]}" for s in STUMPS])
def test_every_stump_has_a_plot_rule(make_stump, data_type, dataset, continuous, request):
    X, y = request.getfixturevalue(dataset)
    stump = make_stump().fit({data_type: X}, y.astype(float) if continuous else y, data_type=data_type,
                             context=None, node_id=None, cache=None)

    rule = stump.get_plot_rule()

    assert rule["label"] == stump.get_rule()["textual_rule"]
    for value in rule.get("media_values", []):
        assert np.ndim(value) == 1 and rule["media_kind"] == "series"


@pytest.fixture
def bumps_and_dips():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, size=80)
    X = rng.normal(scale=0.3, size=(80, 1, 40))
    X[y == 0, 0, 15:25] += 2
    X[y == 1, 0, 15:25] -= 2
    return X, y


def test_shapelet_drawn_in_its_node(bumps_and_dips):
    pytest.importorskip("numba")
    X, y = bumps_and_dips
    stump = ShapeletTreeStumpClassifier(n_shapelets=5, sliding_window=10, selection="random", random_state=0)
    model = RuleTreeClassifier(max_depth=2, random_state=0, base_stumps=stump).fit(X_ts=X, y=y)

    ax = model.plot_tree()  # drawn also without show_media: the shapelets are saved in the stump

    insets = media_insets(ax)
    assert set(insets) == {f"media-{node_id}-0" for node_id in model.get_predicates()}
    renderer = ax.figure.canvas.get_renderer()
    for node_id, node in model.get_predicates().items():
        inset = insets[f"media-{node_id}-0"]
        assert inset.get_title() == "shp"
        np.testing.assert_allclose(inset.lines[0].get_ydata(), node.stump.get_plot_rule()["media_values"][0])
        # the plot is inside the box of the node
        box = next(text for text in ax.texts if text.get_gid() == node_id).get_bbox_patch().get_window_extent(renderer)
        plot = inset.get_window_extent(renderer)
        assert box.x0 <= plot.x0 and plot.x1 <= box.x1 and box.y0 <= plot.y0 and plot.y1 <= box.y1


def test_proximity_shapelets_on_the_arrows(bumps_and_dips):
    """As the proximity pivots: the green shapelet on the left arrow, the red one on the right arrow."""
    pytest.importorskip("numba")
    X, y = bumps_and_dips
    stump = ProximityTreeStumpClassifier(n_shapelets=4, sliding_window=10, selection="random", random_state=0)
    model = RuleTreeClassifier(max_depth=1, random_state=0, base_stumps=stump).fit(X_ts=X, y=y)

    insets = media_insets(model.plot_tree())
    shapelets = model.root.stump.get_plot_rule()["media_values"]

    for side, shapelet, name, color in [("true", shapelets[0], "green_shp", "tab:green"),
                                        ("false", shapelets[1], "red_shp", "tab:red")]:
        inset = insets[f"media-R-{side}"]
        assert inset.get_title() == name
        np.testing.assert_allclose(inset.lines[0].get_ydata(), shapelet)
        assert to_hex(inset.lines[0].get_color()) == to_hex(color)


def test_wrap_splits_rules_before_the_operator_and_leaves_between_words():
    from RuleTree.utils.plotting import _wrap

    assert _wrap("mean fractal dimension <= 0.062", 12) == ["mean fractal", "dimension", "<= 0.062"]
    assert _wrap("worst area > 718.65", 12) == ["worst area", "> 718.65"]
    assert _wrap("class = malignant", 12, rule=False) == ["class =", "malignant"]
    assert _wrap("petal width <= 0.8", None) == ["petal width <= 0.8"]
    assert _wrap("short <= 1", 12) == ["short <= 1"]


def test_wrap_makes_the_boxes_narrower(iris):
    X, y = iris
    names = ["sepal length", "sepal width", "petal length", "petal width"]
    model = RuleTreeClassifier(max_depth=2, random_state=0).fit(X, y)

    ax = model.plot_tree(columns_names=names, wrap=12, fontsize=10)

    assert lines(ax, "R") == ["R", "petal width", "<= 0.8", "samples = 150"]
    assert max(len(line) for text in ax.texts if text.get_gid() for line in text.get_text().split("\n")) <= 13


def test_new_figure_has_no_margins(classifier):
    ax = classifier.plot_tree()

    assert tuple(ax.get_position().bounds) == (0, 0, 1, 1)


@pytest.fixture(scope="module")
def digits():
    """Four digits: images of 8 x 8 pixels."""
    from sklearn.datasets import load_digits

    data = load_digits()
    four = np.isin(data.target, [0, 1, 4, 7])
    return data.images[four], data.target[four]


class WordCounter:
    """A stateless embedder of texts: the number of times each word of a small vocabulary occurs."""

    vocabulary = ["goal", "match", "team", "bread", "garlic", "chef"]

    def get_params(self, deep=True):
        return {}

    def set_params(self, **params):
        return self

    def fit(self, X=None, y=None):
        return self

    def transform(self, X):
        return np.array([[text.lower().count(word) for word in self.vocabulary] for text in X], dtype=float)


def media_insets(ax):
    """The axes of the media, by gid."""
    return {inset.get_gid(): inset for inset in ax.child_axes if str(inset.get_gid()).startswith("media-")}


def test_pivots_not_drawn_by_default(digits):
    images, y = digits
    model = RuleTreeClassifier(max_depth=2, random_state=0, base_stumps=PivotTreeStumpClassifier())
    model.fit(X_img=images, y=y)

    assert media_insets(model.plot_tree()) == {}


def test_pivot_images_in_their_nodes(digits):
    images, y = digits
    model = RuleTreeClassifier(max_depth=2, random_state=0, base_stumps=PivotTreeStumpClassifier())
    model.fit(X_img=images, y=y)

    insets = media_insets(model.plot_tree(show_media=True))

    assert set(insets) == {f"media-{node_id}-0" for node_id in model.get_predicates()}
    for node_id, node in model.get_predicates().items():
        inset = insets[f"media-{node_id}-0"]
        pivot = node.stump.get_pivots()["used"]
        assert inset.get_title() == f"P_{pivot}"
        assert np.array_equal(inset.images[0].get_array(), images[pivot])


def test_proximity_pivots_on_the_arrows(digits):
    """The first pivot of the rule on the arrow of the left child (true), the second on the right one."""
    images, y = digits
    model = RuleTreeClassifier(max_depth=2, random_state=0, base_stumps=ProximityPivotTreeStumpClassifier())
    model.fit(X_img=images, y=y)

    insets = media_insets(model.plot_tree(show_media=True))

    for node_id, node in model.get_predicates().items():
        first, second = node.stump.get_pivots()["used"]
        assert insets[f"media-{node_id}-true"].get_title() == f"P_{first}"
        assert insets[f"media-{node_id}-false"].get_title() == f"P_{second}"
    assert len(insets) == 2 * len(model.get_predicates())


def test_oblique_pivots_side_by_side(digits):
    images, y = digits
    model = RuleTreeClassifier(max_depth=1, random_state=0, base_stumps=ObliquePivotTreeStumpClassifier())
    model.fit(X_img=images, y=y)

    insets = media_insets(model.plot_tree(show_media=True))

    assert [insets[f"media-R-{i}"].get_title() for i in range(len(insets))] == \
        [f"P_{p}" for p in model.root.stump.get_pivots()["used"]]


def test_pivot_series_and_texts(time_series):
    X, y = time_series
    texts = np.array(["goal of the team" if label else "bread with garlic" for label in y])
    series_model = RuleTreeClassifier(max_depth=1, random_state=0, base_stumps=ProximityPivotTreeStumpClassifier())
    text_model = RuleTreeClassifier(max_depth=1, random_state=0,
                                    base_stumps=ProximityPivotTreeStumpClassifier(embedder=WordCounter()))
    series_model.fit(X_ts=X, y=y)
    text_model.fit(X_txt=texts, y=y)
    first = series_model.root.stump.get_pivots()["used"][0]
    second = text_model.root.stump.get_pivots()["used"][1]

    lines_of_series = media_insets(series_model.plot_tree(show_media=True))["media-R-true"].lines
    texts_of_pivots = media_insets(text_model.plot_tree(show_media=True))["media-R-false"].texts

    np.testing.assert_allclose(lines_of_series[0].get_ydata(), X[first, 0])
    assert texts_of_pivots[0].get_text().replace("\n", " ") == texts[second]


def test_tabular_pivots_not_drawn(iris):
    X, y = iris
    model = RuleTreeClassifier(max_depth=2, random_state=0, base_stumps=PivotTreeStumpClassifier()).fit(X, y)

    assert media_insets(model.plot_tree(show_media=True)) == {}


def test_show_media_must_be_a_boolean(digits):
    images, y = digits
    model = RuleTreeClassifier(max_depth=1, random_state=0, base_stumps=PivotTreeStumpClassifier())
    model.fit(X_img=images, y=y)

    with pytest.raises(ValueError, match="True or False"):
        model.plot_tree(show_media="image")


@pytest.mark.parametrize("stump", [PivotTreeStumpClassifier, ProximityPivotTreeStumpClassifier,
                                   ObliquePivotTreeStumpClassifier])
def test_saved_tree_draws_its_pivots(stump, digits, tmp_path):
    """The pivots are saved in the tree: a tree loaded from its JSON file draws them without the data."""
    images, y = digits
    model = RuleTreeClassifier(max_depth=2, random_state=0, base_stumps=stump()).fit(X_img=images, y=y)
    model.to_dict(tmp_path / "tree.json")
    loaded = RuleTree.from_dict(tmp_path / "tree.json")

    ax = loaded.plot_tree(show_media=True)
    insets = media_insets(ax)

    assert {gid: inset.get_title() for gid, inset in insets.items()} == \
        {gid: inset.get_title() for gid, inset in media_insets(model.plot_tree(show_media=True)).items()}
    for inset in insets.values():
        assert np.array_equal(inset.images[0].get_array(), images[int(inset.get_title()[2:])])
    for extension in ("png", "svg", "pdf"):
        ax.figure.savefig(tmp_path / f"tree.{extension}")
        assert (tmp_path / f"tree.{extension}").stat().st_size > 0
    assert (tmp_path / "tree.svg").read_text().count("<image") == len(insets)


def test_media_of_each_data_type_in_a_multimodal_tree(digits):
    """Pivot stumps on the images and on the time series: each node draws its pivots as images or series."""
    images, y = digits
    rng = np.random.default_rng(0)
    series = rng.normal(size=(len(y), 1, 30))
    series[y == 1, 0, 10:20] += 3
    stumps = [(0.5, PivotTreeStumpClassifier(), ["image"]), (0.5, ProximityPivotTreeStumpClassifier(), ["ts"])]
    model = RuleTreeClassifier(max_depth=3, random_state=0, stump_selection="best", base_stumps=stumps)
    model.fit(X_img=images, X_ts=series, y=y)
    data_types = {node.stump.data_type_ for node in model.get_predicates().values()}

    insets = media_insets(model.plot_tree(show_media=True))

    assert data_types == {"image", "ts"}
    for node_id, node in model.get_predicates().items():
        used = node.stump.get_pivots()["used"]
        first = used[0] if isinstance(used, list) else used
        inset = insets[f"media-{node_id}-true" if isinstance(used, list) else f"media-{node_id}-0"]
        if node.stump.data_type_ == "image":
            assert np.array_equal(inset.images[0].get_array(), images[first])
        else:
            np.testing.assert_allclose(inset.lines[0].get_ydata(), series[first, 0])
