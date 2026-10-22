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
from RuleTree.stumps.classification import ProximityTreeStumpClassifier, ShapeletTreeStumpClassifier  # noqa: E402
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
    for shape in rule.get("shapes", []):
        assert np.ndim(shape) == 1


@pytest.fixture
def bumps_and_dips():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, size=80)
    X = rng.normal(scale=0.3, size=(80, 1, 40))
    X[y == 0, 0, 15:25] += 2
    X[y == 1, 0, 15:25] -= 2
    return X, y


@pytest.mark.parametrize("make_stump, n_shapes", [
    (lambda: ShapeletTreeStumpClassifier(n_shapelets=5, sliding_window=10, selection="random", random_state=0), 1),
    (lambda: ProximityTreeStumpClassifier(n_shapelets=4, sliding_window=10, selection="random", random_state=0), 2),
])
def test_shapelets_drawn_in_their_nodes(make_stump, n_shapes, bumps_and_dips):
    pytest.importorskip("numba")
    X, y = bumps_and_dips
    model = RuleTreeClassifier(max_depth=2, random_state=0, base_stumps=make_stump()).fit(X_ts=X, y=y)

    ax = model.plot_tree()

    insets = {inset.get_gid(): inset for inset in ax.child_axes}
    assert set(insets) == {f"shapes-{node_id}" for node_id in model.get_predicates()}
    for node_id, node in model.get_predicates().items():
        inset = insets[f"shapes-{node_id}"]
        assert len(inset.lines) == n_shapes
        np.testing.assert_allclose(inset.lines[0].get_ydata(), node.stump.get_plot_rule()["shapes"][0])
        # the plot is inside the box of the node
        renderer = ax.figure.canvas.get_renderer()
        box = next(text for text in ax.texts if text.get_gid() == node_id).get_bbox_patch().get_window_extent(renderer)
        plot = inset.get_window_extent(renderer)
        assert box.x0 <= plot.x0 and plot.x1 <= box.x1 and box.y0 <= plot.y0 and plot.y1 <= box.y1


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
