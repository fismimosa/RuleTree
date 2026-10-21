"""
Figures of the user guide, drawn with matplotlib from models fitted on fixed data:

    python docs/make_figures.py            # writes docs/_static/figures/*.svg

Run it after changes that affect the figures (e.g. the splits of a stump); tests/test_figures.py
checks that it still runs.
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

OUTPUT_DIR = Path(__file__).resolve().parent / "_static" / "figures"

TEXT = "#2b2b2b"
MUTED = "#6b6b6b"
CLASS_COLORS = ["#2f6db3", "#e07b22", "#3a9a5b"]  # blue, orange, green
LEFT_COLOR, RIGHT_COLOR = "#d6e4f5", "#fbe3cc"  # regions of the left and right child
BOX_COLOR, BOX_EDGE = "#f4f6f8", "#9aa5b1"

plt.rcParams.update({
    "font.size": 9,
    "font.family": "DejaVu Sans",
    "text.color": TEXT,
    "axes.labelcolor": TEXT,
    "axes.edgecolor": "#b0b0b0",
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
    "svg.hashsalt": "ruletree",  # the same file at every run
})


def save(figure, output_dir: Path, name: str):
    """Save a figure as SVG, without the creation date (so that the file changes only with the figure)."""
    output_dir.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_dir / f"{name}.svg", bbox_inches="tight", dpi=150, metadata={"Date": None})
    plt.close(figure)


def box(ax, x, y, text, width, height, color=BOX_COLOR, edge=BOX_EDGE, fontsize=9, family=None, weight=None):
    """A rounded box with centered text, in data coordinates."""
    ax.add_patch(FancyBboxPatch((x - width / 2, y - height / 2), width, height,
                                boxstyle="round,pad=0.02,rounding_size=0.08", facecolor=color, edgecolor=edge,
                                linewidth=1))
    ax.text(x, y, text, ha="center", va="center", fontsize=fontsize, family=family, weight=weight)


def arrow(ax, start, end, text=None, style="-|>", color=MUTED, linestyle="-", text_offset=(0, 0.12)):
    """An arrow between two points, with an optional label at its middle."""
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle=style, mutation_scale=10, color=color, linewidth=1,
                                 linestyle=linestyle, shrinkA=2, shrinkB=2))
    if text:
        ax.text((start[0] + end[0]) / 2 + text_offset[0], (start[1] + end[1]) / 2 + text_offset[1], text,
                ha="center", va="center", fontsize=8, color=MUTED)


# --------------------------------------------------------------------------- splits of the stumps

def figure_splits(output_dir):
    """The root split of four kinds of stumps on the same two-dimensional data."""
    from RuleTree.stumps.classification import (DecisionTreeStumpClassifier, MultiplePivotTreeStumpClassifier,
                                                ObliqueDecisionTreeStumpClassifier, PivotTreeStumpClassifier)

    # two classes on the two sides of a diagonal, elongated along it
    rng = np.random.default_rng(3)
    y = np.repeat([0, 1], 80)
    centers = np.array([[-0.7, -0.5], [0.7, 0.5]])
    covariance = np.array([[0.7, -0.55], [-0.55, 0.7]])
    X = centers[y] + rng.multivariate_normal([0, 0], covariance, size=len(y))

    stumps = [
        ("DecisionTreeStumpClassifier", DecisionTreeStumpClassifier(random_state=0), "threshold on one feature"),
        ("ObliqueDecisionTreeStumpClassifier", ObliqueDecisionTreeStumpClassifier("bivariate", random_state=0),
         "threshold on a combination of features"),
        ("PivotTreeStumpClassifier", PivotTreeStumpClassifier(random_state=0), "distance from a pivot"),
        ("MultiplePivotTreeStumpClassifier", MultiplePivotTreeStumpClassifier(random_state=0),
         "closer to one pivot than to another"),
    ]

    margin = 0.4
    xx, yy = np.meshgrid(np.linspace(X[:, 0].min() - margin, X[:, 0].max() + margin, 300),
                         np.linspace(X[:, 1].min() - margin, X[:, 1].max() + margin, 300))
    grid = np.column_stack([xx.ravel(), yy.ravel()])

    figure, axes = plt.subplots(2, 2, figsize=(8, 7.2))
    for ax, (name, stump, description) in zip(axes.ravel(), stumps):
        stump.fit(X, y, context=None, node_id=None, cache=None)
        sides = stump.apply(grid).reshape(xx.shape)
        regions = ax.contourf(xx, yy, sides, levels=[0.5, 1.5, 2.5], colors=[LEFT_COLOR, RIGHT_COLOR])
        regions.set_rasterized(True)  # a small image instead of thousands of polygons
        ax.contour(xx, yy, sides, levels=[1.5], colors=[MUTED], linewidths=1)
        for label in (0, 1):
            ax.scatter(*X[y == label].T, s=12, color=CLASS_COLORS[label], edgecolor="white", linewidth=0.4,
                       label=f"class {label}")

        if hasattr(stump, "get_pivots"):
            used = stump.get_pivots()["used"]
            for pivot in ([used] if np.ndim(used) == 0 else used):
                ax.scatter(*X[pivot], marker="*", s=220, color="#ffd23f", edgecolor=TEXT, linewidth=0.8, zorder=5)
                ax.annotate(f"P_{pivot}", X[pivot], xytext=(6, 6), textcoords="offset points", fontsize=8)

        ax.set_title(f"{name}\n{description}", fontsize=9)
        ax.text(0.02, 0.02, stump.get_rule()["textual_rule"], transform=ax.transAxes, fontsize=8, family="monospace",
                bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="#cccccc"))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel("X_0")
        ax.set_ylabel("X_1")

    handles = [plt.Rectangle((0, 0), 1, 1, color=LEFT_COLOR), plt.Rectangle((0, 0), 1, 1, color=RIGHT_COLOR),
               *axes[0, 0].get_legend_handles_labels()[0]]
    figure.legend(handles, ["left child (rule true)", "right child", "class 0", "class 1"], loc="lower center",
                  ncol=4, frameon=False)
    figure.tight_layout(rect=(0, 0.04, 1, 1))
    save(figure, output_dir, "splits")


# --------------------------------------------------------------------------- components of a stump

def figure_stump_components(output_dir):
    """A Stump: the transformer turns the data into a feature matrix, the splitter splits it."""
    figure, ax = plt.subplots(figsize=(10, 3.4))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 3.6)
    ax.axis("off")

    y = 2.5
    box(ax, 1.0, y, "data of one type\nX, X_ts, X_img\n(all the instances)", 1.8, 0.9)
    box(ax, 3.5, y, "transformer", 1.7, 0.9, color="#dbe9f7", edge="#5b8fc7", weight="bold")
    box(ax, 6.5, y, "feature matrix\n(instances of the node\n× features)", 2.3, 0.9)
    box(ax, 9.4, y, "splitter", 1.5, 0.9, color="#fbe6d2", edge="#d98b45", weight="bold")
    box(ax, 11.4, y, "left (1)\nright (2)", 1.0, 0.9)
    arrow(ax, (1.9, y), (2.65, y))
    arrow(ax, (4.35, y), (5.35, y), "transform", text_offset=(0, 0.18))
    arrow(ax, (7.65, y), (8.65, y), "fit", text_offset=(0, 0.18))
    arrow(ax, (10.15, y), (10.9, y), "apply", text_offset=(0, 0.18))

    arrow(ax, (9.4, y - 0.45), (3.9, y - 0.45), "select(chosen feature): keep only what it needs",
          linestyle="--", style="-|>", text_offset=(0, -0.17))

    notes = [
        (3.5, 1.05, "MatrixTransformer, PivotTransformer,\nShapeletTransformer, ObliqueTransformer, ...\n"
                    "fit on the whole data + indices of the node;\ncache of the tree shared by the nodes"),
        (9.4, 1.15, "ClassificationSplitter, RegressionSplitter,\nXGBoostSplitter, FairSplitter, ...\n"
                    "feature_, threshold_, gain_"),
    ]
    for x, y_note, text in notes:
        ax.text(x, y_note, text, ha="center", va="center", fontsize=8, color=MUTED)
    ax.text(6.5, 0.2, "to_dict saves the parameters of both and their fitted state (get_state); "
                      "from_dict restores it (set_state)", ha="center", fontsize=8, color=MUTED, style="italic")
    save(figure, output_dir, "stump_components")


# --------------------------------------------------------------------------- nodes of a tree

def _layout(node, depth=0, positions=None, leaves=None):
    """x (order of the leaves) and depth of each node of a subtree."""
    positions = {} if positions is None else positions
    leaves = [0] if leaves is None else leaves
    if node.is_leaf():
        positions[node.node_id] = (leaves[0], depth)
        leaves[0] += 1
    else:
        _layout(node.node_l, depth + 1, positions, leaves)
        _layout(node.node_r, depth + 1, positions, leaves)
        positions[node.node_id] = ((positions[node.node_l.node_id][0] + positions[node.node_r.node_id][0]) / 2,
                                   depth)
    return positions


def draw_tree(ax, model, columns_names, class_names):
    """Draw the nodes of a fitted RuleTreeClassifier: identifier, rule or prediction, number of instances."""
    positions = _layout(model.root)
    nodes = model.get_predicates() | model.get_leaf_nodes()
    scale_x, scale_y = 2.4, 1.5
    coordinates = {node_id: (x * scale_x, -depth * scale_y) for node_id, (x, depth) in positions.items()}

    for node_id, node in nodes.items():
        if node.is_leaf():
            continue
        for child, label in ((node.node_l, "true"), (node.node_r, "false")):
            start, end = coordinates[node_id], coordinates[child.node_id]
            arrow(ax, (start[0], start[1] - 0.35), (end[0], end[1] + 0.35), label,
                  text_offset=(0.32 if child is node.node_r else -0.32, 0.05))

    for node_id, node in nodes.items():
        x, y = coordinates[node_id]
        if node.is_leaf():
            text = f"{node_id}\nclass {class_names[node.prediction]}\nsamples = {node.samples}"
            color = CLASS_COLORS[int(node.prediction)]
            box(ax, x, y, text, 1.9, 0.75, color="white", edge=color)
        else:
            rule = node.stump.get_rule(columns_names=columns_names)["textual_rule"]
            box(ax, x, y, f"{node_id}\n{rule}\nsamples = {node.samples}", 2.1, 0.75)

    xs = [x for x, _ in coordinates.values()]
    ys = [y for _, y in coordinates.values()]
    ax.set_xlim(min(xs) - 1.2, max(xs) + 1.2)
    ax.set_ylim(min(ys) - 0.6, max(ys) + 0.6)
    ax.axis("off")


def figure_node_ids(output_dir):
    """The tree of the guides on iris: node identifiers, rules and leaves."""
    from sklearn.datasets import load_iris

    from RuleTree import RuleTreeClassifier

    data = load_iris()
    model = RuleTreeClassifier(max_depth=2, random_state=0).fit(data.data, data.target)

    figure, ax = plt.subplots(figsize=(7, 3.6))
    draw_tree(ax, model, list(data.feature_names), list(data.target_names))
    save(figure, output_dir, "node_ids")


# --------------------------------------------------------------------------- time series

def figure_time_series(output_dir):
    """The data of the multimodal guide, the shapelet of the root split and the shapelets of a proximity split."""
    from RuleTree import RuleTreeClassifier
    from RuleTree.stumps.classification import ProximityTreeStumpClassifier, ShapeletTreeStumpClassifier

    # the data of docs/guides/multimodal.rst
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, size=200)
    X_ts = rng.normal(scale=0.4, size=(200, 1, 50))
    starts = rng.integers(5, 35, size=200)
    for i in np.where(y == 1)[0]:
        X_ts[i, 0, starts[i]:starts[i] + 10] += 1.0

    shapelets = ShapeletTreeStumpClassifier(n_shapelets=10, sliding_window=10, selection="random")
    model = RuleTreeClassifier(max_depth=2, base_stumps=shapelets, random_state=0).fit(X_ts=X_ts, y=y)
    root = model.root.stump

    rng_shape = np.random.default_rng(1)
    y_shape = rng_shape.integers(0, 2, size=200)
    X_shape = rng_shape.normal(scale=0.4, size=(200, 1, 50))
    for i, start in enumerate(rng_shape.integers(5, 35, size=200)):
        X_shape[i, 0, start:start + 10] += 1.0 if y_shape[i] == 0 else -1.0
    proximity = ProximityTreeStumpClassifier(n_shapelets=6, sliding_window=10, selection="random")
    proximity_model = RuleTreeClassifier(max_depth=1, base_stumps=proximity, random_state=0)
    proximity_model.fit(X_ts=X_shape, y=y_shape)

    figure, axes = plt.subplots(1, 3, figsize=(10, 2.8), gridspec_kw={"width_ratios": [1.6, 1, 1]})
    for label in (0, 1):
        for i, index in enumerate(np.where(y == label)[0][:3]):
            offset = 3 * (label * 3 + i)
            axes[0].plot(X_ts[index, 0] + offset, color=CLASS_COLORS[label], linewidth=1)
            if label == 1:  # where the bump is
                axes[0].axvspan(starts[index], starts[index] + 9, ymin=(offset + 1.2) / 19, ymax=(offset + 3.9) / 19,
                                color=CLASS_COLORS[1], alpha=0.15, linewidth=0)
    axes[0].set_ylim(-1.5, 17.5)
    axes[0].set_title("time series of class 0 (blue) and 1 (orange)\nthe bump of class 1 (shaded) moves", fontsize=9)
    axes[0].set_yticks([])
    axes[0].set_xlabel("time")

    shape = root.st.shapelets[root.feature_original, 0]
    axes[1].plot(shape, color=TEXT, linewidth=2)
    axes[1].set_title(f"shapelet of the root split (one of 10 drawn\nat random): "
                      f"{root.get_rule(float_precision=2)['textual_rule']}", fontsize=9)
    axes[1].set_xlabel("time")

    proximity_stump = proximity_model.root.stump
    for k, color in enumerate(("tab:green", "tab:red")):
        axes[2].plot(proximity_stump.st.shapelets[k, 0], color=color, linewidth=2)
    axes[2].set_title("shapelets of a proximity split\ngreen_shp and red_shp", fontsize=9)
    axes[2].set_xlabel("time")
    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    figure.tight_layout()
    save(figure, output_dir, "time_series")


# --------------------------------------------------------------------------- saved tree

def figure_saved_tree(output_dir):
    """The flat list of nodes of to_dict and the tree that from_dict rebuilds from it."""
    figure, ax = plt.subplots(figsize=(10, 4.4))
    ax.set_xlim(0, 11.5)
    ax.set_ylim(0, 5.2)
    ax.axis("off")

    ax.text(2.3, 5.0, 'dictionary["nodes"] (any order)', ha="center", fontsize=9, weight="bold")
    entries = [("R", "Rl", "Rr", True), ("Rr", "Rrl", "Rrr", True), ("Rrr", None, None, False),
               ("Rrl", None, None, False), ("Rl", None, None, False)]
    list_positions = {}
    for i, (node_id, left, right, internal) in enumerate(entries):
        y = 4.4 - i * 0.82
        text = (f'"node_id": "{node_id}"\n"left_node": {left and repr(left).replace(chr(39), chr(34))}, '
                f'"right_node": {right and repr(right).replace(chr(39), chr(34))}')
        text += "\n+ stump: args, transformer, splitter" if internal else "\n(leaf: prediction, samples, ...)"
        box(ax, 2.3, y, text, 4.4, 0.7, fontsize=7.5, family="monospace")
        list_positions[node_id] = (4.55, y)

    ax.text(8.8, 5.0, "rebuilt tree", ha="center", fontsize=9, weight="bold")
    tree_positions = {"R": (8.8, 4.2), "Rl": (7.7, 2.9), "Rr": (9.9, 2.9), "Rrl": (9.2, 1.6), "Rrr": (10.6, 1.6)}
    for parent, children in {"R": ("Rl", "Rr"), "Rr": ("Rrl", "Rrr")}.items():
        for child in children:
            start, end = tree_positions[parent], tree_positions[child]
            arrow(ax, (start[0], start[1] - 0.3), (end[0], end[1] + 0.3))
    for node_id, (x, y) in tree_positions.items():
        internal = node_id in ("R", "Rr")
        box(ax, x, y, node_id, 0.8, 0.5, color="#dbe9f7" if internal else "white", edge="#5b8fc7")

    for node_id in ("R", "Rr"):
        start = list_positions[node_id]
        end = tree_positions[node_id]
        arrow(ax, start, (end[0] - 0.42, end[1]), linestyle="--", color="#9aa5b1")

    ax.text(5.75, 0.35, "from_dict: RuleTreeNode.dict_to_node for each entry (the stump with the dict_to_node of its "
                       "class),\nthen the children are linked by left_node / right_node and \"R\" is the root",
            ha="center", fontsize=8, color=MUTED, style="italic")
    save(figure, output_dir, "saved_tree")


# --------------------------------------------------------------------------- size and accuracy

def figure_size_accuracy(output_dir):
    """Accuracy on a test set and number of leaves of trees of increasing depth, from the evaluation functions."""
    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.tree import DecisionTreeClassifier

    from RuleTree import RuleTreeClassifier
    from RuleTree.evaluation import evaluate_classification, evaluate_interpretability
    from RuleTree.stumps.classification import ObliqueDecisionTreeStumpClassifier

    X, y = load_breast_cancer(return_X_y=True)
    X = StandardScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)

    models = {
        "RuleTreeClassifier": lambda depth: RuleTreeClassifier(max_depth=depth, random_state=0),
        "RuleTreeClassifier, oblique stumps": lambda depth: RuleTreeClassifier(
            max_depth=depth, random_state=0, base_stumps=ObliqueDecisionTreeStumpClassifier("bivariate")),
        "sklearn DecisionTreeClassifier": lambda depth: DecisionTreeClassifier(max_depth=depth, random_state=0),
    }
    markers = ["o", "s", "^"]

    figure, ax = plt.subplots(figsize=(6.5, 3.8))
    for (name, make_model), marker, color in zip(models.items(), markers, CLASS_COLORS):
        leaves, accuracies = [], []
        for depth in range(1, 7):
            model = make_model(depth).fit(X_train, y_train)
            leaves.append(evaluate_interpretability(model)["n_leaves"])
            accuracies.append(evaluate_classification(y_test, model.predict(X_test))["accuracy"])
        ax.plot(leaves, accuracies, marker=marker, color=color, label=name, linewidth=1.2, markersize=5)
        ax.annotate("depth 1", (leaves[0], accuracies[0]), xytext=(4, -10), textcoords="offset points", fontsize=7,
                    color=color)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("number of leaves (rules), evaluate_interpretability")
    ax.set_ylabel("test accuracy, evaluate_classification")
    ax.set_title("breast cancer: depth 1 to 6", fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=3)
    save(figure, output_dir, "size_accuracy")


FIGURES = {
    "splits": figure_splits,
    "stump_components": figure_stump_components,
    "node_ids": figure_node_ids,
    "time_series": figure_time_series,
    "saved_tree": figure_saved_tree,
    "size_accuracy": figure_size_accuracy,
}


def main(output_dir=OUTPUT_DIR, names=None):
    """Draw the figures (all of them, or the given names) into output_dir."""
    import warnings

    warnings.simplefilter("ignore")
    for name in names or FIGURES:
        FIGURES[name](Path(output_dir))
        print(f"{name}.svg")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("names", nargs="*", choices=[[], *FIGURES], help="figures to draw (default: all)")
    parser.add_argument("--output-dir", default=OUTPUT_DIR, type=Path)
    arguments = parser.parse_args()
    main(arguments.output_dir, arguments.names)
