"""
Drawing of a tree with matplotlib (``RuleTree.plot_tree`` and ``RuleTreeNode.plot_tree``), in the
style of ``sklearn.tree.plot_tree``.

Each internal node shows what its stump returns from ``get_plot_rule``: the rule as text and,
for some stumps (shapelets, proximity), the shapes to plot below it. Each leaf shows its
prediction. The leaves are placed from left to right in the order of the tree, and each internal
node above the middle of its children, so the boxes never overlap.
"""
import re
import textwrap
from typing import Optional

import numpy as np

from RuleTree.utils.optional import import_optional

# colors of the drawing; the text uses the color of matplotlib (rcParams["text.color"]) when None
DEFAULT_COLORS = {
    "text": None,
    "split": "#f4f6f8",  # background and border of the internal nodes
    "split_edge": "#9aa5b1",
    "leaf": "#ffffff",  # background and border of the leaves (filled=False)
    "leaf_edge": "#9aa5b1",
    "edge": "#6b6b6b",  # arrows and their labels
    "classes": ["#2f6db3", "#e07b22", "#3a9a5b", "#9b59b6", "#c0392b", "#17becf", "#8c564b", "#e377c2",
                "#7f7f7f", "#bcbd22"],  # leaves of each class (or cluster), cycled
    "values": "viridis",  # colormap of the leaves of a regression tree
}

PLACEHOLDER = "(...)"  # the children of the nodes at max_depth
SHAPE_LINES = 4  # empty lines reserved below the rule for the plot of the shapes


def plot_tree(node, ax=None, *, columns_names: Optional[list] = None, class_names: Optional[list] = None,
              scaler=None, float_precision: Optional[int] = 3, max_depth: Optional[int] = None,
              node_ids: bool = True, samples: bool = True, filled: bool = True,
              fontsize: Optional[float] = None, colors: Optional[dict] = None, wrap: Optional[int] = None):
    """
    Draw the subtree of a node with matplotlib. See ``RuleTree.plot_tree`` for the parameters.

    Returns
    -------
    ax : matplotlib.axes.Axes
        The axes of the drawing. Each box is a text of ``ax.texts`` whose gid is the id of its node
        (``"(...)"`` for the subtrees cut by ``max_depth``).
    """
    plt = import_optional("matplotlib.pyplot", "plot_tree")
    colors = DEFAULT_COLORS | (colors or {})
    text_color = colors["text"] or plt.rcParams["text.color"]

    boxes = _layout(node, max_depth)
    leaf_colors = _leaf_colors(boxes, colors) if filled else {}
    for box in boxes:
        _describe(box, columns_names, class_names, scaler, float_precision, node_ids, samples, leaf_colors, colors,
                  wrap)

    n_leaves = 1 + max(box["x"] for box in boxes)
    depth = max(box["depth"] for box in boxes)
    if ax is None:  # wide enough for the longest line at about 10 points
        characters = max(len(line) for box in boxes for line in box["label"].split("\n"))
        lines = max(box["label"].count("\n") + 1 for box in boxes)
        figure = plt.figure(figsize=(min(max(4.0, n_leaves * max(1.3, 0.09 * characters)), 30.0),
                                     max(2.5, (depth + 1) * max(1.3, 0.2 * lines + 0.5))))
        ax = figure.add_axes((0, 0, 1, 1))  # no margins: the axes are hidden
    ax.set_xlim(-0.5, n_leaves - 0.5)
    ax.set_ylim(-depth - 0.5, 0.5)
    ax.set_axis_off()

    texts = {}
    for box in boxes:
        text = ax.text(box["x"], -box["depth"], box["label"], ha="center", va="center", color=text_color,
                       fontsize=10 if fontsize is None else fontsize, zorder=3,
                       bbox=dict(boxstyle="round,pad=0.4", facecolor=box["face"], edgecolor=box["edge"],
                                 linewidth=1))
        text.set_gid(PLACEHOLDER if box["node"] is None else box["node"].node_id)
        texts[box["key"]] = text

    renderer = ax.figure.canvas.get_renderer()
    if fontsize is None:  # the largest font with which the boxes do not overlap
        size = 10 * _fit_scale(ax, texts.values(), renderer)
        for text in texts.values():
            text.set_fontsize(size)
    size = next(iter(texts.values())).get_fontsize()

    extents = {key: _data_extent(ax, text, renderer) for key, text in texts.items()}
    for box in boxes:  # the arrows, from the bottom of the parent to the top of the child
        if box["parent"] is None:
            continue
        (x0, _), (x1, y_bottom) = extents[box["parent"]]
        (u0, y_top), (u1, _) = extents[box["key"]]
        start, end = ((x0 + x1) / 2, y_bottom), ((u0 + u1) / 2, y_top)
        ax.annotate("", xy=end, xytext=start, zorder=2,
                    arrowprops=dict(arrowstyle="-|>", color=colors["edge"], linewidth=1, shrinkA=0, shrinkB=1))
        left = box["side"] == "true"
        ax.annotate(box["side"], ((start[0] + end[0]) / 2, (start[1] + end[1]) / 2),
                    xytext=(-3 if left else 3, 2), textcoords="offset points", ha="right" if left else "left",
                    va="bottom", color=colors["edge"], fontsize=size * 0.9, zorder=4)

    for box in boxes:
        if box["shapes"]:
            _plot_shapes(ax, texts[box["key"]], box["shapes"], renderer, text_color)

    return ax


def _describe(box, columns_names, class_names, scaler, float_precision, node_ids, samples, leaf_colors, colors,
              wrap):
    """The text, the colors and the shapes of a box."""
    current = box["node"]
    box["shapes"] = None
    if current is None:
        box["label"], box["face"], box["edge"] = PLACEHOLDER, colors["leaf"], colors["leaf_edge"]
        return

    if current.is_leaf():
        label = _leaf_label(current, class_names, float_precision)
        box["face"], box["edge"] = leaf_colors.get(box["key"], (colors["leaf"], colors["leaf_edge"]))
    else:
        rule = current.stump.get_plot_rule(columns_names=columns_names, scaler=scaler,
                                           float_precision=float_precision)
        label, box["face"], box["edge"] = rule["label"], colors["split"], colors["split_edge"]
        if rule.get("shapes"):
            box["shapes"] = rule

    lines = ([current.node_id] if node_ids else []) + _wrap(label, wrap, rule=not current.is_leaf())
    if samples and getattr(current, "samples", None) is not None:
        lines.append(f"samples = {current.samples}")
    box["label"] = "\n".join(lines) + "\n" * (SHAPE_LINES if box["shapes"] else 0)


def _wrap(text, width, rule=True):
    """
    The lines of a text of at most width characters (if possible): a rule is split before its
    comparison operator (``mean area`` / ``<= 901.95``), then each part between words; the
    prediction of a leaf only between words (``class =`` / ``malignant``).
    """
    if width is None or len(text) <= width:
        return [text]
    operator = re.search(r" (<=|>=|==|!=|<|>|=) ", text) if rule else None
    parts = [text[:operator.start()], text[operator.start() + 1:]] if operator else [text]
    return [line for part in parts for line in (textwrap.wrap(part, width, break_long_words=False) or [part])]


def _data_extent(ax, text, renderer):
    """The corners (bottom left, top right) of the box of a text, in data coordinates."""
    text.update_bbox_position_size(renderer)
    extent = text.get_bbox_patch().get_window_extent(renderer)
    (x0, y0), (x1, y1) = ax.transData.inverted().transform([(extent.x0, extent.y0), (extent.x1, extent.y1)])
    return (x0, y1), (x1, y0)


def _layout(node, max_depth):
    """
    The boxes to draw: x (order of the leaves), depth, node (None for the placeholder of a subtree
    cut by max_depth), key and key of the parent, side ("true" for the left child).
    """
    boxes = []
    n_leaves = 0

    def visit(current, depth, parent_key, side):
        nonlocal n_leaves
        key = len(boxes)
        box = {"node": current, "depth": depth, "key": key, "parent": parent_key, "side": side}
        boxes.append(box)
        if max_depth is not None and depth > max_depth:
            box["node"] = None
        if box["node"] is None or current.is_leaf():
            box["x"] = n_leaves
            n_leaves += 1
        else:
            left = visit(current.node_l, depth + 1, key, "true")
            right = visit(current.node_r, depth + 1, key, "false")
            box["x"] = (left + right) / 2
        return box["x"]

    visit(node, 0, None, None)
    return boxes


def _leaf_label(node, class_names, float_precision):
    """The prediction of a leaf, as text."""
    prediction = node.prediction
    if _is_clustering(node):
        return f"cluster = {prediction}"
    if _is_classification(node):
        if class_names is not None:
            prediction = class_names[list(node.classes).index(prediction)]
        return f"class = {prediction}"
    values = np.round(np.asarray(prediction, dtype=float), float_precision) if float_precision is not None \
        else np.asarray(prediction, dtype=float)
    return f"value = {values.tolist()}"


def _is_classification(node):
    classes = getattr(node, "classes", None)
    return classes is not None and not _is_clustering(node) and np.ndim(node.prediction_probability) == 1 \
        and len(node.prediction_probability) == len(classes)


def _is_clustering(node):
    classes = getattr(node, "classes", None)
    return classes is not None and len(classes) == 1 and str(classes[0]) == "NA"


def _leaf_colors(boxes, colors):
    """Background and border of each leaf: the color of its class or cluster, or of its value."""
    from matplotlib import colormaps
    from matplotlib.colors import to_rgb

    def tint(color):  # a light background of the color of the border
        return (*to_rgb(color), 0.15)

    leaves = {box["key"]: box["node"] for box in boxes if box["node"] is not None and box["node"].is_leaf()}
    result = {}
    palette = colors["classes"]
    for key, leaf in leaves.items():
        if _is_classification(leaf):
            color = palette[list(leaf.classes).index(leaf.prediction) % len(palette)]
            result[key] = (tint(color), color)

    clusters = sorted({leaf.prediction for leaf in leaves.values() if _is_clustering(leaf)}, key=str)
    for key, leaf in leaves.items():
        if _is_clustering(leaf):
            color = palette[clusters.index(leaf.prediction) % len(palette)]
            result[key] = (tint(color), color)

    values = {key: float(leaf.prediction) for key, leaf in leaves.items()
              if key not in result and np.ndim(leaf.prediction) == 0 and np.isfinite(float(leaf.prediction))}
    if values:
        low, high = min(values.values()), max(values.values())
        colormap = colormaps[colors["values"]]
        for key, value in values.items():
            color = colormap(0.15 + 0.7 * ((value - low) / (high - low) if high > low else 0.5))
            result[key] = (tint(color), color)
    return result


def _fit_scale(ax, texts, renderer):
    """How much the font can grow (or must shrink) so that the largest box fits in a cell of the layout."""
    for text in texts:
        text.update_bbox_position_size(renderer)
    extents = [text.get_bbox_patch().get_window_extent(renderer) for text in texts]
    width = max(extent.width for extent in extents)
    height = max(extent.height for extent in extents)
    origin, unit = ax.transData.transform([(0, 0), (1, 1)])
    cell_width, cell_height = abs(unit[0] - origin[0]), abs(unit[1] - origin[1])
    return min(0.92 * cell_width / width, 0.8 * cell_height / height, 1.4)


def _plot_shapes(ax, text, rule, renderer, text_color):
    """The shapes of a node (e.g. shapelets) in the lines reserved at the bottom of its box."""
    text.update_bbox_position_size(renderer)
    extent = text.get_bbox_patch().get_window_extent(renderer)
    (x0, y0), (x1, y1) = ax.transAxes.inverted().transform([(extent.x0, extent.y0), (extent.x1, extent.y1)])
    n_lines = text.get_text().count("\n") + 1
    height = (y1 - y0) * SHAPE_LINES / n_lines
    inset = ax.inset_axes([x0 + 0.08 * (x1 - x0), y0 + 0.12 * height, 0.84 * (x1 - x0), 0.85 * height],
                          zorder=5)
    shape_colors = rule.get("colors") or [None] * len(rule["shapes"])
    for shape, color in zip(rule["shapes"], shape_colors):
        inset.plot(np.asarray(shape), color=color, linewidth=1.2)
    inset.set_xticks([])
    inset.set_yticks([])
    inset.patch.set_alpha(0)
    for side, spine in inset.spines.items():
        spine.set_visible(side in ("left", "bottom"))
        spine.set_color(text_color)
        spine.set_alpha(0.4)
    inset.set_gid(f"shapes-{text.get_gid()}")
