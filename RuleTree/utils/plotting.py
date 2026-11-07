"""
Drawing of a tree, or of a subtree from its root node, with matplotlib (``RuleTree.plot_tree``), in the
style of ``sklearn.tree.plot_tree``, and as a Graphviz graph (``export_graphviz``) with the same boxes.

Each internal node shows what its stump returns from ``get_plot_rule``: the rule as text and the
*media* of its split, saved in the stump and drawn as images, time series or texts: e.g. the shapelets,
always drawn, and the pivots, drawn with ``show_media``. They are drawn below the rule or, for the
splits that compare two of them (e.g. the proximity stumps), on the arrows of the two children. Each leaf
shows its prediction. The leaves are placed from left to right in the order of the tree, and each
internal node above the middle of its children, so the boxes never overlap.
"""
import html
import os
import re
import tempfile
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
MEDIA_LINES = 7  # empty lines reserved below the rule for its media


def plot_tree(node, ax=None, *, columns_names: Optional[list] = None, class_names: Optional[list] = None,
              scaler=None, float_precision: Optional[int] = 3, max_depth: Optional[int] = None,
              node_ids: bool = True, samples: bool = True, filled: bool = True,
              fontsize: Optional[float] = None, colors: Optional[dict] = None, wrap: Optional[int] = None,
              show_media: bool = False):
    """
    Draw the subtree of a node with matplotlib. See ``RuleTree.plot_tree`` for the parameters.

    Returns
    -------
    ax : matplotlib.axes.Axes
        The axes of the drawing. Each box is a text of ``ax.texts`` whose gid is the id of its node
        (``"(...)"`` for the subtrees cut by ``max_depth``); each medium is an axes of
        ``ax.child_axes`` whose gid is ``media-<node id>-<position>`` (``true`` or ``false`` on the arrows).
    """
    plt = import_optional("matplotlib.pyplot", "plot_tree")
    if not isinstance(show_media, (bool, np.bool_)):
        raise ValueError(f"show_media must be True or False, not {show_media!r}")
    colors = DEFAULT_COLORS | (colors or {})
    text_color = colors["text"] or plt.rcParams["text.color"]

    boxes = _layout(node, max_depth)
    leaf_colors = _leaf_colors(boxes, colors) if filled else {}
    for box in boxes:
        _describe(box, columns_names, class_names, scaler, float_precision, node_ids, samples, leaf_colors, colors,
                  wrap, show_media)
    on_edges = [box for box in boxes if box["media"] and box["media_on_edges"]]

    n_leaves = 1 + max(box["x"] for box in boxes)
    depth = max(box["depth"] for box in boxes)
    if ax is None:  # wide enough for the longest line at about 10 points
        characters = max(len(line) for box in boxes for line in box["label"].split("\n"))
        lines = max(box["label"].count("\n") + 1 for box in boxes)
        texts_on_edges = any(medium["kind"] == "text" for box in on_edges for medium in box["media"])
        level_height = max(1.3, 0.2 * lines + 0.5) + (1.4 if on_edges else 0)  # room for the media on the arrows
        leaf_width = max(1.3, 0.09 * characters, (2.2 if texts_on_edges else 1.6) if on_edges else 0)
        figure = plt.figure(figsize=(min(max(4.0, n_leaves * leaf_width), 30.0), max(2.5, (depth + 1) * level_height)))
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
        left = box["side"] == "true"
        parent = boxes[box["parent"]]
        medium_on_edge = parent["media"] and parent["media_on_edges"]
        arrow = dict(arrowstyle="-|>", color=colors["edge"], linewidth=1, shrinkA=0, shrinkB=1)
        if medium_on_edge:  # interrupted where the medium is drawn, also on a transparent background
            above, below = (start[0] + 0.37 * (end[0] - start[0]), start[1] + 0.37 * (end[1] - start[1])), \
                (start[0] + 0.79 * (end[0] - start[0]), start[1] + 0.79 * (end[1] - start[1]))
            ax.annotate("", xy=above, xytext=start, zorder=2, arrowprops=arrow | dict(arrowstyle="-"))
            ax.annotate("", xy=end, xytext=below, zorder=2, arrowprops=arrow)
        else:
            ax.annotate("", xy=end, xytext=start, zorder=2, arrowprops=arrow)
        where = 0.12 if medium_on_edge else 0.5  # the label above the medium drawn on the arrow
        ax.annotate(box["side"], (start[0] + where * (end[0] - start[0]), start[1] + where * (end[1] - start[1])),
                    xytext=(-3 if left else 3, 2), textcoords="offset points", ha="right" if left else "left",
                    va="bottom", color=colors["edge"], fontsize=size * 0.9, zorder=4)
        if medium_on_edge:
            medium = parent["media"][0 if left else 1]
            width, height = 0.55 if medium["kind"] == "text" else 0.42, 0.42 * (start[1] - end[1])
            center = (start[0] + 0.58 * (end[0] - start[0]), start[1] + 0.58 * (end[1] - start[1]))
            inset = ax.inset_axes([center[0] - width / 2, center[1] - height / 2, width, height],
                                  transform=ax.transData, zorder=5)
            _draw_medium(inset, medium, text_color, colors, size, renderer)
            inset.set_gid(f"media-{parent['node'].node_id}-{box['side']}")

    for box in boxes:
        if box["media"] and not box["media_on_edges"]:
            _plot_media(ax, texts[box["key"]], box["media"], renderer, text_color, colors, size)

    return ax


def export_graphviz(node, graph=None, *, columns_names: Optional[list] = None, class_names: Optional[list] = None,
                    scaler=None, float_precision: Optional[int] = 3, max_depth: Optional[int] = None,
                    node_ids: bool = True, samples: bool = True, filled: bool = True, colors: Optional[dict] = None,
                    wrap: Optional[int] = None, show_media: bool = False):
    """
    The subtree of a node as a Graphviz graph, with the boxes, colors and media of ``plot_tree`` (see
    ``RuleTree.export_graphviz`` for the parameters). The media are drawn with matplotlib into PNG
    files of a temporary directory, which the graph references.

    Returns
    -------
    graph : graphviz.Digraph
    """
    graphviz = import_optional("graphviz", "export_graphviz")
    to_hex = import_optional("matplotlib.colors", "export_graphviz").to_hex  # colors of the leaves, media
    if not isinstance(show_media, (bool, np.bool_)):
        raise ValueError(f"show_media must be True or False, not {show_media!r}")
    colors = DEFAULT_COLORS | (colors or {})
    text_color = colors["text"] or "black"

    boxes = _layout(node, max_depth)
    leaf_colors = _leaf_colors(boxes, colors) if filled else {}
    for box in boxes:
        _describe(box, columns_names, class_names, scaler, float_precision, node_ids, samples, leaf_colors, colors,
                  wrap, show_media)

    directory = tempfile.mkdtemp(prefix="ruletree-graphviz-") if any(box["media"] for box in boxes) else None
    graph = graphviz.Digraph(name="RuleTree") if graph is None else graph
    graph.attr("node", shape="box", style="rounded,filled", fontcolor=text_color, fontsize="10")
    graph.attr("edge", color=colors["edge"], fontcolor=colors["edge"], fontsize="9", arrowsize="0.7")
    names = {box["key"]: PLACEHOLDER if box["node"] is None else box["node"].node_id for box in boxes}
    for box in boxes:  # placeholders get a unique name, as the parent id plus the side
        if box["node"] is None:
            names[box["key"]] = f"{names[box['parent']]}-{box['side']}"

    for box in boxes:
        lines = [html.escape(line) for line in box["label"].rstrip("\n").split("\n")]
        image = _media_png(box["media"], directory, names[box["key"]], text_color, colors) \
            if box["media"] and not box["media_on_edges"] else None
        graph.node(names[box["key"]], label=_html_label("<BR/>".join(lines), image),
                   fillcolor=to_hex(box["face"], keep_alpha=True), color=to_hex(box["edge"], keep_alpha=True))

        if box["parent"] is not None:
            parent = boxes[box["parent"]]
            image = None
            if parent["media"] and parent["media_on_edges"]:  # the medium of the child next to its arrow
                medium = parent["media"][0 if box["side"] == "true" else 1]
                image = _media_png([medium], directory, f"{names[parent['key']]}-{box['side']}", text_color, colors)
            graph.edge(names[parent["key"]], names[box["key"]], label=_html_label(box["side"], image))

    return graph


def _html_label(text, image=None):
    """A Graphviz HTML label: the text (already escaped) and, below it, the image of a PNG file."""
    if image is None:
        return f"<{text}>"
    return f'<<TABLE BORDER="0" CELLSPACING="0"><TR><TD>{text}</TD></TR>' \
           f'<TR><TD><IMG SRC="{html.escape(image)}"/></TD></TR></TABLE>>'


def _media_png(media, directory, name, text_color, colors):
    """The media of a node, drawn side by side as in ``plot_tree`` into the PNG file ``<name>.png``."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    width = 1.6 if media[0]["kind"] == "text" else 1.2
    figure = Figure(figsize=(width * len(media), 1.0))
    canvas = FigureCanvasAgg(figure)
    renderer = canvas.get_renderer()
    for inset, medium in zip(figure.subplots(1, len(media), squeeze=False)[0], media):
        _draw_medium(inset, medium, text_color, colors, 9, renderer)
    path = os.path.join(directory, f"{name}.png")
    figure.savefig(path, dpi=100, bbox_inches="tight", pad_inches=0.02, transparent=True)
    return path


def _describe(box, columns_names, class_names, scaler, float_precision, node_ids, samples, leaf_colors, colors,
              wrap, show_media=False):
    """The text, the colors and the media of a box."""
    current = box["node"]
    box["media"], box["media_on_edges"] = None, False
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
        box["media"] = _rule_media(rule, show_media)
        box["media_on_edges"] = bool(box["media"]) and bool(rule.get("media_on_edges")) and len(box["media"]) == 2

    lines = ([current.node_id] if node_ids else []) + _wrap(label, wrap, rule=not current.is_leaf())
    if samples and getattr(current, "samples", None) is not None:
        lines.append(f"samples = {current.samples}")
    reserved = MEDIA_LINES if box["media"] and not box["media_on_edges"] else 0
    box["label"] = "\n".join(lines) + "\n" * reserved


def _rule_media(rule, show_media):
    """
    The media of a node, as dictionaries with ``name``, ``kind``, ``value`` and ``color``: the
    ``media_values`` of the rule, unless they are ``media_optional`` and show_media is False.
    """
    values = rule.get("media_values")
    if values is None or (rule.get("media_optional") and not show_media):
        return None

    kind = rule.get("media_kind", "series")
    names = rule.get("media_names") or [""] * len(values)
    colors = rule.get("media_colors") or [None] * len(values)
    return [{"name": name, "kind": kind, "value": value, "color": color}
            for name, value, color in zip(names, values, colors)]


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


def _plot_media(ax, text, media, renderer, text_color, colors, size):
    """The media of a node, side by side in the lines reserved at the bottom of its box."""
    text.update_bbox_position_size(renderer)
    extent = text.get_bbox_patch().get_window_extent(renderer)
    (x0, y0), (x1, y1) = ax.transAxes.inverted().transform([(extent.x0, extent.y0), (extent.x1, extent.y1)])
    n_lines = text.get_text().count("\n") + 1
    height = (y1 - y0) * MEDIA_LINES / n_lines
    width = 0.9 * (x1 - x0) / len(media)
    for i, medium in enumerate(media):
        inset = ax.inset_axes([x0 + 0.05 * (x1 - x0) + i * width + 0.04 * width, y0 + 0.06 * height,
                               0.92 * width, 0.8 * height], zorder=5)
        _draw_medium(inset, medium, text_color, colors, size, renderer)
        inset.set_gid(f"media-{text.get_gid()}-{i}")


def _draw_medium(inset, medium, text_color, colors, size, renderer):
    """
    A medium in its axes, with its name (e.g. ``P_<i>``) as title: an image (height x width, with the
    channels last), a time series (one or more signals) or a text (shrunk to fit the axes).
    """
    kind = medium["kind"]
    instance = str(medium["value"]) if kind == "text" else np.asarray(medium["value"])
    if medium["name"]:
        inset.set_title(medium["name"], fontsize=size * 0.8, color=medium["color"] or text_color, pad=2)
    inset.set_xticks([])
    inset.set_yticks([])
    if kind == "image":
        if instance.ndim == 3 and instance.shape[-1] == 1:
            instance = instance[..., 0]
        inset.imshow(instance, cmap="gray_r" if instance.ndim == 2 else None)  # the aspect of the image
        inset.set_anchor("C")
        return
    if kind == "series":
        for signal in np.atleast_2d(instance):
            inset.plot(signal, color=medium["color"] or colors["classes"][0], linewidth=1.2)
    else:
        lines = textwrap.wrap(instance, 16)
        lines = lines[:6] + (["..."] if len(lines) > 6 else [])
        label = inset.text(0.5, 0.5, "\n".join(lines), ha="center", va="center", fontsize=size * 0.7,
                           color=text_color, transform=inset.transAxes)
        box, extent = inset.get_window_extent(renderer), label.get_window_extent(renderer)
        label.set_fontsize(label.get_fontsize() * min(1.0, 0.92 * box.width / extent.width,
                                                       0.92 * box.height / extent.height))
    inset.set_facecolor(colors["leaf"])  # opaque, over the arrow
    for spine in inset.spines.values():
        spine.set_color(text_color)
        spine.set_alpha(0.4)
