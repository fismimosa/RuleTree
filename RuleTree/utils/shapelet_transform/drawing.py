"""Drawing of shapelets in the nodes of the Graphviz representation of a tree."""

import tempfile

from RuleTree.utils.optional import import_optional


def graphviz_shapelets(label: str, shapes: list, colors: list = None) -> dict:
    """
    Attributes of a Graphviz node with a label above a plot of one or more shapelets, drawn into a
    temporary PNG file (requires matplotlib).

    Parameters
    ----------
    label : str
        Text of the node, e.g. the rule of the split.
    shapes : list of numpy.ndarray
        The shapelets to draw, each of shape (length,).
    colors : list of str, default=None
        Color of each shapelet. If None, the default colors of matplotlib.

    Returns
    -------
    attributes : dict
        Attributes of the Graphviz node.

    Raises
    ------
    MissingOptionalDependencyError
        If matplotlib is not installed.
    """
    plt = import_optional("matplotlib.pyplot", "The Graphviz drawing of the shapelet stumps")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp_file:
        image = temp_file.name
    plt.figure(figsize=(2, 1))
    for i, shape in enumerate(shapes):
        plt.plot(list(range(shape.shape[0])), shape, **({} if colors is None else {"color": colors[i]}))
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.xlim(0, max(shape.shape[0] for shape in shapes))
    plt.gca().tick_params(axis='both', which='both', length=2, labelsize=6)
    plt.gca().spines['right'].set_color('none')
    plt.gca().spines['top'].set_color('none')
    plt.savefig(image, format="png", dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close()

    return {"image": image, "imagescale": "true", "imagepos": "bc", "label": label, "labelloc": "t",
            "fixedsize": "true", "width": "2", "height": "1.33", "shape": "none", "fontsize": "10"}
