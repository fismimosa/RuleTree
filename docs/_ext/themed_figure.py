"""
The ``themed-figure`` directive: a figure with a light and a dark version, shown according to the
theme of the page.

    .. themed-figure:: /_static/figures/splits.svg
       :alt: ...

       Caption.

It takes the options of ``figure``; the dark version is the same file with ``_dark`` before the
extension (``splits_dark.svg``), as drawn by docs/make_figures.py. The two images get the classes
``only-light`` and ``only-dark``, which the theme hides in the other mode.
"""
from docutils import nodes
from docutils.parsers.rst.directives.images import Figure


class ThemedFigure(Figure):

    def run(self):
        result = super().run()
        figure = next((node for node in result if isinstance(node, nodes.figure)), None)
        if figure is None:  # an error message
            return result

        light = figure.next_node(nodes.image)
        dark = light.deepcopy()
        stem, extension = light["uri"].rsplit(".", 1)
        dark["uri"] = f"{stem}_dark.{extension}"
        light["classes"].append("only-light")
        dark["classes"].append("only-dark")
        light.parent.insert(light.parent.index(light) + 1, dark)
        return result


def setup(app):
    app.add_directive("themed-figure", ThemedFigure)
    return {"parallel_read_safe": True, "parallel_write_safe": True}
