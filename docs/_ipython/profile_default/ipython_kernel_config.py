# IPython profile of the kernels that run the example notebooks while the documentation is built
# (docs/conf.py sets IPYTHONDIR): the figures as SVG, sharp at any size of the page.
c = get_config()  # noqa: F821
c.InlineBackend.figure_formats = {"svg"}
