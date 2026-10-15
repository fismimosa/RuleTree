"""Import of the optional dependencies, with an error that tells how to install them."""

import importlib

from RuleTree.exceptions import MissingOptionalDependencyError

# module -> (package on PyPI, extra of RuleTree that installs it)
OPTIONAL_DEPENDENCIES = {
    "numba": ("numba", "shapelets"),
    "graphviz": ("graphviz", "graphviz"),
    "matplotlib": ("matplotlib", "graphviz"),
    "sklearn_extra": ("scikit-learn-extra", "kmedoids"),
}


def import_optional(module: str, feature: str):
    """
    Import an optional dependency.

    Parameters
    ----------
    module : str
        Module to import, a key of ``OPTIONAL_DEPENDENCIES`` or one of its submodules (e.g.
        ``"matplotlib.pyplot"``).
    feature : str
        What needs the module, used in the error message (e.g. ``"ShapeletTransformer"``).

    Returns
    -------
    module : module
        The imported module.

    Raises
    ------
    MissingOptionalDependencyError
        If the module is not installed.

    Examples
    --------
    >>> from RuleTree.utils.optional import import_optional
    >>> import_optional("numba", "ShapeletTransformer").__name__
    'numba'
    """
    try:
        return importlib.import_module(module)
    except ImportError as error:
        package, extra = OPTIONAL_DEPENDENCIES[module.split(".")[0]]
        raise MissingOptionalDependencyError(
            f"{feature} requires the optional package '{package}': install it with "
            f"pip install \"RuleTree[{extra}]\" (or pip install {package})."
        ) from error
