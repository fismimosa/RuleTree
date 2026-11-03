"""Deprecated names of the library, still importable with a warning, also in the trees saved with them."""

import importlib
import warnings

# module and name of a renamed class -> its current module and name, for the trees saved before
RENAMED_CLASSES = {
    ("RuleTree.stumps.classification.MultiplePivotTreeStumpClassifier", "MultiplePivotTreeStumpClassifier"):
        ("RuleTree.stumps.classification.ProximityPivotTreeStumpClassifier", "ProximityPivotTreeStumpClassifier"),
    ("RuleTree.stumps.classification.MultipleObliquePivotTreeStumpClassifier",
     "MultipleObliquePivotTreeStumpClassifier"):
        ("RuleTree.stumps.classification.ObliqueProximityPivotTreeStumpClassifier",
         "ObliqueProximityPivotTreeStumpClassifier"),
    ("RuleTree.stumps.transformers.MultiplePivotTransformer", "MultiplePivotTransformer"):
        ("RuleTree.stumps.transformers.ProximityPivotTransformer", "ProximityPivotTransformer"),
}


def deprecated_aliases(module_name: str, aliases: dict):
    """
    The ``__getattr__`` of a module where some old names are still available, with a FutureWarning.

    Parameters
    ----------
    module_name : str
        Name of the module (``__name__``).
    aliases : dict
        Old name -> the object that replaces it.

    Returns
    -------
    getattr : callable
        Function to assign to ``__getattr__`` in the module (PEP 562).

    Examples
    --------
    >>> import warnings
    >>> from RuleTree.utils.deprecation import deprecated_aliases
    >>> getattr_ = deprecated_aliases("module", {"old_sum": sum})
    >>> with warnings.catch_warnings(record=True) as caught:
    ...     warnings.simplefilter("always")
    ...     getattr_("old_sum") is sum, str(caught[0].message)
    (True, 'old_sum is deprecated, use sum (builtins) instead')
    """
    def __getattr__(name):
        if name in aliases:
            new = aliases[name]
            warnings.warn(f"{name} is deprecated, use {new.__name__} ({new.__module__}) instead", FutureWarning,
                          stacklevel=2)
            return new
        raise AttributeError(f"module {module_name!r} has no attribute {name!r}")

    return __getattr__


def import_class(module: str, name: str) -> type:
    """
    A class from its module and name, as saved in the dictionary of a tree; a class saved with a name
    that has changed is the current one, with a FutureWarning.

    Parameters
    ----------
    module : str
        Module of the class.
    name : str
        Name of the class.

    Returns
    -------
    cls : type

    Examples
    --------
    >>> from RuleTree.utils.deprecation import import_class
    >>> import_class("RuleTree.tree.RuleTreeClassifier", "RuleTreeClassifier").__name__
    'RuleTreeClassifier'
    """
    if (module, name) in RENAMED_CLASSES:
        new_module, new_name = RENAMED_CLASSES[(module, name)]
        warnings.warn(f"{name} is deprecated, use {new_name} ({new_module}) instead: the tree was saved with the "
                      f"old name", FutureWarning, stacklevel=2)
        module, name = new_module, new_name
    return getattr(importlib.import_module(module), name)
