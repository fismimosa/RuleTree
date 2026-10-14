"""Exceptions and warnings of RuleTree."""

class NoSplitFoundWarning(RuntimeWarning):
    """
    Raised by a stump (or splitter, or transformer) that cannot split the instances of a node.

    The tree catches it and makes the node a leaf. It is a warning, not an error: no split is a
    normal outcome, e.g. for a node whose instances all have the same feature values.
    """
    pass