"""
This file defines the base class for all RuleTree models. It provides an abstract interface
that other RuleTree implementations should adhere to. By inheriting from scikit-learn's BaseEstimator
and Python's Abstract Base Class (ABC), it ensures compatibility with scikit-learn's ecosystem and
enforces the implementation of necessary methods.
"""
from abc import ABC

from sklearn.base import BaseEstimator


class RuleTreeBase(BaseEstimator, ABC):
    """
    Base class for RuleTree models.

    This class serves as an abstract base class for all RuleTree implementations.
    It inherits from scikit-learn's BaseEstimator and Python's Abstract Base Class (ABC).

    Attributes:
        None
    """
    pass
