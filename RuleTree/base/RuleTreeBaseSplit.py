"""
This file defines an abstract base class for splitting strategies in RuleTree models.
It provides a framework that concrete implementations of split strategies must adhere to.
By inheriting from Python's Abstract Base Class (ABC), it enforces the implementation of necessary methods,
ensuring consistency and compatibility across different splitting algorithms used within RuleTree models.
"""
from abc import ABC, abstractmethod

class RuleTreeBaseSplit(ABC):
    """
    Abstract base class for split strategies in RuleTree models.

    This class defines the interface that all concrete implementations of split strategies must follow.
    It is designed to be used within RuleTree models to ensure consistent and compatible splitting behavior
    across different algorithms. By inheriting from Python's Abstract Base Class (ABC), it enforces the
    implementation of necessary methods, such as the constructor that initializes with a machine learning task type.

    Attributes:
        ml_task (str): The type of machine learning task for which the split strategy is being used
                       (e.g., 'classification', 'regression').
    """

    @abstractmethod
    def __init__(self, ml_task):
        """
        Initializes the RuleTreeBaseSplit with a specific machine learning task type.

        Args:
            ml_task (str): The type of machine learning task for which the split strategy is being used.
        """
        self.ml_task = ml_task
