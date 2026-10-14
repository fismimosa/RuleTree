"""Abstract base class of the split strategies with a task, such as PivotSplit."""
from abc import ABC, abstractmethod

class RuleTreeBaseSplit(ABC):
    """
    Abstract base class of the split strategies that depend on the task of the tree, such as
    :class:`RuleTree.stumps.splitters.PivotSplit.PivotSplit`.

    Not to be confused with :class:`RuleTree.base.RuleTreeBaseSplitter.RuleTreeBaseSplitter`, the
    base class of the splitters used by the stumps.

    Parameters
    ----------
    ml_task : str
        Task: MODEL_TYPE_CLF, MODEL_TYPE_REG or MODEL_TYPE_CLU.

    Attributes
    ----------
    ml_task : str
        Task of the split strategy.
    """

    @abstractmethod
    def __init__(self, ml_task):
        """
        Store the task of the split strategy.

        Parameters
        ----------
        ml_task : str
            Task: MODEL_TYPE_CLF, MODEL_TYPE_REG or MODEL_TYPE_CLU.
        """
        self.ml_task = ml_task
