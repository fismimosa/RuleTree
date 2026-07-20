from abc import ABC, abstractmethod
import numpy as np


class IImpurity(ABC):
    """
    Interfaccia per qualsiasi funzione di impurità.
    """

    @staticmethod
    def calculate_gain(sx_mask, y_onehot, stump, Idx) -> tuple:
        pass

    @staticmethod
    def gini(counts: np.ndarray) -> np.ndarray:
        pass

    @staticmethod
    def entropy(counts: np.ndarray) -> np.ndarray:
        pass
