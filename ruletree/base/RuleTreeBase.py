from abc import ABC, abstractmethod

import numpy as np
from sklearn import tree
from sklearn.base import BaseEstimator

from ruletree.RuleTreeNode import RuleTreeNode


class RuleTreeBase(BaseEstimator, ABC):
    pass