from .DecisionTreeStumpClassifier import DecisionTreeStumpClassifier
from .PivotTreeStumpClassifier import PivotTreeStumpClassifier
from .ShapeletTreeStumpClassifier import ShapeletTreeStumpClassifier
from .PartialPivotTreeStumpClassifier import PartialPivotTreeStumpClassifier
from .ObliqueDecisionTreeStumpClassifier import ObliqueDecisionTreeStumpClassifier
from .ProximityTreeStumpClassifier import ProximityTreeStumpClassifier
from .PartialProximityTreeStumpClassifier import PartialProximityTreeStumpClassifier
from .ProximityPivotTreeStumpClassifier import ProximityPivotTreeStumpClassifier
from .ObliquePivotTreeStumpClassifier import ObliquePivotTreeStumpClassifier
from .ObliqueProximityPivotTreeStumpClassifier import ObliqueProximityPivotTreeStumpClassifier

__all__ = [
    "DecisionTreeStumpClassifier",
    "PivotTreeStumpClassifier",
    "ShapeletTreeStumpClassifier",
    "PartialPivotTreeStumpClassifier",
    "ObliqueDecisionTreeStumpClassifier",
    "ProximityTreeStumpClassifier",
    "PartialProximityTreeStumpClassifier",
    "ProximityPivotTreeStumpClassifier",
    "ObliquePivotTreeStumpClassifier",
    "ObliqueProximityPivotTreeStumpClassifier",
]

from RuleTree.utils.deprecation import deprecated_aliases  # noqa: E402

__getattr__ = deprecated_aliases(__name__, {"MultiplePivotTreeStumpClassifier": ProximityPivotTreeStumpClassifier,
                                            "MultipleObliquePivotTreeStumpClassifier":
                                                ObliqueProximityPivotTreeStumpClassifier})
