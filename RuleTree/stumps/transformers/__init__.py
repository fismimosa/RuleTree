from .MatrixTransformer import MatrixTransformer
from .PivotTransformer import PivotTransformer
from .ShapeletTransformer import ShapeletTransformer
from .PartialPivotTransformer import PartialPivotTransformer
from .ObliqueTransformer import ObliqueTransformer
from .HouseholderTransformer import HouseholderTransformer
from .BivariateTransformer import BivariateTransformer
from .ProximityTransformer import ProximityTransformer
from .PartialProximityTransformer import PartialProximityTransformer
from .ProximityPivotTransformer import ProximityPivotTransformer
from .ObliquePivotTransformer import ObliquePivotTransformer

__all__ = ["MatrixTransformer", "PivotTransformer", "ShapeletTransformer", "PartialPivotTransformer",
           "ObliqueTransformer", "HouseholderTransformer", "BivariateTransformer",
           "ProximityTransformer", "PartialProximityTransformer",
           "ProximityPivotTransformer", "ObliquePivotTransformer"]

from RuleTree.utils.deprecation import deprecated_aliases  # noqa: E402

__getattr__ = deprecated_aliases(__name__, {"MultiplePivotTransformer": ProximityPivotTransformer})
