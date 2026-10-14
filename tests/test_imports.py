import subprocess
import sys

import pytest

from RuleTree.utils import profiling

MODULES = [
    "RuleTree",
    "RuleTree.stumps.classification",
    "RuleTree.stumps.regression",
    "RuleTree.stumps.regression.XGBoostStumpRegressor",
    "RuleTree.tree.RuleTreeXGBoostRegressor",
    "RuleTree.ensemble.GBoostRegressor",
    "RuleTree.ensemble.GBoostClassifier",
    "RuleTree.ensemble.XGBoostRegressor",
    "RuleTree.ensemble.EBMRegressor",
]


def run_python(code):
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_import_does_not_change_warning_filters():
    # third-party dependencies install their own filters on import: load them first
    run_python(
        "import warnings\n"
        "import numpy, scipy, pandas, sklearn, numba, category_encoders, matplotlib.pyplot\n"
        "before = list(warnings.filters)\n"
        f"for module in {MODULES!r}: __import__(module)\n"
        "assert warnings.filters == before, 'importing RuleTree changed the warning filters'\n"
    )


def test_import_emits_no_warnings():
    run_python(
        "import warnings\n"
        "import numpy, scipy, pandas, sklearn, numba, category_encoders, matplotlib.pyplot\n"
        "with warnings.catch_warnings(record=True) as caught:\n"
        "    warnings.simplefilter('always')\n"
        f"    for module in {MODULES!r}: __import__(module)\n"
        "assert not caught, [str(w.message) for w in caught]\n"
    )


def test_import_without_line_profiler():
    run_python(
        "import sys\n"
        "sys.modules['line_profiler_pycharm'] = None  # makes the import fail as if not installed\n"
        f"for module in {MODULES!r}: __import__(module)\n"
        "from RuleTree.utils.profiling import profile\n"
        "f = lambda: 1\n"
        "assert profile(f) is f\n"
    )


def test_profile_uses_line_profiler_when_installed():
    line_profiler_pycharm = pytest.importorskip("line_profiler_pycharm")

    assert profiling.profile is line_profiler_pycharm.profile
