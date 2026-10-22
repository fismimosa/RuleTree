import subprocess
import sys

import pytest

from RuleTree.utils import profiling

pytestmark = pytest.mark.slow  # each test starts a new Python process

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


def test_optional_dependencies_are_needed_only_by_their_features():
    run_python(
        "import sys\n"
        "for name in ['numba', 'matplotlib', 'graphviz', 'sklearn_extra']: sys.modules[name] = None\n"
        f"for module in {MODULES!r}: __import__(module)\n"
        "import numpy as np\n"
        "from RuleTree import RuleTreeClassifier\n"
        "from RuleTree.exceptions import MissingOptionalDependencyError\n"
        "from RuleTree.stumps.classification import PartialPivotTreeStumpClassifier, ShapeletTreeStumpClassifier\n"
        "X, y = np.random.default_rng(0).normal(size=(40, 1, 20)), np.arange(40) % 2\n"
        "model = RuleTreeClassifier(max_depth=2).fit(X[:, 0], y)\n"
        "outside = {'context': None, 'node_id': None, 'cache': None}\n"
        "for fit in [lambda: ShapeletTreeStumpClassifier().fit(X, y, data_type='ts', **outside),\n"
        "            lambda: PartialPivotTreeStumpClassifier().fit(X[:, 0], y, **outside),\n"
        "            lambda: model.root.export_graphviz(),\n"
        "            lambda: model.plot_tree()]:\n"
        "    try:\n"
        "        fit()\n"
        "        raise AssertionError('no error')\n"
        "    except MissingOptionalDependencyError as error:\n"
        "        assert 'pip install \"RuleTree[' in str(error), error\n"
    )


def test_profile_uses_line_profiler_when_installed():
    line_profiler_pycharm = pytest.importorskip("line_profiler_pycharm")

    assert profiling.profile is line_profiler_pycharm.profile
