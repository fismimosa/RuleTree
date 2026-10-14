"""
Optional profiling support.

``profile`` is the decorator from ``line_profiler_pycharm`` when that package is installed,
otherwise a no-op decorator, so it can be used anywhere without adding a hard dependency:

    from RuleTree.utils.profiling import profile

    @profile
    def fit(...):
        ...
"""

try:
    from line_profiler_pycharm import profile
except ImportError:
    def profile(func):
        return func
