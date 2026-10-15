"""
Non-regression test: the models must keep producing the reference outputs stored in
tests/reference (predictions, leaves, probabilities, rules), also after being saved with
to_dict and loaded with from_dict.

For intended changes of behavior, regenerate the references with
``python -m tests.reference.generate``.
"""
import json
import re
from pathlib import Path

import numpy as np
import pytest

from RuleTree.tree.RuleTree import RuleTree
from RuleTree.utils.data_utils import json_numpy_encoder
from tests.reference.cases import ALL_CASES, fit_case

# the slowest cases, skipped by pytest -m "not slow"
SLOW_CASES = {"ens_gboost_classifier", "reg_fair"}
CASES = [pytest.param(name, marks=pytest.mark.slow) if name in SLOW_CASES else name for name in ALL_CASES]

REFERENCE_DIR = Path(__file__).parent / "reference"
with open(REFERENCE_DIR / "predictions.json") as f:
    REFERENCE = json.load(f)


def assert_same(actual, expected):
    actual, expected = np.asarray(actual), np.asarray(expected)
    if np.issubdtype(expected.dtype, np.floating):
        np.testing.assert_allclose(actual.astype(float), expected, rtol=1e-9, atol=1e-12)
    else:
        np.testing.assert_array_equal(actual, expected)


# a rule on a feature of the tabular data, as written by get_rule (threshold rounded to 3 decimals)
TABULAR_RULE = re.compile(r"^X_(\d+) <= (\S+)$")


def assert_same_rules(model, inputs, actual, expected):
    """
    The same rules, or splits on another feature that divides the instances of the node in the same way.

    Features that give the same partition of a node have the same gain (e.g. petal length and petal width
    both separate setosa in iris): sklearn chooses among them by floating-point rounding, which changes
    between platforms (macOS ARM, Linux x86) and BLAS libraries. The leaves and the predictions are
    compared exactly anyway.
    """
    assert actual.keys() == expected.keys()
    leaves = np.asarray(model.apply(**inputs)).astype(str)

    for node_id, rule in expected.items():
        if actual[node_id] == rule:
            continue
        match = TABULAR_RULE.match(rule)
        assert match, f"node {node_id}: {actual[node_id]!r} != {rule!r}"

        idx = np.flatnonzero(np.char.startswith(leaves, node_id))  # instances reaching the node
        left = model.get_node_by_id(node_id).stump.apply(model._validate_inputs(**inputs), idx) == 1
        values = np.asarray(inputs["X"])[idx, int(match[1])].astype(float)
        threshold, rounding = float(match[2]), 5e-4

        assert values[left].max() < values[~left].min(), f"node {node_id}: {actual[node_id]!r} != {rule!r}"
        # sklearn puts the threshold halfway between two values: below the first value on the right
        assert values[left].max() - rounding <= threshold < values[~left].min(), \
            f"node {node_id}: {actual[node_id]!r} != {rule!r}"


@pytest.mark.parametrize("name", CASES)
def test_same_outputs(name):
    model, inputs, outputs = fit_case(name)
    expected = REFERENCE[name]

    assert outputs.keys() == expected.keys()
    for key in expected:
        if key == "rules":
            assert_same_rules(model, inputs, outputs[key], expected[key])
        else:
            assert_same(outputs[key], expected[key])


@pytest.mark.parametrize("name", CASES)
def test_same_outputs_after_saving(name):
    model, inputs, _ = fit_case(name)
    if not isinstance(model, RuleTree):
        pytest.skip("not a RuleTree")
    expected = REFERENCE[name]

    loaded = RuleTree.from_dict(json.loads(json.dumps(model.to_dict(), cls=json_numpy_encoder)))

    assert_same(loaded.predict(**inputs), expected["predict"])
    assert_same(loaded.apply(**inputs), expected["apply"])
