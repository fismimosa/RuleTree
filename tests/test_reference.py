"""
Non-regression test: the models must keep producing the reference outputs stored in
tests/reference (predictions, leaves, probabilities, rules), also after being saved with
to_dict and loaded with from_dict.

For intended changes of behavior, regenerate the references with
``python -m tests.reference.generate``.
"""
import json
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


@pytest.mark.parametrize("name", CASES)
def test_same_outputs(name):
    _, _, outputs = fit_case(name)
    expected = REFERENCE[name]

    assert outputs.keys() == expected.keys()
    for key in expected:
        if key == "rules":
            assert outputs[key] == expected[key]
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
