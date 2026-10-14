"""
Non-regression test: the models must keep producing the reference outputs stored in
tests/reference (predictions, leaves, probabilities, rules), and the trees saved with to_dict
by the version that generated them must still load and predict the same.

For intended changes of behavior, regenerate the references with
``python -m tests.reference.generate``.
"""
import json
from pathlib import Path

import numpy as np
import pytest

from RuleTree.tree.RuleTree import RuleTree
from tests.reference.cases import ALL_CASES, fit_case

REFERENCE_DIR = Path(__file__).parent / "reference"
with open(REFERENCE_DIR / "predictions.json") as f:
    REFERENCE = json.load(f)


def assert_same(actual, expected):
    actual, expected = np.asarray(actual), np.asarray(expected)
    if np.issubdtype(expected.dtype, np.floating):
        np.testing.assert_allclose(actual.astype(float), expected, rtol=1e-9, atol=1e-12)
    else:
        np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("name", ALL_CASES)
def test_same_outputs(name):
    _, _, outputs = fit_case(name)
    expected = REFERENCE[name]

    assert outputs.keys() == expected.keys()
    for key in expected:
        if key == "rules":
            assert outputs[key] == expected[key]
        else:
            assert_same(outputs[key], expected[key])


@pytest.mark.parametrize("name", sorted(p.stem for p in (REFERENCE_DIR / "trees").glob("*.json")))
def test_saved_trees_still_load(name):
    _, inputs, _ = fit_case(name)
    expected = REFERENCE[name]

    loaded = RuleTree.from_dict(str(REFERENCE_DIR / "trees" / f"{name}.json"))

    assert_same(loaded.predict(**inputs), expected["predict"])
    assert_same(loaded.apply(**inputs), expected["apply"])
