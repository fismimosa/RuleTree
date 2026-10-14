"""
Regenerate the reference outputs of the non-regression test:

    python -m tests.reference.generate

Run it only for intended changes of behavior: the test checks that the code keeps
producing the outputs stored here.
"""
import json
from pathlib import Path

from RuleTree.utils.data_utils import json_numpy_encoder
from tests.reference.cases import ALL_CASES, fit_case

REFERENCE_DIR = Path(__file__).parent
PREDICTIONS_FILE = REFERENCE_DIR / "predictions.json"


def main():
    predictions = {}
    for name in ALL_CASES:
        _, _, outputs = fit_case(name)
        predictions[name] = outputs
        print(f"{name}: done")

    with open(PREDICTIONS_FILE, "w") as f:
        json.dump(predictions, f, cls=json_numpy_encoder)


if __name__ == "__main__":
    main()
