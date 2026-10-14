"""
Regenerate the reference outputs of the non-regression test:

    python -m tests.reference.generate

Run it only for intended changes of behavior: the test checks that the code keeps
producing the outputs stored here. The trees are also saved with to_dict, to check
that dictionaries saved by this version can still be loaded.
"""
import json
from pathlib import Path

from RuleTree.tree.RuleTree import RuleTree
from RuleTree.utils.data_utils import json_numpy_encoder
from tests.reference.cases import ALL_CASES, fit_case

REFERENCE_DIR = Path(__file__).parent
PREDICTIONS_FILE = REFERENCE_DIR / "predictions.json"
TREES_DIR = REFERENCE_DIR / "trees"


def main():
    TREES_DIR.mkdir(exist_ok=True)
    predictions = {}
    for name in ALL_CASES:
        model, _, outputs = fit_case(name)
        predictions[name] = outputs

        if isinstance(model, RuleTree):
            with open(TREES_DIR / f"{name}.json", "w") as f:
                json.dump(model.to_dict(), f, cls=json_numpy_encoder)
        print(f"{name}: done")

    with open(PREDICTIONS_FILE, "w") as f:
        json.dump(predictions, f, cls=json_numpy_encoder)


if __name__ == "__main__":
    main()
