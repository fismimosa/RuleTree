"""The figures of the user guide: docs/make_figures.py still draws them, and the guides use existing files."""
import importlib.util
import re
from pathlib import Path

import pytest

DOCS_DIR = Path(__file__).parent.parent / "docs"
FIGURES_DIR = DOCS_DIR / "_static" / "figures"


def load_script():
    spec = importlib.util.spec_from_file_location("make_figures", DOCS_DIR / "make_figures.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.slow
def test_make_figures(tmp_path):
    pytest.importorskip("matplotlib")
    script = load_script()

    script.main(tmp_path)

    for name in script.FIGURES:
        assert (tmp_path / f"{name}.svg").read_text().startswith("<?xml")


def test_guides_use_existing_figures():
    referenced = set()
    for page in [*sorted((DOCS_DIR / "guides").glob("*.rst")), DOCS_DIR / "index.rst"]:
        referenced |= set(re.findall(r"\.\. figure:: /_static/figures/(\S+)", page.read_text()))

    assert referenced, "no figure in the guides"
    for name in referenced:
        assert (FIGURES_DIR / name).exists(), name
