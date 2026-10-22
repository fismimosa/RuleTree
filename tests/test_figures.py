"""
The figures of the user guide: docs/make_figures.py still draws them (in the light and dark themes), and
the guides use existing files.
"""
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
        for file in (f"{name}.svg", f"{name}_dark.svg"):
            assert (tmp_path / file).read_text().startswith("<?xml"), file


def test_guides_use_existing_figures():
    referenced = set()
    for page in DOCS_DIR.rglob("*.rst"):
        if "_build" in page.parts or "generated" in page.parts:
            continue
        text = page.read_text()
        referenced |= set(re.findall(r"\.\. figure:: /_static/figures/(\S+)", text))
        for name in re.findall(r"\.\. themed-figure:: /_static/figures/(\S+)", text):  # light and dark
            stem, extension = name.rsplit(".", 1)
            referenced |= {name, f"{stem}_dark.{extension}"}

    assert referenced, "no figure in the guides"
    for name in referenced:
        assert (FIGURES_DIR / name).exists(), name
