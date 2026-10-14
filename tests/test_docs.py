"""
The examples of the guides in docs/guides (and of docs/index.rst) run and print what the guides show.

In each guide, the ``.. code-block:: python`` blocks are run in order in the same namespace. When a
block is followed by a ``.. code-block:: text`` block with ``:caption: Output``, what the code
prints must be that text.
"""
import contextlib
import io
import re
from pathlib import Path

import pytest

DOCS_DIR = Path(__file__).parent.parent / "docs"
GUIDES = sorted((DOCS_DIR / "guides").glob("*.rst")) + [DOCS_DIR / "index.rst"]


def parse_blocks(text: str) -> list[tuple[str, str, str]]:
    """The code blocks of an rst document, as (language, caption, content)."""
    blocks = []
    lines = text.split("\n")
    i = 0
    while i < len(lines):
        match = re.match(r"\.\. code-block:: (\w+)\s*$", lines[i])
        if not match:
            i += 1
            continue

        language, caption = match.group(1), ""
        i += 1
        while i < len(lines) and re.match(r"\s+:\w+:", lines[i]):  # options of the directive
            option = re.match(r"\s+:caption:\s*(.*)$", lines[i])
            caption = option.group(1).strip() if option else caption
            i += 1

        content = []
        while i < len(lines) and (lines[i].strip() == "" or lines[i].startswith("   ")):
            content.append(lines[i])
            i += 1
        indent = min((len(line) - len(line.lstrip()) for line in content if line.strip()), default=0)
        blocks.append((language, caption, "\n".join(line[indent:] for line in content).strip("\n")))

    return blocks


@pytest.mark.parametrize("guide", GUIDES, ids=lambda path: path.stem)
def test_guide_examples(guide, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # files written by the examples
    blocks = parse_blocks(guide.read_text())
    namespace = {"__name__": f"guide_{guide.stem}"}

    for position, (language, caption, content) in enumerate(blocks):
        if language != "python":
            continue

        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            exec(compile(content, f"{guide.name} block {position}", "exec"), namespace)

        following = blocks[position + 1] if position + 1 < len(blocks) else None
        if following is not None and following[0] == "text" and following[1] == "Output":
            assert output.getvalue().strip() == following[2].strip(), f"{guide.name} block {position}"
