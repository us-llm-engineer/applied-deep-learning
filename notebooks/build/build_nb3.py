"""Assemble notebooks/03_project_walkthrough_part2.ipynb from part files (no execution).

Also concatenates every `# NB3-REUSE`-tagged cell from the nb2_part*.py modules into one real,
executed code cell placed near the top of the notebook body (after the title/setup, before the
notebook's own nb3_part1 content) -- this is the literal NB2 source, not a reimplementation. See
plans/round-04/nb2-nb3-growth-contract.md section 4.1 for the exact contract.
"""
import importlib.util
from pathlib import Path

import nbkit

HERE = Path(__file__).parent
NB2_PARTS = ["nb2_part1", "nb2_part2", "nb2_part3", "nb2_part4"]
PARTS = ["nb3_part1", "nb3_part2", "nb3_part3", "nb3_part4", "nb3_part5", "nb3_part6"]

TITLE = nbkit.md("""
# Notebook 3 of 3 — Project walkthrough part 2: augmentation, encoder, and calibration comparison

*Draft scaffold; reuses notebook 2's exposed API against the same in-memory synthetic bundle -- no
new data download, no T4/Colab run this round.* This notebook covers the augmentation comparison
(clean vs. masking vs. acoustic) across the project's three seeds, the frozen-vs-adapted encoder
framing (honestly reporting `EncoderUnavailable` when no network fetch is available), the
calibrated-vs-uncalibrated comparison on the project's own trained-arm logits, a documented
negative/null finding, and the limitations this round's evidence does not settle.
""")


def load(name):
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.cells()


def reused_nb2_cell():
    """Concatenate every NB3-REUSE-tagged cell from nb2_part*.py into one executed code cell."""
    fragments = []
    for name in NB2_PARTS:
        path = HERE / f"{name}.py"
        if not path.exists():
            continue
        for kind, text, tags in load(name):
            if kind == "code" and "NB3-REUSE" in tags:
                fragments.append(f"# --- reused from {name}.py ---\n{text}")
    body = "\n\n".join(fragments) if fragments else "# no NB3-REUSE cells found in nb2_part*.py"
    return nbkit.code(body, tags=("nb3-reuse-import",))


cells = [TITLE, nbkit.setup_cell(), reused_nb2_cell()]
for name in PARTS:
    if (HERE / f"{name}.py").exists():
        cells += load(name)
cells.append(nbkit.code("check_summary()"))
print(nbkit.assemble("03_project_walkthrough_part2.ipynb", cells))
