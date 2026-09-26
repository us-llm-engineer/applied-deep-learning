"""Assemble notebooks/01_research_foundations.ipynb from part files (no execution)."""
import importlib.util
from pathlib import Path

import nbkit

HERE = Path(__file__).parent
TITLE = nbkit.md("""
# Notebook 1 of 3 — Research foundations: calibration, abstention, and acoustic shift

*Draft scaffold on toy data; not a production result.* This notebook builds the measurement toolkit for a study of robust speech-command recognition under acoustic shift. It follows a measure → decide → monitor arc: probability metrics and calibration (measure), selective classification (decide), and conformal risk control (monitor), plus acoustic shift operators and a NoisyMix-style augmentation hypothesis.

Every section carries a provenance tag: `(derived here)` or `(source: paper; q-tag)`. Paper claims are quoted or paraphrased only from the recorded source answers (q-tags) or the independently re-read pages of the selective-classification paper (tag `L`). Toy numbers are never paper results. Companion notebooks: 02 builds the data and training harness on real Speech Commands; 03 reports the trained comparison.
""")
PARTS = ["nb1_part1", "nb1_part2", "nb1_part3", "nb1_part4"]


def load(name):
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.cells()


cells = [TITLE, nbkit.setup_cell()]
for name in PARTS:
    if (HERE / f"{name}.py").exists():
        cells += load(name)
cells.append(nbkit.code("check_summary()"))
print(nbkit.assemble("01_research_foundations.ipynb", cells))
