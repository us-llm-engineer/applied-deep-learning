"""Assemble notebooks/02_project_walkthrough_part1.ipynb from part files (no execution)."""
import importlib.util
from pathlib import Path

import nbkit

HERE = Path(__file__).parent
TITLE = nbkit.md("""
# Notebook 2 of 3 — Project walkthrough part 1: data, QC, cache, and a synthetic baseline audit

*Draft scaffold; runs against an in-memory synthetic bundle (sine/noise waveforms), not a downloaded
Speech Commands archive.* This notebook builds the data and training harness that notebook 1's
measurement toolkit assumes: a real `DatasetBundle`, a real QC/audit pass, a real cache round-trip
with corruption detection, and a real (tiny-scale) CNN-arm comparison. All numbers here are from this
notebook's own synthetic bundle and are not comparable to the paper's speaker-disjoint 88.2% baseline
(see `training-pipeline.md` §11). Notebook 3 reuses this notebook's exposed API (`# NB3-REUSE`
tagged cells) and reports the augmentation/encoder/calibration comparison.
""")
PARTS = ["nb2_part1", "nb2_part2", "nb2_part3", "nb2_part4"]


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
print(nbkit.assemble("02_project_walkthrough_part1.ipynb", cells))
