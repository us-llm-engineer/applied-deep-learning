"""Assemble notebooks/02_small_scale_exploration.ipynb from part files (no execution).

DELIBERATELY SYNTHETIC: this notebook is a cheap sandbox that exercises the whole pipeline on generated data with a
known, learnable class structure. The sibling notebook 04 forbids synthetic data; this one requires it.
"""
from pathlib import Path

import nbkit

HERE = Path(__file__).parent
PARTS = ["nb2_part1", "nb2_part2", "nb2_part3"]

TITLE = nbkit.md("""
# Notebook 2 of 4 — Small-scale exploration on deliberately synthetic data

A cheap, CPU-only sandbox (a few minutes end to end) that exercises the study pipeline -- the four architectures
of `src/models.py`, the training loop with its in-training probes, calibration, and conformal risk control -- on
**synthetic** log-mel-shaped data with a real, learnable, time-varying class structure, and uses it to explore three
directions before anything expensive runs on real audio: an AutoClip percentile sweep, train-time versus post-hoc
calibration, and weighted versus unweighted conformal risk control under simulated shift. **Synthetic data is correct
and required here** (the real-data notebook 4 forbids it). Nothing in this notebook validates a conclusion on real
audio; see Section 9. Every figure is guarded by `assert_varied`, an executable check that a plotted series is not
constant.
""")


def load(name):
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.cells()


cells = [TITLE, nbkit.setup_cell()]
for name in PARTS:
    cells += load(name)
cells.append(nbkit.code("check_summary()"))
print(nbkit.assemble("02_small_scale_exploration.ipynb", cells))
