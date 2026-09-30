"""Assemble notebooks/01_research_foundations.ipynb from part files (no execution).

Measured numbers come from runtime/metrics/ artifacts; literature numbers are quoted from the papers,
attributed by session and turn in the traceability table of section 1.
The only constructed inputs are labelled unit-test fixtures for formulas and estimators.
"""
from pathlib import Path

import nbkit

HERE = Path(__file__).parent
PARTS = ["nb1_part1", "nb1_part2", "nb1_part3", "nb1_part4"]

TITLE = nbkit.md("""
# Notebook 1 of 4 -- Research foundations: the mathematics behind a 0.9008-versus-0.3730 result

A ConvGRU speech-command classifier reached 0.9008 validation accuracy where a globally-average-pooled CNN reached
0.3730. This notebook derives the mathematics that is supposed to explain why, and states honestly where the
literature and our measurements agree, disagree, or cannot be compared. It covers: the traceability table for every
borrowed method; why a recurrent head preserves the temporal order that global average pooling destroys; a real
tension with the CRNN paper's noise-adaptation mechanism (our `gru` is the *worst* arm under `noise_10`); neural
collapse with its terminal-phase precondition and two measurement deviations; adaptive gradient clipping;
calibration (temperature scaling, ECE, MDCA); the salvaged derivations of proper scoring rules and conformal risk
control with a Monte-Carlo check; and limitations. No model is trained here: measured quantities are loaded from
the real run's artifacts, and it runs locally on CPU in well under a minute. Single seed, exploration configuration
(4 stress conditions, 200 bootstrap resamples): none of the parallels drawn below is a causal claim.
""")


def load(name):
    import importlib.util
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
