"""Assemble notebooks/04_real_dataset_walkthrough.ipynb from part files (no execution).

Content is computed from the real-run artifacts (Drive on the execution host, runtime/ mirror locally).
No mock or synthetic data may appear in this notebook, per explicit user instruction.
"""
from pathlib import Path

import nbkit

HERE = Path(__file__).parent
PARTS = ["nb4_part1", "nb4_part2", "nb4_part3", "nb4_part4"]

TITLE = nbkit.md("""
# Notebook 4 of 4 — Real-run walkthrough: architecture race, in-training diagnostics, post-training results

Built entirely from artifacts of a real GPU training run (clean condition, seed 17, three instrumented
architectures: `gru`, `timepool`, `wide`) -- no mock or synthetic data anywhere. Contents: artifact
provenance and architecture comparison (with an honest negative finding); checkpoint verification and an
`offloaded` accuracy re-derivation; in-training diagnostics (AutoClip gradient clipping, SentryCam-style 2D
cluster health, and the raw-space-versus-2D alert comparison as the headline); MM-PHATE-style statistics
(PCA embedding, labelled as a deviation); post-training calibration, selective-risk and conformal results
under **exploration-config** settings (4 stress conditions, 200 bootstrap resamples: not publication numbers);
and limitations. Runs on the remote Colab runtime (Drive artifacts) and, for editing and review, on the
authoring host against the local `runtime/` mirror. Notebooks 1-3 and `src/` are not modified.
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
print(nbkit.assemble("04_real_dataset_walkthrough.ipynb", cells))
