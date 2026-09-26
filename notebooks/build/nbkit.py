"""Notebook builder kit. Part files expose `cells()` returning a list built with md()/code().

Assemble with `python notebooks/build/build_nbN.py`; execute with `python tools/run_notebook.py <path>`.
"""
from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

ROOT = Path(__file__).resolve().parents[2]

SETUP_CODE = '''\
# Shared setup: root-independent imports, seed, palette, labelled self-check helper.
from pathlib import Path
import sys

_here = Path.cwd().resolve()
ROOT = next((p for p in [_here, *_here.parents] if (p / "pyproject.toml").exists()), _here)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import matplotlib.pyplot as plt

RNG_SEED = 20260926
PALETTE = {
    "clean": "#1f77b4", "shifted": "#d62728", "mask": "#ff7f0e", "acoustic": "#2ca02c",
    "frozen": "#9467bd", "adapted": "#8c564b", "reference": "#7f7f7f", "band": "#c7c7c7",
}
plt.rcParams.update({"figure.dpi": 110, "axes.grid": True, "grid.alpha": 0.25, "axes.spines.top": False,
                     "axes.spines.right": False})

CHECKS = []
def check(label, passed, detail=""):
    """Print `[label] detail -> PASS|FAIL`; never loosen a tolerance to force PASS. Use note() for honest misses."""
    CHECKS.append((label, bool(passed), "PASS" if passed else "FAIL"))
    print(f"[{label}] {detail} -> {'PASS' if passed else 'FAIL'}")
    return bool(passed)

def note(label, detail):
    CHECKS.append((label, True, "NOTE"))
    print(f"[{label}] {detail} -> NOTE")

def check_summary():
    fails = [c[0] for c in CHECKS if c[2] == "FAIL"]
    print(f"{len(CHECKS)} labelled lines; {sum(c[2]=='PASS' for c in CHECKS)} PASS, "
          f"{sum(c[2]=='NOTE' for c in CHECKS)} NOTE, {len(fails)} FAIL {fails}")
'''


def md(text: str):
    return ("markdown", dedent(text).strip("\n"), [])


def code(text: str, tags: tuple[str, ...] = ()):
    return ("code", dedent(text).strip("\n"), list(tags))


def setup_cell():
    return ("code", SETUP_CODE.strip("\n"), ["setup"])


def assemble(filename: str, cell_specs) -> Path:
    nb = new_notebook()
    nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
    for kind, text, tags in cell_specs:
        cell = new_markdown_cell(text) if kind == "markdown" else new_code_cell(text)
        if tags:
            cell.metadata["tags"] = list(tags)
        nb.cells.append(cell)
    out = ROOT / "notebooks" / filename
    nbformat.write(nb, out)
    return out
