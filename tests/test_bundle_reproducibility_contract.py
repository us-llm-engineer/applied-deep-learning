"""Cross-interpreter reproducibility contract for capped bundle selection."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

from pipeline_builders import build_dataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROGRAM = """
import json
import sys
from pathlib import Path

from src.pipeline import assemble_bundle

bundle = assemble_bundle(Path(sys.argv[1]), seed=int(sys.argv[2]), cap_per_label=4)
in_distribution = ("train", "val", "cal", "test")
unknown = []
yes_ids = []
for split_name in in_distribution:
    split = bundle.splits[split_name]
    for clip_id, word in zip(split.ids, split.words):
        if word == "yes":
            yes_ids.append(clip_id)
        if word in bundle.train_unknown_words:
            unknown.append([clip_id, word])

print(json.dumps({
    "splits": {name: list(bundle.splits[name].ids) for name in (*in_distribution, "ood")},
    "manifest_sha256": bundle.manifest_sha256,
    "unknown_identities": sorted(unknown),
    "ood_identities": list(zip(bundle.splits["ood"].ids, bundle.splits["ood"].words)),
    "capped_yes_ids": sorted(yes_ids),
}, sort_keys=True, separators=(",", ":")))
"""


def _assemble_in_fresh_interpreter(dataset_root: Path, seed: int, hash_seed: str) -> dict:
    env = os.environ | {
        "PYTHONHASHSEED": hash_seed,
        "PYTHONPATH": os.pathsep.join(filter(None, (str(PROJECT_ROOT), os.environ.get("PYTHONPATH")))),
    }
    completed = subprocess.run(
        [sys.executable, "-c", PROGRAM, str(dataset_root), str(seed)],
        cwd=PROJECT_ROOT,
        env=env,
        check=True,
        text=True,
        capture_output=True,
    )
    return json.loads(completed.stdout)


def test_capped_bundle_selection_is_reproducible_across_python_hash_salts(tmp_path):
    """A seed denotes one manifest even when a new Python process has a new hash salt."""
    dataset_root = build_dataset(
        tmp_path / "speech",
        command_clips=4,
        aux_clips=4,
        clips_by_word={"yes": 11},
    )

    first = _assemble_in_fresh_interpreter(dataset_root, seed=17, hash_seed="1")
    second = _assemble_in_fresh_interpreter(dataset_root, seed=17, hash_seed="2")
    different_experiment = _assemble_in_fresh_interpreter(dataset_root, seed=18, hash_seed="1")

    assert second == first
    assert different_experiment["capped_yes_ids"] != first["capped_yes_ids"]
