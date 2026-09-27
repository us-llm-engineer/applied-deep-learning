"""Public pipeline API: dataset assembly (bundle) and per-arm training/evaluation (arms)."""
from .arms import ArmRun, run_arm, summarise_arm
from .bundle import (
    DatasetBundle, SplitArrays, assemble_bundle, audit_bundle,
    speaker_of, speakers_per_split, speaker_overlap,
)

__all__ = ["ArmRun", "DatasetBundle", "SplitArrays", "assemble_bundle", "audit_bundle", "speaker_of", "speakers_per_split", "speaker_overlap", "run_arm", "summarise_arm"]
