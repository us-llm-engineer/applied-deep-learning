"""Public pipeline API: dataset assembly (bundle) and per-arm training/evaluation (arms)."""
from .arms import ArmRun, run_arm, summarise_arm
from .bundle import DatasetBundle, SplitArrays, assemble_bundle, audit_bundle

__all__ = ["ArmRun", "DatasetBundle", "SplitArrays", "assemble_bundle", "audit_bundle", "run_arm", "summarise_arm"]
