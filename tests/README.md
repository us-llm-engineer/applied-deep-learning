# tests

358 contract tests. They are written **before** the code they cover and then frozen: an implementation
is changed to satisfy a test, not the other way round. A test that needs changing is treated as a
specification change and justified as one.

```bash
python3 -m pytest tests -q        # 358 passed
```

## What is covered

| area | files |
|---|---|
| data and splits | `test_bundle_reproducibility_contract.py`, `test_dataset_contract.py`, `test_data_contract.py`, `test_unknown_holdout.py` |
| audio transforms | `test_audio_contract.py`, `test_rir_reverb_contract.py`, `test_snr_and_cache_integrity.py`, `test_stress_contract.py` |
| silence provenance | `test_silence_windows.py`, `test_silence_provenance_contract.py` |
| speaker leakage | `test_speaker_audit_contract.py`, `test_speaker_audit_edge_contract.py` |
| training loop | `test_train_contract.py`, `test_experiment_contract.py`, `test_pipeline_arm_contract.py`, `test_pipeline_data_contract.py` |
| in-training probes | `test_intraining_probe_contract.py`, `test_sentrycam_diagnostics_contract.py` |
| evaluation | `test_metrics_contract.py`, `test_calibration_cache_contract.py`, `test_calibration_infeasible_alpha_contract.py` |
| resources | `test_latency_contract.py`, `test_budget_contract.py`, `test_encoder_contract.py` |

`pipeline_builders.py` constructs seeded synthetic WAV fixtures so the data path can be exercised
without the 2.4 GB corpus. `INDEX.md` maps each contract to the behaviour it pins.

## Tests worth knowing about

- **The provenance oracle** in `test_pipeline_arm_contract.py` independently reconstructs which clips
  a training run should have seen and asserts the loop saw exactly those, once per epoch. It has
  caught a real leakage bug.
- **`test_intraining_probe_contract.py`** pins that `LogMelCNN` exposes *no* sequence axis while the
  three time-preserving models do. That asymmetry is the architectural claim of the whole study, so
  it is asserted rather than assumed.
- **CUDA tests** are `skipif`-gated on `torch.cuda.is_available()`. They skip on CPU-only machines and
  run for real on a GPU host; skipping is reported, never silently passed.
