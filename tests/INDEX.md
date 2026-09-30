# Round-one and round-two contracts

Run the contract suite from this directory:

```bash
python -m pytest tests -q
```

The tests specify observable behavior, not internal layout.

| Contract | Test file | Oracle |
| --- | --- | --- |
| Disjoint, seeded, label-preserving splits | `test_data_contract.py` | Exact identifier sets and per-label counts |
| Gain, noise, and reverb records | `test_audio_contract.py` | Shape, finite values, deterministic seed, and known gain |
| Probability, calibration, abstention, and interval metrics | `test_metrics_contract.py` | Hand calculations and seeded repeatability |
| Calibration and cached-array provenance | `test_calibration_cache_contract.py` | Ordering, nested sets, conservative threshold, and round trip |
| Compact manifest and duplicate audit | `test_dataset_contract.py` | Stable digest, fixed labels, cap, and collision identifiers |
| Tiny feature/model and paired evaluation | `test_experiment_contract.py` | Deterministic tensors, shared budgets, known metrics, and resources |

## Round 3 contracts

| Contract | Test file | Oracle |
| --- | --- | --- |
| Eleven stress conditions: seeded, batch-independent, realised SNR, gain clipping, reverb tail ordering, validation | `test_stress_contract.py` | Independent SNR definition within 0.05 dB, `clip(x * 10**(G/20))`, tail-energy ordering, row equality via `index_offset` |
| Injected-clock time budget and skip log | `test_budget_contract.py` | Fake clock, exact boundary at the limit, monotone guard sequence, skipped-record list |
| Pipeline data assembly and audit: caps, 70/15/15 partition, unknown/OOD word rules, silence time regions, digest stability, shortfalls, planted duplicates | `test_pipeline_data_contract.py` | Counts within one clip of exact allocation, window location inside noise files, planted-defect bundles |
| Arm training, evaluation and summary: paired stress rows, equal optimizer steps, training-only augmentation, calibration/CRC/AUROC | `test_pipeline_arm_contract.py` | Training-insensitive fixed model, hand-computed NLL/Brier/AUROC, reference `temperature_scale`/`crc_threshold` |
| Encoder helpers: unavailable path, masked mean pool, layer extraction, adapted block | `test_encoder_contract.py` | Hand-pooled tensors, fake duck-typed encoder, exact parameter count 627 |
| RIR reverb `apply_rir_reverb`: seeded, exact DRR, unit-impulse proportionality, RMS level match without clipping, Schroeder RT60 within 20%, late-energy ordering, silence, validation; legacy `apply_reverb` unchanged | `test_rir_reverb_contract.py` | Impulse response recovered as output/level_scale, independent Schroeder fit, DRR to 1e-6 dB, independent legacy kernel |

Shared builders live in `pipeline_builders.py` (not a test module).

## Round 4 contracts

Chosen plan: `plans/round-04/plan-a.md` (single plan — this round's three requirements are Codex's
own frozen `review/round-02.md` Round-3 contract, carried forward since none of it was committed).

| Contract | Test file | Oracle |
| --- | --- | --- |
| `crc_threshold` infeasible-alpha fallback (`alpha*(n+1) < 1` must return exactly 0.0; feasible regime unchanged) | `test_calibration_infeasible_alpha_contract.py` | Hand-computed order-statistic index for the feasible regime; exact-zero assertion (not "minimum") for the infeasible regime |
| `silence_windows` region invariant (no window may spill outside its split's fractional region of its source file; resample-else-raise) plus `audit_bundle`'s real `silence_time_overlap` | `test_silence_provenance_contract.py` | In-region offset/content checks against a short-file regression case; fabricated-provenance `audit_bundle` cases including the val-checked-against-train-region aliasing case |

R4.3 (NB2/NB3 growth) has no pytest contract of its own — its harness is
the notebook growth contract plus the existing scale floor
check, per the project's established notebook-growth convention (frozen baseline + named section
contract, not a separate test file).

## Mutation targets

- Seed noise per batch instead of per example, drop clipping after gain, or invert reverb decay ordering: named stress tests must fail.
- Restart the clock in a second `start()`, let `remaining()` go negative, or record skips on `True`: named budget tests must fail.
- Change a split boundary or reuse one identifier: `test_stratified_split_is_disjoint_reproducible_and_label_preserving` must fail.
- Ignore the noise seed or alter gain conversion: `test_acoustic_transforms_are_seeded_shape_stable_and_finite` or `test_gain_has_the_declared_multiplicative_effect_without_clipping` must fail.
- Replace classwise Brier averaging, reverse confidence ordering, or omit bootstrap seeding: named metric tests must fail.
- Reverse prediction-set threshold ordering, accept stale provenance, or relax the CRC correction: named calibration/cache tests must fail.
- Include an out-of-set label, make manifest hashing order-dependent, or hide a duplicate checksum: named dataset tests must fail.
- Change log-mel frame construction, change a condition's optimization budget, unpair prediction rows, or drop resource fields: named experiment tests must fail.
- Return the empirical minimum instead of exactly 0.0 for an infeasible CRC alpha, or off-by-one the feasible-regime clamp: named `test_calibration_infeasible_alpha_contract.py` tests must fail.
- Let a silence window spill outside its split's region on a short source file, or check `val` provenance against a `val`-shaped region instead of `train`'s: named `test_silence_provenance_contract.py` tests must fail.

## Figure specifications

Figure specifications are generated from prior notebook computation and deliberately contain no stored measurements.
