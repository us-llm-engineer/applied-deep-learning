# src — the library under test

Every module here is covered by a frozen contract test in `tests/`. The notebooks import this code;
they do not reimplement it. Where a notebook needs something the library does not provide — an
experimental loss, a weighted conformal estimator — it is written in the notebook and labelled as
exploratory, rather than smuggled into `src/`.

## Data path

| module | responsibility |
|---|---|
| `bundle.py` | `assemble_bundle(root, seed, cap_per_label)` — the one real ingestion path: reads `<root>/<word>/*.wav` plus `_background_noise_/`, builds the five splits, and records provenance |
| `dataset.py`, `data.py` | split construction, label vocabulary (10 command words + `unknown` + `silence`), deterministic sampling |
| `audio.py` | waveform-domain transforms: gain, additive noise at a target SNR, exponential-decay reverb |
| `stress.py` | the stress conditions applied at evaluation time, separate from training augmentation |
| `cache.py` | feature cache with checksum provenance, so a stale cache cannot silently change a result |

Splits are **sample-level**, which permits the same speaker in more than one partition. That is a
known limitation, measured by `test_speaker_audit_contract.py` rather than assumed away.

## Models

`models.py` holds four architectures sharing one interface, which is what makes the comparison fair:

| class | params | note |
|---|---|---|
| `LogMelCNN` | 14,524 | legacy control. `AdaptiveAvgPool2d((1,1))` collapses both mel and time axes — it has no temporal representation to probe, by construction |
| `TimePoolCNN` | 17,212 | same convolutions, pools mel only, keeps 8 time slots |
| `WideCNN` | 271,692 | width 64, four blocks, time-preserving |
| `ConvGRU` | 291,564 | convolutions with mel-only pooling, then a bidirectional GRU over the time axis |

## Training and diagnostics

**`train.py`** — `train_classifier` is a generic loop. It takes an optional `probe_fn(model, epoch)`
callback and appends whatever it returns to `TrainResult.probe_history`; no paper-specific
instrumentation lives here.

It implements AutoClip inline: the per-step gradient L2 norm is recorded, and the clip threshold is
the `p`-th percentile of the accumulated history. Device is inferred from
`next(model.parameters()).device` — the caller decides placement, the loop respects it.

**`diagnostics.py`** — the paper-specific probes, kept out of the training loop:
`penultimate_and_sequence` (forward hooks that capture the classifier's input and any surviving
sequence axis), `project_2d`, `cluster_drift_metrics`, and `sentrycam_alert_epoch` implementing the
two-condition rule (sustained k=2 trend **and** `|M(t) − M(t−1)| > 0.25σ` against a 10-epoch rolling σ).

`penultimate_and_sequence` returns `None` for the sequence axis when none survives — which is the
honest answer for `LogMelCNN`, and is pinned by a test.

## Evaluation

| module | responsibility |
|---|---|
| `metrics.py` | accuracy, macro-F1, NLL, Brier, ECE at several bin counts, AURC, selective risk, OOD AUROC |
| `calibration.py` | temperature scaling, conformal thresholds, prediction sets |
| `arms.py` | orchestration: `run_arm` trains one arm and `summarise_arm` evaluates it across conditions with bootstrap intervals |
| `latency.py`, `budget.py` | batch-1 latency measurement and a wall-clock budget guard for offloaded runs |
| `encoder.py` | optional pretrained encoder loading; reports `unavailable` rather than substituting when weights are absent |

## Conventions

- **Determinism is a contract, not an aspiration.** Seeds are explicit and split construction is
  reproducible; `test_bundle_reproducibility_contract.py` enforces it.
- **Absence is reported, never substituted.** If a dataset, a checkpoint or an encoder is missing, the
  code says so and stops. No synthetic stand-in is ever silently inserted.
- **Normalization statistics come from the train split only** and are applied to the others — computed
  globally rather than per-utterance, because per-utterance normalization would cancel the very gain
  and noise shifts the study measures.
