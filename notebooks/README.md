# Notebooks

Three notebooks, read in order. The numbering skips `03`: an earlier `01`–`03` ran on synthetic
fallbacks and were retired, and `04` keeps its number because it is the executed artifact the figures
and metrics are named after.

| notebook | data | role |
|---|---|---|
| `01_research_foundations.ipynb` | none (derivations) | the mathematics behind the measured result |
| `02_small_scale_exploration.ipynb` | **synthetic, by design** | cheap sandbox; explores directions the real run could not afford |
| `04_real_dataset_walkthrough.ipynb` | **real Speech Commands v0.02** | the study itself |

Synthetic data is correct in `02` and forbidden in `04`. That split is deliberate: `02` is where a
method may be exercised cheaply; `04` is where every number must trace to real audio.

## Scale

| | cells | figures | checks | equations | errors |
|---|---|---|---|---|---|
| nb1 | 82 | 21 | 196 | 553 | 0 |
| nb2 | 81 | 20 | 237 | 129 | 0 |
| nb4 | 153 | 32 | 306 | 305 | 0 |

Every figure is followed by a **How to read this chart** block; checks are inline and labelled.

Every figure is followed by a **How to read this chart** block naming the data source, the
interpretation, and what observation would falsify it. Checks are inline and labelled (`N1.4.7[...]`),
so a failing assumption is visible in the output rather than buried.

---

## 01 — Research foundations

Derives why the result works, and states where the literature and the measurements disagree.

**§2 Recurrent over convolutional.** Why global average pooling over time destroys order, set against
the source's 229k-parameter CRNN versus its 250k-parameter convolutional baseline (~51% higher false
rejection) and our 291,564 versus 271,692 comparison.

**§3 Where our data contradicts the attribution.** The source attributes the recurrent advantage to
noise adaptation. Under `noise_10` our recurrent model is the worst of the three.

![gru minus wide by condition](../figures/07_gru_minus_wide_by_condition.png)

**§4 Neural collapse.** NC1 = `(1/C)·Tr(Σ_W Σ_B†)` and the NC2 simplex ETF, with the terminal-phase
precondition and the train-versus-held-out measurement deviation stated *before* any parallel is drawn.

![NC trajectories](../figures/05_nc_trajectories.png)

**§5 Adaptive gradient clipping.** AutoClip's percentile rule, plus two fidelity notes: our threshold
uses the prior history before appending, and `p=10` was tuned on speech *separation* with BLSTMs.

**§6 Calibration.** Temperature scaling as one-parameter MLE, ECE and its binning bias, then the MDCA
train-time objective — which drives the optimal temperature toward 1. Our best model's fitted
temperature is already 1.0157; our least stable needs 1.6002.

**§7 Salvaged derivations.** Proper scoring rules, ECE finite-sample bias, and the conformal estimator
with a Monte-Carlo check of its O(1/n) gap.

![CRC gap and exchangeability](../figures/06_crc_gap_and_exchangeability.png)

The left panel confirms the shortfall stays under the 1/(n+1) bound. The right panel shows what
happens when exchangeability is removed: miscoverage climbs from 0.10 to 0.39.

---

## 02 — Small-scale exploration

Synthetic data with class-specific *time-varying* signatures, so that a model which pools away the
time axis is structurally unable to solve the task. An executable `assert_varied()` guard fails any
figure whose plotted series is constant — the defect that made the retired notebooks worthless.

**§6 AutoClip percentile sweep**, `p ∈ {1, 10, 25, 100}` with `p=100` as the unclipped control.

![AutoClip percentile sweep](../figures/09_autoclip_percentile_sweep.png)

**§7 Train-time versus post-hoc calibration.** An MDCA-style auxiliary loss against temperature
scaling, and both together.

**§8 Weighted versus unweighted conformal risk control** under covariate shift and label shift.

![CRC under label shift](../figures/10_crc_label_shift.png)

**§9 How much calibration data a conformal threshold needs.** Bootstrap resampling at increasing
calibration sizes against the √n envelope — context for the real study's n = 1080.

![Coverage versus calibration size](../figures/08_coverage_vs_calibration_size.png)

---

## 04 — Real dataset walkthrough

The study. Real corpus, sha256-verified; real checkpoints reloaded and their accuracy re-derived.

**§2–3 Provenance and architecture comparison**, from sha256-inventoried artifacts.

**§4 Checkpoint verification.** Weights reloaded with `strict=True`, then evaluated on the real
validation split. Two of three arms reproduce their recorded accuracy to 0.0000; the third differs by
one clip in 504.

**§5 In-training diagnostics**, including the raw-versus-projected alert comparison.

![Cluster health in 2D](../figures/12_cluster_health_2d.png)

**§6 MM-PHATE statistics** over a 17,920-node activation tensor, with the PCA substitution labelled.

![MM-PHATE multiway embedding](../figures/11_mmphate_multiway_embedding.png)

**§7 Post-training results** — calibration, selective risk, conformal coverage, OOD detection.

### Offloaded cells

`04` tags its remote-only cells `offloaded`; a local run skips them rather than failing. Its inputs
live under `runtime/` (git-ignored) locally, or on Drive on the remote runtime — the notebook resolves
whichever is present and prints which one it used, so an executed copy shows on its face where its
numbers came from.
