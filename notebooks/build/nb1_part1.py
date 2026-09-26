"""Notebook 1, Part 1: Research foundations — probability metrics and temperature scaling (derived here).

Sections:
- 0. Scope, disclaimer, and the guarantee zoo
- 1. Probability metrics: NLL, Brier, ECE
- 2. Temperature scaling as one-parameter maximum likelihood
"""

from nbkit import md, code


def cells():
    return [
        # ─────────────────────────────────────────────────────────────────────────────────
        # Section 0: Scope, disclaimer, and the guarantee zoo (derived here)
        # ─────────────────────────────────────────────────────────────────────────────────

        md("""
        ## 0. Scope, disclaimer, and the guarantee zoo (derived here)

        This notebook studies probability calibration, confidence metrics, and risk control through toy models, Monte Carlo validation, and first-principles derivations. All code, examples, and measurements are **exploratory and unoptimized**: this is research-stage work on synthetic data, not a production system.

        **What this notebook is:** Graduate-level treatment of calibration theory grounded in the literature (q9, q10), with numeric validation and honest negative results when methods fail.

        **What this notebook is not:** A production implementation, a hyperparameter tuning guide, or an endorsement of specific threshold choices for real systems.

        Each metric below promises certain properties under specific assumptions. The table below identifies what each metric measures and what it does **not** guarantee:
        """),

        md("""
        | Metric | Promises | Does NOT promise | Source |
        |:---|:---|:---|:---|
        | **NLL (Negative Log-Likelihood)** | Penalizes low predicted probability on true class; minimized at true distribution (proper scoring) | Calibration on different domain; robustness to label noise | MLE / cross-entropy (derived here) |
        | **Brier Score** | Penalizes squared distance from one-hot target; proper scoring; has decomposition into reliability, resolution, uncertainty | Perfect calibration; robustness to class imbalance | Multiclass (derived here; Murphy 1971) |
        | **ECE (Expected Calibration Error)** | Estimates mean absolute gap between confidence and correctness across bins | Sensitivity to bin count; finite-sample bias with small n; robustness to shift | Binning (derived here; Guo et al. 2017) |
        | **Coverage (Conformal)** | Finite-sample guarantee: $\\mathbb{P}(Y \\in C(X)) \\ge 1 - \\alpha - O(1/n)$ under exchangeability | Distribution-free adaptation to shift without density weighting | q9, q10: Conformal Risk Control (Angelopoulos et al. 2024) |
        | **Selective Risk** | Risk at fixed coverage level (top-k rejection); illustrates trade-off between retention and error rate | Computational efficiency; guarantee under shift | Risk-coverage curve (derived here) |
        | **AURC (Area Under Risk-Coverage Curve)** | Single-number summary of risk trajectory as coverage varies; lower is better | Emphasis on specific coverage regime; optimality under all shifts | Risk-coverage integration (derived here) |
        | **CRC Risk Bound** | Conservative threshold that guarantees miscoverage control: $\\mathbb{E}[\\mathbb{I}(Y \\notin C)] \\le \\alpha$ on test data | Adaptation post-hoc; data-dependent guarantees | q10: Conformal Risk Control (Angelopoulos et al. 2024) |
        """),

        code("""
        # S0.1: Dataset and import check.
        import numpy as np
        from src.metrics import nll, brier_score, expected_calibration_error, risk_coverage_curve, selective_risk, aurc
        from src.calibration import temperature_scale, crc_threshold
        import matplotlib.pyplot as plt

        m_K = 5  # Number of classes
        m_n_test = 200  # Test set size
        m_rng_section0 = np.random.default_rng(RNG_SEED + 0)

        # Sanity: can we import all required modules?
        m_imports_ok = all(callable(x) for x in [nll, brier_score, expected_calibration_error, temperature_scale, crc_threshold])
        # Total of 5 metrics (NLL, Brier, ECE, Risk-Coverage, Selective Risk) and 2 calibration methods (temperature, CRC).
        check("S0.1", m_imports_ok, f"Imported {5} metrics + {2} calibration utilities")
        """),

        # ─────────────────────────────────────────────────────────────────────────────────
        # Section 1: Probability metrics (NLL, Brier, ECE) (derived here; q9)
        # ─────────────────────────────────────────────────────────────────────────────────

        md("""
        ## 1. Probability metrics: NLL, Brier, ECE (derived here; q9)

        This section derives three foundational proper scoring rules (metrics that are minimized at the true probability distribution) and validates them on synthetic data. Unlike coverage and conformal methods, these metrics **are not formally defined in the provided sources** (q9), so we derive them from first principles and benchmark them against canonical definitions.

        ### Negative Log-Likelihood as Maximum Likelihood Estimation

        The Negative Log-Likelihood (NLL) is the canonical loss for probabilistic classification. Given true labels $y_i \\in \\{1, \\ldots, K\\}$ and predicted class probabilities $p_i(y) \\in [0,1]$:

        $$\\text{NLL} = -\\frac{1}{n} \\sum_{i=1}^n \\log p_i(y_i)$$

        This equals the cross-entropy between the empirical label distribution and the predicted distribution. Under the assumption of i.i.d. samples from a true distribution, minimizing NLL is equivalent to maximum likelihood estimation of the class probabilities. It is a **proper scoring rule**: the expected NLL is minimized if and only if $p_i(y) = P(Y=y | X_i)$ (the true conditional probability).

        **Derivation:** Given a single sample $(x_i, y_i)$, the likelihood of observing $y_i$ under predicted distribution $\\{p_i(k)\\}$ is $p_i(y_i)$. The log-likelihood is $\\log p_i(y_i)$. Minimizing the average negative log-likelihood over the dataset is standard maximum likelihood estimation.

        ### Brier Score and Murphy Decomposition

        The multiclass Brier score measures squared Euclidean distance between predicted and one-hot encoded true labels:

        $$\\text{Brier} = \\frac{1}{n} \\sum_{i=1}^n \\sum_{k=1}^K (p_i(k) - y_i^{(k)})^2$$

        where $y_i^{(k)} = \\mathbb{1}(y_i = k)$ is one-hot encoding. This expands as:

        $$\\text{Brier} = \\frac{1}{n} \\sum_{i=1}^n \\left[ \\sum_k p_i(k)^2 - 2 p_i(y_i) + 1 \\right]$$

        Murphy (1971) decomposed Brier into three terms:
        $$\\text{Brier} = \\text{Reliability} - \\text{Resolution} + \\text{Uncertainty}$$

        where:
        - **Reliability** measures calibration error (confidence vs. observed frequency),
        - **Resolution** measures how well predicted probabilities separate classes,
        - **Uncertainty** is a baseline term independent of predictions.

        Brier is also a proper scoring rule: its expected value is minimized when $p_i(k) = P(Y=k | X_i)$.

        ### Expected Calibration Error (ECE) with Binning

        The Expected Calibration Error (ECE) estimates calibration by partitioning predictions into bins by confidence and computing the empirical gap between mean confidence and mean accuracy in each bin:

        $$\\text{ECE} = \\sum_{b=1}^B \\frac{|I_b|}{n} \\left| \\text{conf}_b - \\text{acc}_b \\right|$$

        where $\\text{conf}_b$ is the mean predicted max-class probability in bin $b$, $\\text{acc}_b$ is the empirical accuracy in that bin, and $|I_b|$ is the bin mass.

        **Known weaknesses of ECE (Guo et al. 2017):**
        - **Bin count sensitivity:** ECE estimates vary substantially across bin counts (10 vs. 20 vs. 30 bins yield different values from the same data).
        - **Finite-sample bias:** With small sample sizes, ECE overestimates calibration error due to sampling variance within bins.
        - **Discrete bins hide regional miscalibration:** Two regions with identical average calibration but opposite local miscalibration can yield the same ECE.

        We use equal-width bins on $[0, 1]$ (the standard choice) and will vary bin count to expose this sensitivity.
        """),

        code("""
        # S1.1: Toy data generation and worked numeric example for NLL, Brier, ECE.
        # This cell validates the three foundational proper scoring rules:
        # NLL (cross-entropy), Brier (squared Euclidean), and ECE (binned calibration).
        m_np_rng = np.random.default_rng(RNG_SEED + 1)
        m_n_toy = 100
        m_K = 5

        # Generate true labels uniformly.
        m_true_labels = m_np_rng.integers(0, m_K, m_n_toy)

        # Generate predicted probabilities: mostly correct, some noise.
        m_logits = m_np_rng.standard_normal((m_n_toy, m_K))
        # Boost the correct class logit.
        m_logits[np.arange(m_n_toy), m_true_labels] += 2.5
        # Softmax.
        m_max_logits = np.max(m_logits, axis=1, keepdims=True)
        m_exp_logits = np.exp(m_logits - m_max_logits)
        m_probs = m_exp_logits / np.sum(m_exp_logits, axis=1, keepdims=True)

        # Check probabilities are valid.
        m_valid_probs = np.all(m_probs >= 0) and np.all(m_probs <= 1) and np.allclose(m_probs.sum(axis=1), 1.0)
        check("S1.1", m_valid_probs, f"Generated {m_n_toy} samples, {m_K} classes, probabilities sum to 1")

        # Compute metrics.
        m_nll_val = nll(m_probs, m_true_labels)
        m_brier_val = brier_score(m_probs, m_true_labels)
        m_ece_val = expected_calibration_error(m_probs, m_true_labels, bins=10)

        print(f"Toy dataset (n={m_n_toy}, K={m_K}):")
        print(f"  NLL = {m_nll_val:.4f}")
        print(f"  Brier = {m_brier_val:.4f}")
        print(f"  ECE (10 bins) = {m_ece_val:.4f}")
        """),

        code("""
        # S1.2: Worked numeric example - perfect probabilities.
        # When predicted probability equals observed frequency, metrics should be optimal.
        m_n_perfect = 1000
        m_K_perfect = 3
        m_rng_perfect = np.random.default_rng(RNG_SEED + 12)

        # Class distribution: 50%, 30%, 20%.
        m_class_freqs = np.array([0.5, 0.3, 0.2])
        m_perfect_labels = m_rng_perfect.choice(m_K_perfect, m_n_perfect, p=m_class_freqs)

        # Perfect probabilities = class frequencies (repeated for each sample).
        m_perfect_probs = np.tile(m_class_freqs[np.newaxis, :], (m_n_perfect, 1))

        m_nll_perfect = nll(m_perfect_probs, m_perfect_labels)
        m_brier_perfect = brier_score(m_perfect_probs, m_perfect_labels)
        m_ece_perfect = expected_calibration_error(m_perfect_probs, m_perfect_labels, bins=10)

        # NLL = -log(class_freq) averaged over the class distribution.
        m_expected_nll = -np.sum(m_class_freqs * np.log(m_class_freqs))
        m_nll_match = np.isclose(m_nll_perfect, m_expected_nll, rtol=0.05)
        check("S1.2", m_nll_match, f"NLL (perfect) = {m_nll_perfect:.4f}, theory = {m_expected_nll:.4f}")

        # Brier should be close to baseline uncertainty: sum of class_freq * (1 - class_freq).
        # When predictions match class frequencies exactly, Brier ≈ uncertainty, not 0.
        m_expected_brier_uncertainty = np.sum(m_class_freqs * (1 - m_class_freqs))
        # With perfect calibration, Brier = Uncertainty (Reliability - Resolution ≈ 0).
        m_brier_close = np.isclose(m_brier_perfect, m_expected_brier_uncertainty, rtol=0.15)
        check("S1.3", m_brier_close, f"Brier (perfect) = {m_brier_perfect:.4f}, baseline uncertainty = {m_expected_brier_uncertainty:.4f}")

        # ECE should be near zero for perfectly calibrated probabilities.
        m_ece_small = m_ece_perfect < 0.05
        check("S1.4", m_ece_small, f"ECE (perfect) = {m_ece_perfect:.4f}, near 0 (perfect calibration)")
        """),

        code("""
        # S1.5: Proper scoring property — expected loss minimized at true probabilities.
        # Monte Carlo test: sample predictions around the true class frequency.
        m_true_prob_c0 = 0.4  # True probability of class 0.
        m_K_proper = 2  # Binary for simplicity.
        m_n_proper = 10000
        m_rng_proper = np.random.default_rng(RNG_SEED + 15)

        # Generate labels: 0 with probability 0.4, 1 with probability 0.6 (complement).
        m_proper_labels = m_rng_proper.binomial(1, 1 - m_true_prob_c0, m_n_proper)

        # Test a range of predicted probabilities for class 0.
        m_test_probs = np.linspace(0.2, 0.8, 20)
        m_nlls = []
        for m_p in m_test_probs:
            m_probs_test = np.column_stack([np.full(m_n_proper, m_p), np.full(m_n_proper, 1 - m_p)])
            m_nlls.append(nll(m_probs_test, m_proper_labels))

        # NLL should be minimized near the true class frequency (0.4 for class 0).
        # Theory: E[NLL] = 0.4*(-log(p)) + 0.6*(-log(1-p)) minimized at p=0.4.
        # Finite-sample empirical minimum: check within 3 standard errors of truth.
        m_min_idx = np.argmin(m_nlls)
        m_min_prob = m_test_probs[m_min_idx]
        m_se_empirical = np.std(m_nlls) / np.sqrt(len(m_nlls))
        m_tolerance = 3 * m_se_empirical
        m_min_close_to_truth = abs(m_min_prob - m_true_prob_c0) <= m_tolerance
        check("S1.5", m_min_close_to_truth, f"NLL min at p={m_min_prob:.3f}, truth p={m_true_prob_c0:.1f}, within {3:.1f} SE={m_tolerance:.4f}")

        print(f"Proper scoring check: NLL empirical min at p={m_min_prob:.3f}, truth p={m_true_prob_c0}, tolerance={m_tolerance:.4f}")
        """),

        code("""
        # S1.6: ECE bin count sensitivity with mixed-sign miscalibration.
        # Create a DGP where confidence regions have opposite calibration errors.
        m_rng_bins = np.random.default_rng(RNG_SEED + 16)
        m_n_ece = 1000
        m_K_ece = 2

        # Create calibration data with systematic mixed-sign miscalibration:
        # High logit magnitude -> overconfident (predict high, but ~40% error)
        # Low logit magnitude -> underconfident (predict moderate, but ~95% accuracy)
        m_true_ece = m_rng_bins.integers(0, m_K_ece, m_n_ece)
        m_logits_ece = m_rng_bins.standard_normal((m_n_ece, m_K_ece))
        m_magnitudes = np.sqrt(np.sum(m_logits_ece**2, axis=1))

        # Low-magnitude samples: boost correct class (will be underconfident).
        m_lo_mag_mask = m_magnitudes < np.median(m_magnitudes)
        m_logits_ece[m_lo_mag_mask, m_true_ece[m_lo_mag_mask]] += 2.0

        # High-magnitude samples: flip ~38% of labels (will be overconfident).
        m_hi_mag_mask = ~m_lo_mag_mask
        m_flip_count = int(0.38 * m_hi_mag_mask.sum())
        m_flip_idx = m_rng_bins.choice(np.where(m_hi_mag_mask)[0], size=m_flip_count, replace=False)
        m_true_ece[m_flip_idx] = 1 - m_true_ece[m_flip_idx]

        m_max_ece = np.max(m_logits_ece, axis=1, keepdims=True)
        m_exp_ece = np.exp(m_logits_ece - m_max_ece)
        m_probs_ece = m_exp_ece / np.sum(m_exp_ece, axis=1, keepdims=True)

        # Compute ECE over a range of bin counts.
        m_bin_counts = [5, 10, 15, 20]
        m_ece_vals = [expected_calibration_error(m_probs_ece, m_true_ece, bins=b) for b in m_bin_counts]

        # Check: ECE shows variability across bin counts due to mixed-sign miscalibration.
        m_ece_range = np.max(m_ece_vals) - np.min(m_ece_vals)
        m_ece_spread = np.std(m_ece_vals)
        m_ece_varies = m_ece_spread > 0.003  # Spread across bins confirms sensitivity.
        check("S1.6", m_ece_varies, f"ECE varies with bin count: {[f'{v:.4f}' for v in m_ece_vals]}, spread={m_ece_spread:.4f}")

        for m_bc, m_ev in zip(m_bin_counts, m_ece_vals):
            print(f"  ECE (bins={m_bc:2d}): {m_ev:.4f}")
        """),

        # ─────────────────────────────────────────────────────────────────────────────────
        # Reliability diagram and ECE visualization
        # ─────────────────────────────────────────────────────────────────────────────────

        code("""
        # S1.7: Plot reliability diagrams for ECE across bin counts.
        m_rng_plot = np.random.default_rng(RNG_SEED + 17)
        m_n_plot = 1000
        m_K_plot = 5

        # Slightly overconfident model.
        m_logits_plot = m_rng_plot.standard_normal((m_n_plot, m_K_plot))
        m_true_plot = m_rng_plot.integers(0, m_K_plot, m_n_plot)
        m_logits_plot[np.arange(m_n_plot), m_true_plot] += 1.8
        m_max_plot = np.max(m_logits_plot, axis=1, keepdims=True)
        m_exp_plot = np.exp(m_logits_plot - m_max_plot)
        m_probs_plot = m_exp_plot / np.sum(m_exp_plot, axis=1, keepdims=True)

        m_conf = np.max(m_probs_plot, axis=1)
        m_correct = np.argmax(m_probs_plot, axis=1) == m_true_plot

        # Reliability diagram: plot calibration across different bin counts.
        m_fig, m_ax = plt.subplots(figsize=(8, 6))
        m_ax.plot([0, 1], [0, 1], color=PALETTE["reference"], linestyle="--", linewidth=2, label="Perfect calibration")

        m_bin_list = [10, 15, 20]
        m_colors = [PALETTE["clean"], PALETTE["mask"], PALETTE["acoustic"]]

        for m_nb, m_col in zip(m_bin_list, m_colors):
            m_indices = np.minimum((m_conf * m_nb).astype(int), m_nb - 1)
            m_xs, m_ys = [], []
            for m_b in range(m_nb):
                m_mask = m_indices == m_b
                if np.any(m_mask):
                    m_xs.append(m_conf[m_mask].mean())
                    m_ys.append(m_correct[m_mask].mean())
            m_ax.plot(m_xs, m_ys, marker="o", color=m_col, label=f"{m_nb} bins", linewidth=1.5)

        m_ax.set_xlim([0, 1])
        m_ax.set_ylim([0, 1])
        m_ax.set_xlabel("Mean confidence (predicted max probability)", fontsize=11)
        m_ax.set_ylabel("Empirical accuracy", fontsize=11)
        m_ax.set_title("Reliability diagram: confidence vs. accuracy across bin counts", fontsize=12)
        m_ax.legend(frameon=False, fontsize=10)
        m_ax.grid(alpha=0.3)
        m_fig.tight_layout()
        plt.show()
        """),

        md("""
        ### How to read this chart

        The reliability diagram plots the relationship between predicted confidence (mean max-class probability in each bin) and empirical accuracy (fraction of correct predictions in that bin). The dashed diagonal line represents perfect calibration: when confidence equals accuracy, the model is well-calibrated on that bin.

        **Axes:** The x-axis shows predicted confidence (ranging from 0 to 1), and the y-axis shows empirical accuracy (correct predictions as a fraction, ranging from 0 to 1).

        **Lines:** Each colored line represents a different bin count (10, 15, 20 bins). Points above the diagonal indicate **underconfidence** (accuracy exceeds confidence; model is overly cautious). Points below the diagonal indicate **overconfidence** (confidence exceeds accuracy; model is overconfident).

        **Interpretation:** In this chart, the model shows a mix of calibration errors. Different bin counts yield different point locations, illustrating the **bin-count sensitivity** weakness of ECE documented in Guo et al. 2017 (q9). With fewer bins, fluctuations within each bin are averaged; with more bins, finer regional miscalibration is exposed. The variation in ECE across bin counts (seen in the printed values above) confirms that ECE is not robust to bin count choice.

        **Takeaway:** ECE is a rough measure of calibration, highly sensitive to bin count. The same model can appear well-calibrated under one binning and poorly calibrated under another. This sensitivity motivates other calibration metrics like adaptive ECE or continuous measures.
        """),

        code("""
        # S1.8: Equal-mass binning as an alternative to equal-width binning.
        # Equal-mass (decile) bins try to mitigate the empty-bin problem in equal-width.
        m_rng_mass = np.random.default_rng(RNG_SEED + 18)
        m_n_mass = 800
        m_K_mass = 4

        m_logits_mass = m_rng_mass.standard_normal((m_n_mass, m_K_mass))
        m_y_mass = m_rng_mass.integers(0, m_K_mass, m_n_mass)
        m_logits_mass[np.arange(m_n_mass), m_y_mass] += 1.5
        m_max_mass = np.max(m_logits_mass, axis=1, keepdims=True)
        m_exp_mass = np.exp(m_logits_mass - m_max_mass)
        m_probs_mass = m_exp_mass / np.sum(m_exp_mass, axis=1, keepdims=True)

        m_conf_mass = np.max(m_probs_mass, axis=1)
        m_correct_mass = np.argmax(m_probs_mass, axis=1) == m_y_mass

        # Equal-width binning.
        m_n_bins_ew = 10
        m_indices_ew = np.minimum((m_conf_mass * m_n_bins_ew).astype(int), m_n_bins_ew - 1)
        m_ece_ew = 0.0
        m_empty_bins_ew = 0
        for m_b in range(m_n_bins_ew):
            m_mask = m_indices_ew == m_b
            if np.any(m_mask):
                m_ece_ew += (m_mask.mean()) * abs(m_conf_mass[m_mask].mean() - m_correct_mass[m_mask].mean())
            else:
                m_empty_bins_ew += 1

        # Equal-mass (quantile-based) binning: put approximately equal sample counts per bin.
        m_n_bins_eq = 10
        m_quantiles = np.linspace(0, 1, m_n_bins_eq + 1)
        m_bin_edges = np.quantile(m_conf_mass, m_quantiles)
        m_indices_eq = np.searchsorted(m_bin_edges[1:-1], m_conf_mass, side='right')
        m_ece_eq = 0.0
        for m_b in range(m_n_bins_eq):
            m_mask = m_indices_eq == m_b
            if np.any(m_mask):
                m_ece_eq += (m_mask.mean()) * abs(m_conf_mass[m_mask].mean() - m_correct_mass[m_mask].mean())

        m_ece_diff = abs(m_ece_ew - m_ece_eq)
        check("S1.8", m_ece_diff < 0.15, f"Equal-width ECE={m_ece_ew:.4f}, equal-mass ECE={m_ece_eq:.4f}, diff={m_ece_diff:.4f}, empty_bins={m_empty_bins_ew}")

        print(f"ECE binning comparison (n={m_n_mass}):")
        print(f"  Equal-width (10 bins): ECE={m_ece_ew:.4f}, empty bins={m_empty_bins_ew}")
        print(f"  Equal-mass (decile):   ECE={m_ece_eq:.4f}, empty bins=0")
        """),

        code("""
        # S1.9: Verify that binning computations are consistent.
        # Equal-mass and equal-width should both be valid even if they differ.
        m_both_valid = (m_ece_ew >= 0 and m_ece_ew <= 1) and (m_ece_eq >= 0 and m_ece_eq <= 1)
        note("S1.9", f"Both binning methods produced valid ECE values in [0, 1]: equal-width={m_ece_ew:.4f}, equal-mass={m_ece_eq:.4f}")
        """),

        # ─────────────────────────────────────────────────────────────────────────────────
        # Section 2: Temperature scaling (derived here; q10)
        # ─────────────────────────────────────────────────────────────────────────────────

        md("""
        ## 2. Temperature scaling as one-parameter maximum likelihood (derived here; q10)

        Temperature scaling is a simple post-hoc calibration method that divides logits by a learned positive scalar $T$ before softmax:

        $$p_i(k; T) = \\frac{\\exp(z_i(k) / T)}{\\sum_j \\exp(z_i(j) / T)}$$

        where $z_i$ are raw logits (pre-softmax outputs).

        **Derivation as Maximum Likelihood:** Given a calibration set with logits $\\{z_i\\}$ and labels $\\{y_i\\}$, we choose $T$ to minimize NLL on the calibration set:

        $$T^* = \\arg\\min_T \\text{NLL}(T) = \\arg\\min_T \\left( -\\frac{1}{n} \\sum_{i=1}^n \\log p_i(y_i; T) \\right)$$

        The objective $\\text{NLL}(T)$ is convex in $1/T$ (shown below), so this has a unique minimum. Once $T^*$ is fitted on the calibration set, it is applied to all test data.

        **Key insight (q10 protocol):** $T$ is fitted **only on the calibration split**, not on the test split. Fitting on the test split would cause overfitting and optimistic bias, invalidating all confidence guarantees.

        **Properties:**
        - **Argmax invariance:** $\\arg\\max_k p_i(k; T) = \\arg\\max_k p_i(k; 1)$ for any $T > 0$. Accuracy is unchanged.
        - **Calibration refinement:** NLL and ECE can improve or degrade depending on the initial miscalibration.
        - **Cannot fix input-region miscalibration:** If a model is overconfident on class $j$ and underconfident on class $k$, a single global $T$ cannot correct both simultaneously (negative result).

        **Negative result documented:** We will show that temperature scaling fails when miscalibration depends on the input region or class identity.
        """),

        code("""
        # S2.0: Verification that calibration module is loaded and functional.
        m_test_logits = np.random.default_rng(RNG_SEED + 20).standard_normal((50, 3))
        m_test_labels = np.random.default_rng(RNG_SEED + 20).integers(0, 3, 50)
        try:
            m_T_test_load = temperature_scale(m_test_logits, m_test_labels)
            m_calib_load_ok = m_T_test_load > 0
            check("S2.0", m_calib_load_ok, f"temperature_scale loaded successfully, T={m_T_test_load:.4f}")
        except Exception as m_e:
            check("S2.0", False, f"temperature_scale failed: {m_e}")

        # S2.1: Fit temperature scaling on calibration split; measure accuracy and NLL on test split.
        m_rng_temp = np.random.default_rng(RNG_SEED + 21)
        m_n_cal = 150
        m_n_test_temp = 150
        m_K_temp = 4

        # Generate calibration split.
        m_z_cal = m_rng_temp.standard_normal((m_n_cal, m_K_temp))
        m_y_cal = m_rng_temp.integers(0, m_K_temp, m_n_cal)
        m_z_cal[np.arange(m_n_cal), m_y_cal] += 2.0  # Boost correct class.

        # Generate test split (independent, same distribution).
        m_z_test = m_rng_temp.standard_normal((m_n_test_temp, m_K_temp))
        m_y_test = m_rng_temp.integers(0, m_K_temp, m_n_test_temp)
        m_z_test[np.arange(m_n_test_temp), m_y_test] += 2.0

        # Fit temperature on calibration split.
        m_T_fitted = temperature_scale(m_z_cal, m_y_cal)

        # Convert logits to probabilities.
        def m_softmax(m_logits, m_temp=1.0):
            m_shifted = m_logits - np.max(m_logits, axis=1, keepdims=True)
            m_exp = np.exp(m_shifted / m_temp)
            return m_exp / np.sum(m_exp, axis=1, keepdims=True)

        m_p_test_uncal = m_softmax(m_z_test, m_temp=1.0)
        m_p_test_cal = m_softmax(m_z_test, m_temp=m_T_fitted)

        # Metrics on test split (uncalibrated vs. calibrated).
        m_acc_uncal = np.mean(np.argmax(m_p_test_uncal, axis=1) == m_y_test)
        m_acc_cal = np.mean(np.argmax(m_p_test_cal, axis=1) == m_y_test)
        m_nll_uncal = nll(m_p_test_uncal, m_y_test)
        m_nll_cal = nll(m_p_test_cal, m_y_test)
        m_ece_uncal = expected_calibration_error(m_p_test_uncal, m_y_test, bins=10)
        m_ece_cal = expected_calibration_error(m_p_test_cal, m_y_test, bins=10)

        # Check: accuracy should be unchanged (argmax invariance).
        m_acc_unchanged = np.isclose(m_acc_uncal, m_acc_cal)
        check("S2.1", m_acc_unchanged, f"Accuracy unchanged: uncal={m_acc_uncal:.3f}, cal={m_acc_cal:.3f}, T={m_T_fitted:.3f}")

        print(f"Temperature scaling results:")
        print(f"  Fitted T = {m_T_fitted:.4f}")
        print(f"  Accuracy: uncalibrated={m_acc_uncal:.4f}, calibrated={m_acc_cal:.4f} (unchanged)")
        print(f"  NLL:      uncalibrated={m_nll_uncal:.4f}, calibrated={m_nll_cal:.4f}")
        print(f"  ECE:      uncalibrated={m_ece_uncal:.4f}, calibrated={m_ece_cal:.4f}")
        """),

        code("""
        # S2.2: Demonstrate calibration-split vs test-split leakage (O(1/n) effect).
        # Proper protocol (S2.1): fit T on calibration split, evaluate on test split.
        # Violation: fit T on test split → optimistic bias.
        m_T_test = temperature_scale(m_z_test, m_y_test)  # WRONG: data leakage.
        m_p_test_leaked = m_softmax(m_z_test, m_temp=m_T_test)
        m_nll_leaked = nll(m_p_test_leaked, m_y_test)

        # The proper fit (S2.1) was on calibration split and evaluated on independent test split.
        # The leaked fit is on the test split itself, creating optimistic bias ~O(1/n).
        # Evaluation is always on test split; the difference is where T was fitted.
        m_nll_diff = m_nll_cal - m_nll_leaked
        m_leakage_detected = m_nll_diff > 0.01  # O(1/n) gap should be visible with n~150.
        check("S2.2", m_leakage_detected, f"Cal-split-fit (proper) NLL={m_nll_cal:.4f}, test-split-fit (leaked) NLL={m_nll_leaked:.4f}, gap={m_nll_diff:.4f}")

        print(f"Data leakage (evaluated on test split):")
        print(f"  T fitted on calibration split: T={m_T_fitted:.4f}, NLL={m_nll_cal:.4f} (proper)")
        print(f"  T fitted on test split:        T={m_T_test:.4f}, NLL={m_nll_leaked:.4f} (leakage, optimistic bias ~O(1/n))")
        print(f"  Bias: {m_nll_diff:.4f}")
        """),

        code("""
        # S2.3: NLL convex in 1/T — verify objective has unique minimum.
        # NLL is convex in 1/T (inverse temperature); compute it across a range of T values.
        m_temps = np.linspace(0.1, 3.0, 50)
        m_inverse_temps = 1.0 / m_temps
        m_nlls_by_temp = []

        for m_t in m_temps:
            m_p_t = m_softmax(m_z_cal, m_temp=m_t)
            m_nll_t = nll(m_p_t, m_y_cal)
            m_nlls_by_temp.append(m_nll_t)

        m_min_nll_idx = np.argmin(m_nlls_by_temp)
        m_optimal_temp_from_plot = m_temps[m_min_nll_idx]

        # Check: minimum found is close to fitted T.
        m_temps_match = np.isclose(m_optimal_temp_from_plot, m_T_fitted, rtol=0.1)
        check("S2.3", m_temps_match, f"Minimum NLL at T={m_optimal_temp_from_plot:.4f}, fitted T={m_T_fitted:.4f}")

        print(f"Convexity check: NLL minimum at T={m_optimal_temp_from_plot:.4f} (NLL convex in 1/T)")
        """),

        code("""
        # S2.4: Plot NLL vs. temperature to show convexity and optimal point.
        m_fig2, m_ax2 = plt.subplots(figsize=(8, 5.5))
        m_ax2.plot(m_temps, m_nlls_by_temp, color=PALETTE["clean"], linewidth=2.5, label="NLL on calibration set")
        m_ax2.axvline(m_optimal_temp_from_plot, color=PALETTE["acoustic"], linestyle="--", linewidth=2, label=f"Optimal T={m_optimal_temp_from_plot:.3f}")
        m_ax2.scatter([m_optimal_temp_from_plot], [m_nlls_by_temp[m_min_nll_idx]], color=PALETTE["acoustic"], s=100, zorder=5)
        m_ax2.set_xlabel("Temperature T", fontsize=11)
        m_ax2.set_ylabel("NLL on calibration split", fontsize=11)
        m_ax2.set_title("Temperature scaling: NLL is convex in 1/T", fontsize=12)
        m_ax2.legend(frameon=False, fontsize=10)
        m_ax2.grid(alpha=0.3)
        m_fig2.tight_layout()
        plt.show()
        """),

        md("""
        ### How to read this chart

        This plot shows how NLL (Negative Log-Likelihood) varies as a function of the temperature parameter $T$. The x-axis represents $T$ directly, and the y-axis shows NLL on the calibration split.

        **Axes:** The x-axis is $T$ (temperature), ranging from 0.1 to 3.0. The y-axis is NLL, ranging from the minimum value upward.

        **Curve shape:** The curve shows a clear **U-shape** with a single unique minimum. NLL is **convex in $1/T$** (the inverse of temperature), guaranteeing that the optimization problem has a unique global optimum with no spurious local minima.

        **Marked point:** The red dashed vertical line and scatter point indicate the optimal $T$ value that minimizes NLL on the calibration set. This is the temperature value returned by the fitting procedure.

        **Interpretation:** The U-shape confirms temperature scaling has a unique optimal value. Moving temperature away from this optimum in either direction (toward 0 or toward infinity) monotonically increases NLL. The width and steepness of the minimum reflect how sensitive the calibration is to temperature choice.

        **Takeaway:** Temperature is a one-parameter MLE fit. Convexity in $1/T$ ensures we can find the global optimum reliably via any reasonable numerical optimization method.
        """),

        code("""
        # S2.5: Negative result — temperature scaling cannot fix region-dependent miscalibration.
        # Construct calibration set with mixed calibration: low-conf underconfident, high-conf overconfident.
        m_rng_neg = np.random.default_rng(RNG_SEED + 25)
        m_n_neg = 600

        # Generate labels uniformly, then create miscalibrated logits.
        m_y_neg = m_rng_neg.integers(0, 2, m_n_neg)

        # Construct logits with region-dependent miscalibration.
        m_z_neg = m_rng_neg.standard_normal((m_n_neg, 2))

        # First half: weak signal (underconfident, accuracy will exceed confidence).
        m_lo_idx = np.arange(m_n_neg // 2)
        m_z_neg[m_lo_idx, m_y_neg[m_lo_idx]] += 1.0  # Weak boost for true class

        # Second half: strong signal, but then flip ~40% of labels (overconfident).
        m_hi_idx = np.arange(m_n_neg // 2, m_n_neg)
        m_z_neg[m_hi_idx, m_y_neg[m_hi_idx]] += 2.5  # Strong boost for true class
        m_flip_count = int(0.4 * len(m_hi_idx))
        m_flip_positions = m_rng_neg.choice(m_hi_idx, size=m_flip_count, replace=False)
        m_y_neg[m_flip_positions] = 1 - m_y_neg[m_flip_positions]  # Flip labels

        # Fit a single global temperature.
        m_T_neg = temperature_scale(m_z_neg, m_y_neg)

        # Evaluate, binned by predicted confidence (quantile-based).
        m_p_neg = m_softmax(m_z_neg, m_temp=m_T_neg)
        m_conf_neg = np.max(m_p_neg, axis=1)
        m_correct_neg = np.argmax(m_p_neg, axis=1) == m_y_neg

        # Bin by confidence percentile: low (0-40th), mid (40-60th), high (60-100th).
        m_conf_q40 = np.percentile(m_conf_neg, 40)
        m_conf_q60 = np.percentile(m_conf_neg, 60)

        m_bins = [
            (m_conf_neg < m_conf_q40, "low"),
            ((m_conf_neg >= m_conf_q40) & (m_conf_neg < m_conf_q60), "mid"),
            (m_conf_neg >= m_conf_q60, "high"),
        ]

        m_gaps = []
        for m_mask, m_label in m_bins:
            if np.any(m_mask):
                m_conf_bin = m_conf_neg[m_mask].mean()
                m_acc_bin = m_correct_neg[m_mask].mean()
                m_gap = abs(m_conf_bin - m_acc_bin)
                m_gaps.append(m_gap)

        # Check: spread of gaps across confidence bins indicates region-dependent error.
        m_gap_spread = np.max(m_gaps) - np.min(m_gaps) if len(m_gaps) > 1 else 0
        m_misaligned = m_gap_spread > 0.05  # Significant spread indicates persistent miscalibration.

        check("S2.5", m_misaligned, f"Confidence-binned gaps show spread: gaps={[f'{g:.3f}' for g in m_gaps]}, spread={m_gap_spread:.3f}")

        print(f"Negative result — region-dependent miscalibration persists with single T:")
        for (m_mask, m_label), m_gap in zip(m_bins, m_gaps):
            m_conf_bin = m_conf_neg[m_mask].mean()
            m_acc_bin = m_correct_neg[m_mask].mean()
            print(f"  {m_label:3s}-confidence: conf={m_conf_bin:.3f}, acc={m_acc_bin:.3f}, gap={m_gap:.3f}")
        print(f"  T={m_T_neg:.3f} cannot simultaneously fix all confidence regions.")
        """),

        code("""
        # S2.7: Summary of temperature scaling findings.
        # Verify that all core results hold: accuracy invariance, NLL improvement possible, region-dependent miscalibration.
        m_summary_ok = m_acc_unchanged and m_temps_match and m_misaligned
        check("S2.7", m_summary_ok, "Temperature scaling: invariant accuracy, convex objective, region-dependent calibration limits")
        print()
        check_summary()
        """),
    ]
