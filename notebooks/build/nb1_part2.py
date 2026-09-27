"""Notebook 1, Part 2: Selective classification and score function scale sensitivity.

Sections:
- 3. Selective classification: selector, coverage, selection risk (source: Liang et al., TMLR 2024; L)
- 4. Score functions and scale sensitivity (source: L Sec 3.1-3.2; derived here for the toy)
- 4b. Selection under covariate shift and novel-class inputs (derived here)
- Conclusion: what the paper does NOT settle
"""

from nbkit import md, code


def cells():
    return [
        # ─────────────────────────────────────────────────────────────────────────────────
        # Section 3: Selective classification (source: Liang et al., TMLR 2024; L)
        # ─────────────────────────────────────────────────────────────────────────────────

        md("""
        ## 3. Selective classification: selector, coverage, selection risk (source: Liang et al., TMLR 2024; L)

        Selective classification extends standard prediction by allowing a model to abstain (reject) on uncertain examples. The framework defines two components:

        **Paraphrase of Liang et al. Eq. (1); L p.3.** A selective classifier is a pair $(f, g)$ where $f$ is the **predictor** (e.g., argmax of softmax probabilities) and $g: \\mathcal{X} \\to \\{0,1\\}$ is the **selector** (abstention rule). On input $x$, the system predicts $f(x)$ if $g(x) = 1$ and abstains otherwise.

        The selector is induced by a **score function** $s: \\mathcal{X} \\to \\mathbb{R}$ and a threshold $\\gamma$:

        $$g_{s,\\gamma}(x) = \\mathbb{1}[s(x) > \\gamma] \\quad \\text{(Eq. 2; L p.3)}$$

        The quality of selective classification is measured by two key metrics defined on the retained subset:

        $$\\varphi = \\mathbb{E}[g(x)] \\quad \\text{(coverage: fraction of samples where } g(x)=1\\text{)}$$

        $$R = \\frac{\\mathbb{E}[\\ell(f(x), y) \\cdot g(x)]}{\\varphi} \\quad \\text{(selection risk: mean 0/1 loss on retained samples; Eq. 3; L p.3)}$$

        where $\\ell(f(x), y) = \\mathbb{1}[f(x) \\ne y]$ is the standard 0/1 classification loss.

        Under a distribution shift, both $\\varphi$ and $R$ change. The paper formalizes this:

        **Paraphrase of Liang et al. Eq. (8); L p.4.** Under a shifted distribution $D'$, coverage and selection risk are re-defined with expectations taken under $D'$; the paper states this generalization assumes no outliers.

        $$\\varphi' = \\mathbb{E}_{D'}[g(x)], \\quad R' = \\frac{\\mathbb{E}_{D'}[\\ell(f(x), y) \\cdot g(x)]}{\\varphi'} \\quad \\text{(Eq. 8; L p.4)}$$

        **What the risk-coverage trade-off reveals:** The risk-coverage (RC) curve plots coverage $\\varphi$ on the x-axis and selection risk $R$ on the y-axis as the threshold $\\gamma$ varies from $-\\infty$ (accept all) to $+\\infty$ (reject all). The curve depends **only on the ranking of scores**, not their absolute magnitude.

        **Derivation:** Suppose we order examples by descending score: $s(x_1) \\ge s(x_2) \\ge \\cdots \\ge s(x_n)$. At coverage level $\\varphi = k/n$, we retain the top-$k$ examples and compute their error rate. If the score scale is multiplied by a constant $\\lambda > 0$, the ranking remains identical, so the RC curve is invariant to positive scaling.

        ### The Area Under the Risk-Coverage Curve (AURC)

        A single-number summary of selective classification performance is the area under the RC curve:

        $$\\text{AURC} = \\int_0^1 R(\\varphi) \\, d\\varphi$$

        where $R(\\varphi)$ is the selection risk at coverage $\\varphi$. Numerically, using the trapezoid rule:

        $$\\text{AURC}_{\\text{trap}} = \\sum_{i=1}^{m-1} \\frac{1}{2} (R_i + R_{i+1}) (\\varphi_{i+1} - \\varphi_i)$$

        **Normalized partial AURC-$\\alpha$:** To focus on a specific coverage regime $[0, \\alpha]$, we compute:

        $$\\text{nAURC}_{\\alpha} = \\frac{1}{\\alpha} \\int_0^\\alpha R(\\varphi) \\, d\\varphi$$

        This is the area-normalized average risk over the first $\\alpha$ fraction of samples (lower is better).

        **Oracle lower bound (derived here):** Let the base classifier have error rate $e$. The oracle score ranks every correct prediction above every error. Then $R(\\varphi)=0$ for $\\varphi \\le 1-e$, and beyond that the retained set contains $(\\varphi-(1-e))n$ errors out of $\\varphi n$ samples, so $R(\\varphi) = 1 - (1-e)/\\varphi$. Integrating,

        $$\\text{AURC}_{\\text{oracle}}(e) = \\int_{1-e}^{1}\\Bigl(1-\\frac{1-e}{\\varphi}\\Bigr)d\\varphi = e + (1-e)\\ln(1-e).$$

        This is strictly positive for $0<e<1$: even a perfect ranker cannot reach AURC $=0$, because at coverage 1 the risk is the base error $e$. For $e=0.3$ it equals $0.3+0.7\\ln 0.7$, evaluated in the code below.

        **Random-score reference:** If the score is independent of correctness, the expected RC curve is flat at the base error rate $e$, so the expected AURC is $e$ (a finite-$n$ curve fluctuates around this line, most at small coverage where few samples are retained).
        """),

        code("""
        # S3.1: Toy dataset generation and risk-coverage curve computation.
        import numpy as np
        from src.metrics import risk_coverage_curve, selective_risk, aurc, bootstrap_interval
        import matplotlib.pyplot as plt

        # Generate a simple 4-class dataset for selective classification.
        s_n_toy3 = 100
        s_n_classes3 = 4
        s_rng3 = np.random.default_rng(RNG_SEED + 300)

        # Generate predicted probabilities with varying confidence.
        s_logits3 = s_rng3.standard_normal((s_n_toy3, s_n_classes3))
        s_true_labels3 = s_rng3.integers(0, s_n_classes3, s_n_toy3)
        # Boost correct class logit to simulate a working classifier.
        s_logits3[np.arange(s_n_toy3), s_true_labels3] += 2.0
        # Softmax to get probabilities.
        s_max_logits3 = np.max(s_logits3, axis=1, keepdims=True)
        s_exp_logits3 = np.exp(s_logits3 - s_max_logits3)
        s_probs3 = s_exp_logits3 / np.sum(s_exp_logits3, axis=1, keepdims=True)

        # Extract max-softmax confidence as score.
        s_conf3 = np.max(s_probs3, axis=1)
        # Compute correctness (0/1 loss).
        s_pred3 = np.argmax(s_probs3, axis=1)
        s_correct3 = (s_pred3 == s_true_labels3)

        # Compute risk-coverage curve.
        s_coverage3, s_risk3 = risk_coverage_curve(s_conf3, s_correct3)

        check("S3.1", len(s_coverage3) == s_n_toy3 and s_coverage3[-1] == 1.0 and abs(s_risk3[-1] - np.mean(~s_correct3)) < 1e-12,
              f"RC curve has {len(s_coverage3)} points (one per row); risk at coverage 1 is {s_risk3[-1]:.4f} = base error {np.mean(~s_correct3):.4f}")

        print(f"Risk-Coverage Curve (n={s_n_toy3}, K={s_n_classes3}):")
        print(f"  Coverage range: [{s_coverage3[0]:.3f}, {s_coverage3[-1]:.3f}]")
        print(f"  Risk range: [{s_risk3[0]:.3f}, {s_risk3[-1]:.3f}]")
        """),

        code("""
        # S3.2: Worked numeric example - hand-computed AURC on 8 points (no randomness, n=8).
        from fractions import Fraction
        s_scores_tiny = np.array([0.95, 0.90, 0.75, 0.65, 0.55, 0.50, 0.40, 0.25])
        s_errors_tiny = np.array([False, False, True, False, True, True, False, True])  # True = the prediction is wrong.
        s_correct_tiny = ~s_errors_tiny  # risk_coverage_curve expects `correct`, not `errors`.

        s_cov_tiny, s_risk_tiny = risk_coverage_curve(s_scores_tiny, s_correct_tiny)
        s_aurc_tiny = aurc(s_cov_tiny, s_risk_tiny)

        # Independent hand computation with exact fractions: scores are already sorted descending.
        s_cum_err = np.cumsum(s_errors_tiny.astype(int))
        s_hand_risk = [Fraction(int(s_cum_err[k]), k + 1) for k in range(8)]
        s_hand_aurc = sum((s_hand_risk[k] + s_hand_risk[k + 1]) / 2 * Fraction(1, 8) for k in range(7))
        s_base_error_tiny = float(np.mean(s_errors_tiny))  # 4/8

        check("S3.2a", abs(s_aurc_tiny - float(s_hand_aurc)) < 1e-12,
              f"aurc()={s_aurc_tiny:.6f} vs exact-fraction hand value {float(s_hand_aurc):.6f} = {s_hand_aurc}")
        check("S3.2b", 0.0 < s_aurc_tiny < s_base_error_tiny,
              f"AURC {s_aurc_tiny:.4f} lies strictly below base error {s_base_error_tiny:.4f} because the score ranks errors low")
        check("S3.2c", np.allclose(s_risk_tiny, [float(r) for r in s_hand_risk], atol=1e-12) and s_risk_tiny[-1] == s_base_error_tiny,
              f"risk at full coverage {s_risk_tiny[-1]:.4f} equals base error {s_base_error_tiny:.4f}")

        print("Tiny example (n=8, 4 errors):")
        print(f"  Coverage: {[f'{c:.3f}' for c in s_cov_tiny]}")
        print(f"  Risk:     {[f'{r:.3f}' for r in s_risk_tiny]}")
        print(f"  AURC (trapezoid) = {s_aurc_tiny:.4f}; hand value = {s_hand_aurc} = {float(s_hand_aurc):.4f}")
        """),

        code("""
        # S3.3: Verify that RC curve depends only on ranking, not scale.
        # Scale the confidence scores and check RC curve invariance.
        s_conf_scaled1 = s_conf3.copy()
        s_conf_scaled2 = s_conf3 * 2.0
        s_conf_scaled3 = s_conf3 * 0.1

        s_cov1, s_risk1 = risk_coverage_curve(s_conf_scaled1, s_correct3)
        s_cov2, s_risk2 = risk_coverage_curve(s_conf_scaled2, s_correct3)
        s_cov3, s_risk3_scaled = risk_coverage_curve(s_conf_scaled3, s_correct3)

        # Check that curves are identical (coverage and risk should not change).
        s_max_dev3 = max(float(np.max(np.abs(s_risk1 - s_risk2))), float(np.max(np.abs(s_risk1 - s_risk3_scaled))))
        check("S3.3", s_max_dev3 < 1e-12,
              f"RC curve scale-invariant: max |risk diff| across scales x1, x2, x0.1 = {s_max_dev3:.1e} (< 1e-12)")

        s_aurc1 = aurc(s_cov1, s_risk1)
        s_aurc2 = aurc(s_cov2, s_risk2)
        s_aurc3 = aurc(s_cov3, s_risk3_scaled)
        print(f"AURC invariance test:")
        print(f"  AURC (original):       {s_aurc1:.4f}")
        print(f"  AURC (scale × 2):      {s_aurc2:.4f}")
        print(f"  AURC (scale × 0.1):    {s_aurc3:.4f}")
        """),

        code("""
        # S3.4: Monte Carlo - random score vs oracle score against closed forms (n=2000, reps=100, seed RNG_SEED+304).
        s_n_monte3 = 2000
        s_n_reps3 = 100
        s_p_err3 = 0.3  # true base error of the binary toy classifier
        s_rng_monte3 = np.random.default_rng(RNG_SEED + 304)

        s_aurc_random_scores, s_aurc_oracle_scores, s_oracle_closed_hat = [], [], []
        for s_rep in range(s_n_reps3):
            s_correct_mc = s_rng_monte3.uniform(0, 1, s_n_monte3) >= s_p_err3  # correct w.p. 0.7
            s_e_hat = float(np.mean(~s_correct_mc))
            s_scores_random = s_rng_monte3.uniform(0, 1, s_n_monte3)
            s_aurc_random_scores.append(aurc(*risk_coverage_curve(s_scores_random, s_correct_mc)))
            s_scores_oracle = s_correct_mc.astype(float)  # correct rows rank first
            s_aurc_oracle_scores.append(aurc(*risk_coverage_curve(s_scores_oracle, s_correct_mc)))
            s_oracle_closed_hat.append(s_e_hat + (1 - s_e_hat) * np.log(1 - s_e_hat))

        s_aurc_random_arr = np.array(s_aurc_random_scores)
        s_aurc_oracle_arr = np.array(s_aurc_oracle_scores)
        s_oracle_closed = s_p_err3 + (1 - s_p_err3) * np.log(1 - s_p_err3)  # e + (1-e) ln(1-e)
        s_random_mean, s_oracle_mean = s_aurc_random_arr.mean(), s_aurc_oracle_arr.mean()
        s_random_se = s_aurc_random_arr.std(ddof=1) / np.sqrt(s_n_reps3)
        s_oracle_se = s_aurc_oracle_arr.std(ddof=1) / np.sqrt(s_n_reps3)
        # Per-replicate comparison to the closed form evaluated at that replicate's realised error rate.
        s_oracle_gap = s_aurc_oracle_arr - np.array(s_oracle_closed_hat)

        check("S3.4a", abs(s_random_mean - s_p_err3) <= 3 * s_random_se,
              f"random-score AURC mean {s_random_mean:.4f} vs base error {s_p_err3} (|diff|={abs(s_random_mean - s_p_err3):.4f} <= 3 SE = {3 * s_random_se:.4f})")
        check("S3.4b", abs(s_oracle_mean - s_oracle_closed) <= 3 * s_oracle_se,
              f"oracle AURC mean {s_oracle_mean:.4f} vs closed form e+(1-e)ln(1-e) = {s_oracle_closed:.4f} (|diff|={abs(s_oracle_mean - s_oracle_closed):.4f} <= 3 SE = {3 * s_oracle_se:.4f})")
        check("S3.4c", s_oracle_closed > 0.03 and s_oracle_mean > 0.03,
              f"oracle AURC is strictly positive (closed form {s_oracle_closed:.4f}, measured {s_oracle_mean:.4f}); the value 0 is not attained")
        check("S3.4d", np.max(np.abs(s_oracle_gap)) < 1e-6,
              f"per-replicate trapezoid vs closed form at realised error: max |gap|={np.max(np.abs(s_oracle_gap)):.2e} (< 1e-6; discretisation error at n={s_n_monte3})")

        print(f"Monte Carlo (n={s_n_monte3}, reps={s_n_reps3}, base error {s_p_err3}):")
        print(f"  Random score AURC: mean={s_random_mean:.4f}, std={np.std(s_aurc_random_arr, ddof=1):.4f}, SE={s_random_se:.4f}")
        print(f"  Oracle score AURC: mean={s_oracle_mean:.4f}, std={np.std(s_aurc_oracle_arr, ddof=1):.4f}, SE={s_oracle_se:.4f}; closed form={s_oracle_closed:.4f}")
        """),

        code("""
        # S3.5: The trapezoid rule used by src.metrics.aurc versus a manual sum and versus a plain mean of risks.
        s_aurc_trap = aurc(s_cov_tiny, s_risk_tiny)
        s_manual = sum(0.5 * (s_risk_tiny[k] + s_risk_tiny[k + 1]) * (s_cov_tiny[k + 1] - s_cov_tiny[k]) for k in range(len(s_cov_tiny) - 1))
        s_aurc_discrete_avg = float(np.mean(s_risk_tiny))
        # A plain mean over all 8 points weights the first point 1/8 and includes the left edge; the trapezoid starts at coverage 1/8.
        s_expected_gap = abs(s_aurc_trap - s_aurc_discrete_avg)

        print("Tiny example AURC comparison:")
        print(f"  Trapezoid rule (src.metrics.aurc): {s_aurc_trap:.4f}")
        print(f"  Manual trapezoid sum:              {s_manual:.4f}")
        print(f"  Plain mean of the 8 risks:         {s_aurc_discrete_avg:.4f}   (gap {s_expected_gap:.4f})")
        check("S3.5", abs(s_aurc_trap - s_manual) < 1e-12 and s_expected_gap > 1e-3,
              f"aurc() equals the manual trapezoid sum to 1e-12, and differs from the plain mean by {s_expected_gap:.4f}")
        """),

        # ─────────────────────────────────────────────────────────────────────────────────
        # Plot 1: Risk-coverage curves and AURC
        # ─────────────────────────────────────────────────────────────────────────────────

        code("""
        # S3.6: Plot risk-coverage curves for random, imperfect, and oracle scores (n=200, seed RNG_SEED+306).
        s_rng_plot3 = np.random.default_rng(RNG_SEED + 306)
        s_n_plot3 = 200
        s_labels_plot3 = s_rng_plot3.integers(0, 2, s_n_plot3)
        s_preds_plot3 = np.where(s_rng_plot3.uniform(0, 1, s_n_plot3) < 0.75, s_labels_plot3, 1 - s_labels_plot3)
        s_correct_plot3 = (s_preds_plot3 == s_labels_plot3)
        s_error_rate_plot3 = float(np.mean(~s_correct_plot3))

        # Three score functions:
        s_scores_rand_plot = s_rng_plot3.uniform(0, 1, s_n_plot3)  # Random.
        s_scores_real_plot = 0.3 * s_correct_plot3.astype(float) + 0.25 * s_rng_plot3.standard_normal(s_n_plot3)  # Imperfect score: weak correctness signal plus noise.
        s_scores_oracle_plot = s_correct_plot3.astype(float)  # Oracle.

        s_cov_rand, s_risk_rand = risk_coverage_curve(s_scores_rand_plot, s_correct_plot3)
        s_cov_real, s_risk_real = risk_coverage_curve(s_scores_real_plot, s_correct_plot3)
        s_cov_oracle, s_risk_oracle = risk_coverage_curve(s_scores_oracle_plot, s_correct_plot3)

        s_aurc_rand_plot = aurc(s_cov_rand, s_risk_rand)
        s_aurc_real_plot = aurc(s_cov_real, s_risk_real)
        s_aurc_oracle_plot = aurc(s_cov_oracle, s_risk_oracle)

        plt.figure(figsize=(8, 5.5))
        plt.plot(s_cov_rand, s_risk_rand, label=f"Random score (AURC={s_aurc_rand_plot:.4f})", color=PALETTE["reference"], linewidth=2, linestyle="--")
        plt.plot(s_cov_real, s_risk_real, label=f"Imperfect score (AURC={s_aurc_real_plot:.4f})", color=PALETTE["clean"], linewidth=2.5)
        plt.plot(s_cov_oracle, s_risk_oracle, label=f"Oracle score (AURC={s_aurc_oracle_plot:.4f})", color=PALETTE["frozen"], linewidth=2, linestyle=":")
        # Baseline: flat line at error rate.
        plt.axhline(s_error_rate_plot3, color=PALETTE["band"], linewidth=1.5, linestyle="-.", alpha=0.7, label=f"Base error rate={s_error_rate_plot3:.3f}")
        plt.xlabel("Coverage φ", fontsize=12)
        plt.ylabel("Selection Risk R", fontsize=12)
        plt.title("Risk-Coverage Curves: Effect of Score Quality", fontsize=13, fontweight="bold")
        plt.legend(fontsize=10, loc="best")
        plt.xlim(0, 1)
        plt.ylim(0, max(s_risk_rand) * 1.05)
        plt.tight_layout()
        plt.show()

        check("S3.6", s_aurc_oracle_plot < s_aurc_real_plot < s_aurc_rand_plot,
              f"oracle {s_aurc_oracle_plot:.4f} < imperfect {s_aurc_real_plot:.4f} < random {s_aurc_rand_plot:.4f} (n={s_n_plot3}, base error {s_error_rate_plot3:.3f})")
        """),

        md("""
        ### How to read this chart

        The risk-coverage curve plots selection risk on the y-axis against coverage (fraction of samples retained) on the x-axis. Each curve shows what happens as we vary the threshold $\\gamma$ to accept or reject examples:

        - **x-axis (Coverage):** Moves from left (reject most, keep only the most confident) to right (accept all). Coverage $\\varphi = 0$ means reject all; $\\varphi = 1$ means accept all.
        - **y-axis (Selection Risk):** The error rate among retained samples. Lower is better.
        - **Oracle score (dotted line):** Sits at risk 0 until coverage reaches one minus the base error, then climbs toward the base error at coverage 1, following $1-(1-e)/\\varphi$. It is an unattainable lower bound, and its AURC is positive, not zero.
        - **Imperfect score (solid line):** Lies above the oracle and below the random curve; risk falls as coverage decreases because low-score rows are rejected, but it does not reach zero because the score is noisy.
        - **Random score (dashed line):** Fluctuates around the base error rate (most at small coverage where few rows are retained) because random scores carry no ranking information.
        - **Base error rate (dash-dot line):** The classifier's overall error rate if we predict on all samples (coverage = 1).

        **Takeaway:** AURC is the average selection risk over coverage levels, so a smaller area means the score keeps errors out of the retained set more effectively. A good score bends the curve downward as coverage shrinks; a useless score stays flat near the base error. The area between the imperfect and random curves quantifies the score's utility, and the area between the imperfect and oracle curves is the headroom that remains.
        """),

        code("""
        # S3.7: Normalized partial AURC-alpha at multiple coverage levels.
        # Compute AURC only over the first alpha fraction of samples.
        s_alphas = np.array([0.25, 0.5, 0.75, 1.0])
        s_n_alphas = len(s_alphas)
        s_naurcs_rand = []
        s_naurcs_real = []

        for s_alpha in s_alphas:
            # Find the index where coverage reaches alpha.
            s_idx_alpha_rand = np.searchsorted(s_cov_rand, s_alpha, side="right")
            s_idx_alpha_real = np.searchsorted(s_cov_real, s_alpha, side="right")

            if s_idx_alpha_rand > 0:
                s_cov_alpha_rand = s_cov_rand[:s_idx_alpha_rand]
                s_risk_alpha_rand = s_risk_rand[:s_idx_alpha_rand]
                # Ensure endpoint is exactly at alpha.
                if s_cov_alpha_rand[-1] < s_alpha:
                    s_cov_alpha_rand = np.append(s_cov_alpha_rand, s_alpha)
                    s_idx_linear = np.interp(s_alpha, s_cov_rand, np.arange(len(s_cov_rand)))
                    s_risk_interp = np.interp(s_alpha, s_cov_rand, s_risk_rand)
                    s_risk_alpha_rand = np.append(s_risk_alpha_rand, s_risk_interp)
                s_partial_aurc_rand = aurc(s_cov_alpha_rand, s_risk_alpha_rand)
                s_naurcs_rand.append(s_partial_aurc_rand / s_alpha)
            else:
                s_naurcs_rand.append(np.nan)

            if s_idx_alpha_real > 0:
                s_cov_alpha_real = s_cov_real[:s_idx_alpha_real]
                s_risk_alpha_real = s_risk_real[:s_idx_alpha_real]
                if s_cov_alpha_real[-1] < s_alpha:
                    s_cov_alpha_real = np.append(s_cov_alpha_real, s_alpha)
                    s_risk_interp = np.interp(s_alpha, s_cov_real, s_risk_real)
                    s_risk_alpha_real = np.append(s_risk_alpha_real, s_risk_interp)
                s_partial_aurc_real = aurc(s_cov_alpha_real, s_risk_alpha_real)
                s_naurcs_real.append(s_partial_aurc_real / s_alpha)
            else:
                s_naurcs_real.append(np.nan)

        print(f"Normalized partial AURC-alpha:")
        print(f"  Alpha    | nAURC (random) | nAURC (real)")
        for s_a, s_nr, s_nreal in zip(s_alphas, s_naurcs_rand, s_naurcs_real):
            print(f"  {s_a:.2f}    | {s_nr:14.4f} | {s_nreal:.4f}")

        check("S3.7", all(~np.isnan(s_naurcs_real)) and all(s_naurcs_real[i] <= s_naurcs_rand[i] for i in range(len(s_alphas))),
              f"nAURC-alpha computed for {len(s_alphas)} levels; real <= random everywhere")
        """),

        # ─────────────────────────────────────────────────────────────────────────────────
        # Section 4: Score functions and scale sensitivity (source: L Sec 3.1-3.2, derived here)
        # ─────────────────────────────────────────────────────────────────────────────────

        md("""
        ## 4. Score functions and scale sensitivity (source: L Sec 3.1-3.2; derived here for the toy)

        Selective classification ranks examples by a score $s(x)$. The paper distinguishes two families: **softmax-response (SR) scores**, computed from the softmax output and therefore sensitive to the overall scale of the logits, and **margin scores**, computed directly from the logits (or from distances to the decision hyperplanes) and therefore insensitive to it.

        ### Softmax-response scores (Eq. 7; L p.4)

        With $p = \\text{softmax}(z)$ for logits $z \\in \\mathbb{R}^K$, the paper defines three SR scores named $\\text{SR}_{\\max}$, $\\text{SR}_{\\text{doctor}}$ and $\\text{SR}_{\\text{ent}}$. The forms used in this notebook are

        $$s_{\\max}(x) = \\max_k p_k, \\qquad s_{\\text{doctor}}(x) = 1 - \\frac{1}{\\sum_k p_k^2}, \\qquad s_{\\text{ent}}(x) = \\sum_k p_k \\log p_k = -H(p).$$

        **Provenance.** These are the three softmax-response scores of Eq. (7) as printed in the paper (L p.4): max-softmax, the DOCTOR score $1 - 1/\\lVert p\\rVert_2^2$, and the negative entropy $\\sum_k p_k \\log p_k$; each is oriented so that a larger value means more confident. The DOCTOR score is a strictly increasing function of $\\sum_k p_k^2$, so ranking by that explicitly named order key gives the same ordering (and risk-coverage curve) as ranking by the displayed DOCTOR score. A risk-coverage curve depends only on the ordering of the scores, so any strictly increasing transform of these forms gives the same curve. The top-two probability gap $p_{(1)} - p_{(2)}$ is a different score and is not used here.

        Multiplying the logits by a scale $\\lambda > 0$ changes the softmax output:

        $$\\text{softmax}(\\lambda z) \\to \\text{uniform} \\ \\text{ as } \\lambda \\to 0, \\qquad \\text{softmax}(\\lambda z) \\to \\text{one-hot at } \\arg\\max_k z_k \\ \\text{ as } \\lambda \\to \\infty .$$

        Every SR score is a function of $p$, so its values and its **ordering** of examples can change with $\\lambda$.

        ### Margin scores (Eqs. 11-13; L p.8)

        The confidence margin is defined on the logits, and the geometric margin on signed distances $d_j = (w_j \\cdot \\phi(x) + b_j)/\\lVert w_j \\rVert$ to the class hyperplanes:

        $$s_{\\text{conf-M}}(x) = z_{(1)} - z_{(2)}, \\qquad s_{\\text{geo-M}}(x) = d_{(1)} - d_{(2)} .$$

        Only the confidence margin is implemented below. Scaling gives $s_{\\text{conf-M}}(\\lambda z) = \\lambda\\, s_{\\text{conf-M}}(z)$, a positive multiple, so the ordering of examples, and hence the whole RC curve, is unchanged.

        ### Lemma 3.1 (paraphrase; L pp.6-7)

        **Paraphrase of Liang et al. Lemma 3.1; L pp.6-7.** As the logit scale $\\lambda$ grows without bound, the ordering induced by each of the three SR scores approaches the ordering induced by the confidence margin $z_{(1)} - z_{(2)}$.

        ### What we derive and test here (derived here)

        Write $d_k = z_{(1)} - z_{(k)} \\ge 0$ for $k = 2, \\dots, K$ (so $d_2$ is the confidence margin) and $S_\\lambda = \\sum_{k \\ge 2} e^{-\\lambda d_k}$. Then $p_{(1)} = 1/(1+S_\\lambda)$ and $p_{(k)} = e^{-\\lambda d_k}/(1+S_\\lambda)$. In floating point $p_{(1)}$ rounds to exactly 1 once $S_\\lambda < 10^{-16}$, so the displayed $s_{\\max}=p_{(1)}$ rounds to 1 and creates ties. We therefore keep $\\log(1-s_{\\max})$ as a log-domain complement, used only to recover the actual score or preserve its ranking. For entropy the log-domain quantity is likewise a log-complement; for DOCTOR we use the explicitly named $\\text{doctor\\_order\\_key}=-\\log\\sum_k p_k^2$. This key decreases strictly as the displayed DOCTOR score $s_{\\text{doctor}}=1-1/\\sum_k p_k^2$ increases, so its negation gives the same descending ranking, including at high logit scales.

        $$1 - s_{\\max} = \\frac{S_\\lambda}{1+S_\\lambda}, \\qquad \\sum_k p_k^2 = \\frac{1+Q_\\lambda}{(1+S_\\lambda)^2}, \\qquad H = \\log(1+S_\\lambda) + \\frac{\\lambda \\sum_{k\\ge2} d_k e^{-\\lambda d_k}}{1+S_\\lambda}.$$

        Here $Q_\\lambda = \\sum_{k\\ge2} e^{-2\\lambda d_k}$, and therefore $\\text{doctor\\_order\\_key}=2\\log(1+S_\\lambda)-\\log(1+Q_\\lambda)$. Since $s_{\\text{doctor}}=1-\\exp(\\text{doctor\\_order\\_key})$, decreasing the key strictly increases the actual score. For entropy, $-\\sum_k p_k \\log p_k = \\sum_k p_k(\\lambda d_k + \\log(1+S_\\lambda))$ with $d_1 = 0$. The log-domain keys avoid rounding ties while preserving the displayed-score ranking.

        **A bound for $s_{\\max}$.** Since $S_\\lambda = e^{-\\lambda d_2}\\bigl(1 + \\sum_{k\\ge3} e^{-\\lambda(d_k - d_2)}\\bigr)$ and $d_k \\ge d_2$ for $k \\ge 3$, the log-complement satisfies

        $$\\lambda d_2 - \\log(K-1) \\;\\le\\; -\\log S_\\lambda \\;\\le\\; \\lambda d_2 .$$

        Hence if two examples have margins with $\\lambda\\,(d_2(x) - d_2(x')) > \\log(K-1)$, then $x$ outranks $x'$ under $s_{\\max}$ too. Rank disagreement with the margin can therefore only occur among pairs whose margins differ by less than $\\log(K-1)/\\lambda$: for $K = 4$ and $\\lambda = 100$ that window is $\\log 3 / 100 \\approx 0.011$ in margin units. This is a worked example of why the ordering must converge. It is proved here only for $s_{\\max}$; for $s_{\\text{doctor}}$ and $s_{\\text{ent}}$ the leading term is also a monotone function of $\\lambda d_2$, but this study checks that numerically rather than proving it.

        **What would make the toy conclusion wrong:** the checks below would fail if the log-domain score keys disagreed with their displayed scores where direct arithmetic is accurate, if the DOCTOR order-key relationship failed, or if the bound above were violated by any pair of examples.
        """),

        code("""
        # S4.1: K=4 Gaussian-mixture toy in d=5 with class overlap; multinomial logistic regression fit on n=80 rows.
        # Separate calibration (n=400) and test (n=2000) sets are drawn independently of the training rows. Seed RNG_SEED+400.
        from sklearn.linear_model import LogisticRegression
        from src.calibration import temperature_scale
        from scipy.special import logsumexp
        from scipy.stats import spearmanr

        s_K4, s_d4 = 4, 5
        s_rng4 = np.random.default_rng(RNG_SEED + 400)
        s_centers4 = s_rng4.standard_normal((s_K4, s_d4))

        def s_draw4(n):
            y = s_rng4.integers(0, s_K4, n)
            return s_centers4[y] + s_rng4.standard_normal((n, s_d4)), y

        s_Xtr4, s_ytr4 = s_draw4(80)
        s_Xca4, s_yca4 = s_draw4(400)
        s_Xte4, s_yte4 = s_draw4(2000)
        s_clf4 = LogisticRegression(C=5.0, max_iter=5000).fit(s_Xtr4, s_ytr4)
        s_zca4 = s_clf4.decision_function(s_Xca4)
        s_zte4 = s_clf4.decision_function(s_Xte4)
        s_correct4 = np.argmax(s_zte4, axis=1) == s_yte4
        s_train_acc4 = s_clf4.score(s_Xtr4, s_ytr4)

        check("S4.1a", int(np.sum(~s_correct4)) >= 100,
              f"test set has {int(np.sum(~s_correct4))} errors of {len(s_correct4)} (>= 100 needed for a non-degenerate RC curve)")
        check("S4.1b", s_train_acc4 > float(np.mean(s_correct4)),
              f"train accuracy {s_train_acc4:.3f} > held-out accuracy {np.mean(s_correct4):.3f}: the finite-sample fit is overconfident on new rows")

        print(f"Toy: K={s_K4}, d={s_d4}, train n=80, calibration n=400, test n=2000")
        print(f"  test error = {np.mean(~s_correct4):.3f}; logit range on test = [{s_zte4.min():.2f}, {s_zte4.max():.2f}]")
        """),

        code("""
        # S4.2: Stable SR ranking keys, validated against displayed softmax scores where direct arithmetic is accurate.
        def s_gaps4(z):
            zs = -np.sort(-z, axis=1)
            return zs[:, :1] - zs[:, 1:]  # d_k = z(1) - z(k), k = 2..K; column 0 is the confidence margin.

        def s_log_keys4(z, lam):
            # Returns log(1 - actual SR_max), a stable DOCTOR confidence key, and log(H).
            D = lam * s_gaps4(z)
            logS = logsumexp(-D, axis=1)
            l1pS = np.log1p(np.exp(logS))  # log(1 + S)
            l_max = logS - l1pS
            logQ = logsumexp(-2.0 * D, axis=1)
            # -s_doctor = R / (1 + Q), R = 2S + (S² - Q).
            # Evaluate log(S²-Q) from distinct pairs to avoid cancellation at large lambda.
            s_pair_i4, s_pair_j4 = np.triu_indices(D.shape[1], k=1)
            if len(s_pair_i4):
                logX = np.log(2.0) + logsumexp(-D[:, s_pair_i4] - D[:, s_pair_j4], axis=1)
            else:
                logX = np.full(len(D), -np.inf)
            logR = np.logaddexp(np.log(2.0) + logS, logX)
            doctor_log_magnitude = logR - np.logaddexp(0.0, logQ)
            doctor_order_key = -doctor_log_magnitude
            logT = logsumexp(-D + np.log(np.maximum(D, 1e-300)), axis=1)
            l_log1p = np.where(logS < -30, logS, np.log(np.maximum(l1pS, 1e-300)))
            l_ent = np.logaddexp(l_log1p, logT - l1pS)
            return {"SR_max": l_max, "doctor_order_key": doctor_order_key,
                    "doctor_log_magnitude": doctor_log_magnitude, "SR_ent": l_ent}

        def s_direct4(z, lam):
            a = lam * z
            a = a - a.max(axis=1, keepdims=True)
            p = np.exp(a)
            p /= p.sum(axis=1, keepdims=True)
            return {"SR_max": p.max(axis=1), "SR_doctor": 1 - 1.0 / (p ** 2).sum(axis=1),
                    "SR_ent": (p * np.log(np.clip(p, 1e-300, 1))).sum(axis=1)}

        s_lam_val4 = 0.5
        s_lc4 = s_log_keys4(s_zte4, s_lam_val4)
        s_dr4 = s_direct4(s_zte4, s_lam_val4)
        s_rel_err4 = {}
        s_stable4 = {"SR_max": -np.expm1(s_lc4["SR_max"]),
                     "SR_doctor": -np.exp(s_lc4["doctor_log_magnitude"]),
                     "SR_ent": -np.exp(s_lc4["SR_ent"])}
        s_key_for_rc4 = lambda vals, name: vals["doctor_order_key"] if name == "SR_doctor" else -vals[name]
        for s_name in s_dr4:
            s_ok = np.abs(s_dr4[s_name]) > 1e-9
            s_rel_err4[s_name] = float(np.max(np.abs(s_stable4[s_name][s_ok] - s_dr4[s_name][s_ok]) / np.abs(s_dr4[s_name][s_ok])))
            check(f"S4.2-{s_name}", s_rel_err4[s_name] < 1e-8 and s_ok.mean() > 0.99,
                  f"stable log-domain score vs direct displayed score at lambda={s_lam_val4}: max rel. error {s_rel_err4[s_name]:.1e} on {s_ok.mean():.1%} of rows (< 1e-8)")
        s_doc_key4 = s_lc4["doctor_order_key"]
        s_doctor_from_key4 = -np.exp(-s_doc_key4)
        s_doc_order_ok4 = np.array_equal(np.argsort(s_doctor_from_key4), np.argsort(s_doc_key4))
        check("S4.2-doctor-order", s_doc_order_ok4 and np.allclose(s_doctor_from_key4, s_stable4["SR_doctor"], atol=0, rtol=0),
              f"doctor_order_key=-log(-s_doctor) ranks actual s_doctor=-exp(-doctor_order_key) identically at lambda={s_lam_val4}")

        s_lam_grid4 = np.array([0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0])
        s_keys4 = {lam: s_log_keys4(s_zte4, lam) for lam in s_lam_grid4}
        s_margin4 = s_gaps4(s_zte4)[:, 0]
        # At lambda=1000 the direct displayed SR_max rounds to exactly 1 for many rows; the log-complement key preserves its ordering.
        s_direct_big = s_direct4(s_zte4, 1000.0)["SR_max"]
        print(f"At lambda=1000, direct SR_max rounds to exactly 1 for {np.mean(s_direct_big == 1):.1%} of rows (ties from rounding);")
        print(f"the log-domain log-complement key, used to recover/rank actual SR_max, has {len(np.unique(s_keys4[1000.0]['SR_max']))} distinct values out of {len(s_margin4)}.")
        """),

        code("""
        # S4.2b: Worked numeric example, K=3, two rows whose SR_max order disagrees with the margin order at lambda=1 (no randomness).
        s_row_a = np.array([[3.0, 2.0, -9.0]])   # margin 1.0, but the runner-up logit is close to the top one
        s_row_b = np.array([[1.5, 0.0, 0.0]])    # margin 1.5, two runners-up
        s_marg_a, s_marg_b = float(s_gaps4(s_row_a)[0, 0]), float(s_gaps4(s_row_b)[0, 0])
        # Hand arithmetic: SR_max = 1/(1+S) with S = sum_k exp(-lambda d_k).
        def s_hand_srmax(z, lam):
            d = z[0].max() - np.sort(z[0])[::-1][1:]
            return 1.0 / (1.0 + np.exp(-lam * d).sum())
        s_ha1, s_hb1 = s_hand_srmax(s_row_a, 1.0), s_hand_srmax(s_row_b, 1.0)
        s_ha10, s_hb10 = s_hand_srmax(s_row_a, 10.0), s_hand_srmax(s_row_b, 10.0)
        s_la10 = -s_log_keys4(s_row_a, 10.0)["SR_max"][0]
        s_lb10 = -s_log_keys4(s_row_b, 10.0)["SR_max"][0]

        check("S4.2b-flip", s_marg_b > s_marg_a and s_ha1 > s_hb1,
              f"lambda=1: margin(b)={s_marg_b:.1f} > margin(a)={s_marg_a:.1f} but SR_max(a)={s_ha1:.4f} > SR_max(b)={s_hb1:.4f} (orders disagree)")
        check("S4.2b-agree", s_hb10 > s_ha10 and s_lb10 > s_la10 and 10.0 * (s_marg_b - s_marg_a) > np.log(2.0),
              f"lambda=10: SR_max(b)={s_hb10:.6f} > SR_max(a)={s_ha10:.6f}, log-complement keys agree ({s_lb10:.2f} > {s_la10:.2f}); "
              f"bound premise lambda*(gap)={10.0 * (s_marg_b - s_marg_a):.1f} > log(K-1)={np.log(2.0):.2f}")
        print("Row a: logits [3, 2, -9]; row b: logits [1.5, 0, 0]. The third logit of row b adds a second runner-up, which lowers its softmax mass on the top class at small scale.")
        """),

        code("""
        # S4.3: RC curves and AURC of SR scores versus margin over the lambda grid (test set of S4.1, no randomness).
        s_aurc_sr4 = {name: [] for name in ["SR_max", "SR_doctor", "SR_ent"]}
        s_aurc_margin4 = []
        s_risk_margin4 = []
        s_risk_srmax4 = []
        for lam in s_lam_grid4:
            for name in s_aurc_sr4:
                s_aurc_sr4[name].append(aurc(*risk_coverage_curve(s_key_for_rc4(s_keys4[lam], name), s_correct4)))
            s_cov_m, s_risk_m = risk_coverage_curve(lam * s_margin4, s_correct4)  # margin score at scale lambda
            s_risk_margin4.append(s_risk_m)
            s_aurc_margin4.append(aurc(s_cov_m, s_risk_m))
            s_risk_srmax4.append(risk_coverage_curve(-s_keys4[lam]["SR_max"], s_correct4)[1])

        s_margin_dev4 = max(float(np.max(np.abs(r - s_risk_margin4[0]))) for r in s_risk_margin4)
        s_i_lo, s_i_hi = 0, int(np.where(s_lam_grid4 == 10.0)[0][0])
        s_srmax_dev4 = float(np.max(np.abs(s_risk_srmax4[s_i_lo] - s_risk_srmax4[s_i_hi])))
        s_rho_max_1, _ = spearmanr(-s_keys4[1.0]["SR_max"], s_margin4)

        check("S4.3a", s_margin_dev4 < 1e-12,
              f"margin RC curve identical across all {len(s_lam_grid4)} scales: max |risk diff| = {s_margin_dev4:.1e} (< 1e-12)")
        check("S4.3b", s_srmax_dev4 > 0.0 and s_rho_max_1 < 1 - 1e-3,
              f"SR_max RC curve changes with scale: max |risk diff| lambda=0.1 vs 10 is {s_srmax_dev4:.4f}; Spearman(SR_max, margin) at lambda=1 is {s_rho_max_1:.4f} (< 0.999)")

        print("AURC over the lambda grid (lower is better):")
        print("  lambda     " + " ".join(f"{l:8g}" for l in s_lam_grid4))
        for name in s_aurc_sr4:
            print(f"  {name:10s} " + " ".join(f"{a:8.4f}" for a in s_aurc_sr4[name]))
        print("  margin     " + " ".join(f"{a:8.4f}" for a in s_aurc_margin4))
        """),

        code("""
        # S4.4: Lemma 3.1 on the grid lambda in [0.1, 1000]: Spearman(SR score, margin) and the pairwise bound for SR_max.
        s_rho4 = {name: [] for name in ["SR_max", "SR_doctor", "SR_ent"]}
        for lam in s_lam_grid4:
            for name in s_rho4:
                s_rho4[name].append(float(spearmanr(s_key_for_rc4(s_keys4[lam], name), s_margin4)[0]))

        s_i1 = int(np.where(s_lam_grid4 == 1.0)[0][0])
        s_tol_rho = 1e-4  # from the bound: disagreement only inside a margin window of width log(K-1)/lambda (about 1e-3 at lambda=1000)
        for name in s_rho4:
            check(f"S4.4-{name}", s_rho4[name][-1] >= 1 - s_tol_rho and s_rho4[name][-1] >= s_rho4[name][s_i1],
                  f"Spearman with margin: lambda=1 -> {s_rho4[name][s_i1]:.6f}, lambda=1000 -> {s_rho4[name][-1]:.6f} (need >= {1 - s_tol_rho} and no lower than at lambda=1)")

        # Pairwise bound for SR_max on a 600-row subsample: pairs with lambda*(gap in margin) > log(K-1) must agree in order.
        s_sub4 = np.random.default_rng(RNG_SEED + 404).choice(len(s_margin4), 600, replace=False)
        s_viol_total, s_pairs_total = 0, 0
        for lam in s_lam_grid4:
            s_key_sub = -s_keys4[lam]["SR_max"][s_sub4]
            s_dm = s_margin4[s_sub4][:, None] - s_margin4[s_sub4][None, :]
            s_mask = lam * s_dm > np.log(s_K4 - 1)
            s_dk = s_key_sub[:, None] - s_key_sub[None, :]
            s_viol_total += int(np.sum(s_mask & (s_dk <= 0)))
            s_pairs_total += int(np.sum(s_mask))
        check("S4.4-bound", s_viol_total == 0 and s_pairs_total > 10_000,
              f"pairwise bound lambda*(d2-d2') > log(K-1) => same SR_max order: {s_viol_total} violations among {s_pairs_total} qualifying pairs over the grid")

        # Largest rank displacement (in positions out of n) between each SR ordering and the margin ordering, at lambda=1 and lambda=100.
        def s_max_displacement(key, ref):
            return int(np.max(np.abs(np.argsort(np.argsort(key)) - np.argsort(np.argsort(ref)))))
        s_disp = {lam: {name: s_max_displacement(s_key_for_rc4(s_keys4[lam], name), s_margin4) for name in s_rho4} for lam in (1.0, 100.0)}
        print(f"Largest rank displacement vs margin ordering (n={len(s_margin4)} rows): lambda=1 {s_disp[1.0]}, lambda=100 {s_disp[100.0]}")
        check("S4.4-disp", all(s_disp[100.0][nm] <= s_disp[1.0][nm] for nm in s_rho4),
              "largest rank displacement does not grow from lambda=1 to lambda=100 for any SR score")

        print("Spearman correlation of SR score with confidence margin:")
        print("  lambda     " + " ".join(f"{l:9g}" for l in s_lam_grid4))
        for name in s_rho4:
            print(f"  {name:10s} " + " ".join(f"{r:9.6f}" for r in s_rho4[name]))
        """),

        # ─────────────────────────────────────────────────────────────────────────────────
        # Plot 2: SR and margin AURC across scales
        # ─────────────────────────────────────────────────────────────────────────────────

        code("""
        # S4.5: AURC and rank disagreement versus logit scale (from S4.3-S4.4; K=4 toy, n=2000 test rows).
        plt.figure(figsize=(11, 5.2))
        plt.subplot(1, 2, 1)
        plt.plot(s_lam_grid4, s_aurc_sr4["SR_max"], marker="o", label="SR_max", linewidth=2.2, color=PALETTE["clean"])
        plt.plot(s_lam_grid4, s_aurc_sr4["SR_doctor"], marker="^", label="SR_doctor", linewidth=2.2, color=PALETTE["acoustic"])
        plt.plot(s_lam_grid4, s_aurc_sr4["SR_ent"], marker="d", label="SR_ent", linewidth=2.2, color=PALETTE["shifted"])
        plt.plot(s_lam_grid4, s_aurc_margin4, marker="s", label="Margin (conf-M)", linewidth=2.2, color=PALETTE["frozen"], linestyle="--")
        plt.xscale("log")
        plt.xlabel("Logit scale λ (log axis)", fontsize=12)
        plt.ylabel("AURC (lower is better)", fontsize=12)
        plt.title("AURC versus logit scale", fontsize=13, fontweight="bold")
        plt.legend(fontsize=10)

        plt.subplot(1, 2, 2)
        s_floor4 = 1e-7
        for name, marker, colour in [("SR_max", "o", "clean"), ("SR_doctor", "^", "acoustic"), ("SR_ent", "d", "shifted")]:
            plt.plot(s_lam_grid4, np.maximum(1 - np.array(s_rho4[name]), s_floor4), marker=marker, label=name, linewidth=2.2, color=PALETTE[colour])
        plt.axhline(s_floor4, color=PALETTE["reference"], linestyle=":", linewidth=1.5, label="floor (identical ranking)")
        plt.xscale("log")
        plt.yscale("log")
        plt.xlabel("Logit scale λ (log axis)", fontsize=12)
        plt.ylabel("1 − Spearman(SR, margin) (log axis)", fontsize=12)
        plt.title("Lemma 3.1: rank disagreement", fontsize=13, fontweight="bold")
        plt.legend(fontsize=9)
        plt.tight_layout()
        plt.show()

        s_first_floor4 = {name: next((float(l) for l, r in zip(s_lam_grid4, s_rho4[name]) if 1 - r <= s_floor4), None) for name in s_rho4}
        print("Smallest grid scale at which the SR ordering equals the margin ordering (None = not reached on the grid):", s_first_floor4)
        """),

        md("""
        ### How to read this chart

        The figure has two panels that share the same logit-scale axis (logarithmic, from 0.1 to 1000), computed on one fixed classifier and one fixed test set.

        **Left panel (AURC versus scale).** The y-axis is AURC, so lower is better. The three solid curves are the SR scores; the dashed line is the confidence margin, which is flat by construction because scaling the margin does not change its ordering. A solid curve that sits above the dashed line at some scale means that the SR score keeps errors out of the retained set less effectively than the margin at that scale; a curve below it means the reverse. Whether SR is above or below the margin at moderate scales is a property of this classifier and this draw, so read the printed table rather than assuming a sign. What must hold for every draw is that the SR curves are not identical across scales while the margin line is.

        **Right panel (rank disagreement).** The y-axis is one minus the Spearman rank correlation between an SR score and the margin, on a logarithmic axis, so lower means closer agreement with the margin ordering. The dotted floor marks a point where the two orderings are identical on this test set (a value of exactly one minus one cannot be drawn on a log axis, so it is clipped to the floor). Lemma 3.1 predicts curves that fall toward the floor as the scale grows; a curve that rose at large scale would contradict it.

        **Takeaway.** SR scores rank examples differently at different logit scales, so an SR-based RC curve depends on how the logits happen to be scaled, for instance by temperature scaling or training length. The margin does not. At sufficiently large scale the SR orderings collapse onto the margin ordering, so the two families differ only at the moderate scales that trained networks typically occupy. That last sentence is a statement about this toy; the paper's evidence for real networks is in its experiments (L pp.10-14).
        """),

        md("""
        ### Temperature scaling as a global rescale (derived here)

        Dividing all logits by a fitted temperature $T$ is the same operation as multiplying them by $\\lambda = 1/T$. It therefore leaves the predicted class unchanged and leaves the margin ordering unchanged, but it can change every SR ordering. The paper cites Zhu et al. (2022) as reporting that recent calibration methods may degrade selective-classification performance (L p.4); that is a cited claim, and this toy does not test it. What the toy can test is the mechanism: a calibration step that is harmless for accuracy can still move the RC curve of an SR score.

        The temperature is fitted on a **separate calibration set** by minimizing negative log-likelihood, using the project's `temperature_scale`. For the fit to mean anything the training data must not be separable: with a separable classifier the likelihood keeps improving as $T \\to 0$, and the fitted $T$ is an artefact of the optimizer's bounds. The toy in S4.1 has overlapping classes for that reason, and the first check below asserts that the fitted $T$ is neither $1$ nor at a boundary.

        The paired bootstrap used below resamples **rows** of the test set once per replicate and recomputes both AURCs on the same resample, so the interval for a difference of AURCs reflects the shared rows rather than treating the two scores as independent.
        """),

        code("""
        # S4.6: Temperature fit on the calibration set, applied to the test set; paired bootstrap of AURC differences (B=400, seed RNG_SEED+406).
        from src.metrics import nll

        def s_softmax4(z):
            a = z - z.max(axis=1, keepdims=True)
            p = np.exp(a)
            return p / p.sum(axis=1, keepdims=True)

        def s_aurc_of(score, correct):
            return aurc(*risk_coverage_curve(score, correct))

        def s_paired_boot(score_a, score_b, correct, B=400, seed=0):
            # Bootstrap rows jointly; returns (point, lower, upper) for AURC(score_a) - AURC(score_b) on the same resample.
            rng = np.random.default_rng(seed)
            n = len(correct)
            diffs = np.empty(B)
            for b in range(B):
                idx = rng.integers(0, n, n)
                diffs[b] = s_aurc_of(score_a[idx], correct[idx]) - s_aurc_of(score_b[idx], correct[idx])
            lo, hi = np.quantile(diffs, [0.025, 0.975])
            return s_aurc_of(score_a, correct) - s_aurc_of(score_b, correct), float(lo), float(hi)

        s_T4 = temperature_scale(s_zca4, s_yca4)
        s_nll_1 = nll(s_softmax4(s_zte4), s_yte4)
        s_nll_T = nll(s_softmax4(s_zte4 / s_T4), s_yte4)
        check("S4.6a", 1.05 < s_T4 < 20.0 and s_nll_T < s_nll_1,
              f"fitted T={s_T4:.3f} is interior (not 1, not at the search bounds) and held-out test NLL improves {s_nll_1:.4f} -> {s_nll_T:.4f}")

        s_zte4_T = s_zte4 / s_T4
        s_margin_T = s_gaps4(s_zte4_T)[:, 0]
        s_pred_same = np.array_equal(np.argmax(s_zte4, axis=1), np.argmax(s_zte4_T, axis=1))
        s_risk_m1 = risk_coverage_curve(s_margin4, s_correct4)[1]
        s_risk_mT = risk_coverage_curve(s_margin_T, s_correct4)[1]
        s_margin_T_dev = float(np.max(np.abs(s_risk_m1 - s_risk_mT)))
        check("S4.6b", s_pred_same and s_margin_T_dev < 1e-12,
              f"predictions identical for every test row: {s_pred_same}; margin RC curve before/after T differs by {s_margin_T_dev:.1e} (< 1e-12)")

        s_srmax_1 = -s_log_keys4(s_zte4, 1.0)["SR_max"]
        s_srmax_T = -s_log_keys4(s_zte4, 1.0 / s_T4)["SR_max"]
        s_rho_T = float(spearmanr(s_srmax_1, s_srmax_T)[0])
        s_risk_s1 = risk_coverage_curve(s_srmax_1, s_correct4)[1]
        s_risk_sT = risk_coverage_curve(s_srmax_T, s_correct4)[1]
        s_srT_dev = float(np.max(np.abs(s_risk_s1 - s_risk_sT)))
        check("S4.7", s_rho_T < 1 - 1e-9 and s_srT_dev > 0.0,
              f"SR_max ordering changes under T: Spearman(before, after) = {s_rho_T:.6f}; max |risk diff| along the RC curve = {s_srT_dev:.4f}")

        s_d1 = s_paired_boot(s_srmax_T, s_srmax_1, s_correct4, seed=RNG_SEED + 406)
        s_d2 = s_paired_boot(s_margin4, s_srmax_T, s_correct4, seed=RNG_SEED + 407)
        note("S4.8a", f"AURC(SR_max after T) - AURC(SR_max before): {s_d1[0]:+.4f}, paired-bootstrap 95% CI [{s_d1[1]:+.4f}, {s_d1[2]:+.4f}]"
             + (" (CI excludes 0)" if s_d1[1] > 0 or s_d1[2] < 0 else " (CI contains 0: no detectable AURC change on this toy)"))
        note("S4.8b", f"AURC(margin) - AURC(SR_max after T): {s_d2[0]:+.4f}, paired-bootstrap 95% CI [{s_d2[1]:+.4f}, {s_d2[2]:+.4f}]"
             + (" (CI excludes 0)" if s_d2[1] > 0 or s_d2[2] < 0 else " (CI contains 0)"))
        """),

        # ─────────────────────────────────────────────────────────────────────────────────
        # Section 4b: Covariate shift and novel-class inputs (derived here)
        # ─────────────────────────────────────────────────────────────────────────────────

        md("""
        ## 4b. Selection under covariate shift and novel-class inputs (derived here; the paper's own experiments are vision and text, L)

        The paper distinguishes three kinds of prediction error (L p.2): **Type A** errors on in-distribution inputs, **Type B** errors on inputs whose label is *not in the classifier's label set* (a novel label, so the prediction is always wrong), and **Type C** errors on covariate-shifted inputs that keep the same label set. Its generalization of selective classification to shifted data, Eq. (8), re-defines coverage and selection risk under the shifted distribution $D'$ and assumes that no outliers are present (L p.4). Our toy makes the same three types concrete and then goes one step beyond the paper's formal setting by also pooling Type B rows into the RC curve, to see what an outlier does to each score.

        **Paraphrase of Liang et al. Eq. (8); L p.4.** Under $D'$ the coverage is $\\varphi' = \\mathbb{E}_{D'}[g(x)]$ and the selection risk is $R' = \\mathbb{E}_{D'}[\\ell(f(x), y)\\, g(x)] / \\varphi'$, with the assumption that $D'$ contains no outliers.

        The paper reports (L pp.12-14) that margin scores are best or comparable to SR scores, that OOD-detection scores (it lists RL_max, Energy, KNN and ViM) perform poorly for selective classification, and that on iWildCam (images) and Amazon (text) the margin scores are only on par with SR scores. We test a narrow version of that comparison on a toy with $K = 3$ known classes, using only the two OOD-style scores that can be computed from logits alone: **RL_max** (the largest logit) and **Energy** ($\\log \\sum_k e^{z_k}$, larger meaning more in-distribution). KNN and ViM need training features and are **not run**, so nothing below says anything about them.

        ### Toy data-generating process

        Three known classes sit at the corners of an equilateral triangle of circumradius 2 in the plane. A multinomial logistic regression is fit on 300 in-distribution rows. Test rows are then drawn independently of the training rows:

        1. **Type A:** same distribution as training (isotropic noise with standard deviation 1).
        2. **Type C:** same class centres and label set, but the noise standard deviation is inflated to 1.8.
        3. **Type B:** two novel clusters with labels 3 and 4, which the classifier has never seen: one at the origin (all three classes equally near) and one on the boundary between two known classes.

        By construction every Type B prediction is wrong, so the risk of any selector on Type B rows alone is 1 at every coverage. That case carries no information about the ranking; it only pins down the arithmetic of the AURC integral.
        """),

        code("""
        # S4b.1: K=3 toy with Type A (in-dist), Type C (noise x1.8) and Type B (novel labels 3, 4). n_train=300, A=900, C=900, B=600. Seed RNG_SEED+410.
        from sklearn.metrics import roc_auc_score
        s_rng4b = np.random.default_rng(RNG_SEED + 410)
        s_ang4b = np.array([0.0, 2 * np.pi / 3, 4 * np.pi / 3])
        s_cen4b = 2.0 * np.column_stack([np.cos(s_ang4b), np.sin(s_ang4b)])

        def s_known4b(n, sd):
            y = s_rng4b.integers(0, 3, n)
            return s_cen4b[y] + sd * s_rng4b.standard_normal((n, 2)), y

        s_Xtr4b, s_ytr4b = s_known4b(300, 1.0)
        s_XA, s_yA = s_known4b(900, 1.0)
        s_XC, s_yC = s_known4b(900, 1.8)
        s_nov_centres = np.array([[0.0, 0.0], [1.5, 2.6]])
        s_yB = s_rng4b.integers(3, 5, 600)
        s_XB = s_nov_centres[s_yB - 3] + 0.8 * s_rng4b.standard_normal((600, 2))

        s_clf4b = LogisticRegression(C=1.0, max_iter=5000).fit(s_Xtr4b, s_ytr4b)
        s_Z4b = {k: s_clf4b.decision_function(X) for k, X in [("A", s_XA), ("C", s_XC), ("B", s_XB)]}
        s_pred4b = {k: s_clf4b.predict(X) for k, X in [("A", s_XA), ("C", s_XC), ("B", s_XB)]}
        s_y4b = {"A": s_yA, "C": s_yC, "B": s_yB}
        s_cor4b = {k: s_pred4b[k] == s_y4b[k] for k in s_Z4b}

        check("S4b.1a", set(np.unique(s_yB)) == {3, 4} and not set(np.unique(s_ytr4b)) & {3, 4} and list(s_clf4b.classes_) == [0, 1, 2],
              f"Type B labels {sorted(int(v) for v in np.unique(s_yB))} are absent from the training labels {[int(v) for v in s_clf4b.classes_]}")
        check("S4b.1b", not s_cor4b["B"].any(),
              f"every Type B prediction is wrong (novel label): error rate {1 - s_cor4b['B'].mean():.3f}")
        print(f"Type A error {1 - s_cor4b['A'].mean():.3f}, Type C error {1 - s_cor4b['C'].mean():.3f}, Type B error {1 - s_cor4b['B'].mean():.3f}")
        print(f"Rows: A={len(s_yA)}, C={len(s_yC)}, B={len(s_yB)}; training rows are independent draws, so no test row was used for fitting.")
        """),

        code("""
        # S4b.2: Scores computed from logits. SR_max, SR_doctor, SR_ent (Eq. 7), conf-M margin, RL_max, Energy.
        def s_scores4b(z):
            zs = -np.sort(-z, axis=1)
            p = s_softmax4(z)
            return {"SR_max": p.max(axis=1), "SR_doctor": 1.0 - 1.0 / (p ** 2).sum(axis=1), "SR_ent": (p * np.log(np.clip(p, 1e-300, 1))).sum(axis=1),
                    "margin": zs[:, 0] - zs[:, 1], "RL_max": zs[:, 0], "Energy": logsumexp(z, axis=1)}

        s_S4b = {k: s_scores4b(Z) for k, Z in s_Z4b.items()}
        s_names4b = list(s_S4b["A"].keys())

        # Regime-by-regime AURC for two scores (SR_max and margin).
        s_reg_aurc = {}
        for s_k in ["A", "C", "B"]:
            for s_nm in ["SR_max", "margin"]:
                s_reg_aurc[(s_k, s_nm)] = s_aurc_of(s_S4b[s_k][s_nm], s_cor4b[s_k])

        print("AURC by regime (rows of one type only):")
        for s_k, s_lab in [("A", "Type A"), ("C", "Type C"), ("B", "Type B")]:
            print(f"  {s_lab}: SR_max {s_reg_aurc[(s_k, 'SR_max')]:.4f}   margin {s_reg_aurc[(s_k, 'margin')]:.4f}   base error {1 - s_cor4b[s_k].mean():.4f}")

        s_nB = len(s_yB)
        check("S4b.2a", all(abs(s_reg_aurc[("B", nm)] - (1 - 1 / s_nB)) < 1e-12 for nm in ["SR_max", "margin"]),
              f"Type B AURC = 1 - 1/n = {1 - 1 / s_nB:.6f} exactly for both scores: risk is 1 at every coverage and the trapezoid starts at coverage 1/n")
        check("S4b.2b", all(s_reg_aurc[("C", nm)] > s_reg_aurc[("A", nm)] for nm in ["SR_max", "margin"]),
              f"covariate shift raises AURC for both scores: A {s_reg_aurc[('A', 'SR_max')]:.4f} -> C {s_reg_aurc[('C', 'SR_max')]:.4f} (SR_max), "
              f"A {s_reg_aurc[('A', 'margin')]:.4f} -> C {s_reg_aurc[('C', 'margin')]:.4f} (margin)")
        """),

        code("""
        # S4b.3: OOD detection (known A+C vs novel B) and selective classification on known rows, for six logit-based scores.
        # Paired bootstrap B=400 on the known rows (n=1800), seeds RNG_SEED+430..432.
        s_zk = np.vstack([s_Z4b["A"], s_Z4b["C"]])
        s_ck = np.hstack([s_cor4b["A"], s_cor4b["C"]])
        s_Sk = s_scores4b(s_zk)
        s_Sb = s_S4b["B"]
        s_n1, s_n2 = len(s_ck), len(s_yB)

        def s_hanley_se(auc, n1, n2):
            q1, q2 = auc / (2 - auc), 2 * auc ** 2 / (1 + auc)
            return float(np.sqrt((auc * (1 - auc) + (n1 - 1) * (q1 - auc ** 2) + (n2 - 1) * (q2 - auc ** 2)) / (n1 * n2)))

        s_tab4b = {}
        print(f"{'score':10s} {'AUROC(ID vs novel)':>19s} {'AURC known-only':>16s} {'AURC pooled A+C+B':>18s}")
        for s_nm in s_names4b:
            s_auc = roc_auc_score(np.r_[np.ones(s_n1), np.zeros(s_n2)], np.r_[s_Sk[s_nm], s_Sb[s_nm]])
            s_ak = s_aurc_of(s_Sk[s_nm], s_ck)
            s_ap = s_aurc_of(np.r_[s_Sk[s_nm], s_Sb[s_nm]], np.r_[s_ck, s_cor4b["B"]])
            s_tab4b[s_nm] = (s_auc, s_ak, s_ap)
            print(f"{s_nm:10s} {s_auc:19.3f} {s_ak:16.4f} {s_ap:18.4f}")

        check("S4b.3a", all(v[0] - 0.5 > 3 * s_hanley_se(v[0], s_n1, s_n2) for v in s_tab4b.values()),
              "every score detects the novel rows above chance by > 3 Hanley-McNeil SE (AUROC min "
              f"{min(v[0] for v in s_tab4b.values()):.3f}, SE about {s_hanley_se(min(v[0] for v in s_tab4b.values()), s_n1, s_n2):.3f})")

        s_b_energy = s_paired_boot(s_Sk["margin"], s_Sk["Energy"], s_ck, seed=RNG_SEED + 430)
        s_b_rlmax = s_paired_boot(s_Sk["margin"], s_Sk["RL_max"], s_ck, seed=RNG_SEED + 431)
        s_b_srmax = s_paired_boot(s_Sk["margin"], s_Sk["SR_max"], s_ck, seed=RNG_SEED + 432)
        check("S4b.3b", s_b_energy[2] < 0,
              f"known-only AURC(margin) - AURC(Energy) = {s_b_energy[0]:+.4f}, paired 95% CI [{s_b_energy[1]:+.4f}, {s_b_energy[2]:+.4f}] (upper bound < 0)")
        check("S4b.3c", s_b_rlmax[2] < 0,
              f"known-only AURC(margin) - AURC(RL_max) = {s_b_rlmax[0]:+.4f}, paired 95% CI [{s_b_rlmax[1]:+.4f}, {s_b_rlmax[2]:+.4f}] (upper bound < 0)")
        note("S4b.3d", f"known-only AURC(margin) - AURC(SR_max) = {s_b_srmax[0]:+.4f}, paired 95% CI [{s_b_srmax[1]:+.4f}, {s_b_srmax[2]:+.4f}]"
             + (" (CI excludes 0)" if s_b_srmax[1] > 0 or s_b_srmax[2] < 0 else " (CI contains 0: margin and SR_max are not separable on this toy)"))

        s_auc_vec = np.array([s_tab4b[n][0] for n in s_names4b])
        s_aurc_vec = np.array([s_tab4b[n][1] for n in s_names4b])
        s_rank_corr = float(spearmanr(s_auc_vec, -s_aurc_vec)[0])
        note("S4b.3e", f"across the six scores, Spearman(AUROC for novel detection, -AURC on known rows) = {s_rank_corr:+.3f}; "
             f"best novelty detector: {s_names4b[int(np.argmax(s_auc_vec))]}, best selective classifier on known rows: {s_names4b[int(np.argmin(s_aurc_vec))]}")
        """),

        # ─────────────────────────────────────────────────────────────────────────────────
        # Plot 3: RC curves per score on known rows, and AUROC versus AURC
        # ─────────────────────────────────────────────────────────────────────────────────

        code("""
        # S4b.4: Left: known-only RC curves for the six scores. Right: novelty-detection AUROC against known-only AURC (from S4b.3).
        s_colours4b = dict(zip(s_names4b, ["clean", "acoustic", "mask", "frozen", "shifted", "adapted"]))
        plt.figure(figsize=(11, 5.2))
        plt.subplot(1, 2, 1)
        for s_nm in s_names4b:
            s_cov, s_risk = risk_coverage_curve(s_Sk[s_nm], s_ck)
            plt.plot(s_cov, s_risk, label=f"{s_nm} (AURC={s_tab4b[s_nm][1]:.4f})", color=PALETTE[s_colours4b[s_nm]], linewidth=1.8,
                     linestyle="--" if s_nm in ("RL_max", "Energy") else "-")
        plt.axhline(1 - s_ck.mean(), color=PALETTE["reference"], linestyle=":", linewidth=1.5, label="base error (coverage 1)")
        plt.xlabel("Coverage φ", fontsize=12)
        plt.ylabel("Selection risk R (known rows: Type A + C)", fontsize=12)
        plt.title("Known-label rows: six selectors", fontsize=13, fontweight="bold")
        plt.legend(fontsize=8, loc="upper left")
        plt.xlim(0, 1)

        plt.subplot(1, 2, 2)
        for s_nm in s_names4b:
            plt.scatter(s_tab4b[s_nm][0], s_tab4b[s_nm][1], s=90, color=PALETTE[s_colours4b[s_nm]], zorder=3)
            if s_nm in ("RL_max", "Energy"):
                plt.annotate(s_nm, (s_tab4b[s_nm][0], s_tab4b[s_nm][1]), textcoords="offset points", xytext=(6, 5), fontsize=9)
        s_grp = ["SR_max", "SR_doctor", "SR_ent", "margin"]
        plt.annotate("SR_max, SR_doctor, SR_ent, margin (overlapping)", (np.mean([s_tab4b[n][0] for n in s_grp]), np.mean([s_tab4b[n][1] for n in s_grp])),
                     textcoords="offset points", xytext=(-150, 25), fontsize=9, arrowprops={"arrowstyle": "-", "color": PALETTE["reference"]})
        plt.xlabel("AUROC: known vs novel rows (higher = better novelty detector)", fontsize=11)
        plt.ylabel("AURC on known rows (lower = better selector)", fontsize=11)
        plt.title("Novelty detection vs selection", fontsize=13, fontweight="bold")
        plt.tight_layout()
        plt.show()

        s_pooled_best = s_names4b[int(np.argmin([s_tab4b[n][2] for n in s_names4b]))]
        print(f"Best pooled-AURC score when Type B rows are mixed in: {s_pooled_best}; best on known rows only: {s_names4b[int(np.argmin(s_aurc_vec))]}.")
        """),

        md("""
        ### How to read this chart

        Both panels use only the two kinds of rows that share the classifier's label set (Type A and Type C pooled), which is the setting of Eq. (8) in the paper, except in the right panel where the novelty-detection axis uses the Type B rows as well.

        **Left panel (RC curves).** The x-axis is coverage and the y-axis is the error rate among retained rows. Each line is one score; the SR family and the margin are solid, and the two OOD-style scores (RL_max and Energy) are dashed. The dotted horizontal line is the error rate at full coverage, where every selector must end. A line that stays lower at a given coverage keeps more errors out of the retained set; lines that nearly overlap are scores that rank the rows almost identically; the printed AURCs and the paired intervals say whether any visible gap is larger than resampling noise.

        **Right panel (novelty detection versus selection).** Each point is one score. The x-axis measures how well the score separates known rows from novel rows (further right is better at spotting novelty); the y-axis is the AURC on known rows (lower is a better selective classifier). If good novelty detectors were poor selectors, the points would slope upward to the right. If the two abilities go together, they slope downward.

        **Takeaway.** The printed checks decide what this toy supports, not the eye. The paired-bootstrap intervals on AURC differences are the evidence about RL_max and Energy versus the margin; the printed rank correlation summarizes the right panel. Anything the paper says about KNN and ViM is untouched by this toy because those scores were not computed. A toy in the plane with a linear classifier is far from the deep vision and text models of the paper's experiments (L pp.10-14), so agreement or disagreement here is weak evidence either way.
        """),

        code("""
        # S4b.5: Extension beyond Eq. (8): pool Type B rows with the known rows. What share of the retained rows is novel at coverage 0.5?
        # Pooled set: 1800 known + 600 novel rows; random selection would retain a novel share of 600/2400. Paired bootstrap B=400, seeds RNG_SEED+440..441.
        s_c_pool = np.r_[s_ck, s_cor4b["B"]]
        s_is_novel = np.r_[np.zeros(s_n1, dtype=bool), np.ones(s_n2, dtype=bool)]
        s_share0 = s_n2 / (s_n1 + s_n2)
        s_k_half = (s_n1 + s_n2) // 2
        s_share = {}
        for s_nm in s_names4b:
            s_pool_score = np.r_[s_Sk[s_nm], s_Sb[s_nm]]
            s_top = np.argsort(-s_pool_score, kind="stable")[:s_k_half]
            s_share[s_nm] = float(np.mean(s_is_novel[s_top]))
        s_se_share = float(np.sqrt(s_share0 * (1 - s_share0) / s_k_half))

        print(f"Share of retained rows that are Type B at coverage 0.5 (random selection: {s_share0:.3f}, binomial SE {s_se_share:.3f}):")
        for s_nm in s_names4b:
            print(f"  {s_nm:10s} {s_share[s_nm]:.3f}")

        check("S4b.5a", all(v < s_share0 - 3 * s_se_share for v in s_share.values()),
              f"every score retains fewer novel rows than random selection by more than 3 SE (largest share {max(s_share.values()):.3f} < {s_share0 - 3 * s_se_share:.3f})")

        s_pool = {nm: np.r_[s_Sk[nm], s_Sb[nm]] for nm in s_names4b}
        s_p_energy = s_paired_boot(s_pool["margin"], s_pool["Energy"], s_c_pool, seed=RNG_SEED + 440)
        s_p_srmax = s_paired_boot(s_pool["margin"], s_pool["SR_max"], s_c_pool, seed=RNG_SEED + 441)
        check("S4b.5b", s_p_energy[2] < 0,
              f"pooled AURC(margin) - AURC(Energy) = {s_p_energy[0]:+.4f}, paired 95% CI [{s_p_energy[1]:+.4f}, {s_p_energy[2]:+.4f}] (upper bound < 0)")
        note("S4b.5c", f"pooled AURC(margin) - AURC(SR_max) = {s_p_srmax[0]:+.4f}, paired 95% CI [{s_p_srmax[1]:+.4f}, {s_p_srmax[2]:+.4f}]"
             + (" (CI excludes 0)" if s_p_srmax[1] > 0 or s_p_srmax[2] < 0 else " (CI contains 0)"))
        """),

        code("""
        # S4b.6: Does covariate shift degrade selection? Unpaired bootstrap (B=400) of AURC(Type C) - AURC(Type A) per score, seed RNG_SEED+501.
        def s_unpaired_boot(score_a, cor_a, score_b, cor_b, B=400, seed=0):
            rng = np.random.default_rng(seed)
            diffs = np.empty(B)
            for b in range(B):
                ia = rng.integers(0, len(cor_a), len(cor_a))
                ib = rng.integers(0, len(cor_b), len(cor_b))
                diffs[b] = s_aurc_of(score_a[ia], cor_a[ia]) - s_aurc_of(score_b[ib], cor_b[ib])
            lo, hi = np.quantile(diffs, [0.025, 0.975])
            return s_aurc_of(score_a, cor_a) - s_aurc_of(score_b, cor_b), float(lo), float(hi)

        s_shift_ci = {}
        for s_i, s_nm in enumerate(["SR_max", "margin"]):
            s_shift_ci[s_nm] = s_unpaired_boot(s_S4b["C"][s_nm], s_cor4b["C"], s_S4b["A"][s_nm], s_cor4b["A"], seed=RNG_SEED + 501 + s_i)
            print(f"AURC(C) - AURC(A) for {s_nm}: {s_shift_ci[s_nm][0]:+.4f}, 95% CI [{s_shift_ci[s_nm][1]:+.4f}, {s_shift_ci[s_nm][2]:+.4f}]")

        s_err_a = np.r_[(~s_cor4b["A"]).astype(float)]
        s_err_c = np.r_[(~s_cor4b["C"]).astype(float)]
        s_rng_err = np.random.default_rng(RNG_SEED + 503)
        s_err_diff = np.array([np.mean(s_err_c[s_rng_err.integers(0, len(s_err_c), len(s_err_c))]) - np.mean(s_err_a[s_rng_err.integers(0, len(s_err_a), len(s_err_a))]) for _ in range(1000)])
        s_err_lo, s_err_hi = np.quantile(s_err_diff, [0.025, 0.975])
        print(f"Error rate (C) - (A): {s_err_c.mean() - s_err_a.mean():+.4f}, 95% CI [{s_err_lo:+.4f}, {s_err_hi:+.4f}]")

        check("S4b.6a", s_err_lo > 0, f"covariate shift raises the error rate: CI lower bound {s_err_lo:+.4f} > 0")
        check("S4b.6b", all(v[1] > 0 for v in s_shift_ci.values()),
              "covariate shift raises AURC for both scores: lower CI bounds " + ", ".join(f"{k} {v[1]:+.4f}" for k, v in s_shift_ci.items()) + " (> 0)")
        """),

        code("""
        # S4b.7: Operating point view (Eq. 3 read backwards): largest coverage whose selection risk is at most 10% on the known rows (Type A + C).
        s_target_risk = 0.10
        s_cov_at_target = {}
        for s_nm in s_names4b:
            s_cov, s_risk = risk_coverage_curve(s_Sk[s_nm], s_ck)
            s_ok_idx = np.where(s_risk <= s_target_risk)[0]
            s_cov_at_target[s_nm] = float(s_cov[s_ok_idx[-1]]) if len(s_ok_idx) else 0.0
        print(f"Largest coverage with selection risk <= {s_target_risk:.2f} on known rows (base error {1 - s_ck.mean():.3f}):")
        for s_nm in s_names4b:
            print(f"  {s_nm:10s} coverage {s_cov_at_target[s_nm]:.3f}")
        check("S4b.7", all(v > 0 for v in s_cov_at_target.values()) and max(s_cov_at_target.values()) < 1.0,
              "every score reaches the risk target at a positive coverage, and none reaches it at full coverage (target below the base error)")
        note("S4b.7n", "this is an in-sample operating point on the test rows themselves; choosing a threshold on separate calibration rows is the "
             "calibration-set question the paper leaves open (L p.6), and is not attempted here")
        """),

        # ─────────────────────────────────────────────────────────────────────────────────
        # Conclusion: What the paper does NOT settle
        # ─────────────────────────────────────────────────────────────────────────────────

        md("""
        ## Conclusion: what the paper leaves open (source: Liang et al., TMLR 2024; L)

        **Paraphrase of Liang et al. Sec. 2.5; L p.6.** The paper states that it does not treat how the calibration set is constructed or how the threshold $\\gamma$ is chosen, and instead evaluates the scores directly on test sets.

        This notebook covered selective classification as a framework (Eqs. 1-3 and 8), the RC curve and its AURC summary, the scale sensitivity of SR scores against the scale invariance of margin scores, and a small toy comparison under covariate shift and novel labels. Every numeric statement about the toys comes from the printed output of the checks above, not from this text.

        The paper's own evidence is in its experiments on ImageNet-based benchmarks, iWildCam (images), Amazon (text) and CIFAR (L pp.10-14). None of it is on audio. What remains open for a speech-commands study:

        1. **Calibration-set construction.** Algorithm 1 (L p.4) picks $\\gamma$ from a small i.i.d. calibration set, and the paper leaves the construction of that set outside its scope. For speech this includes how many utterances are needed, whether the set should include shifted conditions, and whether it should be drawn per speaker or globally. These are open design questions, not results.
        2. **Threshold selection under shift.** A threshold chosen for a coverage or risk target on clean calibration data need not deliver that target on shifted test data. Nothing in the material read for this study establishes how far coverage or risk drift; that must be measured on speech data.
        3. **Audio evidence.** The transfer of the paper's finding that margin scores are competitive with, or better than, SR scores to acoustic shift is a hypothesis of this study. The toy above is consistent with the paper's direction for the two OOD-style scores that were run, on data with no relation to speech.
        4. **Guarantees.** The extract of the paper used here records definitions, an evaluation protocol and empirical comparisons; it records no coverage guarantee and no finite-sample bound. Whether the appendices contain one was not checked, so no statement about guarantees is made either way.
        5. **Novel-label inputs.** Eq. (8) assumes no outliers, so Type B inputs sit outside the formal setting. The toy pooled them anyway; the printed pooled AURCs show how the ranking of scores behaves in that extension, which is a derived-here observation and not a claim about the paper.

        **Next steps for this study:** compute the confidence-margin score and the SR scores from the speech-command model's logits under each acoustic shift, compare their AURC with a paired bootstrap on shared utterances as done above, and treat threshold selection as its own experiment.
        """),

        code("""
        # S5.1: Summary of checks run in this part.
        check_summary()

        print("=" * 70)
        print("Part 2 complete: selective classification and scale sensitivity")
        print("=" * 70)
        print("Sections: 3 (RC curve, AURC, oracle closed form), 4 (SR vs margin, Lemma 3.1 grid, temperature), 4b (Types A/B/C, six scores)")
        print("Figures: RC curves for random/imperfect/oracle scores; AURC and rank disagreement versus scale; six selectors and novelty detection")
        print(f"Toy 4 test set: n={len(s_correct4)}, error {np.mean(~s_correct4):.3f}, fitted T={s_T4:.3f}")
        print(f"Toy 4b known-only AURC, best score: {s_names4b[int(np.argmin(s_aurc_vec))]}; best novelty detector: {s_names4b[int(np.argmax(s_auc_vec))]}")
        print("=" * 70)
        """),
    ]
