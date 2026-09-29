"""Notebook Part 3: Conformal risk control (sections 5-7 and what CRC does not establish)."""
from nbkit import md, code, setup_cell


def cells():
    return [
        # ------------------------------------------------------------------ SECTION 5
        md(r"""## 5. Conformal risk control: setup, estimator and guarantee (source: Angelopoulos et al., arXiv:2208.02814; q2, q3)

This section states the guarantee, derives the estimator, and proves the guarantee with an argument that survives inspection. Paper claims and toy results are kept in separate paragraphs. Every number that comes from a run appears only in a code printout.

**Setup (q2).** A base model $f$ is post-processed into $C_\lambda(x)$ with a scalar $\lambda\in\Lambda$ that encodes conservativeness: larger $\lambda$ gives larger, more conservative outputs such as bigger prediction sets. The per-sample loss is $L_i(\lambda)=\ell(C_\lambda(X_i),Y_i)$ and the $n+1$ losses $L_1,\dots,L_{n+1}$ (calibration points plus one test point) are exchangeable random functions of $\lambda$.

**Assumptions (q2).** (1) exchangeability of the $n+1$ loss functions; (2) each $L_i$ is non-increasing in $\lambda$; (3) each $L_i$ is right-continuous; (4) $\sup_\lambda L_i(\lambda)\le B<\infty$ almost surely; (5) $L_i(\lambda_{\max})\le\alpha$ almost surely, so the target is reachable at the most conservative setting.

**Estimator (q2, q3; Equation 4, Section 1.1).** With $\hat R_n(\lambda)=\frac1n\sum_{i\le n}L_i(\lambda)$,
$$\hat\lambda=\inf\Big\{\lambda:\ \tfrac{n}{n+1}\hat R_n(\lambda)+\tfrac{B}{n+1}\le\alpha\Big\}=\inf\Big\{\lambda:\ \hat R_n(\lambda)\le\tfrac{\alpha(n+1)-B}{n}\Big\},$$
and if the set is empty then $\hat\lambda=\lambda_{\max}$ (q2). It is an infimum over a set, not a minimiser of $|\hat R_n-\alpha|$: the set is upward closed because $\hat R_n$ is non-increasing, so $\hat\lambda$ is the smallest, least conservative, setting that passes the corrected test.

**Guarantee (q2, Theorem 1, Section 2.1).**

> Then E[L_{n+1}(λ̂)] ≤ α. (q2, Theorem 1; hypotheses: exchangeable, non-increasing, right-continuous losses with L_i(λ_max) ≤ α and sup_λ L_i(λ) ≤ B < ∞ almost surely)

The trace also lists a lower bound (Theorem 2 and Proposition 1 in q3): $\mathbb{E}[L_{n+1}(\hat\lambda)]\ge\alpha-\frac{2B}{n+1}$ for losses with continuous jumps, described as tight up to $O(1/n)$ (q3). Section 6 tests both sides of that sandwich.

**Testable and untestable parts (q2).** On a finite held-out set one can compute the mean test loss, and inspect monotonicity, boundedness and right-continuity of the loss. One cannot observe the expectation over the joint draw of calibration and test data from a single split, and exchangeability under acoustic shift cannot be proven from one dataset; q2 states that exchangeability between calibration and test audio is violated under shifts such as noise, reverberation or microphone change.
"""),
        md(r"""### Why the correction terms look the way they do (derived here)

Write the empirical risk over all $n+1$ points as $R_{n+1}(\lambda)=\frac1{n+1}\sum_{i=1}^{n+1}L_i(\lambda)=\frac n{n+1}\hat R_n(\lambda)+\frac1{n+1}L_{n+1}(\lambda)$. The test loss is unknown at calibration time, but boundedness gives $L_{n+1}(\lambda)\le B$, hence
$$R_{n+1}(\lambda)\ \le\ \frac n{n+1}\hat R_n(\lambda)+\frac B{n+1}.\tag{1}$$
The right side is computable from calibration data alone. Requiring it to be at most $\alpha$ is therefore a sufficient condition for the unobservable quantity $R_{n+1}(\lambda)\le\alpha$. That is the entire role of the $\frac n{n+1}$ and $\frac B{n+1}$ terms.

**Special case: indicator loss.** If $L_i(\lambda)=\mathbf 1\{\lambda<s_i\}$ for a score $s_i$ (a prediction set misses its label when the threshold is below the score), then $B=1$ and $n\hat R_n(\lambda)=\#\{i\le n:s_i>\lambda\}$. The condition becomes $\#\{s_i>\lambda\}\le\alpha(n+1)-1$, so with $k=\lfloor\alpha(n+1)\rfloor$ the estimator is the $k$-th largest calibration score (and $\lambda_{\max}$ if $k=0$). This is the familiar split-conformal quantile, and it has an exact expected loss: for continuous scores the test point falls above the $k$-th largest of $n$ exchangeable scores with probability $k/(n+1)$. Hence
$$\mathbb{E}[L_{n+1}(\hat\lambda)]=\frac{\lfloor\alpha(n+1)\rfloor}{n+1}\in\Big(\alpha-\tfrac1{n+1},\ \alpha\Big].\tag{2}$$
Equation (2) is a closed form that the Monte Carlo runs in Section 6 must reproduce, which makes it a sharp test of both the estimator and the prose about the $O(1/n)$ gap. It also shows when the procedure cannot work: if $\alpha(n+1)<B$ the set is empty for every calibration draw and $\hat\lambda=\lambda_{\max}$ always.
"""),
        code(r"""# S5.1-S5.4: estimator implementation, worked example, agreement with src.calibration.crc_threshold
# seeds RNG_SEED+50; worked n=9; 500 random draws each for n=50 (alpha=0.10) and n=10 (alpha=0.05)
from src.calibration import crc_threshold

c_K, c_SIG, c_TEMP = 8, 1.5, 1.5            # classes, signal strength, fixed temperature (T in [0.5, 2])
c_GRID = np.linspace(0.0, 1.0, 2001)         # lambda grid; lambda_max = 1 gives the full label set
c_STEP = c_GRID[1] - c_GRID[0]


def c_curve(js, jw, n, grid=c_GRID):
    # Empirical risk R_n(lambda) on the grid for a loss made of jumps: L_i(lam) = sum_j w_ij * 1{lam < s_ij}
    o = np.argsort(js)
    js, cw = js[o], np.concatenate([[0.0], np.cumsum(jw[o])])
    return (cw[-1] - cw[np.searchsorted(js, grid, side="right")]) / n


def c_lambda_hat(Rg, n, B, alpha, grid=c_GRID):
    # lambda_hat = inf{lam in grid : n/(n+1) R_n(lam) + B/(n+1) <= alpha}; lambda_max if the set is empty
    ok = (n / (n + 1)) * Rg + B / (n + 1) <= alpha + 1e-12
    return grid[-1] if not ok.any() else grid[np.argmax(ok)]


def c_oracle(js, jw, N, alpha, grid=c_GRID):
    # lambda' = inf{lam : R_{n+1}(lam) <= alpha} uses ALL n+1 losses (the proof device, not computable in practice)
    ok = c_curve(js, jw, N, grid) <= alpha + 1e-12
    return grid[-1] if not ok.any() else grid[np.argmax(ok)]


def c_exact_indicator(scores, alpha):
    # closed form of the inf rule for L_i = 1{lam < s_i}: k-th largest score, k = floor(alpha (n+1))
    k = int(np.floor(alpha * (len(scores) + 1) + 1e-9))
    return 1.0 if k == 0 else float(np.sort(scores)[::-1][k - 1])


# worked example: n = 9 scores, alpha = 0.2, B = 1  ->  allowed misses = floor(0.2*10 - 1) = 1
c_ex_scores = np.array([0.05, 0.12, 0.20, 0.31, 0.40, 0.47, 0.55, 0.66, 0.80])
c_ex_grid = np.round(np.linspace(0, 1, 101), 10)
c_ex_R = c_curve(c_ex_scores, np.ones(9), 9, c_ex_grid)
c_ex_lam = c_lambda_hat(c_ex_R, 9, 1.0, 0.2, c_ex_grid)
c_ex_i = int(np.searchsorted(c_ex_grid, 0.66))
print(f"worked example: R_9(0.65)={c_ex_R[c_ex_i - 1]:.4f} -> corrected {0.9 * c_ex_R[c_ex_i - 1] + 0.1:.4f} > 0.2 ; "
      f"R_9(0.66)={c_ex_R[c_ex_i]:.4f} -> corrected {0.9 * c_ex_R[c_ex_i] + 0.1:.4f} <= 0.2 ; lambda_hat={c_ex_lam:.2f}")
tau_ex = crc_threshold(1.0 - c_ex_scores, 0.2)
check("S5.1", abs(c_ex_lam - 0.66) < 1e-9, f"grid inf-rule lambda_hat={c_ex_lam:.4f} vs hand value 0.66 (second largest score)")
check("S5.2", abs((1 - tau_ex) - c_ex_lam) < 1e-9, f"src crc_threshold gives lambda=1-tau={1 - tau_ex:.4f} vs estimator {c_ex_lam:.4f}")

c_rng5 = np.random.default_rng(RNG_SEED + 50)
c_diffs, c_src_gap, c_cons = [], [], []
for _ in range(500):
    s = c_rng5.random(50)
    lam_g = c_lambda_hat(c_curve(s, np.ones(50), 50), 50, 1.0, 0.10)
    lam_e = c_exact_indicator(s, 0.10)
    c_diffs.append(lam_g - lam_e)
    c_src_gap.append(abs(crc_threshold(1 - s, 0.10) - (1 - lam_e)))
c_diffs = np.array(c_diffs)
check("S5.3", c_diffs.min() >= -1e-12 and c_diffs.max() <= c_STEP + 1e-12,
      f"grid estimator minus exact k-th largest in [{c_diffs.min():.2e}, {c_diffs.max():.2e}] (one grid step = {c_STEP:.1e}, never below exact)")
check("S5.4", max(c_src_gap) < 1e-12, f"max |crc_threshold - (1 - exact)| over 500 draws = {max(c_src_gap):.2e}")
c_full = [c_lambda_hat(c_curve(c_rng5.random(10), np.ones(10), 10), 10, 1.0, 0.05) == 1.0 for _ in range(500)]
c_srcsmall = crc_threshold(1 - c_rng5.random(10), 0.05)
check("S5.5", all(c_full), f"n=10, alpha=0.05 (alpha(n+1)={0.05 * 11:.2f} < B=1): lambda_hat = lambda_max in {sum(c_full)}/500 draws")
"""),
        md(r"""### Infeasible-alpha fallback: handling the edge case (R4.1, claims C40 and C41)

**Claim C40 (grounded, source: `src/calibration.py:57-67`).** `crc_threshold` returns exactly `0.0` whenever `alpha * (n + 1) < 1` (i.e., when there are too few calibration points for the requested confidence level to be feasible as a finite-sample order statistic), regardless of the score distribution. The return value is a fallback constant, not derived from the data.

**Claim C41 (grounded, same source).** For `alpha * (n + 1) >= 1`, the existing behavior (order statistic at `index = floor(alpha*(n+1) - 1)`, clamped to `[0, n-1]`) is unchanged — this fix touches only the infeasible branch.

This pair of claims was implicit in the paper's design (Angelopoulos et al., arXiv:2208.02814, §1.1, Equation 4) but the implementation had a bug that returned `sorted(scores)[0]` (the calibration minimum) instead of `0.0` in the infeasible regime. The cell below sweeps across the boundary to show the corrected behavior.
"""),
        code(r"""# R4.1-C40-C41: sweep alpha*(n+1) across the discontinuity at 1.0 using the REAL crc_threshold function
from src.calibration import crc_threshold
import numpy as np

# small toy calibration scores
r4_cal_scores = np.array([0.3, 0.5, 0.7, 0.8, 0.9])  # n=5
r4_n = len(r4_cal_scores)

# sweep alpha*(n+1) from 0.2 to 2.0, crossing the critical threshold at 1.0
r4_sweeps = np.linspace(0.2, 2.0, 37)  # 37 points covers the discontinuity with fine resolution
r4_alphas = r4_sweeps / (r4_n + 1)
r4_thresholds = []

print(f"n = {r4_n}, calibration scores = {r4_cal_scores}")
print(f"Sweeping alpha*(n+1) from {r4_sweeps[0]:.2f} to {r4_sweeps[-1]:.2f}, crossing critical value 1.0")
print(f"\nalpha*(n+1)  alpha      threshold  regime")
print("-" * 50)
for r4_s, r4_a in zip(r4_sweeps, r4_alphas):
    r4_tau = crc_threshold(r4_cal_scores, r4_a)
    r4_thresholds.append(r4_tau)
    r4_regime = "INFEASIBLE (fallback 0.0)" if r4_s < 1.0 else "FEASIBLE (order stat)"
    print(f"{r4_s:6.2f}       {r4_a:8.5f}   {r4_tau:8.5f}   {r4_regime}")

r4_thresholds = np.array(r4_thresholds)
"""),
        code(r"""# R4.1-C40-C41: verify the claims with concrete numeric assertions
# C40: all infeasible cases return exactly 0.0
r4_infeasible_mask = r4_sweeps < 1.0
r4_infeasible_thresholds = r4_thresholds[r4_infeasible_mask]
check("R4.1", np.all(r4_infeasible_thresholds == 0.0),
      f"C40: all {np.sum(r4_infeasible_mask)} infeasible cases (alpha*(n+1) < 1) return exactly 0.0; got {np.unique(r4_infeasible_thresholds)}")

# C41: feasible cases match hand-computed order statistics
r4_feasible_mask = r4_sweeps >= 1.0
r4_feasible_sweeps = r4_sweeps[r4_feasible_mask]
r4_feasible_alphas = r4_alphas[r4_feasible_mask]
r4_sorted_scores = np.sort(r4_cal_scores)
r4_feasible_errors = []
for r4_s, r4_a in zip(r4_feasible_sweeps, r4_feasible_alphas):
    r4_idx_computed = int(np.floor(r4_a * (r4_n + 1) - 1.0 + 1e-12))
    r4_idx_clamped = min(max(r4_idx_computed, 0), r4_n - 1)
    r4_expected = float(r4_sorted_scores[r4_idx_clamped])
    r4_actual = crc_threshold(r4_cal_scores, r4_a)
    r4_error = abs(r4_expected - r4_actual)
    r4_feasible_errors.append(r4_error)
    if r4_error > 1e-12:
        print(f"ERROR at alpha*(n+1)={r4_s}: expected {r4_expected}, got {r4_actual}")

r4_feasible_errors = np.array(r4_feasible_errors)
check("R4.2", np.all(r4_feasible_errors < 1e-12),
      f"C41: all {len(r4_feasible_errors)} feasible cases (alpha*(n+1) >= 1) match hand-computed order statistics; max error = {r4_feasible_errors.max():.2e}")
"""),
        code(r"""# Figure R4.1-a: CRC threshold vs alpha*(n+1) showing the corrected discontinuity at 1.0
fig, ax = plt.subplots(figsize=(10, 5.5))
ax.plot(r4_sweeps, r4_thresholds, color=PALETTE["clean"], linewidth=2, label="crc_threshold (fixed)")
ax.axvline(1.0, color=PALETTE["reference"], linestyle="--", linewidth=1.5, label="alpha*(n+1) = 1 (feasibility boundary)")
ax.scatter([1.0], [0.0], color=PALETTE["shifted"], s=100, marker="o", zorder=5, label="discontinuity at boundary")
ax.set_xlabel("alpha * (n + 1)", fontsize=11)
ax.set_ylabel("CRC threshold (tau)", fontsize=11)
ax.set_title(f"Fig R4.1-a: CRC threshold discontinuity at infeasibility boundary (n={r4_n}, sorted scores={sorted(r4_cal_scores)})", fontsize=12)
ax.grid(True, alpha=0.3)
ax.legend(fontsize=9)
ax.set_xlim(0.1, 2.1)
ax.set_ylim(-0.05, 1.0)
plt.tight_layout()
plt.show()
print(f"Figure shows the corrected curve with a clear jump from 0.0 to the first order statistic at alpha*(n+1)=1.0")
"""),
        md(r"""### How to read Fig R4.1-a

The horizontal axis is `alpha*(n+1)`, the quantity that determines feasibility: if it is less than 1, there are not enough calibration points to support the requested confidence level as a finite-sample order statistic. The vertical axis is the CRC threshold returned by `crc_threshold`.

The curve shows a sharp discontinuity at the boundary `alpha*(n+1) = 1.0`, marked by the grey dashed vertical line and the red circle. To the left of the boundary (infeasible regime), the function returns exactly 0.0 regardless of the calibration scores or the exact value of `alpha*(n+1)` — this is the fallback constant, a safe conservative choice that accepts everything. To the right of the boundary (feasible regime), the threshold jumps to the empirical first-order statistic (the minimum calibration score, 0.3 in this example) and increases monotonically as `alpha*(n+1)` grows. The jump itself is not smooth; it depends on the quantile structure of the calibration data.

The point of this figure is to show that the fix works: a smooth or monotone curve, or one with the fallback value stuck at some nonzero value on the left side, would mean the fix changed nothing observable. Here, the sharp jump is clearly visible.
"""),
        md(r"""### Proof of the guarantee via an oracle threshold (derived here)

An earlier draft claimed that $\hat\lambda$ is a symmetric function of the $n+1$ losses. It is not: $\hat\lambda$ is computed from $L_1,\dots,L_n$ only, and swapping the test point with a calibration point changes it. The correct argument introduces an *oracle* threshold that does use the test point, proves the guarantee for the oracle, and then compares.

Define, using all $n+1$ losses,
$$\lambda'=\inf\{\lambda:\ R_{n+1}(\lambda)\le\alpha\}.$$

**Step 1, symmetry.** $R_{n+1}$ is a sum over all $n+1$ losses, so $\lambda'$ is invariant to permuting them. Swapping loss $i$ with loss $n+1$ leaves $(\lambda',\ \text{joint law})$ unchanged.

**Step 2, the oracle is at most $\hat\lambda$.** By inequality (1), any $\lambda$ that passes the calibration-only test also satisfies $R_{n+1}(\lambda)\le\alpha$. So $\{\lambda:\frac n{n+1}\hat R_n+\frac B{n+1}\le\alpha\}\subseteq\{\lambda:R_{n+1}(\lambda)\le\alpha\}$. The infimum over the smaller set is larger, so $\lambda'\le\hat\lambda$. The set on the right is non-empty because $L_i(\lambda_{\max})\le\alpha$, so $\lambda'$ is well defined; if the left set is empty then $\hat\lambda=\lambda_{\max}\ge\lambda'$ trivially.

**Step 3, monotonicity.** Since $L_{n+1}$ is non-increasing and $\hat\lambda\ge\lambda'$, we get $L_{n+1}(\hat\lambda)\le L_{n+1}(\lambda')$ pathwise.

**Step 4, right-continuity gives attainment.** The infimum defining $\lambda'$ is attained because $R_{n+1}$ is right-continuous, so $R_{n+1}(\lambda')\le\alpha$.

**Step 5, exchangeability.** By Step 1 and exchangeability, $\mathbb{E}[L_i(\lambda')]=\mathbb{E}[L_{n+1}(\lambda')]$ for every $i\le n+1$. Averaging over $i$,
$$\mathbb{E}[L_{n+1}(\lambda')]=\mathbb{E}[R_{n+1}(\lambda')]\le\alpha.$$

**Conclusion.** $\mathbb{E}[L_{n+1}(\hat\lambda)]\le\mathbb{E}[L_{n+1}(\lambda')]\le\alpha$. Boundedness is used only in Step 2, monotonicity only in Step 3, right-continuity in Step 4, exchangeability in Step 5, and achievability at $\lambda_{\max}$ only to make $\lambda'$ exist. The code cell below checks every step numerically, including the fact that the naive symmetry claim about $\hat\lambda$ is false.
"""),
        code(r"""# S5.7-S5.14: numerical audit of the proof steps. Indicator (miscoverage) loss, n=24, alpha=0.10, reps=4000, seed RNG_SEED+51
# Uses the multi-label score DGP of Section 6 (defined here so this cell is self-contained).

def c_draw(rng, R, N, sig=c_SIG, sd=1.0):
    # multi-label toy: each sample has m in {1,2,3} true labels among K; logit = sig*1{true} + N(0, sd^2); p = sigmoid(z / T)
    m = rng.integers(1, 4, size=(R, N))
    ranks = rng.random((R, N, c_K)).argsort(-1).argsort(-1)
    Y = ranks < m[..., None]
    z = np.clip(sig * Y + sd * rng.standard_normal((R, N, c_K)), -8, 8)
    return Y, 1.0 - 1.0 / (1.0 + np.exp(-z / c_TEMP))      # score = 1 - p (small = plausible label)


c_n5, c_a5, c_R5 = 24, 0.10, 4000
c_rng = np.random.default_rng(RNG_SEED + 51)
c_Y5, c_S5 = c_draw(c_rng, c_R5, c_n5 + 1)
c_A5 = np.where(c_Y5, c_S5, -1.0).max(-1)                    # jump location of the miscoverage loss
c_ones_n, c_ones_N = np.ones(c_n5), np.ones(c_n5 + 1)
c_lh, c_lp, c_lh_sw, c_lp_sw, c_Lh, c_Lp, c_Rp = [], [], [], [], [], [], []
for r in range(c_R5):
    a = c_A5[r]
    lh = c_lambda_hat(c_curve(a[:c_n5], c_ones_n, c_n5), c_n5, 1.0, c_a5)
    lp = c_oracle(a, c_ones_N, c_n5 + 1, c_a5)
    a2 = a.copy(); a2[[0, c_n5]] = a2[[c_n5, 0]]              # swap calibration point 0 with the test point
    lh2 = c_lambda_hat(c_curve(a2[:c_n5], c_ones_n, c_n5), c_n5, 1.0, c_a5)
    lp2 = c_oracle(a2, c_ones_N, c_n5 + 1, c_a5)
    c_lh.append(lh); c_lp.append(lp); c_lh_sw.append(lh2); c_lp_sw.append(lp2)
    c_Lh.append(float(a[c_n5] > lh)); c_Lp.append(float(a[c_n5] > lp))
    c_Rp.append(float(np.mean(a > lp)))                      # R_{n+1}(lambda') = mean of the n+1 losses at lambda'
c_lh, c_lp, c_lh_sw, c_lp_sw = map(np.array, (c_lh, c_lp, c_lh_sw, c_lp_sw))
c_Lh, c_Lp, c_Rp = map(np.array, (c_Lh, c_Lp, c_Rp))
c_dd = c_Lp - c_Rp
check("S5.7", np.all(c_lp <= c_lh + 1e-12), f"Step 2: lambda' <= lambda_hat in {int(np.sum(c_lp <= c_lh + 1e-12))}/{c_R5} draws; strict in {int(np.sum(c_lp < c_lh - 1e-12))}")
check("S5.8", np.all(c_Lh <= c_Lp), f"Step 3: L_(n+1)(lambda_hat) <= L_(n+1)(lambda') pathwise in {int(np.sum(c_Lh <= c_Lp))}/{c_R5} draws")
check("S5.9", np.all(c_Rp <= c_a5 + 1e-12), f"Step 4: R_(n+1)(lambda') <= alpha in all draws; max = {c_Rp.max():.4f} vs alpha={c_a5}")
check("S5.10", abs(c_dd.mean()) <= 3 * c_dd.std(ddof=1) / np.sqrt(c_R5),
      f"Step 5: E[L_(n+1)(lambda')] - E[R_(n+1)(lambda')] = {c_dd.mean():+.4f} +/- {c_dd.std(ddof=1) / np.sqrt(c_R5):.4f} (SE); symmetry predicts 0")
c_seh, c_sep = c_Lh.std(ddof=1) / np.sqrt(c_R5), c_Lp.std(ddof=1) / np.sqrt(c_R5)
check("S5.11", c_Lh.mean() <= c_Lp.mean() + 3 * np.hypot(c_seh, c_sep) and c_Lp.mean() <= c_a5 + 3 * c_sep,
      f"E[L(lambda_hat)]={c_Lh.mean():.4f} <= E[L(lambda')]={c_Lp.mean():.4f} <= alpha={c_a5}; closed form (2) = {int(np.floor(c_a5 * (c_n5 + 1))) / (c_n5 + 1):.4f}")
c_changed = float(np.mean(np.abs(c_lh - c_lh_sw) > 1e-12))
check("S5.12", c_changed > 0.0, f"lambda_hat is NOT symmetric: swapping a calibration point with the test point changes it in {c_changed:.1%} of draws")
check("S5.13", np.all(np.abs(c_lp - c_lp_sw) < 1e-12), f"lambda' IS symmetric: unchanged by the swap in {int(np.sum(np.abs(c_lp - c_lp_sw) < 1e-12))}/{c_R5} draws")
c_cf5 = int(np.floor(c_a5 * (c_n5 + 1))) / (c_n5 + 1)
check("S5.14", abs(c_Lh.mean() - c_cf5) <= 3 * c_seh, f"non-vacuity and exactness: E[L(lambda_hat)]={c_Lh.mean():.4f} vs closed form (2) {c_cf5:.4f} (3SE={3 * c_seh:.4f}); a degenerate zero loss would sit far from it")
"""),
        code(r"""# Figure 5.1: oracle threshold versus the calibration-only estimator (data from the S5.7 audit above)
fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
c_jit = np.random.default_rng(RNG_SEED + 52).normal(0, 0.002, c_R5)
ax[0].scatter(c_lp + c_jit, c_lh + c_jit, s=6, alpha=0.35, color=PALETTE["clean"], label="draw")
ax[0].plot([0, 1], [0, 1], color=PALETTE["reference"], ls="--", label="lambda_hat = lambda'")
ax[0].set_xlabel("oracle threshold lambda' (uses test loss)"); ax[0].set_ylabel("estimator lambda_hat (calibration only)")
ax[0].set_xlim(0.4, 0.95); ax[0].set_ylim(0.4, 0.95); ax[0].legend(); ax[0].set_title("Step 2: lambda' <= lambda_hat")
ax[1].hist(c_lh, bins=40, alpha=0.6, color=PALETTE["shifted"], label="lambda_hat")
ax[1].hist(c_lp, bins=40, alpha=0.6, color=PALETTE["clean"], label="lambda'")
ax[1].set_xlabel("threshold (larger = larger sets = more conservative)"); ax[1].set_ylabel("count of draws"); ax[1].legend()
ax[1].set_title("Distribution of the two thresholds")
plt.tight_layout(); plt.show()
print(f"mean lambda_hat = {c_lh.mean():.4f}, mean lambda' = {c_lp.mean():.4f}; mean gap = {np.mean(c_lh - c_lp):.4f}")
"""),
        md(r"""### How to read this chart

The left panel plots each Monte Carlo draw as one point. The horizontal axis is the oracle threshold $\lambda'$, which uses the test loss and is unavailable in practice. The vertical axis is the estimator $\hat\lambda$ computed from calibration data alone. The dashed grey diagonal is $\hat\lambda=\lambda'$. Because larger $\lambda$ means larger prediction sets, a point above the diagonal is a draw where the estimator is more conservative than the oracle, which is the direction the proof needs (Step 2). A point below the diagonal would break the proof; none should appear. Points sitting exactly on the diagonal are draws where the test point did not change the threshold.

The right panel shows the two marginal distributions of thresholds. The distribution of $\hat\lambda$ is shifted toward larger values relative to $\lambda'$, and that shift is the price of not seeing the test point. It is also the source of the gap between the achieved expected loss and $\alpha$ that Section 6 measures. Horizontal stripes and vertical stripes in the scatter come from the discrete set of order statistics that the threshold can equal.

Takeaway: the estimator is a conservative stand-in for a symmetric oracle, and the printed mean gap above quantifies how much conservativeness the stand-in costs at this sample size. What would make this conclusion wrong: any point below the diagonal.
"""),
        # ------------------------------------------------------------------ SECTION 6
        md(r"""## 6. Monte Carlo verification of the bound and its O(1/n) gap (derived here; tests claims from q2, q3)

Section 5 argued that CRC is valid and, by the closed form (2), almost exactly tight for indicator losses. This section tests those claims on a data-generating process where the bound is active: risk close to the target $\alpha$, non-trivial prediction sets, real signal in the scores. An earlier version used labels independent of the features and an unbounded temperature; the sets were always full and the measured risk was zero everywhere, which verifies nothing.

**Toy speech-like DGP.** There are $K=8$ classes. Each sample has a random set of $m\in\{1,2,3\}$ true labels (a multi-label stand-in for keyword sets). For each class, a logit equals a signal strength times the indicator that the class is true, plus standard normal noise; probabilities come from a sigmoid with a fixed temperature in $[0.5,2]$, so scores cannot blow up. The prediction set is $C_\lambda(x)=\{k:\ 1-p_k\le\lambda\}$. At $\lambda=0$ almost nothing is included; at $\lambda_{\max}=1$ every class is included and both losses are zero. Larger $\lambda$ therefore means larger sets and a more conservative predictor.

**Two monotone losses, both with $B=1$.**
1. *Set miscoverage* $L^{\rm mis}(\lambda)=\mathbf 1\{\text{some true label}\notin C_\lambda\}=\mathbf 1\{\lambda<\max_{k\in Y}(1-p_k)\}$. One jump per sample, so equation (2) gives its exact expected loss.
2. *False-negative rate* $L^{\rm fnr}(\lambda)=1-|C_\lambda\cap Y|/|Y|=\frac1{|Y|}\sum_{k\in Y}\mathbf 1\{\lambda<1-p_k\}$. Up to three jumps of size $1/|Y|$ per sample. This is a genuinely different loss: $L^{\rm fnr}\le L^{\rm mis}$ pathwise, so its threshold is never larger, and it has no closed form.

Both are non-increasing and right-continuous in $\lambda$, bounded by $B=1$, and zero at $\lambda_{\max}$, so Theorem 1 applies to both (q2).

**Claims under test.**

> Then E[L_{n+1}(λ̂)] ≤ α. (q2, Theorem 1)

> Theorem 2 & Prop. 1: ... for continuous loss jumps, proven tight up to O(1/n) (q3, claims matrix; the lower bound is $\alpha-2B/(n+1)$)

What would make these wrong: measured mean loss above $\alpha$ by more than three standard errors (upper side), or below $\alpha-2B/(n+1)$ by more than three standard errors (lower side). For miscoverage we test the stronger, exact equation (2).
"""),
        code(r"""# S6.1-S6.6: DGP sanity (signal present, sets grow with lambda, losses monotone, jump form == set form). seed RNG_SEED+60, N=4000
c_rng = np.random.default_rng(RNG_SEED + 60)
c_Yd, c_Sd = c_draw(c_rng, 1, 4000)
c_Yd, c_Sd = c_Yd[0], c_Sd[0]
c_pd = 1 - c_Sd
c_gap_p = c_pd[c_Yd].mean() - c_pd[~c_Yd].mean()
print(f"mean p (true labels) = {c_pd[c_Yd].mean():.3f}, mean p (other classes) = {c_pd[~c_Yd].mean():.3f}, temperature T={c_TEMP}, p range [{c_pd.min():.3f}, {c_pd.max():.3f}]")
check("S6.1", c_gap_p > 0.10, f"signal: true-label probability exceeds other classes by {c_gap_p:.3f} (labels independent of features would give ~0)")
check("S6.2", c_pd.min() > 0.0 and c_pd.max() < 1.0 and 0.5 <= c_TEMP <= 2.0, f"bounded scores: p in [{c_pd.min():.4f}, {c_pd.max():.4f}], T={c_TEMP} within [0.5, 2]")
c_lams = np.linspace(0, 1, 201)
c_mis = np.array([(c_Yd & (c_Sd > l)).any(-1).mean() for l in c_lams])                       # loss from sets, not from jumps
c_fnr = np.array([((c_Yd & (c_Sd > l)).sum(-1) / c_Yd.sum(-1)).mean() for l in c_lams])
c_size = np.array([(c_Sd <= l).sum(-1).mean() for l in c_lams])
check("S6.3", np.all(np.diff(c_mis) <= 1e-12) and np.all(np.diff(c_fnr) <= 1e-12), f"both population risk curves are non-increasing in lambda on 201 grid points (mis {c_mis[0]:.3f}->{c_mis[-1]:.3f}, fnr {c_fnr[0]:.3f}->{c_fnr[-1]:.3f})")
check("S6.4", np.all(np.diff(c_size) >= -1e-12) and abs(c_size[-1] - c_K) < 1e-12, f"mean set size non-decreasing in lambda, from {c_size[0]:.2f} to K={c_size[-1]:.0f}: larger lambda = larger sets")
check("S6.5", np.all(c_fnr <= c_mis + 1e-12) and 0.05 < c_mis[100] < 0.95, f"FNR <= miscoverage at every lambda; at lambda=0.5 miscoverage risk={c_mis[100]:.3f} (non-trivial, not 0 or 1)")
c_a_d = np.where(c_Yd, c_Sd, -1).max(-1)
check("S6.6", all(abs(np.mean(c_a_d > l) - c_mis[i]) < 1e-12 for i, l in enumerate(c_lams)), "jump representation used by the estimator agrees with the set-based miscoverage at all 201 lambdas")
"""),
        code(r"""# S6.7-S6.18: Monte Carlo over n in {24, 54, 104, 204} x alpha in {0.05, 0.10, 0.20}; reps=4000 per cell; seeds RNG_SEED+61+cell
def c_se(x):
    return x.std(ddof=1) / np.sqrt(len(x))


def c_mc_cell(n, alpha, reps, seed, sig_cal=c_SIG, sig_test=None):
    # calibrate both losses on n exchangeable samples, evaluate on one fresh test sample; returns per-rep arrays
    rng = np.random.default_rng(seed)
    Yc, Sc = c_draw(rng, reps, n, sig_cal)
    Yt, St = c_draw(rng, reps, 1, sig_cal if sig_test is None else sig_test)
    aC = np.where(Yc, Sc, -1.0).max(-1)
    wC = Yc / Yc.sum(-1, keepdims=True)
    out = {k: np.empty(reps) for k in ("lamA", "lamB", "LA", "LB", "szA", "szB")}
    one = np.ones(n)
    for r in range(reps):
        lamA = c_lambda_hat(c_curve(aC[r], one, n), n, 1.0, alpha)
        lamB = c_lambda_hat(c_curve(Sc[r][Yc[r]], wC[r][Yc[r]], n), n, 1.0, alpha)
        yt, st = Yt[r, 0], St[r, 0]
        out["lamA"][r], out["lamB"][r] = lamA, lamB
        out["LA"][r] = float(np.any(yt & (st > lamA)))
        out["LB"][r] = (yt & (st > lamB)).sum() / yt.sum()
        out["szA"][r], out["szB"][r] = (st <= lamA).sum(), (st <= lamB).sum()
    return out


c_NS, c_AS, c_REPS = [24, 54, 104, 204], [0.05, 0.10, 0.20], 4000
c_res = {}
for ci, (c_n, c_a) in enumerate([(n, a) for n in c_NS for a in c_AS]):
    c_res[(c_n, c_a)] = c_mc_cell(c_n, c_a, c_REPS, RNG_SEED + 61 + ci)
c_lab = 7
print("n     alpha  closed(2)  E[mis]  SE      E[fnr]  SE      mean lam_A  mean lam_B  size_A  size_B")
for (c_n, c_a), o in c_res.items():
    c_cf = np.floor(c_a * (c_n + 1) + 1e-9) / (c_n + 1)
    print(f"{c_n:<5d} {c_a:<6.2f} {c_cf:.4f}     {o['LA'].mean():.4f}  {c_se(o['LA']):.4f}  {o['LB'].mean():.4f}  {c_se(o['LB']):.4f}  "
          f"{o['lamA'].mean():.4f}      {o['lamB'].mean():.4f}      {o['szA'].mean():.2f}    {o['szB'].mean():.2f}")
for (c_n, c_a), o in c_res.items():
    c_cf = np.floor(c_a * (c_n + 1) + 1e-9) / (c_n + 1)
    c_e, c_s = o["LA"].mean(), max(c_se(o["LA"]), 1e-9)
    check(f"S6.{c_lab}", abs(c_e - c_cf) <= 3 * c_s and c_e <= c_a + 3 * c_s,
          f"miscoverage n={c_n} alpha={c_a}: E={c_e:.4f} vs exact (2)={c_cf:.4f} (3SE={3 * c_s:.4f}); upper bound alpha={c_a}")
    c_lab += 1
"""),
        code(r"""# S6.19-S6.34: false-negative-rate loss (S6.19-S6.30 per cell; S6.31-S6.34 across cells). Same runs as above. Upper bound alpha; lower bound alpha - 2B/(n+1) (q3, Theorem 2)
c_lab = 19
c_ratio = []
for (c_n, c_a), o in c_res.items():
    c_e, c_s = o["LB"].mean(), c_se(o["LB"])
    c_ratio.append(c_e / c_a)
    check(f"S6.{c_lab}", (c_e <= c_a + 3 * c_s) and (c_e >= c_a - 2.0 / (c_n + 1) - 3 * c_s),
          f"FNR n={c_n} alpha={c_a}: E={c_e:.4f} in [{c_a - 2.0 / (c_n + 1) - 3 * c_s:.4f}, {c_a + 3 * c_s:.4f}]")
    c_lab += 1
c_pw = all(np.all(o["lamB"] <= o["lamA"] + 1e-12) for o in c_res.values())
check("S6.31", c_pw, "FNR threshold <= miscoverage threshold in every replicate of every cell (pathwise, because L_fnr <= L_mis)")
c_tie = [(n, a) for (n, a) in c_res if a * (n + 1) - 1.0 < 1.0 / 3.0]     # allowed total loss below the smallest FNR jump (1/3): both rules coincide
c_obs_tie = [k for k, o in c_res.items() if np.all(np.abs(o["lamB"] - o["lamA"]) < 1e-12)]
c_szle = all(o["szB"].mean() <= o["szA"].mean() + 1e-12 for o in c_res.values())
check("S6.32", c_szle and sorted(c_obs_tie) == sorted(c_tie) and all(o["szB"].mean() < o["szA"].mean() for k, o in c_res.items() if k not in c_tie),
      f"FNR set size <= miscoverage set size in all 12 cells, strictly smaller except where alpha(n+1)-B < 1/3 (predicted ties {c_tie}, observed ties {c_obs_tie})")
c_zs = min(min(o["LA"].mean() / c_se(o["LA"]), o["LB"].mean() / c_se(o["LB"])) for o in c_res.values())
check("S6.33", c_zs >= 3.0, f"non-vacuity: every cell's measured risk is at least 3 SE above zero for both losses (min mean/SE = {c_zs:.1f}); an all-zero risk would print 0")
c_vac = [k for k in c_res if k[1] - 2.0 / (k[0] + 1) <= 0]
note("S6.33b", f"the q3 lower bound alpha - 2B/(n+1) is vacuous (<= 0) in cells {c_vac}; the exact closed form (2) is the informative test for the miscoverage loss there")
c_m = [np.mean([c_res[(n, 0.10)]["lamA"].mean()]) for n in c_NS]
c_sem = [c_se(c_res[(n, 0.10)]["lamA"]) for n in c_NS]
check("S6.34", all(c_m[i + 1] <= c_m[i] + 3 * np.hypot(c_sem[i], c_sem[i + 1]) for i in range(3)),
      "mean lambda_hat (alpha=0.10) non-increasing in n up to 3SE: " + ", ".join(f"{n}:{v:.4f}" for n, v in zip(c_NS, c_m)))
"""),
        code(r"""# S6.35-S6.44: the O(1/n) gap made visible. Rank-based simulation: the miscoverage estimator depends on scores only through ranks,
# so U(0,1) scores are exact for any continuous score law. For a U(0,1) test score, E[L | calibration] = 1 - lambda_hat exactly, so we average
# that conditional expectation (removes the Bernoulli test noise). alpha=0.10, reps=400000 per n (chunks of 20000), seed RNG_SEED+70+n
def c_rank_sim(n, alpha, reps, seed, chunk=20000):
    # Bernoulli version: draws the test score too (used where the loss is not an indicator of one score)
    rng = np.random.default_rng(seed)
    k = int(np.floor(alpha * (n + 1) + 1e-9))
    miss = 0
    for _ in range(reps // chunk):
        S = rng.random((chunk, n + 1))
        lam = np.ones(chunk) if k == 0 else -np.partition(-S[:, :n], k - 1, axis=1)[:, k - 1]
        miss += int(np.sum(S[:, n] > lam))
    return miss / reps


def c_rank_rb(n, alpha, reps, seed, chunk=20000):
    # Rao-Blackwellised: mean and SE of 1 - lambda_hat, lambda_hat = k-th largest of n uniforms (lambda_max = 1 if k = 0)
    rng = np.random.default_rng(seed)
    k = int(np.floor(alpha * (n + 1) + 1e-9))
    vals = []
    for _ in range(reps // chunk):
        S = rng.random((chunk, n))
        lam = np.ones(chunk) if k == 0 else -np.partition(-S, k - 1, axis=1)[:, k - 1]
        vals.append(1.0 - lam)
    v = np.concatenate(vals)
    return v.mean(), v.std(ddof=1) / np.sqrt(len(v))


c_NG, c_AG, c_RG = [24, 54, 104, 204], 0.10, 400000
c_gap = {}
c_lab = 35
for c_n in c_NG:
    c_e, c_s = c_rank_rb(c_n, c_AG, c_RG, RNG_SEED + 70 + c_n)
    c_cf = np.floor(c_AG * (c_n + 1) + 1e-9) / (c_n + 1)
    c_gap[c_n] = (c_AG - c_e, c_s, c_AG - c_cf)
    print(f"n={c_n}: E={c_e:.5f} closed form={c_cf:.5f}  gap alpha-E={c_AG - c_e:+.5f} (3SE={3 * c_s:.5f})  gap*(n+1)={(c_AG - c_e) * (c_n + 1):.3f} +/- {c_s * (c_n + 1):.3f}")
    check(f"S6.{c_lab}", abs(c_e - c_cf) <= 3 * c_s, f"rank sim n={c_n}: E={c_e:.5f} vs equation (2) {c_cf:.5f} (3SE={3 * c_s:.5f})")
    check(f"S6.{c_lab + 1}", (c_AG - c_e) >= -3 * c_s and (c_AG - c_e) <= 2.0 / (c_n + 1) + 3 * c_s,
          f"n={c_n}: 0 <= gap={c_AG - c_e:+.5f} <= 2B/(n+1)={2.0 / (c_n + 1):.5f} (within 3SE)")
    c_lab += 2
c_x = np.log(np.array(c_NG) + 1.0)
c_y = np.log([c_gap[n][0] for n in c_NG])
c_w = np.array([(c_gap[n][0] / c_gap[n][1]) ** 2 for n in c_NG])          # inverse variance of log gap (delta method)
c_xb = np.sum(c_w * c_x) / np.sum(c_w)
c_slope = np.sum(c_w * (c_x - c_xb) * c_y) / np.sum(c_w * (c_x - c_xb) ** 2)
c_slope_se = 1.0 / np.sqrt(np.sum(c_w * (c_x - c_xb) ** 2))
print(f"log-log slope of the gap against n+1: {c_slope:.3f} +/- {c_slope_se:.3f} (SE); O(1/n) predicts -1")
check("S6.43", abs(c_slope + 1.0) <= 3 * c_slope_se, f"log-log slope {c_slope:.3f} within 3 SE ({3 * c_slope_se:.3f}) of -1")
check("S6.44", abs(c_slope + 0.5) > 3 * c_slope_se, f"falsifier: a 1/sqrt(n) gap would have slope -0.5; measured {c_slope:.3f} is more than 3 SE away")
"""),
        code(r"""# Figure 6.1: achieved risk versus alpha (both losses) and the O(1/n) gap
fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.3))
c_mk = {24: "o", 54: "s", 104: "^", 204: "D"}
for (c_n, c_a), o in c_res.items():
    ax[0].errorbar(c_a, o["LA"].mean(), yerr=3 * c_se(o["LA"]), fmt=c_mk[c_n], color=PALETTE["clean"], ms=5, capsize=2,
                   label=f"miscoverage, n={c_n}" if c_a == 0.05 else None)
    ax[0].errorbar(c_a + 0.004, o["LB"].mean(), yerr=3 * c_se(o["LB"]), fmt=c_mk[c_n], color=PALETTE["acoustic"], ms=5, capsize=2,
                   label=f"FNR, n={c_n}" if c_a == 0.05 else None)
c_x = np.linspace(0.03, 0.22, 10)
ax[0].plot(c_x, c_x, color=PALETTE["reference"], ls="--", label="E = alpha (Theorem 1)")
ax[0].plot(c_x, c_x - 2.0 / 25, color=PALETTE["shifted"], ls=":", label="alpha - 2B/(n+1), n=24")
ax[0].set_xlabel("target alpha"); ax[0].set_ylabel("achieved E[test loss] (bars: 3 SE)"); ax[0].legend(fontsize=7, ncol=2)
ax[0].set_title("Achieved risk versus target alpha")
c_nn = np.array(c_NG)
ax[1].errorbar(c_nn + 1, [max(c_gap[n][0], 1e-5) for n in c_NG], yerr=[3 * c_gap[n][1] for n in c_NG], fmt="o", color=PALETTE["clean"], capsize=3, label="measured gap alpha - E (3 SE)")
ax[1].plot(c_nn + 1, [c_gap[n][2] for n in c_NG], "x", color=PALETTE["frozen"], ms=9, label="closed form, eq. (2)")
c_xx = np.linspace(20, 260, 50)
ax[1].plot(c_xx, 2.0 / c_xx, color=PALETTE["shifted"], ls=":", label="2B/(n+1) (q3 lower bound)")
ax[1].plot(c_xx, 1.0 / c_xx, color=PALETTE["reference"], ls="--", label="B/(n+1) (eq. 2 worst case)")
ax[1].set_xscale("log"); ax[1].set_yscale("log"); ax[1].set_xlabel("n + 1"); ax[1].set_ylabel("gap alpha - E[L]"); ax[1].legend(fontsize=7)
ax[1].set_title("Gap versus n+1 (rank simulation)")
plt.tight_layout(); plt.show()
print("gap*(n+1) by n:", {n: round(c_gap[n][2] * (n + 1), 3) for n in c_NG}, "(closed form)")
"""),
        md(r"""### How to read this chart

Left panel: the horizontal axis is the target $\alpha$; the vertical axis is the mean loss on a fresh test point after calibrating with the CRC rule, with bars showing three Monte Carlo standard errors. Blue markers are set miscoverage and green markers are the false-negative rate (nudged right); marker shape encodes $n$. The dashed diagonal is Theorem 1, $\mathbb{E}\le\alpha$. The red dotted line is the lower bound $\alpha-2B/(n+1)$ drawn for the smallest $n$. A marker on or just below the diagonal means the guarantee holds and is nearly tight; above it beyond its bar would be a violation; far below would mean wasteful conservativeness.

Right panel: log-log axes, horizontal $n+1$, vertical the gap $\alpha-\mathbb{E}[L]$ from the rank simulation, where the test loss is replaced by its exact conditional mean so that the standard errors are small. Blue points are measurements (three-standard-error bars), purple crosses the closed form (2), the red dotted line $2B/(n+1)$, the grey dashed line $B/(n+1)$. Points below both lines and falling parallel to them show the $O(1/n)$ decay; the printed log-log slope quantifies "parallel". For other choices of $n$ the gap would zigzag between zero and $B/(n+1)$, being a fractional part divided by $n+1$.

Takeaway: the achieved risk is close to the target and the shortfall is of order $1/n$. Wrong if any point in the left panel lies above the diagonal beyond its bar, or if the printed slope in the right panel is far from $-1$.
"""),
        code(r"""# Figure 6.2: what the estimator does on one calibration draw, and how conservativeness varies with n and alpha
fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
c_rng = np.random.default_rng(RNG_SEED + 80)
c_cols = [PALETTE["clean"], PALETTE["acoustic"], PALETTE["shifted"]]
for c_n, c_col in zip([24, 104, 404], c_cols):
    Yc, Sc = c_draw(c_rng, 1, c_n)
    a = np.where(Yc[0], Sc[0], -1.0).max(-1)
    Rg = c_curve(a, np.ones(c_n), c_n)
    lam = c_lambda_hat(Rg, c_n, 1.0, 0.10)
    ax[0].plot(c_GRID, Rg, color=c_col, label=f"R_n(lambda), n={c_n}")
    ax[0].axhline((0.10 * (c_n + 1) - 1) / c_n, color=c_col, ls=":", lw=1)
    ax[0].axvline(lam, color=c_col, ls="--", lw=1)
ax[0].axhline(0.10, color=PALETTE["reference"], lw=1, label="alpha")
ax[0].set_xlim(0.3, 1.0); ax[0].set_ylim(0, 0.4); ax[0].set_xlabel("lambda (larger = larger sets)"); ax[0].set_ylabel("empirical miscoverage risk")
ax[0].legend(fontsize=7); ax[0].set_title("Risk curves, lambda_hat (dashed), corrected target (dotted)")
ax[1].plot(c_lams, c_size, color=PALETTE["clean"], label="mean set size")
ax[1].set_xlabel("lambda"); ax[1].set_ylabel("mean |C_lambda| (of K=8)"); ax[1].set_title("Larger lambda, larger sets")
ax2 = ax[1].twinx(); ax2.plot(c_lams, c_mis, color=PALETTE["shifted"], label="population miscoverage"); ax2.set_ylabel("population miscoverage risk"); ax2.grid(False)
for c_a, c_col in zip(c_AS, c_cols):
    ax[2].errorbar(c_NS, [c_res[(n, c_a)]["lamA"].mean() for n in c_NS], yerr=[3 * c_se(c_res[(n, c_a)]["lamA"]) for n in c_NS],
                   marker="o", color=c_col, capsize=3, label=f"alpha={c_a}")
ax[2].set_xlabel("calibration size n"); ax[2].set_ylabel("mean lambda_hat (3 SE bars)"); ax[2].legend(); ax[2].set_title("Threshold versus n and alpha")
plt.tight_layout(); plt.show()
print("mean lambda_hat by n (alpha=0.05, 0.10, 0.20):", {n: tuple(round(c_res[(n, a)]["lamA"].mean(), 3) for a in c_AS) for n in c_NS})
print("mean test set size by n (alpha=0.10, miscoverage loss):", {n: round(c_res[(n, 0.10)]["szA"].mean(), 2) for n in c_NS})
"""),
        md(r"""### How to read this chart

Left panel: the horizontal axis is $\lambda$ and the vertical axis is the empirical miscoverage risk $\hat R_n(\lambda)$ of one calibration draw per sample size (three colours). Each curve is a non-increasing staircase. The grey line is $\alpha$. The dotted line of matching colour is the corrected target $(\alpha(n+1)-B)/n$ that $\hat R_n$ must fall under, and the dashed vertical line is the resulting $\hat\lambda$, the first $\lambda$ where the staircase crosses the dotted line. Note the dotted lines lie below $\alpha$, and by less as $n$ grows: that is the safety margin.

Middle panel: mean prediction-set size (blue, left axis) and population miscoverage risk (red, right axis) against $\lambda$. Sets grow with $\lambda$ while miscoverage falls, so moving right always trades size for coverage. This is what is meant by a larger $\lambda$ being more conservative.

Right panel: mean $\hat\lambda$ against $n$ for three targets. Stricter targets (smaller $\alpha$) give larger thresholds. Within a target the thresholds do not increase with $n$: the safety margin shrinks as $n$ grows, so they approach the population quantile from the conservative side. Bars are three standard errors.

Takeaway: the correction costs extra set size at small $n$ and the cost fades with $n$. Wrong if a dashed line sat left of the crossing point.
"""),
        code(r"""# Figure 6.3: the false-negative-rate loss compared with set miscoverage (same runs as Figure 6.1/6.2)
fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
c_w = 0.008
for c_j, c_n in enumerate([24, 204]):
    ax[0].bar(np.array(c_AS) + (c_j - 0.5) * 2 * c_w, [c_res[(c_n, a)]["szA"].mean() for a in c_AS], width=c_w, color=PALETTE["clean"], alpha=0.5 + 0.4 * c_j,
              label=f"miscoverage, n={c_n}")
    ax[0].bar(np.array(c_AS) + (c_j - 0.5) * 2 * c_w + c_w, [c_res[(c_n, a)]["szB"].mean() for a in c_AS], width=c_w, color=PALETTE["acoustic"], alpha=0.5 + 0.4 * c_j,
              label=f"FNR, n={c_n}")
ax[0].set_xlabel("alpha"); ax[0].set_ylabel("mean test set size"); ax[0].legend(fontsize=7); ax[0].set_title("FNR control needs smaller sets")
c_o = c_res[(104, 0.10)]
ax[1].hist(c_o["LB"], bins=np.linspace(-0.01, 1.01, 12), color=PALETTE["acoustic"], alpha=0.7, label="FNR test loss")
ax[1].axvline(c_o["LB"].mean(), color=PALETTE["acoustic"], ls="--", label="mean FNR")
ax[1].axvline(c_o["LA"].mean(), color=PALETTE["clean"], ls="--", label="mean miscoverage")
ax[1].axvline(0.10, color=PALETTE["reference"], label="alpha")
ax[1].set_xlabel("per-test-point loss"); ax[1].set_ylabel("count of replicates"); ax[1].legend(fontsize=7); ax[1].set_title("n=104, alpha=0.10")
ax[2].scatter(c_o["lamA"] + np.random.default_rng(RNG_SEED + 81).normal(0, 0.002, c_REPS), c_o["lamB"], s=5, alpha=0.3, color=PALETTE["frozen"])
ax[2].plot([0, 1], [0, 1], color=PALETTE["reference"], ls="--")
ax[2].set_xlabel("lambda_hat, miscoverage loss"); ax[2].set_ylabel("lambda_hat, FNR loss"); ax[2].set_title("Pathwise ordering")
ax[2].set_xlim(0.3, 1); ax[2].set_ylim(0.3, 1)
plt.tight_layout(); plt.show()
print(f"n=104 alpha=0.10: mean FNR={c_o['LB'].mean():.4f}, mean miscoverage={c_o['LA'].mean():.4f}, mean size FNR={c_o['szB'].mean():.2f} vs miscoverage={c_o['szA'].mean():.2f}")
print(f"fraction of test points with FNR loss exactly 0: {np.mean(c_o['LB'] == 0):.3f}")
"""),
        md(r"""### How to read this chart

Left panel: the horizontal axis is $\alpha$, the vertical axis is the mean size of the prediction set on the test point. Blue bars are miscoverage control and green bars are false-negative-rate control, each drawn for a small and a large $n$ (lighter and darker shading). Compare neighbouring blue and green bars: a green bar below its blue neighbour means the FNR target is met with a smaller set, because missing one of several true labels costs only a fraction of a unit under FNR but a whole unit under miscoverage. Bars coincide when $\alpha(n+1)-B$ is below the smallest FNR jump, since neither rule can then afford a miss.

Middle panel: histogram of the per-test-point FNR loss over replicates at one setting. Losses are multiples of reciprocal label counts, so the histogram has spikes. The dashed green line is the mean FNR, the dashed blue line the mean miscoverage, and the grey line is $\alpha$. Both means should sit near $\alpha$ because both losses were calibrated to the same target; a mean slightly above it is Monte Carlo error.

Right panel: each point is one replicate; the horizontal coordinate is the miscoverage threshold and the vertical coordinate the FNR threshold. The dashed diagonal is equality. Every point on or below it confirms $\hat\lambda^{\rm fnr}\le\hat\lambda^{\rm mis}$, as $L^{\rm fnr}\le L^{\rm mis}$ demands.

Takeaway: the second loss has its own risk, its own thresholds and smaller sets. Wrong if any point lay above the diagonal.
"""),
        # ------------------------------------------------------------------ SECTION 7
        md(r"""## 7. Controls where the guarantee visibly breaks (derived here; tests claims from q2, q3)

A verification that cannot fail is not a verification. Section 6 showed the bound holding where its assumptions hold. This section removes assumptions one at a time and shows the measured risk leaving the bound. Three controls are used: covariate shift (exchangeability), a non-monotone loss, and a calibration set too small for the correction term.

**Claims under test (q3, Section 5 limitations, as summarised in the trace).**

> Monotonicity Constraint: The procedure strictly requires loss functions L_i(λ) to be non-increasing in λ (q3)

> If losses are non-monotone, risk control can fail arbitrarily (q3, with the failure level E[L_{n+1}(λ̂)] ≥ B − ε)

> Uncompensated distribution shifts invalidate the exchangeability assumption, causing standard CRC to lose validity (q3)

**Control A, covariate shift.** Calibrate at signal strength $1.5$ and test at a weaker signal, a crude stand-in for noise that makes scores less informative. The mean loss at the calibrated threshold should rise above $\alpha$ as the shift grows, while the unshifted control stays at or below it. The paper's own synthetic regression experiment reports the same phenomenon with a different model (q3); our numbers are a toy and are not comparable to it.

**Control B, non-monotone loss.** The trace states the conclusion of Proposition 2 (risk control fails, expected loss at least $B-\epsilon$) but not the construction (q3). We therefore build our own instance of the phenomenon, a selection effect, and do not claim it is the source's counterexample. Each calibration point has, at every non-maximal $\lambda$ on a fine grid, an independent loss in $\{0,1\}$ with mean $\mu$, and loss zero at $\lambda_{\max}$, so the achievability assumption holds. Only monotonicity fails. With many grid points the calibration risk dips below the target somewhere by chance, the inf rule picks that spot, and the fresh test point still has loss mean $\mu$ there. The identity $\mathbb{E}[L_{n+1}(\hat\lambda)]=\mu\cdot\Pr(\hat\lambda<\lambda_{\max})$ is exact and is tested. The remedy in q3, Theorem C.1, replaces $\hat R_n$ by $\hat R_n^\uparrow(\lambda)=\sup_{t\ge\lambda}\hat R_n(t)$; the trace says it gives asymptotic control, so we test only that it removes the failure here.

**Control C, too little calibration data.** The estimator needs $\alpha(n+1)\ge B$ for the set to be non-empty. Below $n_{\min}=\lceil B/\alpha\rceil-1$ the rule returns $\lambda_{\max}$ (full sets) every time. The bound then holds trivially but the predictor is useless.
"""),
        code(r"""# S7.1-S7.8: Control A, covariate shift. n=104, alpha=0.10, reps=4000; test signal in {1.5 (control), 1.4, 1.2, 0.9, 0.5}; common seed RNG_SEED+90
c_SHIFTS = [1.5, 1.4, 1.2, 0.9, 0.5]
c_sh = {sg: c_mc_cell(104, 0.10, 4000, RNG_SEED + 90, sig_cal=1.5, sig_test=sg) for sg in c_SHIFTS}   # same seed: identical calibration sets, only the test law changes
print("test signal   E[mis]   SE      E[fnr]   SE      mean size (mis)")
for sg, o in c_sh.items():
    print(f"{sg:<12.1f} {o['LA'].mean():.4f}  {c_se(o['LA']):.4f}  {o['LB'].mean():.4f}   {c_se(o['LB']):.4f}  {o['szA'].mean():.2f}")
c_c, c_st = c_sh[1.5], c_sh[0.5]
check("S7.1", c_c["LA"].mean() <= 0.10 + 3 * c_se(c_c["LA"]), f"exchangeable control: E[mis]={c_c['LA'].mean():.4f} <= alpha=0.10 + 3SE={3 * c_se(c_c['LA']):.4f}")
check("S7.2", c_c["LB"].mean() <= 0.10 + 3 * c_se(c_c["LB"]), f"exchangeable control: E[fnr]={c_c['LB'].mean():.4f} <= alpha=0.10 + 3SE")
check("S7.3", c_st["LA"].mean() > 0.10 + 3 * c_se(c_st["LA"]), f"strong shift breaks miscoverage control: E={c_st['LA'].mean():.4f} > alpha + 3SE = {0.10 + 3 * c_se(c_st['LA']):.4f}")
check("S7.4", c_st["LB"].mean() > 0.10 + 3 * c_se(c_st["LB"]), f"strong shift breaks FNR control: E={c_st['LB'].mean():.4f} > alpha + 3SE = {0.10 + 3 * c_se(c_st['LB']):.4f}")
c_chain = [c_sh[s]["LA"].mean() for s in c_SHIFTS]
c_chse = [c_se(c_sh[s]["LA"]) for s in c_SHIFTS]
check("S7.5", all(c_chain[i + 1] >= c_chain[i] - 3 * np.hypot(c_chse[i], c_chse[i + 1]) for i in range(4)),
      "miscoverage risk is non-decreasing along the shift sweep (within 3SE of each step): " + ", ".join(f"{v:.3f}" for v in c_chain))
check("S7.6", c_st["LA"].mean() > 2 * c_c["LA"].mean(), f"strongest shift more than doubles the risk: {c_st['LA'].mean():.3f} vs control {c_c['LA'].mean():.3f}")
c_mild = c_sh[1.4]["LA"].mean() > 0.10 + 3 * c_se(c_sh[1.4]["LA"])
note("S7.7", f"mildest shift (test signal 1.4): E[mis]={c_sh[1.4]['LA'].mean():.4f} vs alpha+3SE={0.10 + 3 * c_se(c_sh[1.4]['LA']):.4f}; "
             f"violation {'detected' if c_mild else 'NOT detected'} at this n and reps; margin over the bar = {c_sh[1.4]['LA'].mean() - 0.10 - 3 * c_se(c_sh[1.4]['LA']):+.4f}")
check("S7.8", all(np.array_equal(c_sh[s_]["lamA"], c_c["lamA"]) and np.array_equal(c_sh[s_]["lamB"], c_c["lamB"]) for s_ in c_SHIFTS),
      "the calibrated thresholds are identical arrays under every test-time shift (the rule cannot see the shift)")
check("S7.9", c_st["szA"].mean() <= c_c["szA"].mean() + 3 * np.hypot(c_se(c_st["szA"]), c_se(c_c["szA"])),
      f"sets do not grow to compensate: mean size {c_c['szA'].mean():.2f} (control) -> {c_st['szA'].mean():.2f} (strongest shift)")
"""),
        code(r"""# Figure 7.1: covariate shift sweep
fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
c_xs = np.array(c_SHIFTS)
ax[0].errorbar(c_xs, [c_sh[s]["LA"].mean() for s in c_SHIFTS], yerr=[3 * c_se(c_sh[s]["LA"]) for s in c_SHIFTS], marker="o", color=PALETTE["shifted"], capsize=3, label="miscoverage")
ax[0].errorbar(c_xs, [c_sh[s]["LB"].mean() for s in c_SHIFTS], yerr=[3 * c_se(c_sh[s]["LB"]) for s in c_SHIFTS], marker="s", color=PALETTE["acoustic"], capsize=3, label="FNR")
ax[0].axhline(0.10, color=PALETTE["reference"], ls="--", label="alpha = 0.10")
ax[0].invert_xaxis(); ax[0].set_xlabel("test-time signal strength (calibration = 1.5, shift grows to the right)")
ax[0].set_ylabel("achieved E[test loss] (3 SE bars)"); ax[0].legend(); ax[0].set_title("Exchangeability broken: risk exceeds alpha")
ax[1].plot(c_xs, [c_sh[s]["szA"].mean() for s in c_SHIFTS], marker="o", color=PALETTE["clean"], label="mean set size (miscoverage rule)")
ax[1].plot(c_xs, [c_sh[s]["lamA"].mean() for s in c_SHIFTS], marker="^", color=PALETTE["frozen"], label="mean lambda_hat (unchanged)")
ax[1].invert_xaxis(); ax[1].set_xlabel("test-time signal strength"); ax[1].set_ylabel("value"); ax[1].legend(fontsize=8)
ax[1].set_title("Threshold is blind to the shift")
plt.tight_layout(); plt.show()
print(f"E[mis] from control to strongest shift: {c_c['LA'].mean():.4f} -> {c_st['LA'].mean():.4f}; alpha = 0.10")
"""),
        md(r"""### How to read this chart

Left panel: the horizontal axis is the test-time signal strength, drawn reversed so that the shift grows to the right; the leftmost point equals the calibration signal, so it is the exchangeable control. The vertical axis is the achieved mean test loss with three standard error bars. Red is miscoverage, green is the false-negative rate, and the grey dashed line is the target $\alpha$. A point on or below the dashed line means the guarantee holds; above it, beyond its bar, it failed. The control should sit at or under the line and the shifted points climb above it.

Right panel: the mean threshold $\hat\lambda$ (purple) and the mean test-set size (blue) against the same axis. The threshold was computed before the shift, so its line is exactly flat. The set size on shifted test points does not grow to compensate. A flat threshold and non-growing sets while the risk climbs mean nothing available at calibration time announces the failure.

Takeaway: the finite-sample guarantee is specific to exchangeable data, and the size of the violation grows with the severity of the shift. The mildest shift sits close to the noise floor of the test, and whether it clears the three-standard-error bar depends on the seed and on the number of replicates; the printed note for this run says which side it fell on. Wrong if the shifted points remained under the dashed line.
"""),
        code(r"""# S7.10-S7.18: Control B, non-monotone loss (selection effect). n=10, alpha=0.30, B=1, mu=0.7, grid G=2000, reps=20000, seed RNG_SEED+100
from math import comb
c_nB, c_aB, c_muB, c_GB, c_RB, c_CH = 10, 0.30, 0.7, 2000, 20000, 500
c_rng = np.random.default_rng(RNG_SEED + 100)
c_hit, c_Ltest, c_hit_up, c_Ltest_up, c_wiggle, c_top = [], [], [], [], [], []
for _ in range(c_RB // c_CH):
    L = (c_rng.random((c_CH, c_nB + 1, c_GB)) < c_muB).astype(np.int8)
    L[:, :, -1] = 0                                        # achievability: loss 0 at lambda_max
    Rn = L[:, :c_nB].mean(1)
    ok = (c_nB / (c_nB + 1)) * Rn + 1.0 / (c_nB + 1) <= c_aB + 1e-12
    idx = np.where(ok.any(1), ok.argmax(1), c_GB - 1)
    Rup = np.maximum.accumulate(Rn[:, ::-1], axis=1)[:, ::-1]           # sup_{t >= lambda} R_n(t), q3 Theorem C.1 construction
    ok_up = (c_nB / (c_nB + 1)) * Rup + 1.0 / (c_nB + 1) <= c_aB + 1e-12
    idx_up = np.where(ok_up.any(1), ok_up.argmax(1), c_GB - 1)
    rows = np.arange(c_CH)
    c_hit.append(idx < c_GB - 1); c_Ltest.append(L[rows, c_nB, idx])
    c_hit_up.append(idx_up < c_GB - 1); c_Ltest_up.append(L[rows, c_nB, idx_up])
    c_wiggle.append((np.diff(L[:, 0, :].astype(int), axis=1) > 0).any(1)); c_top.append(L[:, :, -1].max())
c_hit, c_Ltest, c_hit_up, c_Ltest_up, c_wiggle = map(np.concatenate, (c_hit, c_Ltest, c_hit_up, c_Ltest_up, c_wiggle))
c_eB, c_seB = c_Ltest.mean(), c_se(c_Ltest.astype(float))
c_eU, c_seU = c_Ltest_up.mean(), c_se(c_Ltest_up.astype(float))
c_eC = c_rank_sim(c_nB, c_aB, 100000, RNG_SEED + 101, chunk=10000)
c_seC = np.sqrt(c_eC * (1 - c_eC) / 100000)
c_kB = int(np.floor(c_aB * (c_nB + 1) - 1 + 1e-9))                         # largest number of nonzero losses that passes the corrected test
c_pB = sum(comb(c_nB, j) * c_muB ** j * (1 - c_muB) ** (c_nB - j) for j in range(0, c_kB + 1))   # P(one grid point passes): at most c_kB ones among n
c_hit_exact = 1.0 - (1.0 - c_pB) ** (c_GB - 1)
c_cfC = np.floor(c_aB * (c_nB + 1)) / (c_nB + 1)
print(f"non-monotone: P(lambda_hat < lambda_max) = {c_hit.mean():.3f} (exact {c_hit_exact:.3f}), E[L(lambda_hat)] = {c_eB:.4f} +/- {c_seB:.4f} vs alpha = {c_aB}")
print(f"monotonized (sup_(t>=lambda) R_n): P(lambda_hat < lambda_max) = {c_hit_up.mean():.4f}, E = {c_eU:.4f} +/- {c_seU:.4f}")
print(f"monotone control (indicator loss, same n and alpha): E = {c_eC:.4f} +/- {c_seC:.4f} vs closed form {c_cfC:.4f}")
check("S7.10", c_eB > c_aB + 3 * c_seB, f"non-monotone loss breaks control: E={c_eB:.4f} > alpha + 3SE = {c_aB + 3 * c_seB:.4f}")
check("S7.11", abs(c_hit.mean() - c_hit_exact) <= 3 * c_se(c_hit.astype(float)), f"P(hit) = {c_hit.mean():.4f} vs exact 1-(1-p)^(G-1) = {c_hit_exact:.4f} (p = {c_pB:.4f} is a binomial tail)")
check("S7.12", abs(c_eB - c_muB * c_hit_exact) <= 3 * c_seB, f"exact value E[L(lambda_hat)] = mu * P(hit): {c_eB:.4f} vs {c_muB * c_hit_exact:.4f}")
check("S7.13", c_eB > 0.5 * c_muB, f"failure is large, not marginal: E={c_eB:.3f} vs half of mu = {0.5 * c_muB:.3f} (q3: fails up to B - eps)")
check("S7.14", c_eU <= c_aB + 3 * c_seU, f"monotonized empirical risk restores control here: E={c_eU:.4f} <= alpha + 3SE = {c_aB + 3 * c_seU:.4f}")
check("S7.15", c_eC <= c_aB + 3 * c_seC and abs(c_eC - c_cfC) <= 3 * c_seC, f"monotone control at same n, alpha: E={c_eC:.4f} <= alpha and equals eq. (2) {c_cfC:.4f} within 3SE")
check("S7.16", max(c_top) == 0, f"achievability holds (max loss at lambda_max = {max(c_top)} <= alpha), so monotonicity alone is responsible")
check("S7.17", c_wiggle.mean() > 0.99, f"loss paths are non-monotone: an upward step in {c_wiggle.mean():.1%} of sampled paths")
note("S7.18", f"the monotonized rule is very conservative here (P(hit) = {c_hit_up.mean():.4f}, E = {c_eU:.4f} against alpha = {c_aB}): validity is bought with a nearly useless predictor. q3 claims only asymptotic control for Theorem C.1.")
"""),
        code(r"""# Figure 7.2: non-monotone loss versus monotone control versus monotonized remedy
fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
c_names = ["monotone control\n(indicator loss)", "non-monotone loss\n(inf rule)", "non-monotone loss\n(monotonized R_n)"]
c_vals = [c_eC, c_eB, c_eU]; c_errs = [3 * c_seC, 3 * c_seB, 3 * c_seU]
ax[0].bar(c_names, c_vals, yerr=c_errs, color=[PALETTE["clean"], PALETTE["shifted"], PALETTE["adapted"]], capsize=4)
ax[0].axhline(c_aB, color=PALETTE["reference"], ls="--", label="alpha = 0.30"); ax[0].axhline(c_muB, color=PALETTE["mask"], ls=":", label="per-point loss mean mu")
ax[0].set_ylabel("achieved E[test loss] (3 SE)"); ax[0].legend(); ax[0].set_title("Only the non-monotone rule leaves the bound")
c_rr = np.random.default_rng(RNG_SEED + 102)
c_Lx = (c_rr.random((c_nB, c_GB)) < c_muB).astype(float); c_Lx[:, -1] = 0
c_Rx = c_Lx.mean(0); c_Rux = np.maximum.accumulate(c_Rx[::-1])[::-1]
ax[1].plot(np.linspace(0, 1, c_GB), c_Rx, color=PALETTE["shifted"], lw=0.8, label="R_n(lambda), one draw (non-monotone)")
ax[1].plot(np.linspace(0, 1, c_GB), c_Rux, color=PALETTE["adapted"], lw=1.5, label="sup_(t>=lambda) R_n(t)")
ax[1].axhline((c_aB * (c_nB + 1) - 1) / c_nB, color=PALETTE["reference"], ls="--", label="corrected target")
ax[1].set_xlabel("lambda"); ax[1].set_ylabel("empirical risk"); ax[1].legend(fontsize=7); ax[1].set_title("The noisy curve dips below the target by chance")
plt.tight_layout(); plt.show()
print(f"bars: control {c_eC:.4f}, non-monotone {c_eB:.4f}, monotonized {c_eU:.4f}; alpha = {c_aB}")
"""),
        md(r"""### How to read this chart

Left panel: three bars, each an achieved mean test loss with a three-standard-error whisker. The grey dashed line is the target $\alpha$ and the orange dotted line is the mean loss $\mu$ of a single point at a non-maximal setting. A bar at or under the dashed line means the guarantee held. The first bar is the exchangeable, monotone control with the same $n$ and $\alpha$, at the level predicted by equation (2). The second bar is the plain inf rule applied to non-monotone losses; it sits well above $\alpha$, at the level of $\mu$ times the fraction of draws where the rule found a spot to stop, an identity that the printed checks verify against a binomial closed form. The third bar applies the monotonized empirical risk from q3's Theorem C.1 and returns to the safe side.

Right panel: a single calibration draw of the non-monotone empirical risk (red) with its running supremum from the right (brown), and the corrected target (grey dashed). The red curve is jagged: with many candidate settings, chance alone drags it under the target somewhere, and the inf rule stops at the first such spot. That spot is a selection artefact, so the fresh test point sees the full $\mu$. The brown curve, a supremum of everything to its right, stays above the target until the far end.

Takeaway: monotonicity is what protects the rule from selecting a lucky setting. Wrong if the second bar were at or under the dashed line.
"""),
        code(r"""# S7.19-S7.26: Control C, calibration set too small. alpha=0.05, B=1 -> n_min = ceil(B/alpha) - 1; reps=4000 per n; seeds RNG_SEED+110+i
c_aC = 0.05
c_nmin = int(np.ceil(1.0 / c_aC)) - 1
c_NC = [5, 10, 15, c_nmin - 1, c_nmin, c_nmin + 1, 30, 60]
c_sm = {n: c_mc_cell(n, c_aC, 4000, RNG_SEED + 110 + i) for i, n in enumerate(c_NC)}
print(f"n_min = ceil(B/alpha) - 1 = {c_nmin}")
print("n     P(lambda_hat=lambda_max)  mean size  E[mis]   SE      closed form (2)   E[fnr]")
def c_cf2(n, a=c_aC):
    return np.floor(a * (n + 1) + 1e-9) / (n + 1)
for n, o in c_sm.items():
    print(f"{n:<5d} {float(np.mean(o['lamA'] == 1.0)):<25.3f} {o['szA'].mean():<10.2f} {o['LA'].mean():.4f}   {c_se(o['LA']):.4f}  {c_cf2(n):.4f}            {o['LB'].mean():.4f}")
c_below = [n for n in c_NC if n < c_nmin]
c_above = [n for n in c_NC if n >= c_nmin]
check("S7.19", all(np.all(c_sm[n]["lamA"] == 1.0) and np.all(c_sm[n]["lamB"] == 1.0) for n in c_below), f"n < {c_nmin} (n in {c_below}): lambda_hat = lambda_max for both losses in every replicate")
check("S7.20", all(np.all(c_sm[n]["szA"] == c_K) for n in c_below), f"n < {c_nmin}: test set is the full label set (size K={c_K}) in every replicate")
check("S7.21", all(c_sm[n]["LA"].mean() == 0.0 and c_sm[n]["LB"].mean() == 0.0 for n in c_below), f"n < {c_nmin}: measured risk is exactly 0, versus target alpha={c_aC}: valid but wasteful")
c_at = c_sm[c_nmin]
check("S7.22", np.mean(c_at["lamA"] == 1.0) < 1.0 and c_at["szA"].mean() < c_K, f"n = n_min = {c_nmin}: rule becomes satisfiable, P(full) = {np.mean(c_at['lamA'] == 1.0):.3f}, mean size {c_at['szA'].mean():.2f} < K")
check("S7.23", all(abs(c_sm[n]["LA"].mean() - c_cf2(n)) <= 3 * c_se(c_sm[n]["LA"]) for n in c_above),
      "for every n >= n_min the measured miscoverage matches eq. (2) within 3SE: " + ", ".join(f"{n}: {c_sm[n]['LA'].mean():.4f} vs {c_cf2(n):.4f}" for n in c_above))
check("S7.24", all(c_sm[n]["LA"].mean() <= c_aC + 3 * c_se(c_sm[n]["LA"]) for n in c_NC), "the upper bound alpha holds for every n, including the unsatisfiable ones")
c_l0, c_l1 = c_sm[c_nmin]["lamA"], c_sm[30]["lamA"]                                 # both n have k = floor(alpha(n+1)) = 1: lambda_hat is the largest score
check("S7.25", c_l1.mean() > c_l0.mean() + 3 * np.hypot(c_se(c_l1), c_se(c_l0)) and int(np.floor(c_aC * 31)) == int(np.floor(c_aC * (c_nmin + 1))) == 1,
      f"within the k=1 regime lambda_hat is the sample maximum, which grows with n: mean lambda_hat {c_l0.mean():.4f} (n={c_nmin}) -> {c_l1.mean():.4f} (n=30); a decreasing-with-n claim would fail")
c_gaps = {n: c_aC - c_sm[n]["LA"].mean() for n in c_NC}
check("S7.26", all(abs(c_gaps[n] - c_aC) < 1e-12 for n in c_below) and all(abs(c_gaps[n]) < 0.5 * c_aC for n in c_above),
      f"gap alpha - E equals alpha exactly below n_min and is smaller than alpha/2 in absolute value at or above it: " + ", ".join(f"{n}: {g:+.4f}" for n, g in c_gaps.items()))
"""),
        code(r"""# Figure 7.3: what happens when n is below the satisfiability threshold
fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
c_xn = np.array(c_NC)
ax[0].plot(c_xn, [c_sm[n]["szA"].mean() for n in c_NC], marker="o", color=PALETTE["clean"], label="miscoverage rule")
ax[0].plot(c_xn, [c_sm[n]["szB"].mean() for n in c_NC], marker="s", color=PALETTE["acoustic"], label="FNR rule")
ax[0].axhline(c_K, color=PALETTE["reference"], ls="--", label="K = full label set"); ax[0].axvline(c_nmin - 0.5, color=PALETTE["shifted"], ls=":", label="n_min boundary")
ax[0].set_xscale("log"); ax[0].set_xlabel("calibration size n"); ax[0].set_ylabel("mean test set size"); ax[0].legend(fontsize=8); ax[0].set_title("Full sets until alpha(n+1) >= B")
ax[1].errorbar(c_xn, [c_sm[n]["LA"].mean() for n in c_NC], yerr=[3 * c_se(c_sm[n]["LA"]) for n in c_NC], marker="o", color=PALETTE["clean"], capsize=3, label="measured E[mis]")
ax[1].plot(c_xn, [np.floor(c_aC * (n + 1) + 1e-9) / (n + 1) for n in c_NC], "x", color=PALETTE["frozen"], ms=9, label="closed form (2)")
ax[1].axhline(c_aC, color=PALETTE["reference"], ls="--", label="alpha = 0.05"); ax[1].axvline(c_nmin - 0.5, color=PALETTE["shifted"], ls=":")
ax[1].set_xscale("log"); ax[1].set_xlabel("calibration size n"); ax[1].set_ylabel("achieved E[test loss]"); ax[1].legend(fontsize=8); ax[1].set_title("Risk is zero below n_min, then follows eq. (2)")
plt.tight_layout(); plt.show()
print(f"n_min = {c_nmin}; risk at n={c_NC[0]}: {c_sm[c_NC[0]]['LA'].mean():.4f}; at n={c_nmin}: {c_sm[c_nmin]['LA'].mean():.4f}")
"""),
        md(r"""### How to read this chart

Left panel: the horizontal axis (logarithmic) is the calibration size $n$; the vertical axis is the mean test-set size. Blue is the miscoverage rule and green the false-negative-rate rule; the grey dashed line marks the full label set of $K$ classes, and the red dotted vertical line marks the boundary below which $\alpha(n+1)<B$. Points on the grey line mean the estimator returned $\lambda_{\max}$, so any class is possible; points below it mean a smaller certified set.

Right panel: the achieved mean loss against $n$, with three-standard-error bars, the exact value from equation (2) as purple crosses, and the target $\alpha$ as a grey dashed line. Left of the red dotted line the measured risk is exactly zero, far under $\alpha$: the guarantee holds but only because the predictor is trivial. From the boundary on, the measured risk is nonzero and follows the crosses within its bars; it can sit well under $\alpha$ where $\alpha(n+1)$ has a large fractional part, because the floor operation discards it. In the left panel, while $\lfloor\alpha(n+1)\rfloor$ stays at one the threshold is the sample maximum and the set size rises with $n$; it falls when the floor steps up.

Takeaway: the correction term $B/(n+1)$ is a real cost. A non-trivial predictor needs at least about $B/\alpha-1$ calibration points, and for such small $n$ the risk is decided by a floor operation, so it moves in steps. Wrong if the risk below the boundary were nonzero, or if points above it left the dashed line.
"""),
        # ------------------------------------------------------------------ WHAT CRC DOES NOT ESTABLISH
        md(r"""## What conformal risk control does not establish (derived here; boundaries from q2, q3)

**What the runs above did establish (toy DGP only).** On exchangeable, monotone, bounded toy data, the inf-rule estimator hits the exact closed-form expected miscoverage, the false-negative-rate loss stays inside the upper bound and the lower bound, and the gap to $\alpha$ is of order $1/n$. Each of the three controls broke the guarantee in the way the assumption suggests, and a monotonized rule repaired the non-monotone case at the cost of conservativeness.

**What they did not establish.**
- That standard, unweighted CRC is valid under real acoustic shift for speech commands. Our shift is a change of signal strength in a synthetic multi-label model. The trace states the claim only qualitatively (q2): exchangeability between calibration and test audio is violated under noise, reverberation or microphone changes. Whether a given real shift is mild enough to hide inside Monte Carlo noise is an empirical question; our own mildest shift lies close to the detection limit of the test at this number of replicates, so small violations are hard to separate from noise.
- How much calibration data is needed under shift. The toy shift is far simpler than real acoustic conditions. This is a hypothesis-free statement of a limit: nothing here calibrates the size of a real violation.
- Anything about a single test set. Theorem 1 bounds the expectation over the joint draw of calibration and test points (q2). A single held-out set gives one realisation, and staying under $\alpha$ on it is compatible with failure of the expectation and vice versa.

**Shift remedies need knowledge we may not have (q3).** Weighted CRC (Proposition 3) restores exact control under covariate shift given the density ratio $w(x)=dP_{\rm test}/dP_{\rm train}$. The point-dependent threshold weights each calibration loss by $w(X_i)$ and adds a test-point weight $w(x)B$ in place of $B$. The trace reports that in a synthetic regression the ratios estimated by logistic regression matched the true ratios in effect (q3, Appendix D). Proposition 4 gives $\mathbb{E}[L_{n+1}(\hat\lambda)]\le\alpha+B\sum_i\mathrm{TV}(Z_i,Z_{n+1})$ for arbitrary shift, and a matching lower bound. The trace says both routes require knowing or accurately estimating the form of the shift (q3, Section 5). We did not implement weighted CRC, and we did not test how estimation error in $w$ propagates. That is not established here.

**Monotonicity for real speech losses.** Our two losses are monotone by construction. A speech loss that mixes error rate with latency or a learned rejection rule may not be, and if it is not, Control B shows the failure can be large. Whether any specific speech loss is monotone is checkable on held-out data (q2 lists monotonicity and boundedness as testable), but that check has to be run for each loss. We label as hypothesis, not finding, the idea that adaptive thresholds or fine-tuning on shifted data are the usual sources of non-monotonicity.

**Reading a positive empirical test.** A run where mean held-out loss stays under $\alpha$ implies that on that split the observed loss was under $\alpha$. It does not imply the same on a fresh split, on other acoustic conditions, or after drift. The exchangeability assumption cannot be validated from the data it is meant to describe (q2).

**Sources.** Exchangeability violation: q2, testable versus untestable parts. Weighted CRC, total-variation bound, monotonization (Theorem C.1), Proposition 2 failure: q3.
"""),
        code(r"""# Final summary generated from the labelled checks actually run in this part (no hand-typed counts)
c_by_sec = {}
for c_lbl, c_ok, c_kind in CHECKS:
    if c_lbl.startswith("S5.") or c_lbl.startswith("S6.") or c_lbl.startswith("S7."):
        c_by_sec.setdefault(c_lbl.split(".")[0], []).append((c_lbl, c_ok, c_kind))
print("Part 3 (sections 5-7): labelled results by section")
for c_sec, c_items in c_by_sec.items():
    print(f"  {c_sec}: {len(c_items)} labelled lines, {sum(k == 'PASS' for _, _, k in c_items)} PASS, {sum(k == 'NOTE' for _, _, k in c_items)} NOTE, {sum(k == 'FAIL' for _, _, k in c_items)} FAIL")
c_labels = [l for l, _, _ in CHECKS if l[:2] in ("S5", "S6", "S7")]
check("S7.27", len(c_labels) == len(set(c_labels)), f"all {len(c_labels)} part-3 labels are unique (no duplicated check ids)")
check_summary()
"""),
    ]
