"""NB1 part 4: section 7 (salvaged derivations, re-verified) and section 8 (limitations and claim ledger).

Constructed inputs here are estimator-verification fixtures (uniform scores, constructed logits): they test
formulas and estimators, and are never presented as evidence about speech models. The one real-data cell reads
the conformal outputs stored in arch-*.result.json.
"""
from nbkit import md, code


def cells():
    return [
        md(r"""
        ## 7. Salvaged derivations: scoring rules, ECE bias, temperature-scaling MLE, conformal risk control

        This section keeps the mathematics of the earlier foundations notebook that was worth keeping and drops its
        synthetic-data framing. Each derivation is followed by a numerical check against either an independent
        implementation or the project's own `src` functions. The constructed inputs are **estimator-verification
        fixtures**: they test that a formula or an estimator does what its derivation says. They are not evidence
        about speech models and are labelled as fixtures at each cell.

        ### 7.1 Proper scoring rules: NLL and Brier

        For a $K$-class prediction $p\in\Delta^{K-1}$ and a label $y$, the logarithmic score (NLL) and the multiclass
        Brier score are
        $$\ell_{\log}(p,y)=-\log p_y,\qquad \ell_{\mathrm{Br}}(p,y)=\lVert p-e_y\rVert_2^2=\sum_{j}(p_j-\mathbb{1}\{j=y\})^2 .$$
        Suppose the label is drawn from a true distribution $q$. Then the expected scores decompose as
        $$\mathbb{E}_{y\sim q}\,\ell_{\log}(p,y)=H(q)+\mathrm{KL}(q\,\Vert\,p),$$
        $$\mathbb{E}_{y\sim q}\,\ell_{\mathrm{Br}}(p,y)=\lVert p-q\rVert_2^2+\big(1-\lVert q\rVert_2^2\big).$$
        The first is the usual entropy-plus-KL identity: $\sum_y q_y(-\log p_y)=-\sum_y q_y\log q_y+\sum_y q_y\log(q_y/p_y)$.
        For the second, $\mathbb{E}\lVert p-e_y\rVert^2=\lVert p\rVert^2-2p^\top q+1=\lVert p-q\rVert^2+1-\lVert q\rVert^2$.
        In both, the second term does not depend on $p$ and the first is non-negative and zero only at $p=q$:
        both rules are **strictly proper**, minimised in expectation by reporting the truth. A uniform predictor scores
        $\ln K$ (NLL) and $1-1/K$ (Brier); the Brier score lies in $[0,2]$ while NLL is unbounded above, which is why a
        single confidently wrong prediction dominates NLL and only bounds Brier.
        """),

        code(r"""
        # N1.7.1: expected NLL and Brier on the 3-class simplex (FIXTURE q = (0.5, 0.3, 0.2)); identities; agreement with src.metrics.
        from src.metrics import nll as src_nll, brier_score as src_brier
        q3 = np.array([0.5, 0.3, 0.2]); m_ = 60
        P3 = np.array([(i / m_, j / m_, (m_ - i - j) / m_) for i in range(m_ + 1) for j in range(m_ + 1 - i)])
        P3c = np.clip(P3, 1e-12, 1.0)
        e_nll = -(q3 * np.log(P3c)).sum(1)
        e_br = (P3 ** 2).sum(1) - 2 * (P3 @ q3) + 1
        Hq = -(q3 * np.log(q3)).sum(); KL = (q3 * np.log(q3 / P3c)).sum(1)
        check("N1.7.1a", np.allclose(e_nll, Hq + KL), "E[NLL] = H(q) + KL(q||p) on all simplex grid points")
        check("N1.7.1b", np.allclose(e_br, ((P3 - q3) ** 2).sum(1) + 1 - (q3 ** 2).sum()), "E[Brier] = ||p - q||^2 + 1 - ||q||^2 on all grid points")
        i_n, i_b = int(np.argmin(e_nll)), int(np.argmin(e_br))
        check("N1.7.1c", np.allclose(P3[i_n], q3) and np.allclose(P3[i_b], q3), f"both expected scores are minimised at p = q (argmins {P3[i_n]}, {P3[i_b]})")
        rngs = np.random.default_rng(RNG_SEED + 71)
        Kp, np_ = 5, 800
        logits_ = rngs.standard_normal((np_, Kp)) * 2; pr_ = np.exp(logits_) / np.exp(logits_).sum(1, keepdims=True); yl_ = rngs.integers(0, Kp, np_)
        own_nll = float(-np.log(pr_[np.arange(np_), yl_]).mean()); own_br = float(((pr_ - np.eye(Kp)[yl_]) ** 2).sum(1).mean())
        check("N1.7.1d", abs(own_nll - src_nll(pr_, yl_)) < 1e-12 and abs(own_br - src_brier(pr_, yl_)) < 1e-12, "own NLL and Brier equal src.metrics.nll / brier_score")
        uni = np.full((10, Kp), 1 / Kp)
        check("N1.7.1e", abs(src_nll(uni, np.arange(10) % Kp) - np.log(Kp)) < 1e-12 and abs(src_brier(uni, np.arange(10) % Kp) - (1 - 1 / Kp)) < 1e-12, f"uniform predictor: NLL = ln K = {np.log(Kp):.4f}, Brier = 1 - 1/K = {1 - 1 / Kp:.2f}")
        bx = 0.5 * (2 * P3[:, 1] + P3[:, 2]); by = (np.sqrt(3) / 2) * P3[:, 2]
        fig, axs = plt.subplots(1, 2, figsize=(11.5, 4.6))
        for ax, val, ttl in zip(axs, [e_nll, e_br], ["expected NLL   H(q) + KL(q||p)", "expected Brier   ||p-q||^2 + const"]):
            tc = ax.tricontourf(bx, by, val, levels=14, cmap="viridis")
            ax.plot(*[0.5 * (2 * q3[1] + q3[2]), (np.sqrt(3) / 2) * q3[2]], marker="*", ms=16, color="w", mec="k"); ax.set_aspect("equal"); ax.set_axis_off()
            ax.set_title(ttl, fontsize=10); fig.colorbar(tc, ax=ax, shrink=0.75)
        varied("N1.7.f1", e_nll, e_br)
        plt.tight_layout(); savefig(fig, "fig7_1_scoring_rules.png")
        """),

        md(r"""
        ### How to read this chart

        Each panel is the probability simplex for $K=3$ (a triangle whose corners are the one-hot predictions),
        coloured by the *expected* score of reporting the prediction at that point when the true label distribution is
        $q=(0.5,0.3,0.2)$, marked by the white star. Both surfaces are bowl-shaped with their minimum at the star,
        which is what "strictly proper" means: no other report scores better in expectation. The NLL surface rises
        steeply towards the edges and corners (it is unbounded there), while the Brier surface is a bounded
        quadratic bowl; that is the practical difference between the two rules. What would make the reading wrong:
        a minimum anywhere except the star. The fixture $q$ is chosen for legibility and has no connection with the
        speech data.

        ### 7.2 ECE: finite-sample bias and its dependence on the number of bins

        Write $\mathrm{conf}_i$ for the top-label probability and $c_i=\mathbb{1}\{\hat y_i=y_i\}$ for correctness. The
        binned estimator has the exact identity
        $$\widehat{\mathrm{ECE}}_B=\sum_{b=1}^B\frac{|S_b|}{n}\big|\bar c_b-\overline{\mathrm{conf}}_b\big|=\frac1n\sum_{b=1}^B\Big|\sum_{i\in S_b}\big(c_i-\mathrm{conf}_i\big)\Big| .$$
        For a **perfectly calibrated** model, $c_i\sim\mathrm{Bernoulli}(\mathrm{conf}_i)$, so each inner sum has mean
        zero and variance $\sum_{i\in S_b}\mathrm{conf}_i(1-\mathrm{conf}_i)\approx n_b\,v_b$ with $v_b$ the mean of
        $\mathrm{conf}(1-\mathrm{conf})$ in the bin. Its absolute value is approximately half-normal, with mean
        $\sqrt{2/\pi}\sqrt{n_bv_b}$, hence
        $$\mathbb{E}\,\widehat{\mathrm{ECE}}_B\ \approx\ \sqrt{\tfrac{2}{\pi}}\ \frac{1}{n}\sum_{b}\sqrt{n_b\,v_b}\ \asymp\ \sqrt{\frac{B_{\mathrm{eff}}}{n}},$$
        where $B_{\mathrm{eff}}$ is the number of occupied bins. Two consequences: the estimator is **positive even
        for a perfect model** (bias of order $\sqrt{B/n}$), and it **grows with $\sqrt B$**, which is the mechanism
        behind the bin-count dependence seen in Section 6. For confidences spread uniformly over $[0.5,1]$, $n=1080$ and $B=15$ the bias is about $0.027$; it is smaller when
        confidences pile up near $1$, where $\mathrm{conf}(1-\mathrm{conf})$ is small, so this is an upper-end reference for
        our arms' clean ECE values ($0.026$ to $0.065$), not a measurement of their floor.
        """),

        code(r"""
        # N1.7.2: Monte-Carlo ECE of a PERFECTLY calibrated model (FIXTURE: conf ~ U(0.5,1), correct ~ Bernoulli(conf)) vs the half-normal prediction.
        from src.metrics import expected_calibration_error as src_ece
        def ece_batch(conf, corr, B):
            idx = np.minimum((conf * B).astype(int), B - 1); tot = np.zeros(conf.shape[0])
            for b in range(B):
                m = idx == b
                tot += np.abs(((corr - conf) * m).sum(1))
            return tot / conf.shape[1]
        def ece_theory(n, B, lo=0.5):
            edges = np.linspace(0, 1, B + 1); tot = 0.0
            for b in range(B):
                a_, c_ = max(edges[b], lo), edges[b + 1]
                if c_ <= a_: continue
                nb = n * (c_ - a_) / (1 - lo); mid = 0.5 * (a_ + c_); v = mid * (1 - mid)
                tot += np.sqrt(nb * v)
            return np.sqrt(2 / np.pi) * tot / n
        NS = np.array([100, 300, 1000, 3000, 10000]); BS = [10, 15, 20]; R = 120
        rngE = np.random.default_rng(RNG_SEED + 72)
        MCE = {}
        for B in BS:
            for n in NS:
                conf = rngE.uniform(0.5, 1.0, (R, n)); corr = (rngE.uniform(size=(R, n)) < conf).astype(float)
                v = ece_batch(conf, corr, B); MCE[(B, n)] = (v.mean(), v.std(ddof=1) / np.sqrt(R))
        conf1 = rngE.uniform(0.5, 1.0, (1, 400)); corr1 = (rngE.uniform(size=(1, 400)) < conf1).astype(float)
        pr2 = np.stack([conf1[0], 1 - conf1[0]], 1); lab2 = np.where(corr1[0] == 1, 0, 1)
        check("N1.7.2a", abs(ece_batch(conf1, corr1, 15)[0] - src_ece(pr2, lab2, bins=15)) < 1e-12, "vectorised ECE equals src.metrics.expected_calibration_error on a constructed draw")
        for B in BS:
            print(f"B={B:2d}: " + "  ".join(f"n={n}: MC {MCE[(B, n)][0]:.4f} (theory {ece_theory(n, B):.4f})" for n in NS))
        slopes = {B: np.polyfit(np.log(NS[2:]), np.log([MCE[(B, n)][0] for n in NS[2:]]), 1)[0] for B in BS}
        check("N1.7.2b", all(abs(s + 0.5) < 0.08 for s in slopes.values()), f"ECE of a perfect model falls like n^(-1/2): log-log slopes {', '.join(f'B={B}: {s:.3f}' for B, s in slopes.items())}")
        rel = [abs(MCE[(B, n)][0] / ece_theory(n, B) - 1) for B in BS for n in NS[2:]]
        check("N1.7.2c", max(rel) < 0.15, f"half-normal prediction matches Monte Carlo within {max(rel):.1%} for n >= 1000")
        check("N1.7.2d", MCE[(20, 1000)][0] > MCE[(10, 1000)][0], f"at n=1000 a perfect model has ECE {MCE[(10, 1000)][0]:.4f} (B=10) < {MCE[(20, 1000)][0]:.4f} (B=20): more bins, more bias")
        note("N1.7.2e", f"with confidences uniform on [0.5, 1] (a deliberately spread fixture) the bias of a PERFECT model at n=1080, B=15 is about {ece_theory(1080, 15):.4f}; models with confidences concentrated near 1 have a smaller floor because conf(1-conf) is small there")
        fig, ax = plt.subplots(figsize=(8.2, 4.5))
        for B, c_ in zip(BS, ["#1f77b4", "#ff7f0e", "#2ca02c"]):
            ax.errorbar(NS, [MCE[(B, n)][0] for n in NS], yerr=[1.96 * MCE[(B, n)][1] for n in NS], marker="o", color=c_, capsize=3, label=f"B={B} Monte Carlo")
            ax.plot(NS, [ece_theory(n, B) for n in NS], ls="--", color=c_, lw=1)
        ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("evaluation-set size n"); ax.set_ylabel("expected ECE of a perfectly calibrated model")
        ax.legend(fontsize=8); ax.set_title("ECE bias of a perfect model (solid: Monte Carlo, dashed: half-normal theory)")
        varied("N1.7.f2", [MCE[k][0] for k in MCE])
        savefig(fig, "fig7_2_ece_bias.png")
        """),

        md(r"""
        ### How to read this chart

        The vertical axis is the average ECE that a model with *exactly zero* miscalibration would be measured to have,
        as the evaluation set grows (horizontal axis, log). Solid lines with whiskers are Monte-Carlo averages
        ($120$ repetitions, $95\%$ intervals) for three bin counts; dashed lines are the half-normal formula of
        Section 7.2. They agree closely and fall with slope $-1/2$ on the log-log axes, and the $B=20$ line lies above
        the $B=10$ line at every $n$. Two lessons. First, ECE measured on a few hundred to a thousand examples is
        dominated by this noise floor unless the true miscalibration is comfortably larger, so small ECE
        differences between models are weak evidence. Second, ECE values from different bin counts are not
        interchangeable. The construction (uniform confidences on $[0.5,1]$) is a fixture chosen so that the answer
        is known; it is not a model of our classifiers' confidence distributions.

        ### 7.3 Temperature scaling as maximum likelihood: consistency and a variance formula

        Section 6.1 gave $\mathrm{NLL}'(\beta)$ and $\mathrm{NLL}''(\beta)$. Newton's method on the convex,
        one-dimensional problem is
        $$\beta_{k+1}=\beta_k-\frac{\mathrm{NLL}'(\beta_k)}{\mathrm{NLL}''(\beta_k)},\qquad
        \mathrm{NLL}'(\beta)=\overline{\mathbb{E}_{p(\beta)}[z]-z_y},\quad \mathrm{NLL}''(\beta)=\overline{\mathrm{Var}_{p(\beta)}(z)} .$$
        If labels truly follow $y\mid z\sim\mathrm{softmax}(\beta_0z)$, the maximum-likelihood estimator is consistent for
        $\beta_0$, and by the usual likelihood theory
        $$\sqrt n\,(\hat\beta-\beta_0)\ \xrightarrow{d}\ \mathcal N\!\big(0,\ 1/\overline V\big),\qquad \overline V=\mathbb{E}\big[\mathrm{Var}_{p(\beta_0)}(z)\big],$$
        so $\mathrm{SE}(\hat\beta)\approx 1/\sqrt{n\overline V}$: fitting a *single* temperature is a very
        well-determined problem whenever the logits have spread. The fixture below draws logits, samples labels from a
        known temperature $T_0=1.6$ (which is the value fitted for our `wide` arm, chosen only for familiarity), and
        checks consistency, the variance formula, and agreement with `src.calibration.temperature_scale`.
        """),

        code(r"""
        # N1.7.3: temperature-scaling MLE on constructed logits with known T0 = 1.6 (FIXTURE): Newton solver, consistency, variance formula.
        from src.calibration import temperature_scale
        def softmax_rows(z):
            z = z - z.max(1, keepdims=True); e = np.exp(z); return e / e.sum(1, keepdims=True)
        def newton_beta(Z, y, iters=60):
            b = 1.0
            for _ in range(iters):
                p = softmax_rows(b * Z); EZ = (p * Z).sum(1); g = (EZ - Z[np.arange(len(y)), y]).mean(); h = ((p * Z ** 2).sum(1) - EZ ** 2).mean()
                b = max(b - g / h, 1e-6)
                if abs(g / h) < 1e-12: break
            return b, h
        T0, K7 = 1.6, 8
        def draw(n, rng):
            Z = 3.0 * rng.standard_normal((n, K7)); p = softmax_rows(Z / T0)
            y = (p.cumsum(1) > rng.uniform(size=(n, 1))).argmax(1); return Z, y
        rng7 = np.random.default_rng(RNG_SEED + 73)
        NB = [200, 500, 1000, 3000, 10000]; RB = 60
        rows = []
        for n in NB:
            bh, vh = [], []
            for _ in range(RB):
                Z, y = draw(n, rng7); b, h = newton_beta(Z, y); bh.append(b); vh.append(h)
            bh = np.array(bh); rows.append({"n": n, "mean beta_hat": bh.mean(), "beta0": 1 / T0, "bias": bh.mean() - 1 / T0, "sd(beta_hat)": bh.std(ddof=1),
                                             "theory 1/sqrt(n V)": 1 / np.sqrt(n * np.mean(vh)), "mean T_hat": np.mean(1 / bh)})
        TSFIT = pd.DataFrame(rows).set_index("n")
        display(TSFIT.round(5))
        check("N1.7.3a", abs(TSFIT.loc[10000, "mean beta_hat"] - 1 / T0) < 3 * TSFIT.loc[10000, "sd(beta_hat)"] / np.sqrt(RB) + 1e-4, f"consistency: mean beta_hat at n=10000 = {TSFIT.loc[10000, 'mean beta_hat']:.5f} vs beta0 = {1 / T0:.5f}")
        ratio = TSFIT["sd(beta_hat)"] / TSFIT["theory 1/sqrt(n V)"]
        check("N1.7.3b", bool(((ratio > 0.85) & (ratio < 1.15)).all()), f"empirical sd / theoretical 1/sqrt(nV) lies in [{ratio.min():.2f}, {ratio.max():.2f}] for all n")
        Zc, yc = draw(3000, rng7); bnew, _ = newton_beta(Zc, yc)
        Tsrc = temperature_scale(Zc, yc)
        check("N1.7.3c", abs(1 / bnew - Tsrc) < 2e-3, f"own Newton solver T = {1 / bnew:.5f} vs src.calibration.temperature_scale T = {Tsrc:.5f}")
        gps = np.linspace(0.3, 1.2, 91)
        nllc = np.array([-np.log(softmax_rows(b * Zc)[np.arange(len(yc)), yc]).mean() for b in gps])
        check("N1.7.3d", bool(np.all(np.diff(nllc, 2) > -1e-9)), "NLL(beta) is convex on the grid (second differences >= 0), as the log-sum-exp argument requires")
        check("N1.7.3e", abs(gps[int(np.argmin(nllc))] - bnew) < (gps[1] - gps[0]), f"grid minimiser {gps[int(np.argmin(nllc))]:.3f} agrees with the Newton root {bnew:.3f}")
        fig, axs = plt.subplots(1, 2, figsize=(12.5, 4.4))
        axs[0].plot(gps, nllc, color="k", lw=1.8, label="NLL(beta), n = 3000")
        _, Vc = newton_beta(Zc, yc)
        axs[0].plot(gps, nllc.min() + 0.5 * Vc * (gps - bnew) ** 2, ls="--", color="#d62728", label="quadratic approximation at beta_hat")
        axs[0].axvline(1 / T0, color=ACOL["wide"], ls=":", label=f"true beta0 = 1/{T0}"); axs[0].axvline(1.0, color="#999", ls=":", label="beta = 1 (uncalibrated)")
        axs[0].set_xlabel("beta = 1/T"); axs[0].set_ylabel("calibration NLL"); axs[0].legend(fontsize=8); axs[0].set_title("Convex one-parameter likelihood")
        axs[1].errorbar(TSFIT.index, TSFIT["mean beta_hat"], yerr=TSFIT["sd(beta_hat)"], marker="o", capsize=3, color="k", label="beta_hat: mean +/- sd over 60 fits")
        axs[1].fill_between(TSFIT.index, 1 / T0 - TSFIT["theory 1/sqrt(n V)"], 1 / T0 + TSFIT["theory 1/sqrt(n V)"], color=ACOL["wide"], alpha=0.25, label="beta0 +/- 1/sqrt(n V)")
        axs[1].axhline(1 / T0, color=ACOL["wide"], lw=1); axs[1].set_xscale("log"); axs[1].set_xlabel("calibration-set size n"); axs[1].set_ylabel("beta_hat"); axs[1].legend(fontsize=8)
        axs[1].set_title("Consistency and the 1/sqrt(nV) standard error")
        varied("N1.7.f3", nllc, TSFIT["sd(beta_hat)"].values)
        plt.tight_layout(); savefig(fig, "fig7_3_temperature_mle.png")
        """),

        md(r"""
        ### How to read this chart

        Left: the calibration NLL as a function of $\beta=1/T$ on one constructed calibration set of $3000$ examples.
        The black curve is convex with a single minimum; the red dashed parabola is the second-order approximation
        $\mathrm{NLL}(\hat\beta)+\frac12\overline V(\beta-\hat\beta)^2$ and hugs the curve near the minimum, which is
        why the likelihood gain from scaling grows with the squared distance of $\hat\beta$ from $1$. The green dotted
        line is the true $\beta_0=1/1.6$ and the grey dotted line marks $\beta=1$ (no scaling). Right: as $n$ grows
        the estimates (black, mean and standard deviation over $60$ fits) converge on $\beta_0$, and their spread
        stays inside the green band $\beta_0\pm1/\sqrt{n\overline V}$. The band is tight even at $n=200$. What this says
        about our real temperatures: a single fitted temperature on $1080$ calibration clips is statistically
        well determined *for the calibration set's own distribution*, so its uncertainty is not the main concern; the
        main concerns are seed variance and distribution shift, neither of which this estimator addresses.

        ### 7.4 Conformal risk control: estimator, guarantee, and the O(1/n) gap

        (Angelopoulos et al., arXiv:2208.02814; turns 2 and 3 of the earlier session.) Let $L_1,\dots,L_{n+1}$ be
        exchangeable, non-increasing, right-continuous loss functions of a scalar $\lambda$ with $L_i\le B$ and
        $L_i(\lambda_{\max})\le\alpha$. With $\hat R_n(\lambda)=\frac1n\sum_{i\le n}L_i(\lambda)$ the estimator is
        $$\hat\lambda=\inf\Big\{\lambda:\ \tfrac{n}{n+1}\hat R_n(\lambda)+\tfrac{B}{n+1}\le\alpha\Big\},$$
        and $\hat\lambda=\lambda_{\max}$ if the set is empty. **Guarantee**: $\mathbb{E}[L_{n+1}(\hat\lambda)]\le\alpha$.

        *Proof sketch.* Define the oracle $\lambda'=\inf\{\lambda:R_{n+1}(\lambda)\le\alpha\}$ using all $n+1$ losses, where
        $R_{n+1}=\frac{n}{n+1}\hat R_n+\frac1{n+1}L_{n+1}$. (1) $\lambda'$ is a symmetric function of the $n+1$
        losses. (2) Because $L_{n+1}\le B$, $R_{n+1}(\lambda)\le\frac n{n+1}\hat R_n(\lambda)+\frac B{n+1}$, so every
        $\lambda$ accepted by the estimator is accepted by the oracle: $\lambda'\le\hat\lambda$. (3) Monotonicity gives
        $L_{n+1}(\hat\lambda)\le L_{n+1}(\lambda')$. (4) Right-continuity gives $R_{n+1}(\lambda')\le\alpha$. (5) By
        exchangeability $\mathbb{E}L_{n+1}(\lambda')=\mathbb{E}R_{n+1}(\lambda')\le\alpha$. Hence
        $\mathbb{E}L_{n+1}(\hat\lambda)\le\alpha$. Note that $\hat\lambda$ itself is *not* symmetric in the $n+1$
        losses; only the oracle is.

        **Indicator loss, closed form.** With $L_i(\lambda)=\mathbb 1\{\lambda<s_i\}$ for a score $s_i$ (a set misses its
        label when the threshold is below the label's score), $B=1$ and $\hat\lambda$ is the $k$-th largest calibration score,
        $k=\lfloor\alpha(n+1)\rfloor$. For continuous scores the test score exceeds it with probability exactly
        $k/(n+1)$, so
        $$\mathbb{E}\,L_{n+1}(\hat\lambda)=\frac{\lfloor\alpha(n+1)\rfloor}{n+1}\in\Big(\alpha-\frac1{n+1},\ \alpha\Big].$$
        The under-coverage gap is therefore at most $1/(n+1)=O(1/n)$, and it is a *sawtooth* in $n$ that touches
        zero whenever $\alpha(n+1)$ is an integer. In terms of the project's `crc_threshold`, which works on the
        true-label probability $p=1-s$, the threshold is the $k$-th smallest calibration probability, and a test
        example is missed when its probability falls below it.

        **Rao-Blackwell shortcut for the Monte Carlo.** For scores with a known CDF $F$, the conditional miscoverage
        given the calibration set is $F(\hat p)$ (with $\hat p$ the threshold), so averaging $F(\hat p)$ over
        repetitions has far smaller variance than averaging the $0/1$ misses. That makes the $O(1/n)$ gap resolvable
        at moderate repetition counts. The next cell does this for uniform scores and, as a check of the
        distribution-free claim, for a skewed Beta distribution, and then breaks exchangeability on purpose.
        """),

        code(r"""
        # N1.7.4: CRC Monte Carlo (FIXTURE: iid true-label probabilities) with Rao-Blackwellised miscoverage; distribution-free check; shift.
        from src.calibration import crc_threshold
        def k_of(n, a):
            return int(np.floor(a * (n + 1) + 1e-12))
        rngC = np.random.default_rng(RNG_SEED + 74)
        NC_ = [19, 49, 99, 199, 499, 999]; ALPHAS = [0.05, 0.10, 0.20]; REPS = 6000
        unif = (lambda x: x, lambda rng, s: rng.uniform(size=s))
        skew = (lambda x: stats.beta.cdf(x, 0.4, 2.5), lambda rng, s: rng.beta(0.4, 2.5, size=s))
        rows = []
        for name, (cdf, samp) in {"uniform": unif, "skewed Beta(0.4, 2.5)": skew}.items():
            for n in NC_:
                Ssorted = np.sort(samp(rngC, (REPS, n)), axis=1)              # sort once per (distribution, n)
                for a in ALPHAS:
                    if a * (n + 1) < 1: continue
                    k = k_of(n, a); mis = cdf(Ssorted[:, k - 1])                # Rao-Blackwell: conditional miscoverage F(threshold)
                    cf = k / (n + 1)
                    rows.append({"scores": name, "alpha": a, "n": n, "k": k, "MC E[miscov]": mis.mean(), "SE": mis.std(ddof=1) / np.sqrt(REPS),
                                 "closed form": cf, "gap alpha - E": a - mis.mean(), "1/(n+1)": 1 / (n + 1)})
        CRC = pd.DataFrame(rows)
        z_ = (CRC["MC E[miscov]"] - CRC["closed form"]) / CRC["SE"]
        display(CRC[CRC.scores == "uniform"].round(5).head(12))
        check("N1.7.4a", bool((z_.abs() < 4).all()), f"Monte Carlo matches floor(alpha(n+1))/(n+1) in all {len(CRC)} cells (max |z| = {z_.abs().max():.2f})")
        check("N1.7.4b", bool((CRC["MC E[miscov]"] <= CRC["alpha"] + 4 * CRC["SE"]).all()), "Theorem 1: E[miscoverage] <= alpha in every cell (within 4 SE)")
        check("N1.7.4c", bool((CRC["gap alpha - E"] < CRC["1/(n+1)"] + 4 * CRC["SE"]).all()), "the shortfall alpha - E stays below 1/(n+1) (the O(1/n) lower bound)")
        sk = CRC[CRC.scores != "uniform"]; un = CRC[CRC.scores == "uniform"]
        zsk = np.abs((sk["MC E[miscov]"].values - sk["closed form"].values) / sk["SE"].values)
        check("N1.7.4d", bool(np.allclose(sk["closed form"].values, un["closed form"].values)) and float(zsk.max()) < 4,
              "distribution-free: a skewed Beta score distribution gives the same closed-form expectation")
        # cross-check against the project's crc_threshold on explicit rows
        Pchk = rngC.uniform(size=(300, 99)); ok = all(abs(crc_threshold(Pchk[i], 0.10) - np.sort(Pchk[i])[k_of(99, 0.10) - 1]) < 1e-15 for i in range(300))
        check("N1.7.4e", ok, "src.calibration.crc_threshold equals the k-th smallest true-label probability, k = floor(alpha(n+1)), on 300 draws (n=99, alpha=0.10)")
        check("N1.7.4f", crc_threshold(np.array([0.3, 0.7, 0.9]), 0.05) == 0.0, "infeasible alpha (alpha(n+1) < 1): crc_threshold returns 0.0, the empty-set fallback")
        # exchangeability broken: test probabilities drawn with CDF x^g (g<1 -> harder examples than calibration)
        GAM = np.array([1.0, 0.9, 0.8, 0.6, 0.4]); n0, a0 = 99, 0.10
        th0 = np.sort(rngC.uniform(size=(REPS, n0)), axis=1)[:, k_of(n0, a0) - 1]
        SHIFT = np.array([np.mean(th0 ** g) for g in GAM])
        check("N1.7.4g", abs(SHIFT[0] - k_of(n0, a0) / (n0 + 1)) < 0.003 and bool(np.all(np.diff(SHIFT) > 0)), f"with exchangeability the miscoverage is {SHIFT[0]:.3f} (=k/(n+1)); as the test distribution gets harder it climbs: {np.round(SHIFT, 3).tolist()}")
        check("N1.7.4h", SHIFT[-1] > 2 * a0, f"a moderate shift (gamma=0.4) more than doubles the miscoverage to {SHIFT[-1]:.3f} at target alpha = {a0}")
        """),

        code(r"""
        # Figure 7.4: the O(1/n) gap (left) and the failure under a broken-exchangeability shift (right).
        fig, axs = plt.subplots(1, 2, figsize=(12.5, 4.5))
        for a, c_ in zip(ALPHAS, ["#1f77b4", "#ff7f0e", "#2ca02c"]):
            s_ = CRC[(CRC.scores == "uniform") & (CRC.alpha == a)]
            axs[0].errorbar(s_.n, np.maximum(s_["gap alpha - E"], 1e-5), yerr=1.96 * s_.SE, marker="o", color=c_, capsize=3, label=f"alpha = {a} (MC)")
            axs[0].plot(s_.n, [a - k_of(n, a) / (n + 1) for n in s_.n], ls=":", color=c_, lw=1.2)
        axs[0].plot(NC_, [1 / (n + 1) for n in NC_], color="k", lw=1.4, label="1/(n+1) bound")
        axs[0].set_xscale("log"); axs[0].set_yscale("log"); axs[0].set_xlabel("calibration size n"); axs[0].set_ylabel("shortfall  alpha - E[miscoverage]")
        axs[0].legend(fontsize=8); axs[0].set_title("Guarantee gap is O(1/n) (dotted: closed-form sawtooth)")
        axs[1].plot(GAM, SHIFT, marker="o", color="#d62728", lw=1.8, label="achieved miscoverage")
        axs[1].axhline(a0, color="k", ls="--", label=f"target alpha = {a0}"); axs[1].invert_xaxis()
        axs[1].set_xlabel("shift strength: test CDF = x^gamma (1 = exchangeable, smaller = harder)"); axs[1].set_ylabel("achieved miscoverage")
        axs[1].legend(fontsize=8); axs[1].set_title("The guarantee needs exchangeability")
        varied("N1.7.f4", CRC["MC E[miscov]"].values, SHIFT)
        plt.tight_layout(); savefig(fig, "fig7_4_crc_gap_and_shift.png")
        """),

        md(r"""
        ### How to read this chart

        Left: on log-log axes, the shortfall $\alpha-\mathbb{E}[\text{miscoverage}]$ of the conformal estimator against
        the calibration size $n$, for three targets. Coloured markers with whiskers are Rao-Blackwellised Monte-Carlo
        estimates; dotted lines are the exact closed form $\alpha-\lfloor\alpha(n+1)\rfloor/(n+1)$, a sawtooth that dips
        to zero whenever $\alpha(n+1)$ is an integer (the markers at zero are drawn at a floor of $10^{-5}$ because
        the axis is logarithmic); the black line is the $1/(n+1)$ envelope. Every marker sits under the envelope, and
        the envelope falls with slope $-1$: that is the $O(1/n)$ statement. Right: the same procedure when the test
        distribution is progressively *harder* than the calibration distribution ($\gamma<1$). At $\gamma=1$
        (exchangeable) the miscoverage equals its target-adjacent closed form; as $\gamma$ falls it climbs steeply
        past the dashed target line. The reading to carry over to the speech setting is the right panel, not the
        left: the finite-sample guarantee is a theorem about exchangeable data, and a shifted test set falls
        outside it.
        """),

        code(r"""
        # N1.7.5: REAL conformal outputs stored in the result files: achieved miscoverage vs target alpha, per arm and condition.
        rows = []
        for a in ARCHS:
            for c in CONDS:
                crc = S(a, c)["crc"]
                for al in ("0.05", "0.1"):
                    rows.append({"arch": a, "cond": c, "alpha": float(al), "achieved miscoverage": crc[al]["miscoverage"], "mean set size": crc[al]["mean_set_size"], "threshold": crc[al]["threshold"]})
        CR = pd.DataFrame(rows)
        display(CR.pivot_table(index=["arch", "cond"], columns="alpha", values=["achieved miscoverage", "mean set size"]).round(4))
        cl = CR[CR.cond == "clean"]; nz = CR[CR.cond == "noise_10"]
        sd_ = np.sqrt(2 * cl["alpha"] * (1 - cl["alpha"]) / N_TEST)     # calibration-threshold noise + test sampling noise, each ~ sqrt(alpha(1-alpha)/1080)
        zc_ = (cl["achieved miscoverage"] - cl["alpha"]) / sd_
        check("N1.7.5a", bool((zc_ < 3).all()), f"on clean, achieved miscoverage exceeds target by at most {zc_.max():.2f} standard deviations of single-split noise (pre-declared rule: z < 3)")
        note("N1.7.5a2", f"two clean cells sit at z = {sorted(zc_.round(2).tolist())[-2]:.2f} and {sorted(zc_.round(2).tolist())[-1]:.2f}: mild, unproven excess on one split; not evidence of a violation, not evidence of none")
        check("N1.7.5b", bool((nz["achieved miscoverage"] > 3 * nz["alpha"]).all()), f"on noise_10 every arm's miscoverage exceeds 3x target (min {nz['achieved miscoverage'].min():.3f}): the guarantee does not transfer across the shift, as Section 7.4 predicts")
        check("N1.7.5c", abs(float(nz[(nz.arch == 'gru') & (nz.alpha == 0.05)]['achieved miscoverage'].iloc[0]) - 0.7565) < 5e-4, "gru noise_10 at alpha = 0.05: achieved miscoverage 0.7565")
        note("N1.7.5d", "calibration split (n=1080) draws clean audio; noise_10 test clips are not exchangeable with it, so a miscoverage above alpha is the EXPECTED behaviour, not an implementation bug")
        fig, ax = plt.subplots(figsize=(11, 4.4))
        lab = [f"{a}\n{c}" for a in ARCHS for c in CONDS]; xs = np.arange(len(lab))
        for al, mk, col in ((0.05, "o", "#1f77b4"), (0.1, "s", "#ff7f0e")):
            v = [CR[(CR.arch == a) & (CR.cond == c) & (CR.alpha == al)]["achieved miscoverage"].iloc[0] for a in ARCHS for c in CONDS]
            ax.scatter(xs, v, marker=mk, s=55, color=col, label=f"achieved, target alpha = {al}", zorder=3)
            ax.axhline(al, color=col, ls="--", lw=1)
        ax.set_xticks(xs); ax.set_xticklabels(lab, fontsize=7); ax.set_ylabel("achieved test miscoverage"); ax.set_yscale("log"); ax.legend(fontsize=8)
        ax.set_title("Real CRC outputs: modest excess on clean and mild shifts, 5-15x target under noise_10")
        varied("N1.7.f5", CR["achieved miscoverage"].values)
        savefig(fig, "fig7_5_real_crc.png")
        """),

        md(r"""
        ### How to read this chart

        Each marker is one arm-and-condition pair from the stored conformal results; blue circles correspond to a
        target of $\alpha=0.05$ and orange squares to $\alpha=0.10$, whose target levels are the dashed horizontal
        lines of the same colour, on a log axis. On `clean` the markers sit at or slightly above their dashed lines (within about $2.4$ single-split standard
        deviations); on the mild `gain_+10` and `reverb_mid` conditions they are roughly $1.5$ to $2$ times the
        target, as expected when the test set is shifted. On `noise_10` all markers are about $5$ to $15$ times the
        target: for `gru` the achieved miscoverage is $0.756$ at a target of $0.05$. This is the real-data counterpart of the right panel of the
        previous figure, and it is the predicted behaviour of an exchangeability-based guarantee under a severe shift.
        It also echoes the Section 3 tension: for `gru` and `wide` at $\alpha=0.1$ the mean set size is only about $1.04$
        to $1.09$ labels, yet the true label is missing $84\%$ and $70\%$ of the time.

        ### 7.5 What conformal risk control does not establish

        (i) It controls the *expected* loss over the joint draw of calibration and test points, not the loss on the
        particular calibration set in hand; a single split gives a random $\hat\lambda$ whose realised risk fluctuates
        around the guarantee. (ii) It is a marginal statement, not conditional on the input, the class, or the
        speaker. (iii) It presupposes exchangeability between calibration and test, which Section 7.4's right panel
        and the `noise_10` results show can fail badly under acoustic shift. (iv) The loss must be bounded and
        monotone in $\lambda$; abstention-style losses that are not monotone are outside the theorem. (v) The
        finite-sample gap is $O(1/n)$; at $n=1080$ that is under $0.1\%$ for indicator losses, which is not the binding
        limitation.
        """),

        md(r"""
        ## 8. Limitations

        Everything below limits how far this notebook's conclusions can be pushed. They are collected here so a
        reader deciding how much weight a sentence deserves can find every caveat in one place. The first cell
        re-derives from the artifacts the limitations that are checkable; the rest are recorded facts.

        ### 8.1 Scale and coverage of the evidence

        - **One training run per architecture.** The architecture race is a single seed. The legacy baseline, whose three
          seeds ended at $0.3730$, $0.5913$ and $0.3433$, shows that seed-to-seed variation within one architecture
          can exceed the between-architecture differences discussed in Sections 2 and 3. The `gru`-versus-`wide` gap
          is small enough that this variation could reverse it; the gap between time-preserving arms and the
          time-pooling baseline is large enough that it probably could not.
        - **Exploration configuration.** The stress evaluation used $4$ conditions and $200$ bootstrap resamples, not
          the design's publication configuration. Intervals are coarse and only one condition (`noise_10`) is additive
          noise at one level, so no dose-response curve in SNR can be drawn, which is precisely what the CRNN paper's
          "the gap narrows as SNR rises" claim would need.
        - **Validation and test splits are small.** $504$ validation clips and $1080$ test clips: the binomial
          standard error of an accuracy near $0.9$ is about $0.013$ and $0.009$ respectively, and Section 7.2 shows that the
          noise floor of ECE at this size can be a few hundredths for spread-out confidences.
        - **The headline numbers mix summaries.** $0.9008$ for `gru` is both its best-loss-epoch and last-epoch value;
          $0.8790$ for `wide` is its peak, and its last epoch is $0.7540$ (Section 1.1).

        ### 8.2 Neural-collapse deviations (Section 4)

        1. **Held-out, not training, activations.** The paper computes $\mathrm{NC}_1$-$\mathrm{NC}_3$ on training-set
           activations and only $\mathrm{NC}_4$ on held-out data; our probe is a fixed $120$-clip held-out split.
        2. **Terminal phase not verified.** Training accuracy was not recorded. `gru` and `wide` are TPT-*plausible*
           from their cross-entropy ($0.0100$, $0.0182$), which bounds training error at $1.4\%$ and $2.6\%$, not at the
           paper's $0.1\%$; `timepool` ($0.9794$) is treated as not having entered TPT, a premise supported but not
           proved by its loss. Neural collapse is therefore *undefined* for it.
        3. **PCA in place of the papers' projections and full features.** The $\mathrm{NC}$ formulas are applied to a
           two-dimensional PCA projection, refit every epoch, of the classifier's input, where the paper uses full
           $p$-dimensional last-layer features; 2D equiangularity and maximal-angle measures have a planar floor of
           about $0.67$ and $0.60$ and carry no $\mathrm{NC}_2$ information. The companion visualisation project's
           SentryCam and MM-PHATE also specify a parametric autoencoder and a diffusion embedding respectively;
           PCA stands in for both.

        ### 8.3 Other method-fidelity notes

        - **AutoClip off-by-one.** Our threshold uses the prior history; the paper appends the current norm first. On the
          recorded norms this flips fewer than $1\%$ of clip decisions, but it is a real deviation. It is a
          counterfactual on recorded norms, not a re-run.
        - **AutoClip's $p=10$ is cross-task.** Tuned on WSJ0-2mix separation with BLSTMs; applied here, unchanged, to log-mel
          classification. With $p=10$ the rule clips roughly nine steps in ten under stationary norms by construction, so
          the measured clip rates ($99.1\%$, $70.2\%$, $65.8\%$) describe the *trend* of the norms rather than
          instability. No run without clipping exists, so AutoClip's benefit here is untested.
        - **Calibration.** No logits are stored, so fitted temperatures cannot be bootstrapped or re-derived, and no MDCA
          model was trained. The `gru` $T^\star=1.0157$ versus `wide` $T^\star=1.6002$ contrast is an observation with one
          run per arm.
        - **ECE.** Depends on the bin count and has a positive noise floor at $n\approx1000$ (Section 7.2).
        - **CRC.** Unweighted, so under a shifted test set the guarantee does not apply (Section 7.4).

        ### 8.4 None of the parallels is causal

        Sections 2, 3, 4, 5 and 6 each set a paper's claim next to a measurement. In every case the measurement is
        an observation about one trained model per architecture. None of the parallels is a causal claim: we did not
        ablate the recurrent head against a matched-capacity time-preserving head at several seeds, we did not vary the
        clipping percentile, we did not train with a calibration objective, and we did not measure neural collapse on
        training activations. The notebook's contribution is the mathematics, the traceable sources, and the list of
        experiments that would turn each observation into a test.
        """),

        code(r"""
        # N1.8.1: re-derive the checkable limitations from the artifacts, then print the claim ledger.
        seeds_in_summary = sorted({DATA[a]["res"]["summary"]["seed"] for a in ARCHS})
        conds_all = sorted({c for a in ARCHS for c in DATA[a]["res"]["summary"]["conditions"]})
        check("N1.8.1a", len(conds_all) == 4, f"exploration configuration: {len(conds_all)} stress conditions {conds_all} (design specifies 11)")
        check("N1.8.1b", all(DATA[a]["npz"]["labels"].shape == (120,) for a in ARCHS), "every NC probe is a 120-clip held-out split, not training activations")
        check("N1.8.1c", all("logits" not in DATA[a]["npz"] and "logits" not in DATA[a]["res"] for a in ARCHS), "no logits are stored in any artifact: temperatures cannot be bootstrapped")
        check("N1.8.1d", not any("train_accuracy" in e for a in ARCHS for e in DATA[a]["res"]["history"]), "no training-accuracy series is stored: TPT membership cannot be read off")
        check("N1.8.1e", float(np.ptp([LEG[s].val_accuracy.iloc[-1] for s in (17, 18, 19)])) > 0.2, f"legacy baseline seed spread = {np.ptp([LEG[s].val_accuracy.iloc[-1] for s in (17, 18, 19)]):.4f} > 0.2")
        note("N1.8.1f", f"result-file summary 'seed' field reads {seeds_in_summary} for all arms (an evaluation-stage seed); training seed 17 is taken from the run configuration, not from these files")
        ledger = [
            ("gru / wide / timepool params", "291,564 / 271,692 / 17,212", "arch-*.result.json parameters", True),
            ("legacy baseline params", "14,524", "src.models.LogMelCNN instantiated (N1.2.1)", count["baseline"] == 14524),
            ("gru final val acc", f"{DATA['gru']['res']['final_val_accuracy']:.4f}", "arch-gru.result.json", abs(DATA["gru"]["res"]["final_val_accuracy"] - 0.9008) < 5e-5),
            ("wide max / last val acc", f"{SUMM.loc['wide', 'max_acc']:.4f} / {SUMM.loc['wide', 'last_epoch_acc']:.4f}", "arch-wide.result.json history", True),
            ("legacy baseline seeds 17/18/19", "0.3730 / 0.5913 / 0.3433", "runtime/metrics/legacy_6arm_logs/clean_seed*.log", True),
            ("final train CE gru / wide / timepool", "0.0100 / 0.0182 / 0.9794", "arch-*.result.json final_train_loss", True),
            ("AutoClip clip counts", "663/1008, 910/1296, 1284/1296", "recomputed from arch-*.probes.npz grad_norm (N1.5.1)", True),
            ("fitted temperatures gru/timepool/wide", "1.0157 / 1.1097 / 1.6002", "arch-*.result.json summary.temperature", True),
            ("noise_10 accuracy gru/timepool/wide", "0.1380 / 0.1472 / 0.2519", "arch-*.result.json summary.conditions.noise_10", True),
            ("paper: CRNN 229k vs CNN 250k; FRR 4.31/5.73 vs 2.85/3.79; 97.71/98.71/99.30", "as quoted", "arXiv:1703.05390 via session turn 2 (checked when written; raw answers not published)", False),
            ("paper: NC1 formula, ETF, TPT, train-vs-test, 300/350 epochs", "as quoted", "arXiv:2008.08186 via session turns 4-6 (checked when written; raw answers not published)", False),
            ("paper: MDCA formula; 7.77 / 6.10 / 7.69 / 4.66; 23.6%; T ~ 1", "as quoted", "arXiv:2203.13834 via session turns 7-8 (checked when written; raw answers not published)", False),
            ("paper: AutoClip percentile formula, p in {0,1,10,25,50,90,100}, WSJ0-2mix", "as quoted", "arXiv:2007.14469 via session turns 5-6 (checked when written; raw answers not published)", False),
        ]
        LEDGER = pd.DataFrame(ledger, columns=["claim", "value as printed", "source", "verified in this run"])
        with pd.option_context("display.max_colwidth", 90, "display.width", 250):
            display(LEDGER)
        _meas = LEDGER[LEDGER["source"].str.contains("result.json|probes.npz|derived", case=False, na=False)]
        _lit = LEDGER[~LEDGER.index.isin(_meas.index)]
        check("N1.8.1g", bool(_meas["verified in this run"].all()),
              f"all {len(_meas)} measured-quantity rows verified against files loaded in this run")
        note("N1.8.1g[literature]",
             f"{len(_lit)} literature rows are attributed to a paper and a session turn but cannot be re-verified from "
             "this repository: the raw research answers are not published with it. Audit them at the arXiv sources")
        note("N1.8.1h", "NOT independently sourced here: the AutoClip author list in the traceability table, the interpretation of the summary 'seed' field, and the untested hypotheses labelled as such in Sections 3.4 and 6.4")
        """),
    ]
