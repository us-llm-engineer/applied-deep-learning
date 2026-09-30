"""NB1 part 2: section 3 (where our data contradicts the attribution) and section 4 (neural collapse).

Measured quantities come from arch-*.result.json (summary block) and arch-*.probes.npz. Constructed inputs in
section 4 (a simplex ETF and a noise-scaled copy) are labelled unit-test fixtures for the NC formulas only.
"""
from nbkit import md, code


def cells():
    return [
        md(r"""
        ## 3. Where our data contradicts the attribution

        ### 3.1 What the paper says, stated so that it can be tested

        The CRNN paper (Arik et al., arXiv:1703.05390; turns 2 and 3) makes two separable statements.

        1. **A mechanism.** The advantage is attributed to complementary strengths: convolutions capture local
           time-and-frequency structure, the bidirectional recurrent layers capture long-range temporal context,
           and, in the paper's words as returned by the trace, *recurrent layers adapt better to individual noise
           signatures*.
        2. **A dose-response.** The CRNN's advantage **narrows as SNR increases**: at $5$ dB the CNN's false-reject
           rate is about $51\%$ higher, and the CRNN's test accuracy at $0.5$ false alarms per hour rises from
           $97.71\%$ at $5$ dB to $98.71\%$ at $10$ dB and $99.30\%$ at $20$ dB, i.e. cleaner audio leaves less room
           for either model to differ.

        Write $\Delta(s)=\mathrm{acc}_{\text{rec}}(s)-\mathrm{acc}_{\text{conv}}(s)$ for the recurrent model's accuracy
        advantage in condition $s$. The two statements together predict
        $$\Delta(\text{clean})\ \ge 0,\qquad \Delta(\text{noise})\ >\ \Delta(\text{clean}),\qquad \Delta(s)\ \text{decreasing in SNR}.$$
        Our data can test the first two, but only coarsely: the exploration configuration has four conditions
        (`clean`, `gain_+10`, `reverb_mid`, `noise_10`), of which only `noise_10` is an additive-noise condition and
        there is exactly one SNR level. The third prediction (a dose-response curve) cannot be tested at all.

        ### 3.2 What we measured

        Our architecture race was **clean-only**: every arm trained on clean audio and the stress conditions are
        applied at test time only. In that regime the paper's second statement predicts a small `gru`-versus-`wide`
        margin, and that is what we see (about $0.02$ in accuracy on the clean test split). So far the two agree.
        The `noise_10` condition is where the mechanism should show, and it is where the data disagree: `gru` scores
        $0.1380$, the *lowest* of the three arms, and `wide` scores $0.2519$, the *highest*. That is the reverse of
        "recurrent layers adapt better to noise". The next cell loads the numbers for all four conditions with the
        bootstrap intervals recorded in the result files.
        """),

        code(r"""
        # N1.3.1: accuracy by arm x condition (uncalibrated, test split n=1080) with the recorded bootstrap CIs.
        CONDS = ["clean", "gain_+10", "reverb_mid", "noise_10"]
        def S(a, c, kind="uncalibrated"):
            return DATA[a]["res"]["summary"]["conditions"][c][kind]
        N_TEST = DATA["gru"]["res"]["summary"]["n_test"]
        ACC = pd.DataFrame({a: {c: S(a, c)["accuracy"] for c in CONDS} for a in ARCHS})
        LO = pd.DataFrame({a: {c: S(a, c)["ci"]["accuracy"]["lower"] for c in CONDS} for a in ARCHS})
        HI = pd.DataFrame({a: {c: S(a, c)["ci"]["accuracy"]["upper"] for c in CONDS} for a in ARCHS})
        display(ACC.round(4))
        check("N1.3.1a", all(DATA[a]["res"]["summary"]["n_test"] == 1080 for a in ARCHS) and sorted(DATA["gru"]["res"]["summary"]["conditions"]) == sorted(CONDS),
              f"every arm evaluated on n_test={N_TEST} with the same four conditions {CONDS}")
        check("N1.3.1b", [round(ACC.loc["noise_10", a], 4) for a in ("gru", "timepool", "wide")] == [0.1380, 0.1472, 0.2519],
              "noise_10 accuracy: gru 0.1380, timepool 0.1472, wide 0.2519")
        check("N1.3.1c", ACC.loc["noise_10"].idxmin() == "gru" and ACC.loc["noise_10"].idxmax() == "wide", "under noise_10 gru is the worst arm and wide the best")
        check("N1.3.1d", all(LO.loc[c, a] <= ACC.loc[c, a] <= HI.loc[c, a] for a in ARCHS for c in CONDS), "every point estimate lies inside its recorded bootstrap interval")
        CHANCE = 1 / 12
        _mult = ", ".join("%s %.2fx" % (a, ACC.loc["noise_10", a] / CHANCE) for a in ARCHS)
        note("N1.3.1e", f"chance is 1/12 = {CHANCE:.4f}; noise_10 accuracies are {_mult} chance: all three arms are near the floor")
        LN12 = np.log(12)
        nll10 = {a: S(a, "noise_10")["nll"] for a in ARCHS}
        print("noise_10 NLL (uncalibrated):", {a: round(v, 3) for a, v in nll10.items()}, "| uniform predictor NLL = ln 12 =", round(LN12, 4))
        check("N1.3.1f", all(v > LN12 for v in nll10.values()), "under noise_10 every arm has NLL above ln 12: worse than answering with the uniform distribution")
        """),

        md(r"""
        ### 3.3 How big is the contradiction, and of what kind?

        Two different sources of uncertainty must not be confused. The first is **test-set sampling**: with $n=1080$
        test clips and accuracy $\hat p$, the binomial standard error is $\sqrt{\hat p(1-\hat p)/n}$, and the
        difference of two independent arms has variance $\sigma_\Delta^2=\sigma_1^2+\sigma_2^2$. The second is
        **training-run variance** (seed, data order, initialisation), which a single trained model per arm cannot
        estimate at all. The first is small and computable; the second is unknown and, for this project, potentially
        large: the legacy baseline's three seeds ended at $0.3730$, $0.5913$ and $0.3433$ within one architecture.

        The cell below computes $\Delta=\mathrm{acc}_{\text{gru}}-\mathrm{acc}_{\text{wide}}$ per condition with the
        first kind of uncertainty only, and tests the paper's directional prediction against it. The comparison is
        unpaired because per-clip predictions are not stored, so the standard error is conservative in exactly the
        way that pairing would reduce it; that does not change the qualitative reading.
        """),

        code(r"""
        # N1.3.2: Delta(condition) = acc_gru - acc_wide with test-sampling SE only (unpaired binomial).
        def se_bin(p, n=N_TEST):
            return np.sqrt(p * (1 - p) / n)
        rows = []
        for c in CONDS:
            pg, pw = ACC.loc[c, "gru"], ACC.loc[c, "wide"]
            d = pg - pw; se = np.hypot(se_bin(pg), se_bin(pw))
            rows.append({"condition": c, "acc_gru": pg, "acc_wide": pw, "Delta": d, "SE(test sampling)": se, "z": d / se,
                         "95% lo": d - 1.96 * se, "95% hi": d + 1.96 * se})
        DELTA = pd.DataFrame(rows).set_index("condition")
        display(DELTA.round(4))
        d_clean, d_noise = DELTA.loc["clean", "Delta"], DELTA.loc["noise_10", "Delta"]
        check("N1.3.2a", abs(d_clean - 0.0222) < 5e-4, f"Delta(clean) = {d_clean:+.4f}: gru slightly ahead on the clean test split")
        check("N1.3.2b", DELTA.loc["clean", "95% lo"] < 0.0 < DELTA.loc["clean", "95% hi"] or DELTA.loc["clean", "z"] > 1.96,
              f"Delta(clean) z = {DELTA.loc['clean', 'z']:+.2f}: small margin, consistent with the paper's 'gap narrows in clean audio' (not a strong test)")
        check("N1.3.2c", DELTA.loc["noise_10", "z"] < -1.96, f"Delta(noise_10) = {d_noise:+.4f}, z = {DELTA.loc['noise_10', 'z']:+.2f}: wide is ahead by more than test-set sampling can produce")
        note("N1.3.2d", f"paper predicts Delta(noise) > Delta(clean); measured Delta(noise) - Delta(clean) = {d_noise - d_clean:+.4f}: the predicted direction is reversed")
        note("N1.3.2e", "test-set sampling is the ONLY uncertainty in the SEs above; seed-to-seed training variance is unmeasured (one run per arm)")
        """),

        code(r"""
        # Figure 3.1: accuracy by condition and arm with recorded bootstrap CIs; chance line.
        fig, ax = plt.subplots(figsize=(10.5, 4.6))
        w = 0.26
        for k, a in enumerate(ARCHS):
            xs = np.arange(len(CONDS)) + (k - 1) * w
            yv = ACC[a].values; lo = yv - LO[a].values; hi = HI[a].values - yv
            ax.bar(xs, yv, width=w, color=ACOL[a], label=a, yerr=[lo, hi], capsize=3)
            for x_, v_ in zip(xs, yv):
                ax.text(x_, v_ + 0.03, f"{v_:.3f}", ha="center", fontsize=7.5, rotation=90)
        ax.axhline(CHANCE, color="k", ls=":", lw=1); ax.text(3.45, CHANCE + 0.012, "chance 1/12", ha="right", fontsize=8)
        ax.set_xticks(range(len(CONDS))); ax.set_xticklabels(CONDS); ax.set_ylabel("accuracy on test split (n=1080), uncalibrated")
        ax.set_ylim(0, 1.08); ax.legend(ncol=3, loc="upper right")
        ax.set_title("Accuracy per stress condition: gru leads on clean/reverb, trails on noise_10 (single run per arm)")
        varied("N1.3.f1", ACC.values)
        savefig(fig, "fig3_1_accuracy_by_condition.png")
        """),

        md(r"""
        ### How to read this chart

        Each group of three bars is one test condition; bar height is test accuracy, whiskers are the bootstrap
        interval recorded in the result files (exploration configuration: $200$ resamples, so the whiskers are
        coarse), the dotted line is chance at $1/12$. Read across a group to compare architectures under the *same*
        condition. On `clean` and `reverb_mid` the blue `gru` bar is at or above the green `wide` bar; on `gain_+10`
        they are level; on `noise_10` the ordering flips, and all three bars collapse to a height that is only
        $1.7$ to $3.0$ times chance. The point of the chart is the **flip**, not the absolute heights: a mechanism
        of the form "recurrent layers adapt to noise" predicts that the blue bar rises relative to the green bar
        when moving from `clean` to `noise_10`, and it falls. What would make this reading wrong: intervals that
        overlap heavily in the `noise_10` group (they do not; the whiskers for `gru` and `wide` there are separated by
        several bootstrap half-widths) or a second seed that reversed the order (unknown).
        """),

        code(r"""
        # Figure 3.2: Delta = acc_gru - acc_wide per condition, with test-sampling-only 95% intervals, against the paper's predicted sign.
        fig, ax = plt.subplots(figsize=(8.6, 4.4))
        xs = np.arange(len(CONDS))
        ax.errorbar(xs, DELTA["Delta"], yerr=1.96 * DELTA["SE(test sampling)"], fmt="o", color=ACOL["gru"], capsize=5, ms=8, label="measured Delta (95%, test-set sampling only)")
        ax.axhline(0, color="k", lw=0.9)
        ax.fill_between([-0.4, 3.4], 0, 0.3, color=PALETTE["band"], alpha=0.25, label="paper's predicted region under noise (Delta > 0)")
        for x_, v_ in zip(xs, DELTA["Delta"]):
            ax.text(x_ + 0.08, v_ + 0.012, f"{v_:+.3f}", fontsize=9)
        ax.set_xticks(xs); ax.set_xticklabels(CONDS); ax.set_xlim(-0.4, 3.4); ax.set_ylim(-0.2, 0.15)
        ax.set_ylabel("acc(gru) - acc(wide)"); ax.legend(loc="lower left", fontsize=8)
        ax.set_title("The recurrent advantage under each condition")
        varied("N1.3.f2", DELTA["Delta"].values)
        savefig(fig, "fig3_2_gru_minus_wide.png")
        """),

        md(r"""
        ### How to read this chart

        Each dot is $\Delta=\mathrm{acc}_{\text{gru}}-\mathrm{acc}_{\text{wide}}$ for one condition; the vertical bar is
        a $95\%$ interval that includes **only** test-set sampling noise, so it is a floor on the true uncertainty, not
        a full interval. The shaded band is where the paper's mechanism would put the `noise_10` point (positive, and
        larger than on `clean`). The `clean` dot is slightly positive with an interval that reaches zero; the `noise_10`
        dot sits far below zero, outside the band, by roughly seven of its own standard errors. If the dot were a
        random fluctuation of the test set alone it would not be there; that leaves training-run variance (unknown),
        a genuine architectural effect, or an interaction with the specific perturbation as candidate explanations,
        and this notebook cannot separate them.

        ### 3.4 What this does and does not establish

        **What is established.** In this run, with these weights, `gru` is not more robust to `noise_10` than the
        time-preserving `wide` CNN; it is less robust, by more than test-set sampling explains, and its probabilities
        are the worst-behaved (NLL $4.467$ against $4.071$ for `wide`, with ECE at $0.747$: confidently wrong).

        **What is not established.** Whether this is a property of the architectures or of this training run. One
        seed cannot settle it, and the exploration configuration has exactly one additive-noise condition. The paper's
        statement is about noise signatures the model was *exposed to*, and the grounding does not say whether the
        paper's noisy conditions were seen at training time; ours were not. So we report a tension, not a refutation.

        **An untested hypothesis, labelled as such.** A bidirectional GRU integrates the whole clip, so a stationary
        broadband noise floor shifts *every* input to the recurrence and the perturbation can accumulate in the
        state, whereas a convolutional trunk with a short receptive field (about $12$ frames in the baseline, Section
        2) treats each patch locally and then pools coarsely. Nothing in this notebook tests that. It is recorded so
        that a future experiment (noise-augmented training of both arms across several seeds) has a concrete
        statement to confirm or reject.
        """),

        code(r"""
        # N1.3.3: how many training seeds would it take to see a difference this size? Illustrative sigma values, clearly labelled.
        # Minimum detectable difference for a two-sided 5% test at 80% power, k seeds per arm: MDE(k) = (z_{0.975} + z_{0.8}) * sigma * sqrt(2/k).
        zsum = stats.norm.ppf(0.975) + stats.norm.ppf(0.80)
        leg_final = np.array([LEG[s].val_accuracy.iloc[-1] for s in (17, 18, 19)])
        SIGMAS = {"legacy baseline, 3 seeds (different architecture)": leg_final.std(ddof=1),
                  "wide, last-10-epoch sd (epoch noise, not seed noise)": SUMM.loc["wide", "sd(last 10 epochs)"],
                  "gru, last-10-epoch sd (epoch noise, not seed noise)": SUMM.loc["gru", "sd(last 10 epochs)"]}
        ks = np.array([1, 2, 3, 5, 10, 20, 40])
        MDE = pd.DataFrame({name: zsum * s * np.sqrt(2 / ks) for name, s in SIGMAS.items()}, index=pd.Index(ks, name="seeds per arm"))
        display(MDE.round(3))
        target = abs(d_noise)
        need = {name: int(np.ceil(2 * (zsum * s / target) ** 2)) for name, s in SIGMAS.items()}
        for name, k in need.items():
            print(f"seeds per arm needed to detect |Delta(noise_10)| = {target:.3f} if sigma = {SIGMAS[name]:.3f} ({name}): {k}")
        check("N1.3.3a", abs(SIGMAS["legacy baseline, 3 seeds (different architecture)"] - np.std([0.3730, 0.5913, 0.3433], ddof=1)) < 1e-3, f"legacy seed sd = {leg_final.std(ddof=1):.4f}")
        check("N1.3.3b", bool(np.all(np.diff(MDE.values, axis=0) < 0)), "MDE shrinks monotonically with seeds per arm (as 1/sqrt(k))")
        check("N1.3.3c", all(k >= 1 for k in need.values()) and max(need.values()) > min(need.values()), f"seeds needed depend strongly on the unknown sigma: {min(need.values())} to {max(need.values())}")
        note("N1.3.3d", "these sigmas are proxies from other quantities; NONE is an estimate of gru or wide seed-to-seed sd, which requires re-training")
        """),

        code(r"""
        # Figure 3.3: minimum detectable accuracy difference versus seeds per arm, for three illustrative sigma values.
        fig, ax = plt.subplots(figsize=(8.4, 4.4))
        for (name, col), ls in zip(zip(MDE.columns, ["#7f7f7f", ACOL["wide"], ACOL["gru"]]), ["-", "--", ":"]):
            ax.plot(MDE.index, MDE[name], color=col, ls=ls, marker="o", ms=4, label=name)
        ax.axhline(target, color="r", lw=1); ax.text(40, target + 0.01, f"observed |Delta(noise_10)| = {target:.3f}", ha="right", color="r", fontsize=8)
        ax.set_xscale("log"); ax.set_xlabel("seeds per arm (log)"); ax.set_ylabel("minimum detectable difference (80% power)")
        ax.legend(fontsize=7.5); ax.set_title("Seed budget needed to settle the noise_10 tension (illustrative sigmas)")
        varied("N1.3.f3", MDE.values)
        savefig(fig, "fig3_3_seed_budget.png")
        """),

        md(r"""
        ### How to read this chart

        Each curve is the smallest true accuracy difference that a two-sided $5\%$ test would detect with $80\%$
        power, $\mathrm{MDE}(k)=(z_{0.975}+z_{0.80})\,\sigma\sqrt{2/k}$, as a function of the number $k$ of independent
        training runs per arm. The red line is the observed $|\Delta(\text{noise\_10})|$. A curve below the red line
        means $k$ seeds are enough to resolve a difference that large. The three curves use three **different,
        unverified** values of the seed-to-seed standard deviation $\sigma$, because none has been measured for
        `gru` or `wide`: the grey curve borrows the spread of the legacy baseline's three seeds, the green and blue
        ones borrow epoch-to-epoch noise. The lesson is the spread between curves: depending on $\sigma$, the number
        of seeds required ranges from one to more than twenty, so the honest statement is "we do not know how many
        seeds it takes", not "three is enough".

        ## 4. Neural collapse: what the formulas say, what they require, and what our probes can and cannot show

        ### 4.1 Order of business: preconditions before parallels

        Neural collapse (NC) is a phenomenon of the *terminal phase of training* (TPT), and its metrics are defined
        on *training-set* activations. Both facts constrain what we may say about our own numbers, and both are
        stated here, **before** any figure or measurement is discussed, so that no later sentence can lean on a
        parallel the definitions do not license. Sections 4.2 and 4.3 give the mathematics; Section 4.4 states the
        precondition and the two measurement deviations; only then does Section 4.5 look at our data.

        ### 4.2 NC1: within-class variability collapse

        (Papyan, Han and Donoho, arXiv:2008.08186; turn 4.) Let there be $C$ classes and $N$ examples per class, and
        let $h_{i,c}\in\mathbb{R}^p$ be the last-layer feature of example $i$ in class $c$. Define
        $$\mu_G=\mathrm{Ave}_{i,c}\{h_{i,c}\},\qquad \mu_c=\mathrm{Ave}_i\{h_{i,c}\},$$
        $$\Sigma_W=\mathrm{Ave}_{i,c}\big\{(h_{i,c}-\mu_c)(h_{i,c}-\mu_c)^\top\big\},\qquad
        \Sigma_B=\mathrm{Ave}_c\big\{(\mu_c-\mu_G)(\mu_c-\mu_G)^\top\big\},$$
        and the variability-collapse metric
        $$\mathrm{NC}_1=\frac{1}{C}\,\mathrm{Tr}\!\big(\Sigma_W\,\Sigma_B^{\dagger}\big),$$
        where $\dagger$ is the Moore-Penrose pseudo-inverse. Collapse means $\Sigma_W\to\mathbf 0$, hence
        $\mathrm{NC}_1\to0$. Three properties follow directly and are worth having in hand because they decide how
        the metric may be used. **(i) Scale invariance**: replacing $h\mapsto a h$ multiplies both $\Sigma_W$ and
        $\Sigma_B$ by $a^2$, and $\Sigma_B^\dagger$ by $a^{-2}$, so $\mathrm{NC}_1$ is unchanged. A raw
        within-class variance is *not* scale-free, so a large $\mathrm{Tr}\,\Sigma_W$ says nothing about collapse
        unless it is compared with the between-class scale. **(ii) Rotation invariance**: $h\mapsto Qh$ with $Q$
        orthogonal leaves it unchanged. **(iii) Quadratic scaling in the noise**: if $h=m+s\,\varepsilon$ with class
        means $m$ fixed and within-class noise scaled by $s$, then $\Sigma_W\propto s^2$ and
        $\mathrm{NC}_1\propto s^2$, so $\log\mathrm{NC}_1$ against $\log s$ has slope $2$.

        ### 4.3 NC2: the simplex equiangular tight frame

        (Turn 6.) The standard simplex ETF is the set of columns of
        $$M^{*}=\sqrt{\tfrac{C}{C-1}}\Big(I-\tfrac{1}{C}\mathbf 1_C\mathbf 1_C^{\top}\Big),$$
        and a general simplex ETF is $M=\alpha\,U\,M^{*}$ with $\alpha>0$ and $U^\top U=I$. Its columns have equal norm
        and pairwise cosine exactly $-1/(C-1)$, the largest mutual angle that $C$ centred vectors can have. The paper
        splits NC2 into two measurements on the centred class means $\mu_c-\mu_G$, with
        $\cos_\mu(c,c')=\langle\mu_c-\mu_G,\mu_{c'}-\mu_G\rangle/(\lVert\mu_c-\mu_G\rVert\,\lVert\mu_{c'}-\mu_G\rVert)$:
        $$\text{equinormness}=\frac{\mathrm{Std}_c\,\lVert\mu_c-\mu_G\rVert_2}{\mathrm{Avg}_c\,\lVert\mu_c-\mu_G\rVert_2},\qquad
        \text{equiangularity}=\mathrm{Std}_{c\ne c'}\,\cos_\mu(c,c'),$$
        and a third, maximal-angle measure, $\mathrm{Avg}_{c\ne c'}\,\big|\cos_\mu(c,c')+\tfrac{1}{C-1}\big|$. All three tend
        to $0$ in TPT. One geometric fact matters for us: an ETF of $C=12$ classes spans $C-1=11$ dimensions. **In a
        two-dimensional projection, 12 centred vectors cannot be equiangular at $-1/11$**, so the 2D versions of the
        equiangularity and maximal-angle measures have a floor above zero that no amount of training can lower. The
        fixture cell below computes that floor for the best-spread planar configuration (a regular 12-gon).
        """),

        code(r"""
        # N1.4.1: NC formulas as code, verified on labelled FIXTURES (a simplex ETF and noise-scaled copies). Not evidence about speech.
        def nc_stats(h, y):
            # NC1, equinormness, equiangularity and maximal-angle measure for features h (n, p) with integer labels y.
            h = np.asarray(h, dtype=float); classes = np.unique(y); C = len(classes)
            mu_G = h.mean(0); mus = np.stack([h[y == c].mean(0) for c in classes])
            Sw = sum((h[y == c] - mus[i]).T @ (h[y == c] - mus[i]) for i, c in enumerate(classes)) / len(h)   # Ave over all i,c
            M = mus - mu_G; Sb = M.T @ M / C
            nc1 = float(np.trace(Sw @ np.linalg.pinv(Sb)) / C)
            norms = np.linalg.norm(M, axis=1); U = M / norms[:, None]
            cs = (U @ U.T)[np.triu_indices(C, 1)]
            return {"nc1": nc1, "equinorm": float(norms.std() / norms.mean()), "equiang": float(cs.std()),
                    "maxangle": float(np.abs(cs + 1 / (C - 1)).mean()), "trSw": float(np.trace(Sw)), "trSb": float(np.trace(Sb))}

        Ccl, Nper = 12, 10
        Mstar = np.sqrt(Ccl / (Ccl - 1)) * (np.eye(Ccl) - np.ones((Ccl, Ccl)) / Ccl)     # columns = simplex ETF
        gram = Mstar.T @ Mstar
        check("N1.4.1a", np.allclose(np.linalg.norm(Mstar, axis=0), 1.0), "columns of M* have unit norm")
        off = gram[~np.eye(Ccl, dtype=bool)]
        check("N1.4.1b", np.allclose(off, -1 / (Ccl - 1)), f"pairwise cosine of M* columns = -1/(C-1) = {-1 / (Ccl - 1):.4f} exactly")
        check("N1.4.1c", np.allclose(gram, Ccl / (Ccl - 1) * (np.eye(Ccl) - np.ones((Ccl, Ccl)) / Ccl)), "M*^T M* = C/(C-1) (I - 11^T/C): a tight frame")
        check("N1.4.1d", np.linalg.matrix_rank(Mstar) == Ccl - 1, f"rank(M*) = {np.linalg.matrix_rank(Mstar)} = C-1: the simplex ETF spans 11 dimensions, not 2")

        rngf = np.random.default_rng(RNG_SEED + 41)
        y_fx = np.repeat(np.arange(Ccl), Nper)
        eps = rngf.standard_normal((Ccl * Nper, Ccl))
        for _c in range(Ccl):
            eps[y_fx == _c] -= eps[y_fx == _c].mean(0)      # zero class means: only Sigma_W scales with s
        base = 3.0 * Mstar.T[y_fx]                                   # alpha = 3, U = I
        s_grid = np.array([1.0, 0.5, 0.25, 0.125, 0.0625])
        nc1_s = np.array([nc_stats(base + s * eps, y_fx)["nc1"] for s in s_grid])
        slope = np.polyfit(np.log(s_grid), np.log(nc1_s), 1)[0]
        check("N1.4.1e", abs(slope - 2.0) < 0.05, f"fixture: log NC1 vs log(noise scale) has slope {slope:.3f} (theory 2)")
        z0 = nc_stats(base, y_fx)
        check("N1.4.1f", z0["nc1"] < 1e-12 and z0["equinorm"] < 1e-12 and z0["equiang"] < 1e-12 and z0["maxangle"] < 1e-12,
              "fixture at zero within-class noise: NC1, equinormness, equiangularity and maximal-angle measure are all 0")
        h_noisy = base + 0.5 * eps
        Q, _ = np.linalg.qr(rngf.standard_normal((Ccl, Ccl)))
        a0, a1, a2 = nc_stats(h_noisy, y_fx)["nc1"], nc_stats(7.3 * h_noisy, y_fx)["nc1"], nc_stats(h_noisy @ Q, y_fx)["nc1"]
        check("N1.4.1g", abs(a1 / a0 - 1) < 1e-8 and abs(a2 / a0 - 1) < 1e-8, f"NC1 is scale-invariant and rotation-invariant (ratios {a1 / a0:.10f}, {a2 / a0:.10f})")
        tr_ratio = nc_stats(7.3 * h_noisy, y_fx)["trSw"] / nc_stats(h_noisy, y_fx)["trSw"]
        check("N1.4.1h", abs(tr_ratio - 7.3 ** 2) < 1e-6, f"whereas Tr(Sigma_W) is NOT scale-free: it grows by {tr_ratio:.2f} = 7.3^2 under a pure rescaling")

        ang = np.arange(Ccl) * 2 * np.pi / Ccl
        U2 = np.c_[np.cos(ang), np.sin(ang)]; cs2 = (U2 @ U2.T)[np.triu_indices(Ccl, 1)]
        FLOOR2D = {"equiang": float(cs2.std()), "maxangle": float(np.abs(cs2 + 1 / (Ccl - 1)).mean())}
        print("2D floor for a regular 12-gon of class means:", {k: round(v, 3) for k, v in FLOOR2D.items()})
        check("N1.4.1i", FLOOR2D["equiang"] > 0.5 and FLOOR2D["maxangle"] > 0.5, "in two dimensions the equiangularity (0 at ETF) is >= 0.5 even for the best-spread planar arrangement: it cannot be read as 'not collapsed'")
        """),

        md(r"""
        ### 4.4 The precondition (TPT) and the two measurement deviations, stated before any comparison

        **Precondition: the terminal phase of training.** In the paper (turn 5) TPT is the post-zero-error phase: it
        begins at the epoch where *in-sample training classification error first vanishes*, while training continues
        to drive cross-entropy toward zero. The paper operationalises the effective start as training accuracy of
        $99.6\%$ for ImageNet and $99.9\%$ for the other datasets. Neural collapse is a statement about behaviour
        *inside* TPT; the metrics are snapshotted across the whole of training (300 epochs for ImageNet, 350 for the
        other six datasets) so the paper can show them fall, but "collapse" is claimed only where the precondition
        holds. A model that has not reached zero training error is not "failing to collapse": the quantity has no
        claim on it.

        **Deviation 1: training versus held-out data.** In the paper, $\mathrm{NC}_1$, $\mathrm{NC}_2$ and
        $\mathrm{NC}_3$ are computed on **training-set activations**; only $\mathrm{NC}_4$ (agreement with the
        nearest-class-centre rule) is evaluated on held-out data. Our probe is a fixed **120-sample held-out
        split** (12 classes $\times$ 10 clips), not training activations. Held-out activations of a model that
        generalises imperfectly scatter more than training activations, so our within-class spread is biased
        upward against the paper's definition and the two are not the same quantity.

        **Deviation 2: what the recorded features are.** Our 2D coordinates are a PCA projection of the classifier's
        input (the penultimate features), not the paper's full $p$-dimensional last-layer features, and the
        artifacts store only the 2D coordinates and two scalar cluster summaries in raw space, not the activations
        themselves. Two measurements are therefore available: the paper's NC formulas applied to the **2D
        coordinates**, and two **raw-space scalar summaries** whose ratio is a scale-free (but non-standard)
        variability-to-separation number. Neither is the paper's $\mathrm{NC}_1$ in $p$ dimensions.

        **Is a training run in TPT? A bound from the loss.** The artifacts do not record training accuracy, so
        membership in TPT cannot be read off directly. What can be bounded: a misclassified example has some other class
        $j$ with $p_j\ge p_y$, and $p_y+p_j\le1$ forces $p_y\le\tfrac12$, so its loss is at least $\ln2$. Averaging over
        a training set,
        $$\text{training error}\ \le\ \frac{\overline{\mathrm{CE}}}{\ln 2}.$$
        This bound is only useful when it is small. It is also applied to the *recorded* training cross-entropy, which
        is the running mean over minibatches of the last epoch in training mode, not an evaluation-mode pass, so
        it is a heuristic, not a certificate.
        """),

        code(r"""
        # N1.4.2: TPT status per arm from the recorded final training cross-entropy (bound err <= CE / ln 2).
        rows = []
        for a in ARCHS:
            ce = DATA[a]["res"]["final_train_loss"]
            rows.append({"arch": a, "final_train_CE": ce, "err_bound=CE/ln2": ce / np.log(2), "geo-mean p(true)=exp(-CE)": np.exp(-ce),
                         "meets 0.1% operational TPT?": ce / np.log(2) <= 0.001})
        TPT = pd.DataFrame(rows).set_index("arch")
        display(TPT.round(4))
        check("N1.4.2a", [round(TPT.loc[a, "final_train_CE"], 4) for a in ARCHS] == [0.0100, 0.9794, 0.0182], "final train CE: gru 0.0100, timepool 0.9794, wide 0.0182")
        check("N1.4.2b", TPT.loc["gru", "err_bound=CE/ln2"] < 0.02 and TPT.loc["wide", "err_bound=CE/ln2"] < 0.03,
              f"gru and wide: training error bounded by {TPT.loc['gru', 'err_bound=CE/ln2']:.2%} and {TPT.loc['wide', 'err_bound=CE/ln2']:.2%}: near, but not certified at, zero error")
        check("N1.4.2c", TPT.loc["timepool", "err_bound=CE/ln2"] > 1.0, f"timepool: the bound is {TPT.loc['timepool', 'err_bound=CE/ln2']:.0%}, i.e. vacuous; nothing certifies zero training error")
        check("N1.4.2d", not TPT["meets 0.1% operational TPT?"].any(), "no arm is CERTIFIED at the paper's 99.9% operational threshold by this bound (it could still be reached; training accuracy was not recorded)")
        STATUS = {"gru": "TPT plausible, not certified", "wide": "TPT plausible, not certified", "timepool": "TPT not reached (working premise): NC undefined"}
        note("N1.4.2e", "timepool: training accuracy is not recorded; we adopt 'zero training error not reached' as a premise supported by CE 0.9794 (true-class geometric-mean probability 0.376), not proved by it")
        """),

        code(r"""
        # Figure 4.0: recorded training cross-entropy per epoch versus the level that would certify 99.9% training accuracy.
        LEVEL = 0.001 * np.log(2)          # CE at which the error bound CE/ln2 equals the paper's 0.1% operational threshold
        fig, ax = plt.subplots(figsize=(9.2, 4.5))
        for a in ARCHS:
            h_ = DATA[a]["res"]["history"]
            ax.plot(np.arange(1, len(h_) + 1), [e["train_loss"] for e in h_], color=ACOL[a], lw=1.9, label=a)
        ax.axhline(LEVEL, color="r", ls="--"); ax.text(1.5, LEVEL * 1.35, f"CE = 0.1% x ln 2 = {LEVEL:.5f}: the loss level at which the bound certifies 99.9% training accuracy", color="r", fontsize=8)
        ax.set_yscale("log"); ax.set_xlabel("epoch"); ax.set_ylabel("training cross-entropy (epoch mean, log)"); ax.legend(fontsize=9)
        ax.set_title("How far each arm's training loss is from a certifiable terminal phase")
        ratio_ = {a: DATA[a]["res"]["final_train_loss"] / LEVEL for a in ARCHS}
        check("N1.4.2f", min(ratio_.values()) > 5, f"final train CE is {ratio_['gru']:.0f}x (gru), {ratio_['wide']:.0f}x (wide), {ratio_['timepool']:.0f}x (timepool) above the certifying level: none is certified")
        last5 = {a: [e["train_loss"] for e in DATA[a]["res"]["history"]][-5:] for a in ARCHS}
        check("N1.4.2g", all(last5[a][-1] < last5[a][0] for a in ("gru", "wide")), "gru and wide training CE is still falling over the last five epochs (a terminal phase, if any, has only just begun)")
        varied("N1.4.f0", [[e["train_loss"] for e in DATA[a]["res"]["history"]] for a in ARCHS])
        savefig(fig, "fig4_0_train_ce_vs_tpt_level.png")
        """),

        md(r"""
        ### How to read this chart

        Training cross-entropy per epoch, one line per architecture, log axis; the red dashed line is the loss level
        $0.001\ln2\approx0.0007$ below which the bound $\text{error}\le\mathrm{CE}/\ln2$ certifies the paper's $99.9\%$
        operational threshold. `gru` and `wide` decline by two orders of magnitude and are still falling in the
        last epochs, ending about $14$ and $26$ times above the red line, so "plausibly entering TPT" is the most this
        chart supports. `timepool` stays near $1$ ($0.98$ at the end), three orders of magnitude above it. The chart is the honest
        summary of Section 4.4: a terminal phase, if there is one, lies beyond where the two better arms stopped.
        The runs lasted $28$ epochs (`gru`) and $36$ epochs (`wide`, `timepool`), against the $300$ to $350$ epochs
        over which the paper snapshots its metrics.
        """),

        md(r"""
        ### 4.5 What our probes show

        With the precondition and both deviations on the table, the measurements can be read at the right strength.
        The status assignment is: `gru` (train cross-entropy $0.0100$) and `wide` ($0.0182$) are *plausibly* in TPT;
        `timepool` ($0.9794$) never reached zero training error, so **neural collapse is not defined for it**. Its
        rising intra-class variance in the figures below is therefore recorded as a description of its geometry,
        and is **not** to be described as a "failure to collapse". The metrics we compute per epoch are: the paper's
        $\mathrm{NC}_1$ formula and the three NC2 measures applied to the 2D coordinates of the 120 held-out
        probes; and, in raw space, the ratio $\rho_{\text{raw}}=\text{intra}_{\text{raw}}/\text{inter}_{\text{raw}}^2$,
        where intra is the mean squared distance to the class centroid (equal to $\mathrm{Tr}\,\Sigma_W$ for balanced
        classes) and inter is the mean pairwise centroid distance. $\rho_{\text{raw}}$ is scale-free and tracks
        $\mathrm{Tr}\,\Sigma_W/\mathrm{Tr}\,\Sigma_B$ up to a constant; it is a crude cousin of $\mathrm{NC}_1$, not
        $\mathrm{NC}_1$.
        """),

        code(r"""
        # N1.4.3: per-epoch NC-type series for each arm from the real probes (held-out, 120 clips; 2D PCA coordinates + raw scalars).
        NC = {}
        for a in ARCHS:
            z = DATA[a]["npz"]; E_ = len(z["inter_2d"]); y_ = z["labels"]
            per = pd.DataFrame([nc_stats(z["coords_2d"][e].astype(float), y_) for e in range(E_)])
            per["inter_2d"], per["intra_2d"] = z["inter_2d"], z["intra_2d"]
            per["inter_raw"], per["intra_raw"] = z["inter_raw"], z["intra_raw"]
            per["rho_raw"] = per["intra_raw"] / per["inter_raw"] ** 2
            per["rho_2d"] = per["intra_2d"] / per["inter_2d"] ** 2
            NC[a] = per
        summ = pd.DataFrame({a: {"nc1_2d first": NC[a].nc1.iloc[0], "nc1_2d last": NC[a].nc1.iloc[-1], "rho_raw first": NC[a].rho_raw.iloc[0], "rho_raw last": NC[a].rho_raw.iloc[-1],
                                 "rho_raw last/first": NC[a].rho_raw.iloc[-1] / NC[a].rho_raw.iloc[0], "equinorm last": NC[a].equinorm.iloc[-1],
                                 "equiang(2D) last": NC[a].equiang.iloc[-1], "maxangle(2D) last": NC[a].maxangle.iloc[-1]} for a in ARCHS}).T
        display(summ.round(3))
        g, t, w_ = (DATA[a]["npz"] for a in ARCHS)
        check("N1.4.3a", abs(g["inter_2d"][0] - 2.6) < 0.1 and abs(g["inter_2d"][-1] - 4.3) < 0.1, f"gru inter-cluster distance (2D) {g['inter_2d'][0]:.2f} -> {g['inter_2d'][-1]:.2f} (brief: 2.6 -> 4.3)")
        check("N1.4.3b", abs(g["intra_2d"][0] - 6.7) < 0.1 and abs(g["intra_2d"].min() - 1.5) < 0.05, f"gru intra-cluster variance (2D) {g['intra_2d'][0]:.2f} -> min {g['intra_2d'].min():.2f}, last {g['intra_2d'][-1]:.2f} (brief's 1.5 is the minimum, not the last epoch)")
        check("N1.4.3c", abs(t["intra_2d"][0] - 10) < 0.2 and abs(t["intra_2d"][-1] - 29) < 0.5, f"timepool intra (2D) {t['intra_2d'][0]:.1f} -> {t['intra_2d'][-1]:.1f} (max {t['intra_2d'].max():.1f}); brief: 10 -> 29")
        check("N1.4.3d", abs(w_["intra_2d"].min() - 48) < 0.5 and abs(w_["intra_2d"].max() - 135) < 0.5, f"wide intra (2D) oscillates between {w_['intra_2d'].min():.1f} and {w_['intra_2d'].max():.1f} (brief: 48-135)")
        check("N1.4.3e", NC["gru"].nc1.iloc[-1] < 0.2 * NC["gru"].nc1.iloc[0], f"gru 2D NC1-analogue falls {NC['gru'].nc1.iloc[0]:.3f} -> {NC['gru'].nc1.iloc[-1]:.3f}: NC1-consistent behaviour")
        note("N1.4.3f", f"wide 2D NC1-analogue {NC['wide'].nc1.iloc[0]:.3f} -> {NC['wide'].nc1.iloc[-1]:.3f} (flat) while its raw-space ratio falls {NC['wide'].rho_raw.iloc[0]:.3f} -> {NC['wide'].rho_raw.iloc[-1]:.3f}: only weakly NC1-consistent, and only in raw space")
        note("N1.4.3g", f"raw intra grows for EVERY arm (gru {g['intra_raw'][0]:.1f} -> {g['intra_raw'][-1]:.1f}) because feature scale grows; the scale-free ratio rho_raw is the meaningful one")
        check("N1.4.3h", all(NC[a].equiang.iloc[-1] > 0.5 and NC[a].maxangle.iloc[-1] > 0.5 for a in ARCHS), "2D equiangularity / maximal-angle measures sit at their planar floor (>0.5) for all arms: uninformative about NC2 here")
        """),

        code(r"""
        # Figure 4.1: final-epoch 2D geometry of the 120 held-out probes per arm, TPT status annotated.
        fig, axs = plt.subplots(1, 3, figsize=(15, 4.9))
        cmap = plt.get_cmap("tab20")
        for ax, a in zip(axs, ARCHS):
            z = DATA[a]["npz"]; xy = z["coords_2d"][-1]; y_ = z["labels"]
            for c in range(12):
                m_ = y_ == c
                ax.scatter(xy[m_, 0], xy[m_, 1], s=16, color=cmap(c), alpha=0.85)
                ax.scatter(*xy[m_].mean(0), s=90, marker="X", color=cmap(c), edgecolor="k", linewidth=0.7)
            ax.set_title(f"{a}: last epoch ({len(z['inter_2d'])})", color=ACOL[a])
            ax.text(0.02, 0.98, f"{STATUS[a]}\ntrain CE {DATA[a]['res']['final_train_loss']:.4f}\nheld-out probes, 2D PCA",
                    transform=ax.transAxes, va="top", fontsize=8.5, bbox=dict(boxstyle="round", fc="white", ec=ACOL[a]))
            ax.set_xlabel("PC 1"); ax.set_ylabel("PC 2")
        fig.suptitle("Final 2D geometry of held-out probes (X = class centroid). Not training activations; NC is undefined for timepool.", y=1.02)
        varied("N1.4.f1", [DATA[a]["npz"]["coords_2d"][-1] for a in ARCHS])
        plt.tight_layout(); savefig(fig, "fig4_1_final_geometry.png")
        """),

        md(r"""
        ### How to read this chart

        Each panel is one architecture's last-epoch PCA projection of the classifier's input for the 120 held-out
        probe clips; small dots are clips, coloured by class, and the large crosses are class centroids. Read
        *separation between coloured groups* against *spread within one colour*. `gru` shows compact same-colour
        groups spread across the plane; `wide` shows groups spread over a much larger extent (note the axis scale)
        with heavier overlap; `timepool` shows large overlapping clouds. The annotation box states the
        terminal-phase status of each arm. For `timepool` it reads "NC undefined": the arm never reached zero
        training error, so the picture is a description of a model that has not begun TPT, and the overlap must not
        be labelled a collapse failure. Because the probes are held-out and the projection is two-dimensional, this
        is also not a picture of $\mathrm{NC}_2$: twelve classes cannot lie on an ETF in a plane.
        """),

        code(r"""
        # Figure 4.2: trajectories of scale-aware and raw cluster summaries across epochs, per arm. timepool drawn dashed: NC undefined.
        fig, axs = plt.subplots(2, 2, figsize=(13, 8))
        panels = [("nc1", "NC1 formula on 2D coordinates (held-out)", False), ("rho_raw", "raw-space Tr(Sw)/inter^2 ratio  (scale-free)", True),
                  ("intra_2d", "intra-cluster variance, 2D (NOT scale-free)", True), ("inter_2d", "inter-cluster distance, 2D", False)]
        for ax, (col, ttl, logy) in zip(axs.ravel(), panels):
            for a in ARCHS:
                d = NC[a]
                ax.plot(np.arange(1, len(d) + 1), d[col], color=ACOL[a], lw=1.8, ls="--" if a == "timepool" else "-",
                        label=a + (" (NC undefined: TPT not reached)" if a == "timepool" else ""))
            ax.set_title(ttl, fontsize=10); ax.set_xlabel("epoch")
            if logy: ax.set_yscale("log")
        axs[0, 0].legend(fontsize=8)
        varied("N1.4.f2", [NC[a][c].values for a in ARCHS for c in ("nc1", "rho_raw", "intra_2d", "inter_2d")])
        plt.tight_layout(); savefig(fig, "fig4_2_nc_trajectories.png")
        """),

        md(r"""
        ### How to read this chart

        Four panels, one line per architecture across its epochs; the `timepool` line is dashed and labelled because
        neural collapse is undefined for it. Top-left is the paper's $\mathrm{NC}_1$ formula applied to the 2D
        coordinates: only `gru` shows the monotone-looking fall toward zero that the definition describes ($0.46$ to
        $0.03$); `wide` is flat and `timepool` is flat and high. Top-right is the scale-free raw-space ratio on a log
        axis: **all three** arms fall, `gru` the most (about fivefold), `wide` about threefold, `timepool` also
        about threefold (again, a description, not a collapse). Bottom-left is the raw 2D intra-cluster variance,
        which the panel title flags as scale-dependent: `wide`'s $48$ to $135$ oscillation looks alarming in
        absolute terms, but intra is a squared distance, so it should be set against the squared inter-cluster
        distance (bottom-right, $5.8$ to $7.6$, squared about $33$ to $58$): the ratio is near $1.5$ and does not fall.
        The ratio, not the level, is what NC1 measures. The caution to carry away: an absolute intra-cluster level cannot rank
        architectures by collapse. What would make the `gru` reading wrong: the same fall on a training-set probe
        being absent, or a fall driven purely by the projection changing between epochs (PCA is refit every epoch).
        """),

        code(r"""
        # Figure 4.3: NC2-type measures in 2D versus their planar floor; equinormness trajectories.
        fig, axs = plt.subplots(1, 2, figsize=(13, 4.4))
        for a in ARCHS:
            z = DATA[a]["npz"]; xy = z["coords_2d"][-1].astype(float); y_ = z["labels"]
            mus = np.stack([xy[y_ == c].mean(0) for c in range(12)]); M_ = mus - xy.mean(0)
            U_ = M_ / np.linalg.norm(M_, axis=1)[:, None]; cs_ = (U_ @ U_.T)[np.triu_indices(12, 1)]
            axs[0].hist(cs_, bins=np.linspace(-1, 1, 21), alpha=0.45, color=ACOL[a], label=f"{a} (Std {cs_.std():.2f})")
        axs[0].axvline(-1 / 11, color="r", lw=1.5); axs[0].text(-1 / 11 + 0.03, axs[0].get_ylim()[1] * 0.92, "-1/(C-1) = -0.091\n(ETF value)", color="r", fontsize=8)
        axs[0].set_xlabel("pairwise cosine of centred class means (2D)"); axs[0].set_ylabel("number of class pairs"); axs[0].legend(fontsize=8)
        axs[0].set_title("Last-epoch class-mean cosines in 2D vs the ETF value")
        for a in ARCHS:
            axs[1].plot(np.arange(1, len(NC[a]) + 1), NC[a].equinorm, color=ACOL[a], ls="--" if a == "timepool" else "-", lw=1.8, label=a)
        axs[1].set_xlabel("epoch"); axs[1].set_ylabel("equinormness (Std/Avg of ||mu_c - mu_G||), 2D"); axs[1].legend(fontsize=8)
        axs[1].set_title("Equinormness across epochs (2D, held-out)")
        varied("N1.4.f3", [NC[a].equinorm.values for a in ARCHS])
        plt.tight_layout(); savefig(fig, "fig4_3_nc2_2d.png")
        """),

        md(r"""
        ### How to read this chart

        Left: for each arm, a histogram of the $\binom{12}{2}=66$ pairwise cosines between centred class means at the
        last epoch, in the 2D projection; the red line marks the ETF value $-1/11=-0.091$ at which every pair should
        sit in TPT. In a plane the cosines must spread over the whole range $[-1,1]$ (the standard deviations printed in
        the legend are all near $0.68$, close to the $0.668$ of a perfect regular 12-gon), so the histogram cannot
        approach a spike at the red line for *any* model. The panel therefore demonstrates the floor, and should be read
        as the reason the 2D NC2 numbers are not evidence. Right: equinormness, the coefficient of variation of the
        centred-mean norms. It falls for all arms over training (roughly $0.8$-$1.0$ down to $0.5$-$0.65$), which is the
        direction NC2 predicts but is also what a projection that spreads centroids more evenly would produce.

        ### 4.6 What we may say

        1. **Allowed**: `gru`, whose loss is consistent with near-zero training error, shows a fall in the paper's
           $\mathrm{NC}_1$ formula and in a scale-free raw ratio on held-out probes. This is *NC1-consistent
           behaviour*.
        2. **Allowed, weaker**: `wide` shows a fall only in the raw ratio, not in the 2D $\mathrm{NC}_1$.
        3. **Not allowed**: that `timepool` "failed to collapse". It never entered TPT, and the metric is not defined
           on it. Its rising raw intra-cluster level is recorded as geometry.
        4. **Not allowed**: any $\mathrm{NC}_2$ claim from 2D projections, since a plane cannot host an ETF of 12
           classes.
        5. **Not allowed**: to call any of this the paper's measurement. The paper measures training activations at
           $p$ dimensions; we have held-out probes and a PCA projection. The parallel is an observation about
           shape, in the direction the theory predicts, with no test of the theory.
        """),

        md(r"""
        ### 4.7 Where these quantities live in notebook 04, and where they do not

        **Provenance, stated once and plainly.** Every measured number in this notebook -- the accuracies, the
        parameter counts, the clip rates, the fitted temperatures, the cluster geometry -- is read from the artifacts
        of **notebook 04, `04_real_dataset_walkthrough.ipynb`**, via `runtime/metrics/arch-*.result.json` and
        `arch-*.probes.npz`. This notebook trains nothing and measures nothing of its own. Notebook 04 is the
        measurement; this notebook is the mathematics, and notebook 02 is the cheap sandbox that precedes both.

        **The correspondence is partial, and the gap is worth naming.** Notebook 04 reports three families of
        in-training diagnostic. Only one of them is what the neural-collapse formulas above describe:

        | notebook 04 | what it measures | relation to $\mathrm{NC}_1$ / $\mathrm{NC}_2$ |
        |---|---|---|
        | V4 cluster health (`inter_2d`, `intra_2d`) | between- and within-class spread of penultimate features, per epoch | **the same family.** Its intra-cluster variance is a 2D, held-out analogue of $\Sigma_W$; its inter-cluster distance plays the role $\Sigma_B$ plays in $\mathrm{NC}_1$ |
        | V3 latent scatter (`coords_2d`) | the projected features themselves | the raw material both $\mathrm{NC}_1$ and V4 are computed from |
        | V5--V8 MM-PHATE (`mmphate_tensor`) | dispersion of **hidden units** across sequence steps and epochs | **a different object.** Neural collapse is a statement about final-layer *class* geometry at convergence; MM-PHATE's entropies are about *unit* trajectories through time. Neither theory grounds the other |

        So the mathematics in this section founds notebook 04's V3 and V4, and does **not** found its V5--V8. Those
        rest on MM-PHATE's own construction, cited there, and nothing in this notebook derives them. That is an
        honest gap in the series rather than a claim of coverage, and it is repeated in section 8.
        """),

        code(r"""
        # N1.4.7: verify the correspondence claim against the artifacts, rather than asserting it in prose.
        from scipy.stats import spearmanr
        # For each arm: does notebook 04's stored intra_2d behave like a within-class scatter computed
        # directly from its stored coords_2d? If the two disagree, the table above is wrong.
        rows = []
        for a in ARCHS:
            n = DATA[a]["npz"]
            coords, labels = n["coords_2d"], n["labels"]
            recomputed = []
            for t in range(coords.shape[0]):
                pts = coords[t]
                within = np.concatenate([pts[labels == c] - pts[labels == c].mean(axis=0)
                                         for c in np.unique(labels)])
                recomputed.append(float((within ** 2).sum(axis=1).mean()))
            recomputed = np.asarray(recomputed)
            stored = n["intra_2d"]
            rho = float(spearmanr(recomputed, stored).statistic)
            rows.append({"arch": a, "epochs": len(stored), "rho(recomputed, stored intra_2d)": rho,
                         "stored first": float(stored[0]), "stored last": float(stored[-1]),
                         "train_CE": DATA[a]["res"]["final_train_loss"],
                         "in_TPT": "no" if DATA[a]["res"]["final_train_loss"] > 0.1 else "plausibly"})
            check(f"N1.4.7[{a}.same_family]", rho > 0.9,
                  f"within-class scatter recomputed from coords_2d tracks stored intra_2d (Spearman rho={rho:+.3f})")
        CORR = pd.DataFrame(rows).set_index("arch")
        display(CORR.round(4))
        note("N1.4.7[mmphate]", "no quantity derived in this section founds notebook 04's V5-V8 MM-PHATE statistics; "
                               "they measure hidden-unit dispersion across sequence steps, not final-layer class geometry")
        """),

        code(r"""
        # N1.4.8: the same picture as a figure -- stored intra_2d against the scatter recomputed from coords_2d.
        fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
        for ax, a in zip(axes, ARCHS):
            n = DATA[a]["npz"]
            coords, labels = n["coords_2d"], n["labels"]
            rec = np.asarray([float((np.concatenate([coords[t][labels == c] - coords[t][labels == c].mean(axis=0)
                                                     for c in np.unique(labels)]) ** 2).sum(axis=1).mean())
                              for t in range(coords.shape[0])])
            ep = np.arange(len(rec))
            varied(f"{a} intra_2d", rec, n["intra_2d"])
            ax.plot(ep, n["intra_2d"], color=ACOL[a], lw=2, label="stored intra_2d (notebook 04 V4)")
            ax.plot(ep, rec, color="black", lw=1, ls="--", label="recomputed from coords_2d")
            tpt = DATA[a]["res"]["final_train_loss"] <= 0.1
            ax.set(title=f"{a} - {'plausibly in TPT' if tpt else 'never reached TPT'}",
                   xlabel="epoch", ylabel="within-class scatter (2D)")
            ax.grid(alpha=0.3); ax.legend(fontsize=7)
        fig.suptitle("Notebook 04's V4 intra-cluster variance is a within-class scatter: "
                     "the NC1 numerator's 2D, held-out analogue", y=1.04)
        savefig(fig, "fig4_4_nc1_vs_nb04_v4.png")
        """),

        md(r"""
        ### How to read this chart

        **Source.** `runtime/metrics/arch-*.probes.npz`, produced by notebook 04: the stored `intra_2d` series it
        plots as V4, and `coords_2d`, the projected probe features it computed them from.

        **What it shows.** The dashed black line is a within-class scatter recomputed here directly from the stored
        coordinates -- the $\Sigma_W$ trace of section 4.2, in the projected plane. It lies on the coloured stored
        series. That is the evidence for the first row of the table above: notebook 04's V4 is the same quantity the
        neural-collapse literature calls within-class variability, measured in 2D on held-out probes.

        **What would falsify the reading.** If the dashed and solid lines diverged, the claim that V4 and
        $\mathrm{NC}_1$ are the same family would be wrong, and section 4.5's parallels with it. The per-arm
        Spearman correlation in the table above is the check: it must exceed $0.9$.

        **What it does not show.** Panel titles carry each arm's TPT status. `timepool` never reached near-zero
        training error, so the neural-collapse reading does not apply to it at all -- its curve is geometry, not
        collapse. And nothing here speaks to MM-PHATE.
        """),
    ]
