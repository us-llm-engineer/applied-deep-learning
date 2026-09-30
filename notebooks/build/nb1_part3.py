"""NB1 part 3: section 5 (adaptive gradient clipping) and section 6 (calibration: temperature scaling, ECE, MDCA).

Measured: grad_norm / clip_threshold arrays from arch-*.probes.npz; temperatures, ECE and NLL from the summary
blocks of arch-*.result.json. Literature numbers: attributed by session and turn in section 1. Constructed inputs
(Monte-Carlo norm sequences, an MDCA counterexample fixture) are labelled fixtures for a formula or a mechanism.
"""
from nbkit import md, code


def cells():
    return [
        md(r"""
        ## 5. Adaptive gradient clipping (AutoClip)

        ### 5.1 The algorithm, exactly as the source states it

        (AutoClip, arXiv:2007.14469; NotebookLM turns 5 and 6 of the signal notebook.) At optimizer step $t$ the
        unclipped gradient norm is $g_t=\lVert\nabla_\theta f(X_t;\theta_{t-1})\rVert_2$. The full cumulative history
        is $G_h(t)=[g_1,\dots,g_t]$, it is **not** a sliding window (a window is listed as future work), and the
        threshold is a fixed percentile of it,
        $$\eta_c(t)=\mathrm{Percentile}_p\big(G_h(t)\big),\qquad p\ \text{fixed (here } p=10).$$
        The gradient is then rescaled by the ordinary norm-clipping multiplier
        $$h_c=\min\!\Big(\frac{\eta_c(t)}{\lVert g_t\rVert_2},\,1\Big),\qquad \nabla_\theta^{\text{clipped}}=h_c\,\nabla_\theta f(X_t;\theta_{t-1}),$$
        so a step is *clipped* exactly when $g_t>\eta_c(t)$, and a clipped step has its norm set to $\eta_c(t)$. The paper
        swept $p\in\{0,1,10,25,50,90,100\}$ ($p=0$ is min-clipping, $p=100$ is no clipping) on WSJ0-2mix speech
        *separation* with five loss functions, and reports $p=10$ (with $p=1$) as the recommended "set-and-forget"
        value (turn 6).

        For a sorted history $x_{(1)}\le\dots\le x_{(m)}$ the $p$-th percentile with linear interpolation (the
        `numpy` default, and what the training code calls) is
        $$q_p=x_{(k+1)}+\phi\,\big(x_{(k+2)}-x_{(k+1)}\big),\qquad k+\phi=\tfrac{p}{100}(m-1),\ \ k=\lfloor\cdot\rfloor .$$

        ### 5.2 Two fidelity notes, recorded before the numbers

        1. **An off-by-one against the paper.** The paper *appends $g_t$ to the history first* and then takes the
           percentile, so $\eta_c(t)$ includes step $t$'s own norm. Our training loop takes the percentile of the
           **prior** history $[g_1,\dots,g_{t-1}]$ and appends afterwards (step $1$ therefore has $\eta=\infty$ and is
           never clipped). We quantify the consequence below on the recorded norms.
        2. **A cross-task percentile.** $p=10$ was tuned for speech *separation* with BLSTM networks trained on
           STFT magnitudes, where gradient explosions are a genuine problem. Our networks are log-mel
           *classifiers* with BatchNorm convolutions and a small GRU. Nothing in the paper says $p=10$ is good for
           such a task, and we did not tune it. The optimizer is Adam at $10^{-3}$ (`src/train.py`), which is
           invariant to a constant rescale of the gradient, so clipping matters here only through the
           **relative** scale it imposes across steps.

        ### 5.3 Reproducing the recorded thresholds and clip counts

        The next cell recomputes every threshold from the recorded gradient norms with an independent implementation
        (a sorted list with interpolation), compares it with `clip_threshold`, and counts the clipped steps.
        """),

        code(r"""
        # N1.5.1: reproduce the thresholds from grad_norm; clip counts and rates per arm.
        import bisect
        def prior_quantile_series(g, p=10.0, inclusive=False):
            # Percentile_p of the history at each step, linear interpolation; inclusive=True appends g_t first (paper), False = ours.
            hist, out = [], np.empty(len(g))
            for t, x in enumerate(g):
                if inclusive:
                    bisect.insort(hist, float(x))
                m = len(hist)
                if m == 0:
                    out[t] = np.inf
                else:
                    pos = (m - 1) * p / 100.0; lo = int(np.floor(pos)); hi = min(lo + 1, m - 1)
                    out[t] = hist[lo] * (1 - (pos - lo)) + hist[hi] * (pos - lo)
                if not inclusive:
                    bisect.insort(hist, float(x))
            return out

        CLIP = {}
        for a in ARCHS:
            z = DATA[a]["npz"]; g = z["grad_norm"].astype(float); thr = z["clip_threshold"].astype(float)
            mine = prior_quantile_series(g, 10.0)
            err = np.max(np.abs(mine[1:] - thr[1:]) / thr[1:])
            clipped = g > thr
            CLIP[a] = {"g": g, "thr": thr, "clipped": clipped, "steps": len(g), "n_clip": int(clipped.sum()), "rate": clipped.mean(), "relerr": err}
            print(f"{a:9s} steps={len(g):5d} clipped={int(clipped.sum()):5d} rate={clipped.mean():.4f}  max rel. error of recomputed threshold = {err:.2e}  step-0 threshold = {thr[0]}")
            check(f"N1.5.1[{a}.thr]", err < 1e-9 and np.isinf(thr[0]), f"threshold(t) == percentile_10 of PRIOR history (max rel err {err:.1e}); step 0 is +inf (prior-history convention confirmed)")
        check("N1.5.1a", (CLIP["timepool"]["n_clip"], CLIP["timepool"]["steps"]) == (1284, 1296), "timepool clipped 1284 / 1296 steps")
        check("N1.5.1b", (CLIP["wide"]["n_clip"], CLIP["wide"]["steps"]) == (910, 1296), "wide clipped 910 / 1296 steps")
        check("N1.5.1c", (CLIP["gru"]["n_clip"], CLIP["gru"]["steps"]) == (663, 1008), "gru clipped 663 / 1008 steps")
        check("N1.5.1d", [round(100 * CLIP[a]["rate"], 1) for a in ("timepool", "wide", "gru")] == [99.1, 70.2, 65.8], "clip rates 99.1% / 70.2% / 65.8% (timepool / wide / gru)")
        """),

        md(r"""
        ### 5.4 What clip rate should we expect? A stationary-norm argument

        The clip rate is not a free outcome; a percentile rule with $p=10$ nearly fixes it. Suppose the gradient norms
        were exchangeable draws from one continuous distribution. Then step $t$'s norm is equally likely to occupy any
        rank among $g_1,\dots,g_t$, so the probability that it exceeds the $p$-th percentile of the *earlier* norms
        tends to $1-p/100$:
        $$\Pr\big(g_t>\eta_c(t)\big)\ \longrightarrow\ 1-\frac{p}{100}=0.90\quad(p=10).$$
        **A percentile of 10 clips about nine steps in ten by construction when norms are stationary.** AutoClip
        with $p=10$ is therefore close to *normalised* gradient descent (each update has norm near a low quantile of
        the history), not a rare-event safeguard. The clip rate departs from $0.90$ only through a *trend*. Model the
        norms as $g_t=\exp(\delta\,t+\sigma\varepsilon_t)$ with $\varepsilon_t$ standard normal, and write
        $d=\delta/\sigma$ for the drift per step in units of the noise. If $d<0$ (norms shrink as training
        proceeds) the current norm tends to sit *below* older, larger ones, and since the $10$th percentile of a
        history dominated by large old values sits in its low tail, $g_t$ often falls under $\eta_c$ and is not clipped:
        the rate drops below $0.90$. If $d>0$ (norms grow) $g_t$ exceeds essentially the whole past and the rate goes to
        $1$. The next cell checks this by simulation and then estimates each arm's own $d$ from its recorded norms.
        """),

        code(r"""
        # N1.5.2: Monte-Carlo clip rate versus drift d (FIXTURE: synthetic log-normal norm sequences), then each arm's estimated d.
        D_GRID = np.linspace(-0.006, 0.006, 13)
        def sim_rate(d, n=1300, reps=12, seed=0):
            rng = np.random.default_rng(seed); r = []
            for _ in range(reps):
                g = np.exp(d * np.arange(n) + rng.standard_normal(n))
                th = prior_quantile_series(g, 10.0); r.append((g[1:] > th[1:]).mean())
            return float(np.mean(r)), float(np.std(r, ddof=1) / np.sqrt(reps))
        SIM = np.array([sim_rate(d, seed=RNG_SEED + i) for i, d in enumerate(D_GRID)])
        i0 = int(np.argmin(np.abs(D_GRID)))
        print("simulated clip rate at d = 0:", SIM[i0].round(4), "(stationary theory 1 - p/100 = 0.9)")
        check("N1.5.2a", abs(SIM[i0, 0] - 0.9) < 0.02, f"stationary norms give clip rate {SIM[i0, 0]:.3f} vs 1 - p/100 = 0.900")
        check("N1.5.2b", bool(np.all(np.diff(SIM[:, 0]) > -0.01)), "simulated clip rate is (weakly) increasing in the drift d")
        check("N1.5.2c", SIM[0, 0] < 0.6 and SIM[-1, 0] > 0.98, f"falling norms clip {SIM[0, 0]:.2f}, rising norms clip {SIM[-1, 0]:.3f}")
        ARMD = {}
        for a in ARCHS:
            lg = np.log(CLIP[a]["g"]); x = np.arange(len(lg))
            slope, ic = np.polyfit(x, lg, 1); sd = (lg - (slope * x + ic)).std()
            d = slope / sd; pred = float(np.interp(d, D_GRID, SIM[:, 0]))
            ARMD[a] = {"d": d, "pred": pred, "obs": CLIP[a]["rate"], "first_decile_mean": CLIP[a]["g"][: len(lg) // 10].mean(), "last_decile_mean": CLIP[a]["g"][-(len(lg) // 10):].mean()}
            print(f"{a:9s} drift d = {d:+.5f}   predicted clip rate {pred:.3f}   observed {CLIP[a]['rate']:.3f}   mean norm first/last decile {ARMD[a]['first_decile_mean']:.2f} / {ARMD[a]['last_decile_mean']:.2f}")
        check("N1.5.2d", ARMD["gru"]["d"] < 0 and ARMD["wide"]["d"] < 0 and ARMD["timepool"]["d"] > 0, "gru and wide have falling gradient norms (d < 0); timepool has rising norms (d > 0)")
        check("N1.5.2e", all(abs(ARMD[a]["pred"] - ARMD[a]["obs"]) < 0.08 for a in ARCHS), "drift-only model reproduces each arm's observed clip rate to within 0.08")
        note("N1.5.2f", "timepool's rate is the worst-predicted (rising, heavy-tailed norms); this is a one-parameter log-normal caricature, not a fit of the norm process")
        """),

        code(r"""
        # N1.5.3: the paper's convention (append g_t first) as a counterfactual on the RECORDED norms; and the mean applied scale h_c.
        rows = []
        for a in ARCHS:
            g = CLIP[a]["g"]; thr_paper = prior_quantile_series(g, 10.0, inclusive=True)
            c_ours = g[1:] > CLIP[a]["thr"][1:]; c_paper = g[1:] > thr_paper[1:]
            hc = np.minimum(1.0, CLIP[a]["thr"][1:] / g[1:])
            rows.append({"arch": a, "steps": len(g) - 1, "clipped (ours)": int(c_ours.sum()), "clipped (paper convention)": int(c_paper.sum()),
                         "decisions that flip": int((c_ours != c_paper).sum()), "flip fraction": float((c_ours != c_paper).mean()),
                         "mean h_c (ours)": float(hc.mean()), "median h_c": float(np.median(hc)), "min h_c": float(hc.min())})
        OFF = pd.DataFrame(rows).set_index("arch")
        display(OFF.round(4))
        check("N1.5.3a", (OFF["flip fraction"] < 0.01).all(), f"the off-by-one changes fewer than 1% of clip decisions (max {OFF['flip fraction'].max():.2%}); it is real but small")
        check("N1.5.3b", (OFF["mean h_c (ours)"] < 1).all(), "on average every arm's updates are shrunk (mean h_c < 1)")
        check("N1.5.3c", OFF.loc["timepool", "mean h_c (ours)"] < OFF.loc["gru", "mean h_c (ours)"] and OFF.loc["timepool", "mean h_c (ours)"] < OFF.loc["wide", "mean h_c (ours)"],
              f"timepool's updates are shrunk hardest: mean h_c = {OFF.loc['timepool', 'mean h_c (ours)']:.3f} vs gru {OFF.loc['gru', 'mean h_c (ours)']:.3f}, wide {OFF.loc['wide', 'mean h_c (ours)']:.3f}")
        note("N1.5.3d", "the paper-convention rows are a counterfactual on the recorded norms: a different threshold would have changed the trajectory that produced later norms, so this is not a re-run")
        """),

        code(r"""
        # N1.5.4: counterfactual percentile sweep on the RECORDED norms: what clip rate would each p imply?
        PS = [1, 10, 25, 50, 75, 90]
        SWEEP = pd.DataFrame({a: [float((CLIP[a]["g"][1:] > prior_quantile_series(CLIP[a]["g"], float(p_))[1:]).mean()) for p_ in PS] for a in ARCHS}, index=pd.Index(PS, name="p"))
        SWEEP["stationary 1-p/100"] = [1 - p_ / 100 for p_ in PS]
        display(SWEEP.round(3))
        check("N1.5.4a", all(bool(np.all(np.diff(SWEEP[a].values) < 0)) for a in ARCHS), "clip rate falls monotonically as p rises, for every arm")
        check("N1.5.4b", abs(SWEEP.loc[10, "gru"] - CLIP["gru"]["rate"]) < 0.002 and abs(SWEEP.loc[10, "timepool"] - CLIP["timepool"]["rate"]) < 0.002, "the p = 10 row reproduces the recorded clip rates")
        check("N1.5.4c", float(SWEEP.loc[50, "timepool"]) > 0.9 and float(SWEEP.loc[50, "gru"]) < 0.6, f"even p = 50 clips {SWEEP.loc[50, 'timepool']:.0%} of timepool steps but {SWEEP.loc[50, 'gru']:.0%} of gru steps: the trend decides, not p alone")
        note("N1.5.4d", "counterfactual on recorded norms; a different p would have produced a different training trajectory and different norms")
        fig, ax = plt.subplots(figsize=(8.2, 4.3))
        for a in ARCHS:
            ax.plot(PS, SWEEP[a], marker="o", color=ACOL[a], lw=1.8, label=a)
        ax.plot(PS, SWEEP["stationary 1-p/100"], color="r", ls="--", label="stationary 1 - p/100")
        ax.axvline(10, color="#888", ls=":"); ax.text(11, 0.05, "p = 10 (used)", fontsize=8)
        ax.set_xlabel("percentile p"); ax.set_ylabel("implied clip rate"); ax.legend(fontsize=8); ax.set_title("Clip rate versus percentile on the recorded gradient norms")
        varied("N1.5.f3", SWEEP.values)
        savefig(fig, "fig5_3_percentile_sweep.png")
        """),

        md(r"""
        ### How to read this chart

        Each line is the clip rate that the percentile rule would have produced on an arm's *recorded* gradient-norm
        sequence, for a range of $p$; the red dashed line is the stationary prediction $1-p/100$. The lines fall as
        $p$ rises (a higher percentile means a higher threshold and fewer clipped steps). `timepool`'s line stays
        above the red line at every $p$ (rising norms), while `gru`'s and `wide`'s dip below it (falling norms). The
        vertical dotted line is the $p=10$ that was used. The chart shows that the clip rate is a
        property of the percentile and the trend together; it does not say which $p$ would have trained a better
        model, and because it holds the norm sequence fixed it cannot: changing $p$ changes the trajectory.
        """),

        code(r"""
        # Figure 5.1: per-step gradient norm and threshold, per arm. Clipped steps in red.
        fig, axs = plt.subplots(1, 3, figsize=(15, 4.4), sharey=False)
        for ax, a in zip(axs, ARCHS):
            g, thr, cl = CLIP[a]["g"], CLIP[a]["thr"], CLIP[a]["clipped"]
            st = np.arange(len(g))
            ax.scatter(st[~cl], g[~cl], s=4, color="#9a9a9a", label="not clipped")
            ax.scatter(st[cl], g[cl], s=4, color="#d62728", alpha=0.6, label="clipped")
            ax.plot(st[1:], thr[1:], color="k", lw=1.6, label="threshold eta_c(t) (p=10)")
            ax.set_yscale("log"); ax.set_xlabel("optimizer step"); ax.set_title(f"{a}: {CLIP[a]['n_clip']}/{CLIP[a]['steps']} clipped ({CLIP[a]['rate']:.1%})", color=ACOL[a])
        axs[0].set_ylabel("gradient norm (log)"); axs[0].legend(fontsize=8, markerscale=3)
        varied("N1.5.f1", [CLIP[a]["g"] for a in ARCHS])
        plt.tight_layout(); savefig(fig, "fig5_1_grad_norm_and_threshold.png")
        """),

        md(r"""
        ### How to read this chart

        One panel per architecture; each dot is one optimizer step, plotted at its pre-clip gradient norm on a log
        axis; red dots exceeded the threshold and were clipped, grey dots did not; the black line is the running
        $10$th-percentile threshold. The threshold is a **lagging, monotone-ish floor** because a percentile of a
        cumulative history moves slowly. In `gru` and `wide` the norms fall over training, so the black line is
        eventually *above* many of the new dots (grey dots appear mid-to-late training) and the clip rate settles at
        $66$ to $70\%$. In `timepool` the norms rise after the first few hundred steps and almost every dot is red:
        the threshold is pinned to the earliest small norms and clips essentially every step ($99.1\%$). Takeaway: a
        high clip rate here says the percentile is low relative to the recent norm scale, not that training is unstable.
        What would make this reading wrong: grey and red dots interleaved at the same height early in training,
        which would indicate the rule is doing something other than tracking a history quantile.
        """),

        code(r"""
        # Figure 5.2: clip rates (bars vs stationary 0.90), rolling clip rate over 100 steps, and the drift model.
        fig, axs = plt.subplots(1, 3, figsize=(16, 4.4))
        rates = [CLIP[a]["rate"] for a in ARCHS]
        axs[0].bar(range(3), rates, color=[ACOL[a] for a in ARCHS])
        axs[0].axhline(0.9, color="r", ls="--"); axs[0].text(2.45, 0.91, "stationary theory 0.90", color="r", ha="right", fontsize=8)
        for i, r_ in enumerate(rates):
            axs[0].text(i, r_ + 0.01, f"{r_:.1%}", ha="center", fontsize=9)
        axs[0].set_xticks(range(3)); axs[0].set_xticklabels(ARCHS); axs[0].set_ylim(0, 1.1); axs[0].set_ylabel("fraction of steps clipped"); axs[0].set_title("Real clip rates")
        for a in ARCHS:
            c_ = CLIP[a]["clipped"].astype(float); w_ = 100
            roll = np.convolve(c_, np.ones(w_) / w_, mode="valid")
            axs[1].plot(np.arange(len(roll)) + w_, roll, color=ACOL[a], lw=1.6, label=a)
        axs[1].axhline(0.9, color="r", ls="--", lw=1); axs[1].set_xlabel("optimizer step"); axs[1].set_ylabel("rolling clip rate (100 steps)"); axs[1].legend(fontsize=8); axs[1].set_title("Clip rate through training")
        axs[2].errorbar(D_GRID, SIM[:, 0], yerr=1.96 * SIM[:, 1], color="#555", marker="o", ms=3, label="simulated (log-normal, drift d)")
        for a in ARCHS:
            axs[2].scatter(ARMD[a]["d"], ARMD[a]["obs"], s=90, color=ACOL[a], edgecolor="k", zorder=5, label=f"{a} observed")
        axs[2].axhline(0.9, color="r", ls="--", lw=1); axs[2].set_xlabel("drift d of log gradient norm per step (in sd units)"); axs[2].set_ylabel("clip rate"); axs[2].legend(fontsize=7.5); axs[2].set_title("Drift explains the rate")
        varied("N1.5.f2", rates, SIM[:, 0])
        plt.tight_layout(); savefig(fig, "fig5_2_clip_rates.png")
        """),

        md(r"""
        ### How to read this chart

        Left: the three measured clip rates against the $0.90$ that stationary norms would give (red dashed).
        Middle: the same rate computed over a rolling window of $100$ steps; `timepool` sits near $1$ throughout,
        while `gru` and `wide` start near $0.9$ or above and sag as their norms fall. Right: the simulated
        relationship between clip rate and the drift $d$ of log gradient norms (grey, with $95\%$ Monte-Carlo
        error bars) and each arm's measured $(d,\ \text{rate})$ as a coloured dot. The dots lie close to the curve,
        which supports the reading that *the trend of the norm decides the clip rate*, and that $p=10$ makes the
        mechanism largely independent of any notion of "instability". Caution: this is a one-parameter caricature of
        the norm process; it explains the *rate*, not whether clipping helped or hurt the trained model. We ran no
        arm without clipping, so this notebook makes no claim about AutoClip's benefit for classification.
        """),

        md(r"""
        ## 6. Calibration: temperature scaling, ECE, and MDCA

        ### 6.1 Temperature scaling as a one-parameter maximum-likelihood problem

        Let $z_i\in\mathbb{R}^K$ be the logits of example $i$ with label $y_i$, and let $\beta=1/T>0$. The tempered
        model is $p_i(\beta)=\mathrm{softmax}(\beta z_i)$, and the negative log-likelihood of the calibration set is
        $$\mathrm{NLL}(\beta)=\frac1n\sum_{i=1}^n\Big[-\beta\,z_{i,y_i}+\log\sum_{j=1}^K e^{\beta z_{ij}}\Big].$$
        The log-sum-exp is convex in $\beta$, so $\mathrm{NLL}$ is convex with a unique minimiser (or none, if the
        data are separable). Its derivatives are
        $$\mathrm{NLL}'(\beta)=\frac1n\sum_i\Big(\mathbb{E}_{p_i(\beta)}[z_i]-z_{i,y_i}\Big),\qquad
        \mathrm{NLL}''(\beta)=\frac1n\sum_i\mathrm{Var}_{p_i(\beta)}(z_i)\ \ge 0 .$$
        The stationarity condition $\mathrm{NLL}'(\beta^\star)=0$ is **moment matching**: the model-expected logit,
        averaged over the calibration set, equals the average logit of the observed label. Near the optimum,
        $$\mathrm{NLL}(1)-\mathrm{NLL}(\beta^\star)\ \approx\ \tfrac12\,\overline{V}\,(1-\beta^\star)^2,$$
        with $\overline V$ the mean logit variance, so the likelihood gain from temperature scaling grows with the
        *square of the distance of $T^\star$ from one*. Temperature scaling cannot change the arg-max, hence never
        changes accuracy; it changes NLL, Brier score and ECE. $T^\star>1$ flattens an overconfident model and
        $T^\star<1$ sharpens an underconfident one. The stored artifacts carry the fitted $T^\star$ per arm, and
        the NLL and ECE before and after; they do **not** carry logits, so nothing below can re-derive $T^\star$
        or bootstrap its uncertainty. Section 7 verifies the estimator itself on constructed logits.

        ### 6.2 ECE and its binning sensitivity

        The top-label expected calibration error with $B$ equal-width confidence bins $S_1,\dots,S_B$ is
        $$\mathrm{ECE}_B=\sum_{b=1}^B\frac{|S_b|}{n}\,\Big|\mathrm{acc}(S_b)-\mathrm{conf}(S_b)\Big|,$$
        with $\mathrm{conf}(S_b)$ the mean top-label probability and $\mathrm{acc}(S_b)$ the fraction correct in the
        bin. It depends on $B$ (more bins expose more within-bin structure but each bin is noisier), it is biased
        upward at finite $n$ even for a perfectly calibrated model, and it is not the quantity temperature scaling
        minimises, which is NLL. The result files record $\mathrm{ECE}_{10}$, $\mathrm{ECE}_{15}$ and
        $\mathrm{ECE}_{20}$ for every arm, condition and calibration state.
        """),

        code(r"""
        # N1.6.1: ECE at B = 10/15/20 for every arm x condition, uncalibrated and calibrated; TS effect; the fitted temperatures.
        TEMP = {a: DATA[a]["res"]["summary"]["temperature"] for a in ARCHS}
        rows = []
        for a in ARCHS:
            for c in CONDS:
                u, k = S(a, c, "uncalibrated"), S(a, c, "calibrated")
                rows.append({"arch": a, "cond": c, "T": TEMP[a], "ECE10_u": u["ece_10"], "ECE15_u": u["ece_15"], "ECE20_u": u["ece_20"],
                             "ECE15_cal": k["ece_15"], "dECE15(TS)": k["ece_15"] - u["ece_15"], "NLL_u": u["nll"], "NLL_cal": k["nll"], "dNLL(TS)": k["nll"] - u["nll"]})
        CAL = pd.DataFrame(rows)
        display(CAL.round(4).set_index(["arch", "cond"]))
        check("N1.6.1a", [round(TEMP[a], 4) for a in ("gru", "timepool", "wide")] == [1.0157, 1.1097, 1.6002], f"fitted temperatures gru {TEMP['gru']:.4f}, timepool {TEMP['timepool']:.4f}, wide {TEMP['wide']:.4f}")
        clean = CAL[CAL.cond == "clean"].set_index("arch")
        check("N1.6.1b", (clean["dNLL(TS)"] <= 1e-9).all(), "on the clean test split TS never raises NLL (the objective it minimises, fitted on the calibration split)")
        check("N1.6.1c", clean["dNLL(TS)"].idxmin() == "wide" and abs(np.log(TEMP["wide"])) > abs(np.log(TEMP["timepool"])) > abs(np.log(TEMP["gru"])),
              "clean NLL gain from TS is largest for wide, whose |ln T| is also largest (quadratic-gain law, qualitatively)")
        note("N1.6.1d", f"timepool: TS lowers clean NLL ({clean.loc['timepool', 'dNLL(TS)']:+.4f}) but RAISES clean ECE_15 {clean.loc['timepool', 'ECE15_u']:.4f} -> {clean.loc['timepool', 'ECE15_cal']:.4f}: ECE and NLL are different objectives")
        spread = CAL[["ECE10_u", "ECE15_u", "ECE20_u"]]
        rel_spread = ((spread.max(axis=1) - spread.min(axis=1)) / spread.mean(axis=1))
        print("relative spread of uncalibrated ECE across B in {10,15,20}: max =", round(float(rel_spread.max()), 3), "| median =", round(float(rel_spread.median()), 3))
        orders = {c: tuple(clean[c].sort_values().index) for c in ["ECE10_u", "ECE15_u", "ECE20_u"]}
        rank_same = len(set(orders.values())) == 1
        note("N1.6.1e", f"clean ranking of arms by uncalibrated ECE is {'the same' if rank_same else 'NOT the same'} for B = 10, 15, 20: {list(orders['ECE15_u'])}")
        check("N1.6.1f", rel_spread.max() > 0.10, f"the same model's ECE moves by up to {rel_spread.max():.0%} of its mean when only B changes: absolute ECE values are bin-count dependent")
        """),

        code(r"""
        # Figure 6.1: ECE sensitivity to bin count, and the TS effect on ECE_15 for clean vs noise_10.
        fig, axs = plt.subplots(1, 2, figsize=(14, 4.6), gridspec_kw={"width_ratios": [1.5, 1]})
        lab = [f"{a}\n{c}" for a in ARCHS for c in CONDS]
        xs = np.arange(len(lab))
        for col, mk, off in zip(["ECE10_u", "ECE15_u", "ECE20_u"], ["o", "s", "^"], [-0.2, 0, 0.2]):
            axs[0].scatter(xs + off, CAL[col].values, marker=mk, s=28, label=col.replace("_u", "").replace("ECE", "B="), color="#333")
        axs[0].set_yscale("log"); axs[0].set_xticks(xs); axs[0].set_xticklabels(lab, fontsize=6.5); axs[0].set_ylabel("uncalibrated ECE (log)"); axs[0].legend(fontsize=8)
        axs[0].set_title("ECE of the same predictions under 3 bin counts")
        w = 0.36
        for j, c in enumerate(["clean", "noise_10"]):
            sub = CAL[CAL.cond == c].set_index("arch").loc[ARCHS]
            axs[1].bar(np.arange(3) + (j - 0.5) * w, sub["ECE15_u"], width=w * 0.45, color=[ACOL[a] for a in ARCHS], alpha=0.5 + 0.5 * j, label=None)
            axs[1].bar(np.arange(3) + (j - 0.5) * w + w * 0.5, sub["ECE15_cal"], width=w * 0.45, color=[ACOL[a] for a in ARCHS], hatch="//", alpha=0.5 + 0.5 * j)
        axs[1].set_yscale("log"); axs[1].set_xticks(np.arange(3)); axs[1].set_xticklabels([f"{a}\nT={TEMP[a]:.3f}" for a in ARCHS])
        axs[1].set_ylabel("ECE_15 (log)"); axs[1].set_title("Before (solid) / after (hatched) TS; light = clean, dark = noise_10")
        varied("N1.6.f1", CAL[["ECE10_u", "ECE15_u", "ECE20_u", "ECE15_cal"]].values)
        plt.tight_layout(); savefig(fig, "fig6_1_ece_bins_and_ts.png")
        """),

        md(r"""
        ### How to read this chart

        Left: for each of the twelve arm-and-condition pairs, the uncalibrated ECE computed with $10$, $15$ and $20$
        bins (three markers per column, log axis). Where the markers stack, the choice of $B$ does not matter; where
        they spread, it does. On the noise conditions the three markers nearly coincide at very high values, which suggests the
        confidences are concentrated in few bins. On the clean condition, where ECE is a few percent, the same model
        can differ visibly between bin counts, which is why ECE differences of a percentage point between models
        should not be over-read. Right: ECE$_{15}$ before (solid) and after (hatched) temperature scaling, light bars
        for `clean` and dark bars for `noise_10`, with each arm's fitted $T$ in the axis label. `wide`, with $T=1.60$,
        gets a large ECE reduction on `clean`; `gru`, with $T=1.016$, barely moves; `timepool`'s clean ECE *rises*.
        On `noise_10` every arm stays at ECE near $0.5$ to $0.75$ after scaling: a temperature fitted on clean data
        cannot repair a shifted distribution.

        ### 6.3 MDCA: the formula, the reported numbers, and what it does and does not guarantee

        (Hebbalaguppe et al., arXiv:2203.13834; turns 7 and 8.) MDCA is an auxiliary loss added to a primary loss
        (cross-entropy or focal). With $K$ classes and a minibatch of $N_b$ examples, $s_i[j]$ the softmax
        probability of class $j$ for example $i$ and $q_i[j]\in\{0,1\}$ the one-hot ground truth,
        $$\mathcal{L}_{\mathrm{MDCA}}=\frac1K\sum_{j=1}^K\left|\frac1{N_b}\sum_{i=1}^{N_b}s_i[j]-\frac1{N_b}\sum_{i=1}^{N_b}q_i[j]\right| .$$
        It compares, for each class, the batch-average predicted probability with the batch frequency of that class,
        needs no bins, and is differentiable almost everywhere:
        $\partial\mathcal{L}_{\mathrm{MDCA}}/\partial s_i[j]=\frac{1}{KN_b}\mathrm{sign}(\bar s_j-\bar q_j)$, i.e. it nudges all
        examples' probability of class $j$ down by an equal amount when class $j$ is over-predicted on the batch, and up
        when under-predicted. On PASCAL-VOC 2012 with DeepLabV3+ and an Xception65 backbone the paper reports:

        | training loss | post-hoc temperature scaling | ECE (%) |
        |---|---|---|
        | NLL | no | 7.77 |
        | NLL | yes | 6.10 |
        | focal loss | no | 7.69 |
        | focal loss + MDCA | no | 4.66 |

        The improvement of FL + MDCA over post-hoc scaling is $(6.10-4.66)/6.10=23.6\%$ relative. The paper also
        tests composition and reports that **MDCA does not compose with temperature scaling**: scaling after MDCA
        training leaves calibration unchanged or slightly degraded, because the optimal temperature found after MDCA
        training is $T\approx1$ (turn 8).

        **A note on what MDCA-zero means.** The loss is zero when, for every class, the batch mean probability equals
        the class frequency. That is a *marginal* (class-average) condition. It is weaker than the temperature
        stationarity $\mathrm{NLL}'(\beta^\star)=0$ and weaker than top-label calibration. So "$T^\star\approx1$ after
        MDCA training" is the paper's *empirical* finding for FL + MDCA, where the focal or cross-entropy term is
        still present, and not a consequence of the MDCA term alone. The fixture below makes that concrete.
        """),

        code(r"""
        # N1.6.2: MDCA as code, and a FIXTURE showing MDCA = 0 does not imply calibration or T* = 1.
        from src.calibration import temperature_scale
        from src.metrics import expected_calibration_error
        def mdca(probs, labels):
            K = probs.shape[1]; q = np.eye(K)[labels]
            return float(np.mean(np.abs(probs.mean(0) - q.mean(0))))
        def softmax_(z):
            z = z - z.max(1, keepdims=True); e = np.exp(z); return e / e.sum(1, keepdims=True)

        rngm = np.random.default_rng(RNG_SEED + 62)
        Kc, n_fx = 3, 3000
        y_fx2 = np.repeat(np.arange(Kc), n_fx // Kc)
        # Fixture A: overconfident and uninformative -- predicted class uniform at random, independent of the label.
        pred = rngm.integers(0, Kc, n_fx)
        zA = np.full((n_fx, Kc), 0.0); zA[np.arange(n_fx), pred] = 8.0
        pA = softmax_(zA)
        # Fixture B: informative, calibrated by construction (beta = 2 for a unit mean shift), then over-sharpened by a factor 1.6.
        zB = 1.6 * 2.0 * (2.0 * np.eye(Kc)[y_fx2] + rngm.standard_normal((n_fx, Kc)))
        pB = softmax_(zB)
        lossA, lossB = mdca(pA, y_fx2), mdca(pB, y_fx2)
        eceA, eceB = expected_calibration_error(pA, y_fx2, bins=15), expected_calibration_error(pB, y_fx2, bins=15)
        TA, TB = temperature_scale(zA, y_fx2), temperature_scale(zB, y_fx2)
        print(f"fixture A (overconfident, label-independent): MDCA={lossA:.4f}  ECE_15={eceA:.3f}  acc={np.mean(pA.argmax(1) == y_fx2):.3f}  fitted T*={TA:.2f}")
        print(f"fixture B (informative, over-sharpened by 1.6)   : MDCA={lossB:.4f}  ECE_15={eceB:.3f}  acc={np.mean(pB.argmax(1) == y_fx2):.3f}  fitted T*={TB:.2f}")
        ref = float(np.mean([abs(pA[:, j].mean() - (y_fx2 == j).mean()) for j in range(Kc)]))
        check("N1.6.2a", abs(lossA - ref) < 1e-12, "MDCA implementation equals the explicit (1/K) sum_j |mean s[j] - mean q[j]|")
        check("N1.6.2b", lossA < 0.03 and eceA > 0.5, f"fixture A: MDCA = {lossA:.4f} (near 0) while ECE_15 = {eceA:.2f}: MDCA-zero does not imply calibration")
        check("N1.6.2c", TA > 2.5, f"fixture A: post-hoc temperature would still be T* = {TA:.2f}, far from 1: MDCA-zero does not imply T* = 1")
        check("N1.6.2d", lossB < 0.03 and 1.4 < TB < 1.8, f"fixture B: an informative but over-sharpened model has small MDCA ({lossB:.3f}) and T* = {TB:.2f}, close to the built-in 1.6")
        check("N1.6.2e", abs((6.10 - 4.66) / 6.10 - 0.236) < 5e-4, f"relative improvement of FL+MDCA over NLL+TS = {(6.10 - 4.66) / 6.10:.4f} = 23.6% (paper)")
        PASCAL = {"NLL": 7.77, "NLL + TS": 6.10, "focal": 7.69, "focal + MDCA": 4.66}
        note("N1.6.2f", f"paper: MDCA vs plain NLL baseline is {(7.77 - 4.66) / 7.77:.1%} relative; the 23.6% figure is against the stronger post-hoc-TS comparator")
        """),

        code(r"""
        # N1.6.3: different objectives, different optimal temperatures (FIXTURE B logits from N1.6.2; true T0 = 1.6 by construction).
        Ts = np.exp(np.linspace(np.log(0.7), np.log(3.0), 70))
        acc_T, nll_T, br_T, ece_T = [], [], [], []
        for T_ in Ts:
            pT = softmax_(zB / T_)
            acc_T.append(np.mean(pT.argmax(1) == y_fx2)); nll_T.append(float(-np.log(pT[np.arange(n_fx), y_fx2]).mean()))
            br_T.append(float(((pT - np.eye(Kc)[y_fx2]) ** 2).sum(1).mean())); ece_T.append(expected_calibration_error(pT, y_fx2, bins=15))
        acc_T, nll_T, br_T, ece_T = map(np.array, (acc_T, nll_T, br_T, ece_T))
        T_nll, T_br, T_ece = Ts[nll_T.argmin()], Ts[br_T.argmin()], Ts[ece_T.argmin()]
        print(f"argmin T: NLL {T_nll:.3f} | Brier {T_br:.3f} | ECE_15 {T_ece:.3f}   (built-in over-sharpening factor 1.6, fitted src T* {TB:.3f})")
        check("N1.6.3a", float(np.ptp(acc_T)) == 0.0, "temperature scaling never changes accuracy (argmax is invariant to T)")
        check("N1.6.3b", abs(T_nll - TB) < 0.06 and abs(T_nll - 1.6) < 0.15, f"grid argmin of NLL ({T_nll:.3f}) matches src.calibration.temperature_scale ({TB:.3f}) and the built-in 1.6")
        check("N1.6.3c", abs(T_br - T_nll) < 0.3, f"Brier is minimised at a nearby but not identical temperature ({T_br:.3f} vs {T_nll:.3f}): both proper, different curvature")
        note("N1.6.3d", f"ECE_15 is minimised at T = {T_ece:.3f}, NLL at {T_nll:.3f}: the ECE curve is flat and noisy near its floor, so minimising ECE is a poorly conditioned target")
        fig, axs = plt.subplots(1, 3, figsize=(15, 4.1))
        for ax, y_, ttl, tb in zip(axs, [nll_T, br_T, ece_T], ["NLL", "Brier", "ECE_15"], [T_nll, T_br, T_ece]):
            ax.plot(Ts, y_, color="k", lw=1.8); ax.axvline(tb, color="#d62728", ls="--", label=f"argmin T = {tb:.2f}"); ax.axvline(1.6, color=ACOL["wide"], ls=":", label="built-in 1.6")
            ax.set_xscale("log"); ax.set_xlabel("temperature T (log)"); ax.set_title(ttl); ax.legend(fontsize=8)
        varied("N1.6.f3", nll_T, br_T, ece_T)
        plt.tight_layout(); savefig(fig, "fig6_3_objective_optima.png")
        """),

        md(r"""
        ### How to read this chart

        One panel per calibration objective, each a function of temperature $T$ on a constructed, informative,
        deliberately over-sharpened classifier (a fixture; the true temperature is $1.6$, green dotted). Red dashed
        lines mark each objective's minimiser. NLL, the objective that temperature scaling fits, has a smooth bowl
        with its minimum near $1.6$. Brier has its minimum nearby but not at exactly the same place. ECE has a much
        flatter valley with a jagged floor of binning noise, so its minimiser is poorly determined. This is the
        reason Section 6 found that scaling can lower NLL while raising ECE for `timepool`: the two are different
        targets, and a train-time term such as MDCA that acts on class-average probabilities is a third. Accuracy
        (checked, not plotted) is constant across all $T$.
        """),

        code(r"""
        # Figure 6.2: PASCAL-VOC numbers from the paper, and the fitted temperatures of our three arms against T = 1.
        fig, axs = plt.subplots(1, 2, figsize=(12.5, 4.4))
        names = list(PASCAL); vals = [PASCAL[k] for k in names]
        bars = axs[0].bar(range(4), vals, color=["#888", "#4c72b0", "#c9a", "#3A8D68"])
        for i, v in enumerate(vals):
            axs[0].text(i, v + 0.12, f"{v:.2f}", ha="center")
        axs[0].annotate("", xy=(3, 4.66), xytext=(1, 6.10), arrowprops=dict(arrowstyle="->", color="r"))
        axs[0].text(2.0, 6.6, "-23.6% relative", color="r", ha="center", fontsize=9)
        axs[0].set_xticks(range(4)); axs[0].set_xticklabels(names, fontsize=8.5); axs[0].set_ylabel("ECE (%)"); axs[0].set_ylim(0, 9)
        axs[0].set_title("Paper (PASCAL-VOC 2012, DeepLabV3+/Xception65)")
        tv = [TEMP[a] for a in ARCHS]
        axs[1].bar(range(3), np.array(tv) - 1.0, bottom=1.0, color=[ACOL[a] for a in ARCHS])
        axs[1].axhline(1.0, color="k", lw=1); axs[1].text(2.45, 1.015, "T = 1 (what MDCA training reaches per the paper)", ha="right", fontsize=8)
        for i, v in enumerate(tv):
            axs[1].text(i, v + 0.02, f"{v:.4f}", ha="center")
        axs[1].set_xticks(range(3)); axs[1].set_xticklabels([f"{a}\nacc {SUMM.loc[a, 'acc@best_epoch']:.3f}" for a in ARCHS]); axs[1].set_ylim(0.9, 1.75)
        axs[1].set_ylabel("fitted post-hoc temperature T*"); axs[1].set_title("Ours: fitted T* per arm (no MDCA used)")
        varied("N1.6.f2", vals, tv)
        plt.tight_layout(); savefig(fig, "fig6_2_mdca_and_temperatures.png")
        """),

        md(r"""
        ### How to read this chart

        Left: the paper's four ECE values on PASCAL-VOC, as bars, with the $23.6\%$ relative reduction of
        FL + MDCA over NLL + temperature scaling as the red arrow. These are the paper's numbers, for a
        semantic-segmentation task, and are shown for scale; they are not comparable to our ECE values, which are
        for 12-way clip classification. Right: our own fitted temperatures, one per architecture, as bars anchored at
        $T=1$ (the black line, where the paper says MDCA training leaves the optimal temperature). `gru` ($1.0157$) is
        essentially on the line, `timepool` ($1.1097$) a little above it, and `wide` ($1.6002$) far above. No arm was
        trained with MDCA here.

        ### 6.4 An observation worth testing, and no more

        **The observation.** The paper's mechanism of interest is that a train-time objective can make the optimal
        post-hoc temperature approach $1$. Our best model, `gru`, already has $T^\star=1.0157$ with *no* train-time
        calibration objective; our worst-calibrated (on clean data) model, `wide`, needs $T^\star=1.6002$, and scaling
        cuts its clean ECE$_{15}$ from $0.0652$ to $0.0248$. If a train-time calibration term were added to
        `wide`, the paper's account predicts its $T^\star$ should fall toward $1$; for `gru` it predicts little
        change, because there is little to remove.

        **Why this is not a result.** (i) One training run per arm: $T^\star$ carries seed noise we cannot estimate.
        (ii) $T^\star$ was fitted on a single calibration split of $1080$ clips; logits are not stored, so it cannot be
        bootstrapped here. (iii) `gru`'s temperature is near $1$ on *clean* data only: on `noise_10` its calibrated
        NLL is still $4.404$, worse than a uniform predictor ($\ln12=2.485$), so "nearly calibrated in distribution"
        must not be read as "calibrated". (iv) Accuracy and calibration are entangled: `gru` is also the most
        accurate arm, and better-separated logits change $T^\star$ for reasons unrelated to any calibration
        objective. (v) A relation between fitted $T^\star$ and a training objective that we did not train is a
        hypothesis about a counterfactual. **The test that would settle it:** train `gru` and `wide` at several
        seeds with and without an MDCA term and record $|\ln T^\star|$ on the same calibration split; the paper's
        account predicts a drop for `wide` and no change for `gru`.
        """),
    ]
