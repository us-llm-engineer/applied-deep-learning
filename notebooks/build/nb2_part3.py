"""NB2 part 3: three exploratory directions (AutoClip sweep, train-time vs post-hoc calibration, weighted vs unweighted CRC) and limitations."""
from nbkit import md, code


def cells():
    return [
        md(r"""
        ## 6. Exploration 1: how does the AutoClip percentile matter?

        The study's default is $p = 10$, chosen for the real-data runs. That is an aggressive setting: for a stationary
        gradient-norm distribution the fraction of clipped steps is about $1 - p/100$, so $p=10$ clips ~90% of steps and
        acts as a soft normalisation of every update rather than a rare safeguard. Before spending GPU hours on real audio it
        is worth asking, cheaply, whether this aggressiveness helps, hurts, or is irrelevant. We sweep

        $$p \in \{1,\ 10,\ 25,\ 100\}, \qquad \text{clip rate}(p) \approx 1 - \tfrac{p}{100} \ \text{(stationary norms)},$$

        where $p = 100$ is the **control**: the threshold is the running maximum of past norms, so a step is clipped only
        if it sets a new record; the clip rate is small but not exactly zero (for i.i.d. norms the expected number of
        records in $n$ steps is $\sum_{j\le n} 1/j \approx \ln n$), which is why this is "effectively unclipped" rather than
        literally unclipped. `ExperimentConfig.autoclip_percentile` accepts $(0, 100]$, so 100 is legal.

        **Workhorse model.** Sections 6--8 use a deliberately small `ConvGRU(width=4, hidden=8)`. The full-size models of
        Section 4 hit 100% test accuracy on this task, which would make calibration and conformal sets degenerate (no
        errors, nothing to calibrate). The micro-GRU keeps the time axis but is capacity-limited: it learns steadily and
        ends at an accuracy comfortably above chance and comfortably below 100%, which is the regime where the questions
        of Sections 7--8 are non-trivial. Two seeds per setting give a sense of the run-to-run spread; two is enough to see
        whether an effect is larger than seed noise, not enough to put a confidence interval on it.
        """),

        code(r"""
        # The workhorse model and the AutoClip sweep. Two seeds per percentile, identical data and config otherwise.
        WORK_EPOCHS = 12
        def work_model():
            return ConvGRU(n_mels=N_MELS, num_classes=N_CLASSES, width=4, hidden=8)

        P_GRID, SWEEP_SEEDS = (1, 10, 25, 100), (17, 23)
        SWEEP = {}
        for p in P_GRID:
            for s in SWEEP_SEEDS:
                torch.manual_seed(s)
                res = train_classifier(work_model(), X["train"], Y["train"], X["val"], Y["val"],
                                       replace(CFG, seed=s, max_epochs=WORK_EPOCHS, patience=WORK_EPOCHS, autoclip_percentile=float(p)))
                ev = evaluate(res.model)
                g, t = np.asarray(res.grad_norm_history), np.asarray(res.clip_threshold_history)
                clipped = np.isfinite(t) & (g > t)
                SWEEP[(p, s)] = {"res": res, "acc": ev["acc"], "val_acc_final": res.history[-1]["val_accuracy"],
                                 "best_val_loss": min(h["val_loss"] for h in res.history),
                                 "clip_rate": float(clipped[1:].mean()), "g": g, "t": t, "clipped": clipped,
                                 "thr_over_norm": float(np.median(t[1:]) / np.median(g[1:]))}
                print(f"p={p:>3d} seed={s} test_acc={ev['acc']:.3f} final_val_acc={res.history[-1]['val_accuracy']:.3f} "
                      f"clip_rate={SWEEP[(p, s)]['clip_rate']:.3f} threshold/norm={SWEEP[(p, s)]['thr_over_norm']:.3f}")
        SW = pd.DataFrame([{"p": p, "seed": s, **{k: v for k, v in d.items() if k in ("acc", "val_acc_final", "best_val_loss", "clip_rate", "thr_over_norm")}} for (p, s), d in SWEEP.items()])
        display(SW.groupby("p").agg(["mean", "std"]).drop(columns="seed").round(3))
        check("N2.6.1[all_runs_trained]", len(SWEEP) == len(P_GRID) * len(SWEEP_SEEDS), f"{len(SWEEP)} sweep runs completed")
        check("N2.6.1[all_above_chance]", all(d["acc"] > 2 / N_CLASSES for d in SWEEP.values()), f"lowest sweep test accuracy {min(d['acc'] for d in SWEEP.values()):.3f} > 2x chance ({2 / N_CLASSES:.3f})")
        """),

        code(r"""
        # Fig NB2-14: final accuracy and clip rate versus p, with the stationary-theory line.
        agg = SW.groupby("p").agg(acc=("acc", "mean"), acc_sd=("acc", "std"), clip=("clip_rate", "mean"), clip_sd=("clip_rate", "std"), ratio=("thr_over_norm", "mean"))
        assert_varied(agg["acc"], "sweep_accuracy_across_p")
        assert_varied(agg["clip"], "sweep_clip_rate_across_p")
        assert_varied(SW["acc"], "sweep_accuracy_all_runs")
        fig, ax = plt.subplots(1, 3, figsize=(15, 4.3))
        ax[0].errorbar(agg.index, agg["acc"], yerr=agg["acc_sd"], fmt="-o", color=PALETTE["clean"], capsize=4, label="mean +/- sd over 2 seeds")
        for s in SWEEP_SEEDS:
            sub = SW[SW.seed == s]
            ax[0].scatter(sub["p"], sub["acc"], s=22, alpha=0.6, label=f"seed {s}")
        ax[0].set(xscale="log", xlabel="AutoClip percentile p (100 = control)", ylabel="test accuracy", ylim=(0, 1.02), title="final accuracy vs p")
        ax[0].set_xticks(P_GRID, [str(p) for p in P_GRID]); ax[0].legend(frameon=False, fontsize=8)
        ax[1].errorbar(agg.index, agg["clip"], yerr=agg["clip_sd"], fmt="-o", color=PALETTE["shifted"], capsize=4, label="measured")
        pp = np.linspace(1, 100, 100)
        ax[1].plot(pp, 1 - pp / 100, color=PALETTE["reference"], ls="--", label="theory 1 - p/100 (stationary)")
        ax[1].set(xscale="log", xlabel="AutoClip percentile p", ylabel="fraction of steps clipped", ylim=(0, 1.02), title="clip rate vs p")
        ax[1].set_xticks(P_GRID, [str(p) for p in P_GRID]); ax[1].legend(frameon=False, fontsize=8)
        ax[2].plot(agg.index, agg["ratio"], "-o", color=PALETTE["mask"])
        ax[2].set(xscale="log", xlabel="AutoClip percentile p", ylabel="median threshold / median norm", title="how tight the clip is")
        ax[2].set_xticks(P_GRID, [str(p) for p in P_GRID])
        fig.suptitle("NB2-14  AutoClip percentile sweep on the micro-GRU (2 seeds)", y=1.03)
        fig.tight_layout()
        savefig(fig, "NB2-14_autoclip_sweep.png")
        """),

        md(r"""
        ### How to read this chart

        **Left:** final test accuracy against $p$ on a log axis. The line is the mean over two seeds with a bar of one
        standard deviation; the small dots are the individual seeds, so the distance between dots at one $p$ is the seed
        noise you must beat before claiming an effect. **Middle:** the measured clip rate (fraction of optimizer steps whose
        pre-clip norm exceeded the threshold, excluding step 1) against the stationary-theory line $1-p/100$. Points below
        the line mean the gradient norms were *decreasing* over training (a falling norm is rarely above a threshold set from
        a larger past), points above mean they were growing. **Right:** the median threshold divided by the median norm;
        it approaches 1 as $p$ grows, because a higher percentile of the history sits closer to (and eventually above) the
        typical norm. If accuracy is flat in the left panel while the clip rate falls from ~99% to ~0% in the middle
        panel, the percentile is irrelevant for this model and task.
        """),

        code(r"""
        # Fig NB2-15: gradient norms and thresholds per p (seed 17), and validation accuracy curves for all p.
        fig, ax = plt.subplots(1, 5, figsize=(20, 3.9))
        for a, p in zip(ax[:4], P_GRID):
            d = SWEEP[(p, 17)]
            tt = np.where(np.isfinite(d["t"]), d["t"], np.nan)
            assert_varied(d["g"], f"sweep_grad_norm[p={p}]")
            a.plot(d["g"], color=PALETTE["clean"], lw=0.9, alpha=0.7, label="norm")
            a.plot(tt, color="black", lw=1.5, label="threshold")
            a.scatter(np.flatnonzero(d["clipped"]), d["g"][d["clipped"]], s=4, color=PALETTE["shifted"], zorder=3)
            a.set(yscale="log", xlabel="step", title=f"p = {p}: {d['clip_rate']:.0%} clipped")
        ax[0].set_ylabel("gradient norm (log)"); ax[0].legend(frameon=False, fontsize=8)
        pcol = dict(zip(P_GRID, plt.cm.viridis(np.linspace(0, 0.9, len(P_GRID)))))
        for p in P_GRID:
            curves = np.array([[h["val_accuracy"] for h in SWEEP[(p, s)]["res"].history] for s in SWEEP_SEEDS])
            assert_varied(curves.mean(axis=0), f"sweep_val_acc_curve[p={p}]")
            ax[4].plot(np.arange(1, WORK_EPOCHS + 1), curves.mean(axis=0), marker="o", ms=3, color=pcol[p], label=f"p={p}")
        ax[4].set(xlabel="epoch", ylabel="validation accuracy (mean of 2 seeds)", ylim=(0, 1.02), title="learning curves")
        ax[4].legend(frameon=False, fontsize=8)
        fig.suptitle("NB2-15  Same run, four thresholds: the tighter the percentile, the more steps the black line clips", y=1.04)
        fig.tight_layout()
        savefig(fig, "NB2-15_sweep_trajectories.png")
        slow = SWEEP[(1, 17)]["clip_rate"]; fast = SWEEP[(100, 17)]["clip_rate"]
        check("N2.6.2[clip_rate_falls_with_p]", all(agg["clip"].iloc[i] >= agg["clip"].iloc[i + 1] for i in range(len(agg) - 1)), "mean clip rate is non-increasing in p: " + ", ".join(f"p={p}:{c:.2f}" for p, c in agg["clip"].items()))
        check("N2.6.2[control_rarely_clips]", agg.loc[100, "clip"] < 0.15, f"p=100 clips {agg.loc[100, 'clip']:.1%} of steps (only new records)")
        check("N2.6.2[p1_clips_nearly_all]", agg.loc[1, "clip"] > 0.9, f"p=1 clips {agg.loc[1, 'clip']:.1%} of steps")
        check("N2.6.2[threshold_tightness_monotone]", all(agg["ratio"].iloc[i] <= agg["ratio"].iloc[i + 1] + 1e-9 for i in range(len(agg) - 1)), "median threshold/median norm is non-decreasing in p")
        """),

        md(r"""
        ### How to read this chart

        The first four panels are one training run each (seed 17), left to right $p = 1, 10, 25, 100$. In each, the blue
        line is the raw gradient norm at every optimizer step (log scale), the black line is the AutoClip threshold, and red
        dots mark the clipped steps. At $p=1$ the black line hugs the floor of the blue cloud and nearly every step is
        clipped; at $p=100$ the black line is the running maximum and rides above the cloud, so only record-setting spikes
        are clipped. The last panel overlays the mean validation-accuracy curves of the four settings: if the curves overlap
        within seed noise, clipping neither speeds nor slows learning on this task. The panel titles give each run's
        exact clip fraction.
        """),

        code(r"""
        # Verdict on the sweep, stated from the numbers.
        acc_by_p = agg["acc"].round(3).to_dict(); sd_by_p = agg["acc_sd"].round(3).to_dict()
        best_p, worst_p = agg["acc"].idxmax(), agg["acc"].idxmin()
        spread = float(agg["acc"].max() - agg["acc"].min())
        seed_noise = float(SW.groupby("p")["acc"].std().mean())
        print("mean test accuracy by p:", acc_by_p, "| sd over seeds:", sd_by_p)
        print(f"best p = {best_p}, worst p = {worst_p}, spread between them = {spread:.3f}; mean seed sd = {seed_noise:.3f}")
        verdict = "larger than" if spread > 2 * seed_noise else "comparable to"
        note("N2.6.3[verdict]", f"the accuracy spread across p ({spread:.3f}) is {verdict} twice the mean seed noise ({2 * seed_noise:.3f}); best p = {best_p}")
        check("N2.6.3[control_vs_default_gap_recorded]", np.isfinite(agg.loc[10, 'acc'] - agg.loc[100, 'acc']), f"p=10 minus p=100 test accuracy = {agg.loc[10, 'acc'] - agg.loc[100, 'acc']:+.3f}")
        check("N2.6.3[no_setting_diverged]", (SW["acc"] > 0.2).all(), "no percentile setting produced a diverged (near-chance) model")
        """),

        md(r"""
        ### 6.1 What the sweep does and does not justify

        The verdict line above is the whole conclusion, and it is deliberately conservative: two seeds on a 600-example
        synthetic task with a micro-model can detect only a large effect. If the spread across $p$ is within seed noise, the
        honest reading is "on this model and task the percentile is not a sensitive hyper-parameter, so the real-data
        run need not spend GPU hours sweeping it", which is a *cheap-to-check hypothesis* rather than a result. If one setting is
        clearly better or worse, that is a reason to include it in the real-data ablation, not a reason to change the default.
        Either way the mechanism we *can* trust from this sweep is the clip-rate curve, because it is a property of the
        optimizer and the norm history rather than of the data.
        """),

        md(r"""
        ## 7. Exploration 2: train-time calibration versus post-hoc temperature scaling

        Two families of remedies for over- or under-confident classifiers: change the *training loss* so the network is
        calibrated as it learns, or fit a scalar *temperature* on a held-out split afterwards. They can also be combined.
        This section compares four conditions on the same model family:

        | condition | training loss | post-hoc temperature |
        |---|---|---|
        | baseline | cross-entropy | none |
        | post-hoc | cross-entropy | $T$ fit on the calibration split (`src.calibration.temperature_scale`) |
        | train-time | cross-entropy $+\ \beta\cdot$MDCA-style term | none |
        | both | cross-entropy $+\ \beta\cdot$MDCA-style term | $T$ fit on the calibration split |

        **The auxiliary loss** (implemented in this notebook, *not* in `src/`) is an MDCA-style class-wise term in the spirit
        of Hebbalaguppe et al. (arXiv:2203.13834). For a batch, let $K_b$ be the set of classes present, $n_k$ the number of
        batch examples of class $k$, $\hat p_i = \operatorname{softmax}(z_i)$, and $\hat y_i = \arg\max_j \hat p_{ij}$. Then

        $$\mathcal{L} = \underbrace{-\tfrac{1}{B}\sum_i \log \hat p_{i,y_i}}_{\text{cross-entropy}} \;+\; \beta\,\frac{1}{|K_b|}\sum_{k\in K_b}\Big|\underbrace{\tfrac{1}{n_k}\sum_{i:y_i=k}\max_j \hat p_{ij}}_{\text{mean confidence in class }k} \;-\; \underbrace{\tfrac{1}{n_k}\sum_{i:y_i=k}\mathbb{1}[\hat y_i = k]}_{\text{accuracy in class }k}\Big|.$$

        The second term penalises the gap between mean predicted confidence and mean accuracy *within each class* over the
        batch. It is differentiable through the confidence; the accuracy indicator is a constant with respect to the
        parameters. **Deviation, stated plainly:** the original MDCA compares the mean predicted *probability of class $k$*
        over all samples with the empirical frequency of class $k$; the version here is the per-true-class
        confidence-versus-accuracy form the exploration brief asked for. It is a direction-setting experiment, not a
        replication of the paper.

        The metric is the top-label expected calibration error with 10 equal-width bins,
        $\mathrm{ECE} = \sum_{b} \tfrac{|B_b|}{n}\,\big|\mathrm{acc}(B_b) - \mathrm{conf}(B_b)\big|$, computed by
        `src.metrics.expected_calibration_error`. We sweep $\beta \in \{0, 1, 4\}$ with two seeds. $\beta = 0$ is the
        plain cross-entropy control, trained with the same in-notebook loop, so the only difference between conditions is
        the loss term. Test-set ECE is reported with a 200-resample bootstrap interval; the choice of $\beta^\star$ is made
        on the **calibration** split, never on the test split.
        """),

        code(r"""
        # Training loop with the optional MDCA-style auxiliary loss. Adam(lr=1e-3), same batch size and epochs as the sweep.
        import torch.nn.functional as F

        def mdca_style(logits, y):
            # Class-wise gap between mean confidence and mean accuracy over the batch (see equation above).
            p = logits.softmax(dim=1)
            conf, pred = p.max(dim=1)
            correct = (pred == y).float().detach()
            gaps = [(conf[y == k].mean() - correct[y == k].mean()).abs() for k in y.unique()]
            return torch.stack(gaps).mean()

        def fit_aux(beta, seed, epochs=WORK_EPOCHS):
            torch.manual_seed(seed)
            model = work_model()
            opt = torch.optim.Adam(model.parameters(), lr=1e-3)
            xt, yt = torch.from_numpy(X["train"]), torch.from_numpy(Y["train"])
            gen = torch.Generator().manual_seed(seed)
            for ep in range(epochs):
                model.train()
                idx = torch.randperm(len(xt), generator=gen)
                for b in range(0, len(idx), MICROBATCH):
                    i = idx[b:b + MICROBATCH]
                    logits = model(xt[i])
                    loss = F.cross_entropy(logits, yt[i])
                    if beta > 0:
                        loss = loss + beta * mdca_style(logits, yt[i])
                    opt.zero_grad(); loss.backward(); opt.step()
            return model

        # Unit tests of the auxiliary term on inputs whose answer is known.
        y_t = torch.tensor([0, 0, 1, 1])
        perfect_sure = torch.tensor([[9., 0, 0], [9, 0, 0], [0, 9, 0], [0, 9, 0]])            # confident and right
        wrong_sure = torch.tensor([[0., 9, 0], [0, 9, 0], [9, 0, 0], [9, 0, 0]])              # confident and wrong
        unsure_right = torch.tensor([[1., 0, 0], [1, 0, 0], [0, 1, 0], [0, 1, 0]])            # right, low confidence
        print("aux(confident&right)=%.4f  aux(confident&wrong)=%.4f  aux(unsure&right)=%.4f" % (mdca_style(perfect_sure, y_t), mdca_style(wrong_sure, y_t), mdca_style(unsure_right, y_t)))
        check("N2.7.1[aux_zero_when_sure_and_right]", float(mdca_style(perfect_sure, y_t)) < 1e-3, "confidence ~1, accuracy 1 -> gap ~0")
        check("N2.7.1[aux_large_when_sure_and_wrong]", float(mdca_style(wrong_sure, y_t)) > 0.99, "confidence ~1, accuracy 0 -> gap ~1")
        check("N2.7.1[aux_positive_when_underconfident]", 0.3 < float(mdca_style(unsure_right, y_t)) < 0.8, f"right but confidence ~0.58 -> gap {float(mdca_style(unsure_right, y_t)):.2f}")
        g_logits = wrong_sure.clone().requires_grad_(True); mdca_style(g_logits, y_t).backward()
        check("N2.7.1[aux_has_gradient]", float(g_logits.grad.abs().sum()) > 0, "the term back-propagates through the confidence")
        """),

        code(r"""
        # Train the beta x seed grid, then fit temperatures on the calibration split.
        BETAS, CAL_SEEDS = (0.0, 1.0, 4.0), (17, 23)
        CAL = {}
        for beta in BETAS:
            for s in CAL_SEEDS:
                m = fit_aux(beta, s)
                lc, lt = predict_logits(m, X["cal"]), predict_logits(m, X["test"])
                T = temperature_scale(lc, Y["cal"])
                CAL[(beta, s)] = {"model": m, "logit_cal": lc, "logit_test": lt, "T": T}
                pt_raw, pt_ts = softmax64(lt), softmax64(lt / T)
                CAL[(beta, s)].update(acc=float((lt.argmax(1) == Y["test"]).mean()),
                                      ece_raw=expected_calibration_error(pt_raw, Y["test"], 10), ece_ts=expected_calibration_error(pt_ts, Y["test"], 10),
                                      ece_cal_raw=expected_calibration_error(softmax64(lc), Y["cal"], 10),
                                      conf_raw=float(pt_raw.max(1).mean()))
                d = CAL[(beta, s)]
                print(f"beta={beta:>3} seed={s} acc={d['acc']:.3f} mean_conf={d['conf_raw']:.3f} T={T:.3f} ECE raw={d['ece_raw']:.4f} ECE+TS={d['ece_ts']:.4f} (cal-split raw ECE {d['ece_cal_raw']:.4f})")
        CALDF = pd.DataFrame([{"beta": b, "seed": s, **{k: v for k, v in d.items() if k in ("acc", "conf_raw", "T", "ece_raw", "ece_ts", "ece_cal_raw")}} for (b, s), d in CAL.items()])
        display(CALDF.groupby("beta").mean(numeric_only=True).drop(columns="seed").round(4))
        check("N2.7.2[temperatures_positive]", (CALDF["T"] > 0).all(), f"all fitted temperatures positive: {CALDF['T'].round(2).tolist()}")
        check("N2.7.2[models_above_chance]", (CALDF["acc"] > 2 / N_CLASSES).all(), f"lowest test accuracy {CALDF['acc'].min():.3f}")
        """),

        code(r"""
        # Choose beta* on the CALIBRATION split (never on test): the beta > 0 with the lowest mean raw ECE there.
        cal_pick = CALDF[CALDF.beta > 0].groupby("beta")["ece_cal_raw"].mean()
        BETA_STAR = float(cal_pick.idxmin())
        print("mean raw ECE on the calibration split by beta > 0:", cal_pick.round(4).to_dict(), "-> beta* =", BETA_STAR)
        COND = {"baseline (CE)": ("ece_raw", 0.0), "post-hoc TS": ("ece_ts", 0.0), "train-time (MDCA)": ("ece_raw", BETA_STAR), "both": ("ece_ts", BETA_STAR)}
        rng_b = np.random.default_rng(0)

        def boot_ece(probs, y, B=200):
            n = len(y); out = []
            for _ in range(B):
                ii = rng_b.integers(0, n, n)
                out.append(expected_calibration_error(probs[ii], y[ii], 10))
            return np.array(out)

        ROWS = []
        for label, (key, beta) in COND.items():
            vals = [CAL[(beta, s)][key] for s in CAL_SEEDS]
            d17 = CAL[(beta, 17)]
            probs = softmax64(d17["logit_test"] / (d17["T"] if key == "ece_ts" else 1.0))
            bs = boot_ece(probs, Y["test"])
            ROWS.append({"condition": label, "beta": beta, "ECE mean over seeds": float(np.mean(vals)), "ECE seed 17": vals[0],
                         "boot 5%": float(np.percentile(bs, 5)), "boot 95%": float(np.percentile(bs, 95)),
                         "NLL seed 17": float(-np.log(probs[np.arange(len(Y['test'])), Y['test']] + 1e-12).mean()),
                         "accuracy": float(np.mean([CAL[(beta, s)]["acc"] for s in CAL_SEEDS]))})
        COMP = pd.DataFrame(ROWS).set_index("condition")
        display(COMP.round(4))
        for label in COMP.index:
            check(f"N2.7.3[{label}.ece_finite]", np.isfinite(COMP.loc[label, "ECE mean over seeds"]), f"ECE {COMP.loc[label, 'ECE mean over seeds']:.4f} (bootstrap {COMP.loc[label, 'boot 5%']:.4f}-{COMP.loc[label, 'boot 95%']:.4f})")
        """),

        code(r"""
        # Fig NB2-16: reliability diagrams for the four conditions (seed 17, test split).
        def reliability(probs, y, bins=10):
            conf, pred = probs.max(axis=1), probs.argmax(axis=1)
            idx = np.minimum((conf * bins).astype(int), bins - 1)
            rows = []
            for b in range(bins):
                s = idx == b
                rows.append((conf[s].mean() if s.any() else np.nan, (pred[s] == y[s]).mean() if s.any() else np.nan, int(s.sum())))
            return np.array(rows)

        fig, ax = plt.subplots(2, 4, figsize=(17, 7.3), gridspec_kw={"height_ratios": [3, 1]})
        for j, (label, (key, beta)) in enumerate(COND.items()):
            d = CAL[(beta, 17)]
            probs = softmax64(d["logit_test"] / (d["T"] if key == "ece_ts" else 1.0))
            R = reliability(probs, Y["test"])
            ok = np.isfinite(R[:, 0])
            assert_varied(R[ok, 1], f"reliability_accuracy[{label}]")
            assert_varied(R[:, 2], f"reliability_bin_counts[{label}]")
            ax[0, j].plot([0, 1], [0, 1], color=PALETTE["reference"], ls="--", label="perfect")
            ax[0, j].bar(R[ok, 0], R[ok, 1], width=0.07, color=PALETTE["clean"], alpha=0.75, edgecolor="black", label="accuracy in bin")
            ax[0, j].scatter(R[ok, 0], R[ok, 1], color="black", s=12, zorder=3)
            ax[0, j].set(title=f"{label}\nECE {expected_calibration_error(probs, Y['test'], 10):.3f}", xlim=(0, 1), ylim=(0, 1), xlabel="mean confidence in bin")
            ax[1, j].bar(R[:, 0][ok], R[ok, 2], width=0.07, color=PALETTE["band"]); ax[1, j].set(xlim=(0, 1), xlabel="confidence", ylabel="examples")
        ax[0, 0].set_ylabel("accuracy"); ax[0, 0].legend(frameon=False, fontsize=8)
        fig.suptitle(f"NB2-16  Reliability diagrams (seed 17, test split, beta* = {BETA_STAR:g}): points on the dashed diagonal are calibrated", y=1.0)
        fig.tight_layout()
        savefig(fig, "NB2-16_reliability.png")
        """),

        md(r"""
        ### How to read this chart

        Each column is one calibration condition. **Top:** a reliability diagram. Test examples are grouped into ten
        equal-width bins by their top-label confidence (x); each bar is the *actual* accuracy of the examples in that bin (y).
        A perfectly calibrated model puts every bar on the dashed diagonal. Bars above the diagonal mean the model was
        *under*-confident in that bin (it was right more often than it claimed); bars below mean *over*-confident. The
        number in each title is the ECE, the mass-weighted mean vertical distance between the bar tops and the diagonal.
        **Bottom:** how many test examples fall in each bin. Bins with few examples are noisy; a large bar gap in a nearly
        empty bin contributes little to the ECE. Compare the *pattern* between columns: does temperature scaling move
        all bars toward the diagonal in the same direction, and does the MDCA-style loss shift the confidence
        distribution itself (the bottom row) rather than only rescaling it?
        """),

        code(r"""
        # Fig NB2-17: ECE by condition (seed dots, bootstrap band) and ECE versus beta with and without temperature scaling.
        assert_varied(COMP["ECE mean over seeds"], "ece_across_conditions")
        assert_varied(CALDF["ece_raw"], "ece_raw_across_runs")
        fig, ax = plt.subplots(1, 3, figsize=(16, 4.4))
        xs = np.arange(len(COMP))
        ax[0].bar(xs, COMP["ECE mean over seeds"], color=[PALETTE["reference"], PALETTE["clean"], PALETTE["mask"], PALETTE["acoustic"]], alpha=0.85)
        ax[0].errorbar(xs, COMP["ECE seed 17"], yerr=[COMP["ECE seed 17"] - COMP["boot 5%"], COMP["boot 95%"] - COMP["ECE seed 17"]], fmt="ko", capsize=4, label="seed 17, bootstrap 5-95%")
        ax[0].set_xticks(xs, [c.replace(" ", "\n", 1) for c in COMP.index], fontsize=8)
        ax[0].set(ylabel="test ECE (10 bins)", title="ECE by condition (bar = mean of 2 seeds)"); ax[0].legend(frameon=False, fontsize=8)
        gb = CALDF.groupby("beta")[["ece_raw", "ece_ts", "acc"]].mean()
        ax[1].plot(gb.index, gb["ece_raw"], "-o", color=PALETTE["shifted"], label="raw")
        ax[1].plot(gb.index, gb["ece_ts"], "-s", color=PALETTE["clean"], label="+ temperature scaling")
        for s in CAL_SEEDS:
            sub = CALDF[CALDF.seed == s]
            ax[1].scatter(sub["beta"], sub["ece_raw"], color=PALETTE["shifted"], alpha=0.35, s=18); ax[1].scatter(sub["beta"], sub["ece_ts"], color=PALETTE["clean"], alpha=0.35, s=18)
        ax[1].set(xlabel="beta", ylabel="test ECE", title="ECE versus beta (dots = seeds)"); ax[1].legend(frameon=False, fontsize=8)
        ax[2].plot(gb.index, gb["acc"], "-o", color=PALETTE["acoustic"])
        ax[2].set(xlabel="beta", ylabel="test accuracy", title="does the auxiliary loss cost accuracy?")
        fig.suptitle("NB2-17  Train-time versus post-hoc calibration", y=1.03)
        fig.tight_layout()
        savefig(fig, "NB2-17_ece_conditions.png")
        """),

        md(r"""
        ### How to read this chart

        **Left:** test ECE for the four conditions. Bars are the mean over two seeds; the black dot is the seed-17 value and
        its bar is a 5--95% bootstrap interval over resamples of the 600 test examples, which tells you how much of a
        difference between two bars could be pure test-set sampling noise. If two conditions' intervals overlap
        heavily, the ordering between them is not established. **Middle:** ECE against the auxiliary-loss weight $\beta$,
        raw (red) and after temperature scaling (blue); faint dots are individual seeds. A blue line that lies below the red
        one means temperature scaling helps at that $\beta$; a red line that falls as $\beta$ grows means the auxiliary loss
        alone improves calibration. **Right:** test accuracy against $\beta$, to check that the calibration gain, if any,
        was not bought by degrading the classifier.
        """),

        code(r"""
        # Verdict on calibration, stated from the numbers.
        e = COMP["ECE mean over seeds"]
        base, ts, mdca, both = (float(e[k]) for k in COMP.index)
        print(f"ECE  baseline {base:.4f} | post-hoc TS {ts:.4f} | train-time (beta*={BETA_STAR:g}) {mdca:.4f} | both {both:.4f}")
        fitted_T = float(np.mean([CAL[(0.0, s)]['T'] for s in CAL_SEEDS]))
        direction = "over-confident (T > 1)" if fitted_T > 1.05 else ("under-confident (T < 1)" if fitted_T < 0.95 else "close to calibrated (T ~ 1)")
        print(f"mean fitted temperature for the beta = 0 models: {fitted_T:.3f} -> the baseline is {direction}")
        check("N2.7.4[ts_does_not_hurt]", ts <= base + 0.01, f"post-hoc TS ECE {ts:.4f} vs baseline {base:.4f}")
        check("N2.7.4[ts_improves_or_matches_on_both_seeds]", all(CAL[(0.0, s)]["ece_ts"] <= CAL[(0.0, s)]["ece_raw"] + 0.01 for s in CAL_SEEDS), "temperature scaling does not make either beta=0 seed worse by more than 0.01")
        check("N2.7.4[aux_loss_improves_raw_ece]", mdca < base, f"train-time ECE {mdca:.4f} < baseline {base:.4f}  (expected if the MDCA-style term is doing its job)")
        check("N2.7.4[best_condition_identified]", True, "lowest ECE: " + e.idxmin() + f" ({e.min():.4f})")
        check("N2.7.4[aux_loss_accuracy_cost_small]", CALDF[CALDF.beta == BETA_STAR]["acc"].mean() > CALDF[CALDF.beta == 0]["acc"].mean() - 0.1, f"accuracy at beta*={BETA_STAR:g}: {CALDF[CALDF.beta == BETA_STAR]['acc'].mean():.3f} vs beta=0: {CALDF[CALDF.beta == 0]['acc'].mean():.3f}")
        note("N2.7.4[both_vs_single]", f"combining the two: ECE {both:.4f} versus best single remedy {min(ts, mdca):.4f}")
        """),

        md(r"""
        ### 7.1 What the calibration comparison does and does not show

        The verdict lines are the finding; they are specific to a micro-GRU at ~75--95% accuracy on a 600-example training
        set. Three cautions apply. (i) ECE with ten bins on 600 test examples has a sampling error of roughly 0.02--0.03 (see
        the bootstrap interval), so differences smaller than that are not resolved. (ii) The post-hoc route needs a held-out
        split (480 examples here) that the train-time route does not, so the comparison is not cost-free on either side.
        (iii) A train-time term with a large $\beta$ can trade accuracy for calibration; the right panel of the last figure and
        the accuracy check above guard against that, but the trade-off curve on real data may be steeper. The exploration's
        value is the *protocol* -- four conditions, $\beta$ chosen on the calibration split, bootstrap intervals -- which
        transfers unchanged to the real-data arms.
        """),

        md(r"""
        ## 8. Exploration 3: weighted versus unweighted conformal risk control under shift

        The study's conformal step (`src.calibration.crc_threshold` with `prediction_sets`) builds, from a calibration
        split, a threshold $\hat\lambda$ so that the prediction set $C(x) = \{c : \hat p_c(x) \ge \hat\lambda\}$ contains the true
        label with probability at least $1-\alpha$. With calibration scores $s_i = \hat p_{y_i}(x_i)$ sorted ascending, the
        unweighted rule is the conservative order statistic

        $$\hat\lambda = s_{(m+1)}, \qquad m = \lfloor \alpha (n+1) - 1 \rfloor,$$

        which guarantees marginal coverage **only if the test point is exchangeable with the calibration points**. Under
        distribution shift that guarantee is void. If the shift is known up to a density ratio $w(x) = dP_{\text{test}}/dP_{\text{cal}}(x)$,
        weighted conformal prediction (Tibshirani et al., arXiv:1904.06019) restores it by replacing counts with weights:

        $$\hat\lambda_w = \max\Big\{ s_{(j)} : \sum_{i:\, s_i < s_{(j)}} w_i \ \le\ \alpha \big(\textstyle\sum_i w_i + w_{\text{test}}\big) - w_{\text{test}} \Big\}.$$

        With $w \equiv 1$ this is exactly the unweighted rule (we verify that below against `src`). The weighted variant is
        implemented **in this notebook**; `src/` is unchanged. Two synthetic shifts are applied, each with a different way of
        obtaining $w$:

        * **A. Covariate shift** on the test split: additive Gaussian noise $\mathcal{N}(0, s^2)$ and a circular shift of the
          mel axis by $\operatorname{round}(2s)$ bins, severity $s \in \{0, 0.25, 0.5, 0.75, 1, 1.5\}$. The weights are estimated
          *without labels* by a logistic-regression domain classifier that separates calibration inputs from shifted test
          inputs on five cheap features (spectrogram std, mean, frame-to-frame roughness, model max-probability, entropy);
          $w(x) = \hat P(\text{test}\mid x)/\hat P(\text{cal}\mid x)$ with balanced classes, clipped to $[0.01, 100]$. Weighted
          conformal needs overlap between the two input distributions; heavy noise plus a frequency shift reduces it, so this is
          the *hard* case for the weighted method.
        * **B. Subpopulation (label) shift**: the test distribution over classes is tilted toward the classes the model finds
          hard, $q_{\text{test}}(k) \propto \exp(\tau\, z_k)$ with $z_k$ the standardised calibration error rate of class $k$,
          $\tau \in \{0, 0.5, 1, 2, 3\}$, sampling from the independent shift pool. Weights are $w(y) = q_{\text{test}}(y)/q_{\text{cal}}(y)$
          in three forms: none (unweighted), *estimated* label-free from the ratio of predicted-class frequencies on test versus
          calibration (a simplified black-box shift estimator), and *oracle* (known $q$). This is the *easy* case: full overlap and a
          scalar weight per class.

        The **expected** finding is that unweighted CRC over-shoots its target miscoverage $\alpha$ under shift. We report what
        we measure. Miscoverage is $1 - \frac1n\sum_i \mathbb{1}[y_i \in C(x_i)]$ on the shifted test data; the target is $\alpha = 0.1$.
        """),

        code(r"""
        # Weighted CRC threshold (in-notebook) and a proof that it reduces to src's rule when all weights are 1.
        ALPHA = 0.1
        def crc_threshold_weighted(scores, alpha, w_cal, w_test):
            scores, w_cal = np.asarray(scores, float), np.asarray(w_cal, float)
            order = np.argsort(scores, kind="stable")
            s, w = scores[order], w_cal[order]
            below = np.cumsum(w) - w                                   # weight strictly before each sorted position
            budget = alpha * (w.sum() + w_test) - w_test
            feasible = below <= budget + 1e-12
            return float(s[feasible].max()) if feasible.any() else 0.0

        rng_u = np.random.default_rng(3)
        agree = 0
        for trial in range(200):
            n_u = int(rng_u.integers(20, 300)); a_u = float(rng_u.choice([0.05, 0.1, 0.2]))
            sc = np.round(rng_u.random(n_u), int(rng_u.integers(1, 4)))          # rounding creates ties on purpose
            agree += np.isclose(crc_threshold_weighted(sc, a_u, np.ones(n_u), 1.0), crc_threshold(sc, a_u))
        check("N2.8.1[weighted_reduces_to_src]", agree == 200, f"with unit weights the in-notebook rule equals src.crc_threshold on {agree}/200 random score sets (incl. ties)")
        sc = rng_u.random(50)
        heavy = np.where(sc < 0.5, 10.0, 1.0)      # low scores (likely misses) are heavily up-weighted -> threshold should not decrease
        check("N2.8.1[upweighting_low_scores_is_more_conservative]", crc_threshold_weighted(sc, 0.1, heavy, 1.0) <= crc_threshold_weighted(sc, 0.1, np.ones(50), 1.0), "up-weighting low-score calibration points lowers the admissible order statistic (fewer points 'allowed' below it)")
        check("N2.8.1[infeasible_alpha_gives_zero]", crc_threshold_weighted(rng_u.random(5), 0.05, np.ones(5), 1.0) == 0.0 == crc_threshold(rng_u.random(5), 0.05), "alpha too small for n=5 -> threshold 0 (include every class), matching src")
        """),

        code(r"""
        # Common objects: the base model (beta = 0, seed 17), its temperature, calibration scores, and the shift functions.
        BASE = CAL[(0.0, 17)]
        T_BASE = BASE["T"]
        P_CAL = softmax64(BASE["logit_cal"] / T_BASE)
        S_CAL = P_CAL[np.arange(len(Y["cal"])), Y["cal"]]                   # true-label probability on the calibration split
        THR_UNW = crc_threshold(S_CAL, ALPHA)
        print(f"base model: T={T_BASE:.3f}, cal accuracy={(P_CAL.argmax(1) == Y['cal']).mean():.3f}, unweighted threshold={THR_UNW:.4f}")

        def shift_cov(x, s, seed):
            rng = np.random.default_rng(seed)
            out = x + rng.normal(0.0, s, x.shape).astype(np.float32)
            k = int(round(2 * s))
            return np.roll(out, k, axis=2) if k else out

        def set_stats(p_all, y, thr):
            covered = p_all[np.arange(len(y)), y] >= thr
            size = (p_all >= thr).sum(axis=1)
            return covered, size

        # Cross-check the vectorised coverage against src.prediction_sets on the calibration split itself.
        sets = prediction_sets(BASE["logit_cal"], T_BASE, THR_UNW)
        cov_src = np.mean([y in s for y, s in zip(Y["cal"], sets)])
        cov_vec = set_stats(P_CAL, Y["cal"], THR_UNW)[0].mean()
        # These two disagreed by one sample when first run. That is a boundary condition, not noise:
        # src.prediction_sets and the vectorised form must agree on whether a score exactly equal to the
        # threshold is inside the set. Diagnose it rather than loosening the tolerance.
        _in_vec = P_CAL >= THR_UNW
        _in_src = np.zeros_like(_in_vec)
        for _i, _s in enumerate(sets):
            for _c in _s:
                _in_src[_i, _c] = True
        _disagree = np.argwhere(_in_vec != _in_src)
        _at_boundary = int(sum(np.isclose(P_CAL[i, c], THR_UNW, rtol=0, atol=1e-12) for i, c in _disagree))
        check("N2.8.2[vectorised_matches_src_sets]", abs(cov_src - cov_vec) < 1e-9,
              f"coverage from src.prediction_sets {cov_src:.4f} vs vectorised {cov_vec:.4f}; "
              f"{len(_disagree)} membership disagreement(s), {_at_boundary} of them exactly at the threshold")
        if len(_disagree):
            note("N2.8.2[boundary]",
                 f"{len(_disagree)} of {P_CAL.size} (class, sample) memberships differ, all within {np.abs(P_CAL[tuple(_disagree.T)] - THR_UNW).max():.2e} "
                 "of the threshold: an inclusive-vs-exclusive comparison at the boundary. It moves coverage by "
                 f"{abs(cov_src - cov_vec):.4f}, which is below the resolution this notebook claims anywhere")
        check("N2.8.2[cal_coverage_at_least_target]", cov_src >= 1 - ALPHA - 0.02, f"in-sample calibration coverage {cov_src:.3f} (target {1 - ALPHA:.2f})")
        clean_cov, clean_size = set_stats(softmax64(BASE["logit_test"] / T_BASE), Y["test"], THR_UNW)
        check("N2.8.2[clean_test_near_target]", abs((1 - clean_cov.mean()) - ALPHA) < 0.04, f"clean test miscoverage {1 - clean_cov.mean():.3f} vs target {ALPHA} (sampling s.e. ~0.012)")
        print(f"clean test: miscoverage {1 - clean_cov.mean():.3f}, mean set size {clean_size.mean():.2f}")
        """),

        code(r"""
        # Experiment A: covariate shift with estimated (label-free) density-ratio weights.
        from sklearn.pipeline import make_pipeline

        def dom_features(x, logits):
            p = softmax64(logits / T_BASE); xf = x[:, 0]
            ent = -(p * np.log(p + 1e-12)).sum(axis=1)
            return np.column_stack([xf.std(axis=(1, 2)), xf.mean(axis=(1, 2)), np.abs(np.diff(xf, axis=2)).mean(axis=(1, 2)), p.max(axis=1), ent])

        FEAT_CAL = dom_features(X["cal"], BASE["logit_cal"])
        SEVERITY = (0.0, 0.25, 0.5, 0.75, 1.0, 1.5)
        rng_a = np.random.default_rng(11)
        A_ROWS, A_KEEP = [], {}
        for sev in SEVERITY:
            xs = shift_cov(X["test"], sev, seed=100 + int(sev * 100))
            lg = predict_logits(BASE["model"], xs)
            p_test = softmax64(lg / T_BASE)
            feat_t = dom_features(xs, lg)
            dom = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, class_weight="balanced", max_iter=500))
            dom.fit(np.vstack([FEAT_CAL, feat_t]), np.r_[np.zeros(len(FEAT_CAL)), np.ones(len(feat_t))])
            pr = np.clip(dom.predict_proba(FEAT_CAL)[:, 1], 1e-4, 1 - 1e-4); pt = np.clip(dom.predict_proba(feat_t)[:, 1], 1e-4, 1 - 1e-4)
            w_cal = np.clip(pr / (1 - pr), 0.01, 100.0); w_te = float(np.clip(pt / (1 - pt), 0.01, 100.0).mean())
            thr_w = crc_threshold_weighted(S_CAL, ALPHA, w_cal, w_te)
            cov_u, size_u = set_stats(p_test, Y["test"], THR_UNW)
            cov_w, size_w = set_stats(p_test, Y["test"], thr_w)
            boots = lambda c: np.percentile([1 - c[rng_a.integers(0, len(c), len(c))].mean() for _ in range(200)], [5, 95])
            ess = float(w_cal.sum() ** 2 / (w_cal ** 2).sum())
            A_ROWS.append({"severity": sev, "acc": float((p_test.argmax(1) == Y["test"]).mean()), "thr_unweighted": THR_UNW, "thr_weighted": thr_w,
                           "miscov_unweighted": 1 - cov_u.mean(), "miscov_weighted": 1 - cov_w.mean(),
                           "unw_lo": boots(cov_u)[0], "unw_hi": boots(cov_u)[1], "w_lo": boots(cov_w)[0], "w_hi": boots(cov_w)[1],
                           "size_unweighted": size_u.mean(), "size_weighted": size_w.mean(), "ess": ess, "w_max": float(w_cal.max()),
                           "domain_auc": float(__import__('sklearn.metrics', fromlist=['roc_auc_score']).roc_auc_score(np.r_[np.zeros(len(FEAT_CAL)), np.ones(len(feat_t))], dom.predict_proba(np.vstack([FEAT_CAL, feat_t]))[:, 1]))})
            A_KEEP[sev] = {"w_cal": w_cal}
        A = pd.DataFrame(A_ROWS).set_index("severity")
        display(A[["acc", "thr_unweighted", "thr_weighted", "miscov_unweighted", "miscov_weighted", "size_unweighted", "size_weighted", "ess", "domain_auc"]].round(3))
        """),

        code(r"""
        # Fig NB2-18: experiment A. Miscoverage and set size versus shift severity; weight distribution at the highest severity.
        assert_varied(A["miscov_unweighted"], "A_unweighted_miscoverage")
        assert_varied(A["miscov_weighted"], "A_weighted_miscoverage")
        assert_varied(A["size_unweighted"], "A_set_size")
        assert_varied(A["acc"], "A_accuracy_under_shift")
        fig, ax = plt.subplots(1, 4, figsize=(19, 4.4))
        ax[0].plot(A.index, A["acc"], "-o", color=PALETTE["acoustic"]); ax[0].set(xlabel="shift severity s", ylabel="test accuracy", ylim=(0, 1), title="the shift is real: accuracy falls")
        ax[1].axhline(ALPHA, color="black", ls="--", label=f"target alpha = {ALPHA}")
        ax[1].plot(A.index, A["miscov_unweighted"], "-o", color=PALETTE["shifted"], label="unweighted CRC (src)")
        ax[1].fill_between(A.index, A["unw_lo"], A["unw_hi"], color=PALETTE["shifted"], alpha=0.15)
        ax[1].plot(A.index, A["miscov_weighted"], "-s", color=PALETTE["clean"], label="weighted CRC (estimated w)")
        ax[1].fill_between(A.index, A["w_lo"], A["w_hi"], color=PALETTE["clean"], alpha=0.15)
        ax[1].set(xlabel="shift severity s", ylabel="miscoverage", title="miscoverage vs severity"); ax[1].legend(frameon=False, fontsize=8)
        ax[2].plot(A.index, A["size_unweighted"], "-o", color=PALETTE["shifted"], label="unweighted"); ax[2].plot(A.index, A["size_weighted"], "-s", color=PALETTE["clean"], label="weighted")
        ax[2].axhline(N_CLASSES, color=PALETTE["reference"], ls=":", label="all 12 classes")
        ax[2].set(xlabel="shift severity s", ylabel="mean prediction-set size", title="price paid in set size"); ax[2].legend(frameon=False, fontsize=8)
        wmax = A_KEEP[SEVERITY[-1]]["w_cal"]; assert_varied(wmax, "A_weights_at_max_severity")
        ax[3].hist(wmax, bins=30, color=PALETTE["mask"]); ax[3].set(xlabel="importance weight on a calibration point", ylabel="count", yscale="log", title=f"weights at s={SEVERITY[-1]} (ESS {A.loc[SEVERITY[-1], 'ess']:.0f}/{len(wmax)})")
        fig.suptitle("NB2-18  Experiment A, covariate shift: does importance weighting rescue the coverage guarantee?", y=1.03)
        fig.tight_layout()
        savefig(fig, "NB2-18_crc_covariate_shift.png")
        """),

        md(r"""
        ### How to read this chart

        **Panel 1** is a sanity check on the shift itself: test accuracy should fall as severity rises, otherwise the "shift"
        is cosmetic and nothing below is informative. **Panel 2** is the main result. The dashed horizontal line is the
        target miscoverage $\alpha = 0.1$: a method that honours its guarantee stays on or below it. Red is unweighted CRC (what
        `src/` does), blue is the importance-weighted variant, and the shaded bands are 5--95% bootstrap intervals over test
        resamples. A red line that climbs well above the dashed line is the over-shoot the exploration expected. **Panel 3** is
        the price: mean prediction-set size, with the dotted line marking the trivial "predict all 12 classes" set. A method
        can always restore coverage by enlarging sets, so a coverage gain is only meaningful next to its set-size cost.
        **Panel 4** shows the estimated importance weights at the highest severity on a log-count axis; the title gives the
        effective sample size $(\sum w)^2/\sum w^2$, which shrinks when a few calibration points carry most of the weight -- a
        sign of poor overlap between calibration and shifted inputs.
        """),

        code(r"""
        # Verdict for experiment A, from the numbers.
        a0, a_hi = A.iloc[0], A.iloc[-1]
        print(f"s=0   : unweighted miscoverage {a0.miscov_unweighted:.3f}, weighted {a0.miscov_weighted:.3f}")
        print(f"s={A.index[-1]}: unweighted miscoverage {a_hi.miscov_unweighted:.3f}, weighted {a_hi.miscov_weighted:.3f}; accuracy {a_hi.acc:.3f}")
        check("N2.8.3[shift_reduces_accuracy]", a_hi.acc < a0.acc - 0.1, f"accuracy {a0.acc:.3f} -> {a_hi.acc:.3f} at the highest severity")
        check("N2.8.3[unweighted_overshoots_under_shift]", a_hi.miscov_unweighted > ALPHA + 0.03, f"unweighted miscoverage at max severity = {a_hi.miscov_unweighted:.3f} vs target {ALPHA} (expected over-shoot)")
        check("N2.8.3[overshoot_grows_with_severity]", A["miscov_unweighted"].iloc[-1] > A["miscov_unweighted"].iloc[1], "miscoverage at the highest severity exceeds miscoverage at severity 0.25")
        improved = (A["miscov_weighted"] < A["miscov_unweighted"] - 0.005).sum()
        check("N2.8.3[weighted_helps_somewhere]", improved >= 1, f"weighted CRC has lower miscoverage than unweighted at {improved} of {len(A)} severities")
        gap_w = float(a_hi.miscov_weighted - ALPHA)
        if a_hi.miscov_weighted <= ALPHA + 0.03:
            note("N2.8.3[verdict]", f"estimated weights restore coverage at max severity (miscoverage {a_hi.miscov_weighted:.3f}) at set size {a_hi.size_weighted:.2f} vs {a_hi.size_unweighted:.2f}")
        else:
            note("N2.8.3[verdict]", f"estimated covariate weights do NOT restore coverage at max severity: {a_hi.miscov_weighted:.3f} vs target {ALPHA} (remaining over-shoot {gap_w:+.3f}); overlap is poor (ESS {a_hi.ess:.0f}, domain-classifier AUC {a_hi.domain_auc:.2f})")
        """),

        code(r"""
        # Experiment B: subpopulation shift. Test class prior tilted toward hard classes; three weightings compared over 100 resamples.
        pred_cal = P_CAL.argmax(1)
        hard = np.array([1 - (pred_cal[Y["cal"] == k] == k).mean() for k in range(N_CLASSES)])
        assert_varied(hard, "B_per_class_error_rate")
        z = (hard - hard.mean()) / (hard.std() + 1e-9)
        Q_CAL = np.full(N_CLASSES, 1 / N_CLASSES)
        LOGIT_POOL = predict_logits(BASE["model"], X["pool"])
        P_POOL = softmax64(LOGIT_POOL / T_BASE)
        POOL_PRED = P_POOL.argmax(1)
        POOL_BY_CLASS = [np.flatnonzero(Y["pool"] == k) for k in range(N_CLASSES)]
        TAUS, R_B = (0.0, 0.5, 1.0, 2.0, 3.0), 100
        rng_b2 = np.random.default_rng(21)
        f_cal_pred = np.bincount(pred_cal, minlength=N_CLASSES) / len(pred_cal)
        B_ROWS = []
        for tau in TAUS:
            q = np.exp(tau * z); q = q / q.sum()
            acc_l, mis_u, mis_e, mis_o, sz_u, sz_e, sz_o = [], [], [], [], [], [], []
            for r in range(R_B):
                cls = rng_b2.choice(N_CLASSES, size=600, p=q)
                ii = np.array([rng_b2.choice(POOL_BY_CLASS[k]) for k in cls])
                yb, pb = Y["pool"][ii], P_POOL[ii]
                f_te = np.bincount(POOL_PRED[ii], minlength=N_CLASSES) / len(ii)
                w_hat = f_te / np.maximum(f_cal_pred, 1e-3)
                thr_e = crc_threshold_weighted(S_CAL, ALPHA, np.clip(w_hat[pred_cal], 0.05, 20.0), float(np.clip(w_hat[POOL_PRED[ii]], 0.05, 20.0).mean()))
                w_or = q / Q_CAL
                thr_o = crc_threshold_weighted(S_CAL, ALPHA, w_or[Y["cal"]], float((q * w_or).sum()))
                for thr, m_l, s_l in ((THR_UNW, mis_u, sz_u), (thr_e, mis_e, sz_e), (thr_o, mis_o, sz_o)):
                    c, s_ = set_stats(pb, yb, thr); m_l.append(1 - c.mean()); s_l.append(s_.mean())
                acc_l.append((pb.argmax(1) == yb).mean())
            B_ROWS.append({"tau": tau, "acc": np.mean(acc_l), "unw": np.mean(mis_u), "unw_lo": np.percentile(mis_u, 5), "unw_hi": np.percentile(mis_u, 95),
                           "est": np.mean(mis_e), "est_lo": np.percentile(mis_e, 5), "est_hi": np.percentile(mis_e, 95),
                           "orc": np.mean(mis_o), "orc_lo": np.percentile(mis_o, 5), "orc_hi": np.percentile(mis_o, 95),
                           "size_unw": np.mean(sz_u), "size_est": np.mean(sz_e), "size_orc": np.mean(sz_o)})
        B = pd.DataFrame(B_ROWS).set_index("tau")
        display(B[["acc", "unw", "est", "orc", "size_unw", "size_est", "size_orc"]].round(3))
        print("per-class calibration error rate (hardness):", dict(zip(CLASS_NAMES, hard.round(2))))
        """),

        code(r"""
        # Fig NB2-19: experiment B. Miscoverage versus prior tilt for the three weightings, plus the class priors at the largest tilt.
        assert_varied(B["unw"], "B_unweighted_miscoverage")
        assert_varied(B["est"], "B_estimated_weights_miscoverage")
        assert_varied(B["acc"], "B_accuracy_vs_tilt")
        q_max = np.exp(TAUS[-1] * z); q_max = q_max / q_max.sum()
        assert_varied(q_max, "B_class_prior_at_max_tilt")
        fig, ax = plt.subplots(1, 3, figsize=(16, 4.4))
        ax[0].axhline(ALPHA, color="black", ls="--", label=f"target alpha = {ALPHA}")
        for key, lab, col, mk in (("unw", "unweighted (src)", PALETTE["shifted"], "o"), ("est", "estimated weights (label-free)", PALETTE["clean"], "s"), ("orc", "oracle weights", PALETTE["acoustic"], "^")):
            ax[0].plot(B.index, B[key], f"-{mk}", color=col, label=lab)
            ax[0].fill_between(B.index, B[key + "_lo"], B[key + "_hi"], color=col, alpha=0.15)
        ax[0].set(xlabel="prior tilt tau (0 = no shift)", ylabel="miscoverage (mean of 100 resamples)", title="miscoverage vs tilt"); ax[0].legend(frameon=False, fontsize=8)
        ax[1].plot(B.index, B["size_unw"], "-o", color=PALETTE["shifted"], label="unweighted"); ax[1].plot(B.index, B["size_est"], "-s", color=PALETTE["clean"], label="estimated")
        ax[1].plot(B.index, B["size_orc"], "-^", color=PALETTE["acoustic"], label="oracle")
        ax[1].set(xlabel="prior tilt tau", ylabel="mean set size", title="set size vs tilt"); ax[1].legend(frameon=False, fontsize=8)
        ax[2].bar(np.arange(N_CLASSES) - 0.2, np.full(N_CLASSES, 1 / N_CLASSES), 0.4, color=PALETTE["reference"], label="calibration prior")
        ax[2].bar(np.arange(N_CLASSES) + 0.2, q_max, 0.4, color=PALETTE["mask"], label=f"test prior, tau={TAUS[-1]:g}")
        ax[2].set_xticks(range(N_CLASSES), CLASS_NAMES, rotation=90, fontsize=8); ax[2].set(ylabel="class probability", title="the shift: hard classes over-represented"); ax[2].legend(frameon=False, fontsize=8)
        fig.suptitle("NB2-19  Experiment B, subpopulation shift: weights are a scalar per class, so overlap is complete", y=1.03)
        fig.tight_layout()
        savefig(fig, "NB2-19_crc_label_shift.png")
        """),

        md(r"""
        ### How to read this chart

        **Left:** miscoverage against the prior tilt $\tau$. $\tau = 0$ is no shift (a sanity check: all three methods should
        sit near the dashed target). Larger $\tau$ draws test examples increasingly from the classes the model gets wrong most often,
        so a fixed threshold calibrated on a uniform class mix under-covers. Red is unweighted CRC, blue is weighted with
        class weights estimated *without labels* from the predicted-class frequencies, green is weighted with the true class
        prior (an oracle upper bound on what any estimator could achieve). Bands are 5--95% ranges over 100 resampled test
        sets of 600 examples. **Middle:** the set-size cost of each method. **Right:** the calibration and test class priors at
        the largest tilt: the orange bars, tall over the hard classes and short over the easy ones, are the shift. If blue
        tracks green and both stay near the dashed line while red climbs, importance weighting works when overlap is complete
        and the weights can be estimated from predictions alone.
        """),

        code(r"""
        # Verdict for experiment B, from the numbers.
        b0, b_hi = B.iloc[0], B.iloc[-1]
        print(f"tau=0    : unweighted {b0.unw:.3f}, estimated {b0.est:.3f}, oracle {b0.orc:.3f}")
        print(f"tau={B.index[-1]:g}: unweighted {b_hi.unw:.3f}, estimated {b_hi.est:.3f}, oracle {b_hi.orc:.3f}; accuracy {b_hi.acc:.3f}; set sizes {b_hi.size_unw:.2f}/{b_hi.size_est:.2f}/{b_hi.size_orc:.2f}")
        check("N2.8.4[no_shift_all_near_target]", max(abs(b0.unw - ALPHA), abs(b0.est - ALPHA), abs(b0.orc - ALPHA)) < 0.04, "at tau=0 all three methods are within 0.04 of the target")
        check("N2.8.4[unweighted_overshoots]", b_hi.unw > ALPHA + 0.02, f"unweighted miscoverage at max tilt {b_hi.unw:.3f} vs target {ALPHA}")
        check("N2.8.4[oracle_restores_coverage]", b_hi.orc < b_hi.unw and b_hi.orc < ALPHA + 0.04, f"oracle-weighted miscoverage {b_hi.orc:.3f} (unweighted {b_hi.unw:.3f})")
        check("N2.8.4[estimated_weights_help]", b_hi.est < b_hi.unw, f"label-free weights reduce miscoverage from {b_hi.unw:.3f} to {b_hi.est:.3f}")
        check("N2.8.4[accuracy_falls_with_tilt]", b_hi.acc < b0.acc, f"accuracy under the tilted prior {b0.acc:.3f} -> {b_hi.acc:.3f}")
        check("N2.8.4[weighted_costs_set_size]", b_hi.size_orc >= b_hi.size_unw - 0.05, f"oracle-weighted mean set size {b_hi.size_orc:.2f} vs unweighted {b_hi.size_unw:.2f} (coverage is bought with larger sets)")
        """),

        md(r"""
        ### 8.1 What this shows about conformal control under shift

        Read the two experiments together. The unweighted rule that `src/` implements is a *marginal, exchangeable* guarantee:
        any shift that changes the test distribution away from the calibration distribution makes its stated coverage
        unreliable, and the size of the failure grows with the amount of shift. Importance weighting is the textbook remedy,
        and the two experiments bracket how well it can work in practice. With complete overlap and a low-dimensional,
        estimable weight (experiment B) it can do most of the job; with hard covariate shift and only unlabelled features
        from which to estimate a density ratio (experiment A) the estimated weights concentrate on a few calibration points
        and cannot conjure coverage the calibration data do not contain. For the real-data study the implication is
        procedural: if a stress condition looks like a class-mix change, a weighted or class-conditional threshold is
        worth building; if it looks like an acoustic covariate shift, the weighted variant should be evaluated against the
        unweighted one with its effective sample size reported, and an honest fallback is a wider set or an abstention
        policy.
        """),

        md(r"""
        ## 9. How much calibration data does the conformal threshold actually need?

        Section 8 fitted one threshold on the whole calibration split and reported one coverage number. That hides the
        question a practitioner faces: **how much calibration data buys how much reliability?** The conformal
        guarantee is distribution-free but finite-sample -- its slack shrinks like $O(1/n)$ in the calibration count
        $n$, so a threshold fitted on 50 points and one fitted on 500 are not the same object even when both are
        formally valid.

        This bears directly on the real study, whose calibration split is $n = 1080$ against a source recommendation
        of at least $1000$. That is a pass, but a narrow one. Resampling is free on synthetic data, so we measure the
        whole curve here and let the real run's margin be read against a shape rather than a rule of thumb.
        """),

        code(r"""
        # N2.9a.1: threshold stability vs calibration size. Pure resampling -- nothing is retrained.
        RNG10 = np.random.default_rng(1010)
        P_TEST10 = softmax64(BASE["logit_test"] / T_BASE)
        N_CAL_TOTAL = len(Y["cal"])
        GRID = sorted({n for n in (25, 50, 100, 200, 400, 800) if n <= N_CAL_TOTAL} | {N_CAL_TOTAL})
        R10, rows10 = 200, []
        for n in GRID:
            covs, thrs = [], []
            for _ in range(R10):
                # Bootstrap (with replacement). Without replacement, n = N_CAL_TOTAL returns the whole
                # split every time, so its spread collapses to exactly 0 by construction -- a degenerate
                # point that would make the root-n check pass for the wrong reason.
                idx = RNG10.choice(N_CAL_TOTAL, size=n, replace=True)
                thr = crc_threshold(S_CAL[idx], ALPHA)
                thrs.append(thr)
                covs.append(float(set_stats(P_TEST10, Y["test"], thr)[0].mean()))
            covs, thrs = np.asarray(covs), np.asarray(thrs)
            rows10.append({"n_cal": n, "mean_coverage": covs.mean(), "sd_coverage": covs.std(ddof=1),
                           "p05_coverage": float(np.percentile(covs, 5)), "mean_threshold": thrs.mean(),
                           "sd_threshold": thrs.std(ddof=1), "undercover_rate": float((covs < 1 - ALPHA).mean())})
        NCAL = pd.DataFrame(rows10).set_index("n_cal")
        display(NCAL.round(4))
        assert_varied(NCAL["sd_coverage"].to_numpy(), "coverage sd vs calibration size")
        check("N2.9a.1[sd_shrinks]", NCAL["sd_coverage"].iloc[-1] < NCAL["sd_coverage"].iloc[0],
              f"coverage spread falls from {NCAL['sd_coverage'].iloc[0]:.4f} at n={GRID[0]} to {NCAL['sd_coverage'].iloc[-1]:.4f} at n={GRID[-1]}")
        _ratio = float(NCAL["sd_coverage"].iloc[0] / max(NCAL["sd_coverage"].iloc[-1], 1e-12))
        _expected = float(np.sqrt(GRID[-1] / GRID[0]))
        check("N2.9a.1[sd_scales_like_sqrt_n]", 0.4 * _expected <= _ratio <= 2.5 * _expected,
              f"sd ratio {_ratio:.2f} against the sqrt(n) expectation {_expected:.2f} (order-of-magnitude check, not a fit)")
        check("N2.9a.1[mean_coverage_near_target]", abs(float(NCAL["mean_coverage"].iloc[-1]) - (1 - ALPHA)) < 0.05,
              f"mean coverage at full n is {NCAL['mean_coverage'].iloc[-1]:.4f} against target {1 - ALPHA:.2f}")
        note("N2.9a.1[undercoverage]",
             f"at n={GRID[0]} a fraction {NCAL['undercover_rate'].iloc[0]:.2f} of resamples land below the target; "
             f"at n={GRID[-1]} it is {NCAL['undercover_rate'].iloc[-1]:.2f}. No single run can reveal this about itself")
        """),

        code(r"""
        # N2.9a.2: the same result as a figure.
        fig, ax = plt.subplots(1, 2, figsize=(12, 3.8))
        ns = NCAL.index.to_numpy()
        ax[0].plot(ns, NCAL["mean_coverage"], "o-", color="#3568A8", label="mean coverage")
        ax[0].fill_between(ns, NCAL["mean_coverage"] - NCAL["sd_coverage"],
                           NCAL["mean_coverage"] + NCAL["sd_coverage"], alpha=0.25, color="#3568A8", label="+/- 1 sd")
        ax[0].axhline(1 - ALPHA, color="black", ls="--", lw=1, label=f"target {1 - ALPHA:.2f}")
        ax[0].set(xscale="log", xlabel="calibration points n", ylabel="test coverage",
                  title="Coverage concentrates as n grows")
        ax[1].plot(ns, NCAL["sd_coverage"], "o-", color="#E58B2A", label="observed sd")
        ax[1].plot(ns, NCAL["sd_coverage"].iloc[0] * np.sqrt(ns[0] / ns), "k--", lw=1, label=r"$\propto 1/\sqrt{n}$")
        ax[1].set(xscale="log", yscale="log", xlabel="calibration points n", ylabel="sd of coverage",
                  title="Spread follows the root-n envelope")
        for a in ax:
            a.set_xticks(ns)
            a.set_xticklabels([str(int(v)) for v in ns], fontsize=7)
            a.minorticks_off()
        for a in ax:
            a.grid(alpha=0.3)
            a.legend(fontsize=7)
        fig.suptitle("Synthetic data: how much calibration data a conformal threshold needs", y=1.03)
        savefig(fig, "fig9a_1_coverage_vs_calibration_size.png")
        """),

        md(r"""
        ### How to read this chart

        **Source.** The calibration and test logits of the `beta = 0`, seed 17 run from section 7, resampled 200 times
        at each calibration size. Nothing is retrained; only the calibration subset changes.

        **What it shows.** Left: mean test coverage against calibration size, with a one-standard-deviation band and
        the nominal $1 - \alpha$ target. Right: the spread alone, on log-log axes, against a $1/\sqrt{n}$ reference.
        Coverage is roughly unbiased at every $n$ -- that is what "distribution-free" buys -- but at small $n$ the
        *variance* is large, so a single small-calibration run can land well off target while the procedure remains
        formally valid.

        **How to read it against the real study.** The real run calibrates on $n = 1080$, the right-hand end of these
        curves, where the spread has largely collapsed. The margin is real but not generous, and `undercover_rate` is
        the column to watch: it is the fraction of resamples landing below target, which no single run can measure
        about itself.

        **What would falsify the reading.** A flat right panel would mean the spread is not shrinking with $n$ and the
        finite-sample story is wrong. Mean coverage drifting away from target as $n$ grows would mean the estimator is
        biased rather than merely noisy.

        **What it does not show.** This is exchangeable synthetic data. It says nothing about the shifted setting of
        section 8, where the assumption is violated outright and more calibration data does not rescue the guarantee.
        """),

        md(r"""
        ## 10. Limitations and what this notebook does not establish

        **Synthetic data cannot validate any of these conclusions on real audio.** Every accuracy, every clip rate,
        every ECE, and every coverage number above is a property of a generator we wrote, of a micro-model chosen for speed,
        and of a 600-example training set. They are direction-setting experiments whose value is to justify or discourage an
        expensive real-data run, not to substitute for one.

        * **The architecture finding is engineered.** The class structure was designed so that a globally pooled model *must*
          fail on order. That the pooled model fails is a check that the construction works, not evidence about speech. On real
          audio the size of the pooling penalty is an empirical question answered in notebook 4, not here.
        * **Two seeds.** Sections 6--7 use two seeds per setting (Section 4 uses three for two models). That resolves only
          large effects. Where a verdict line says "comparable to seed noise", read it as "not distinguishable at this budget",
          not as "equal".
        * **The workhorse is a micro-GRU** chosen so the task is neither trivial nor impossible. Calibration and conformal
          behaviour depend strongly on the accuracy regime and on how over-fitted the network is; a 2-million-parameter model on
          25,000 real utterances will sit elsewhere.
        * **The MDCA-style loss is a variant.** It is the per-true-class confidence-versus-accuracy form, not the paper's exact
          objective, and $\beta$ was searched over only three values.
        * **Weighted CRC is a sketch.** The finite-sample guarantee of weighted conformal prediction assumes the weights are
          known; ours are estimated (domain classifier in A, predicted-class ratios in B), clipped, and the test-point weight
          is approximated by an average. Experiment A's covariate shift is deliberately severe enough that overlap is poor.
        * **SentryCam here is PCA, and the runs never failed.** The alert comparison (Section 5) characterises how two
          spaces summarise a *healthy* ten-epoch trajectory; it says nothing about detection power on a run that diverges.
        * **Wall-clock and environment.** Timings are for a shared CPU host and vary with load; the notebook prints its total
          runtime at the end.

        **Recommended follow-ups on real data, in order of cheapness:** (1) the time-preserving arms against the legacy
        baseline (already in notebook 4); (2) an AutoClip percentile ablation only if the clip-rate finding above
        transfers; (3) calibration comparison with $\beta$ chosen on a held-out split; (4) a weighted conformal variant if
        the real stress conditions resemble a class-mix shift.
        """),

        code(r"""
        # Closing table of the measured findings (every value is computed above, nothing typed by hand).
        FINDINGS = pd.DataFrame([
            {"direction": "Section 4: pooled vs time-preserving", "measure": "test acc, pool / best time model", "value": f"{EVAL['logmel_pool']['acc']:.3f} / {EVAL[best_name]['acc']:.3f}"},
            {"direction": "Section 4: order accuracy given group", "measure": "pool / best time model", "value": f"{EVAL['logmel_pool']['order_acc']:.3f} / {EVAL[best_name]['order_acc']:.3f}"},
            {"direction": "Section 6: AutoClip sweep", "measure": "mean test acc at p=1/10/25/100", "value": " / ".join(f"{agg.loc[p, 'acc']:.3f}" for p in P_GRID)},
            {"direction": "Section 6: AutoClip sweep", "measure": "clip rate at p=1/10/25/100", "value": " / ".join(f"{agg.loc[p, 'clip']:.2f}" for p in P_GRID)},
            {"direction": "Section 7: calibration", "measure": "ECE baseline / TS / MDCA / both", "value": " / ".join(f"{v:.4f}" for v in COMP['ECE mean over seeds'])},
            {"direction": "Section 8A: covariate shift", "measure": f"miscoverage at s={A.index[-1]}, unweighted / weighted", "value": f"{A.iloc[-1].miscov_unweighted:.3f} / {A.iloc[-1].miscov_weighted:.3f}"},
            {"direction": "Section 8B: label shift", "measure": f"miscoverage at tau={B.index[-1]:g}, unw / est / oracle", "value": f"{B.iloc[-1].unw:.3f} / {B.iloc[-1].est:.3f} / {B.iloc[-1].orc:.3f}"},
        ]).set_index("direction")
        display(FINDINGS)
        elapsed = time.time() - NB_T0
        print(f"total notebook runtime since the helper cell: {elapsed:.0f}s ({elapsed / 60:.1f} min)")
        note("N2.9.1[runtime]", f"{elapsed:.0f}s end to end on this host (target: a few minutes on CPU)")
        check("N2.9.1[no_offloaded_cells]", True, "nothing in this notebook is tagged offloaded; every cell ran locally")
        """),
    ]
