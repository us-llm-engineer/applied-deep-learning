"""NB4 part 2: in-training diagnostics V1-V4 and the raw-vs-2D SentryCam alert comparison (the headline)."""
from nbkit import md, code


def cells():
    return [
        md(r"""
        ## 5. In-training diagnostics (V1--V4) on the three instrumented architectures

        The training loop of the real run recorded, for `gru`, `timepool`, and `wide`: the per-epoch loss and accuracy
        history; the L2 norm of every optimizer step's gradient together with the AutoClip threshold that was
        current at that step; and, after every epoch, a probe of the penultimate-layer representation of the fixed
        120-sample probe set. This section reads those recordings back. Nothing here retrains anything.

        Two methods are involved, and each carries a deviation label that is repeated in the figure titles.

        - **AutoClip** (arXiv:2007.14469). The research file records the threshold as
          $\eta_c(t) = \mathrm{percentile}_p\big(G_h(t)\big)$ over the *full cumulative* history $G_h(t)$ of per-step
          gradient L2 norms (no window), with default $p=10$. The percentile formula and the full-cumulative-history scope are
          reproduced exactly (check `N4.5.3` verifies this), but the threshold at step $t$ is computed from the prior
          history *before* the current norm is appended, whereas the paper appends first. See Section 8.2 for the full note.
        - **SentryCam** (arXiv:2405.15135). The research file records a live per-epoch instability alert from two scalars
          computed on a 2D *projected* latent space (not on raw activations): inter-cluster distance and intra-cluster
          variance. The alert fires when a metric shows a 2-consecutive-epoch unhealthy trend **and**
          $|\Delta| > 0.25\,\sigma_{10}$, where $\sigma_{10}$ is the rolling standard deviation over 10 epochs.
          **Deviation:** SentryCam specifies a trained parametric autoencoder for the projection; we use **PCA**.
          This is labelled in the title of every figure that uses it.

        All results are from a single exploration run (clean, seed 17). Epoch numbers in figures are 1-based; the
        alert function `sentrycam_alert_epoch` returns 0-based indices, and tables show both.
        """),

        md(r"""
        ### 5.1 V1: training curves

        Loss values are the *true* cross-entropy (the accumulation-scale accounting bug of earlier rounds is fixed in
        this run), so a network at chance on 12 classes should sit near $\ln 12 = 2.4849$. The cell below prints the
        first-epoch numbers so that this claim can be checked rather than asserted.
        """),

        code(r"""
        # N4.5.1: V1 numbers: first-epoch loss vs ln(12), val-loss range and late volatility.
        LN12 = float(np.log(12))
        rows = []
        for a in ARCHS:
            h = DATA[a]["res"]["history"]
            tl = np.array([r["train_loss"] for r in h]); vl = np.array([r["val_loss"] for r in h]); va = np.array([r["val_accuracy"] for r in h])
            rows.append({"arch": a, "train_loss_ep1": tl[0], "val_loss_ep1": vl[0], "ln12": LN12, "val_loss_min": vl.min(), "val_loss_max": vl.max(),
                         "val_loss_std_diff_last10": np.std(np.diff(vl[-10:])), "val_acc_std_diff_last10": np.std(np.diff(va[-10:])),
                         "train_loss_last": tl[-1], "val_loss_last": vl[-1]})
        V1T = pd.DataFrame(rows).set_index("arch")
        display(V1T.round(4))
        for a in ARCHS:
            check(f"N4.5.1[{a}.ep1_le_ln12]", V1T.loc[a, "train_loss_ep1"] <= LN12 + 1e-6 and V1T.loc[a, "train_loss_ep1"] > 0.5 * LN12,
                  f"first-epoch mean train loss {V1T.loc[a, 'train_loss_ep1']:.4f} vs ln12={LN12:.4f}")
            check(f"N4.5.1[{a}.train_below_val]", V1T.loc[a, "train_loss_last"] < V1T.loc[a, "val_loss_last"], "final train loss below final val loss (generalisation gap present)")
        check("N4.5.1[wide_val_range]", V1T.loc["wide", "val_loss_min"] < 0.5 and V1T.loc["wide", "val_loss_max"] > 2.2,
              f"wide val loss spans {V1T.loc['wide', 'val_loss_min']:.2f} .. {V1T.loc['wide', 'val_loss_max']:.2f}")
        check("N4.5.1[gru_smoothest]", V1T["val_loss_std_diff_last10"].idxmin() == "gru", "gru has the smallest late val-loss volatility of the three")
        """),

        md(r"""
        ### Figure NB4-3: V1 training curves, three architectures overlaid
        """),

        code(r"""
        # N4.5.2: V1 figure.
        fig, ax = plt.subplots(1, 2, figsize=(13, 4.4))
        for a in ARCHS:
            h = DATA[a]["res"]["history"]; ep = [r["epoch"] + 1 for r in h]
            ax[0].plot(ep, [r["train_loss"] for r in h], color=ACOL[a], label=f"{a} train")
            ax[0].plot(ep, [r["val_loss"] for r in h], color=ACOL[a], ls="--", label=f"{a} val")
            ax[1].plot(ep, [r["val_accuracy"] for r in h], color=ACOL[a], label=a)
        ax[0].axhline(LN12, color="#667085", ls=":", label="chance CE = ln 12")
        ax[0].set(xlabel="epoch", ylabel="cross-entropy", title="V1a  loss (true CE)")
        ax[1].axhline(1 / 12, color="#667085", ls=":", label="chance 1/12")
        ax[1].set(xlabel="epoch", ylabel="validation accuracy", ylim=(0, 1), title="V1b  validation accuracy")
        ax[0].legend(frameon=False, fontsize=7, ncol=2); ax[1].legend(frameon=False, fontsize=8)
        fig.suptitle("NB4-3  V1: loss and accuracy per epoch (clean, seed 17, exploration run)", y=1.03)
        fig.tight_layout()
        savefig(fig, "NB4-3_V1_training_curves.png")
        print("[N4.5.2] V1 drawn")
        """),

        md(r"""
        ### How to read this chart

        **Data source:** the `history` list (per-epoch `train_loss`, `val_loss`, `val_accuracy`) in
        `<ARTIFACT_ROOT>/metrics/arch-{gru,timepool,wide}.result.json`.

        **Axes:** horizontal is epoch (1-based). Left panel: cross-entropy, solid = training loss, dashed = validation
        loss, dotted grey = $\ln 12 \approx 2.4849$, the loss of a uniform guess over 12 classes. Right panel: validation
        accuracy with chance $1/12$ dotted.

        **Interpretation:** all three first-epoch training losses ($2.10$ to $2.33$) sit just below $\ln 12$, which is
        what a network starting at chance and learning during its first epoch would produce; an accounting error (for
        example a loss divided by the number of accumulation steps) would put the start far below $2.48$, so this is an
        independent confirmation that the true-CE accounting is correct. The three validation traces then separate
        sharply. `gru`'s validation loss falls smoothly to about $0.31$ and its accuracy plateaus near $0.9$. `wide`'s
        validation loss is violently unstable, roughly $0.43$ up to $2.31$ within the run, even as its training loss
        keeps falling. `timepool`'s validation loss stays high and noisy.

        **Falsifier:** if the first-epoch losses had been near $0$ or far above $2.48$, the loss accounting would be
        suspect. If `wide`'s validation loss were smooth it would contradict the instability claim; the printed
        volatility of the last ten epochs (about $0.80$ for `wide` against $0.02$ for `gru`) is the quantitative test.
        """),

        md(r"""
        ### 5.2 V2: AutoClip, gradient norm versus adaptive threshold

        For every optimizer step $t$, the run stored the pre-clip gradient L2 norm $g_t$ and the AutoClip threshold
        $\eta_c(t)$ applied at that step. Step $0$ has no history, so its threshold is infinite (no clipping). A step is
        counted as **clipped** when $g_t > \eta_c(t)$ with $\eta_c(t)$ finite.

        Because the threshold is the $10$th percentile of the history, $90\%$ of a stationary gradient distribution
        would lie *above* it. A clip rate near $90\%$ therefore does not indicate pathological gradients by itself; it is
        what the $p=10$ setting produces. What departs from $90\%$ carries information: a rate well below $90\%$ means the
        recent gradients are smaller than the early history, and a rate near $100\%$ means they are larger.
        The rates are recomputed from the arrays below, not hard-coded.
        """),

        code(r"""
        # N4.5.3: clip rates recomputed from the arrays; verify the threshold equals the 10th percentile of PRIOR history.
        rows = []
        for a in ARCHS:
            g, c = DATA[a]["npz"]["grad_norm"], DATA[a]["npz"]["clip_threshold"]
            finite = np.isfinite(c); bit = finite & (g > c)
            ref = np.array([np.percentile(g[:t], 10) if t > 0 else np.inf for t in range(len(g))])
            match = np.allclose(ref[finite], c[finite], rtol=1e-4, atol=1e-6)
            rows.append({"arch": a, "steps": len(g), "finite_threshold_steps": int(finite.sum()), "clipped": int(bit.sum()),
                         "clip_rate_of_all_steps": bit.sum() / len(g), "clip_rate_of_finite": bit.sum() / finite.sum(),
                         "threshold_first": c[finite][0], "threshold_last": c[finite][-1],
                         "grad_median_first100": np.median(g[:100]), "grad_median_last100": np.median(g[-100:]),
                         "matches_P10_of_prior_history": bool(match)})
        CLIP = pd.DataFrame(rows).set_index("arch")
        display(CLIP.round(4))
        for a in ARCHS:
            check(f"N4.5.3[{a}.P10_prior_history]", CLIP.loc[a, "matches_P10_of_prior_history"], "stored threshold == np.percentile(grad_norm[:t], 10) for every step t>0")
            check(f"N4.5.3[{a}.step0_inf]", not np.isfinite(DATA[a]["npz"]["clip_threshold"][0]), "step-0 threshold is infinite (no history yet)")
        check("N4.5.3[counts]", (CLIP["clipped"].to_dict() == {"gru": 663, "timepool": 1284, "wide": 910}), f"clipped counts {CLIP['clipped'].to_dict()} (expected: gru 663, timepool 1284, wide 910)")
        check("N4.5.3[timepool_rate]", abs(CLIP.loc["timepool", "clip_rate_of_all_steps"] - 0.991) < 0.001, f"timepool clip rate {CLIP.loc['timepool', 'clip_rate_of_all_steps']:.4f}")
        """),

        md(r"""
        ### Figure NB4-4: V2 gradient norm, AutoClip threshold, and clipped steps
        """),

        code(r"""
        # N4.5.4: V2 figure.
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), sharey=True)
        for ax, a in zip(axes, ARCHS):
            g, c = DATA[a]["npz"]["grad_norm"], DATA[a]["npz"]["clip_threshold"]
            step = np.arange(len(g)); finite = np.isfinite(c); bit = finite & (g > c)
            ax.plot(step, g, lw=0.7, color=ACOL[a], label="grad norm (pre-clip)")
            ax.plot(step[finite], c[finite], lw=1.6, color="#d62728", label=r"AutoClip $\eta_c = P_{10}$ of history")
            ax.scatter(step[bit], g[bit], s=3, color="#d62728", alpha=0.45, label=f"clipped ({bit.sum()}/{len(g)})")
            ax.set(xlabel="optimizer step", yscale="log", title=a)
            ax.legend(frameon=False, fontsize=7)
        axes[0].set_ylabel("gradient L2 norm")
        fig.suptitle("NB4-4  V2: AutoClip (arXiv:2007.14469) gradient norm vs adaptive threshold (threshold from prior history, see 8.2)", y=1.03)
        fig.tight_layout()
        savefig(fig, "NB4-4_V2_autoclip.png")
        print("[N4.5.4] V2 drawn")
        """),

        md(r"""
        ### How to read this chart

        **Data source:** `grad_norm` and `clip_threshold` in `<ARTIFACT_ROOT>/metrics/arch-*.probes.npz`; one entry per
        optimizer step ($1008$ for `gru`, $1296$ for `timepool` and `wide`).

        **Axes:** horizontal is the optimizer step; vertical is the gradient L2 norm on a log scale. The thin coloured line
        is the pre-clip gradient norm, the red line the threshold $\eta_c(t)$, and red dots mark steps where the gradient
        exceeded the threshold (clipping bit).

        **Interpretation:** the threshold is not a ratchet that only decreases. It **tracks the gradient distribution**.
        For `gru` and `wide`, whose gradients shrink as training converges (median of the last 100 steps well below the
        first 100), the threshold falls with them, and the clip rate is $663/1008 = 65.8\%$ and $910/1296 = 70.2\%$, below
        the $90\%$ that a stationary distribution would give. For `timepool` the gradient norm *grows* late in training
        (median of the last 100 steps about six times the first 100), the threshold lags it, and the clip rate is
        $1284/1296 = 99.1\%$: almost every step is clipped, which says the optimizer is running against a threshold set by
        smaller, earlier gradients. That is a real pathology of the `timepool` run, not an artefact of the plot.

        **Falsifier:** if the threshold were a one-way ratchet, the red line would be monotonically non-increasing; for
        `timepool` it rises. If the stored threshold were not the $10$th percentile of prior history, the reconstruction
        check above would fail; it passes for every step of every architecture.
        """),

        md(r"""
        ### Figure NB4-5: clip rate per epoch and threshold tracking

        A single global clip rate hides *when* clipping happened. The next figure bins the steps by epoch
        ($36$ steps per epoch for `gru`, $36$ for the others: the number of steps divided by the epochs run) and plots the
        within-epoch clip rate, together with the ratio of the threshold to the running median of the gradient norm.
        """),

        code(r"""
        # N4.5.5: per-epoch clip rate and threshold/median-gradient ratio.
        fig, ax = plt.subplots(1, 2, figsize=(13, 4.2))
        per_epoch = {}
        for a in ARCHS:
            g, c = DATA[a]["npz"]["grad_norm"], DATA[a]["npz"]["clip_threshold"]
            E = DATA[a]["res"]["epochs_run"]; spe = len(g) // E
            check(f"N4.5.5[{a}.steps_per_epoch]", spe * E == len(g), f"{len(g)} steps = {E} epochs x {spe} steps")
            bit = (np.isfinite(c) & (g > c)).astype(float)
            rate = bit[: spe * E].reshape(E, spe).mean(axis=1)
            per_epoch[a] = rate
            ax[0].plot(np.arange(1, E + 1), rate, color=ACOL[a], label=a)
            finite = np.isfinite(c)
            run_med = np.array([np.median(g[max(0, t - spe): t + 1]) for t in range(len(g))])
            ax[1].plot(np.arange(len(g))[finite], (c[finite] / run_med[finite]), color=ACOL[a], lw=0.9, label=a)
        ax[0].axhline(0.9, color="#667085", ls=":", label="0.9 (stationary expectation for P10)")
        ax[0].set(xlabel="epoch", ylabel="fraction of steps clipped", ylim=(0, 1.02), title="within-epoch clip rate")
        ax[1].axhline(1.0, color="#667085", ls=":")
        ax[1].set(xlabel="optimizer step", ylabel=r"$\eta_c$ / running median of grad norm (last epoch of steps)", yscale="log", title="threshold relative to recent gradient")
        ax[0].legend(frameon=False, fontsize=8); ax[1].legend(frameon=False, fontsize=8)
        fig.suptitle("NB4-5  V2 (continued): when clipping happened and how far the threshold lags the gradient", y=1.03)
        fig.tight_layout()
        savefig(fig, "NB4-5_V2_clip_rate.png")
        for a in ARCHS:
            print(f"{a}: mean within-epoch clip rate first 5 epochs {per_epoch[a][:5].mean():.3f}, last 5 epochs {per_epoch[a][-5:].mean():.3f}")
        check("N4.5.5[timepool_late_rate]", per_epoch["timepool"][-5:].mean() > 0.95, "timepool clips >95% of steps in its last 5 epochs")
        """),

        md(r"""
        ### How to read this chart

        **Data source:** the same `grad_norm` and `clip_threshold` arrays as Figure NB4-4, reshaped into
        (epochs $\times$ steps per epoch).

        **Axes:** left, epoch versus the fraction of that epoch's steps that were clipped, with the $0.9$ level marked.
        Right, optimizer step versus the ratio of the threshold to the running median of the most recent epoch's worth of
        gradient norms, on a log scale; a ratio of $1$ means the threshold equals the recent typical gradient, values
        below $1$ mean the threshold sits under the typical gradient (so most steps are clipped).

        **Interpretation:** a healthy adaptive threshold keeps the ratio roughly stable. A ratio that falls steadily
        below $1$ (`timepool`) is the signature of gradients that grow relative to history and are being clipped almost
        every step. `gru` and `wide` show ratios that stay in a narrower band because their gradients decay with the
        threshold following.

        **Falsifier:** if the clip rate were near $0.9$ throughout for all three arms, the $P_{10}$ rule would be doing
        nothing informative and the global counts above would be an artefact of the percentile choice alone. The
        `timepool` late-epoch rate near $1.0$ and the lower `gru`/`wide` rates contradict that.
        """),

        md(r"""
        ### 5.3 V3: SentryCam 2D latent scatter over epochs

        After each epoch the probe set was passed through the model and the penultimate-layer vector for each of the 120
        samples was projected to 2D. Panels below show that 2D scatter at eight roughly evenly spaced epochs plus the two
        alert epochs (the raw-space alert, and the 2D alert, both from Section 5.5), coloured by true class.

        **Deviation label:** the projection is **PCA**, not SentryCam's trained parametric autoencoder. Each epoch's PCA is
        fitted separately, so axis orientation and sign are arbitrary from one panel to the next; only the *spacing* of the
        clusters within a panel is meaningful. Axis limits are free per panel for the same reason.
        """),

        code(r"""
        # N4.5.6: helper: alert epochs from src.diagnostics on both raw and 2D series, then the V3 grid.
        from src.diagnostics import sentrycam_alert_epoch, cluster_drift_metrics, project_2d

        ALERT = {}
        for a in ARCHS:
            n = DATA[a]["npz"]
            ALERT[a] = {"raw": sentrycam_alert_epoch(n["inter_raw"], n["intra_raw"]), "2d": sentrycam_alert_epoch(n["inter_2d"], n["intra_2d"])}
        print({a: ALERT[a] for a in ARCHS})

        def v3_grid(arch, fname, fig_no):
            co, lab = DATA[arch]["npz"]["coords_2d"], DATA[arch]["npz"]["labels"]
            E = co.shape[0]
            picks = sorted(set(np.linspace(0, E - 1, 8).astype(int).tolist() + [ALERT[arch]["raw"], ALERT[arch]["2d"]]))
            picks = [p for p in picks if p is not None]
            fig, axes = plt.subplots(2, 5, figsize=(17, 6.6))
            for ax in axes.ravel():
                ax.set_visible(False)
            for ax, e in zip(axes.ravel(), picks):
                ax.set_visible(True)
                ax.scatter(co[e, :, 0], co[e, :, 1], c=lab, cmap="tab20", s=12)
                tag = []
                if e == ALERT[arch]["raw"]: tag.append("raw-space alert")
                if e == ALERT[arch]["2d"]: tag.append("2D alert")
                ax.set_title(f"epoch {e + 1}" + (" | " + ", ".join(tag) if tag else ""), fontsize=9, color="#d62728" if tag else "black")
                ax.set_xticks([]); ax.set_yticks([])
                if tag:
                    for sp in ax.spines.values():
                        sp.set_edgecolor("#d62728"); sp.set_linewidth(2)
            fig.suptitle(f"NB4-{fig_no}  V3: penultimate-layer 2D projection per epoch, {arch} (PCA, NOT SentryCam's parametric autoencoder); colour = true class", y=1.0)
            fig.tight_layout()
            savefig(fig, fname)
            return picks

        p_gru = v3_grid("gru", "NB4-6_V3_latent_gru.png", 6)
        print("[N4.5.6] V3 gru panels at epochs (0-based)", p_gru)
        """),

        md(r"""
        ### How to read this chart

        **Data source:** `coords_2d` (shape $(E,120,2)$) and `labels` in `<ARTIFACT_ROOT>/metrics/arch-gru.probes.npz`.

        **Axes:** each panel is one epoch; the two axes are the first two principal components of that epoch's
        penultimate-layer activations (units arbitrary, orientation arbitrary between panels). Colour is the true class
        of each of the 120 probe samples (12 classes, 10 points each). Red-framed panels mark alert epochs.

        **Interpretation:** for the healthy `gru`, points that start as a diffuse cloud in the first panel condense into
        compact, separated same-colour groups, and stay that way for the remaining epochs. Tighter same-colour groups and
        larger gaps between groups are exactly what a rising inter-cluster distance and a falling intra-cluster variance
        (Figure NB4-8) quantify.

        **Falsifier:** if clusters were equally diffuse at the last epoch as at the first, the accuracy of about $0.9$
        would be hard to explain by this projection, and the inter/intra curves would be flat. Note that two principal
        components cannot show separation that lives in other directions, so a panel that looks mixed is weaker evidence
        than a panel that looks separated.
        """),

        code(r"""
        # N4.5.7: V3 for timepool and wide (same helper).
        p_tp = v3_grid("timepool", "NB4-7a_V3_latent_timepool.png", "7a")
        print("[N4.5.7] V3 timepool panels (0-based)", p_tp)
        """),

        code(r"""
        # N4.5.8: V3 wide.
        p_wd = v3_grid("wide", "NB4-7b_V3_latent_wide.png", "7b")
        print("[N4.5.8] V3 wide panels (0-based)", p_wd)
        """),

        md(r"""
        ### How to read these two charts (`timepool`, then `wide`)

        **Data source:** `coords_2d` and `labels` in `<ARTIFACT_ROOT>/metrics/arch-timepool.probes.npz` and
        `arch-wide.probes.npz`; layout, colouring, and the PCA deviation label are identical to the `gru` figure.

        **Axes:** as in the previous figure. Note that axis limits are free per panel, so the *absolute* spread of these
        two architectures (their intra-cluster variance is an order of magnitude larger than `gru`'s) is visible in the
        tick-free panels only through how much of the panel the clusters fill and how much they overlap.

        **Interpretation:** neither architecture reaches the compact, separated clusters of `gru` by the end. `timepool`'s
        same-class points remain spread and overlapping, consistent with its low accuracy. `wide` shows some class
        structure but with large within-class dispersion that does not settle from epoch to epoch, consistent with its
        oscillating validation loss.

        **Falsifier:** if either architecture's last-epoch panel looked as tight as `gru`'s, the ranking by intra-cluster
        variance in Figure NB4-8 would be contradicted. Qualitative panel impressions are secondary to the computed
        metrics, which use all 120 points and all epochs.
        """),

        md(r"""
        ### 5.4 V4: cluster health computed in 2D, with the significance band

        For each epoch, the two SentryCam scalars are computed on the 2D coordinates:
        inter-cluster distance $D_t$ (mean pairwise Euclidean distance between the 12 class centroids) and
        intra-cluster variance $V_t$ (unweighted mean over classes of the mean squared distance to the class centroid).
        Both are stored in the probe file as `inter_2d` and `intra_2d`, and the check cell below recomputes them from
        `coords_2d` with `src.diagnostics.cluster_drift_metrics` to confirm the stored series.

        The alert's significance margin is $\alpha\,\sigma_{10}(t)$ with $\alpha = 0.25$ and $\sigma_{10}(t)$ the standard
        deviation of the previous (up to) 10 epochs. It is drawn as a band $x_t \pm \alpha\,\sigma_{10}(t)$ around each
        series. **Deviation:** the 2D space is a PCA projection, not a parametric autoencoder.
        """),

        code(r"""
        # N4.5.9: recompute the 2D metrics from coords_2d and compare with the stored series.
        for a in ARCHS:
            n = DATA[a]["npz"]
            re_inter, re_intra = zip(*[cluster_drift_metrics(n["coords_2d"][t], n["labels"]) for t in range(n["coords_2d"].shape[0])])
            check(f"N4.5.9[{a}.inter_2d]", np.allclose(re_inter, n["inter_2d"], rtol=1e-3, atol=1e-4), f"max abs diff {np.max(np.abs(np.array(re_inter) - n['inter_2d'])):.2e}")
            check(f"N4.5.9[{a}.intra_2d]", np.allclose(re_intra, n["intra_2d"], rtol=1e-3, atol=1e-4), f"max abs diff {np.max(np.abs(np.array(re_intra) - n['intra_2d'])):.2e}")
        """),

        md(r"""
        ### Figure NB4-8: V4 cluster health in 2D
        """),

        code(r"""
        # N4.5.10: V4 figure with rolling-sigma band and alert epochs.
        ALPHA, WINDOW = 0.25, 10
        def rolling_sigma(series, window=WINDOW):
            return np.array([np.std(series[max(0, t - window):t]) if t >= 2 else np.nan for t in range(len(series))])

        fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
        for ax, a in zip(axes, ARCHS):
            n = DATA[a]["npz"]; ep = np.arange(1, len(n["inter_2d"]) + 1)
            for series, col, lbl in ((n["inter_2d"], "#3568A8", "inter-cluster distance (2D)"), (n["intra_2d"], "#E58B2A", "intra-cluster variance (2D)")):
                ax.plot(ep, series, color=col, label=lbl)
                sg = rolling_sigma(series)
                ax.fill_between(ep, series - ALPHA * sg, series + ALPHA * sg, color=col, alpha=0.25)
            if ALERT[a]["2d"] is not None:
                ax.axvline(ALERT[a]["2d"] + 1, color="#d62728", ls="--", lw=1.4, label=f"2D alert (epoch {ALERT[a]['2d'] + 1})")
            if ALERT[a]["raw"] is not None:
                ax.axvline(ALERT[a]["raw"] + 1, color="#7f7f7f", ls=":", lw=1.4, label=f"raw-space alert (epoch {ALERT[a]['raw'] + 1})")
            ax.set(xlabel="epoch", title=a); ax.legend(frameon=False, fontsize=7)
        fig.suptitle(r"NB4-8  V4: SentryCam cluster health in 2D (PCA, not the paper's autoencoder), band = $\pm\alpha\sigma_{10}$, $\alpha=0.25$", y=1.03)
        fig.tight_layout()
        savefig(fig, "NB4-8_V4_cluster_health.png")
        print("[N4.5.10] V4 drawn")
        """),

        md(r"""
        ### How to read this chart

        **Data source:** `inter_2d` and `intra_2d` in `<ARTIFACT_ROOT>/metrics/arch-*.probes.npz`; the alert epochs are computed
        in the notebook by `src.diagnostics.sentrycam_alert_epoch` (raw-space alert from `inter_raw`/`intra_raw`).

        **Axes:** horizontal is epoch (1-based); vertical is the metric value in 2D PCA units. Blue is inter-cluster
        distance (larger = better separated), orange is intra-cluster variance (smaller = tighter). The shaded ribbon around
        each line is the alert's significance margin $\pm\alpha\sigma_{10}$: a one-epoch change must exceed the ribbon's
        half-width to count. Red dashed = 2D alert epoch; grey dotted = raw-space alert epoch.

        **Interpretation:** `gru`'s blue line rises quickly from about $2.6$ to about $4.3$ and its orange line falls from
        about $6.7$ to about $1.5$--$1.8$: separation up, dispersion down, then a plateau in which the ribbon is very
        narrow. `timepool`'s orange line *rises* (roughly $10$ to $29$ at the end, peaking above $34$) while its blue line
        also rises modestly, i.e. the classes drift apart and spread at the same time. `wide`'s orange line swings between
        about $48$ and $135$ without settling.

        **Falsifier:** a healthy run should not show a persistent rise in intra-cluster variance; `gru`'s does not.
        Conversely, if `wide`'s intra-cluster variance were flat, its unstable validation loss would lack a matching
        geometric signature. Epoch-by-epoch correlation between these metrics and validation loss is quantified in
        Section 5.6 and is weaker than the visual impression for two of the three arms.
        """),

        md(r"""
        ### 5.5 Headline: raw-space alert versus 2D alert

        SentryCam's alert can be evaluated on two different series that the run recorded for every epoch:
        the cluster metrics computed on the **raw penultimate activations** (`inter_raw`, `intra_raw`) and the same
        metrics computed on the **2D projection** (`inter_2d`, `intra_2d`). The research file records that SentryCam
        computes its metrics on a 2D *projected* latent space, "not raw activations". The reason for projecting first is
        not written in that research file; it is stated in this project's own docstring of `project_2d`
        (`src/diagnostics.py`) as the paper's motivation: raw high-dimensional distances are noise-dominated
        (the curse of dimensionality). That is the project's gloss on the paper, not a verified quotation, and it is
        treated here as a hypothesis the data can test: if it is right, the raw-space alert should be *less informative*
        than the 2D one.

        The cell below runs the identical `sentrycam_alert_epoch(k=2, alpha=0.25, window=10)` on both series for all
        three architectures and separates the two internal conditions (distance-decrease and variance-increase) by feeding
        the other series as a constant, which can never fire.
        """),

        code(r"""
        # N4.5.11: raw-vs-2D alert table, with each condition separated.
        rows = []
        for a in ARCHS:
            n = DATA[a]["npz"]; one = np.ones(len(n["inter_2d"]))
            for space in ("raw", "2d"):
                i, v = n[f"inter_{space}"], n[f"intra_{space}"]
                al = sentrycam_alert_epoch(i, v)
                rows.append({"arch": a, "space": space, "alert_idx0": al, "alert_epoch_1based": None if al is None else al + 1,
                             "distance_cond_idx0": sentrycam_alert_epoch(i, one), "variance_cond_idx0": sentrycam_alert_epoch(one, v),
                             "inter_first": float(i[0]), "inter_last": float(i[-1]), "intra_first": float(v[0]), "intra_last": float(v[-1]),
                             "intra_min": float(v.min()), "intra_max": float(v.max())})
        ALERT_TABLE = pd.DataFrame(rows)
        display(ALERT_TABLE.round(3))
        wide_table = ALERT_TABLE.pivot(index="arch", columns="space", values="alert_idx0")
        print("alert epoch, 0-based (raw vs 2D):")
        display(wide_table)
        check("N4.5.11[raw_all_same]", wide_table["raw"].nunique() == 1, f"raw-space alert epoch identical for all three archs: {wide_table['raw'].tolist()}")
        check("N4.5.11[raw_gru_fires]", wide_table.loc["gru", "raw"] is not None and not np.isnan(wide_table.loc["gru", "raw"]), "raw-space alert fires on the healthy gru (uninformative)")
        check("N4.5.11[matches_json]", all(DATA[a]["res"]["instability_alert_epoch"] == ALERT[a]["raw"] for a in ARCHS), "stored instability_alert_epoch equals the raw-space alert recomputed here")
        check("N4.5.11[2d_differs]", wide_table["2d"].nunique() > 1, f"2D alert epochs differ across archs: {wide_table['2d'].tolist()}")
        gru2, tp2, wd2 = (ALERT[a]["2d"] for a in ("gru", "timepool", "wide"))
        note("N4.5.11[gru_2d_alert]", f"the 2D alert ALSO fires on the healthy gru, at 0-based epoch {gru2} (1-based {gru2 + 1}); it is late (plateau) versus epoch {tp2} for timepool and {wd2} for wide, but it is not silent")
        """),

        md(r"""
        ### Figure NB4-9: raw and 2D series, side by side
        """),

        code(r"""
        # N4.5.12: raw vs 2D series for all three archs, alert epochs marked.
        fig, axes = plt.subplots(3, 4, figsize=(17, 10))
        for r_i, a in enumerate(ARCHS):
            n = DATA[a]["npz"]; ep = np.arange(1, len(n["inter_2d"]) + 1)
            for c_i, (key, ttl, col) in enumerate((("inter_raw", "raw inter-cluster distance", "#3568A8"), ("intra_raw", "raw intra-cluster variance", "#E58B2A"),
                                                   ("inter_2d", "2D inter-cluster distance (PCA)", "#3568A8"), ("intra_2d", "2D intra-cluster variance (PCA)", "#E58B2A"))):
                ax = axes[r_i, c_i]; s = n[key]
                ax.plot(ep, s, color=col); sg = rolling_sigma(s)
                ax.fill_between(ep, s - ALPHA * sg, s + ALPHA * sg, color=col, alpha=0.25)
                space = "raw" if key.endswith("raw") else "2d"
                if ALERT[a][space] is not None:
                    ax.axvline(ALERT[a][space] + 1, color="#d62728", ls="--", lw=1.3)
                ax.set_title(f"{a}: {ttl}" + (f" | alert e{ALERT[a][space] + 1}" if ALERT[a][space] is not None else ""), fontsize=8.5)
                if r_i == 2: ax.set_xlabel("epoch")
        fig.suptitle("NB4-9  Raw-space vs 2D (PCA, not SentryCam's autoencoder) cluster metrics; red dashed = alert epoch in that space", y=1.0)
        fig.tight_layout()
        savefig(fig, "NB4-9_raw_vs_2d.png")
        print("[N4.5.12] raw-vs-2D figure drawn")
        """),

        md(r"""
        ### How to read this chart

        **Data source:** `inter_raw`, `intra_raw`, `inter_2d`, `intra_2d` in `<ARTIFACT_ROOT>/metrics/arch-*.probes.npz`; alert
        epochs computed by `sentrycam_alert_epoch`.

        **Axes:** rows are architectures; the left two columns are the raw-space metrics, the right two the 2D metrics.
        Horizontal is epoch; vertical is the metric in its own units (the raw and 2D scales are not comparable). Ribbons
        are $\pm\alpha\sigma_{10}$; the red dashed line is that column-pair's alert epoch.

        **Interpretation:** in raw space every architecture, including the healthy `gru`, shows the same early pattern
        (both metrics grow over the first epochs while the representation is still forming), so the alert fires at the
        same early epoch, $2$ (0-based), for all three: it cannot tell a healthy run from an unhealthy one. In 2D the
        *trajectories* discriminate: `gru`'s intra-cluster variance falls from about $6.7$ to about $1.5$--$1.8$ while its
        inter-cluster distance rises from about $2.6$ to $4.3$; `timepool`'s intra-cluster variance rises from about
        $9.9$ to $28.7$ at the end; `wide`'s oscillates between about $48$ and $135$. This ordering matches the stability
        ordering in V1.

        **What the numbers do not show, stated plainly:** the 2D *alert epoch* is not a clean discriminator. It fires at
        0-based epoch $2$ for `timepool` and `wide` (early, as expected for unhealthy runs) but it also fires for the healthy
        `gru`, late, at 0-based epoch $17$. Both conditions trigger there together, on the plateau where the rolling
        $\sigma_{10}$ is small and a tiny wobble exceeds $\alpha\sigma_{10}$. The discrimination is in the *trajectories*
        and in *when* the alert fires, not in *whether* it fires.

        **Falsifier:** if the raw-space alert epochs differed across architectures and matched the 2D ones, the claim that
        raw-space is uninformative would fail; they are identical. If `gru`'s 2D alert never fired, the "not silent"
        caveat would be wrong; it fires.
        """),

        md(r"""
        ### 5.6 Sensitivity and how well the geometry tracks validation behaviour

        Two further checks temper the headline. First, the alert has a free choice of $k$ (consecutive epochs) and
        $\alpha$ (margin multiplier); the paper's values are $k=2$, $\alpha=0.25$. The sweep shows how much the alert epoch
        moves. Second, the Spearman rank correlation, across epochs, between each 2D metric and the validation loss or
        accuracy tells whether the geometric signal follows the behaviour it is supposed to foreshadow.
        Both are computed from the same files.
        """),

        code(r"""
        # N4.5.13: alert sensitivity to (alpha, k) for the 2D series.
        srows = []
        for a in ARCHS:
            n = DATA[a]["npz"]
            for alpha in (0.1, 0.25, 0.5, 1.0):
                for k in (2, 3):
                    srows.append({"arch": a, "alpha": alpha, "k": k, "alert_2d_idx0": sentrycam_alert_epoch(n["inter_2d"], n["intra_2d"], alpha=alpha, k=k),
                                  "alert_raw_idx0": sentrycam_alert_epoch(n["inter_raw"], n["intra_raw"], alpha=alpha, k=k)})
        SENS = pd.DataFrame(srows)
        display(SENS.pivot_table(index=["alpha", "k"], columns="arch", values=["alert_2d_idx0", "alert_raw_idx0"], aggfunc="first"))
        check("N4.5.13[paper_setting]", (SENS[(SENS.alpha == 0.25) & (SENS.k == 2)].set_index("arch")["alert_2d_idx0"].to_dict() == {a: ALERT[a]["2d"] for a in ARCHS}), "sweep at the paper setting reproduces the headline 2D alerts")
        n_none = int(SENS["alert_2d_idx0"].isna().sum())
        note("N4.5.13", f"{n_none} of {len(SENS)} (arch, alpha, k) cells never fire in 2D; the alert epoch is sensitive to k and alpha, so it should not be over-read")
        """),

        code(r"""
        # N4.5.14: Spearman correlation across epochs between 2D geometry and validation behaviour.
        crow = []
        for a in ARCHS:
            n = DATA[a]["npz"]; h = DATA[a]["res"]["history"]
            vl = np.array([r["val_loss"] for r in h]); va = np.array([r["val_accuracy"] for r in h])
            row = {"arch": a}
            for name, s in (("inter_2d", n["inter_2d"]), ("intra_2d", n["intra_2d"]), ("inter_raw", n["inter_raw"]), ("intra_raw", n["intra_raw"])):
                row[f"{name}~val_loss"] = spearmanr(s, vl)[0]; row[f"{name}~val_acc"] = spearmanr(s, va)[0]
            crow.append(row)
        CORR = pd.DataFrame(crow).set_index("arch")
        display(CORR.round(3))
        check("N4.5.14[gru_intra_tracks_loss]", CORR.loc["gru", "intra_2d~val_loss"] > 0.5, f"gru: rho(intra_2d, val_loss)={CORR.loc['gru', 'intra_2d~val_loss']:.3f}")
        note("N4.5.14[weak_for_others]", f"timepool rho(intra_2d, val_loss)={CORR.loc['timepool', 'intra_2d~val_loss']:.3f}; wide {CORR.loc['wide', 'intra_2d~val_loss']:.3f}: the geometry does not track epoch-level validation loss for the two unstable arms")
        """),

        md(r"""
        **Reading the two tables.** The sweep shows the 2D alert epoch for `timepool` and `wide` is $2$ across almost the
        whole grid (robustly early), whereas for `gru` it ranges from $8$ to $18$ and is absent for none of the 2D cells
        at $k=2$. The alert is therefore best read as a coarse timing signal. The correlation table is a warning against
        over-claiming: for `gru`, intra-cluster variance and validation loss rise and fall together (a strongly positive
        Spearman coefficient), which supports the reading that tight clusters accompany low loss; for `timepool` the sign is
        even negative and for `wide` it is near zero, so for the unstable arms the geometry differs from the validation
        loss at the epoch level even though its overall *level* (variance about ten times larger) separates them from
        `gru`. Ten points per class is a small probe set, and single-run rank correlations over $28$--$36$ epochs carry
        wide uncertainty; none of this is a significance test.
        """),

        md(r"""
        ### 5.7 Further in-training analyses

        The remaining cells use the same recorded arrays to test the main story from angles the headline figures do not
        cover: the *distribution* of gradient norms rather than only their threshold, the geometry *per class* rather than
        averaged over classes, a classification-style score computed inside the 2D projection, and how far the alert epochs
        sit from the epoch the saved model actually comes from. Everything is descriptive and single-run.
        """),

        code(r"""
        # N4.5.15: per-epoch quantiles of the pre-clip gradient norm.
        QS = (10, 50, 90, 99)
        GQ = {}
        for a in ARCHS:
            g = DATA[a]["npz"]["grad_norm"]; E = DATA[a]["res"]["epochs_run"]; spe = len(g) // E
            per = g[: E * spe].reshape(E, spe)
            GQ[a] = {q: np.percentile(per, q, axis=1) for q in QS}
            GQ[a]["max"] = per.max(axis=1); GQ[a]["spe"] = spe
            check(f"N4.5.15[{a}.finite]", np.isfinite(per).all(), f"all {per.size} gradient norms finite")
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), sharey=True)
        for ax, a in zip(axes, ARCHS):
            ep = np.arange(1, DATA[a]["res"]["epochs_run"] + 1)
            ax.fill_between(ep, GQ[a][10], GQ[a][90], color=ACOL[a], alpha=0.25, label="10th to 90th percentile")
            ax.plot(ep, GQ[a][50], color=ACOL[a], label="median")
            ax.plot(ep, GQ[a][99], color="black", lw=0.8, ls="--", label="99th percentile")
            ax.plot(ep, GQ[a]["max"], color="#d62728", lw=0.8, ls=":", label="max")
            ax.set(xlabel="epoch", yscale="log", title=a); ax.legend(frameon=False, fontsize=7)
        axes[0].set_ylabel("pre-clip gradient L2 norm")
        fig.suptitle("NB4-5b  Distribution of per-step gradient norms within each epoch (AutoClip input, arXiv:2007.14469)", y=1.03)
        fig.tight_layout()
        savefig(fig, "NB4-5b_grad_quantiles.png")
        tail = pd.DataFrame({a: {"median ep1": GQ[a][50][0], "median last": GQ[a][50][-1], "max/median (whole run)": DATA[a]["npz"]["grad_norm"].max() / np.median(DATA[a]["npz"]["grad_norm"]),
                                 "p99/p50 mean over epochs": float(np.mean(GQ[a][99] / GQ[a][50]))} for a in ARCHS}).T
        display(tail.round(3))
        check("N4.5.15[timepool_grows]", GQ["timepool"][50][-1] > 2 * GQ["timepool"][50][0], "timepool's median gradient norm at least doubles from the first to the last epoch")
        check("N4.5.15[gru_shrinks]", GQ["gru"][50][-1] < 0.5 * GQ["gru"][50][0], "gru's median gradient norm at least halves from the first to the last epoch")
        """),

        md(r"""
        ### How to read this chart

        **Data source:** `grad_norm` in `<ARTIFACT_ROOT>/metrics/arch-*.probes.npz`, reshaped to (epochs $\times$ steps per
        epoch).

        **Axes:** horizontal is epoch; vertical is the gradient L2 norm on a log scale. The band is the $10$th to $90$th
        percentile of that epoch's step norms, the solid line the median, the dashed line the $99$th percentile, the dotted
        line the maximum.

        **Interpretation:** this figure shows the distribution the AutoClip percentile is applied to. For `gru` the whole
        band drifts down by roughly an order of magnitude as training converges, so a cumulative $P_{10}$ threshold that
        remembers early large gradients ends up *above* recent typical gradients, which is why fewer late steps are
        clipped. For `timepool` the band drifts *up*, so the cumulative $P_{10}$ is left far below the recent gradients and
        nearly all steps are clipped. Spikes in the maximum line without any change in the median are isolated large
        gradients, which AutoClip is meant to catch.

        **Falsifier:** if the medians were constant across epochs for all three architectures, the differences in clip
        rate would have to come from tail behaviour alone; the printed median ratios show they come from a shift of the
        bulk.
        """),

        code(r"""
        # N4.5.16: per-class 2D geometry. Labels are integer indices; we ASSUME index i corresponds to STUDY_LABELS[i] (src/dataset.py).
        from src.dataset import STUDY_LABELS
        CLASSES = list(STUDY_LABELS)

        def per_class_intra(coords, labels):
            return np.array([np.mean(np.sum((coords[labels == k] - coords[labels == k].mean(axis=0)) ** 2, axis=1)) for k in range(12)])

        def centroid_matrix(coords, labels):
            cen = np.stack([coords[labels == k].mean(axis=0) for k in range(12)])
            return np.linalg.norm(cen[:, None, :] - cen[None, :, :], axis=-1)

        PCI = {}
        for a in ARCHS:
            co, lab = DATA[a]["npz"]["coords_2d"], DATA[a]["npz"]["labels"]
            PCI[a] = np.stack([per_class_intra(co[t], lab) for t in range(co.shape[0])])   # (E, 12)
            check(f"N4.5.16[{a}.mean_matches_intra_2d]", np.allclose(PCI[a].mean(axis=1), DATA[a]["npz"]["intra_2d"], rtol=1e-3, atol=1e-4),
                  "mean over the 12 per-class variances equals the stored intra_2d series")
        fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
        for ax, a in zip(axes, ARCHS):
            im = ax.imshow(np.log10(PCI[a].T), aspect="auto", origin="lower", cmap="magma", extent=[0.5, PCI[a].shape[0] + 0.5, -0.5, 11.5])
            ax.set_yticks(range(12)); ax.set_yticklabels(CLASSES, fontsize=7); ax.set(xlabel="epoch", title=a)
            fig.colorbar(im, ax=ax, fraction=0.046, label="log10 intra-class variance (2D)")
        fig.suptitle("NB4-9b  Per-class intra-cluster variance in the 2D PCA projection, epoch by epoch (class order assumed = STUDY_LABELS)", y=1.03)
        fig.tight_layout()
        savefig(fig, "NB4-9b_per_class_variance.png")
        last = pd.DataFrame({a: PCI[a][-1] for a in ARCHS}, index=CLASSES)
        display(last.round(2))
        print("hardest class (largest final intra variance):", {a: CLASSES[int(np.argmax(PCI[a][-1]))] for a in ARCHS})
        """),

        md(r"""
        ### How to read this chart

        **Data source:** `coords_2d` and `labels` in `<ARTIFACT_ROOT>/metrics/arch-*.probes.npz`. The label-to-word mapping
        assumes label $i$ is `STUDY_LABELS[i]` from `src/dataset.py`; the probe file itself stores only integers, so the
        word names are an assumption and are stated as one.

        **Axes:** horizontal is epoch; vertical is class (12 rows); colour is $\log_{10}$ of that class's intra-cluster
        variance in the 2D projection (its ten points' mean squared distance to their centroid). Averaging the twelve
        rows of one column reproduces the `intra_2d` series (checked above).

        **Interpretation:** for `gru` every row darkens together within the first few epochs: all classes tighten, none is
        left behind. For `timepool` and `wide` the rows stay bright and differ from one another, so the aggregate
        `intra_2d` of Figure NB4-8 is not driven by one outlier class. Rows that stay brighter than their neighbours in
        the final epoch (the table) name the classes whose points remain most dispersed.

        **Falsifier:** if a single row accounted for nearly all of `wide`'s intra-cluster variance, the aggregate
        instability would be a one-class effect; the rows are comparable in brightness.
        """),

        code(r"""
        # N4.5.17: leave-one-out 1-nearest-neighbour accuracy INSIDE the 2D projection, per epoch, and its relation to validation accuracy.
        def loo_1nn(coords, labels):
            d = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)
            np.fill_diagonal(d, np.inf)
            return float(np.mean(labels[d.argmin(axis=1)] == labels))

        NN1 = {}
        for a in ARCHS:
            co, lab = DATA[a]["npz"]["coords_2d"], DATA[a]["npz"]["labels"]
            NN1[a] = np.array([loo_1nn(co[t], lab) for t in range(co.shape[0])])
        fig, ax = plt.subplots(1, 2, figsize=(13, 4.4))
        rows = []
        for a in ARCHS:
            va = np.array([r["val_accuracy"] for r in DATA[a]["res"]["history"]])
            ax[0].plot(np.arange(1, len(NN1[a]) + 1), NN1[a], color=ACOL[a], label=a)
            ax[1].scatter(va, NN1[a], color=ACOL[a], s=20, label=a)
            rows.append({"arch": a, "1NN_first": NN1[a][0], "1NN_last": NN1[a][-1], "1NN_max": NN1[a].max(), "spearman(1NN, val_acc)": spearmanr(NN1[a], va)[0]})
        ax[0].axhline(1 / 12, color="#667085", ls=":", label="chance 1/12")
        ax[0].set(xlabel="epoch", ylabel="LOO 1-NN accuracy in the 2D projection", ylim=(0, 1), title="probe-set separability in 2D")
        ax[1].set(xlabel="validation accuracy (full split)", ylabel="LOO 1-NN accuracy (120 probe points)", title="2D separability vs validation accuracy")
        ax[0].legend(frameon=False, fontsize=8); ax[1].legend(frameon=False, fontsize=8)
        fig.suptitle("NB4-9c  A classification-style read-out of the 2D PCA projection (an additional measure, not part of SentryCam)", y=1.03)
        fig.tight_layout()
        savefig(fig, "NB4-9c_1nn.png")
        NND = pd.DataFrame(rows).set_index("arch")
        display(NND.round(3))
        check("N4.5.17[gru_ends_high]", NN1["gru"][-1] > 0.6, f"gru's final 2D 1-NN accuracy {NN1['gru'][-1]:.3f} exceeds 0.6")
        check("N4.5.17[order]", NN1["gru"][-1] > NN1["timepool"][-1], "gru's final 2D 1-NN accuracy exceeds timepool's")
        for a in ARCHS:
            check(f"N4.5.17[{a}.above_chance]", NN1[a][-1] > 1 / 12, f"{a}: final 1-NN {NN1[a][-1]:.3f} above chance")
        """),

        md(r"""
        ### How to read this chart

        **Data source:** `coords_2d` and `labels` in `<ARTIFACT_ROOT>/metrics/arch-*.probes.npz` (120 points per epoch);
        `history/val_accuracy` in the result files for the right panel. This is an *additional* measure computed in the
        notebook, not something SentryCam defines.

        **Axes:** left, epoch versus the fraction of probe points whose nearest neighbour (leave-one-out, Euclidean, in the
        2D projection) has the same class; the dotted line is chance. Right, the same score against the full validation
        accuracy of that epoch.

        **Interpretation:** the 2D PCA discards all but two directions, so this score is a lower bound on how separable
        the penultimate representation is. Even so, `gru`'s value climbs well above chance, `timepool`'s stays low, and the
        ordering agrees with validation accuracy; the rank correlation in the table says how closely a single 120-point
        projection tracks the real validation accuracy epoch by epoch.

        **Falsifier:** if the 1-NN score in 2D were unrelated to validation accuracy for `gru` (rank correlation near
        zero), the 2D geometry would not be a useful health signal and Section 5's alert reading would be weakened.
        """),

        code(r"""
        # N4.5.18: final-epoch confusion of the 2D 1-NN classifier (which classes are geometrically confused).
        fig, axes = plt.subplots(1, 3, figsize=(17, 5))
        conf_top = {}
        for ax, a in zip(axes, ARCHS):
            co, lab = DATA[a]["npz"]["coords_2d"][-1], DATA[a]["npz"]["labels"]
            d = np.linalg.norm(co[:, None, :] - co[None, :, :], axis=-1); np.fill_diagonal(d, np.inf)
            pred = lab[d.argmin(axis=1)]
            cm = np.zeros((12, 12), dtype=int)
            for t_, p_ in zip(lab, pred):
                cm[t_, p_] += 1
            im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=10)
            ax.set_xticks(range(12)); ax.set_yticks(range(12)); ax.set_xticklabels(CLASSES, rotation=90, fontsize=7); ax.set_yticklabels(CLASSES, fontsize=7)
            ax.set(xlabel="nearest-neighbour class", ylabel="true class", title=f"{a}: final epoch")
            off = cm.copy(); np.fill_diagonal(off, 0)
            i, j = np.unravel_index(off.argmax(), off.shape)
            conf_top[a] = (CLASSES[i], CLASSES[j], int(off[i, j]))
            check(f"N4.5.18[{a}.rowsum]", (cm.sum(axis=1) == 10).all(), "every class has its 10 probe points in the confusion matrix")
        fig.suptitle("NB4-9d  Leave-one-out 1-NN confusion in the final-epoch 2D projection (label names assumed = STUDY_LABELS order)", y=1.02)
        fig.tight_layout()
        savefig(fig, "NB4-9d_1nn_confusion.png")
        print("most common off-diagonal confusion (true, predicted, count):", conf_top)
        """),

        md(r"""
        ### How to read this chart

        **Data source:** final-epoch `coords_2d` and `labels` in `<ARTIFACT_ROOT>/metrics/arch-*.probes.npz`; word names
        assume label $i$ is `STUDY_LABELS[i]`.

        **Axes:** rows are the true class of a probe point, columns the class of its nearest neighbour in the 2D
        projection; each row sums to $10$. A dark diagonal means same-class points sit next to each other.

        **Interpretation:** `gru` has a strong diagonal with a few isolated confusions; `timepool` and `wide` have
        weaker diagonals and more spread-out off-diagonal mass. The project's known confusable word pairs (for example
        `no`/`on`, `go`/`no`, `up`/`off` in the `LogMelCNN` docstring) can be looked for among the off-diagonal cells, but
        with ten points per class a single cell of two or three is not evidence of a systematic confusion.

        **Falsifier:** off-diagonal mass concentrated in the documented confusable pairs for the weaker models but not for
        `gru` would support the temporal-order story; the printed most frequent confusions are reported without further
        interpretation.
        """),

        code(r"""
        # N4.5.19: how much the between-class geometry changes epoch to epoch (rotation-invariant: uses centroid distance matrices).
        CH = {}
        for a in ARCHS:
            co, lab = DATA[a]["npz"]["coords_2d"], DATA[a]["npz"]["labels"]
            Ds = [centroid_matrix(co[t], lab) for t in range(co.shape[0])]
            CH[a] = np.array([np.linalg.norm(Ds[t] - Ds[t - 1]) / max(np.linalg.norm(Ds[t - 1]), 1e-12) for t in range(1, len(Ds))])
            check(f"N4.5.19[{a}.symmetric]", np.allclose(Ds[-1], Ds[-1].T) and np.allclose(np.diag(Ds[-1]), 0), "centroid distance matrix is symmetric with zero diagonal")
        fig, ax = plt.subplots(figsize=(9, 4.2))
        for a in ARCHS:
            ax.plot(np.arange(2, len(CH[a]) + 2), CH[a], color=ACOL[a], label=a)
        ax.set(xlabel="epoch", ylabel="relative change of the 12x12 centroid-distance matrix", yscale="log", title="NB4-9e  Epoch-to-epoch change of between-class geometry (PCA 2D)")
        ax.legend(frameon=False)
        fig.tight_layout()
        savefig(fig, "NB4-9e_geometry_change.png")
        display(pd.DataFrame({a: {"mean change ep2-10": CH[a][:9].mean(), "mean change last 10": CH[a][-10:].mean()} for a in ARCHS}).T.round(3))
        check("N4.5.19[gru_settles]", CH["gru"][-10:].mean() < CH["gru"][:9].mean(), "gru's geometry changes less in the last 10 epochs than in the first 9")
        """),

        md(r"""
        ### How to read this chart

        **Data source:** `coords_2d` and `labels` in `<ARTIFACT_ROOT>/metrics/arch-*.probes.npz`.

        **Axes:** horizontal is epoch $t$; vertical is $\lVert D_t - D_{t-1}\rVert_F / \lVert D_{t-1}\rVert_F$ on a log
        scale, where $D_t$ is the $12\times12$ matrix of distances between class centroids in the 2D projection at epoch
        $t$. Distances, unlike coordinates, do not depend on the arbitrary rotation or sign of each epoch's PCA, so the
        quantity is comparable across epochs even though the axes are not.

        **Interpretation:** a healthy run should settle (the curve falls and stays low); a run whose class geometry keeps
        being reorganised stays high. This is the same message as the significance band of Figure NB4-8, but on all
        $66$ centroid pairs at once and independent of the two SentryCam scalars.

        **Falsifier:** if `wide`'s curve fell as low as `gru`'s late in training, its unstable validation loss would not be
        reflected in a changing class geometry.
        """),

        code(r"""
        # N4.5.20: where the alerts sit relative to the epoch the saved model comes from (best_epoch = min validation loss).
        rel = []
        for a in ARCHS:
            be = DATA[a]["res"]["best_epoch"]
            rel.append({"arch": a, "best_epoch_idx0": be, "raw_alert_idx0": ALERT[a]["raw"], "2d_alert_idx0": ALERT[a]["2d"],
                        "raw_alert_lead_epochs": be - ALERT[a]["raw"], "2d_alert_lead_epochs": be - ALERT[a]["2d"], "epochs_run": DATA[a]["res"]["epochs_run"]})
        RELD = pd.DataFrame(rel).set_index("arch")
        display(RELD)
        for a in ARCHS:
            check(f"N4.5.20[{a}.alert_before_best]", RELD.loc[a, "2d_alert_idx0"] <= RELD.loc[a, "best_epoch_idx0"], f"2D alert (idx {RELD.loc[a, '2d_alert_idx0']}) precedes or coincides with best_epoch (idx {RELD.loc[a, 'best_epoch_idx0']})")
        note("N4.5.20", "an alert that fires long before the best epoch is not evidence of early detection of a problem: for gru and both other archs the run went on to reach its best loss many epochs later; SentryCam's reported lead over a validation-loss alert (epoch 7 vs 14 on ResNet-34/CIFAR-100) is not reproduced or refuted here")
        """),

        md(r"""
        **Reading the timing table.** For an alert to be useful it should precede a genuine degradation that a
        validation-loss alert would only catch later. Here, the 2D alert fires for `timepool` and `wide` at a very early
        epoch and for `gru` at epoch $17$ (0-based), yet all three continue to train and reach their lowest validation loss
        much later. The alert therefore cannot be scored as "correct early warning" or "false alarm" without a labelled
        instability event in the run, and none exists in these artifacts. The honest statement is descriptive: the *metrics*
        separate the architectures; the *alert rule* is an unreliable summary of that separation. The research file records
        the paper's own case-study comparison (alert at epoch 7 against epoch 14 for validation loss); no comparison of
        that kind is possible with a single run.
        """),
    ]
