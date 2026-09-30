"""NB4 part 3: MM-PHATE statistics (V5-V8), post-training results from the summary block, limitations."""
from nbkit import md, code


def cells():
    return [
        md(r"""
        ## 6. MM-PHATE statistics (V5--V8), post-training

        MM-PHATE (arXiv:2406.01969) is, per the research file, a **post-hoc** analysis tool: it builds a 4-way tensor of
        activations (epochs $\times$ samples $\times$ sequence steps $\times$ units), an intra-/inter-step diffusion
        kernel over the nodes, an MDS embedding, and four summary statistics (intra-step entropy, inter-step entropy,
        signed flow alignment, epoch-to-epoch change magnitude). Its cost is $O((nsm)^2)$ memory and $O((nsm)^3)$ time,
        and the paper states it has no incremental update, so it cannot run live during training. This section therefore
        works on the tensors the probe hook stored after each epoch, and never during training.

        **Deviation, stated up front and repeated in every figure title.** MM-PHATE embeds the multislice kernel with
        diffusion potentials and MDS. We embed the same node matrix with **PCA**. The *statistics* below follow the
        structure of the paper's names; the *embedding they are computed on does not*. The research file records only
        the names of the four statistics, not their exact formulas, so the definitions below are **this notebook's own
        implementation** written for this study, and are not verified to be the paper's
        formulas. Nothing in this section should be quoted as "the MM-PHATE result".

        **Definitions used here.** Index epoch by $\tau$, sequence step by $\omega$, unit by $i$. For each $(\tau,\omega)$,
        the $32$ units are treated as points, each described by its $120$ probe-sample responses, and projected to 2D by
        PCA, giving $y_{\tau,\omega,i}\in\mathbb{R}^2$. With $\hat H_{\mathrm{KDE}}(S) = -\frac{1}{|S|}\sum_{y\in S}\log \hat p(y)$
        the Gaussian-KDE differential-entropy estimate of a point set $S$:

        - intra-step entropy $H^{\mathrm{intra}}_{\tau,\omega} = \hat H_{\mathrm{KDE}}\big(\{y_{\tau,\omega,i}\}_i\big)$ (V5),
        - inter-step entropy $H^{\mathrm{inter}}_{\tau,i} = \hat H_{\mathrm{KDE}}\big(\{y_{\tau,\omega,i}\}_\omega\big)$ (V6),
        - flow alignment $A_\tau = \big\lVert \tfrac{1}{W-1}\sum_{\omega}\hat u_{\tau,\omega}\big\rVert_2$, where
          $\hat u_{\tau,\omega}$ is the unit vector of the displacement of the unit-centroid from step $\omega$ to $\omega+1$ (V7).

        **A correction to how V7 is described.** $A_\tau$ as implemented is the *magnitude* of the mean unit direction,
        so it lies in $[0,1]$ and is **unsigned**. Calling it a "signed flow alignment" would be wrong for this
        implementation; the paper's signed quantity is not what is plotted. Because each $(\tau,\omega)$ has its own PCA,
        orientation is arbitrary between steps, which is a second reason the magnitude, not the sign, is the only
        defensible reading. Entropies are also not scale-free: differential entropy shifts by $\log$ of the scale, so
        absolute values across epochs partly reflect the changing spread of the activations.
        """),

        code(r"""
        # N4.6.1: MM-PHATE statistics (kde_entropy, pca_embed, mmphate_stats), implemented for this study.
        def kde_entropy(points):
            # Differential entropy via KDE: H = -mean(log p(x_i)). Returns nan if degenerate.
            p = np.asarray(points, dtype=np.float64)
            if p.ndim == 1:
                p = p[:, None]
            if p.shape[0] < 3:
                return np.nan
            p = p + np.random.default_rng(0).normal(0, 1e-9, p.shape)  # break exact duplicates
            try:
                kde = gaussian_kde(p.T)
                return float(-np.mean(np.log(np.clip(kde(p.T), 1e-300, None))))
            except Exception:
                return np.nan

        def pca_embed(x, k=2):
            x = np.asarray(x, dtype=np.float64)
            c = x - x.mean(axis=0, keepdims=True)
            _, _, vt = np.linalg.svd(c, full_matrices=False)
            return c @ vt[:k].T

        def mmphate_stats(tensor):
            # tensor (epochs, samples, steps, units) -> intra-step H(tau, omega), inter-step H(tau, i), flow A_tau
            n_ep, n_s, n_w, n_u = tensor.shape
            intra = np.full((n_ep, n_w), np.nan); inter = np.full((n_ep, n_u), np.nan); flow = np.full(n_ep, np.nan)
            for t in range(n_ep):
                emb = np.stack([pca_embed(tensor[t, :, w, :].T, 2) for w in range(n_w)])  # (steps, units, 2)
                for w in range(n_w):
                    intra[t, w] = kde_entropy(emb[w])
                for i in range(n_u):
                    inter[t, i] = kde_entropy(emb[:, i, :])
                cent = emb.mean(axis=1)
                d = np.diff(cent, axis=0)
                nrm = np.linalg.norm(d, axis=1, keepdims=True)
                u = d / np.where(nrm > 0, nrm, 1)
                flow[t] = float(np.linalg.norm(u.mean(axis=0))) if len(u) else np.nan
            return intra, inter, flow

        import time
        MM = {}
        for a in ARCHS:
            t0 = time.time()
            MM[a] = mmphate_stats(DATA[a]["npz"]["mmphate_tensor"])
            print(f"{a}: tensor {DATA[a]['npz']['mmphate_tensor'].shape} -> stats in {time.time() - t0:.1f}s; intra {MM[a][0].shape}, inter {MM[a][1].shape}, flow {MM[a][2].shape}")
            check(f"N4.6.1[{a}.finite]", all(np.isfinite(x).all() for x in MM[a]), "no NaN in intra/inter/flow")
            check(f"N4.6.1[{a}.flow_range]", np.all((MM[a][2] >= 0) & (MM[a][2] <= 1 + 1e-9)), "flow alignment lies in [0, 1] (magnitude of a mean of unit vectors)")
        """),

        code(r"""
        # N4.6.2: compare with the precomputed mmphate_stats.npz (if present on this host) and check the tensor's per-unit standardisation.
        pre = MET / "mmphate_stats.npz"
        if pre.exists():
            st = np.load(pre)
            for a in ARCHS:
                for k, arr in zip(("intra", "inter", "flow"), MM[a]):
                    check(f"N4.6.2[{a}.{k}_vs_npz]", np.allclose(arr, st[f"{a}_{k}"], equal_nan=True), f"recomputed {k} equals precomputed npz array {st[f'{a}_{k}'].shape}")
        else:
            note("N4.6.2", f"mmphate_stats.npz not present under {MET}; using the notebook-recomputed statistics only")
        for a in ARCHS:
            T = DATA[a]["npz"]["mmphate_tensor"]
            m = np.abs(T.mean(axis=(1, 2))).max(); s = T.std(axis=(1, 2)).mean()
            print(f"{a}: max |mean| per (epoch, unit) = {m:.4f}; mean std = {s:.4f}")
            check(f"N4.6.2[{a}.zscored]", m < 0.1 and abs(s - 1) < 0.05, "tensor is per-unit z-scored over (samples, steps) as the probe hook documents")
        """),

        md(r"""
        ### 6.1 Subsampling actually used and the cost model

        The recorded tensor for each architecture is $(E, 120, \omega, 32)$: $E$ epochs, $120$ probe samples,
        $\omega$ sequence steps ($20$ for `gru`, $8$ for `timepool` and `wide`), and $32$ units per step. Two subsamplings
        are already baked into these arrays: the **probe set is 120 samples** (12 classes $\times$ 10) rather than the
        validation set, and the **unit axis is 32**, although the `ConvGRU` bidirectional output has $2\times 64 = 128$
        features per step and `WideCNN`'s final feature map has $128$ channels. The rule that reduced $128 \to 32$ is not
        documented in the artifacts available here (the worker script on Drive would say), so it is reported as an
        unknown rather than guessed. `timepool`'s $32$ channels are its full width.

        Node counts and the resulting kernel cost depend on what is counted as a node. Two readings are tabulated, because
        the research file states the cost as $O((nsm)^2)$ memory and $O((nsm)^3)$ time without defining $n$, $s$, $m$
        in the text available here: (A) the nodes this notebook actually embeds, $E\times\omega\times32$ (epoch, step, unit);
        (B) the product $E\times120\times\omega$ used in this notebook's planning (epochs $\times$ probe samples $\times$ steps).
        Memory is for a dense float64 kernel, $N^2\times 8$ bytes; time is reported as the raw $N^3$ operation count
        (no constant factor claimed).
        """),

        code(r"""
        # N4.6.3: cost model with the real numbers.
        crow = []
        for a in ARCHS:
            E, S, W, U = DATA[a]["npz"]["mmphate_tensor"].shape
            for label, N in (("A: epoch x step x unit (embedded here)", E * W * U), ("B: epoch x sample x step (planning-note reading)", E * S * W)):
                crow.append({"arch": a, "E": E, "samples": S, "steps": W, "units": U, "reading": label, "nodes N": N,
                             "kernel entries N^2": N ** 2, "dense float64 GB": N ** 2 * 8 / 1e9, "N^3 (x1e12)": N ** 3 / 1e12})
        COST = pd.DataFrame(crow)
        pd.set_option("display.float_format", lambda v: f"{v:,.3f}")
        display(COST)
        pd.reset_option("display.float_format")
        by = COST.set_index(["arch", "reading"])["nodes N"]
        check("N4.6.3[gru_brief_count]", by.loc[("gru", "B: epoch x sample x step (planning-note reading)")] == 28 * 120 * 20, f"gru (B) = 28 x 120 x 20 = {28 * 120 * 20:,}")
        check("N4.6.3[tp_brief_count]", by.loc[("timepool", "B: epoch x sample x step (planning-note reading)")] == 36 * 120 * 8, f"timepool (B) = 36 x 120 x 8 = {36 * 120 * 8:,}")
        check("N4.6.3[wide_brief_count]", by.loc[("wide", "B: epoch x sample x step (planning-note reading)")] == 36 * 120 * 8, f"wide (B) = 36 x 120 x 8 = {36 * 120 * 8:,}")
        check("N4.6.3[embedded_gru]", by.loc[("gru", "A: epoch x step x unit (embedded here)")] == 28 * 20 * 32, "gru (A) = 28 x 20 x 32 = 17,920 nodes embedded")
        print("Reading A: PCA of a 32x120 matrix per (epoch, step) is trivial; the full O(N^3) kernel + MDS at N=17,920 would be ~5.8e12 operations and ~2.6 GB of kernel memory,")
        print("Reading B would need ~36 GB for gru's kernel: the reason a PCA embedding replaces the diffusion+MDS pipeline in this notebook.")
        """),

        md(r"""
        Reading (A) is what this notebook embeds and what the PCA runs on. Under reading (B), `gru` would need a dense
        kernel of about $36$ GB and `timepool`/`wide` about $9.6$ GB, which is the practical reason the
        diffusion-plus-MDS embedding is not attempted here and PCA is substituted. The numbers are produced from the
        array shapes, not asserted.
        """),

        md(r"""
        ### 6.2 V5: intra-step entropy over epoch $\times$ sequence step
        """),

        code(r"""
        # N4.6.4: V5 heatmaps.
        fig, axes = plt.subplots(1, 3, figsize=(16, 4.4))
        for ax, a in zip(axes, ARCHS):
            intra = MM[a][0]
            im = ax.imshow(intra.T, aspect="auto", origin="lower", cmap="magma", extent=[0.5, intra.shape[0] + 0.5, 0.5, intra.shape[1] + 0.5])
            ax.set(xlabel="epoch", ylabel="sequence step", title=a); fig.colorbar(im, ax=ax, fraction=0.046, label="H_KDE (nats)")
        fig.suptitle("NB4-10  V5: intra-step entropy per (epoch, sequence step) on a PCA embedding (NOT MM-PHATE's diffusion+MDS)", y=1.03)
        fig.tight_layout()
        savefig(fig, "NB4-10_V5_intra_entropy.png")
        for a in ARCHS:
            m = MM[a][0].mean(axis=1)
            print(f"{a}: mean intra-step entropy epoch 1 = {m[0]:.3f}, last epoch = {m[-1]:.3f}, min = {m.min():.3f} at epoch {int(m.argmin()) + 1}")
        """),

        md(r"""
        ### How to read this chart

        **Data source:** `mmphate_tensor` in `<ARTIFACT_ROOT>/metrics/arch-*.probes.npz`, passed through the notebook's copy of
        `mmphate_stats`; equals the precomputed `mmphate_stats.npz` when that file is present.

        **Axes:** horizontal is epoch, vertical is sequence step ($20$ rows for `gru`, $8$ for the others), colour is
        $H^{\mathrm{intra}}_{\tau,\omega}$ in nats (brighter = the 32 units are more spread out in the 2D embedding at that
        step).

        **Interpretation:** in the processing-versus-compression picture, high early entropy corresponds to units carrying
        diverse, uncommitted responses and a later decline to units becoming redundant (compression). Here the decline is
        real but modest: mean entropy per epoch moves only from about $6.0$ to about $5.5$ nats for `gru`, from about $6.0$
        to $5.6$ for `timepool`, and from about $5.9$ to $5.1$ for `wide` (exact values printed above). The step-to-step
        structure within an epoch is at least as large as the epoch-to-epoch drift, so most of the picture is *which step*,
        not *which epoch*. This is a weak signal, not a phase diagram.

        **Falsifier:** if entropy were flat in epoch for every step, there would be no compression trend to report;
        the printed epoch-1-to-last change is small but non-zero. A reading that treats the colour gradient as a
        demonstrated processing/compression transition is **not** supported: the embedding is PCA, the probe set is 120
        samples, and there is one run.
        """),

        md(r"""
        ### 6.3 V6: inter-step entropy per unit across epochs
        """),

        code(r"""
        # N4.6.5: V6 heatmaps plus the across-unit mean and interquartile band.
        fig, axes = plt.subplots(2, 3, figsize=(16, 8))
        for j, a in enumerate(ARCHS):
            inter = MM[a][1]; ep = np.arange(1, inter.shape[0] + 1)
            im = axes[0, j].imshow(inter.T, aspect="auto", origin="lower", cmap="viridis", extent=[0.5, inter.shape[0] + 0.5, 0.5, inter.shape[1] + 0.5])
            axes[0, j].set(xlabel="epoch", ylabel="unit", title=f"{a}: per-unit inter-step entropy"); fig.colorbar(im, ax=axes[0, j], fraction=0.046)
            q1, med, q3 = np.percentile(inter, [25, 50, 75], axis=1)
            axes[1, j].plot(ep, inter.mean(axis=1), color=ACOL[a], label="mean over units")
            axes[1, j].plot(ep, med, color="black", lw=0.8, ls="--", label="median")
            axes[1, j].fill_between(ep, q1, q3, color=ACOL[a], alpha=0.25, label="IQR over units")
            axes[1, j].set(xlabel="epoch", ylabel="H_KDE (nats)"); axes[1, j].legend(frameon=False, fontsize=8)
        fig.suptitle("NB4-11  V6: inter-step entropy per unit (PCA embedding, NOT MM-PHATE's diffusion+MDS)", y=1.0)
        fig.tight_layout()
        savefig(fig, "NB4-11_V6_inter_entropy.png")
        for a in ARCHS:
            m = MM[a][1].mean(axis=1)
            print(f"{a}: mean inter-step entropy epoch 1 = {m[0]:.3f}, max = {m.max():.3f} at epoch {int(m.argmax()) + 1}, last = {m[-1]:.3f}")
        """),

        md(r"""
        ### How to read this chart

        **Data source:** the same tensors, statistic $H^{\mathrm{inter}}_{\tau,i}$, one value per (epoch, unit).

        **Axes:** top row: horizontal is epoch, vertical is the unit index ($32$ units), colour the entropy of that unit's
        2D trajectory across the sequence steps. Bottom row: the mean across units (coloured), the median (dashed), and the
        interquartile band as a function of epoch.

        **Interpretation:** a low value means a unit's embedded position is concentrated across steps (it responds the same
        way at every time step); a high value means its position moves around across steps (temporal selectivity). The
        measured pattern is **not a collapse**: the mean rises early (for `gru` from about $4.2$ at epoch 1 to about $5.3$
        within a few epochs; for `timepool` from about $3.8$ to above $5$) and then plateaus or eases slightly. The
        per-unit heterogeneity (the band and the horizontal stripes in the heatmap) is large compared with the epoch
        trend. The expected "temporal selectivity collapse" is therefore not observed in these runs; if it exists at other
        scales or settings it is not shown here.

        **Falsifier:** if a sustained decline in mean inter-step entropy appeared in the late epochs for the healthy
        `gru` and not for the others, it would support a selectivity-collapse reading. The printed maxima and last
        values show only a small decline after the peak.
        """),

        md(r"""
        ### 6.4 V7: flow alignment across epochs
        """),

        code(r"""
        # N4.6.6: V7 curves. A_tau is the magnitude of the mean unit displacement of the unit-centroid across steps (unsigned).
        fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
        for ax, a in zip(axes, ARCHS):
            flow = MM[a][2]
            ax.plot(np.arange(1, len(flow) + 1), flow, color=ACOL[a], marker="o", ms=3)
            ax.axhline(flow.mean(), color="#667085", ls=":", label=f"mean {flow.mean():.2f}")
            ax.set(xlabel="epoch", ylim=(0, 1), title=a); ax.legend(frameon=False, fontsize=8)
        axes[0].set_ylabel(r"$A_\tau$ = |mean unit step displacement|")
        fig.suptitle("NB4-12  V7: flow alignment (unsigned magnitude, PCA embedding; NOT MM-PHATE's signed statistic)", y=1.03)
        fig.tight_layout()
        savefig(fig, "NB4-12_V7_flow_alignment.png")
        for a in ARCHS:
            f = MM[a][2]
            print(f"{a}: A_tau mean {f.mean():.3f}, std {f.std():.3f}, range {f.min():.3f} .. {f.max():.3f}; rho(A_tau, epoch) = {spearmanr(np.arange(len(f)), f)[0]:+.3f}")
        """),

        md(r"""
        ### How to read this chart

        **Data source:** the same tensors, statistic $A_\tau$.

        **Axes:** horizontal is epoch; vertical is $A_\tau\in[0,1]$. A value near $1$ means the unit-centroid moves in
        nearly the same direction at every step of the sequence (a coherent drift through the embedding); near $0$ means
        the step-to-step displacements point in varying directions and cancel.

        **Interpretation:** all three curves sit low (typically $0.03$ to $0.3$) and jump around from epoch to epoch,
        with no monotone trend (the printed Spearman coefficients against epoch are the test). Because every $(\tau,\omega)$
        has an independent PCA whose orientation is arbitrary, consecutive-step displacements are not expressed in a common
        frame, and the low values are what randomly oriented frames would produce. The statistic is therefore close to
        uninformative in this implementation; it is shown for completeness and labelled as such, not as evidence of a flow
        regime.

        **Falsifier:** a genuine flow structure would appear as $A_\tau$ near $1$ for some epochs or a systematic
        trend; neither occurs. A version computed in a fixed, shared frame (for example the diffusion-plus-MDS embedding
        the paper uses) might differ, which cannot be tested with the PCA embedding here.
        """),

        md(r"""
        ### 6.5 V8: multiway node embedding

        Every node $(\tau,\omega,i)$ is a $120$-dimensional vector (that unit's responses over the probe set at that
        epoch and step). All nodes of an architecture are stacked and embedded together by **one** PCA to 2D, then
        drawn twice: coloured by epoch and coloured by sequence step. Node counts are $28\times20\times32=17{,}920$ for
        `gru` and $36\times8\times32=9{,}216$ for `timepool` and `wide` (reading A in Section 6.1). **Deviation:** PCA, not
        diffusion + MDS.
        """),

        code(r"""
        # N4.6.7: V8 for gru (two colourings).
        def v8_nodes(arch):
            T = DATA[arch]["npz"]["mmphate_tensor"]; E, S, W, U = T.shape
            nodes, ep_id, w_id = [], [], []
            for t in range(E):
                for w in range(W):
                    nodes.append(T[t, :, w, :].T); ep_id += [t] * U; w_id += [w] * U
            emb = pca_embed(np.concatenate(nodes), 2)
            return emb, np.array(ep_id), np.array(w_id)

        def v8_figure(arch, fname, fig_no):
            emb, ep_id, w_id = v8_nodes(arch)
            fig, ax = plt.subplots(1, 2, figsize=(13, 5))
            for k, (c, lbl, cm) in enumerate([(ep_id + 1, "epoch", "viridis"), (w_id + 1, "sequence step", "plasma")]):
                s = ax[k].scatter(emb[:, 0], emb[:, 1], c=c, cmap=cm, s=4, alpha=0.6)
                ax[k].set(title=f"{arch}: coloured by {lbl}", xlabel="PC1", ylabel="PC2"); fig.colorbar(s, ax=ax[k], label=lbl)
            fig.suptitle(f"NB4-{fig_no}  V8: node = (epoch, step, unit); {len(emb):,} nodes; PCA embedding (deviation from MM-PHATE's diffusion+MDS)", y=1.02)
            fig.tight_layout()
            savefig(fig, fname)
            return emb, ep_id, w_id

        emb_g, ep_g, w_g = v8_figure("gru", "NB4-13_V8_multiway_gru.png", 13)
        check("N4.6.7[gru_nodes]", len(emb_g) == 28 * 20 * 32, f"{len(emb_g):,} nodes for gru")
        rho_ep = spearmanr(ep_g, emb_g[:, 0])[0], spearmanr(ep_g, emb_g[:, 1])[0]
        rho_w = spearmanr(w_g, emb_g[:, 0])[0], spearmanr(w_g, emb_g[:, 1])[0]
        print(f"gru: rank correlation of PC1/PC2 with epoch = {rho_ep[0]:+.3f}/{rho_ep[1]:+.3f}; with step = {rho_w[0]:+.3f}/{rho_w[1]:+.3f}")
        """),

        md(r"""
        ### How to read this chart

        **Data source:** `mmphate_tensor` in `<ARTIFACT_ROOT>/metrics/arch-gru.probes.npz` (shape $(28,120,20,32)$).

        **Axes:** each point is one (epoch, step, unit) node; the two axes are the first two principal components of the
        stacked $120$-dimensional node vectors. Left panel colours nodes by epoch, right panel by sequence step.

        **Interpretation:** if training reorganised the units, nodes from early and late epochs would occupy different
        regions (left panel would show a colour gradient), and if units were step-specific, the right panel would show
        step-ordered structure. The printed rank correlations of PC1 and PC2 with epoch and with step quantify how much of
        that structure a 2D PCA can see; the scatter itself is dominated by the arrangement of units, and the
        colour-by-epoch panel is mixed rather than a clean gradient. Because the tensor is per-unit z-scored, a large part of
        the cloud's shape is the standardisation.

        **Falsifier:** a strong monotone epoch (or step) gradient across the cloud would mean training or position dominates
        the variance; correlations near zero, as here, mean neither dominates in the leading two components.
        """),

        code(r"""
        # N4.6.8: V8 for timepool and wide.
        emb_t, ep_t, w_t = v8_figure("timepool", "NB4-14a_V8_multiway_timepool.png", "14a")
        check("N4.6.8[tp_nodes]", len(emb_t) == 36 * 8 * 32, f"{len(emb_t):,} nodes for timepool")
        """),

        code(r"""
        # N4.6.9: V8 wide.
        emb_w, ep_w, w_w = v8_figure("wide", "NB4-14b_V8_multiway_wide.png", "14b")
        check("N4.6.9[wide_nodes]", len(emb_w) == 36 * 8 * 32, f"{len(emb_w):,} nodes for wide")
        for a, (e, ep, w) in (("timepool", (emb_t, ep_t, w_t)), ("wide", (emb_w, ep_w, w_w))):
            print(f"{a}: rank corr PC1/PC2 with epoch = {spearmanr(ep, e[:, 0])[0]:+.3f}/{spearmanr(ep, e[:, 1])[0]:+.3f}; with step = {spearmanr(w, e[:, 0])[0]:+.3f}/{spearmanr(w, e[:, 1])[0]:+.3f}")
        """),

        md(r"""
        ### How to read these two charts (`timepool`, `wide`)

        **Data source:** `mmphate_tensor` in `arch-timepool.probes.npz` and `arch-wide.probes.npz`, shape
        $(36,120,8,32)$ each; $9{,}216$ nodes per figure.

        **Axes and colouring:** identical to the `gru` figure.

        **Interpretation:** with only $8$ sequence steps the step colouring has fewer levels, and the two architectures
        differ from `gru` mainly in how the units of one step spread; the printed correlations again say how much epoch or
        step structure the first two components carry. As with `gru`, none of this is evidence about MM-PHATE's own
        findings, since the embedding is different.

        **Falsifier:** if `wide` (whose validation loss oscillates) showed a much stronger epoch gradient than `gru`, that
        would suggest the instability reorganises the representation; the printed values are the test and are reported as
        computed.
        """),

        md(r"""
        ### 6.6 Further MM-PHATE-style read-outs: epoch-to-epoch change and the step profile

        The research file lists four MM-PHATE statistics: intra-step entropy, inter-step entropy, signed flow alignment,
        and epoch-to-epoch change magnitude. Sections 6.2 to 6.4 covered the first three. The fourth is added here in the
        same spirit as the others: **this notebook's own implementation** on the PCA embedding, defined as the Euclidean
        norm of the change of the epoch's intra-step entropy vector (one entry per sequence step) between consecutive
        epochs, $\Delta_\tau = \lVert H^{\mathrm{intra}}_{\tau,\cdot} - H^{\mathrm{intra}}_{\tau-1,\cdot}\rVert_2$. It is
        not claimed to be the paper's definition. The step profile (mean over epochs of $H^{\mathrm{intra}}_{\tau,\omega}$ as
        a function of $\omega$) is also shown, since the heatmaps suggested the step dimension carries more variation than
        the epoch dimension.
        """),

        code(r"""
        # N4.6.10: epoch-to-epoch change magnitude and the step profile of intra-step entropy.
        fig, axes = plt.subplots(1, 2, figsize=(14, 4.2))
        DELTA, PROF = {}, {}
        for a in ARCHS:
            intra = MM[a][0]
            DELTA[a] = np.linalg.norm(np.diff(intra, axis=0), axis=1)
            PROF[a] = intra.mean(axis=0)
            axes[0].plot(np.arange(2, intra.shape[0] + 1), DELTA[a], color=ACOL[a], label=a)
            axes[1].plot(np.arange(1, intra.shape[1] + 1), PROF[a], color=ACOL[a], marker="o", ms=3, label=a)
            check(f"N4.6.10[{a}.delta_len]", len(DELTA[a]) == intra.shape[0] - 1, f"{len(DELTA[a])} epoch-to-epoch changes for {intra.shape[0]} epochs")
        axes[0].set(xlabel="epoch", ylabel=r"$\|H_\tau - H_{\tau-1}\|_2$ (nats)", title="epoch-to-epoch change of intra-step entropy")
        axes[1].set(xlabel="sequence step", ylabel="mean H_intra over epochs (nats)", title="step profile of intra-step entropy")
        for x in axes: x.legend(frameon=False, fontsize=8)
        fig.suptitle("NB4-12b  MM-PHATE-style change magnitude and step profile (PCA embedding; own implementation, not the paper's definition)", y=1.03)
        fig.tight_layout()
        savefig(fig, "NB4-12b_change_and_profile.png")
        tab = pd.DataFrame({a: {"mean change ep2-6": DELTA[a][:5].mean(), "mean change last 5": DELTA[a][-5:].mean(),
                                "step-profile range": PROF[a].max() - PROF[a].min(),
                                "epoch-trend range": MM[a][0].mean(axis=1).max() - MM[a][0].mean(axis=1).min()} for a in ARCHS}).T
        display(tab.round(3))
        for a in ARCHS:
            check(f"N4.6.10[{a}.change_settles]", tab.loc[a, "mean change last 5"] < tab.loc[a, "mean change ep2-6"], "change magnitude is smaller in the last five epochs than in epochs 2-6")
        """),

        md(r"""
        ### How to read this chart

        **Data source:** the `intra` array of the MM-PHATE statistics computed in Section 6 from `mmphate_tensor` in
        `<ARTIFACT_ROOT>/metrics/arch-*.probes.npz`.

        **Axes:** left, epoch versus the L2 norm (over sequence steps) of the change of the intra-step entropy vector
        from the previous epoch, in nats. Right, sequence step versus the intra-step entropy averaged over all epochs.

        **Interpretation:** the left panel says how quickly the entropy landscape stops changing: a decaying curve is the
        MM-PHATE-style signature of a representation that has stopped reorganising. The right panel is descriptive: its
        range across steps is compared in the table with the range across epochs, and it shows that structure along the
        sequence axis (for example edge steps of the recurrent or pooled sequence differing from interior steps) is at least
        as large as structure along the epoch axis.

        **Falsifier:** if the change magnitude did not decrease for the healthy `gru` while validation loss had
        stabilised, the entropy statistic would not reflect convergence. The table compares early and late means.
        """),

        md(r"""
        ## 7. Post-training results from the `summary` block

        > **EXPLORATION CONFIGURATION.** Every number in this section comes from the `summary` block of a result file
        > produced with **4 stress conditions** (`clean`, `gain_+10`, `noise_10`, `reverb_mid`) and **200 bootstrap
        > resamples**, for a single trained model per architecture (clean training, seed 17). These are exploration
        > results. They are not publication numbers: the full 11-condition / 2000-resample evaluation was never run.
        > Bootstrap intervals from 200 resamples are noisy at the third decimal.

        Each architecture's `summary` holds, per stress condition and per calibration mode
        (`uncalibrated`, `calibrated` with the temperature fitted on the calibration split), accuracy, macro-F1, NLL,
        Brier, ECE at 10/15/20 bins, AURC, selective risk at coverage $0.8$ and $0.9$, OOD AUROC, CRC results at
        $\alpha\in\{0.05,0.10\}$ (threshold, miscoverage, mean set size), and bootstrap intervals for four metrics
        (accuracy, ECE with 10 bins, NLL, AURC).

        **What is *not* available here, and therefore not plotted.** The result files hold summary numbers, not logits.
        NB3's reliability diagram and risk-coverage curve are built from per-example logits; those cannot be rebuilt from
        these artifacts. Instead this section presents the reliability-style information that *is* stored (ECE at three bin
        counts, with and without temperature scaling) and the risk-coverage information that is stored (selective risk at two
        coverages and AURC). This mirrors NB3's plotting conventions (grouped uncalibrated/calibrated bars, annotated
        values, "How to read this chart" blocks) without inventing data.
        """),

        code(r"""
        # N4.7.1: flatten the summary block of the three result files into one long table.
        CONDS = DATA["gru"]["res"]["summary"]["conditions"]
        COND_LIST = list(CONDS)
        MODES = ("uncalibrated", "calibrated")
        SCALARS = ["accuracy", "macro_f1", "nll", "brier", "ece_10", "ece_15", "ece_20", "aurc", "selective_risk_0.8", "selective_risk_0.9", "ood_auroc"]
        rows = []
        for a in ARCHS:
            for c in COND_LIST:
                for m in MODES:
                    d = DATA[a]["res"]["summary"]["conditions"][c][m]
                    row = {"arch": a, "condition": c, "mode": m}
                    row.update({k: d[k] for k in SCALARS})
                    for al in ("0.05", "0.1"):
                        row[f"crc{al}_threshold"] = d["crc"][al]["threshold"]
                        row[f"crc{al}_miscoverage"] = d["crc"][al]["miscoverage"]
                        row[f"crc{al}_set_size"] = d["crc"][al]["mean_set_size"]
                    for k, v in d["ci"].items():
                        row[f"{k}_lo"], row[f"{k}_hi"] = v["lower"], v["upper"]
                    rows.append(row)
        SUM = pd.DataFrame(rows)
        display(SUM[["arch", "condition", "mode", "accuracy", "macro_f1", "nll", "brier", "ece_10", "aurc", "ood_auroc"]].round(4))
        check("N4.7.1[rows]", len(SUM) == 3 * 4 * 2, f"{len(SUM)} rows = 3 archs x 4 conditions x 2 modes")
        check("N4.7.1[conditions]", COND_LIST == ["clean", "gain_+10", "noise_10", "reverb_mid"], f"conditions {COND_LIST}")
        """),

        code(r"""
        # N4.7.2: internal-consistency checks on the summary block (each is an individual PASS/FAIL).
        for a in ARCHS:
            for c in COND_LIST:
                u = SUM[(SUM.arch == a) & (SUM.condition == c) & (SUM["mode"] == "uncalibrated")].iloc[0]
                k = SUM[(SUM.arch == a) & (SUM.condition == c) & (SUM["mode"] == "calibrated")].iloc[0]
                check(f"N4.7.2[{a}.{c}.argmax_invariance]", abs(u["accuracy"] - k["accuracy"]) < 1e-12, "temperature scaling leaves accuracy unchanged")
                for m_name, r in (("uncal", u), ("cal", k)):
                    ok = all(r[f"{q}_lo"] <= r[q] <= r[f"{q}_hi"] for q in ("accuracy", "ece_10", "nll", "aurc"))
                    check(f"N4.7.2[{a}.{c}.{m_name}.ci_contains_point]", ok, "bootstrap interval contains the point estimate for accuracy, ece_10, nll and aurc")
        """),

        md(r"""
        ### 7.1 Accuracy and its bootstrap interval under stress

        The headline of the post-training results is a robustness gap. The trained models were fitted on clean audio
        only. Under `gain_+10` and `reverb_mid` the drop is moderate, but under `noise_10` accuracy falls **to or near
        chance** for the two best clean models. The next figure shows accuracy with the recorded $95\%$-style bootstrap
        intervals (the interval level is whatever `summarise_arm` used; only the lower and upper values are stored).
        """),

        code(r"""
        # N4.7.3: accuracy with bootstrap intervals, three archs x four conditions, uncalibrated.
        fig, ax = plt.subplots(figsize=(11, 4.6)); w = 0.26
        for j, a in enumerate(ARCHS):
            d = SUM[(SUM.arch == a) & (SUM["mode"] == "uncalibrated")].set_index("condition").loc[COND_LIST]
            x = np.arange(len(COND_LIST)) + (j - 1) * w
            ax.bar(x, d["accuracy"], w, color=ACOL[a], label=a, alpha=0.9,
                   yerr=[d["accuracy"] - d["accuracy_lo"], d["accuracy_hi"] - d["accuracy"]], capsize=3)
            for xi, v in zip(x, d["accuracy"]):
                ax.annotate(f"{v:.2f}", (xi, v), xytext=(0, 4), textcoords="offset points", ha="center", fontsize=7.5)
        ax.axhline(1 / 12, color="#667085", ls=":", label="chance 1/12")
        ax.set_xticks(np.arange(len(COND_LIST))); ax.set_xticklabels(COND_LIST)
        ax.set(ylabel="test accuracy", ylim=(0, 1.05), title="NB4-15  Test accuracy under four stress conditions (EXPLORATION: 200 bootstrap resamples), clean-trained, seed 17")
        ax.legend(frameon=False)
        fig.tight_layout()
        savefig(fig, "NB4-15_accuracy_ci.png")
        print(SUM[(SUM["mode"] == "uncalibrated")].pivot(index="condition", columns="arch", values="accuracy").round(3))
        """),

        md(r"""
        ### How to read this chart

        **Data source:** `summary/conditions/<cond>/uncalibrated/accuracy` and `summary/.../ci/accuracy/{lower,upper}` in
        `arch-*.result.json`. Calibrated accuracy is identical by construction (argmax invariance, checked above).

        **Axes:** horizontal is the stress condition; vertical is accuracy on the $1080$-clip test split. Bars are
        coloured by architecture, error bars are the recorded bootstrap interval (200 resamples), the dotted line is chance
        ($1/12$).

        **Interpretation:** on `clean`, `gru` ($0.88$) and `wide` ($0.86$) are far above `timepool` ($0.55$). `gain_+10`
        and `reverb_mid` cost `gru` about $5$ points and `wide` up to about $10$, and `timepool` more in relative terms.
        `noise_10` is different in kind: `gru` falls to about $0.14$, `timepool` to about $0.15$, `wide` to about $0.25$, all
        close to chance for `gru` and `timepool`. The most accurate clean model is therefore the most fragile to additive
        noise in this run. Because it is a single seed with a clean-only training condition, the finding is that
        *clean training does not transfer to noise here*, not that `gru` is inherently fragile.

        **Falsifier:** if error bars overlapped between `gru` and `wide` on `clean`, their ordering would not be
        established; the intervals are narrow (a few points). If the `noise_10` bars were near their `clean` height, the
        stress had no effect; the gap is the finding.
        """),

        md(r"""
        ### 7.2 Calibration: ECE at three bin counts, before and after temperature scaling
        """),

        code(r"""
        # N4.7.4: ECE (10/15/20 bins) uncalibrated vs calibrated, per arch, four conditions.
        fig, axes = plt.subplots(1, 3, figsize=(16, 4.4), sharey=True); w = 0.13
        for ax, a in zip(axes, ARCHS):
            for bi, bins in enumerate((10, 15, 20)):
                for mi, m in enumerate(MODES):
                    d = SUM[(SUM.arch == a) & (SUM["mode"] == m)].set_index("condition").loc[COND_LIST]
                    x = np.arange(len(COND_LIST)) + (bi * 2 + mi - 2.5) * w
                    ax.bar(x, d[f"ece_{bins}"], w, color=("#1f77b4" if m == "uncalibrated" else "#2ca02c"), alpha=0.55 + 0.15 * bi,
                           label=f"{bins} bins, {m}" if a == ARCHS[0] else None)
            ax.set_xticks(np.arange(len(COND_LIST))); ax.set_xticklabels(COND_LIST, fontsize=8); ax.set(title=a, yscale="log")
        axes[0].set_ylabel("ECE (log scale)"); axes[0].legend(frameon=False, fontsize=7, ncol=2)
        fig.suptitle("NB4-16  ECE at 10/15/20 bins, uncalibrated (blue) vs temperature-scaled (green); EXPLORATION config", y=1.03)
        fig.tight_layout()
        savefig(fig, "NB4-16_ece_bins.png")
        t = SUM.pivot_table(index=["arch", "condition"], columns="mode", values="ece_10")
        t["ratio cal/uncal"] = t["calibrated"] / t["uncalibrated"]
        display(t.round(4))
        print("fitted temperatures:", {a: round(DATA[a]["res"]["summary"]["temperature"], 3) for a in ARCHS})
        check("N4.7.4[wide_helped]", (t.loc["wide"]["ratio cal/uncal"] < 1).all(), "temperature scaling lowers ece_10 for wide on all four conditions")
        check("N4.7.4[gru_T_near_1]", abs(DATA["gru"]["res"]["summary"]["temperature"] - 1) < 0.05, "gru's fitted temperature is close to 1 (already near-calibrated on clean)")
        """),

        md(r"""
        ### How to read this chart

        **Data source:** `ece_10`, `ece_15`, `ece_20` under `summary/conditions/<cond>/{uncalibrated,calibrated}` in each
        result file, and the fitted temperature `summary/temperature`.

        **Axes:** panels are architectures; horizontal is the stress condition; vertical is ECE on a log axis.
        Blue bars are uncalibrated, green calibrated; within each colour the three shades are $10$, $15$, $20$ bins.

        **Interpretation:** ECE is not much affected by the choice of bin count *within* a condition, which is a small
        robustness check on the reading. Temperature scaling helps `wide` on every condition (fitted $T\approx1.6$,
        i.e. an overconfident model), leaves `gru` essentially unchanged ($T\approx1.02$), and is mixed for `timepool` on
        `clean` (ECE with 10 bins rises). On `noise_10` all ECEs are enormous (about $0.5$--$0.75$) and temperature scaling,
        fitted on *clean* calibration data, barely reduces them: recalibrating on clean audio cannot repair miscalibration
        that arises from a distribution shift it never saw.

        **Falsifier:** if calibrated ECE were far lower than uncalibrated on `noise_10`, clean-fit temperature scaling
        would be robust to shift; it is not. ECE with only $1080$ test points and $10$--$20$ bins carries visible sampling
        noise, so differences of a few thousandths are not meaningful.
        """),

        md(r"""
        ### 7.3 NLL and AURC with bootstrap intervals
        """),

        code(r"""
        # N4.7.5: NLL and AURC with intervals, uncalibrated vs calibrated (gru and wide differ visibly under noise).
        fig, axes = plt.subplots(2, 3, figsize=(16, 7.4), sharex=True)
        for j, a in enumerate(ARCHS):
            for ri, q in enumerate(("nll", "aurc")):
                ax = axes[ri, j]
                for mi, m in enumerate(MODES):
                    d = SUM[(SUM.arch == a) & (SUM["mode"] == m)].set_index("condition").loc[COND_LIST]
                    x = np.arange(len(COND_LIST)) + (mi - 0.5) * 0.36
                    ax.bar(x, d[q], 0.36, color=("#1f77b4" if m == "uncalibrated" else "#2ca02c"), alpha=0.85, label=m,
                           yerr=[d[q] - d[f"{q}_lo"], d[f"{q}_hi"] - d[q]], capsize=2)
                ax.set_xticks(np.arange(len(COND_LIST))); ax.set_xticklabels(COND_LIST, fontsize=8); ax.set(title=f"{a}: {q.upper()}", ylabel=q.upper())
                if ri == 0 and j == 0: ax.legend(frameon=False, fontsize=8)
        fig.suptitle("NB4-17  NLL (top) and AURC (bottom) with bootstrap intervals (EXPLORATION: 200 resamples)", y=1.0)
        fig.tight_layout()
        savefig(fig, "NB4-17_nll_aurc.png")
        print(SUM[SUM["mode"] == "uncalibrated"].pivot(index="condition", columns="arch", values="aurc").round(3))
        """),

        md(r"""
        ### How to read this chart

        **Data source:** `nll`, `aurc` and `ci/nll`, `ci/aurc` in each result file's `summary` block.

        **Axes:** rows are NLL (top) and AURC (bottom), columns architectures, horizontal the stress condition. Blue is
        uncalibrated, green calibrated; whiskers are the recorded bootstrap interval (200 resamples).

        **Interpretation:** NLL grows from roughly $0.4$ (`gru`, clean) to about $4.5$ under `noise_10`, i.e. the model is
        confidently wrong. AURC, the area under the risk-coverage curve (lower is better, it measures how well confidence
        ranks correct above incorrect predictions), rises from about $0.017$ (`gru`, clean) to about $0.75$ under noise:
        under noise, confidence stops being a useful ranking of correctness. The largest calibration gain in NLL is `wide`
        under `noise_10` ($4.07 \rightarrow 2.71$), a case where temperature scaling does something visible but still leaves NLL
        far above its clean value.

        **Falsifier:** if AURC under `noise_10` were close to the clean AURC, selective prediction (abstaining on
        low-confidence inputs) would still be reliable under this shift. It is not: the values differ by more than an
        order of magnitude, far outside the bootstrap whiskers.
        """),

        md(r"""
        ### 7.4 Selective risk at fixed coverage
        """),

        code(r"""
        # N4.7.6: selective risk at coverage 0.8 and 0.9 (risk-coverage information that is stored), uncalibrated vs calibrated.
        fig, axes = plt.subplots(1, 3, figsize=(16, 4.4), sharey=True); w = 0.2
        for ax, a in zip(axes, ARCHS):
            for ci_, cov in enumerate(("0.8", "0.9")):
                for mi, m in enumerate(MODES):
                    d = SUM[(SUM.arch == a) & (SUM["mode"] == m)].set_index("condition").loc[COND_LIST]
                    x = np.arange(len(COND_LIST)) + (ci_ * 2 + mi - 1.5) * w
                    ax.bar(x, d[f"selective_risk_{cov}"], w, color=("#1f77b4" if m == "uncalibrated" else "#2ca02c"), alpha=0.5 + 0.4 * ci_, label=f"coverage {cov}, {m}" if a == ARCHS[0] else None)
            ax.set_xticks(np.arange(len(COND_LIST))); ax.set_xticklabels(COND_LIST, fontsize=8); ax.set(title=a)
        axes[0].set_ylabel("selective risk (error among retained)"); axes[0].legend(frameon=False, fontsize=7)
        fig.suptitle("NB4-18  Selective risk at coverage 0.8 / 0.9; the full risk-coverage curve needs per-example logits (not stored)", y=1.03)
        fig.tight_layout()
        savefig(fig, "NB4-18_selective_risk.png")
        d80 = SUM[SUM["mode"] == "uncalibrated"].pivot(index="condition", columns="arch", values="selective_risk_0.8")
        display(d80.round(3))
        check("N4.7.6[risk_order]", (SUM["selective_risk_0.8"] <= SUM["selective_risk_0.9"] + 1e-9).mean() > 0.5, "selective risk at 0.8 coverage is usually <= that at 0.9 coverage")
        """),

        md(r"""
        ### How to read this chart

        **Data source:** `selective_risk_0.8` and `selective_risk_0.9` in each result file's `summary` block.

        **Axes:** horizontal is the stress condition; vertical is the error rate among the retained examples when the model
        keeps the most confident $80\%$ (or $90\%$) of the test set. Colour is the calibration mode; darker shades are the
        higher coverage.

        **Interpretation:** for `gru` on `clean`, keeping the most confident $80\%$ of examples brings the error rate down to
        about $0.025$ against an overall error of about $0.12$: abstention works. Under `noise_10` the retained error is about
        $0.89$ at $80\%$ coverage, *worse* than the unfiltered error, because the model is confidently wrong on the examples it
        keeps. Temperature scaling does not change the ranking of confidences, so it barely moves these bars.

        **Falsifier:** if selective risk at $0.8$ coverage were not below the overall error on `clean`, confidence would not
        be informative even in-distribution; it is well below. Where selective risk exceeds the overall error, confidence
        is anti-correlated with correctness, which is what `noise_10` shows.
        """),

        md(r"""
        ### 7.5 Conformal risk control: threshold, miscoverage, mean set size

        Conformal risk control (CRC) turns a confidence threshold into prediction *sets*, aiming for a target miscoverage
        $\alpha$: the fraction of test points whose true class falls outside the set should not exceed $\alpha$ (up to a
        small finite-sample term) when calibration and test data are exchangeable. The result files record, for
        $\alpha=0.05$ and $\alpha=0.10$: the fitted threshold, the achieved miscoverage on the test split, and the mean set
        size. **The CRC used here is unweighted.** Under acoustic shift the exchangeability assumption fails, and
        the grounding names unweighted CRC as the failure mode under shift. This is flagged, not fixed.

        Angelopoulos, Bates, Fisch, Lei, Schuster, *Conformal Risk Control*, arXiv:2208.02814 (ICLR 2024). This is the source
        of the estimator used here and of the O(1/n) tightness bound. It is also the source of the failure mode this section
        is already subject to: the guarantee assumes exchangeability between calibration and test data, which acoustic shift
        breaks, and the estimator here is **unweighted**. Where achieved miscoverage exceeds the target alpha below, that is
        the predicted behaviour under non-exchangeability, not an implementation error.
        """),

        code(r"""
        # N4.7.7: CRC: achieved miscoverage vs target alpha, and mean set size.
        fig, axes = plt.subplots(2, 3, figsize=(16, 8))
        for j, a in enumerate(ARCHS):
            for ri, (col, ylabel) in enumerate((("miscoverage", "achieved miscoverage"), ("set_size", "mean prediction-set size"))):
                ax = axes[ri, j]
                for ai, (al, alname) in enumerate((("0.05", 0.05), ("0.1", 0.10))):
                    for mi, m in enumerate(MODES):
                        d = SUM[(SUM.arch == a) & (SUM["mode"] == m)].set_index("condition").loc[COND_LIST]
                        x = np.arange(len(COND_LIST)) + (ai * 2 + mi - 1.5) * 0.2
                        ax.bar(x, d[f"crc{al}_{col}"], 0.2, color=("#1f77b4" if m == "uncalibrated" else "#2ca02c"), alpha=0.5 + 0.4 * ai,
                               label=f"alpha={alname}, {m}" if (j == 0 and ri == 0) else None)
                    if ri == 0:
                        ax.axhline(alname, color="#d62728", ls=":", lw=1)
                ax.set_xticks(np.arange(len(COND_LIST))); ax.set_xticklabels(COND_LIST, fontsize=8); ax.set(title=f"{a}", ylabel=ylabel)
                if ri == 0 and j == 0: ax.legend(frameon=False, fontsize=7)
        fig.suptitle("NB4-19  Unweighted CRC (red dotted = target alpha 0.05 and 0.10): miscoverage (top), mean set size (bottom); EXPLORATION config", y=1.0)
        fig.tight_layout()
        savefig(fig, "NB4-19_crc.png")
        cov = []
        for a in ARCHS:
            for c in COND_LIST:
                for m in MODES:
                    r = SUM[(SUM.arch == a) & (SUM.condition == c) & (SUM["mode"] == m)].iloc[0]
                    for al, alname in (("0.05", 0.05), ("0.1", 0.10)):
                        cov.append({"arch": a, "condition": c, "mode": m, "alpha": alname, "miscoverage": r[f"crc{al}_miscoverage"], "set_size": r[f"crc{al}_set_size"],
                                    "target_met": r[f"crc{al}_miscoverage"] <= alname})
        COV = pd.DataFrame(cov)
        display(COV[COV["mode"] == "uncalibrated"].pivot_table(index=["arch", "condition"], columns="alpha", values=["miscoverage", "set_size"]).round(3))
        met = COV.groupby("condition")["target_met"].mean()
        print("fraction of (arch, mode, alpha) cells meeting the target, by condition:", met.round(3).to_dict())
        check("N4.7.7[clean_gru_target]", bool(COV[(COV.arch == "gru") & (COV.condition == "clean") & (COV["mode"] == "uncalibrated") & (COV.alpha == 0.05)]["target_met"].iloc[0]) is False, "gru clean alpha=0.05: target NOT met (0.072 > 0.05), recorded honestly")
        check("N4.7.7[noise_fails]", (COV[COV.condition == "noise_10"]["target_met"] == False).all(), "under noise_10 every CRC cell misses its target (unweighted CRC under shift)")
        note("N4.7.7[clean_exceedance]", "CRC misses its target even on `clean` (e.g. gru 0.072 vs 0.05) where exchangeability should hold; cause not investigated here (untested candidates: split structure, finite-sample slack)")
        """),

        md(r"""
        ### How to read this chart

        **Data source:** `summary/conditions/<cond>/<mode>/crc/<alpha>/{threshold,miscoverage,mean_set_size}` in each
        result file.

        **Axes:** top row, achieved miscoverage per stress condition with the target $\alpha$ as red dotted lines; bottom
        row, the mean number of classes in the prediction set. Colour is calibration mode, shade is $\alpha$
        (darker = $0.10$).

        **Interpretation:** a valid CRC procedure would keep the top-row bars at or below their red lines. On `clean` they
        are close but frequently slightly above for `gru` and `wide` (for instance $0.072$ against a target of $0.05$ for
        `gru`); on `gain_+10` and `reverb_mid` they are higher; on `noise_10` they are at $0.5$ to $0.84$, an order
        of magnitude above target. The bottom row shows the price: `timepool`'s sets contain about $4$ to $5$ classes on
        `clean` (it is unsure, so its sets are wide), while `gru`'s stay near $1$ to $1.4$, and under noise the sets do not
        widen enough to compensate. This is the expected failure of *unweighted* CRC when the test distribution differs from
        the calibration distribution, shown rather than merely asserted.

        **Falsifier:** if miscoverage stayed near $\alpha$ under noise while set sizes grew, the unweighted procedure
        would be robust to this shift; the opposite happens. If the target were also missed by a wide margin on `clean`,
        that would point to the procedure or the split rather than to shift; the clean exceedances are small but real and
        are left unexplained.
        """),

        md(r"""
        ### 7.6 OOD detection and fitted temperature
        """),

        code(r"""
        # N4.7.8: OOD AUROC per (arch, condition), uncalibrated vs calibrated.
        fig, axes = plt.subplots(1, 2, figsize=(14, 4.4), gridspec_kw={"width_ratios": [3, 1]})
        w = 0.13
        for j, a in enumerate(ARCHS):
            for mi, m in enumerate(MODES):
                d = SUM[(SUM.arch == a) & (SUM["mode"] == m)].set_index("condition").loc[COND_LIST]
                x = np.arange(len(COND_LIST)) + (j * 2 + mi - 2.5) * w
                axes[0].bar(x, d["ood_auroc"], w, color=ACOL[a], alpha=0.95 if m == "calibrated" else 0.45, label=f"{a} {m}")
        axes[0].axhline(0.5, color="#667085", ls=":", label="chance 0.5")
        axes[0].set_xticks(np.arange(len(COND_LIST))); axes[0].set_xticklabels(COND_LIST); axes[0].set(ylabel="OOD AUROC", ylim=(0.3, 1.0), title="max-softmax OOD detection vs held-out words")
        axes[0].legend(frameon=False, fontsize=7, ncol=3)
        axes[1].bar(ARCHS, [DATA[a]["res"]["summary"]["temperature"] for a in ARCHS], color=[ACOL[a] for a in ARCHS])
        axes[1].axhline(1.0, color="#667085", ls=":"); axes[1].set(ylabel="fitted temperature T", title="temperature")
        fig.suptitle("NB4-20  OOD AUROC (300 held-out-word clips) and fitted temperature; EXPLORATION config", y=1.03)
        fig.tight_layout()
        savefig(fig, "NB4-20_ood_temperature.png")
        display(SUM[SUM["mode"] == "uncalibrated"].pivot(index="condition", columns="arch", values="ood_auroc").round(3))
        check("N4.7.8[clean_above_chance]", (SUM[SUM.condition == "clean"]["ood_auroc"] > 0.5).all(), "clean-condition OOD AUROC above 0.5 for all archs and modes")
        check("N4.7.8[gru_noise_below_chance]", DATA["gru"]["res"]["summary"]["conditions"]["noise_10"]["uncalibrated"]["ood_auroc"] < 0.5, "gru's OOD AUROC under noise_10 is below chance (confidence inverted)")
        """),

        md(r"""
        ### How to read this chart

        **Data source:** `summary/conditions/<cond>/<mode>/ood_auroc` (AUROC of the maximum softmax probability as a score to
        separate in-distribution from the $300$ held-out-word OOD clips) and `summary/temperature`.

        **Axes:** left panel, stress condition against AUROC, coloured by architecture (light = uncalibrated, dark =
        calibrated); dotted line is chance ($0.5$). Right panel, the fitted temperature per architecture.

        **Interpretation:** on `clean`, the confidence score separates held-out words from in-distribution ones reasonably
        well for `gru` ($0.83$) and `wide` ($0.81$) and poorly for `timepool` ($0.60$). Under `noise_10` the AUROC for
        `gru` is about $0.47$, i.e. *below* chance: the model is, if anything, more confident on the noisy in-distribution
        clips than on the OOD ones. Temperature scaling rescales confidences monotonically, so the ranking, and therefore the
        AUROC, barely changes.

        **Falsifier:** if AUROC under `noise_10` stayed near the clean value, the confidence score would still detect
        novelty under this shift; it does not. An AUROC exactly equal for uncalibrated and calibrated would indicate the
        temperature had no effect; the two are close but not identical.
        """),

        md(r"""
        ### 7.7 Further post-training analyses

        The last block of analyses works only from the `summary` numbers already tabulated. It measures degradation
        relative to `clean`, whether architecture differences exceed the bootstrap intervals, how much temperature
        scaling changes each metric, and how the calibration and set-size metrics relate to accuracy across the
        $3\times4$ cells. As everywhere in this section: **exploration configuration** (4 conditions, 200 resamples), a
        single seed, no significance tests beyond interval overlap.
        """),

        code(r"""
        # N4.7.9: degradation relative to the clean condition, uncalibrated.
        base = SUM[(SUM["mode"] == "uncalibrated")].set_index(["arch", "condition"])
        DEG = []
        for a in ARCHS:
            for c in COND_LIST:
                r, r0 = base.loc[(a, c)], base.loc[(a, "clean")]
                DEG.append({"arch": a, "condition": c, "acc_drop": r0["accuracy"] - r["accuracy"], "acc_retained": r["accuracy"] / r0["accuracy"],
                            "ece_ratio": r["ece_10"] / r0["ece_10"], "nll_increase": r["nll"] - r0["nll"], "aurc_ratio": r["aurc"] / r0["aurc"],
                            "ood_auroc_drop": r0["ood_auroc"] - r["ood_auroc"]})
        DEGD = pd.DataFrame(DEG)
        display(DEGD[DEGD.condition != "clean"].set_index(["arch", "condition"]).round(3))
        fig, ax = plt.subplots(1, 2, figsize=(13, 4.2))
        piv = DEGD.pivot(index="arch", columns="condition", values="acc_retained").loc[ARCHS, COND_LIST]
        im = ax[0].imshow(piv.values, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
        ax[0].set_xticks(range(4)); ax[0].set_xticklabels(COND_LIST); ax[0].set_yticks(range(3)); ax[0].set_yticklabels(ARCHS)
        for i in range(3):
            for j in range(4):
                ax[0].text(j, i, f"{piv.values[i, j]:.2f}", ha="center", va="center", fontsize=9)
        ax[0].set_title("fraction of clean accuracy retained"); fig.colorbar(im, ax=ax[0], fraction=0.046)
        piv2 = DEGD.pivot(index="arch", columns="condition", values="nll_increase").loc[ARCHS, COND_LIST]
        for a in ARCHS:
            ax[1].plot(COND_LIST, piv2.loc[a], marker="o", color=ACOL[a], label=a)
        ax[1].set(ylabel="NLL increase over clean", title="NLL increase relative to clean"); ax[1].legend(frameon=False)
        fig.suptitle("NB4-20b  Degradation relative to the clean condition (EXPLORATION config)", y=1.03)
        fig.tight_layout()
        savefig(fig, "NB4-20b_degradation.png")
        for a in ARCHS:
            check(f"N4.7.9[{a}.noise_worst]", DEGD[(DEGD.arch == a) & (DEGD.condition == "noise_10")]["acc_retained"].iloc[0] == DEGD[(DEGD.arch == a)]["acc_retained"].min(), "noise_10 is the most damaging condition for this architecture")
            check(f"N4.7.9[{a}.clean_retained_1]", abs(DEGD[(DEGD.arch == a) & (DEGD.condition == "clean")]["acc_retained"].iloc[0] - 1) < 1e-12, "clean retains 100% of itself (sanity)")
        """),

        md(r"""
        ### How to read this chart

        **Data source:** `summary/conditions/<cond>/uncalibrated/{accuracy,nll}` in `arch-*.result.json`.

        **Axes:** left, a heatmap (architecture by condition) of the fraction of clean accuracy retained under each stress;
        right, NLL increase over the clean condition per architecture across the four conditions.

        **Interpretation:** `noise_10` is the most damaging condition for all three architectures and retains only about
        $0.16$ (`gru`) to $0.30$ (`wide`) of clean accuracy. `gain_+10` and `reverb_mid` retain most of it for `gru` and
        `wide`. Retention is highest for `wide` on `noise_10`, even though `wide` is not the most accurate on clean:
        the ordering by clean accuracy is not the ordering by robustness.

        **Falsifier:** if retained accuracy under `noise_10` were similar across the three architectures, robustness would
        not discriminate between them; the retained fraction differs by nearly a factor of two.
        """),

        code(r"""
        # N4.7.10: do bootstrap intervals overlap between architectures? (accuracy, uncalibrated) An overlap test only, not a significance test.
        pairs = [("gru", "wide"), ("gru", "timepool"), ("wide", "timepool")]
        OV = []
        for c in COND_LIST:
            for x, y in pairs:
                rx, ry = base.loc[(x, c)], base.loc[(y, c)]
                overlap = not (rx["accuracy_lo"] > ry["accuracy_hi"] or ry["accuracy_lo"] > rx["accuracy_hi"])
                OV.append({"condition": c, "pair": f"{x} vs {y}", "acc_x": rx["accuracy"], "acc_y": ry["accuracy"], "width_x": rx["accuracy_hi"] - rx["accuracy_lo"],
                           "width_y": ry["accuracy_hi"] - ry["accuracy_lo"], "intervals_overlap": overlap})
        OVD = pd.DataFrame(OV)
        display(OVD.round(3))
        check("N4.7.10[clean_gru_timepool]", not OVD[(OVD.condition == "clean") & (OVD.pair == "gru vs timepool")]["intervals_overlap"].iloc[0], "gru and timepool accuracy intervals do not overlap on clean")
        check("N4.7.10[clean_gru_wide]", bool(OVD[(OVD.condition == "clean") & (OVD.pair == "gru vs wide")]["intervals_overlap"].iloc[0]), "gru and wide accuracy intervals overlap on clean: their ordering there is not established")
        check("N4.7.10[widths_small]", (OVD[["width_x", "width_y"]] < 0.1).all().all(), "all recorded accuracy intervals are narrower than 0.1")
        noise = OVD[OVD.condition == "noise_10"]
        print("noise_10 overlaps:", dict(zip(noise["pair"], noise["intervals_overlap"])))
        """),

        md(r"""
        **Reading the overlap table.** On `clean`, the $0.88$ (`gru`) versus $0.86$ (`wide`) difference sits inside
        overlapping bootstrap intervals, so their ordering is *not* established by this evidence, in spite of `gru`'s
        higher value; both are separated from `timepool` by a wide margin. The bootstrap resamples the *test split*
        only (a single trained model), so it captures test-sampling noise and not training-seed variance, which is the
        larger source and is absent here (Limitation 2). Overlap is a conservative screen, not a hypothesis test.
        """),

        code(r"""
        # N4.7.11: effect of temperature scaling on each metric (calibrated minus uncalibrated), all cells.
        METRICS = ["nll", "brier", "ece_10", "ece_15", "ece_20", "aurc", "ood_auroc", "selective_risk_0.8", "selective_risk_0.9"]
        cal = SUM[SUM["mode"] == "calibrated"].set_index(["arch", "condition"])[METRICS]
        unc = SUM[SUM["mode"] == "uncalibrated"].set_index(["arch", "condition"])[METRICS]
        DELTAS = (cal - unc)
        display(DELTAS.round(4))
        summary_rows = []
        for m in METRICS:
            better = (DELTAS[m] < 0) if m != "ood_auroc" else (DELTAS[m] > 0)
            summary_rows.append({"metric": m, "cells improved": int(better.sum()), "cells worsened": int((~better & (DELTAS[m] != 0)).sum()), "cells unchanged": int((DELTAS[m] == 0).sum()),
                                 "mean delta": DELTAS[m].mean()})
        TSUM = pd.DataFrame(summary_rows).set_index("metric")
        display(TSUM.round(4))
        check("N4.7.11[nll_helps_mostly]", TSUM.loc["nll", "cells improved"] >= 9, f"temperature scaling improves NLL in {TSUM.loc['nll', 'cells improved']} of 12 cells (fit target is NLL)")
        check("N4.7.11[wide_biggest]", DELTAS.loc["wide", "nll"].mean() < DELTAS.loc["gru", "nll"].mean(), "wide gains more NLL from temperature scaling than gru (it starts more overconfident)")
        note("N4.7.11", "temperature was fitted on CLEAN calibration data only; its effect under noise_10 is a transfer test, not a fit")
        """),

        code(r"""
        # N4.7.12: how the calibration/set-size metrics relate to accuracy across the 12 (arch, condition) cells.
        cells_u = SUM[SUM["mode"] == "uncalibrated"].reset_index(drop=True)
        fig, axes = plt.subplots(1, 3, figsize=(16, 4.2))
        rels = {}
        for ax, (col, lbl) in zip(axes, (("ece_10", "ECE (10 bins)"), ("crc0.05_set_size", "mean CRC set size (alpha=0.05)"), ("crc0.05_miscoverage", "CRC miscoverage (alpha=0.05)"))):
            for a in ARCHS:
                d = cells_u[cells_u.arch == a]
                ax.scatter(d["accuracy"], d[col], color=ACOL[a], s=45, label=a)
                for _, r in d.iterrows():
                    ax.annotate(r["condition"].replace("_", ""), (r["accuracy"], r[col]), fontsize=6.5, xytext=(3, 3), textcoords="offset points")
            rels[col] = spearmanr(cells_u["accuracy"], cells_u[col])[0]
            ax.set(xlabel="test accuracy", ylabel=lbl, title=f"rho = {rels[col]:+.2f}")
        axes[0].legend(frameon=False, fontsize=8)
        fig.suptitle("NB4-20c  Calibration and conformal metrics versus accuracy across the 12 (architecture, condition) cells (uncalibrated)", y=1.03)
        fig.tight_layout()
        savefig(fig, "NB4-20c_metric_relations.png")
        print("Spearman rho across the 12 cells:", {k: round(v, 3) for k, v in rels.items()})
        check("N4.7.12[ece_vs_acc]", rels["ece_10"] < -0.5, f"ECE falls as accuracy rises across cells: rho={rels['ece_10']:+.2f}")
        check("N4.7.12[miscov_vs_acc]", rels["crc0.05_miscoverage"] < -0.5, f"CRC miscoverage falls as accuracy rises: rho={rels['crc0.05_miscoverage']:+.2f}")
        """),

        md(r"""
        ### How to read this chart

        **Data source:** the uncalibrated `ece_10`, `crc/0.05/{mean_set_size,miscoverage}` and `accuracy` values of
        each of the twelve (architecture, condition) cells in the `summary` blocks.

        **Axes:** horizontal is test accuracy; vertical is, left to right, ECE, mean CRC set size, and CRC miscoverage
        at $\alpha=0.05$. Points are coloured by architecture and labelled by condition; the title shows the Spearman
        rank correlation across the twelve points.

        **Interpretation:** the cloud is dominated by the two extremes, clean-ish conditions at high accuracy with low ECE
        and low miscoverage, and `noise_10` at low accuracy with ECE and miscoverage near their maxima. The strong
        negative rank correlations reflect that split more than a fine-grained relationship. Set size behaves
        differently: it is largest for `timepool` on the *clean-ish* conditions (a model that knows it is unsure
        emits wide sets) and *smaller* under noise, because a confidently-wrong model does not widen its sets, which is the
        unweighted-CRC failure again.

        **Falsifier:** if set size grew with distribution shift, CRC would be adapting to the shift; the middle panel shows
        it does not. With twelve dependent points (three architectures, four conditions of one model each) the
        correlations are descriptive, not tests.
        """),

        code(r"""
        # N4.7.13: completeness of the summary block (every expected key present, finite, and in range).
        expect_scalar = SCALARS
        n_ok = n_tot = 0
        for a in ARCHS:
            for c in COND_LIST:
                for m in MODES:
                    d = DATA[a]["res"]["summary"]["conditions"][c][m]
                    for k in expect_scalar:
                        n_tot += 1; n_ok += int(k in d and np.isfinite(d[k]))
                    for al in ("0.05", "0.1"):
                        for k in ("threshold", "miscoverage", "mean_set_size"):
                            n_tot += 1; n_ok += int(al in d["crc"] and np.isfinite(d["crc"][al][k]))
                    for k in ("accuracy", "ece_10", "nll", "aurc"):
                        n_tot += 1; n_ok += int(k in d["ci"] and all(q in d["ci"][k] for q in ("point", "lower", "upper")))
        check("N4.7.13[complete]", n_ok == n_tot, f"{n_ok}/{n_tot} expected summary entries present and finite")
        rng = {"accuracy": (0, 1), "macro_f1": (0, 1), "ece_10": (0, 1), "aurc": (0, 1), "ood_auroc": (0, 1), "brier": (0, 2)}
        for k, (lo, hi) in rng.items():
            check(f"N4.7.13[range.{k}]", bool(SUM[k].between(lo, hi).all()), f"{k} within [{lo}, {hi}] in all 24 rows")
        for al in ("0.05", "0.1"):
            check(f"N4.7.13[crc{al}.sizes]", bool((SUM[f"crc{al}_set_size"] >= 1 - 1e-9).all() and (SUM[f"crc{al}_set_size"] <= 12).all()), f"mean CRC set size for alpha={al} in [1, 12]")
        check("N4.7.13[thresholds_ordered]", bool((SUM["crc0.05_threshold"] <= SUM["crc0.1_threshold"] + 1e-9).mean() > 0.5), "CRC threshold is usually lower at alpha=0.05 than at alpha=0.10 (see table)")
        """),

        md(r"""
        ### 7.8 Ledger of the numbers quoted in this notebook's prose

        The markdown above quotes a number of values. The table below recomputes each from the artifacts, so a
        reader can check the prose against the data in one place. If a value here ever disagrees with the prose, the
        prose is wrong, not the table.
        """),

        code(r"""
        # N4.8.1: recompute the headline numbers quoted in the markdown.
        s = {a: DATA[a]["res"]["summary"]["conditions"] for a in ARCHS}
        ledger = [
            ("gru best/final val acc", f"{DATA['gru']['res']['best_val_accuracy']:.4f} / {DATA['gru']['res']['final_val_accuracy']:.4f}", "0.9127 / 0.9008"),
            ("timepool best/final val acc", f"{DATA['timepool']['res']['best_val_accuracy']:.4f} / {DATA['timepool']['res']['final_val_accuracy']:.4f}", "0.5556 / 0.4107"),
            ("wide best/final val acc", f"{DATA['wide']['res']['best_val_accuracy']:.4f} / {DATA['wide']['res']['final_val_accuracy']:.4f}", "0.8790 / 0.7540"),
            ("clip counts gru/timepool/wide", f"{CLIP.loc['gru', 'clipped']}/{CLIP.loc['timepool', 'clipped']}/{CLIP.loc['wide', 'clipped']}", "663/1284/910"),
            ("2D alert idx0 gru/timepool/wide", f"{ALERT['gru']['2d']}/{ALERT['timepool']['2d']}/{ALERT['wide']['2d']}", "17/2/2"),
            ("raw alert idx0 (all three)", f"{ALERT['gru']['raw']}/{ALERT['timepool']['raw']}/{ALERT['wide']['raw']}", "2/2/2"),
            ("gru inter_2d first/last", f"{DATA['gru']['npz']['inter_2d'][0]:.2f}/{DATA['gru']['npz']['inter_2d'][-1]:.2f}", "2.62/4.29"),
            ("gru intra_2d first/min", f"{DATA['gru']['npz']['intra_2d'][0]:.2f}/{DATA['gru']['npz']['intra_2d'].min():.2f}", "6.65/1.50"),
            ("timepool intra_2d first/last", f"{DATA['timepool']['npz']['intra_2d'][0]:.1f}/{DATA['timepool']['npz']['intra_2d'][-1]:.1f}", "9.9/28.7"),
            ("wide intra_2d min/max", f"{DATA['wide']['npz']['intra_2d'].min():.0f}/{DATA['wide']['npz']['intra_2d'].max():.0f}", "48/135"),
            ("clean acc gru/wide/timepool", f"{s['gru']['clean']['uncalibrated']['accuracy']:.3f}/{s['wide']['clean']['uncalibrated']['accuracy']:.3f}/{s['timepool']['clean']['uncalibrated']['accuracy']:.3f}", "0.879/0.856/0.549"),
            ("noise_10 acc gru/timepool/wide", f"{s['gru']['noise_10']['uncalibrated']['accuracy']:.3f}/{s['timepool']['noise_10']['uncalibrated']['accuracy']:.3f}/{s['wide']['noise_10']['uncalibrated']['accuracy']:.3f}", "0.138/0.147/0.252"),
            ("gru clean CRC miscoverage a=0.05", f"{s['gru']['clean']['uncalibrated']['crc']['0.05']['miscoverage']:.3f}", "0.072"),
            ("gru noise_10 OOD AUROC", f"{s['gru']['noise_10']['uncalibrated']['ood_auroc']:.3f}", "0.473"),
            ("fitted T gru/timepool/wide", "/".join(f"{DATA[a]['res']['summary']['temperature']:.2f}" for a in ("gru", "timepool", "wide")), "1.02/1.11/1.60"),
        ]
        LED = pd.DataFrame(ledger, columns=["quantity", "recomputed from artifacts", "value quoted in prose"])
        LED["match"] = LED["recomputed from artifacts"] == LED["value quoted in prose"]
        display(LED)
        for _, r in LED.iterrows():
            check(f"N4.8.1[{r['quantity']}]", bool(r["match"]), f"recomputed {r['recomputed from artifacts']} vs prose {r['value quoted in prose']}")
        """),
    ]
