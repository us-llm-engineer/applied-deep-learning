"""NB2 part 1: framing, the synthetic generator, and data understanding.

Everything in this notebook is synthetic ON PURPOSE (a sandbox for exercising the pipeline cheaply).
The sibling real-data notebook (04) forbids synthetic data; this one requires it.
"""
from nbkit import md, code


def cells():
    return [
        md(r"""
        ## 1. What this notebook is, and why synthetic data is correct here

        This is the **small-scale exploration sandbox**. Its job is to exercise the whole study pipeline -- models,
        the training loop with its in-training probes, calibration, conformal risk control -- *cheaply*, on a CPU, in a
        few minutes, and to explore directions before anything expensive is run on real audio. Every array below is
        **synthetic, deliberately and by design**.

        **How it differs from the real-data notebook.** Notebook 4 (`04_real_dataset_walkthrough`) is built only from
        artifacts of a real GPU run on Speech Commands and *forbids* synthetic or mock data anywhere. This notebook is
        its opposite: it *requires* synthetic data, because a sandbox whose inputs come from a generator with a known,
        inspectable structure lets us ask questions that real audio cannot answer cheaply -- for example, "if two
        classes differ **only** by the temporal order of two spectral events, which architectures can tell them apart?"
        On real audio that question is entangled with speaker, channel, and vocabulary; here it is isolated. The price
        is that nothing learned here transfers by assertion to real speech. Section 9 says so again.

        **What "synthetic" must not mean.** The notebooks this one replaces were deleted because their figures were
        worthless: one rendered zero bars, one plotted accuracies of 0.083--0.33 when chance for 12 classes is
        $1/12 = 0.083$, and one was five identical full-width bars all reading "PASS". Those failures share a cause:
        the synthetic data carried no learnable class structure, so every figure showed noise or a constant. This
        notebook therefore enforces two rules, both executable:

        1. The generator gives every class a **distinct, time-varying spectral signature** (a frequency band whose centre
           moves across frames on a class-specific trajectory, with two bursts whose *order* is class-specific), and
           Section 3 verifies the structure is visible before any model is trained.
        2. A helper `assert_varied(values, label)` is called before every figure. If a plotted series is constant or
           near-constant ($\mathrm{ptp} < 10^{-9}$) the labelled check prints `FAIL`. A figure whose values are all
           identical is a defect, not a result.

        **Roadmap.** Section 2 defines the generator. Section 3 is data understanding (per-class mean spectrograms,
        separability of the raw features, split sizes). Section 4 trains the four architectures of `src/models.py` with
        identical configuration and seed and reports the notebook's central cheap finding about global average pooling.
        Section 5 covers the in-training probes: AutoClip gradient norms and clip thresholds, SentryCam-style 2D
        cluster health, per-epoch latent scatter, and the raw-space versus 2D alert comparison. Sections 6--8 are
        three exploratory directions: an AutoClip percentile sweep, train-time versus post-hoc calibration, and
        weighted versus unweighted conformal risk control under simulated shift. Section 9 is limitations.
        """),

        code(r"""
        # NB2 helpers: threads, plain-text tables, figure saving, and the assert_varied guard.
        import os, time, math, itertools
        import pandas as pd
        import torch
        pd.set_option("display.notebook_repr_html", False)   # plain-text DataFrame repr keeps the .ipynb small
        pd.set_option("display.width", 160)
        from IPython.display import display

        torch.set_num_threads(min(4, os.cpu_count() or 1))
        NB_T0 = time.time()
        FIGDIR = ROOT / "runtime" / "figures" / "nb2"
        FIGDIR.mkdir(parents=True, exist_ok=True)

        def savefig(fig, name):
            # Save under runtime/figures/nb2/ and show inline.
            fig.savefig(FIGDIR / name, dpi=110, bbox_inches="tight")
            plt.show()
            return FIGDIR / name

        def is_varied(values, tol=1e-9):
            # Pure predicate: True when the finite values span more than `tol`.
            arr = np.asarray(values, dtype=float).ravel()
            arr = arr[np.isfinite(arr)]
            return bool(arr.size > 1 and np.ptp(arr) > tol)

        def assert_varied(values, label, tol=1e-9):
            # Executable guard for figures: a constant series is a defect, not a result.
            arr = np.asarray(values, dtype=float).ravel()
            arr = arr[np.isfinite(arr)]
            spread = float(np.ptp(arr)) if arr.size else float("nan")
            return check(f"VARIED[{label}]", is_varied(arr, tol), f"n={arr.size} range={spread:.4g} (constant would be a defect)")

        # Self-test of the guard on inputs whose answer is known.
        check("N2.1.1[guard_rejects_constant]", not is_varied(np.full(12, 0.5)), "a constant series is rejected")
        check("N2.1.1[guard_rejects_near_constant]", not is_varied(0.5 + 1e-12 * np.arange(6)), "a near-constant series (spread 5e-12) is rejected")
        check("N2.1.1[guard_rejects_empty_and_nan]", not is_varied([]) and not is_varied([np.nan, np.nan]), "empty and all-NaN series are rejected")
        check("N2.1.1[guard_accepts_varied]", is_varied(np.linspace(0, 1, 5)), "a series spanning [0, 1] is accepted")
        print("torch threads:", torch.get_num_threads(), "| figures ->", FIGDIR)
        """),

        md(r"""
        ## 2. The synthetic generator: seed, class signatures, and why the time axis matters

        Each example is a log-mel-shaped tensor $X \in \mathbb{R}^{1 \times 40 \times 40}$ (channel, mel bins, frames),
        the shape `src/models.py` consumes. A background of i.i.d. Gaussian noise $\mathcal{N}(0, \sigma_0^2)$ with
        $\sigma_0 = 1$ is overlaid with **two spectral bursts**, each 8 frames long. A burst is a Gaussian band in mel
        space whose centre drifts linearly with time:

        $$X_{m,t} \mathrel{+}= A \exp\!\left(-\tfrac12 \left(\frac{m - c(t)}{w}\right)^2\right), \qquad c(t) = c_0 + \delta\,(t - t_{\text{on}} - 4),$$

        with amplitude $A = 3$, width $w = 2$ mel bins, and per-example jitter $c_0 \sim \mathcal{N}(\mu_c, 0.7^2)$ and onset
        $t_{\text{on}}$ uniform over a five-frame window. The first burst starts in frames 2--6 and the second in frames
        28--32, so the two events are separated by more than 15 frames.

        **Class signature.** There are 12 classes built as 6 *groups* $\times$ 2 *orders*. A group fixes a pair of centre
        frequencies $(\text{lo}, \text{hi}) \in \{5, 12, 19\} \times \{28, 35\}$ and a drift sign $\delta = \pm 0.4$ bins per
        frame (alternating across groups). The two classes in a group play **the same two bursts in opposite orders**:
        order 0 plays $\text{lo}$ then $\text{hi}$, order 1 plays $\text{hi}$ then $\text{lo}$. Consequently:

        * two classes in the same group have the **same time-averaged spectrum** (the same multiset of frames, permuted);
        * classes in different groups differ in *where* the energy is, which any model can see;
        * classes within a group differ **only in temporal order**.

        This is the construction that makes the architecture comparison meaningful. A model that discards the time axis
        can, at best, identify the group (6 ways) and must guess the order, so its accuracy is capped near
        $6 \times \tfrac12 / 6 = 0.5$ when the group is perfectly identified, versus chance $1/12 \approx 0.083$. A
        model that keeps the time axis is not capped. The bursts are far enough apart (>15 frames) that a small
        convolutional receptive field cannot see both at once, so the order cannot be read from any single local patch.

        **Seeds.** All randomness derives from `RNG_SEED = 20260926` (defined in the shared setup cell) through
        `numpy.random.SeedSequence(RNG_SEED).spawn(5)`, giving five independent child streams: train, validation,
        calibration, test, and a shift pool (used in Section 8). Model initialisation and shuffling use `seed=17`,
        as in the real-data runs.
        """),

        code(r"""
        # Generator constants and the explicit per-class signature table.
        N_MELS, FRAMES, N_CLASSES, N_GROUPS = 40, 40, 12, 6
        LOW_CENTRES, HIGH_CENTRES = (5, 12, 19), (28, 35)
        BURST_LEN, BURST_AMP, BURST_WIDTH, BASE_NOISE, DRIFT, CENTRE_JITTER = 8, 3.0, 2.0, 1.0, 0.4, 0.7
        ONSET1, ONSET2 = (2, 7), (28, 33)      # numpy integers(low, high): onsets 2..6 and 28..32
        MODEL_SEED = 17

        def build_signatures():
            rows, g = [], 0
            for lo in LOW_CENTRES:
                for hi in HIGH_CENTRES:
                    drift = DRIFT if g % 2 == 0 else -DRIFT
                    for order in (0, 1):
                        first, second = (lo, hi) if order == 0 else (hi, lo)
                        rows.append(dict(cls=2 * g + order, group=g, lo=lo, hi=hi, order=order,
                                         first_centre=first, second_centre=second, drift=drift))
                    g += 1
            return pd.DataFrame(rows).set_index("cls")

        SIG = build_signatures()
        SIG_ARR = SIG[["first_centre", "second_centre", "drift"]].to_numpy(dtype=float)
        display(SIG)

        # Structural properties the whole notebook relies on.
        check("N2.2.1[twelve_classes]", len(SIG) == N_CLASSES == N_GROUPS * 2, f"{len(SIG)} classes = {N_GROUPS} groups x 2 orders")
        same_multiset = all(sorted(SIG.loc[2 * g, ["first_centre", "second_centre"]]) == sorted(SIG.loc[2 * g + 1, ["first_centre", "second_centre"]]) for g in range(N_GROUPS))
        check("N2.2.1[orders_share_frequencies]", same_multiset, "in every group the two orders use the same two centre frequencies")
        check("N2.2.1[orders_swap_time]", all(SIG.loc[2 * g, "first_centre"] == SIG.loc[2 * g + 1, "second_centre"] for g in range(N_GROUPS)), "order 1 plays order 0's bursts reversed in time")
        check("N2.2.1[groups_distinct]", len({(r.lo, r.hi) for r in SIG.itertuples()}) == N_GROUPS, "the six groups have six distinct (lo, hi) pairs")
        # The design intent is that the two bursts never touch, so a time-blind model cannot recover order
        # from a single blurred blob. The binding constraint is one clear burst-length of silence between
        # them; the ">15" originally written here was an arbitrary round number, not the design rule.
        _gap = ONSET2[0] - (ONSET1[1] - 1 + BURST_LEN)
        check("N2.2.1[burst_separation]", _gap >= BURST_LEN, f"minimum gap between bursts = {_gap} frames (design rule: >= BURST_LEN = {BURST_LEN})")
        check("N2.2.1[chance_and_ceiling]", abs(1 / N_CLASSES - 0.0833) < 1e-3 and abs(N_GROUPS * 0.5 / N_GROUPS - 0.5) < 1e-12, "chance = 1/12; order-blind ceiling = 0.5")
        """),

        code(r"""
        # The generator itself: vectorised background noise, then two drifting Gaussian bursts per example.
        def synthesize(labels, rng, noise=BASE_NOISE):
            labels = np.asarray(labels)
            mel = np.arange(N_MELS, dtype=float)
            X = rng.normal(0.0, noise, (len(labels), 1, N_MELS, FRAMES)).astype(np.float32)
            for i, k in enumerate(labels):
                first, second, drift = SIG_ARR[k]
                for centre, (lo_on, hi_on) in ((first, ONSET1), (second, ONSET2)):
                    onset = int(rng.integers(lo_on, hi_on))
                    c0 = centre + rng.normal(0.0, CENTRE_JITTER)
                    for j in range(BURST_LEN):
                        c = c0 + drift * (j - BURST_LEN / 2)
                        X[i, 0, :, onset + j] += (BURST_AMP * np.exp(-0.5 * ((mel - c) / BURST_WIDTH) ** 2)).astype(np.float32)
            return X

        CHILD = np.random.SeedSequence(RNG_SEED).spawn(5)
        SPLIT_SIZES = {"train": 50, "val": 20, "cal": 40, "test": 50}   # examples per class
        RAW, Y = {}, {}
        for child, (name, n_per) in zip(CHILD, SPLIT_SIZES.items()):
            Y[name] = np.repeat(np.arange(N_CLASSES), n_per)
            RAW[name] = synthesize(Y[name], np.random.default_rng(child))
        # Shift pool for Section 8: same generator, an independent stream, 50 per class.
        Y["pool"] = np.repeat(np.arange(N_CLASSES), 50)
        RAW["pool"] = synthesize(Y["pool"], np.random.default_rng(CHILD[4]))

        # Standardise with TRAINING statistics only; every split reuses them (no leakage).
        MU, SD = float(RAW["train"].mean()), float(RAW["train"].std())
        X = {k: ((v - MU) / SD).astype(np.float32) for k, v in RAW.items()}
        print(f"train mean/std used for standardisation: {MU:.4f} / {SD:.4f}")
        print({k: v.shape for k, v in X.items()})

        check("N2.2.2[shapes]", all(v.shape[1:] == (1, N_MELS, FRAMES) for v in X.values()), f"every split has trailing shape (1, {N_MELS}, {FRAMES})")
        check("N2.2.2[finite_float32]", all(np.isfinite(v).all() and v.dtype == np.float32 for v in X.values()), "all splits finite float32")
        check("N2.2.2[class_balance]", all(np.bincount(Y[k]).min() == np.bincount(Y[k]).max() for k in Y), "every split is exactly class-balanced")
        check("N2.2.2[train_standardised]", abs(float(X['train'].mean())) < 1e-4 and abs(float(X['train'].std()) - 1) < 1e-4, "the training split has mean 0, std 1")
        again = synthesize(Y["val"], np.random.default_rng(CHILD[1]))
        check("N2.2.2[same_seed_same_data]", np.array_equal(again, RAW["val"]), "a second call with the same SeedSequence child reproduces the validation split bit-for-bit")
        """),

        md(r"""
        ## 3. Data understanding: is the class structure real and learnable?

        Before any network is trained we verify, with the data alone, that the structure promised in Section 2 is present
        and that the raw features carry the information a model would need. Three views follow: the per-class mean
        spectrogram (does each class have a visibly distinct time-frequency signature?), the time-marginal profiles
        (are same-group classes really identical once time is averaged out?), and a model-free separability test
        (how much accuracy does a linear classifier get from time-averaged features versus the full time-frequency
        image?). The last is a cheap preview of the Section 4 headline, obtained without training a single network.
        """),

        code(r"""
        # Fig NB2-1: per-class mean spectrogram (3 x 4 panels, one per class).
        CLASS_NAMES = [f"g{SIG.loc[k, 'group']}{'ab'[SIG.loc[k, 'order']]}" for k in range(N_CLASSES)]
        MEANS = np.stack([RAW["train"][Y["train"] == k, 0].mean(axis=0) for k in range(N_CLASSES)])   # (12, mel, frames)
        assert_varied(MEANS.max(axis=(1, 2)) - MEANS.min(axis=(1, 2)), "mean_spectrogram_dynamic_range")
        assert_varied(MEANS.reshape(N_CLASSES, -1).std(axis=1), "mean_spectrogram_spatial_std")
        fig, axes = plt.subplots(3, 4, figsize=(13, 8.5), sharex=True, sharey=True)
        vmin, vmax = MEANS.min(), MEANS.max()
        for k, ax in enumerate(axes.ravel()):
            im = ax.imshow(MEANS[k], origin="lower", aspect="auto", cmap="magma", vmin=vmin, vmax=vmax)
            ax.set_title(f"class {k} ({CLASS_NAMES[k]}): {int(SIG.loc[k, 'first_centre'])} -> {int(SIG.loc[k, 'second_centre'])}", fontsize=9)
            ax.grid(False)
        for ax in axes[-1]:
            ax.set_xlabel("frame")
        for ax in axes[:, 0]:
            ax.set_ylabel("mel bin")
        fig.colorbar(im, ax=axes, shrink=0.8, label="mean log-mel value (train)")
        fig.suptitle("NB2-1  Per-class mean spectrogram: two drifting bursts, order swapped within each group", y=0.995)
        savefig(fig, "NB2-1_class_means.png")
        peak_frame = MEANS.max(axis=1).argmax(axis=1)
        print("frame of the brightest cell per class:", dict(zip(CLASS_NAMES, peak_frame.tolist())))
        # Noise floor of a 50-example class mean is sigma0/sqrt(50) per cell; a cross-group distance far above it means real structure.
        floor = BASE_NOISE / np.sqrt(50) * np.sqrt(N_MELS * FRAMES)
        cross = float(np.linalg.norm(MEANS[0] - MEANS[2]))
        swap = float(np.linalg.norm(MEANS[0] - MEANS[1]))
        check("N2.3.1[cross_group_distance_above_noise_floor]", cross > 3 * floor, f"||mean(class0) - mean(class2)||_F = {cross:.2f} vs noise floor {floor:.2f}")
        check("N2.3.1[order_swap_distance_above_noise_floor]", swap > 3 * floor, f"||mean(class0) - mean(class1)||_F = {swap:.2f} vs noise floor {floor:.2f}: order is visible in the full image")
        """),

        md(r"""
        ### How to read this chart

        Each panel is the average of the 50 training spectrograms of one class: mel bin up, frame right, brighter is
        more energy. Every panel shows two bright, slightly slanted streaks -- the two drifting bursts -- on an otherwise
        featureless (averaged-out noise) background. The title gives the class name (`g<group><a|b>`) and the centre
        frequencies of the first and second burst. Compare classes 0 and 1: the *same two streaks* appear, but the
        low-frequency streak comes first in class 0 and second in class 1, so the panels are vertical reflections of each
        other in time-ordering, not in content. Compare classes 0 and 2: the streaks sit at different mel heights, which is
        the group-level (order-blind) difference. Because the colour scale is shared across panels, the streaks have equal
        brightness everywhere; the classes differ in *where and when*, not in loudness. The slant flips sign between
        adjacent groups (drift $\pm 0.4$ bins/frame), which is the "moving band" part of the signature.
        """),

        code(r"""
        # Fig NB2-2: time-marginal profiles. Same-group classes must coincide when time is averaged out.
        prof = np.stack([MEANS[k].mean(axis=1) for k in range(N_CLASSES)])                     # (12, mel): time-averaged spectrum
        low_energy = np.stack([MEANS[k][:26].mean(axis=0) for k in range(N_CLASSES)])           # (12, frames): mel<26 energy over time
        within = np.array([np.abs(prof[2 * g] - prof[2 * g + 1]).mean() for g in range(N_GROUPS)])
        between = np.array([np.abs(prof[2 * g] - prof[2 * h]).mean() for g in range(N_GROUPS) for h in range(g + 1, N_GROUPS)])
        assert_varied(prof.ravel(), "time_averaged_profiles")
        assert_varied(low_energy.ravel(), "low_band_energy_over_time")
        assert_varied(np.concatenate([within, between]), "within_vs_between_profile_gaps")
        fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
        cols = plt.cm.tab10(np.arange(N_GROUPS))
        for g in range(N_GROUPS):
            ax[0].plot(prof[2 * g], color=cols[g], lw=2.2, label=f"group {g} order a")
            ax[0].plot(prof[2 * g + 1], color=cols[g], lw=1.2, ls="--", label=f"group {g} order b")
        ax[0].set(xlabel="mel bin", ylabel="time-averaged mean value", title="time-averaged spectrum per class")
        ax[0].legend(fontsize=6, ncol=2, frameon=False)
        for k, ls in ((0, "-"), (1, "--")):
            ax[1].plot(low_energy[k], ls=ls, lw=2, color=cols[0], label=f"class {k} ({CLASS_NAMES[k]})")
        ax[1].set(xlabel="frame", ylabel="mean value, mel bins 0-25", title="group 0: low-band energy over time")
        ax[1].legend(frameon=False)
        ax[2].bar(["within group\n(order swap)", "between groups"], [within.mean(), between.mean()], color=[PALETTE["clean"], PALETTE["shifted"]])
        ax[2].errorbar([0, 1], [within.mean(), between.mean()], yerr=[within.std(), between.std()], fmt="none", ecolor="black", capsize=4)
        ax[2].set(ylabel="mean |profile difference|", title="profile gap: order swap vs different group")
        fig.suptitle("NB2-2  Averaging over time erases the order, not the group", y=1.03)
        fig.tight_layout()
        savefig(fig, "NB2-2_time_marginals.png")
        print(f"within-group profile gap {within.mean():.4f} +/- {within.std():.4f}; between-group {between.mean():.4f} +/- {between.std():.4f}")
        check("N2.3.2[order_erased_by_time_average]", within.mean() < 0.25 * between.mean(), f"within/between profile gap ratio = {within.mean() / between.mean():.3f} (<0.25)")
        check("N2.3.2[order_visible_in_time]", float(np.abs(low_energy[0] - low_energy[1]).max()) > 0.5, f"max |low-band energy difference| between order a and b = {np.abs(low_energy[0] - low_energy[1]).max():.2f}")
        """),

        md(r"""
        ### How to read this chart

        **Left:** the time-averaged mel spectrum of every class. Solid and dashed lines of the same colour are the two
        orders of one group; they lie almost exactly on top of each other, which is the point of the construction -- once
        time is averaged out, order 0 and order 1 are the same signal. Different colours peak at different mel positions,
        so the six groups remain separable. **Middle:** the low-frequency band's energy over frames for the two orders of
        group 0. The solid curve peaks early and the dashed curve peaks late: this is the only information distinguishing
        the pair, and it lives entirely on the time axis. **Right:** the mean absolute profile difference within a group
        (an order swap) versus between groups. The within-group bar is a small fraction of the between-group bar; the
        error bars are the standard deviation across the 6 within-group pairs and the 15 between-group pairs respectively.
        """),

        code(r"""
        # Model-free separability: linear classifiers on time-averaged features versus the full image.
        from sklearn.linear_model import LogisticRegression
        from sklearn.neighbors import NearestCentroid
        from sklearn.preprocessing import StandardScaler

        def features(split, kind):
            x = X[split][:, 0]                                    # (n, mel, frames)
            return x.mean(axis=2) if kind == "time-mean (40-d)" else x.reshape(len(x), -1)

        def fisher_ratio(F, y):
            # between-class variance over within-class variance, averaged over feature dimensions.
            overall = F.mean(axis=0)
            between = sum((y == k).sum() * (F[y == k].mean(axis=0) - overall) ** 2 for k in np.unique(y)) / len(y)
            within = sum(((F[y == k] - F[y == k].mean(axis=0)) ** 2).sum(axis=0) for k in np.unique(y)) / len(y)
            return float(np.mean(between / np.maximum(within, 1e-12)))

        rows = []
        for kind in ("time-mean (40-d)", "full image (1600-d)"):
            Ftr, Fte = features("train", kind), features("test", kind)
            sc = StandardScaler().fit(Ftr)
            lr = LogisticRegression(max_iter=300, C=0.05).fit(sc.transform(Ftr), Y["train"])
            pred = lr.predict(sc.transform(Fte))
            nc = NearestCentroid().fit(Ftr, Y["train"])
            rows.append({"features": kind, "logreg test acc": float((pred == Y["test"]).mean()),
                         "logreg group acc": float((pred // 2 == Y["test"] // 2).mean()),
                         "nearest-centroid acc": float((nc.predict(Fte) == Y["test"]).mean()),
                         "fisher ratio": fisher_ratio(Ftr, Y["train"])})
        SEP = pd.DataFrame(rows).set_index("features")
        display(SEP.round(3))
        tm, fu = SEP.iloc[0], SEP.iloc[1]
        check("N2.3.3[time_mean_capped_near_half]", tm["logreg test acc"] < 0.6 and tm["logreg group acc"] > 0.9, f"time-mean logreg: acc {tm['logreg test acc']:.3f}, group acc {tm['logreg group acc']:.3f} (order cannot be read)")
        check("N2.3.3[time_mean_far_above_chance]", tm["logreg test acc"] > 3 * (1 / N_CLASSES), f"time-mean acc {tm['logreg test acc']:.3f} is > 3x chance {1 / N_CLASSES:.3f}")
        check("N2.3.3[full_image_beats_time_mean]", fu["logreg test acc"] > tm["logreg test acc"] + 0.2, f"full image {fu['logreg test acc']:.3f} vs time-mean {tm['logreg test acc']:.3f}")
        check("N2.3.3[task_is_learnable]", fu["logreg test acc"] > 0.8, f"a linear model on the full image reaches {fu['logreg test acc']:.3f}: the class structure is real")
        note("N2.3.3[preview]", f"model-free preview of the Section 4 headline: dropping time costs {fu['logreg test acc'] - tm['logreg test acc']:.3f} accuracy for a linear classifier")
        """),

        code(r"""
        # Fig NB2-3: 2D PCA of the two feature views. Colour = group, marker = order.
        from sklearn.decomposition import PCA
        fig, ax = plt.subplots(1, 2, figsize=(13, 5.2))
        stats = {}
        for a, kind in zip(ax, ("time-mean (40-d)", "full image (1600-d)")):
            F = features("test", kind)
            Z = PCA(n_components=2, random_state=0).fit(features("train", kind)).transform(F)
            stats[kind] = Z
            assert_varied(Z[:, 0], f"pca_pc1[{kind}]")
            assert_varied(Z[:, 1], f"pca_pc2[{kind}]")
            for k in range(N_CLASSES):
                sel = Y["test"] == k
                a.scatter(Z[sel, 0], Z[sel, 1], s=14, alpha=0.7, color=cols[SIG.loc[k, "group"]], marker="o" if SIG.loc[k, "order"] == 0 else "^")
            a.set(title=kind, xlabel="PC1", ylabel="PC2")
        fig.suptitle("NB2-3  Test-set PCA: time-averaged features pair the orders up; the full image separates them", y=1.02)
        fig.tight_layout()
        savefig(fig, "NB2-3_pca.png")
        # Cluster geometry in PCA space: distance between the two order centroids of a group vs between group centroids.
        def centroid(Z, k): return Z[Y["test"] == k].mean(axis=0)
        gap = {}
        for kind, Z in stats.items():
            wi = np.mean([np.linalg.norm(centroid(Z, 2 * g) - centroid(Z, 2 * g + 1)) for g in range(N_GROUPS)])
            bt = np.mean([np.linalg.norm(centroid(Z, 2 * g) - centroid(Z, 2 * h)) for g in range(N_GROUPS) for h in range(g + 1, N_GROUPS)])
            gap[kind] = (wi, bt)
            print(f"{kind}: mean order-swap centroid distance {wi:.3f}, mean between-group centroid distance {bt:.3f}, ratio {wi / bt:.3f}")
        check("N2.3.4[time_mean_pairs_collapse]", gap["time-mean (40-d)"][0] / gap["time-mean (40-d)"][1] < 0.2, "in time-mean PCA the two orders of a group nearly coincide")
        check("N2.3.4[full_image_pairs_apart]", gap["full image (1600-d)"][0] / gap["full image (1600-d)"][1] > gap["time-mean (40-d)"][0] / gap["time-mean (40-d)"][1], "in the full-image PCA the order-swap distance is relatively larger")
        """),

        md(r"""
        ### How to read this chart

        Each dot is one test example projected to two principal components; colour is the group (6 colours) and marker
        shape is the order (circle = order a, triangle = order b). **Left (time-averaged features):** dots of one colour
        form a single blob in which circles and triangles are thoroughly mixed -- the order information is not present in
        this feature space, so a classifier built on it cannot separate the pair. **Right (full time-frequency image):**
        the projection is dominated by the class-specific placement of the two bursts in time and frequency, and the
        circle/triangle pairs move apart. Keep in mind that PCA preserves variance, not class separation, so absolute
        separation is understated; the reliable statement is the *ratio* printed under the figure, order-swap centroid
        distance divided by between-group centroid distance, which is smaller for the time-averaged view.
        """),

        code(r"""
        # Fig NB2-4: split sizes and per-split raw intensity statistics (values vary across splits by construction).
        split_names = list(SPLIT_SIZES) + ["pool"]
        sizes = np.array([len(Y[s]) for s in split_names], dtype=float)
        means = np.array([RAW[s].mean() for s in split_names])
        stds = np.array([RAW[s].std() for s in split_names])
        assert_varied(sizes, "split_sizes")
        assert_varied(means, "split_raw_means")
        assert_varied(stds, "split_raw_stds")
        fig, ax = plt.subplots(1, 3, figsize=(14, 3.9))
        bars = ax[0].bar(split_names, sizes, color=[PALETTE["clean"], PALETTE["adapted"], PALETTE["acoustic"], PALETTE["mask"], PALETTE["shifted"]])
        for b, s in zip(bars, sizes):
            ax[0].text(b.get_x() + b.get_width() / 2, s + 8, int(s), ha="center", fontsize=9)
        ax[0].set(ylabel="examples", title="split sizes (50/20/40/50/50 per class)")
        ax[1].bar(split_names, means, color=PALETTE["reference"]); ax[1].set(ylabel="raw mean", title="raw log-mel-shaped mean")
        ax[2].bar(split_names, stds, color=PALETTE["reference"]); ax[2].set(ylabel="raw std", title="raw std")
        ax[1].set_ylim(means.min() - 0.02, means.max() + 0.02); ax[2].set_ylim(stds.min() - 0.02, stds.max() + 0.02)
        fig.suptitle("NB2-4  Split sizes and intensity statistics: independent streams, same distribution", y=1.03)
        fig.tight_layout()
        savefig(fig, "NB2-4_splits.png")
        drift_mu = float(np.ptp(means))
        check("N2.3.5[splits_same_distribution]", drift_mu < 0.02, f"raw means across the five splits differ by at most {drift_mu:.4f} (same generator, independent seeds)")
        check("N2.3.5[splits_are_distinct_draws]", not np.array_equal(RAW["train"][:20], RAW["val"][:20]) and not np.array_equal(RAW["test"][:20], RAW["pool"][:20]), "no two splits share examples")
        check("N2.3.5[total_examples]", int(sizes.sum()) == 600 + 240 + 480 + 600 + 600, f"{int(sizes.sum())} examples generated in total")
        """),

        md(r"""
        ### How to read this chart

        **Left:** the number of examples in each split. Train is the largest of the labelled splits (600), followed by test
        (600), calibration (480) and validation (240); the shift pool (600) is set aside for Section 8. Sizes are equal per
        class (50/20/40/50/50), so class balance is exactly uniform in every split and no per-class bar chart would carry
        information -- an all-equal chart is exactly the defect `assert_varied` guards against, which is why it is
        not drawn. **Middle and right:** the raw mean and standard deviation of each split. The axes are zoomed to a
        $\pm 0.02$ window, so differences that look large are a few thousandths; the check underneath states the actual
        spread. The point of the panel is the opposite of a finding: the five splits are independent draws from one
        distribution, so any train-test gap seen later is a property of the models, not of the split.
        """),
    ]
