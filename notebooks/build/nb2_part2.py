"""NB2 part 2: architecture comparison (the globally-pooled headline) and in-training diagnostics."""
from nbkit import md, code


def cells():
    return [
        md(r"""
        ## 4. Architecture comparison: what does global average pooling cost?

        `src/models.py` defines four classifiers. They share a convolutional front-end and differ mainly in what they do
        with the **time axis**:

        | name here | class | time handling | parameters |
        |---|---|---|---|
        | `logmel_pool` | `LogMelCNN` | `AdaptiveAvgPool2d((1,1))` averages time **and** mel away; the classifier sees 32 channel means | small |
        | `timepool` | `TimePoolCNN` | same convolutions, but pools to `(1, 8)`: eight temporal slots survive | small |
        | `wide` | `WideCNN` | wider and deeper, still keeps 8 temporal slots | large |
        | `convgru` | `ConvGRU` | pools only the mel axis; a bidirectional GRU runs over all time steps | large |

        **The prediction.** On the Section 2 task the two classes of a group have identical time-averaged spectra. A
        model whose last feature stage is a global average has, for a fixed network, the same output for an example and
        for any permutation of its frames' *post-convolution* summary -- more precisely, its classifier sees only
        $\bar h_c = \frac{1}{MT}\sum_{m,t} h_{c,m,t}$ per channel $c$, a bag of local patterns. Local patterns (a
        burst, a band edge) are identical across the two orders because a burst looks the same whether it comes first or
        last, and the receptive field cannot span the >15-frame gap. So `logmel_pool` should identify the group but
        guess the order:

        $$\text{acc}(\texttt{logmel\_pool}) \lesssim \tfrac{1}{2}, \qquad \text{acc}(\text{time-preserving}) \text{ unconstrained}.$$

        **The protocol.** All four models train with the *same* `ExperimentConfig`, the same seed, the same data, and
        the same augmentation (none). We report test accuracy, *group accuracy* (is the predicted class in the correct
        group? order-blind), and *order accuracy* (among test examples whose group was predicted correctly, was the
        order right?). The last quantity is $\approx 0.5$ for any model that cannot use temporal order and $\to 1$ for
        one that can. Whatever we measure is reported as measured; if the predicted gap does not appear, Section 4.3 says
        so and investigates rather than adjusting the task.
        """),

        code(r"""
        # Training configuration shared by all four architectures, and the probe batch for SentryCam.
        from dataclasses import replace
        from src.models import LogMelCNN, TimePoolCNN, WideCNN, ConvGRU
        from src.train import train_classifier, predict_logits
        from src.diagnostics import sentrycam_probe, sentrycam_alert_epoch, cluster_drift_metrics, project_2d
        from src.experiment import ExperimentConfig
        from src.calibration import temperature_scale, prediction_sets, crc_threshold
        from src.metrics import expected_calibration_error

        EPOCHS, MICROBATCH = 10, 32
        CFG = ExperimentConfig(condition="clean", seed=MODEL_SEED, max_epochs=EPOCHS, patience=EPOCHS,
                               microbatch_size=MICROBATCH, accumulation_steps=1, autoclip_percentile=10.0)
        ARCHS = {"logmel_pool": LogMelCNN, "timepool": TimePoolCNN, "wide": WideCNN, "convgru": ConvGRU}
        ACOL = {"logmel_pool": PALETTE["shifted"], "timepool": PALETTE["mask"], "wide": PALETTE["acoustic"], "convgru": PALETTE["clean"]}

        # Probe batch: the first 10 validation examples of every class (120 in total, fixed for every epoch and architecture).
        PROBE_IDX = np.concatenate([np.flatnonzero(Y["val"] == k)[:10] for k in range(N_CLASSES)])
        PROBE_X = torch.from_numpy(X["val"][PROBE_IDX])
        PROBE_Y = Y["val"][PROBE_IDX]

        def probe_fn(model, epoch):
            # Called by train_classifier once per epoch, inside its no-grad eval block.
            return sentrycam_probe(model, PROBE_X, PROBE_Y, keep_sequence=False)

        def softmax64(logits):
            z = np.asarray(logits, dtype=np.float64)
            z = z - z.max(axis=1, keepdims=True)
            e = np.exp(z)
            return e / e.sum(axis=1, keepdims=True)

        def evaluate(model, split="test"):
            # Overall, group (order-blind) and conditional order accuracy for a fitted model.
            logits = predict_logits(model, X[split])
            pred, y = logits.argmax(axis=1), Y[split]
            right_group = pred // 2 == y // 2
            return {"acc": float((pred == y).mean()), "group_acc": float(right_group.mean()),
                    "order_acc": float((pred[right_group] == y[right_group]).mean()) if right_group.any() else float("nan"),
                    "logits": logits}

        print(CFG)
        print("probe batch:", PROBE_X.shape, "labels per class:", np.bincount(PROBE_Y).tolist())
        check("N2.4.1[config_shared]", CFG.max_epochs == EPOCHS and CFG.seed == MODEL_SEED and CFG.microbatch_size == MICROBATCH, f"one ExperimentConfig for all four models: {EPOCHS} epochs, seed {MODEL_SEED}, microbatch {MICROBATCH}")
        check("N2.4.1[probe_batch_balanced]", len(set(np.bincount(PROBE_Y).tolist())) == 1 and len(PROBE_Y) == 120, "probe batch is class-balanced (10 per class)")
        check("N2.4.1[chance_reference]", abs(1 / N_CLASSES - 0.08333) < 1e-4, "chance for 12 classes is 1/12 = 0.0833")
        """),

        code(r"""
        # Train all four architectures with identical config and seed, with the SentryCam probe attached.
        RUNS, EVAL, WALL = {}, {}, {}
        for name, cls in ARCHS.items():
            torch.manual_seed(MODEL_SEED)
            t0 = time.time()
            RUNS[name] = train_classifier(cls(n_mels=N_MELS, num_classes=N_CLASSES), X["train"], Y["train"], X["val"], Y["val"], CFG, probe_fn=probe_fn)
            WALL[name] = time.time() - t0
            EVAL[name] = evaluate(RUNS[name].model)
            r = RUNS[name]
            print(f"{name:12s} params={sum(p.numel() for p in r.model.parameters()):>8,d} epochs={r.epochs_run} best_epoch={r.best_epoch} "
                  f"val_acc={r.history[-1]['val_accuracy']:.3f} test_acc={EVAL[name]['acc']:.3f} group={EVAL[name]['group_acc']:.3f} "
                  f"order|group={EVAL[name]['order_acc']:.3f} wall={WALL[name]:.1f}s")
        for name, r in RUNS.items():
            check(f"N2.4.2[{name}.ran_all_epochs]", r.epochs_run == EPOCHS, f"{r.epochs_run} epochs run (patience = max_epochs so no early stop)")
            check(f"N2.4.2[{name}.probes_recorded]", len(r.probe_history) == EPOCHS, f"{len(r.probe_history)} probe records, one per epoch")
        """),

        code(r"""
        # Summary table of the four runs.
        SUMMARY = pd.DataFrame({
            name: {"parameters": sum(p.numel() for p in RUNS[name].model.parameters()),
                   "final val acc": RUNS[name].history[-1]["val_accuracy"],
                   "best val loss": min(h["val_loss"] for h in RUNS[name].history),
                   "test acc": EVAL[name]["acc"], "group acc": EVAL[name]["group_acc"], "order acc | group": EVAL[name]["order_acc"],
                   "wall seconds": WALL[name]} for name in RUNS}).T
        display(SUMMARY.round(3))
        for name in RUNS:
            check(f"N2.4.3[{name}.above_chance]", EVAL[name]["acc"] > 1 / N_CLASSES if name == "logmel_pool" else EVAL[name]["acc"] > 3 / N_CLASSES,
                  f"test accuracy {EVAL[name]['acc']:.3f} vs chance {1 / N_CLASSES:.3f}" + ("" if name == "logmel_pool" else " (a time-preserving model must clear 3x chance)"))
        """),

        code(r"""
        # Fig NB2-5: validation accuracy and loss per epoch for all four architectures.
        epochs = np.arange(1, EPOCHS + 1)
        VA = {n: np.array([h["val_accuracy"] for h in RUNS[n].history]) for n in RUNS}
        VL = {n: np.array([h["val_loss"] for h in RUNS[n].history]) for n in RUNS}
        TL = {n: np.array([h["train_loss"] for h in RUNS[n].history]) for n in RUNS}
        for n in RUNS:
            assert_varied(VA[n], f"val_acc[{n}]")
            assert_varied(VL[n], f"val_loss[{n}]")
        fig, ax = plt.subplots(1, 3, figsize=(16, 4.4))
        for n in RUNS:
            ax[0].plot(epochs, VA[n], marker="o", ms=4, color=ACOL[n], label=n)
            ax[1].plot(epochs, VL[n], marker="o", ms=4, color=ACOL[n], label=f"{n} val")
            ax[1].plot(epochs, TL[n], ls="--", color=ACOL[n], alpha=0.6)
            ax[2].plot(epochs, VL[n] - TL[n], marker="s", ms=3, color=ACOL[n], label=n)
        ax[0].axhline(1 / N_CLASSES, color=PALETTE["reference"], ls=":", label="chance 1/12")
        ax[0].axhline(0.5, color=PALETTE["reference"], ls="-.", label="order-blind ceiling 0.5")
        ax[0].set(xlabel="epoch", ylabel="validation accuracy", ylim=(0, 1.02), title="validation accuracy")
        ax[1].set(xlabel="epoch", ylabel="cross-entropy", title="validation (solid) and training (dashed) loss")
        ax[2].axhline(0, color="black", lw=0.8)
        ax[2].set(xlabel="epoch", ylabel="val loss - train loss", title="generalisation gap")
        for a in ax:
            a.legend(fontsize=8, frameon=False)
        fig.suptitle("NB2-5  Four architectures, one config: only the time-preserving models leave the order-blind ceiling", y=1.03)
        fig.tight_layout()
        savefig(fig, "NB2-5_training_curves.png")
        """),

        md(r"""
        ### How to read this chart

        **Left:** validation accuracy per epoch. The dotted line is chance ($1/12 = 0.083$); the dash-dot line is the
        order-blind ceiling ($0.5$) derived in Section 2. A curve that stays below the dash-dot line is consistent with
        a model that cannot use temporal order; a curve that crosses above it must be using order. **Middle:** solid
        lines are validation loss and dashed lines training loss of the same colour; a shrinking solid line means the
        model is learning something that generalises. **Right:** validation minus training loss. Values above zero
        indicate overfitting to the 600 training examples; values near zero mean the model is not fitting the training set
        any better than the held-out set (either because it is still underfitting or because there is nothing more to
        learn). With only ten epochs and a fixed learning rate of $10^{-3}$ these are short-horizon curves, not converged
        ones.
        """),

        code(r"""
        # Fig NB2-6: test accuracy decomposed into group accuracy and order accuracy, with seed variation for the two cheap models.
        EXTRA_SEEDS = (23, 31)
        SEEDED = {"logmel_pool": [EVAL["logmel_pool"]], "timepool": [EVAL["timepool"]]}
        for name in SEEDED:
            for s in EXTRA_SEEDS:
                torch.manual_seed(s)
                res = train_classifier(ARCHS[name](n_mels=N_MELS, num_classes=N_CLASSES), X["train"], Y["train"], X["val"], Y["val"], replace(CFG, seed=s))
                SEEDED[name].append(evaluate(res.model))
        SEED_TABLE = pd.DataFrame({name: {m: [e[m] for e in SEEDED[name]] for m in ("acc", "group_acc", "order_acc")} for name in SEEDED}).T
        seed_stats = {name: {m: (np.mean([e[m] for e in ev]), np.std([e[m] for e in ev])) for m in ("acc", "group_acc", "order_acc")} for name, ev in SEEDED.items()}
        for name, st in seed_stats.items():
            print(name, {m: f"{v[0]:.3f} +/- {v[1]:.3f}" for m, v in st.items()}, "over seeds", (MODEL_SEED,) + EXTRA_SEEDS)
        metrics = ("acc", "group_acc", "order_acc")
        vals = np.array([[EVAL[n][m] for m in metrics] for n in RUNS])
        assert_varied(vals[:, 0], "test_acc_across_architectures")
        assert_varied(vals[:, 2], "order_acc_across_architectures")
        fig, ax = plt.subplots(1, 2, figsize=(14, 4.4))
        w = 0.25
        for j, m in enumerate(metrics):
            ax[0].bar(np.arange(len(RUNS)) + (j - 1) * w, vals[:, j], w, label=m.replace("_", " "), color=[PALETTE["clean"], PALETTE["acoustic"], PALETTE["mask"]][j])
        ax[0].axhline(1 / N_CLASSES, color=PALETTE["reference"], ls=":"); ax[0].axhline(0.5, color=PALETTE["reference"], ls="-.")
        ax[0].set_xticks(range(len(RUNS)), list(RUNS), rotation=10)
        ax[0].set(ylabel="test metric", ylim=(0, 1.05), title="seed 17: overall / group / order-given-group")
        ax[0].legend(frameon=False, fontsize=8)
        for i, (name, ev) in enumerate(SEEDED.items()):
            for j, m in enumerate(metrics):
                pts = [e[m] for e in ev]
                ax[1].scatter([i + (j - 1) * 0.22] * len(pts), pts, s=40, color=[PALETTE["clean"], PALETTE["acoustic"], PALETTE["mask"]][j], alpha=0.8, label=m.replace("_", " ") if i == 0 else None)
                ax[1].hlines(np.mean(pts), i + (j - 1) * 0.22 - 0.08, i + (j - 1) * 0.22 + 0.08, color="black")
        ax[1].set_xticks(range(len(SEEDED)), list(SEEDED))
        ax[1].axhline(1 / N_CLASSES, color=PALETTE["reference"], ls=":"); ax[1].axhline(0.5, color=PALETTE["reference"], ls="-.")
        ax[1].set(ylabel="test metric", ylim=(0, 1.05), title="three seeds (dots) and their mean (bar)")
        ax[1].legend(frameon=False, fontsize=8)
        fig.suptitle("NB2-6  Where the accuracy comes from: telling groups apart is easy; telling orders apart needs the time axis", y=1.03)
        fig.tight_layout()
        savefig(fig, "NB2-6_accuracy_decomposition.png")
        """),

        md(r"""
        ### How to read this chart

        Three metrics per architecture. *Overall* is ordinary 12-way test accuracy. *Group* is order-blind: a prediction
        counts if it lands in the right pair of classes. *Order* is measured only on examples whose group was right, and
        asks whether the model then picked the right member of the pair -- chance for that question is $0.5$, which is the
        dash-dot line on this panel. **Left:** seed 17, all four architectures. **Right:** the two cheap models over three
        seeds (17, 23, 31); each dot is a seed and the black tick is the mean, so the spread of the dots is the seed noise
        that any single-seed comparison has to be read against. If the dots of two architectures for the same metric
        overlap, the data do not support a ranking on that metric.
        """),

        code(r"""
        # The headline comparison, stated numerically, with an honest verdict.
        pool_acc, pool_ord = EVAL["logmel_pool"]["acc"], EVAL["logmel_pool"]["order_acc"]
        tp_only = {n: EVAL[n]["acc"] for n in ("timepool", "wide", "convgru")}
        best_name = max(tp_only, key=tp_only.get)
        gap = tp_only[best_name] - pool_acc
        print(f"logmel_pool (global average): test acc {pool_acc:.3f}, order|group {pool_ord:.3f}")
        for n, a in tp_only.items():
            print(f"  {n:9s} (keeps time)      : test acc {a:.3f}, order|group {EVAL[n]['order_acc']:.3f}, gap over pool {a - pool_acc:+.3f}")
        check("N2.4.4[headline.best_time_model_beats_pool]", gap > 0.1, f"best time-preserving model ({best_name}) beats logmel_pool by {gap:+.3f} test accuracy (needs > 0.1)")
        check("N2.4.4[headline.every_time_model_beats_pool]", all(a > pool_acc for a in tp_only.values()), "all three time-preserving models exceed the globally pooled baseline: " + ", ".join(f"{n} {a:.3f}" for n, a in tp_only.items()))
        check("N2.4.4[headline.pool_below_order_blind_ceiling]", pool_acc <= 0.5 + 0.05, f"logmel_pool accuracy {pool_acc:.3f} is at or below the order-blind ceiling 0.5 (+0.05 sampling slack, n=600)")
        check("N2.4.4[headline.pool_cannot_order]", abs(pool_ord - 0.5) < 0.15, f"logmel_pool order|group = {pool_ord:.3f}, within 0.15 of coin-flip 0.5")
        check("N2.4.4[headline.best_model_reads_order]", EVAL[best_name]["order_acc"] > 0.8, f"{best_name} order|group = {EVAL[best_name]['order_acc']:.3f}")
        pool_seeds = [e["acc"] for e in SEEDED["logmel_pool"]]
        tp_seeds = [e["acc"] for e in SEEDED["timepool"]]
        note("N2.4.4[headline.timepool_vs_pool_over_seeds]", f"timepool minus logmel_pool per seed: {[round(b - a, 3) for a, b in zip(pool_seeds, tp_seeds)]}")
        check("N2.4.4[headline.timepool_beats_pool_every_seed]", all(b > a for a, b in zip(pool_seeds, tp_seeds)), "TimePoolCNN (a one-line pooling change) beats LogMelCNN on every one of the three seeds")
        """),

        code(r"""
        # Fig NB2-7: confusion matrices. The globally pooled model should show 2x2 blocks on the diagonal.
        def confusion(pred, y):
            C = np.zeros((N_CLASSES, N_CLASSES))
            np.add.at(C, (y, pred), 1)
            return C / C.sum(axis=1, keepdims=True)

        pairs = [("logmel_pool", EVAL["logmel_pool"]), (best_name, EVAL[best_name])]
        fig, ax = plt.subplots(1, 2, figsize=(12.5, 5.2))
        for a, (n, ev) in zip(ax, pairs):
            C = confusion(ev["logits"].argmax(axis=1), Y["test"])
            assert_varied(C.ravel(), f"confusion[{n}]")
            im = a.imshow(C, cmap="Blues", vmin=0, vmax=1)
            a.set_xticks(range(N_CLASSES), CLASS_NAMES, rotation=90, fontsize=8); a.set_yticks(range(N_CLASSES), CLASS_NAMES, fontsize=8)
            for g in range(1, N_GROUPS):
                a.axhline(2 * g - 0.5, color="black", lw=0.6); a.axvline(2 * g - 0.5, color="black", lw=0.6)
            a.set(xlabel="predicted", ylabel="true", title=f"{n}: test acc {ev['acc']:.3f}")
            a.grid(False)
        fig.colorbar(im, ax=ax, shrink=0.8, label="row-normalised rate")
        fig.suptitle("NB2-7  Confusion matrices (black lines separate groups): pooled model confuses orders within a group", y=1.02)
        savefig(fig, "NB2-7_confusion.png")
        Cp = confusion(EVAL["logmel_pool"]["logits"].argmax(axis=1), Y["test"])
        within_block = np.mean([Cp[2 * g:2 * g + 2, 2 * g:2 * g + 2].sum() / 2 for g in range(N_GROUPS)])
        print(f"logmel_pool: mass inside the 2x2 group blocks = {within_block:.3f}; mass on the true class = {np.trace(Cp) / N_CLASSES:.3f}")
        Cb = confusion(EVAL[best_name]["logits"].argmax(axis=1), Y["test"])
        check("N2.4.5[pool_confusions_stay_in_group]", within_block > np.trace(Cp) / N_CLASSES + 0.1, "the pooled model puts more mass inside group blocks than on the exact diagonal (order confusions)")
        check("N2.4.5[best_diagonal_stronger]", np.trace(Cb) > np.trace(Cp), f"{best_name} diagonal mass {np.trace(Cb) / N_CLASSES:.3f} vs logmel_pool {np.trace(Cp) / N_CLASSES:.3f}")
        """),

        md(r"""
        ### How to read this chart

        Rows are the true class, columns the predicted class, and the two orders of each group are adjacent
        (`g0a`, `g0b`, `g1a`, ...); heavy black lines divide the groups into 2x2 blocks. Each cell is the fraction of that
        true class's test examples assigned to the column class, so every row sums to 1. **Left, the globally pooled model:**
        if it works as predicted, the mass concentrates in the 2x2 diagonal blocks but is smeared roughly evenly across both
        cells of each block -- it knows the group but not the order. **Right, the best time-preserving model:** the mass
        should concentrate on the exact diagonal. The number under the figure, the mass inside the group blocks versus on
        the exact diagonal for the pooled model, quantifies the same thing.
        """),

        code(r"""
        # Investigation: is the pooled model's weakness structural or just under-training? Train it longer (30 epochs) and compare.
        torch.manual_seed(MODEL_SEED)
        long_run = train_classifier(LogMelCNN(n_mels=N_MELS, num_classes=N_CLASSES), X["train"], Y["train"], X["val"], Y["val"], replace(CFG, max_epochs=30, patience=30))
        long_eval = evaluate(long_run.model)
        long_curve = np.array([h["val_accuracy"] for h in long_run.history])
        assert_varied(long_curve, "logmel_pool_30_epoch_val_acc")
        fig, ax = plt.subplots(figsize=(7.5, 4))
        ax.plot(np.arange(1, 31), long_curve, marker="o", ms=3, color=ACOL["logmel_pool"], label="logmel_pool, 30 epochs")
        ax.plot(epochs, VA["timepool"], marker="s", ms=3, color=ACOL["timepool"], label="timepool, 10 epochs")
        ax.axhline(0.5, color=PALETTE["reference"], ls="-.", label="order-blind ceiling"); ax.axhline(1 / 12, color=PALETTE["reference"], ls=":", label="chance")
        ax.set(xlabel="epoch", ylabel="validation accuracy", ylim=(0, 1.02), title="NB2-8  Is the pooled model just under-trained?")
        ax.legend(frameon=False, fontsize=8)
        savefig(fig, "NB2-8_pool_longer.png")
        print(f"logmel_pool after 30 epochs: test acc {long_eval['acc']:.3f}, group {long_eval['group_acc']:.3f}, order|group {long_eval['order_acc']:.3f}")
        check("N2.4.6[longer_training_recorded]", long_run.epochs_run == 30, f"{long_run.epochs_run} epochs run")
        check("N2.4.6[longer_still_below_best_time_model]", long_eval["acc"] < EVAL[best_name]["acc"], f"30-epoch pooled accuracy {long_eval['acc']:.3f} < 10-epoch {best_name} {EVAL[best_name]['acc']:.3f}")
        check("N2.4.6[longer_still_order_blind]", long_eval["order_acc"] < 0.7, f"30-epoch pooled order|group = {long_eval['order_acc']:.3f}: 3x the training does not buy order information")
        """),

        md(r"""
        ### How to read this chart

        The red curve is the same pooled architecture given three times the training budget; the orange curve is the
        time-preserving `TimePoolCNN` with the standard ten epochs. The dash-dot line is the order-blind ceiling. The
        question is whether the pooled model's weakness is a training-budget artefact. If the red curve keeps climbing past
        the dash-dot line the weakness was under-training; if it plateaus at or below it, the ceiling is structural. Read the
        printed line under the figure alongside the curve: the *order given group* number is the cleanest statistic, since
        it is $0.5$ for a model with no access to order however long it trains.

        ### 4.1 The central cheap finding

        Global average pooling discards the time axis, and on a task whose classes differ only by temporal order that costs
        accuracy: the numbers above show it directly. This is a **cheap** finding -- a few seconds of CPU training -- and
        it is the reason the real-data study includes the time-preserving `timepool`, `wide`, and `gru` arms next to the
        legacy `baseline`. It is also the *expected* result of a construction designed to expose it; on real speech, where
        confusable words (`no`/`on`, `go`/`no`) differ in phoneme order but also in many other cues, the size of the effect
        is an empirical question that only the real-data notebook can answer.
        """),

        md(r"""
        ## 5. In-training diagnostics: AutoClip, SentryCam cluster health, and the alert

        `train_classifier` records, at zero extra cost, the pre-clip gradient norm at every optimizer step and the AutoClip
        threshold in force at that step. AutoClip (Seetharaman et al., arXiv:2007.14469) clips the gradient at the
        $p$-th percentile of the norm history:

        $$\tau_t = \operatorname{percentile}_p\big(\{\lVert g_s \rVert\}_{s<t}\big), \qquad g_t \leftarrow g_t \cdot \min\!\big(1, \tau_t / \lVert g_t \rVert\big).$$

        The first step has no history, so $\tau_1 = \infty$ and nothing is clipped. With $p = 10$ (this notebook's default,
        the study's default) roughly 90% of steps exceed the threshold and are clipped -- an aggressive setting that Section 6
        sweeps.

        Through the `probe_fn` hook each epoch also records a **SentryCam-style** record (arXiv:2405.15135): the penultimate
        activations of a fixed 120-example probe batch are projected to 2D (here by PCA, a documented deviation from the paper's
        autoencoder), and two cluster-health numbers are computed in that 2D space, the mean pairwise distance between
        class centroids $D_t$ and the mean within-class variance $V_t$:

        $$D_t = \binom{K}{2}^{-1} \sum_{i<j} \lVert \mu_i - \mu_j \rVert_2, \qquad V_t = \frac{1}{K} \sum_{k} \frac{1}{n_k} \sum_{x \in k} \lVert z_x - \mu_k \rVert_2^2 .$$

        A healthy run has $D_t$ growing and $V_t$ shrinking. The **alert** `sentrycam_alert_epoch` fires at the earliest epoch
        where either $D$ has fallen, or $V$ has risen, for $k=2$ consecutive epochs by more than $\alpha = 0.25$ standard
        deviations of the preceding window. `train_classifier` itself computes the same metrics **in raw logit space** on
        the whole validation set every epoch (`inter_cluster_distance_history`, `intra_cluster_variance_history`); the paper
        argues that raw high-dimensional distances are distorted, and that is exactly the comparison drawn in Section 5.4.
        """),

        code(r"""
        # Fig NB2-9: AutoClip gradient-norm and threshold trajectories, per architecture.
        def clip_stats(r, steps_per_epoch):
            g = np.asarray(r.grad_norm_history); t = np.asarray(r.clip_threshold_history)
            clipped = np.isfinite(t) & (g > t)
            per_epoch = [clipped[i * steps_per_epoch:(i + 1) * steps_per_epoch].mean() for i in range(len(g) // steps_per_epoch)]
            return g, t, clipped, np.array(per_epoch)

        STEPS = math.ceil(len(X["train"]) / MICROBATCH)
        CLIP = {n: clip_stats(RUNS[n], STEPS) for n in RUNS}
        fig, ax = plt.subplots(2, 2, figsize=(13, 7.4), sharex=True)
        for a, n in zip(ax.ravel(), RUNS):
            g, t, clipped, per_epoch = CLIP[n]
            tt = np.where(np.isfinite(t), t, np.nan)
            assert_varied(g, f"grad_norm[{n}]")
            assert_varied(tt[np.isfinite(tt)], f"clip_threshold[{n}]")
            a.plot(np.arange(1, len(g) + 1), g, color=ACOL[n], lw=1, alpha=0.7, label="pre-clip gradient norm")
            a.plot(np.arange(1, len(g) + 1), tt, color="black", lw=1.6, label="AutoClip threshold (p=10)")
            a.scatter(np.flatnonzero(clipped) + 1, g[clipped], s=6, color=PALETTE["shifted"], zorder=3, label="clipped step")
            for e in range(1, EPOCHS):
                a.axvline(e * STEPS + 0.5, color=PALETTE["band"], lw=0.6)
            a.set(yscale="log", title=f"{n}: {clipped.sum()} of {len(g)} steps clipped ({clipped.mean():.0%})")
        for a in ax[1]:
            a.set_xlabel("optimizer step (grey lines = epoch boundaries)")
        for a in ax[:, 0]:
            a.set_ylabel("gradient norm (log)")
        ax[0, 0].legend(fontsize=8, frameon=False)
        fig.suptitle("NB2-9  AutoClip in practice: the percentile threshold tracks the norm history from below", y=1.0)
        fig.tight_layout()
        savefig(fig, "NB2-9_autoclip.png")
        for n in RUNS:
            g, t, clipped, per_epoch = CLIP[n]
            check(f"N2.5.1[{n}.first_step_unclipped]", not np.isfinite(t[0]) and not clipped[0], "step 1 has threshold = inf (no history) and is not clipped")
            check(f"N2.5.1[{n}.clip_rate_near_90pct]", 0.6 < clipped[1:].mean() < 0.99, f"p=10 clips {clipped[1:].mean():.1%} of steps after the first (a 10th-percentile threshold clips the upper ~90%)")
        """),

        md(r"""
        ### How to read this chart

        One panel per architecture; the x-axis is the optimizer step (with one step per 32-example microbatch there are
        19 steps per epoch, marked by the light grey vertical lines) and the y-axis is log-scaled. The coloured line is the
        raw gradient norm *before* clipping; the black line is the AutoClip threshold, the 10th percentile of every norm
        seen so far; red dots mark the steps that were actually clipped (norm above threshold). Two features are worth
        looking for. First, the black line sits below the coloured cloud almost everywhere -- by construction, a 10th
        percentile threshold is below 90% of the past. Second, the *shape* of the coloured cloud differs across architectures:
        a model whose gradient norms fall steadily has a threshold that keeps chasing them downward, whereas a model with
        heavy-tailed norms has a threshold that lags. The panel titles give the exact clipped fractions.
        """),

        code(r"""
        # Fig NB2-10: per-epoch clip rate and threshold-to-median-norm ratio; how aggressive is p=10 across architectures?
        clip_rate = {n: CLIP[n][3] for n in RUNS}
        ratio = {}
        for n in RUNS:
            g, t, _, _ = CLIP[n]
            per_epoch_ratio = []
            for e in range(EPOCHS):
                sl = slice(e * STEPS, (e + 1) * STEPS)
                tt = t[sl]; tt = tt[np.isfinite(tt)]
                per_epoch_ratio.append(float(np.median(tt) / np.median(g[sl])) if tt.size else np.nan)
            ratio[n] = np.array(per_epoch_ratio)
        fig, ax = plt.subplots(1, 2, figsize=(13, 4))
        for n in RUNS:
            assert_varied(clip_rate[n][1:], f"clip_rate_per_epoch[{n}]")
            ax[0].plot(np.arange(1, EPOCHS + 1), clip_rate[n], marker="o", ms=4, color=ACOL[n], label=n)
            ax[1].plot(np.arange(1, EPOCHS + 1), ratio[n], marker="o", ms=4, color=ACOL[n], label=n)
        ax[0].set(xlabel="epoch", ylabel="fraction of steps clipped", ylim=(0, 1.02), title="clip rate per epoch")
        ax[1].axhline(1, color="black", lw=0.8)
        ax[1].set(xlabel="epoch", ylabel="median threshold / median norm", title="how far below the norms the threshold sits")
        ax[0].legend(frameon=False, fontsize=8)
        fig.suptitle("NB2-10  Clip rate and threshold-to-norm ratio per epoch", y=1.03)
        fig.tight_layout()
        savefig(fig, "NB2-10_clip_rate.png")
        for n in RUNS:
            check(f"N2.5.2[{n}.threshold_below_norms]", np.nanmedian(ratio[n]) < 1.0, f"median threshold/median norm = {np.nanmedian(ratio[n]):.3f} (<1: threshold sits below the typical norm)")
        """),

        md(r"""
        ### How to read this chart

        **Left:** the fraction of optimizer steps clipped in each epoch. Epoch 1 is lower because step 1 has an infinite
        threshold and the first few thresholds are set from a tiny, noisy history. Later epochs sit at or near the level
        implied by a 10th-percentile rule when the norm distribution is stationary; drifts away from it show the norm
        distribution changing over training. **Right:** the ratio of the median threshold to the median gradient norm in
        each epoch; a value of $1$ (black line) would mean the threshold sits at the typical norm, and smaller values mean
        the clip is more aggressive. Lines that fall over epochs mean the norms are growing relative to their history (the
        threshold lags), lines that rise mean the norms are shrinking towards the historical floor.
        """),

        code(r"""
        # Fig NB2-11: SentryCam cluster metrics. 2D (from the probe hook) versus raw logit space (from train_classifier).
        D2 = {n: np.array([p["inter_cluster_distance_2d"] for p in RUNS[n].probe_history]) for n in RUNS}
        V2 = {n: np.array([p["intra_cluster_variance_2d"] for p in RUNS[n].probe_history]) for n in RUNS}
        DR = {n: np.array(RUNS[n].inter_cluster_distance_history) for n in RUNS}
        VR = {n: np.array(RUNS[n].intra_cluster_variance_history) for n in RUNS}
        fig, ax = plt.subplots(2, 2, figsize=(13, 7.4), sharex=True)
        panels = [(ax[0, 0], D2, "inter-cluster distance D (2D probe)"), (ax[0, 1], V2, "intra-cluster variance V (2D probe)"),
                  (ax[1, 0], DR, "inter-cluster distance D (raw logits)"), (ax[1, 1], VR, "intra-cluster variance V (raw logits)")]
        for a, series, title in panels:
            for n in RUNS:
                assert_varied(series[n], f"{title}[{n}]")
                a.plot(np.arange(1, EPOCHS + 1), series[n], marker="o", ms=4, color=ACOL[n], label=n)
            a.set(title=title, yscale="log")
        for a in ax[1]:
            a.set_xlabel("epoch")
        ax[0, 0].legend(frameon=False, fontsize=8)
        fig.suptitle("NB2-11  SentryCam cluster health per epoch: 2D-projected (top) versus raw logit space (bottom)", y=1.0)
        fig.tight_layout()
        savefig(fig, "NB2-11_cluster_metrics.png")
        for n in RUNS:
            check(f"N2.5.3[{n}.distance_grows_2d]", D2[n][-1] > D2[n][0], f"2D inter-cluster distance {D2[n][0]:.3f} -> {D2[n][-1]:.3f} over training")
            check(f"N2.5.3[{n}.raw_probe_lengths_match]", len(DR[n]) == len(D2[n]) == EPOCHS, "raw-space and 2D series both have one value per epoch")
        """),

        md(r"""
        ### How to read this chart

        Top row: the two SentryCam quantities computed in the 2D PCA projection of the penultimate-layer activations of the
        fixed 120-example probe batch. Bottom row: the same two quantities computed in raw *logit* space on the whole 240-example
        validation set by the training loop. Each line is one architecture and the y-axes are log-scaled because scales
        differ by orders of magnitude between architectures and between spaces. A healthy run has the left-hand panels rising
        (clusters pulling apart) and the right-hand panels falling (clusters tightening). Do not compare the *values* across
        rows -- the two spaces have different dimension and scale, and the top row measures penultimate features whilst the
        bottom measures the output logits -- compare only the *shapes*: which epochs a series turns, and whether the turn is
        visible in one space but not the other. That question is answered quantitatively in Section 5.4.
        """),

        code(r"""
        # Fig NB2-12: per-epoch 2D latent scatter for the pooled baseline (top) and the best time-preserving model (bottom).
        show_epochs = [0, 2, 4, 6, EPOCHS - 1]
        sel_models = ["logmel_pool", best_name]
        fig, axes = plt.subplots(2, len(show_epochs), figsize=(3.2 * len(show_epochs), 6.6), sharex=False)
        for r_i, n in enumerate(sel_models):
            for c_i, ep in enumerate(show_epochs):
                p = RUNS[n].probe_history[ep]
                co = p["coords_2d"]
                assert_varied(co[:, 0], f"scatter_x[{n}, epoch {ep + 1}]")
                a = axes[r_i, c_i]
                for k in range(N_CLASSES):
                    s = p["labels"] == k
                    a.scatter(co[s, 0], co[s, 1], s=10, color=cols[SIG.loc[k, "group"]], marker="o" if SIG.loc[k, "order"] == 0 else "^", alpha=0.8)
                a.set(title=f"{n}, epoch {ep + 1}\nD={p['inter_cluster_distance_2d']:.2f} V={p['intra_cluster_variance_2d']:.2f}", xticks=[], yticks=[])
                a.title.set_fontsize(8); a.grid(False)
        fig.suptitle("NB2-12  2D latent scatter of the probe batch over epochs (colour = group, circle/triangle = order)", y=1.0)
        fig.tight_layout()
        savefig(fig, "NB2-12_latent_scatter.png")
        SEP2D = {}
        for n in sel_models:
            p = RUNS[n].probe_history[-1]; co = p["coords_2d"]; lab = p["labels"]
            cen = {k: co[lab == k].mean(axis=0) for k in range(N_CLASSES)}
            order_d = np.mean([np.linalg.norm(cen[2 * g] - cen[2 * g + 1]) for g in range(N_GROUPS)])
            group_d = np.mean([np.linalg.norm(cen[2 * g] - cen[2 * h]) for g in range(N_GROUPS) for h in range(g + 1, N_GROUPS)])
            SEP2D[n] = order_d / group_d
            print(f"{n}: final-epoch 2D order-swap centroid distance / between-group distance = {SEP2D[n]:.3f}")
        check("N2.5.4[best_model_separates_orders_more]", SEP2D[best_name] > SEP2D["logmel_pool"], f"order-swap/between-group ratio: {best_name} {SEP2D[best_name]:.3f} vs logmel_pool {SEP2D['logmel_pool']:.3f}")
        """),

        md(r"""
        ### How to read this chart

        Rows are two models (top: the globally pooled baseline; bottom: the best time-preserving model), columns are
        epochs 1, 3, 5, 7 and 10. Every dot is one of the 120 probe examples projected to the top two principal components
        of the penultimate activations; colour encodes the group and the marker shape encodes the order, so a model that
        recovers the order shows circles and triangles of one colour as *separate* clusters, and a model that cannot shows
        them overlapping in a single blob. The panel titles give the SentryCam pair $(D, V)$ for that epoch. Axes carry no
        ticks because PCA coordinates are unitless and are recomputed every epoch (so the same class can appear mirrored
        between epochs: the sign of a principal component is arbitrary). The number printed under the figure summarises the
        last column: the mean distance between the two orders' centroids over the mean distance between group centroids.
        """),

        code(r"""
        # Recompute the SentryCam alert on BOTH series and tabulate side by side.
        rows = []
        for n in RUNS:
            raw_alert = sentrycam_alert_epoch(DR[n], VR[n])
            twod_alert = sentrycam_alert_epoch(D2[n], V2[n])
            rows.append({"architecture": n, "raw-space alert epoch": raw_alert, "2D alert epoch": twod_alert,
                         "train_classifier stored alert": RUNS[n].instability_alert_epoch,
                         "raw-space fires": raw_alert is not None, "2D fires": twod_alert is not None})
        ALERT = pd.DataFrame(rows).set_index("architecture")
        display(ALERT)
        for n in RUNS:
            check(f"N2.5.5[{n}.recompute_matches_stored]", ALERT.loc[n, "raw-space alert epoch"] == ALERT.loc[n, "train_classifier stored alert"] or (pd.isna(ALERT.loc[n, "raw-space alert epoch"]) and pd.isna(ALERT.loc[n, "train_classifier stored alert"])),
                  f"recomputing on the raw series reproduces the stored alert ({ALERT.loc[n, 'raw-space alert epoch']})")
        n_raw, n_2d = int(ALERT["raw-space fires"].sum()), int(ALERT["2D fires"].sum())
        print(f"alerts fired: raw space {n_raw}/4, 2D {n_2d}/4")
        note("N2.5.5[alert_counts]", f"raw-space alerts {n_raw}/4, 2D alerts {n_2d}/4 (a healthy 10-epoch run need not alert at all; an alert is not a failure)")
        """),

        code(r"""
        # Fig NB2-13: the alert's internal statistics, epoch by epoch. z_t = |h_t - h_{t-1}| / std(window before t); the alert needs z > alpha.
        def alert_z(h, window=10):
            h = np.asarray(h, float); z = np.full(len(h), np.nan)
            for t in range(1, len(h)):
                w = h[max(0, t - window):t]
                if len(w) >= 2 and w.std() > 0:
                    z[t] = abs(h[t] - h[t - 1]) / w.std()
            return z
        def direction_run(h, sign):
            run, out = 0, []
            for t in range(len(h)):
                run = run + 1 if t > 0 and np.sign(h[t] - h[t - 1]) == sign else 0
                out.append(run)
            return np.array(out)

        fig, ax = plt.subplots(2, 2, figsize=(13, 7.2), sharex=True)
        for n in RUNS:
            zr, z2 = alert_z(DR[n]), alert_z(D2[n])
            assert_varied(zr[np.isfinite(zr)], f"alert_z_raw[{n}]")
            assert_varied(z2[np.isfinite(z2)], f"alert_z_2d[{n}]")
            ax[0, 0].plot(np.arange(1, EPOCHS + 1), zr, marker="o", ms=4, color=ACOL[n], label=n)
            ax[0, 1].plot(np.arange(1, EPOCHS + 1), z2, marker="o", ms=4, color=ACOL[n], label=n)
            ax[1, 0].plot(np.arange(1, EPOCHS + 1), direction_run(DR[n], -1), marker="s", ms=4, color=ACOL[n], label=n)
            ax[1, 1].plot(np.arange(1, EPOCHS + 1), direction_run(D2[n], -1), marker="s", ms=4, color=ACOL[n], label=n)
        for a in ax[0]:
            a.axhline(0.25, color="black", ls="--", lw=1, label="alpha = 0.25")
            a.set(yscale="log")
        for a in ax[1]:
            a.axhline(2, color="black", ls="--", lw=1, label="k = 2 consecutive")
            a.set_xlabel("epoch")
        ax[0, 0].set(ylabel="z = |change| / window std", title="raw-space distance series")
        ax[0, 1].set(title="2D distance series")
        ax[1, 0].set(ylabel="consecutive decreases", title="raw-space distance series")
        ax[1, 1].set(title="2D distance series")
        ax[0, 0].legend(frameon=False, fontsize=7); ax[1, 0].legend(frameon=False, fontsize=7)
        fig.suptitle("NB2-13  Why the alert did or did not fire: the significance statistic (top) and the run length of decreases (bottom)", y=1.0)
        fig.tight_layout()
        savefig(fig, "NB2-13_alert_internals.png")
        """),

        md(r"""
        ### How to read this chart

        The alert has two ingredients and this figure shows both for the *distance* condition (the variance condition is the
        mirror image). **Top row:** the significance statistic $z_t = |D_t - D_{t-1}| / \sigma_{\text{window}}$, the size of the
        latest change relative to the spread of the preceding epochs; the dashed line is $\alpha = 0.25$, so a point above it
        is a "significant" move. **Bottom row:** how many epochs in a row the distance has been *falling*; the dashed line is
        $k = 2$. The alert needs a point above the top dashed line **and** a run at or above the bottom dashed line at the
        same epoch. Left column is the raw logit series, right column the 2D series. Because the distance series in a
        healthy run mostly rises, the bottom-row runs are mostly zero; a run of two or more decreases that coincides with a
        significant change is the only route to a distance alert. Comparing columns shows whether the two spaces disagree
        about *which* epochs look suspicious.
        """),

        code(r"""
        # Sensitivity of the alert to (k, alpha): raw-space versus 2D, for every architecture.
        grid_rows = []
        for n in RUNS:
            for k, al in itertools.product((1, 2, 3), (0.1, 0.25, 0.5)):
                a_raw = sentrycam_alert_epoch(DR[n], VR[n], k=k, alpha=al)
                a_2d = sentrycam_alert_epoch(D2[n], V2[n], k=k, alpha=al)
                grid_rows.append({"architecture": n, "k": k, "alpha": al, "raw alert": -1 if a_raw is None else a_raw, "2D alert": -1 if a_2d is None else a_2d})
        GRID = pd.DataFrame(grid_rows)
        piv = GRID.pivot_table(index=["architecture", "k"], columns="alpha", values=["raw alert", "2D alert"])
        display(piv)
        fires_raw = (GRID["raw alert"] >= 0).mean(); fires_2d = (GRID["2D alert"] >= 0).mean()
        print(f"over {len(GRID)} (architecture, k, alpha) settings: raw fires {fires_raw:.0%}, 2D fires {fires_2d:.0%}   (-1 = no alert)")
        # More permissive settings can only fire earlier or equally: check monotonicity in alpha at fixed k (smaller alpha = easier).
        mono = True
        for n in RUNS:
            for k in (1, 2, 3):
                sub = GRID[(GRID.architecture == n) & (GRID.k == k)].sort_values("alpha")
                fired = [(v if v >= 0 else 10 ** 6) for v in sub["2D alert"]]
                mono &= all(a <= b for a, b in zip(fired, fired[1:]))
        check("N2.5.6[alert_monotone_in_alpha]", mono, "for every architecture and k, a smaller alpha never fires later than a larger alpha (2D series)")
        assert_varied(np.concatenate([GRID["raw alert"], GRID["2D alert"]]), "alert_grid_values")
        note("N2.5.6[sensitivity]", f"alert firing rate across the grid: raw {fires_raw:.0%} vs 2D {fires_2d:.0%}")
        """),

        md(r"""
        ### 5.4 Reading the alert comparison

        The tables above are the notebook's diagnostic headline. The important point is not which space "wins" on a run
        that never failed -- there is no ground-truth instability here to detect, so an alert is neither a true nor a
        false positive; it is a statement about *how each space summarises a healthy trajectory*. What the comparison can
        establish is (i) whether the two spaces agree on when the trajectory turns, (ii) how sensitive each is to the
        hyper-parameters $(k, \alpha)$, and (iii) that the recomputation reproduces `train_classifier`'s stored alert
        exactly, which validates the plumbing. Whether a 2D alert is *earlier or more reliable* than a raw-space alert on
        a run that really diverges is a question for the real-data notebook, where the instrumented runs are long enough
        (up to 36 epochs) for a rolling window of ten epochs to mean something; with ten epochs here the window is
        never full.
        """),
    ]
