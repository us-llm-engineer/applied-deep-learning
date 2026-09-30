"""NB4 part 1: charter, provenance root, architecture comparison, checkpoint reload verification.

Everything below is computed from the copied-back real-run artifacts in runtime/metrics/ (git-ignored).
No synthetic or mock data appears anywhere in this notebook.
"""
from nbkit import md, code


def cells():
    return [
        md(r"""
        ## 1. Charter, provenance, and what this notebook is not

        This notebook is built **only** from artifacts that a real GPU training run produced (on the Colab runtime,
        stored on Google Drive) and that are mirrored locally in `<ARTIFACT_ROOT>/metrics/`: three `arch-*.result.json` files (real `ArmRun` output), three
        `arch-*.probes.npz` files (per-epoch in-training probes), three per-arm worker logs, one precomputed
        `mmphate_stats.npz`, and the superseded legacy training logs. Every printed number and every plotted point
        traces to one of those files. Where something cannot be computed from them, the text says so and the analysis
        is skipped -- nothing is substituted.

        **Where this notebook runs.** Its execution host is the remote Colab runtime, which has the GPU, Google Drive
        mounted at `/content/drive/MyDrive`, and therefore the checkpoints and the Speech Commands bundle next to the
        compute. The local machine is only the *authoring* host: it holds the editable source of truth
        (`notebooks/build/nb4_part*.py`) and a mirror of the `metrics/` folder for editing and review. Cells that need
        the remote-only resources (checkpoints, the dataset, a GPU) are tagged `offloaded`, so a local run skips
        them instead of failing; everything else runs anywhere. A single cell below resolves one `ARTIFACT_ROOT`
        (the Drive folder when it exists, otherwise the local `runtime/` mirror) and **prints which root and which host
        it used**, so a reader of an executed copy can see where the numbers came from. The local `runtime/` folder is
        **git-ignored**; a fresh clone contains no artifacts, and the resolution cell fails loudly rather than
        substituting anything.

        **Run configuration.** The run behind every result is `condition=clean`, `seed=17`, at most 36 epochs with
        early-stopping patience 8 (the `gru` arm early-stopped at epoch 28). The post-training summary block was
        produced with the **EXPLORATION configuration: 4 stress conditions and 200 bootstrap resamples**. These are
        exploration results and must never be quoted as publication numbers; the full 11-condition / 2000-resample
        publication evaluation was never run (see Section 8).

        **The four architectures.**

        | name | class in `src/models.py` | parameters | instrumented in this run |
        |---|---|---|---|
        | `baseline` | `LogMelCNN` | 14,524 | no -- legacy control, its `AdaptiveAvgPool2d((1,1))` leaves no temporal axis |
        | `timepool` | `TimePoolCNN` | 17,212 | yes |
        | `wide` | `WideCNN` | 271,692 | yes |
        | `gru` | `ConvGRU` | 291,564 | yes |

        **Sections.** Section 2 inventories the artifacts (sha256 and array shapes: the provenance root). Section 3
        compares the architectures and states an honest negative finding. Section 4 reloads the checkpoints and re-derives the recorded accuracy on the real
        validation split (offloaded). Section 5 covers the in-training diagnostics V1--V4 (training curves, AutoClip, SentryCam) and the
        headline raw-space-versus-2D alert comparison. Section 6 covers the MM-PHATE statistics V5--V8. Section 7 is
        the post-training results section. Section 8 is the limitations section.
        """),

        code(r"""
        # NB4 helpers: single ARTIFACT_ROOT resolution, artifact loading, figure saving. No silent fallback, no synthetic data.
        import json, hashlib, re, os, platform
        import pandas as pd
        # Plain-text DataFrame repr: Colab otherwise wraps every table in ~6KB of interactive HTML,
        # which bloats the .ipynb and the synced outputs without adding information.
        pd.set_option("display.notebook_repr_html", False)
        try:  # Colab also registers its own interactive-table HTML formatter for DataFrame;
            # drop it so tables render as plain text instead of ~3KB of embedded CSS/JS each.
            get_ipython().display_formatter.formatters["text/html"].type_printers.pop(pd.DataFrame, None)
        except Exception:
            pass
        # Plain-text DataFrame repr: Colab otherwise wraps every table in ~6KB of interactive HTML,
        # which bloats the .ipynb and the synced outputs without adding information.
        pd.set_option("display.notebook_repr_html", False)
        from IPython.display import display
        from scipy.stats import gaussian_kde, spearmanr

        DRIVE_ROOT = Path("/content/drive/MyDrive/deep-learning-first-project")
        LOCAL_ROOT = ROOT / "runtime"
        if DRIVE_ROOT.is_dir():
            ARTIFACT_ROOT, HOST = DRIVE_ROOT, "remote Colab runtime (Drive mounted)"
        elif LOCAL_ROOT.is_dir():
            ARTIFACT_ROOT, HOST = LOCAL_ROOT, "local authoring host (runtime/ mirror; Drive not mounted)"
        else:
            raise FileNotFoundError(f"neither {DRIVE_ROOT} nor {LOCAL_ROOT} exists; runtime/ is git-ignored and this notebook has no synthetic fallback")
        MET = ARTIFACT_ROOT / "metrics"
        CKPT_DIR = ARTIFACT_ROOT / "checkpoints"
        # Figures persist next to the artifacts they are derived from, so a remote run does not
        # lose them when the runtime recycles (ARTIFACT_ROOT is Drive on the execution host).
        FIGDIR = ARTIFACT_ROOT / "figures" / "nb4"
        FIGDIR.mkdir(parents=True, exist_ok=True)
        ARCHS = ["gru", "timepool", "wide"]
        ACOL = {"gru": "#3568A8", "timepool": "#E58B2A", "wide": "#3A8D68"}

        if not all((MET / f"arch-{a}.result.json").exists() for a in ARCHS):
            raise FileNotFoundError(f"arch-*.result.json not found under {MET} (resolved on: {HOST}); there is deliberately no synthetic fallback.")
        try:
            import torch as _torch
            CUDA = _torch.cuda.is_available()
        except Exception:
            CUDA = False

        DATA = {}
        for a in ARCHS:
            _npz = np.load(MET / f"arch-{a}.probes.npz")
            DATA[a] = {"res": json.loads((MET / f"arch-{a}.result.json").read_text()),
                       "npz": {k: _npz[k] for k in _npz.files}}

        def savefig(fig, name):
            # Save under runtime/figures/nb4/ and show inline.
            fig.savefig(FIGDIR / name, dpi=110, bbox_inches="tight")
            plt.show()
            return FIGDIR / name

        print("host          :", HOST, "| node:", platform.node(), "| cuda available:", CUDA)
        print("ARTIFACT_ROOT :", ARTIFACT_ROOT)
        print("metrics dir   :", MET)
        print("checkpoint dir:", CKPT_DIR, "(exists)" if CKPT_DIR.is_dir() else "(does not exist on this host)")
        print("figure dir   :", FIGDIR)
        for a in ARCHS:
            print(f"loaded {a}: epochs_run={DATA[a]['res']['epochs_run']} probe arrays={len(DATA[a]['npz'])}")
        check("N4.1.1", True, f"all three result.json + probes.npz loaded from {MET}")
        """),

        md(r"""
        ## 2. Artifact inventory and data understanding

        Warden, *Speech Commands: A Dataset for Limited-Vocabulary Speech Recognition*, arXiv:1804.03209 (2018) — the corpus
        every number in this notebook derives from, and the source of the 12-label task definition (10 command words plus
        `unknown` and `silence`). It is the dataset citation, not one of the study's theory-bearing selections.

        ### 2.1 Provenance root: every file under `<ARTIFACT_ROOT>/metrics`, its size and sha256

        The table below lists every file under the resolved `metrics/` folder with its byte size and full sha256. It is the
        provenance root of this notebook: anything plotted later is a function of these bytes. If a reader re-copies the
        artifacts and any hash differs, the figures no longer correspond to what is documented here.
        """),

        code(r"""
        # N4.2.1: inventory of every file under MET with size and sha256.
        rows = []
        for p in sorted(MET.rglob("*")):
            if p.is_file():
                h = hashlib.sha256(p.read_bytes()).hexdigest()
                rows.append({"file": str(p.relative_to(MET)), "bytes": p.stat().st_size, "sha256": h})
        INVENTORY = pd.DataFrame(rows)
        pd.set_option("display.max_colwidth", 80, "display.width", 200)
        display(INVENTORY)
        print(f"{len(INVENTORY)} files, {INVENTORY['bytes'].sum()/1e6:.2f} MB total")
        check("N4.2.1", len(INVENTORY) >= 9, f"{len(INVENTORY)} files inventoried")
        for _, r in INVENTORY.iterrows():
            check(f"N4.2.1[{r['file']}]", r["bytes"] > 0 and len(r["sha256"]) == 64, f"{r['bytes']} bytes, sha256 {r['sha256'][:12]}...")
        """),

        md(r"""
        ### 2.2 Array shapes inside each `.npz`

        The probe arrays are the raw material for Sections 5 and 6. The expected shapes follow from the run
        configuration: $E$ epochs actually run ($E=28$ for `gru`, $E=36$ for `timepool` and `wide`), $120$ probe samples
        (12 classes $\times$ 10 per class), and the sequence tensor shape recorded as `mmphate_shape` in each result file.
        The check cells compare every array's shape with the value implied by the result JSON, so a mismatch between two
        artifacts of the same run is caught here rather than surfacing as a confusing figure later.
        """),

        code(r"""
        # N4.2.2: print array shapes for every npz and compare with the shape implied by the result JSON.
        for a in ARCHS:
            res, npz = DATA[a]["res"], DATA[a]["npz"]
            E = res["epochs_run"]
            steps = len(npz["grad_norm"])
            expected = {
                "coords_2d": (E, 120, 2), "labels": (120,),
                "inter_2d": (E,), "intra_2d": (E,), "inter_raw": (E,), "intra_raw": (E,),
                "grad_norm": (steps,), "clip_threshold": (steps,),
                "mmphate_tensor": tuple(res["mmphate_shape"]),
            }
            print(f"--- arch-{a}.probes.npz  (E={E}, optimizer steps={steps})")
            for k in sorted(npz):
                print(f"   {k:16s} shape={str(npz[k].shape):18s} dtype={npz[k].dtype}")
                check(f"N4.2.2[{a}.{k}]", npz[k].shape == expected[k], f"shape {npz[k].shape} vs expected {expected[k]}")
        if (MET / "mmphate_stats.npz").exists():
            st = np.load(MET / "mmphate_stats.npz")
            print("--- mmphate_stats.npz")
            for k in sorted(st.files):
                print(f"   {k:16s} shape={str(st[k].shape)}")
            check("N4.2.2[mmphate_stats keys]", len(st.files) == 9, f"{len(st.files)} arrays: intra/inter/flow x 3 archs")
        else:
            note("N4.2.2[mmphate_stats]", f"mmphate_stats.npz not present under {MET}; Section 6 recomputes the statistics from the tensors instead")
        """),

        code(r"""
        # N4.2.3: cross-artifact consistency between result.json history and probes.npz.
        for a in ARCHS:
            res, npz = DATA[a]["res"], DATA[a]["npz"]
            h = res["history"]
            check(f"N4.2.3[{a}.history_len]", len(h) == res["epochs_run"], f"len(history)={len(h)} vs epochs_run={res['epochs_run']}")
            check(f"N4.2.3[{a}.probe_epochs]", npz["inter_2d"].shape[0] == len(h), f"probe epochs {npz['inter_2d'].shape[0]} vs history {len(h)}")
            va = np.array([r["val_accuracy"] for r in h])
            check(f"N4.2.3[{a}.best_acc]", abs(va.max() - res["best_val_accuracy"]) < 1e-6, f"max history val_acc {va.max():.4f} vs best_val_accuracy {res['best_val_accuracy']:.4f}")
            vl_ = np.array([r["val_loss"] for r in h])
            check(f"N4.2.3[{a}.best_epoch_is_min_val_loss]", int(vl_.argmin()) == res["best_epoch"], f"argmin val_loss epoch {int(vl_.argmin())} vs best_epoch {res['best_epoch']} (early stopping selects on val LOSS, src/train.py)")
            note(f"N4.2.3[{a}.best_acc_vs_best_epoch]", f"best_val_accuracy {res['best_val_accuracy']:.4f} is the max over epochs (epoch idx {int(va.argmax())}); accuracy at the restored best_epoch {res['best_epoch']} is {va[res['best_epoch']]:.4f}")
            check(f"N4.2.3[{a}.final_acc]", abs(va[-1] - res["final_val_accuracy"]) < 1e-6, f"last history val_acc {va[-1]:.4f} vs final_val_accuracy {res['final_val_accuracy']:.4f}")
            check(f"N4.2.3[{a}.final_loss]", abs(h[-1]["train_loss"] - res["final_train_loss"]) < 1e-6 and abs(h[-1]["val_loss"] - res["final_val_loss"]) < 1e-6, "final train/val loss match last history row")
            check(f"N4.2.3[{a}.params]", res["parameters"] == res["resources"]["parameters"], f"parameters {res['parameters']} == resources.parameters")
            check(f"N4.2.3[{a}.labels]", sorted(np.unique(npz["labels"]).tolist()) == list(range(12)) and np.bincount(npz["labels"]).tolist() == [10] * 12, "120 probe samples = 12 classes x 10")
        """),

        md(r"""
        ### 2.3 The probe set and the summary block

        The probe set is a fixed batch of 120 samples, $12$ classes $\times$ $10$ each, pushed through the model after
        every epoch. It is small on purpose: the MM-PHATE kernel is $O((nsm)^2)$ in memory and $O((nsm)^3)$ in time, so
        the number of samples $n$ directly controls what is affordable. It also means every cluster metric in Section 5
        is estimated from ten points per class, which is a real source of noise and is revisited in the limitations.

        The `summary` block of each result file holds the post-training evaluation. Its header fields are printed next.
        Note the field `seed` inside `summary`: the notebook prints it rather than assuming it equals the training seed.
        """),

        code(r"""
        # N4.2.4: header fields of the summary block for each arch.
        hdr = []
        for a in ARCHS:
            s = DATA[a]["res"]["summary"]
            hdr.append({"arch": a, "summary.seed": s["seed"], "n_cal": s["n_cal"], "n_test": s["n_test"], "n_ood": s["n_ood"],
                        "temperature": round(s["temperature"], 4), "conditions": ", ".join(s["conditions"])})
        display(pd.DataFrame(hdr))
        for a in ARCHS:
            s = DATA[a]["res"]["summary"]
            check(f"N4.2.4[{a}.conditions]", len(s["conditions"]) == 4, f"{len(s['conditions'])} stress conditions (exploration config)")
            check(f"N4.2.4[{a}.temperature]", s["temperature"] > 0, f"fitted temperature T={s['temperature']:.4f}")
        note("N4.2.4", "summary.seed prints as %s, not 17: the summary seed is most likely the bootstrap/evaluation seed, not the training seed; not verifiable from these files" % sorted({DATA[a]['res']['summary']['seed'] for a in ARCHS}))
        """),

        md(r"""
        ## 3. Architecture comparison

        ### 3.1 The real table

        One row per instrumented architecture, straight from the result JSON files. `device` is where the run executed
        (the training itself ran on a GPU on the Colab side; this notebook never touches a GPU). `latency_cpu_median_ms`
        is the recorded CPU median forward latency. The `best_val_accuracy` is the maximum of the per-epoch
        validation accuracy; because early stopping selects and restores the epoch with the lowest validation *loss*
        (`src/train.py`), the accuracy of the saved model is `val_acc_at_best_epoch`, a separate column. The `baseline` row is **not** in this table: the legacy control
        predates the instrumentation and has no `arch-baseline.result.json`; it is handled from its legacy logs in
        Section 3.3, with its provenance stated there.
        """),

        code(r"""
        # N4.3.1: architecture comparison table from the three result files.
        rows = []
        for a in ARCHS:
            r = DATA[a]["res"]; rs = r["resources"]
            rows.append({"arch": a, "parameters": r["parameters"], "epochs_run": r["epochs_run"], "best_epoch": r["best_epoch"],
                         "best_val_accuracy": round(r["best_val_accuracy"], 4),
                         "val_acc_at_best_epoch": round(r["history"][r["best_epoch"]]["val_accuracy"], 4), "final_val_accuracy": round(r["final_val_accuracy"], 4),
                         "final_train_loss": round(r["final_train_loss"], 4), "final_val_loss": round(r["final_val_loss"], 4),
                         "device": rs["device"], "peak_memory_mb": round(rs["peak_memory_mb"], 1),
                         "latency_cpu_median_ms": round(rs["latency_cpu_median_ms"], 3)})
        ARCH_TABLE = pd.DataFrame(rows).set_index("arch")
        display(ARCH_TABLE)
        check("N4.3.1", len(ARCH_TABLE) == 3, "3 instrumented architectures tabulated")
        check("N4.3.1[order]", ARCH_TABLE.loc["gru", "best_val_accuracy"] > ARCH_TABLE.loc["wide", "best_val_accuracy"] > ARCH_TABLE.loc["timepool", "best_val_accuracy"],
              "best val accuracy ordering gru > wide > timepool")
        """),

        md(r"""
        ### Figure NB4-1: Resource and accuracy comparison
        """),

        code(r"""
        # N4.3.2: resources vs accuracy figure.
        fig, ax = plt.subplots(1, 4, figsize=(16, 3.8))
        for a in ARCHS:
            r = DATA[a]["res"]; rs = r["resources"]
            ax[0].bar(a, r["best_val_accuracy"], color=ACOL[a], alpha=0.9)
            ax[0].bar(a, r["final_val_accuracy"], color="none", edgecolor="black", hatch="//")
            ax[1].bar(a, rs["parameters"], color=ACOL[a])
            ax[2].bar(a, rs["model_size_bytes"] / 1e6, color=ACOL[a])
            ax[3].bar(a, rs["latency_cpu_median_ms"], color=ACOL[a])
        ax[0].set(title="best (solid) / final (hatched) val accuracy", ylim=(0, 1))
        ax[1].set(title="parameters (log)", yscale="log")
        ax[2].set(title="model size (MB)")
        ax[3].set(title="CPU median latency (ms)")
        fig.suptitle("NB4-1  Accuracy versus cost for the three instrumented architectures (exploration run, seed 17, clean)", y=1.04)
        fig.tight_layout()
        savefig(fig, "NB4-1_resources.png")
        print("[N4.3.2] resource figure drawn")
        """),

        md(r"""
        ### How to read this chart

        **Data source:** `<ARTIFACT_ROOT>/metrics/arch-{gru,timepool,wide}.result.json`, fields `best_val_accuracy`,
        `final_val_accuracy`, `parameters`, `resources/model_size_bytes` and `resources/latency_cpu_median_ms`.

        **Axes:** each of the four panels has the architecture on the horizontal axis. Panel one shows validation
        accuracy on $[0,1]$ (solid bar: best epoch; hatched outline: final epoch, so a large gap between them means the
        run ended far from its best point). Panels two to four show parameter count (log scale), serialized model size
        in MB, and the recorded CPU median forward latency in ms.

        **Interpretation:** `gru` and `wide` are roughly $16$--$17\times$ larger than `timepool` in parameters, but
        `gru` is the only one with a high and stable accuracy; `wide` reaches a high best accuracy yet ends visibly lower.
        Cost alone does not explain accuracy: `wide` and `gru` have nearly the same parameter count and very different
        stability, which is the question Section 5 examines.

        **Falsifier:** if accuracy simply tracked parameter count, `wide` and `gru` would tie and both would clearly beat
        `timepool`; the recorded gap between `wide`'s best and final accuracy, and `gru`'s smaller gap, contradict a
        pure-capacity reading. Any bar with height zero would indicate a missing field rather than a real result.
        """),

        md(r"""
        ### 3.2 Recorded timing from the worker logs

        Each arm was trained by a separate worker process that wrote its own timestamped log. The cell parses the
        `run_arm done in ...` and `probe overhead ...` lines. It exists to keep the cost discussion honest: the
        in-training probes are not free, and the table shows how large the overhead was relative to training.
        """),

        code(r"""
        # N4.3.3: parse per-arm worker logs for timing lines.
        trows = []
        for a in ARCHS:
            txt = (MET / f"arch-{a}.log").read_text()
            m_run = re.search(r"run_arm done in ([\d.]+)s \(probe overhead ([\d.]+)s total\)", txt)
            m_sum = re.search(r"summarise_arm done in ([\d.]+)s", txt)
            m_best = re.search(r"DONE best_val_acc=([\d.]+)", txt)
            trows.append({"arch": a, "run_arm_s": float(m_run.group(1)) if m_run else np.nan,
                          "probe_overhead_s": float(m_run.group(2)) if m_run else np.nan,
                          "summarise_arm_s": float(m_sum.group(1)) if m_sum else np.nan,
                          "log_best_val_acc": float(m_best.group(1)) if m_best else np.nan})
        TIMING = pd.DataFrame(trows).set_index("arch")
        TIMING["probe_overhead_frac"] = (TIMING["probe_overhead_s"] / TIMING["run_arm_s"]).round(3)
        display(TIMING)
        for a in ARCHS:
            check(f"N4.3.3[{a}.log_vs_json]", abs(TIMING.loc[a, "log_best_val_acc"] - DATA[a]["res"]["best_val_accuracy"]) < 5e-4,
                  f"log best {TIMING.loc[a, 'log_best_val_acc']:.4f} vs json {DATA[a]['res']['best_val_accuracy']:.4f}")
            check(f"N4.3.3[{a}.timing_parsed]", np.isfinite(TIMING.loc[a, "run_arm_s"]), f"run_arm {TIMING.loc[a, 'run_arm_s']:.1f}s, probe overhead {TIMING.loc[a, 'probe_overhead_s']:.1f}s")
        """),

        md(r"""
        ### 3.3 The legacy `baseline` control, from its logs

        The `baseline` (`LogMelCNN`) has no result JSON and no probe file, because its `AdaptiveAvgPool2d((1,1))` leaves
        no temporal axis for the sequence probes. The only record of it is the set of superseded legacy training logs in
        `<ARTIFACT_ROOT>/metrics/legacy_6arm_logs/`. They come from an earlier pipeline (before the true-CE loss accounting fix),
        so **their loss values are not comparable to Section 5's**, and accuracy comparison is at best indicative. The
        cell parses every `epoch n/36 done: ... val_accuracy=...` line and reports per-run best and final validation
        accuracy. No row is invented: the table is exactly what the logs contain.
        """),

        code(r"""
        # N4.3.4: parse legacy logs (superseded runs) into a table of best/final val accuracy.
        pat = re.compile(r"epoch (\d+)/(\d+) done: train_loss=([\d.]+) val_loss=([\d.]+) val_accuracy=([\d.]+)")
        LEG = {}
        lrows = []
        for p in sorted((MET / "legacy_6arm_logs").glob("*.log")):  # empty if the folder is absent on this host
            ms = [pat.search(l) for l in p.read_text().splitlines()]
            ms = [m for m in ms if m]
            ep = np.array([int(m.group(1)) for m in ms]); acc = np.array([float(m.group(5)) for m in ms])
            LEG[p.stem] = {"epoch": ep, "train": np.array([float(m.group(3)) for m in ms]), "val": np.array([float(m.group(4)) for m in ms]), "acc": acc}
            cond, seed = p.stem.split("_seed")
            lrows.append({"condition": cond, "seed": int(seed), "epochs_logged": len(ep), "best_val_acc": acc.max(), "best_epoch": int(ep[acc.argmax()]), "final_val_acc": acc[-1]})
        LEGACY = pd.DataFrame(lrows)
        display(LEGACY.round(4))
        check("N4.3.4", len(LEGACY) == 9, f"{len(LEGACY)} legacy logs parsed under {MET / 'legacy_6arm_logs'}")
        clean_leg = LEGACY[LEGACY.condition == "clean"]
        print("legacy clean-condition final val acc by seed:", clean_leg.set_index("seed")["final_val_acc"].round(4).to_dict())
        print("legacy clean-condition best  val acc by seed:", clean_leg.set_index("seed")["best_val_acc"].round(4).to_dict())
        hit = [k for k, v in LEG.items() if np.isclose(v["acc"], 0.4742, atol=5e-5).any()]
        for k in hit:
            e = LEG[k]["epoch"][np.isclose(LEG[k]["acc"], 0.4742, atol=5e-5)]
            print(f"value 0.4742 occurs in {k} only at epochs {e.tolist()}; that run's final val_acc is {LEG[k]['acc'][-1]:.4f}")
        note("N4.3.4[0.4742]", "the value 0.4742 is a mid-run epoch value of legacy clean_seed17, NOT its final val accuracy (0.3730); no artifact records 0.4742 as a final baseline accuracy")
        for _, r in LEGACY.iterrows():
            check(f"N4.3.4[{r['condition']}_{r['seed']}]", 1 <= r["epochs_logged"] <= 36, f"{r['epochs_logged']} epochs logged, best {r['best_val_acc']:.4f}, final {r['final_val_acc']:.4f}")
        """),

        md(r"""
        ### Figure NB4-2: Legacy baseline validation accuracy versus the new architectures
        """),

        code(r"""
        # N4.3.5: legacy clean-condition curves vs new architectures' val accuracy per epoch.
        fig, ax = plt.subplots(1, 2, figsize=(13, 4.2), sharey=True)
        for k, v in LEG.items():
            if k.startswith("clean"):
                ax[0].plot(v["epoch"], v["acc"], lw=1.2, label=f"legacy baseline {k}")
        for a in ARCHS:
            h = DATA[a]["res"]["history"]
            ax[1].plot([r["epoch"] + 1 for r in h], [r["val_accuracy"] for r in h], color=ACOL[a], label=a)
        for x in ax:
            x.axhline(1 / 12, color="#667085", ls=":", label="chance 1/12")
            x.set(xlabel="epoch", ylim=(0, 1))
        ax[0].set(ylabel="validation accuracy", title="legacy LogMelCNN, clean condition, 3 seeds (superseded pipeline)")
        ax[1].set(title="instrumented architectures, clean, seed 17")
        ax[0].legend(frameon=False, fontsize=8); ax[1].legend(frameon=False, fontsize=8)
        fig.suptitle("NB4-2  Legacy baseline versus new architectures (different pipelines: indicative only)", y=1.03)
        fig.tight_layout()
        savefig(fig, "NB4-2_legacy_vs_new.png")
        print("[N4.3.5] legacy-vs-new figure drawn")
        """),

        md(r"""
        ### How to read this chart

        **Data source:** left panel from `<ARTIFACT_ROOT>/metrics/legacy_6arm_logs/clean_seed{17,18,19}.log` (parsed by regex);
        right panel from `history` in `arch-*.result.json`.

        **Axes:** horizontal is epoch, vertical is validation accuracy on $[0,1]$; the dotted line is chance for 12
        classes, $1/12 \approx 0.083$. The two panels share the vertical axis so the spread is directly comparable.

        **Interpretation:** the legacy baseline's validation accuracy swings by tens of points from epoch to epoch and
        differs greatly across seeds (final values of about $0.37$, $0.59$, and $0.34$). `timepool` lands inside that same
        noisy band, whereas `gru` climbs to roughly $0.9$ and stays there. The seed-to-seed spread of the legacy control is
        itself a caution against reading any single-seed comparison of `timepool` and `baseline` as decisive.

        **Falsifier:** if `timepool`'s curve sat clearly above every legacy seed at every late epoch, the claim below that
        time retention alone is not sufficient would be contradicted. It does not: at several late epochs the legacy seeds
        are above `timepool`'s final value.
        """),

        md(r"""
        ### 3.4 The honest negative finding: retaining the time axis alone did not help reliably

        `timepool` is `LogMelCNN` with a single change: the final pool keeps $8$ temporal slots instead of one, so the
        classifier can see *when* energy occurred. It was built to test whether the temporal collapse alone is the
        ceiling. Its best validation accuracy is $0.5556$ and its final is $0.4107$.

        An assumption carried into this notebook's planning was that `timepool` performs **worse** than the `baseline` and quoted a
        recorded `baseline` final accuracy of $0.4742$. **The artifacts on disk do not support that exact statement.**
        The only occurrence of $0.4742$ is a mid-run value of legacy `clean_seed17` (epochs 22 and 23); that run finished
        at $0.3730$, and the three legacy clean seeds finish at $0.3730$, $0.5913$, and $0.3433$. So `timepool` (final
        $0.4107$) is not below the baseline in any way these files can demonstrate; it lies inside the baseline's own
        seed-to-seed spread. What *is* supported is the weaker but still real negative result: **retaining the time axis
        by itself produced no reliable improvement** -- `timepool`'s late-epoch validation loss and accuracy are as
        erratic as the legacy control's, and it is unstable in the in-training diagnostics too (Section 5).

        The contrast is the interesting part. `gru`, which adds a recurrence over the same time axis, reaches best
        $0.9127$ and final $0.9008$ with a smooth validation loss. The inductive bias of the recurrence, not merely the
        presence of temporal positions in the classifier input, is what accompanies the large improvement in this run.
        That is a single-seed, single-condition observation; the caveat is developed in Section 8.
        """),

        code(r"""
        # N4.3.6: quantify the negative finding from the files: timepool vs legacy baseline seed band vs gru.
        tp, gr, wd = (DATA[a]["res"] for a in ("timepool", "gru", "wide"))
        band_final = (clean_leg["final_val_acc"].min(), clean_leg["final_val_acc"].max())
        band_best = (clean_leg["best_val_acc"].min(), clean_leg["best_val_acc"].max())
        print(f"legacy baseline clean, final val acc range over 3 seeds: {band_final[0]:.4f} .. {band_final[1]:.4f}")
        print(f"legacy baseline clean, best  val acc range over 3 seeds: {band_best[0]:.4f} .. {band_best[1]:.4f}")
        print(f"timepool best {tp['best_val_accuracy']:.4f}, final {tp['final_val_accuracy']:.4f}")
        print(f"gru      best {gr['best_val_accuracy']:.4f}, final {gr['final_val_accuracy']:.4f}")
        va_tp = np.array([r["val_accuracy"] for r in tp["history"]]); va_gr = np.array([r["val_accuracy"] for r in gr["history"]])
        for a, va in (("timepool", va_tp), ("gru", va_gr)):
            print(f"{a}: std of epoch-to-epoch val-acc change over the last 10 epochs = {np.std(np.diff(va[-10:])):.4f}")
        check("N4.3.6[timepool_in_band]", band_final[0] <= tp["final_val_accuracy"] <= band_final[1], "timepool final val acc lies inside the legacy baseline seed band")
        check("N4.3.6[gru_above_band]", gr["final_val_accuracy"] > band_final[1] + 0.2, "gru final val acc exceeds the whole legacy seed band by > 0.2")
        check("N4.3.6[timepool_gain_claim]", tp["final_val_accuracy"] < 0.5, "timepool final val acc < 0.5: no reliable improvement")
        check("N4.3.6[gru_smoother]", np.std(np.diff(va_gr[-10:])) < np.std(np.diff(va_tp[-10:])), "gru's late val-acc changes are smaller than timepool's")
        """),

        md(r"""
        ### 3.5 Reconstructing the early-stopping decision from the recorded history

        Early stopping in `src/train.py` keeps a running best validation loss, resets a counter when the loss improves,
        increments it otherwise, and stops when the counter reaches the patience ($8$ in this run). The recorded
        `history`, `best_epoch` and `epochs_run` are enough to replay that rule exactly, which gives an independent check
        that the three result files really are the output of one consistent training loop, and clarifies which epoch the
        saved weights come from. Nothing is inferred beyond the rule stated in the source.
        """),

        code(r"""
        # N4.3.7: replay the early-stopping rule (strict improvement, patience 8) on each recorded validation-loss series.
        PATIENCE, MAX_EPOCHS = 8, 36
        ES = []
        for a in ARCHS:
            res = DATA[a]["res"]; vl = [r["val_loss"] for r in res["history"]]
            best, best_ep, counter, stop_after = float("inf"), 0, 0, None
            for ep, v in enumerate(vl):
                if v < best:
                    best, best_ep, counter = v, ep, 0
                else:
                    counter += 1
                if counter >= PATIENCE:
                    stop_after = ep + 1
                    break
            ran = stop_after if stop_after is not None else len(vl)
            ES.append({"arch": a, "replayed_best_epoch": best_ep, "recorded_best_epoch": res["best_epoch"], "replayed_epochs_run": ran,
                       "recorded_epochs_run": res["epochs_run"], "stopped_early": stop_after is not None, "epochs_after_best": ran - 1 - best_ep,
                       "min_val_loss": best})
        ESD = pd.DataFrame(ES).set_index("arch")
        display(ESD.round(4))
        for a in ARCHS:
            check(f"N4.3.7[{a}.best_epoch_replay]", ESD.loc[a, "replayed_best_epoch"] == ESD.loc[a, "recorded_best_epoch"], "replayed best epoch equals recorded best_epoch")
            check(f"N4.3.7[{a}.epochs_run_replay]", ESD.loc[a, "replayed_epochs_run"] == ESD.loc[a, "recorded_epochs_run"], "replayed stopping epoch equals recorded epochs_run")
        check("N4.3.7[gru_stopped_early]", bool(ESD.loc["gru", "stopped_early"]) and ESD.loc["gru", "recorded_epochs_run"] == 28, "gru early-stopped at epoch 28 (best epoch 19 + 8 non-improving epochs)")
        check("N4.3.7[others_hit_cap]", all(ESD.loc[a, "recorded_epochs_run"] == MAX_EPOCHS for a in ("timepool", "wide")), "timepool and wide ran to the 36-epoch cap without triggering patience")
        note("N4.3.7", "timepool and wide were still improving their best validation loss within 2 epochs of the cap (best_epoch 34): neither had converged when training ended")
        """),

        md(r"""
        **What the replay shows.** The rule reproduces the recorded stopping point for all three architectures, so the
        history and the summary fields are mutually consistent. `gru` stopped at epoch 28 because $8$ consecutive epochs
        after its best validation loss (epoch index 19) failed to improve on it. `timepool` and `wide` reached the
        36-epoch cap with their best validation loss at epoch index $34$, i.e. still moving; a longer run could have
        changed their numbers, and this notebook cannot say by how much. That is a limitation of the *race*, not of the
        analysis, and it applies most to `wide`, whose best-loss epoch is also its best-accuracy epoch.
        """),

        md(r"""
        ### 3.6 Legacy loss curves and their volatility

        The legacy logs contain validation loss too. Plotting them beside the instrumented runs puts the volatility of the
        legacy control in the same frame as `timepool` and `wide`. As before, the legacy loss values come from the earlier
        accounting pipeline and are not numerically comparable to Section 5; only the *shape* (steady versus erratic) is
        compared.
        """),

        code(r"""
        # N4.3.8: legacy validation-loss curves per training condition, and a volatility table.
        fig, axes = plt.subplots(1, 3, figsize=(16, 4.2), sharey=True)
        vol_rows = []
        for ax, cond in zip(axes, ("clean", "masking", "acoustic")):
            for k, v in LEG.items():
                if k.startswith(cond):
                    ax.plot(v["epoch"], v["val"], lw=1.2, label=k.replace("_", " "))
                    ax.plot(v["epoch"], v["train"], lw=0.8, ls="--", color=ax.lines[-1].get_color())
                    vol_rows.append({"run": k, "epochs": len(v["epoch"]), "val_loss_min": v["val"].min(), "val_loss_last": v["val"][-1],
                                     "val_loss_std_diff_last10": np.std(np.diff(v["val"][-10:])) if len(v["val"]) >= 11 else np.nan,
                                     "val_acc_std_diff_last10": np.std(np.diff(v["acc"][-10:])) if len(v["acc"]) >= 11 else np.nan})
            ax.set(xlabel="epoch", title=f"legacy, {cond} training"); ax.legend(frameon=False, fontsize=7)
        axes[0].set_ylabel("cross-entropy (legacy accounting; solid = val, dashed = train)")
        fig.suptitle("NB4-2b  Legacy control: validation (solid) and training (dashed) loss, three training conditions x three seeds", y=1.03)
        fig.tight_layout()
        savefig(fig, "NB4-2b_legacy_loss.png")
        LV = pd.DataFrame(vol_rows).set_index("run")
        display(LV.round(4))
        check("N4.3.8[all_runs]", len(LV) == 9, f"{len(LV)} legacy runs plotted")
        clean_vol = LV.loc[[i for i in LV.index if i.startswith("clean")], "val_acc_std_diff_last10"].mean()
        print(f"legacy clean: mean late val-acc volatility {clean_vol:.4f} vs timepool {np.std(np.diff(va_tp[-10:])):.4f}, gru {np.std(np.diff(va_gr[-10:])):.4f}")
        check("N4.3.8[gru_calmer_than_legacy]", np.std(np.diff(va_gr[-10:])) < clean_vol, "gru's late validation accuracy is steadier than the legacy clean baseline's average")
        """),

        md(r"""
        ### How to read this chart

        **Data source:** the legacy logs `<ARTIFACT_ROOT>/metrics/legacy_6arm_logs/{clean,masking,acoustic}_seed{17,18,19}.log`,
        parsed by regular expression for `epoch n/36 done: train_loss=... val_loss=... val_accuracy=...`.

        **Axes:** horizontal is epoch; vertical is cross-entropy under the *legacy* loss accounting, shared across the
        three panels. Solid lines are validation loss, dashed lines training loss, one colour per seed.

        **Interpretation:** in every legacy condition the training loss declines steadily while the validation loss stays
        far above it and fluctuates from epoch to epoch (the legacy control generalises poorly and inconsistently). The
        `acoustic` runs stop after only $6$--$7$ epochs (early stopping), at a validation accuracy near $0.27$, so
        augmenting acoustically hurt this small legacy model rather than helping. The `clean` runs are the relevant
        control for this notebook, and their late-epoch instability is of the same order as `timepool`'s.

        **Falsifier:** if the legacy validation loss tracked its training loss closely, the claim that `timepool`'s
        erratic behaviour is not a new pathology (it resembles the control) would be undercut. The wide gap and the
        oscillations are visible directly.
        """),

        md(r"""
        ### 3.7 How tightly validation loss and validation accuracy move together

        Loss and accuracy are different functionals of the same logits. When a model becomes overconfident, validation loss
        can rise while accuracy stays flat or even improves; when both move together the model is simply getting better or
        worse. The rank correlation across epochs, computed per architecture, says which regime each is in.
        """),

        code(r"""
        # N4.3.9: per-epoch val_loss vs val_accuracy scatter and Spearman rho, with train/val gap.
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
        LA = []
        for ax, a in zip(axes, ARCHS):
            h = DATA[a]["res"]["history"]
            vl = np.array([r["val_loss"] for r in h]); va = np.array([r["val_accuracy"] for r in h]); tl = np.array([r["train_loss"] for r in h])
            sc = ax.scatter(vl, va, c=np.arange(1, len(vl) + 1), cmap="viridis", s=26)
            ax.set(xlabel="validation loss", ylabel="validation accuracy", title=a); fig.colorbar(sc, ax=ax, label="epoch")
            rho = spearmanr(vl, va)[0]
            LA.append({"arch": a, "spearman(val_loss, val_acc)": rho, "mean train/val gap": float(np.mean(vl - tl)), "gap_last": float(vl[-1] - tl[-1]),
                       "epochs_with_val_loss_up_and_acc_up": int(np.sum((np.diff(vl) > 0) & (np.diff(va) > 0)))})
        fig.suptitle("NB4-2c  Validation loss vs validation accuracy per epoch (colour = epoch)", y=1.03)
        fig.tight_layout()
        savefig(fig, "NB4-2c_loss_vs_acc.png")
        LAD = pd.DataFrame(LA).set_index("arch")
        display(LAD.round(3))
        for a in ARCHS:
            check(f"N4.3.9[{a}.rho_negative]", LAD.loc[a, "spearman(val_loss, val_acc)"] < 0, f"rho={LAD.loc[a, 'spearman(val_loss, val_acc)']:+.3f}: lower loss goes with higher accuracy")
        """),

        md(r"""
        ### How to read this chart

        **Data source:** `history` in `<ARTIFACT_ROOT>/metrics/arch-*.result.json` (`val_loss`, `val_accuracy`, per epoch).

        **Axes:** each point is one epoch; horizontal is validation loss, vertical validation accuracy, colour the epoch.

        **Interpretation:** a clean downward-sloping cloud means loss and accuracy agree. `gru`'s points form a tight,
        monotone curve running from high loss and low accuracy (early, dark) to low loss and high accuracy (late, bright).
        `wide`'s cloud is broad and its later epochs scatter across a large range of loss at broadly similar accuracy,
        the signature of confidence swings rather than of the predicted classes changing. `timepool` sits in a narrow band
        at high loss. The table counts epochs in which loss and accuracy rose together, a direct measure of that decoupling.

        **Falsifier:** if `wide`'s late points lay on the same curve as its early ones, its loss oscillations would just be
        accuracy oscillations; the loose cloud indicates a part of the instability is confidence, not classification.
        """),

        md(r"""
        ### 3.8 Cost efficiency, tabulated

        Accuracy per parameter and per millisecond of CPU latency, computed from the recorded fields only. These are
        descriptive ratios on one seed, not a Pareto claim.
        """),

        code(r"""
        # N4.3.10: accuracy-per-cost ratios derived from the result files.
        CE = []
        for a in ARCHS:
            r = DATA[a]["res"]; rs = r["resources"]; acc = r["history"][r["best_epoch"]]["val_accuracy"]
            CE.append({"arch": a, "acc_at_best_epoch": acc, "parameters": rs["parameters"], "size_MB": rs["model_size_bytes"] / 1e6,
                       "bytes_per_param": rs["model_size_bytes"] / rs["parameters"], "latency_ms": rs["latency_cpu_median_ms"],
                       "acc_per_100k_params": acc / rs["parameters"] * 1e5, "acc_per_ms": acc / rs["latency_cpu_median_ms"],
                       "peak_memory_mb": rs["peak_memory_mb"], "device": rs["device"]})
        CED = pd.DataFrame(CE).set_index("arch")
        display(CED.round(4))
        for a in ARCHS:
            check(f"N4.3.10[{a}.bytes_per_param]", 3.9 < CED.loc[a, "bytes_per_param"] < 4.5, f"{CED.loc[a, 'bytes_per_param']:.3f} bytes/param (float32 = 4 plus buffers)")
        check("N4.3.10[latency_order]", CED.loc["timepool", "latency_ms"] < CED.loc["gru", "latency_ms"] or CED.loc["timepool", "latency_ms"] < CED.loc["wide", "latency_ms"], "timepool is cheaper than at least one of the large models on CPU latency")
        note("N4.3.10", "latency was measured with a small number of repeats on the training host; treat differences of a few tenths of a millisecond as noise")
        """),

        md(r"""
        ## 4. Checkpoint reload and accuracy re-derivation

        The instrumented checkpoints live at `<ARTIFACT_ROOT>/checkpoints/arch-{gru,timepool,wide}_clean_seed17_instrumented.pt`
        (on the execution host, `/content/drive/MyDrive/deep-learning-first-project/checkpoints/`). Verification has two
        layers, of different strength, and they prove different things.

        **Layer 1, architecture reconstruction (Section 4.1).** `torch.load(..., map_location="cpu")`, instantiate the
        matching class from `src/models.py`, `load_state_dict(..., strict=True)`, assert zero missing and zero unexpected
        keys, assert the live parameter count equals the `parameters` field of that arch's result JSON, and cross-check any
        metrics stored inside the checkpoint against the result JSON. This proves the architecture code reconstructs the
        trained weights exactly and that the artifacts are self-consistent. It does **not** by itself show the recorded
        accuracy is right.

        **Layer 2, accuracy re-derivation (Section 4.2).** Rebuild the real Speech Commands bundle with the project's own
        `assemble_bundle`, recompute the features and the train-split normalisation statistics exactly as
        `src/arms.py::run_arm` does, run each loaded checkpoint over the real validation split, and compare the computed
        accuracy with the recorded `best_val_accuracy` and `final_val_accuracy`. This needs the dataset and (optionally)
        a GPU, so it is tagged `offloaded` and gated on the bundle being present. It never substitutes synthetic audio.
        """),

        code(r"""
        # N4.4.1a: live parameter count of every class in src.models vs the recorded count (needs no checkpoint; runs anywhere).
        import torch
        from src.models import ConvGRU, TimePoolCNN, WideCNN, LogMelCNN

        CLASS_OF = {"gru": ConvGRU, "timepool": TimePoolCNN, "wide": WideCNN}
        for a in ARCHS:
            live = sum(p.numel() for p in CLASS_OF[a]().parameters())
            check(f"N4.4.1a[{a}.live_param_count]", live == DATA[a]["res"]["parameters"], f"src.models.{CLASS_OF[a].__name__} has {live} params vs recorded {DATA[a]['res']['parameters']}")
        base_n = sum(p.numel() for p in LogMelCNN().parameters())
        check("N4.4.1a[baseline_params]", base_n == 14524, f"LogMelCNN parameter count from src.models = {base_n} (charter says 14,524; no artifact exists for it)")
        """),

        md(r"""
        ### 4.1 Checkpoint reload (strict) and cross-check against the result JSON
        """),

        code(r"""
        # N4.4.1: strict checkpoint reload and consistency check. Meant to run on the host that has the checkpoints.
        LOADED = {}
        for a in ARCHS:
            res = DATA[a]["res"]
            path = CKPT_DIR / f"arch-{a}_clean_seed17_instrumented.pt"
            if not path.is_file():
                note(f"N4.4.1[{a}.checkpoint]", f"MISSING on {HOST}: looked for {path}")
                continue
            model = CLASS_OF[a]()
            blob = torch.load(path, map_location="cpu")
            state = blob
            if isinstance(blob, dict):
                for key in ("state_dict", "model_state_dict", "model"):
                    if key in blob and isinstance(blob[key], dict):
                        state = blob[key]
                        break
            out = model.load_state_dict(state, strict=True)
            check(f"N4.4.1[{a}.strict_load]", len(out.missing_keys) == 0 and len(out.unexpected_keys) == 0, f"missing={len(out.missing_keys)} unexpected={len(out.unexpected_keys)}")
            check(f"N4.4.1[{a}.loaded_param_count]", sum(p.numel() for p in model.parameters()) == res["parameters"], "parameter count after load equals result JSON")
            if isinstance(blob, dict):
                extra = [k for k in ("best_val_accuracy", "final_val_accuracy", "epochs_run", "best_epoch", "parameters") if k in blob]
                print(f"{a}: checkpoint top-level keys {sorted(blob)[:8]}{'...' if len(blob) > 8 else ''}; comparable metric keys: {extra}")
                for key in extra:
                    check(f"N4.4.1[{a}.ckpt_{key}]", abs(float(blob[key]) - float(res[key])) < 1e-6, f"checkpoint {key}={blob[key]} vs json {res[key]}")
            LOADED[a] = model.eval()
        print("checkpoints loaded:", sorted(LOADED))
        """, tags=("offloaded",)),

        md(r"""
        ### 4.2 Accuracy re-derivation on the real validation split

        `run_arm` builds features as `log_mel(int16 / 32768)`, takes the global mean and standard deviation of the
        **train** split's features, normalises validation features with those statistics, and early-stops on the clean
        validation split. This cell repeats exactly those steps with the project's own helpers
        (`src.arms._features`, `_feature_stats`, `_normalize`, `_to_float` and `src.train.predict_logits`) so no new
        pipeline is invented, then evaluates each loaded checkpoint on the validation split and prints the delta against
        the recorded accuracies.

        Two caveats, stated now rather than after the fact. First, the checkpoint saved by `run_arm` is the model
        returned by `train_classifier`, which (`src/train.py`, the `load_state_dict(best_model_state)` after the loop)
        is restored to the epoch with the lowest validation **loss**. The accuracy it should reproduce is therefore the
        history's `val_accuracy` at `best_epoch` (column `val_acc_at_best_epoch`), *not* `best_val_accuracy` (the
        maximum over epochs) and not `final_val_accuracy`. The cell reports the delta against all three so the
        comparison does not depend on this reading. Second, the bundle is rebuilt with `assemble_bundle(seed=17)` and its default
        `cap_per_label`; the value used by the original run is not recorded in the files available here, so a non-zero
        delta could reflect a different cap as easily as an evaluation error. A match to within float noise is
        strong evidence; a mismatch is a finding to investigate, not a verdict.
        """),

        md(r"""
        ### 4.2 Re-loading the prepared dataset splits

        The corpus was assembled **once** -- `src.bundle.assemble_bundle(seed=17)` over the real
        `speech_commands_v0.02` archive -- and the resulting five splits were persisted to
        `<ARTIFACT_ROOT>/train-val-split/splits_seed17.npz`. This notebook re-loads that file rather than
        re-downloading 2.43 GB and re-running the split construction on every execution. The splitting logic
        is `assemble_bundle`'s, unchanged; nothing here re-implements it.

        The cell verifies the reloaded sizes against the counts the result JSON recorded (`n_cal`, `n_test`,
        `n_ood`) before anything downstream uses them. That check is what makes the accuracy comparison in
        4.3 meaningful: `cap_per_label` is not recorded in these artifacts, and if it had differed the
        validation split would be a *different set of clips*, making any accuracy delta uninterpretable.
        """),

        code(r"""
        # N4.4.2: re-load the prepared splits from Drive. No download, no re-splitting, no synthetic audio.
        SPLIT_NPZ = ARTIFACT_ROOT / "train-val-split" / "splits_seed17.npz"
        SPLITS, LABELS, SPLITS_MATCH = {}, None, False
        if not SPLIT_NPZ.is_file():
            note("N4.4.2", f"prepared splits not found at {SPLIT_NPZ} (host: {HOST}); section 4.3 cannot run and nothing is substituted")
        else:
            _z = np.load(SPLIT_NPZ, allow_pickle=False)
            for _name in ("train", "val", "cal", "test", "ood"):
                SPLITS[_name] = {"waveforms": _z[f"{_name}__waveforms"], "y": _z[f"{_name}__y"]}
            LABELS = _z["labels"] if "labels" in _z.files else None
            sizes = {k: len(v["y"]) for k, v in SPLITS.items()}
            _sm = DATA[ARCHS[0]]["res"]["summary"]
            _exp = {"cal": _sm["n_cal"], "test": _sm["n_test"], "ood": _sm["n_ood"]}
            for _k, _v in _exp.items():
                check(f"N4.4.2[split_size.{_k}]", sizes.get(_k) == _v, f"reloaded {_k}={sizes.get(_k)} vs recorded {_v}")
            check("N4.4.2[split_size.train_val]", (sizes.get("train"), sizes.get("val")) == (4536, 504),
                  f"reloaded train/val = {sizes.get('train')}/{sizes.get('val')} vs the original run's 4536/504")
            SPLITS_MATCH = all(sizes.get(k) == v for k, v in _exp.items()) and (sizes.get("train"), sizes.get("val")) == (4536, 504)
            _w = SPLITS["val"]["waveforms"]
            print(f"loaded {SPLIT_NPZ.name} ({SPLIT_NPZ.stat().st_size/1024**2:.1f} MB) on {HOST}")
            print("split sizes :", sizes)
            print("val waveform:", _w.shape, _w.dtype, "| labels:", None if LABELS is None else list(LABELS))
            check("N4.4.2[waveform_dtype]", _w.dtype == np.int16, f"waveforms stored as {_w.dtype} (assemble_bundle yields int16 PCM)")
        """, tags=("offloaded",)),

        md(r"""
        ### 4.3 Accuracy re-derivation on the real validation split

        `run_arm` builds features as `log_mel(int16 / 32768)`, takes the global mean and standard deviation of
        the **train** split's features, and normalises validation features with those statistics. This cell
        repeats exactly those steps with the project's own helpers (`src.arms._features`, `_feature_stats`,
        `_normalize`, `_to_float` and `src.train.predict_logits`), then evaluates each loaded checkpoint.

        One caveat stated before the numbers appear: the checkpoint `run_arm` saves is the model
        `train_classifier` returns, which (`src/train.py`, the `load_state_dict(best_model_state)` after the
        loop) is restored to the epoch with the lowest validation **loss**. The accuracy it should reproduce
        is therefore the history's `val_accuracy` at `best_epoch` (column `val_acc_at_best_epoch`), *not*
        `best_val_accuracy` (the maximum over epochs) and not `final_val_accuracy`. The cell reports the delta
        against all three so the comparison does not depend on that reading.
        """),

        code(r"""
        # N4.4.3: re-derive validation accuracy from the reloaded splits. Gated on N4.4.2 having loaded them.
        if not SPLITS:
            note("N4.4.3", "no splits loaded in N4.4.2; accuracy re-derivation not executed (nothing substituted)")
        elif not LOADED:
            note("N4.4.3", "no checkpoint was loaded in N4.4.1; nothing to evaluate")
        else:
            from src.arms import _features, _feature_stats, _normalize, _to_float
            from src.train import predict_logits
            dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            t0 = time.time()
            train_feats = _features(_to_float(SPLITS["train"]["waveforms"]))
            f_mean, f_std = _feature_stats(train_feats)
            val_x = _normalize(_features(_to_float(SPLITS["val"]["waveforms"])), f_mean, f_std)
            val_y = SPLITS["val"]["y"]
            print(f"features: train {train_feats.shape}, val {val_x.shape} | train stats mean={f_mean:.4f} std={f_std:.4f} | {time.time()-t0:.1f}s | device {dev}")
            REDERIVE = []
            for a, model in LOADED.items():
                model = model.to(dev)
                pred = predict_logits(model, val_x).argmax(axis=1)
                acc = float((pred == val_y).mean()); res = DATA[a]["res"]
                at_best = res["history"][res["best_epoch"]]["val_accuracy"]
                REDERIVE.append({"arch": a, "recomputed_val_acc": acc, "recorded_at_best_epoch": at_best,
                                 "delta_vs_at_best_epoch": acc - at_best,
                                 "recorded_final": res["final_val_accuracy"], "delta_vs_final": acc - res["final_val_accuracy"],
                                 "recorded_max": res["best_val_accuracy"], "delta_vs_max": acc - res["best_val_accuracy"], "n_val": len(val_y)})
            RD = pd.DataFrame(REDERIVE).set_index("arch")
            display(RD.round(4))
            if not SPLITS_MATCH:
                note("N4.4.3[comparison]", "reloaded split sizes differ from the recorded run: the validation set is not the same set of clips, so the deltas above are NOT evidence about the checkpoints and no accuracy claim is made")
            for a, r in RD.iterrows():
                check(f"N4.4.3[{a}.rederived]", SPLITS_MATCH and abs(r["delta_vs_at_best_epoch"]) < 5e-3,
                      f"recomputed {r['recomputed_val_acc']:.4f}; delta vs accuracy at restored best_epoch {r['delta_vs_at_best_epoch']:+.4f} (vs final {r['delta_vs_final']:+.4f}, vs max {r['delta_vs_max']:+.4f})")
        """, tags=("offloaded",)),

        md(r"""
        **What the offloaded cells contribute when executed.** Section 4.1 confirms the weights and the code agree and
        that any metadata stored in the checkpoint matches the result JSON. Section 4.2 is the actual check on the
        accuracy numbers used throughout this notebook. If either cell has not been executed on the remote runtime, its
        output above will be absent (the cells are tagged `offloaded`), and the accuracy figures should then be read as
        the *recorded* values from the result JSON, not as independently re-derived ones.
        """),
    ]
