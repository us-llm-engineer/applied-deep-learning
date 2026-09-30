"""NB4 part 4: limitations.

Markdown, plus one cell that re-derives from the artifacts the two claims that can be checked
(the number of stress conditions, and which architectures carry probe instrumentation).
No mock or synthetic data appears anywhere in this notebook.
"""
from nbkit import md, code


def cells():
    return [
        md(r"""
        ## 8. Limitations

        Everything below is a limitation of *this run and this analysis*, not of the pipeline. They are stated
        here rather than distributed through the text so that a reader deciding how much weight to put on any
        number in Sections 3--7 can find the caveats in one place. The first cell re-derives from the artifacts
        the two claims that are checkable; the rest are recorded facts about how the run was configured and
        what was never executed.
        """),

        code(r"""
        # N4.8.2: re-derive the checkable limitation claims from the artifacts themselves.
        conds = sorted({c for a in ARCHS for c in DATA[a]["res"]["summary"]["conditions"]})
        check("N4.8.2[exploration_conditions]", len(conds) == 4,
              f"summary block carries {len(conds)} stress conditions: {conds} (the publication design specifies 11)")
        instrumented = sorted(a for a in ARCHS if "coords_2d" in DATA[a]["npz"])
        check("N4.8.2[instrumented_archs]", instrumented == ["gru", "timepool", "wide"],
              f"probe arrays exist only for {instrumented}; the legacy 'baseline' (LogMelCNN) has none")
        seeds = sorted({DATA[a]["res"]["summary"]["seed"] for a in ARCHS})
        print("distinct summary seeds across the three arms:", seeds, "| conditions:", conds)
        print("NOTE: the bootstrap resample count is not recorded in the result JSON; 200 is the")
        print("      exploration value carried in PROVENANCE, not something re-derivable from these files.")
        """),

        md(r"""
        ### 8.1 Scale and coverage of the evaluation

        - **Exploration configuration, not publication numbers.** The post-training summary block was produced with
          **4 stress conditions and 200 bootstrap resamples**. The design in `plans/round-02/training-pipeline.md`
          specifies 11 conditions and 2000 resamples; that evaluation **was never run**. Every interval in Section 7
          is therefore wider and coarser than the intended design, and no number in this notebook should be quoted
          as a publication result.
        - **One seed, one training condition.** The architecture race is `condition=clean`, `seed=17` only. Earlier
          rounds on the legacy baseline showed validation accuracy swinging 0.343 to 0.591 across seeds at
          essentially constant training cross-entropy, so single-seed differences between architectures here cannot
          be separated from seed variance. The `gru`-versus-`wide` gap is large enough to survive that concern; the
          `timepool`-versus-`baseline` comparison is not.
        - **The `baseline` control is not instrumented.** `LogMelCNN`'s `AdaptiveAvgPool2d((1,1))` collapses both the
          mel and time axes, so there is no penultimate sequence axis to probe. V2--V8 cover the three new
          architectures only, and the baseline appears solely through its recorded logs.
        - **A reproducibility anomaly that is not explained.** Across two races, `wide` and `gru` reproduce
          bit-exact, while `timepool`'s final validation accuracy differs (0.3631 versus 0.4107). The cause was not
          identified. Treat `timepool`'s exact numbers as uncertain at that magnitude.

        ### 8.2 Method fidelity: where this notebook deviates from its sources

        These deviations are labelled in the figure titles as well, so a figure lifted out of this notebook carries
        its own caveat.

        - **PCA is substituted for SentryCam's projection.** SentryCam (arXiv:2405.15135) specifies a *parametric
          autoencoder* (`d -> d/2 -> ... -> 2`, with GroupNorm and BatchNorm) to reach the 2D space in which its two
          cluster metrics are computed. This notebook uses a linear PCA projection. The paper's stated *reason* for
          projecting at all -- that raw high-dimensional space distorts distances -- is what Section 5.5 tests, and
          that argument does not depend on which projector is used; but the specific 2D coordinates and therefore the
          exact alert epochs are not the paper's.
        - **PCA is substituted for MM-PHATE's embedding.** MM-PHATE (arXiv:2406.01969) builds a multiway diffusion
          kernel and embeds via diffusion potential plus MDS. Section 6's V8 is a PCA scatter of the same z-scored
          activation tensor. The *statistics* (intra-step entropy, inter-step entropy, flow alignment) follow the
          paper's definitions; the *embedding geometry* does not, and should not be read as a PHATE plot.
        - **The alert rule is evaluated post-hoc.** SentryCam's two-condition rule is designed for live per-epoch
          logging. Here it is replayed over stored series. That changes nothing about the arithmetic, but the
          "fired 7 epochs early" style of claim in the paper concerns an online setting this notebook does not test.
        - **AutoClip threshold is computed one step early.** The paper appends the current step's gradient norm to the
          history $G_h(t)$ and *then* takes the percentile, so its threshold at step $t$ includes step $t$'s own norm.
          This implementation takes the percentile of the prior history and appends afterwards -- which is exactly what
          check `N4.5.3` verifies and passes (`threshold == percentile(grad_norm[:t], 10)`, step 0 infinite). The
          numerical difference is negligible beyond the first handful of steps, but it is an off-by-one against the
          published algorithm and is recorded here rather than left unstated.
        - **`p = 10` is inherited from a different task.** AutoClip's percentile sweep was run on WSJ0-2mix speech
          *separation* -- 4x BLSTM layers, STFT magnitude features, evaluated in SI-SDR -- where gradients genuinely
          explode. This notebook applies it to log-mel classification with BatchNorm convolutions, which do not have
          that pathology. The measured clip rates (`timepool` 99.1%, `wide` 70.2%, `gru` 65.8%) are where that
          difference shows, so the paper's conclusion that "more clipping is generally better" should not be assumed to
          carry over.

        ### 8.3 Known-unsound components carried forward deliberately

        - **Conformal risk control here is *unweighted*.** Under acoustic shift the calibration and test
          distributions are not exchangeable, and unweighted CRC in exactly that setting is the *named failure mode*
          in the grounding: the achieved risk can exceed the target. The miscoverage numbers in Section 7.5 should
          be read with that in mind -- where achieved miscoverage exceeds the target alpha, that is the predicted
          behaviour, not a bug in the implementation. The weighted estimator was designed but never implemented.
        - **The abstention loss CRC requires must be monotone**, and the natural abstention objective is not. This
          is flagged in the grounding as an open limitation and is not addressed here.
        - **The calibration split is n = 1080.** The source wants n >= 1000 for a tight O(1/n) margin. We clear it,
          barely -- so the conformal thresholds are at the edge of the regime where the guarantee is tight.

        ### 8.4 Research-integrity notes on the grounding itself

        Recorded because a reader tracing a claim back to its source would otherwise find these the hard way.

        - **Source 1 was never ingested.** In the `speech-shift` research notebook, what was captured for Source 1 is
          a Cloudflare/OpenReview gateway challenge page, not the selective-classification paper it was meant to be.
          Any claim attributed to Source 1 in earlier rounds is therefore **unsupported** and has not been relied on
          in this notebook.
        - **One research answer is empty.** Turn 7 of that same notebook (the "unresolved limitations" question)
          returned no answer; only the re-ask at Turn 10 produced content. Round summaries that cite Turn 7 are
          citing nothing.
        - **Two research answers survive only locally.** The live chat history for the visualization notebook
          truncated its first two answers; locally stored copies in a private research workspace are now the only
          not a backing store and should not be treated as one.
        - **Researched but unused.** The `training-diag-signal` NotebookLM notebook spent four of its eight turns on CLUE
          (arXiv:2505.22803), an error-driven uncertainty-aware training objective. It is implemented nowhere in this
          project and appears nowhere else in this notebook. Recorded so the research effort is accounted for rather than
          silently dropped.

        ### 8.5 What the artifacts cannot tell us

        - **`cap_per_label` for the original run is not recorded** in any file available here. Section 4.2 rebuilds
          the bundle with `assemble_bundle`'s default, so a non-zero accuracy delta there could reflect a different
          cap rather than an evaluation error. This is why that cell reports a delta against three different
          recorded accuracies instead of asserting one.
        - **The bootstrap resample count is likewise not in the result JSON** -- 200 is carried in the provenance
          record, not re-derivable from these bytes.
        - **Sections 4.1 and 4.2 are `offloaded`.** If they were not executed on the remote runtime, the accuracy
          figures throughout this notebook are the *recorded* values from the result JSON and have not been
          independently re-derived. The cell output above is what tells you which of the two you are reading.
        """),
    ]
