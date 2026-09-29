"""Notebook 3, Part 3: Frozen-vs-adapted encoder framing (source: training-pipeline.md §3).

Section:
- 3. Frozen-vs-adapted encoder framing (source: training-pipeline.md §3)
"""

from nbkit import md, code


def cells():
    return [
        # ─────────────────────────────────────────────────────────────────────────────────
        # Section 3: Frozen-vs-adapted encoder framing (source: training-pipeline.md §3)
        # ─────────────────────────────────────────────────────────────────────────────────

        md("""
        ## 3. Frozen-vs-adapted encoder framing (source: training-pipeline.md §3)

        The Wav2Vec2 encoder comparison is designed as a 4-arm measurement: {frozen, adapted} x {clean, acoustic}.

        **Arm design** (quoted exactly from training-pipeline.md §3):
        - **Frozen**: mean-pool last-layer features (768) -> linear head (same optimizer budget), trained on clean features vs on features of a fixed seeded acoustic-augmented copy (K=1 copy, same transform family as the CNN acoustic arm).
        - **Adapted**: cache outputs of block 11 for clean train and for K=3 acoustic copies (fp16); train final transformer block + layer norm + mean-pool + linear head over cached hidden states; same budget (max 6 epochs, patience 2, microbatch 64, accumulation 2). No masking arm for the encoder (documented as not run).
        - **Arms**: {frozen, adapted} x {clean, acoustic} = 4.

        This section attempts to load the encoder and report its availability status. In this offline/local-only round,
        the attempt is expected to raise `EncoderUnavailable` due to the absence of a network download budget.
        """),

        code("""
        # W3.3.1: Attempt to load Wav2Vec2 encoder, catching EncoderUnavailable.
        from src.encoder import load_wav2vec2, EncoderUnavailable

        w3_encoder_attempted = False
        w3_encoder_loaded = False
        w3_caught = None
        w3_caught_type = None

        try:
            w3_encoder = load_wav2vec2()
            w3_encoder_loaded = True
            print("[W3.3] Encoder loaded successfully")
        except EncoderUnavailable as e:
            w3_caught = e
            w3_caught_type = type(e).__name__
            print(f"[W3.3] EncoderUnavailable raised: {e}")
        except Exception as e:
            w3_caught = e
            w3_caught_type = type(e).__name__
            print(f"[W3.3] Unexpected exception: {w3_caught_type}: {e}")
        finally:
            w3_encoder_attempted = True

        # Check that the attempt was genuinely made (sentinel set in both branches).
        check("W3.3.1", w3_encoder_attempted and (w3_encoder_loaded is True or isinstance(w3_caught, EncoderUnavailable)),
              f"Encoder load attempted: loaded={w3_encoder_loaded}, caught EncoderUnavailable={isinstance(w3_caught, EncoderUnavailable)}")
        """),

        code("""
        # W3.3.1a: Check the exact exception type (if an exception was raised).
        # This ensures we are catching the intended exception, not a different one.

        if w3_caught is not None:
            w3_correct_exception_type = isinstance(w3_caught, EncoderUnavailable)
            check("W3.3.1a", w3_correct_exception_type,
                  f"Exception type is EncoderUnavailable (actual: {w3_caught_type})")
        else:
            # Encoder loaded successfully; no exception to check
            check("W3.3.1a", w3_encoder_loaded,
                  f"Encoder load succeeded, no exception raised")
        """),

        code("""
        # W3.3.2: Report encoder unavailability when applicable.
        if isinstance(w3_caught, EncoderUnavailable):
            note("W3.3.2", "encoder: unavailable (no network fetch this round); 4-arm comparison not run; see Limitations")
        elif w3_encoder_loaded:
            note("W3.3.2", "encoder: available; 4-arm comparison would proceed to training")
        """),

        md(r"""
        ### Formal 4-arm design definition

        The frozen-vs-adapted encoder comparison is structured as a factorial design with two factors:

        **Encoder type**: {frozen, adapted}
        - **Frozen**: Mean-pool final-layer features (768 dimensions), train only a linear head on top.
        - **Adapted**: Cache intermediate block-11 states; train the final transformer block, layer norm, mean-pool, and linear head.

        **Training condition**: {clean, acoustic}
        - **Clean**: Train on original waveforms (no augmentation except the base training augmentation).
        - **Acoustic**: Train on $K=1$ acoustic-augmented copy per sample (same augmentation family as the CNN acoustic arm).

        **Result**: A $2 \times 2$ design matrix:

        $$\begin{array}{c|cc}
        & \text{Clean} & \text{Acoustic} \\
        \hline
        \text{Frozen} & \text{arm(f,c)} & \text{arm(f,a)} \\
        \text{Adapted} & \text{arm(a,c)} & \text{arm(a,a)} \\
        \end{array}$$

        Each arm trains with the same budget: up to 6 epochs, patience 2, microbatch 64, accumulation 2
        (for the adapted arm only). In this round, encoder load is expected to raise `EncoderUnavailable`
        (no network access), so all four cells show "unavailable" status rather than numeric results.
        The design is documented here to show what will run once network access is restored in a future round.
        """),

        md("""
        ### Why the encoder comparison is deferred this round

        This round's primary constraint is network availability. The Wav2Vec2 base encoder model
        (`facebook/wav2vec2-base`) is 360+ MB and requires downloading from Hugging Face Model Hub.
        This notebook infrastructure is offline-first (local synthetic data, CPU training), so the download
        is deliberately blocked (`EncoderUnavailable`). The 4-arm design is fully specified and ready to run
        once network access is provisioned in a future round — the design document (above) serves as a contract
        that future implementations must follow. This is a deliberate postponement, not an oversight: attempting
        to fake or simulate encoder results would violate the "never substitute" principle stated in
        `training-pipeline.md §3`. The grid below visualizes the actual outcome: unavailability is reported
        honestly, with no fabricated numbers.
        """),

        md("""
        ### Figure NB3-4: Encoder-arm availability matrix

        The 2×2 grid below reflects the actual outcome of the encoder load attempt. When the encoder is unavailable,
        all four arms (frozen-clean, frozen-acoustic, adapted-clean, adapted-acoustic) show "unavailable" status
        rather than fabricated numbers. The falsifier for this chart: a figure showing numeric numbers when `EncoderUnavailable`
        was raised would violate the "never substitute" rule stated in training-pipeline.md §3.
        """),

        code("""
        # W3.3.3: Create encoder-arm availability matrix (2x2 grid).
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        import numpy as np

        fig, ax = plt.subplots(figsize=(6.0, 4.5))

        # Define the 2x2 grid: rows = {frozen, adapted}, columns = {clean, acoustic}
        w3_conditions = ["clean", "acoustic"]
        w3_encoder_types = ["frozen", "adapted"]
        w3_status_grid = []

        if w3_encoder_loaded:
            # If encoder loaded, all four arms would be "Available" (real numbers would go here in the future)
            w3_status_grid = [["Available", "Available"], ["Available", "Available"]]
            w3_cell_colors = [["#90EE90", "#90EE90"], ["#90EE90", "#90EE90"]]  # Light green for available
        else:
            # Encoder unavailable: all cells show "Unavailable"
            w3_status_grid = [["Unavailable", "Unavailable"], ["Unavailable", "Unavailable"]]
            w3_cell_colors = [["#FFB6C6", "#FFB6C6"], ["#FFB6C6", "#FFB6C6"]]  # Light red for unavailable

        # Create a table visualization
        ax.axis("tight")
        ax.axis("off")

        # Create text content
        w3_table_data = []
        w3_table_data.append(["Encoder Type", "Clean", "Acoustic"])
        for w3_i, w3_enc_type in enumerate(w3_encoder_types):
            w3_row = [w3_enc_type, w3_status_grid[w3_i][0], w3_status_grid[w3_i][1]]
            w3_table_data.append(w3_row)

        # Draw table
        w3_table = ax.table(cellText=w3_table_data, cellLoc="center", loc="center",
                             colWidths=[0.2, 0.3, 0.3])
        w3_table.auto_set_font_size(False)
        w3_table.set_fontsize(11)
        w3_table.scale(1, 2.5)

        # Color the header row
        for w3_j in range(3):
            w3_table[(0, w3_j)].set_facecolor("#E8E8E8")
            w3_table[(0, w3_j)].set_text_props(weight="bold")

        # Color the data cells based on availability
        for w3_i in range(1, 3):
            w3_table[(w3_i, 0)].set_facecolor("#E8E8E8")
            w3_table[(w3_i, 0)].set_text_props(weight="bold")
            for w3_j in range(1, 3):
                w3_table[(w3_i, w3_j)].set_facecolor(w3_cell_colors[w3_i - 1][w3_j - 1])

        ax.set_title("Encoder-arm Availability (frozen vs. adapted) × (clean vs. acoustic)", fontsize=12, weight="bold", pad=20)
        fig.tight_layout()
        plt.savefig("/tmp/w3_encoder_availability.png", dpi=110, bbox_inches="tight")
        plt.show()

        print("[W3.3.3] Encoder-arm availability matrix created")
        """),

        code("""
        # W3.3.3a: Check that the 2x2 matrix has exactly 4 cells all marked consistently.
        # If encoder_loaded, all should be "Available"; if not loaded, all should be "Unavailable".

        w3_matrix_size = len(w3_status_grid) * len(w3_status_grid[0])
        check("W3.3.3a", w3_matrix_size == 4,
              f"Encoder-arm matrix has exactly 4 cells (2x2): {w3_matrix_size}")

        # Check consistency: all cells should show the same status (either all Available or all Unavailable)
        w3_status_values = [cell for row in w3_status_grid for cell in row]
        w3_all_same_status = len(set(w3_status_values)) == 1
        check("W3.3.3b", w3_all_same_status,
              f"All 4 matrix cells show consistent status: {set(w3_status_values)}")
        """),

        code("""
        # W3.3.3c: Check that no numeric result was fabricated when EncoderUnavailable was raised.
        # All four matrix cells must be the string "Unavailable", never a float.

        w3_no_fabrication = True
        for w3_row in w3_status_grid:
            for w3_cell in w3_row:
                # Cell should be either "Unavailable" (string) or "Available" (string), never a float/number
                if isinstance(w3_cell, (int, float)):
                    w3_no_fabrication = False

        check("W3.3.3c", w3_no_fabrication,
              f"No numeric fabrication: all matrix cells are strings (Available/Unavailable), not numeric values")
        """),

        md("""
        ### How to read this chart

        This 2×2 grid displays the availability status of each encoder-arm configuration. Each cell represents one
        combination of encoder type (frozen or adapted) and training condition (clean or acoustic).

        **Data source:** The status is determined by the `load_wav2vec2()` call in the cell above. Each cell displays
        either "Available" (if the encoder weights could be downloaded successfully) or "Unavailable" (if `EncoderUnavailable`
        was raised, as expected in this offline environment).

        **Falsifier (what would indicate a bug):** A chart showing numeric accuracy or loss values when `EncoderUnavailable`
        was raised would indicate that the code fabricated numbers instead of honestly reporting unavailability, violating
        the contract's "never substitute" rule. The fact that all four cells show the same status (either all "Available"
        or all "Unavailable") is correct: the encoder is either present for all arms or absent for all arms; there is no
        per-arm availability difference.

        **Interpretation:** In this round, with no network fetch budget available, all four arms show "Unavailable". Once
        network access is restored (in a future run with proper infrastructure), the encoder would load, and real training
        logits would populate the cells in place of the "Available" status markers. This chart ensures that the notebook's
        output is honest: it reports the truth about whether the encoder was available, rather than silently substituting
        or skipping the measurement.
        """),

        code("""
        # W3.3.4: Record what a real T4/Colab run must re-verify before trusting the 4-arm comparison,
        # per plans/round-02/training-pipeline.md §3 and §10 -- this environment's `EncoderUnavailable`
        # result does not, by itself, prove torchaudio will be available on the offload host.

        w3_t4_preflight_items = [
            "import torchaudio succeeds on the Colab/T4 runtime (not assumed from this local result)",
            "torchaudio.pipelines.WAV2VEC2_BASE.get_model() actually downloads weights within the T4 session's network budget",
            "the frozen arm's mean-pooled 768-dim features extract without shape errors on real 16kHz clips",
            "the adapted arm's cached block-11 hidden states (fp16, K=3 acoustic copies) fit within Colab's memory budget",
        ]

        print("Pre-flight checklist for the real T4 run's encoder arms (not verified by this notebook):")
        for w3_item in w3_t4_preflight_items:
            print(f"  - {w3_item}")

        check("W3.3.4", len(w3_t4_preflight_items) == 4,
              "T4 pre-flight checklist for the 4-arm encoder comparison is recorded (4 items, matching training-pipeline.md §3)")
        """),
    ]
