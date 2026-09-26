"""Notebook part 4: acoustic shift operators, a noise-injected mixup toy, and synthesis.

Sections 8, 9, 10: measured shifts (gain, additive noise, synthetic-RIR reverb), a NoisyMix-style toy
analogue, and the recap of what the sources do and do not settle.
"""

from nbkit import md, code


def cells():
    return [
        md(r"""
## 8. Acoustic shift operators with measured severity (derived here)

This section operationalises three canonical acoustic perturbations for a fixed-vocabulary speech-command setting: gain change, additive white noise, and reverberation. Each operator is driven by one requested severity (gain in dB, signal-to-noise ratio in dB, reverberation time $T_{60}$ together with a direct-to-reverberant ratio). The point of the section is not to *use* the operators but to **measure whether the severity actually delivered matches the severity requested**, because every later claim of the form "accuracy at 10 dB SNR" or "calibration at $T_{60}=0.6$ s" silently assumes it does.

**Why this matters for the sources.** The paragraphs below are paper-level statements, each restricted to what the raw traces contain.

- Conformal risk control (Angelopoulos et al., 2208.02814) guarantees expected risk only for exchangeable data. Section 5 of the paper, as quoted in q12, says extensions to non-exchangeable data require knowledge about the form of the shift. A clean-calibrated model deployed on noisy or reverberant audio is exactly that situation (q8, "Failure of Standard Risk Control under Shift").
- q8 lists two transferable hypotheses for acoustic shift: a weighted risk-control threshold if the shift can be modelled as covariate shift with an estimable density ratio, and a total-variation bound for shifts that cannot. Both are labelled hypotheses in q8; neither is tested on audio in this notebook.
- NoisyMix (Erichson et al., AISTATS 2024) states in its conclusion: *"A limitation of NoisyMix is that it is tailored towards computer vision tasks and not directly applicable to natural language processing tasks, or time series tasks."* (q6, q7, q8, q12). Any audio analogue therefore needs audio-native perturbations, which is why we build and validate the operators here.

**What is derived here, not taken from a paper.** The synthetic waveform, the SNR algebra, the clipping emulation, and the verification of the room-impulse-response (RIR) reverb are all our own constructions. Nothing in this section is a claim about real speech.

**Conventions.** Signal power is $P_s=\frac1N\sum_t s_t^2$ (a *power*, not an RMS amplitude). All ratios in dB are $10\log_{10}$ of a power ratio, or equivalently $20\log_{10}$ of an amplitude ratio. Every number that comes from a run is printed by a code cell; the prose only states what to look for.
        """),

        code(r"""
# S8.0 setup: synthetic 1 s speech-like waveform (4 harmonics + fricative burst), fs=16 kHz, n=16000 samples, seed RNG_SEED+8, reps=1
import sys; sys.path.insert(0, str(ROOT))
from src import audio

n_sr = 16000
n_duration = 1.0
n_samples = int(n_sr * n_duration)
n_rng = np.random.default_rng(RNG_SEED + 8)
n_time = np.arange(n_samples, dtype=np.float32) / n_sr

# amplitude envelope: 50 ms ramp-in, 700 ms plateau, exponential release
n_attack, n_sustain = int(0.05 * n_sr), int(0.7 * n_sr)
n_release = n_samples - n_attack - n_sustain
n_env = np.ones(n_samples, dtype=np.float32)
n_env[:n_attack] = np.linspace(0, 1, n_attack)
n_env[n_attack + n_sustain:] = np.exp(-np.linspace(0, 4, n_release))

# harmonic part: 110 Hz fundamental, partials 2,3,5,8 (all below 1 kHz)
n_overtones = np.array([2, 3, 5, 8])
n_harmonic = np.zeros(n_samples, dtype=np.float32)
for n_ratio in n_overtones:
    n_harmonic += np.sin(2 * np.pi * 110.0 * n_ratio * n_time).astype(np.float32) / len(n_overtones)
n_harmonic *= n_env

# broadband burst 0.30-0.40 s (stand-in for a fricative); zero outside that window
n_burst_start, n_burst_len = int(0.3 * n_sr), int(0.1 * n_sr)
n_fric = np.zeros(n_samples, dtype=np.float32)
n_fric[n_burst_start:n_burst_start + n_burst_len] = 0.3 * n_rng.standard_normal(n_burst_len).astype(np.float32)
n_fric *= n_env

n_waveform = (n_harmonic + n_fric).astype(np.float32)
n_waveform = n_waveform / (np.abs(n_waveform).max() + 1e-8)          # peak-normalise to full scale
n_sig_pow = float(np.mean(n_waveform.astype(np.float64) ** 2))       # power, used by every SNR formula below
n_burst_end = n_burst_start + n_burst_len

print(f"[S8.0] waveform {n_waveform.shape}, power P_s={n_sig_pow:.4f}, RMS={np.sqrt(n_sig_pow):.4f}, peak={np.abs(n_waveform).max():.6f}")
check("S8.0", n_waveform.shape == (n_samples,) and np.isfinite(n_waveform).all() and abs(np.abs(n_waveform).max() - 1.0) < 1e-6
      and not n_fric[:n_burst_start].any() and not n_fric[n_burst_end:].any() and n_fric[n_burst_start:n_burst_end].any(),
      f"shape/finite/peak=1, and the broadband burst is confined to samples [{n_burst_start},{n_burst_end})")
        """),

        md(r"""
### Gain operator and an emulated full-scale limiter

A gain of $g$ dB multiplies the waveform by $a=10^{g/20}$, so power scales by $a^2=10^{g/10}$. The operator `src.audio.apply_gain` performs exactly this multiplication in 32-bit float and **does not clip**: the output can exceed $\pm1$. A real microphone chain or a fixed-point file does clip at full scale, so to study that failure we emulate a hard limiter $x_c=\mathrm{clip}(a\,x,-1,1)$ as a separate, explicit step in the notebook.

Because our synthetic waveform is peak-normalised to full scale, *any* positive gain pushes the peak beyond $\pm1$; whether that matters depends on how many samples exceed full scale and how much distortion power $D=\frac1N\sum_t (x_{c,t}-a x_t)^2$ the limiter injects. The signal-to-distortion ratio $\mathrm{SDR}=10\log_{10}(P_{ax}/D)$ measures how badly the clipped waveform departs from the ideal scaled waveform.

We check two things: that the operator delivers the requested scale to float32 precision, and that the number of samples above full scale follows from the waveform's own amplitude distribution ($|x_t|>10^{-g/20}$) and is monotone in $g$.
        """),

        code(r"""
# S8.1 gain + emulated clipping: g in {-6, 0, +3, +12} dB on the peak-normalised waveform; deterministic (no random draws), n=16000
n_gains_db = [-6.0, 0.0, 3.0, 12.0]
n_gain = {}
n_x64 = n_waveform.astype(np.float64)
for n_g in n_gains_db:
    n_out, _ = audio.apply_gain(n_waveform, gain_db=n_g)               # float32, unclipped
    n_clipped = np.clip(n_out, -1.0, 1.0)                              # emulated hard limiter at full scale
    n_over = int(np.sum(np.abs(n_out) > 1.0))
    n_over_expected = int(np.sum(np.abs(n_x64) > 10.0 ** (-n_g / 20.0)))  # from the original amplitudes only
    n_dist = float(np.mean((n_clipped.astype(np.float64) - n_out.astype(np.float64)) ** 2))
    n_pow = float(np.mean(n_out.astype(np.float64) ** 2))
    n_scale_err = abs(np.sqrt(n_pow / n_sig_pow) / 10.0 ** (n_g / 20.0) - 1.0)
    n_gain[n_g] = dict(out=n_out, clip=n_clipped, over=n_over, over_expected=n_over_expected, dist=n_dist, power=n_pow, scale_err=n_scale_err)
    n_sdr = 10 * np.log10(n_pow / n_dist) if n_dist > 0 else float("inf")
    print(f"gain {n_g:+5.1f} dB: RMS ratio error {n_scale_err:.2e}, samples over full scale {n_over} (expected {n_over_expected}), "
          f"clip distortion power D={n_dist:.2e}, SDR={n_sdr:.2f} dB")

check("S8.1", max(v["scale_err"] for v in n_gain.values()) < 1e-5,
      f"apply_gain RMS ratio matches 10^(g/20): worst relative error {max(v['scale_err'] for v in n_gain.values()):.2e} < 1e-5 (float32 rounding)")
n_counts = [n_gain[g]["over"] for g in n_gains_db]
check("S8.2", n_counts[0] == 0 and n_counts[1] == 0 and n_counts[2] > 0 and n_counts == sorted(n_counts)
      and all(n_gain[g]["over"] == n_gain[g]["over_expected"] for g in n_gains_db),
      f"over-full-scale counts {n_counts} are monotone in gain, zero for g<=0, and equal the count implied by |x|>10^(-g/20)")
        """),

        md(r"""
### Noise operator: additive white noise at a requested SNR, and how far the realised SNR may stray

For a signal $s$ with power $P_s=\frac1N\sum_t s_t^2$, white Gaussian noise at a requested SNR of $\rho$ dB is drawn with **variance**

$$\sigma_n^2=\frac{P_s}{10^{\rho/10}},\qquad\text{equivalently noise RMS }\sigma_n=\sqrt{P_s}\;10^{-\rho/20},$$

and the degraded signal is $x=s+n$ with $n_t\sim\mathcal N(0,\sigma_n^2)$. The two forms are the same statement: the exponent is $\rho/10$ when we talk about power and $\rho/20$ when we talk about RMS amplitude. The **realised** SNR uses the noise actually drawn, $\hat P_n=\frac1N\sum_t n_t^2$:

$$\widehat{\mathrm{SNR}}=10\log_{10}\frac{P_s}{\hat P_n}.$$

**The realised SNR is random, so the tolerance must come from its sampling distribution.** Since $N\hat P_n/\sigma_n^2\sim\chi^2_N$, we have $\mathrm{Var}(\hat P_n/\sigma_n^2)=2/N$. The delta method for $g(u)=-10\log_{10}u$ at $u=1$ gives

$$\mathrm{SD}(\widehat{\mathrm{SNR}})\approx\frac{10}{\ln 10}\sqrt{\frac{2}{N}}.$$

Worked example: $N=16000$ gives $4.343\times\sqrt{1.25\times10^{-4}}=4.343\times0.01118\approx0.049$ dB. So a single realisation typically lands within about $0.05$ dB of the request, and $3$ standard errors is about $0.15$ dB; a fixed tolerance such as $0.5$ dB would be more than ten standard errors and could hide a genuine bias. The mean over $R$ independent noise draws has standard error $\mathrm{SD}/\sqrt R$, and we test the mean error against $3$ of *those*. We also test that the empirical spread matches the $\chi^2$ prediction, which would catch a noise generator with the wrong variance structure even if its mean were right. Because $\hat P_n/\sigma_n^2$ does not depend on the signal, the same tolerance applies to any waveform with the same number of samples $N$.
        """),

        code(r"""
# S8.3 realised vs requested SNR: targets 30/20/10/0 dB, R=200 noise seeds each, n=16000, reference = the unclipped clean waveform
n_R_snr = 200
n_snr_targets = [30.0, 20.0, 10.0, 0.0]
n_se_db = (10.0 / np.log(10.0)) * np.sqrt(2.0 / n_samples)          # single-realisation SD of realised SNR (delta method)
n_snr_real = {}
for n_t in n_snr_targets:
    n_vals = []
    for n_r in range(n_R_snr):
        n_noisy, _ = audio.add_white_noise(n_waveform, sample_rate=n_sr, snr_db=n_t, seed=RNG_SEED + 8000 + 1000 * int(n_t) + n_r)
        n_noise = n_noisy.astype(np.float64) - n_x64
        n_vals.append(10.0 * np.log10(n_sig_pow / np.mean(n_noise ** 2)))
    n_snr_real[n_t] = np.array(n_vals)

n_inline_vs_lib = abs(n_vals[-1] - audio.measure_snr_db(n_waveform, n_noisy))
n_mean_ok, n_sd_ok, n_lines = [], [], []
for n_t in n_snr_targets:
    n_v = n_snr_real[n_t]
    n_tol_mean = 3 * n_se_db / np.sqrt(n_R_snr)
    n_sd_ratio = n_v.std(ddof=1) / n_se_db
    n_tol_sd = 3 / np.sqrt(2 * (n_R_snr - 1))
    n_mean_ok.append(abs(n_v.mean() - n_t) <= n_tol_mean)
    n_sd_ok.append(abs(n_sd_ratio - 1) <= n_tol_sd)
    n_lines.append(f"target {n_t:5.1f}: mean err {n_v.mean() - n_t:+.4f} dB (tol {n_tol_mean:.4f}), SD/theory {n_sd_ratio:.3f} (tol +-{n_tol_sd:.3f})")
print("\n".join(n_lines))
print(f"theory: single-draw SD = {n_se_db:.4f} dB; 3 SD = {3 * n_se_db:.3f} dB")
check("S8.3", all(n_mean_ok), f"mean realised-SNR error within 3 SE of the mean for all 4 targets ({sum(n_mean_ok)}/4)")
check("S8.4", all(n_sd_ok), f"empirical SD of realised SNR matches the chi-square prediction within 3 SE ({sum(n_sd_ok)}/4)")
check("S8.5", n_inline_vs_lib < 1e-9, f"inline power formula equals audio.measure_snr_db: |diff|={n_inline_vs_lib:.2e} dB")
        """),

        md(r"""
### Clipping breaks the SNR bookkeeping: compare against the right reference

`add_white_noise` sets the noise power from *whatever waveform it is given*. If that waveform was first clipped, the requested SNR is honoured **relative to the clipped signal**. But the clipping itself is an error relative to the ideal, unclipped signal $a\,x$, so an evaluation that scores "SNR versus the intended signal" sees the noise **plus** the clipping distortion.

Let $x_c$ be the clipped signal and $D$ the clip distortion power from the previous cell. With noise variance $\sigma_n^2=P_c/10^{\rho/10}$ independent of $x_c$, the residual against the unclipped reference is $n+(x_c-a x)$, whose expected power is $\sigma_n^2+D$ (the cross term has zero mean because $n$ is zero-mean and independent). Therefore

$$\mathbb E\big[\widehat{\mathrm{SNR}}_{\text{vs unclipped}}\big]\approx10\log_{10}\frac{P_{ax}}{\sigma_n^2+D},\qquad\mathbb E\big[\widehat{\mathrm{SNR}}_{\text{vs clipped}}\big]\approx\rho.$$

This has a sharp consequence: as $\rho\to\infty$ the first expression saturates at $10\log_{10}(P_{ax}/D)$, the SDR, no matter how quiet the added noise is. We test three things. (i) Against the *clipped* reference the error is just Monte Carlo noise (the operator does what it says). (ii) Against the *unclipped* reference the measured SNR matches the formula above within $3$ standard errors of the mean over noise draws. (iii) The control gain $0$ dB, where the limiter does nothing, gives identical values for the two references, and at $+12$ dB the shortfall is larger than $3$ standard errors of the paired difference.
        """),

        code(r"""
# S8.4 clipping vs the SNR reference: gain 0 dB (control, limiter inactive) and +12 dB; noise targets 30/20/10/0 dB; R=50 seeds; n=16000
n_R_clip = 50
n_clip_res = {}
for n_g in (0.0, 12.0):
    n_xu = n_gain[n_g]["out"].astype(np.float64)            # ideal (unclipped) reference a*x
    n_xc32 = n_gain[n_g]["clip"]                             # what the operator is actually given
    n_xc = n_xc32.astype(np.float64)
    n_pu, n_pc, n_d = float(np.mean(n_xu ** 2)), float(np.mean(n_xc ** 2)), n_gain[n_g]["dist"]
    for n_t in n_snr_targets:
        n_vs_c, n_vs_u = [], []
        for n_r in range(n_R_clip):
            n_noisy, _ = audio.add_white_noise(n_xc32, sample_rate=n_sr, snr_db=n_t, seed=RNG_SEED + 8500 + 100 * int(n_t) + n_r)
            n_nz = n_noisy.astype(np.float64)
            n_vs_c.append(10 * np.log10(n_pc / np.mean((n_nz - n_xc) ** 2)))
            n_vs_u.append(10 * np.log10(n_pu / np.mean((n_nz - n_xu) ** 2)))
        n_pred = 10 * np.log10(n_pu / (n_pc / 10 ** (n_t / 10) + n_d))
        n_clip_res[(n_g, n_t)] = dict(vs_c=np.array(n_vs_c), vs_u=np.array(n_vs_u), pred=n_pred)
        print(f"gain {n_g:+5.1f} dB target {n_t:5.1f}: vs clipped {np.mean(n_vs_c):7.3f} | vs unclipped {np.mean(n_vs_u):7.3f} | predicted {n_pred:7.3f} dB")

n_tol_c = 3 * n_se_db / np.sqrt(n_R_clip)
n_ok_c = all(abs(n_clip_res[(12.0, t)]["vs_c"].mean() - t) <= n_tol_c for t in n_snr_targets)
n_ok_u = all(abs(n_clip_res[(12.0, t)]["vs_u"].mean() - n_clip_res[(12.0, t)]["pred"]) <= 3 * n_clip_res[(12.0, t)]["vs_u"].std(ddof=1) / np.sqrt(n_R_clip) for t in n_snr_targets)
n_ctrl = all(np.allclose(n_clip_res[(0.0, t)]["vs_c"], n_clip_res[(0.0, t)]["vs_u"], atol=1e-9) for t in n_snr_targets)
n_dif = n_clip_res[(12.0, 30.0)]["vs_c"] - n_clip_res[(12.0, 30.0)]["vs_u"]
n_short_lo = n_dif.mean() - 3 * n_dif.std(ddof=1) / np.sqrt(n_R_clip)
check("S8.6", n_ok_c, f"+12 dB: SNR vs clipped reference equals the request within 3 SE (tol {n_tol_c:.4f} dB) for all 4 targets")
check("S8.7", n_ok_u, "+12 dB: SNR vs unclipped reference matches 10log10(P_ax/(sigma^2+D)) within 3 SE of the mean for all 4 targets")
check("S8.8", n_ctrl and n_short_lo > 0, f"control (0 dB) gives identical references; at +12 dB / 30 dB the shortfall {n_dif.mean():.2f} dB has lower 3-SE bound {n_short_lo:.2f} dB > 0")
n_sdr12 = 10 * np.log10(n_gain[12.0]["power"] / n_gain[12.0]["dist"])
print(f"SNR ceiling when scored against the unclipped signal at +12 dB gain: SDR = {n_sdr12:.2f} dB")
        """),

        md(r"""
### Reverberation: why the earlier low-pass-like kernel was not a room, and the RIR operator that replaces it

The first version of the reverb operator, `src.audio.apply_reverb(decay_seconds)`, built a kernel from a decaying exponential **normalised to sum one** and then added $0.5$ to the first tap. That kernel has two properties that make it a poor stand-in for a room. Its sum-to-one tail is a smooth, non-negative, one-sided decay, hence a low-pass filter; and the added first tap carries almost all of the kernel's energy, so it behaves mostly as a fixed attenuation that is nearly the same for every `decay_seconds`. Nothing in it controls the *direct-to-reverberant ratio*, and the decay time is not measured, only requested. We demonstrate this below rather than assert it.

The replacement, `src.audio.apply_rir_reverb(waveform, *, sample_rate, rt60_seconds, drr_db=0.0, seed=0)`, uses a synthetic room impulse response with the standard exponential-decay model of diffuse reverberation:

$$h[0]=1,\qquad h[k]=c\,\varepsilon_k\,10^{-3k/(f_s\,T_{60})}\quad(k\ge1),\quad\varepsilon_k\sim\mathcal N(0,1).$$

The amplitude envelope $10^{-3t/T_{60}}$ has power $10^{-6t/T_{60}}$, i.e. exactly $-60$ dB at $t=T_{60}$, which is the definition of the reverberation time. The scalar $c$ is chosen so that $\mathrm{DRR}=10\log_{10}\big(h[0]^2/\sum_{k\ge1}h[k]^2\big)$ equals the request. The output is the convolution of the input with $h$, rescaled to keep the input RMS ("level matched") so that reverberation is not confounded with a gain change. The record returned by the operator carries `measured_drr_db` and `measured_rt60_seconds`.

**We do not trust those two record fields by themselves.** We recover the applied impulse response independently by passing a unit impulse through the operator (a convolution with $\delta$ returns $h$ up to the known level scale) and re-measure both quantities with our own code. DRR is scale-invariant. For $T_{60}$ we use Schroeder backward integration: the energy decay curve is $\mathrm{EDC}(t)=10\log_{10}\big(\sum_{\tau\ge t}h[\tau]^2/\sum_\tau h[\tau]^2\big)$; a line fitted over the $-5$ to $-35$ dB range is extrapolated to $-60$ dB, giving $\hat T_{60}=-60/\text{slope}$. Because the tail is random noise, $\hat T_{60}$ varies with the seed, so we test the *mean over seeds* against the request within $3$ standard errors.
        """),

        code(r"""
# S8.5 verify apply_rir_reverb: recover h by impulse response; RT60 in {0.1,0.3,0.6} s, DRR in {-5,0,+5} dB; R=100 seeds at DRR=0, R=20 otherwise
def n_recover_h(rt60, drr, seed):
    n_len = int(np.ceil(rt60 * n_sr)) + 1                       # RIR length used by the operator
    n_delta = np.zeros(n_len, dtype=np.float32); n_delta[0] = 1.0
    n_o, n_rec = audio.apply_rir_reverb(n_delta, sample_rate=n_sr, rt60_seconds=rt60, drr_db=drr, seed=seed)
    return n_o.astype(np.float64), n_rec

def n_drr_of(h):
    return 10 * np.log10(h[0] ** 2 / np.sum(h[1:] ** 2))

def n_edc_db(h):
    n_e = np.cumsum((h ** 2)[::-1])[::-1]
    return 10 * np.log10(np.maximum(n_e / n_e[0], 1e-30))

def n_rt60_of(h):
    n_edc = n_edc_db(h)
    n_i = np.where((n_edc <= -5.0) & (n_edc >= -35.0))[0]
    return -60.0 / np.polyfit(n_i / n_sr, n_edc[n_i], 1)[0]

n_rir = {}
for n_rt in (0.1, 0.3, 0.6):
    for n_drr in (-5.0, 0.0, 5.0):
        n_reps = 100 if n_drr == 0.0 else 20
        n_hs = [n_recover_h(n_rt, n_drr, RNG_SEED + 8700 + n_s) for n_s in range(n_reps)]
        n_rir[(n_rt, n_drr)] = dict(
            drr=np.array([n_drr_of(h) for h, _ in n_hs]), rt60=np.array([n_rt60_of(h) for h, _ in n_hs]),
            rec_drr=np.array([r["measured_drr_db"] for _, r in n_hs]), rec_rt60=np.array([r["measured_rt60_seconds"] for _, r in n_hs]),
            first_h=n_hs[0][0])

n_drr_err = max(np.abs(v["drr"] - d).max() for (r, d), v in n_rir.items())
n_rec_drr_err = max(np.abs(v["rec_drr"] - v["drr"]).max() for v in n_rir.values())
n_rec_rt_err = max(np.abs(v["rec_rt60"] - v["rt60"]).max() for v in n_rir.values())
n_z = {k: (v["rt60"].mean() - k[0]) / (v["rt60"].std(ddof=1) / np.sqrt(len(v["rt60"]))) for k, v in n_rir.items()}
for n_k, n_v in n_rir.items():
    print(f"RT60 req {n_k[0]:.1f} s, DRR req {n_k[1]:+.0f} dB: recovered DRR {n_v['drr'].mean():+.5f} dB; RT60 mean {n_v['rt60'].mean():.4f} s, SD {n_v['rt60'].std(ddof=1):.4f} s (z={n_z[n_k]:+.2f}, R={len(n_v['rt60'])})")
check("S8.9", n_drr_err < 1e-5, f"recovered DRR equals requested DRR: worst |error| {n_drr_err:.2e} dB < 1e-5 dB (float32 storage)")
check("S8.10", max(abs(z) for z in n_z.values()) <= 3.0, f"mean recovered RT60 within 3 SE of request in all 9 conditions: worst |z|={max(abs(z) for z in n_z.values()):.2f}")
check("S8.11", n_rec_drr_err < 1e-5 and n_rec_rt_err < 1e-6, f"record fields agree with independent recovery: DRR diff {n_rec_drr_err:.1e} dB, RT60 diff {n_rec_rt_err:.1e} s")
        """),

        code(r"""
# S8.6 level matching, determinism and legacy-kernel demonstration; waveform n_waveform, seeds fixed, no Monte Carlo
n_lvl = []
for n_rt in (0.1, 0.3, 0.6):
    for n_drr in (-5.0, 0.0, 5.0):
        n_y, n_rec = audio.apply_rir_reverb(n_waveform, sample_rate=n_sr, rt60_seconds=n_rt, drr_db=n_drr, seed=RNG_SEED + 8)
        n_lvl.append(abs(np.sqrt(np.mean(n_y.astype(np.float64) ** 2) / n_sig_pow) - 1.0))
n_y1, _ = audio.apply_rir_reverb(n_waveform, sample_rate=n_sr, rt60_seconds=0.3, seed=5)
n_y2, _ = audio.apply_rir_reverb(n_waveform, sample_rate=n_sr, rt60_seconds=0.3, seed=5)
n_y3, _ = audio.apply_rir_reverb(n_waveform, sample_rate=n_sr, rt60_seconds=0.3, seed=6)
check("S8.12", max(n_lvl) < 1e-5, f"output RMS equals input RMS in all 9 conditions: worst relative error {max(n_lvl):.2e}")
check("S8.13", np.array_equal(n_y1, n_y2) and np.abs(n_y1 - n_y3).max() > 1e-3, f"same seed reproduces output bit-for-bit; different seed differs by up to {np.abs(n_y1 - n_y3).max():.3f}")

# legacy kernel recovered from an impulse: audio.apply_reverb(decay_seconds)
n_legacy = {}
for n_dec in (0.1, 0.3, 0.6):
    n_len = int(round(n_sr * n_dec))
    n_d0 = np.zeros(2 * n_len, dtype=np.float32); n_d0[0] = 1.0
    n_h_old = audio.apply_reverb(n_d0, sample_rate=n_sr, decay_seconds=n_dec)[0].astype(np.float64)[:n_len]
    n_w_old = audio.apply_reverb(n_waveform, sample_rate=n_sr, decay_seconds=n_dec)[0].astype(np.float64)
    n_spec = np.abs(np.fft.rfft(n_h_old, 4 * n_sr)); n_f = np.fft.rfftfreq(4 * n_sr, 1 / n_sr)
    n_legacy[n_dec] = dict(h=n_h_old, drr=n_drr_of(n_h_old), first_frac=n_h_old[0] ** 2 / np.sum(n_h_old ** 2),
                           rms_ratio=np.sqrt(np.mean(n_w_old ** 2) / n_sig_pow), mag_1k=float(n_spec[np.argmin(np.abs(n_f - 1000.0))]))
    print(f"legacy decay {n_dec:.1f} s: DRR {n_legacy[n_dec]['drr']:.2f} dB, energy share of tap 0 {n_legacy[n_dec]['first_frac']:.4f}, "
          f"output/input RMS {n_legacy[n_dec]['rms_ratio']:.3f}, |H(1 kHz)| {n_legacy[n_dec]['mag_1k']:.3f}")
note("S8.14", f"legacy kernel is not a room: DRR is fixed by construction (about {np.mean([v['drr'] for v in n_legacy.values()]):.1f} dB averaged over decays, "
     f"no control), output/input RMS spans only {min(v['rms_ratio'] for v in n_legacy.values()):.3f}-{max(v['rms_ratio'] for v in n_legacy.values()):.3f} across a 6x range of decay, "
     f"so it acts mostly as an attenuation; the RIR operator at DRR 0 dB is level matched and controls both quantities")
        """),

        md(r"""
### Figure: what the four conditions look like

The next cell draws the waveform and a Hann-windowed spectrogram (512-sample window, 50 % overlap) for the clean signal, the $+12$ dB signal after the emulated limiter, the noise condition at a $10$ dB requested SNR, and the RIR reverb at $T_{60}=0.3$ s with $\mathrm{DRR}=0$ dB. Colour is power spectral density in dB relative to the loudest cell of the clean spectrogram, over an $80$ dB range, so panels are directly comparable.

Before looking at the picture, note what we can quantify. The clean signal has a broadband burst in $[0.30,0.40]$ s and harmonics below $1$ kHz. Any energy above $2$ kHz *after* $0.40$ s must therefore come from the operator, not from the source. The cell after the figure measures that energy in a window $[0.42,0.52]$ s for $T_{60}\in\{0.1,0.3,0.6\}$ s. Since a fixed DRR fixes the total tail energy while a longer $T_{60}$ spreads it over more time, the fraction of tail energy landing in a window $[t_1,t_2]$ after the burst is approximately $e^{-2\alpha t_1}-e^{-2\alpha t_2}$ with $\alpha=3\ln10/T_{60}$ (ignoring truncation and the burst's own duration). With $t_1=0.02$ s and $t_2=0.12$ s after the burst ends, this expression is an increasing function of $T_{60}$ over the three values used here. That is the prediction we check.
        """),

        code(r"""
# S8.7 figure 1: waveforms + spectrograms for clean / +12 dB clipped / SNR 10 dB / RIR reverb (T60=0.3 s, DRR 0 dB); single seeds
from scipy.signal import spectrogram as sp_spec

n_noisy10, _ = audio.add_white_noise(n_waveform, sample_rate=n_sr, snr_db=10.0, seed=RNG_SEED + 8100)
n_rev03, n_rev_rec = audio.apply_rir_reverb(n_waveform, sample_rate=n_sr, rt60_seconds=0.3, drr_db=0.0, seed=RNG_SEED + 8200)
n_panels = [("Clean", n_waveform, PALETTE["clean"]), ("+12 dB, clipped at +-1", n_gain[12.0]["clip"], PALETTE["shifted"]),
            ("Noise, SNR 10 dB", n_noisy10, PALETTE["mask"]), ("RIR reverb, T60 0.3 s, DRR 0 dB", n_rev03, PALETTE["acoustic"])]
n_ref_db = 10 * np.log10(sp_spec(n_waveform, fs=n_sr, window="hann", nperseg=512, noverlap=256)[2].max() + 1e-12)

n_fig, n_ax = plt.subplots(len(n_panels), 2, figsize=(14, 10))
for n_i, (n_lab, n_w, n_col) in enumerate(n_panels):
    n_ax[n_i, 0].plot(n_time, n_w, color=n_col, linewidth=0.5)
    n_ax[n_i, 0].axvspan(n_burst_start / n_sr, n_burst_end / n_sr, color=PALETTE["band"], alpha=0.5, lw=0)
    n_ax[n_i, 0].set_ylabel(n_lab, fontsize=8); n_ax[n_i, 0].set_xlim(0, n_duration)
    n_f_, n_t_, n_sxx = sp_spec(n_w, fs=n_sr, window="hann", nperseg=512, noverlap=256)
    n_im = n_ax[n_i, 1].pcolormesh(n_t_, n_f_, 10 * np.log10(n_sxx + 1e-12) - n_ref_db, cmap="viridis", shading="auto", vmin=-80, vmax=0)
    n_ax[n_i, 1].set_ylabel("Frequency (Hz)"); n_ax[n_i, 1].axvline(n_burst_end / n_sr, color="white", ls="--", lw=0.8)
n_ax[-1, 0].set_xlabel("Time (s)"); n_ax[-1, 1].set_xlabel("Time (s)")
n_fig.colorbar(n_im, ax=n_ax[:, 1], label="dB re clean maximum")
plt.show()
        """),

        md(r"""
### How to read this chart

Left column: waveform amplitude against time. The grey band marks the source burst, $0.30$ to $0.40$ s. Right column: spectrogram, frequency against time, with colour equal to power in dB relative to the loudest cell of the clean panel; the dashed white line marks the end of the burst. Rows are the four conditions in the order clean, $+12$ dB with the emulated limiter, additive noise at a requested $10$ dB SNR, and RIR reverb.

What to look for. In the limiter row the waveform's extremes are cut at $\pm1$ instead of continuing to the larger scaled values, and the spectrogram shows energy at frequencies above the source's harmonics outside the burst (the cut waveform is no longer a sum of those few sinusoids). In the noise row the spectrogram floor is raised across all frequencies and times, including the silent tail of the utterance. In the reverb row, compare the time extent of the broadband content around the burst with the clean row: a room adds energy to the right of the dashed line, where the clean panel has none above the harmonics. The reverb row is level matched, so its amplitudes are not a gain change.

The next cell replaces that visual impression with a number: energy above $2$ kHz after the burst has ended. The takeaway is that each operator leaves a distinct, measurable signature, and that none of these panels is evidence about real speech.
        """),

        code(r"""
# S8.8 reverb tail signature: energy above 2 kHz in the window 0.42-0.52 s (after the burst ends at 0.40 s); 5 seeds per T60, DRR 0 dB
def n_hf_tail(x):
    n_seg = x[int(0.42 * n_sr):int(0.52 * n_sr)].astype(np.float64) * np.hanning(int(0.52 * n_sr) - int(0.42 * n_sr))
    n_p = np.abs(np.fft.rfft(n_seg)) ** 2
    return float(n_p[np.fft.rfftfreq(len(n_seg), 1 / n_sr) > 2000.0].sum())

n_hf_burst = float(np.sum(np.abs(np.fft.rfft(n_waveform[n_burst_start:n_burst_end].astype(np.float64) * np.hanning(n_burst_len))) ** 2
                          * (np.fft.rfftfreq(n_burst_len, 1 / n_sr) > 2000.0)))
n_hf_clean = n_hf_tail(n_waveform)
n_hf = {}
for n_rt in (0.1, 0.3, 0.6):
    n_hf[n_rt] = float(np.mean([n_hf_tail(audio.apply_rir_reverb(n_waveform, sample_rate=n_sr, rt60_seconds=n_rt, drr_db=0.0, seed=RNG_SEED + 8300 + n_s)[0]) for n_s in range(5)]))
print(f"HF energy (>2 kHz) in burst window {n_hf_burst:.3e}; clean tail window {n_hf_clean:.3e}")
print("reverb tail window: " + ", ".join(f"T60 {r:.1f} s -> {v:.3e}" for r, v in n_hf.items()))
check("S8.15", n_hf_clean < 1e-6 * n_hf_burst, f"the clean tail window has no HF content: {n_hf_clean:.2e} < 1e-6 x burst-window energy {n_hf_burst:.2e}")
check("S8.16", n_hf_clean < n_hf[0.1] < n_hf[0.3] < n_hf[0.6], "HF energy after the burst is zero for clean and increases with T60 (0.1 < 0.3 < 0.6 s), as predicted from the exponential tail")
        """),

        code(r"""
# Figure 2 (S8.9): SNR accuracy (left: residual realised-requested, R=200 draws per target) and clipping (right: SNR scored against clipped vs unclipped reference)
n_fig2, (n_a, n_b) = plt.subplots(1, 2, figsize=(13, 4.5))
for n_i, n_t in enumerate(n_snr_targets):
    n_v = n_snr_real[n_t] - n_t
    n_a.scatter(np.full(len(n_v), n_i) + np.random.default_rng(RNG_SEED + 8).uniform(-0.15, 0.15, len(n_v)), n_v, s=6, color=PALETTE["acoustic"], alpha=0.35)
    n_a.errorbar(n_i, n_v.mean(), yerr=3 * n_se_db / np.sqrt(n_R_snr), color="black", capsize=6, marker="o")
n_a.axhspan(-3 * n_se_db, 3 * n_se_db, color=PALETTE["band"], alpha=0.5, label="single draw: +-3 theoretical SD")
n_a.axhline(0, color=PALETTE["reference"], ls="--")
n_a.set_xticks(range(4)); n_a.set_xticklabels([f"{t:.0f}" for t in n_snr_targets])
n_a.set_xlabel("Requested SNR (dB)"); n_a.set_ylabel("Realised - requested SNR (dB)"); n_a.set_title("Additive noise: realised SNR error"); n_a.legend(fontsize=8)

n_tt = np.array(n_snr_targets)
n_b.plot([-2, 32], [-2, 32], color=PALETTE["reference"], ls="--", label="y = x")
n_b.plot(n_tt, [n_clip_res[(12.0, t)]["vs_c"].mean() for t in n_snr_targets], "o-", color=PALETTE["clean"], label="+12 dB, vs clipped reference")
n_b.plot(n_tt, [n_clip_res[(12.0, t)]["vs_u"].mean() for t in n_snr_targets], "s-", color=PALETTE["shifted"], label="+12 dB, vs unclipped reference")
n_b.plot(n_tt, [n_clip_res[(12.0, t)]["pred"] for t in n_snr_targets], "k:", label="prediction 10log10(P/(sigma^2+D))")
n_b.axhline(n_sdr12, color=PALETTE["shifted"], alpha=0.4, lw=1)
n_b.set_xlabel("Requested SNR (dB)"); n_b.set_ylabel("Realised SNR (dB)"); n_b.set_title("Clipping: which reference?"); n_b.legend(fontsize=8)
plt.tight_layout(); plt.show()
        """),

        md(r"""
### How to read this chart

Left panel. Each small green point is one noise draw; the horizontal axis is the requested SNR, the vertical axis is realised minus requested SNR in dB, so $0$ is perfect agreement (dashed line). The grey band is $\pm3$ theoretical single-draw standard deviations from the delta-method formula, and the black marker with a bar is the mean over the $200$ draws with its $\pm3$ standard-error bar. Points scattered inside the band with the marker on the dashed line mean the operator delivers the requested SNR up to sampling noise; a marker off the line by more than its bar would be a bias.

Right panel. The dashed diagonal is the ideal. The blue curve scores SNR against the clipped signal that was handed to the operator: it should hug the diagonal. The red curve scores against the unclipped scaled signal: it should fall below the diagonal at high requested SNR and flatten toward the horizontal red line, the signal-to-distortion ratio of the limiter. The black dotted curve is the closed-form prediction; agreement between it and the red curve is the test.

Takeaway: additive-noise severity is reliable, but if clipping precedes it the *effective* SNR against the intended signal has a ceiling set by the clipping, so the reported severity must name its reference.
        """),

        code(r"""
# Figure 3 (S8.10): energy decay curves of the applied RIR vs the legacy kernel (left), recovered RT60 vs requested with 3-SE bars (right); DRR 0 dB, R=100 seeds
n_fig3, (n_l, n_r) = plt.subplots(1, 2, figsize=(13, 4.5))
n_cols = [PALETTE["clean"], PALETTE["acoustic"], PALETTE["shifted"]]
for n_c, n_rt in zip(n_cols, (0.1, 0.3, 0.6)):
    n_h = n_rir[(n_rt, 0.0)]["first_h"]
    n_l.plot(np.arange(len(n_h)) / n_sr * 1000, n_edc_db(n_h), color=n_c, label=f"RIR, T60={n_rt:.1f} s")
    n_ho = n_legacy[n_rt]["h"]
    n_l.plot(np.arange(len(n_ho)) / n_sr * 1000, n_edc_db(n_ho), color=n_c, ls="--", alpha=0.8, label=f"legacy kernel, decay={n_rt:.1f} s")
n_l.axhline(-5, color=PALETTE["reference"], lw=0.8); n_l.axhline(-35, color=PALETTE["reference"], lw=0.8)
n_l.set_ylim(-65, 2); n_l.set_xlim(0, 620)
n_l.set_xlabel("Time (ms)"); n_l.set_ylabel("EDC (dB)"); n_l.set_title("Energy decay curves"); n_l.legend(fontsize=7)

n_req = np.array([0.1, 0.3, 0.6])
n_mean = np.array([n_rir[(r, 0.0)]["rt60"].mean() for r in n_req])
n_se3 = np.array([3 * n_rir[(r, 0.0)]["rt60"].std(ddof=1) / np.sqrt(100) for r in n_req])
n_r.plot([0, 0.7], [0, 0.7], color=PALETTE["reference"], ls="--", label="y = x")
n_r.errorbar(n_req, n_mean, yerr=n_se3, fmt="o", color=PALETTE["acoustic"], capsize=5, label="mean of 100 seeds +-3 SE")
n_r.set_xlabel("Requested T60 (s)"); n_r.set_ylabel("Recovered T60 (s)"); n_r.set_title("Reverberation time check"); n_r.legend(fontsize=8)
plt.tight_layout(); plt.show()
        """),

        md(r"""
### How to read this chart

Left panel. The vertical axis is the Schroeder energy decay curve in dB (energy remaining after time $t$, relative to total), the horizontal axis is time in milliseconds. Solid curves are the applied RIR for three requested $T_{60}$ values; dashed curves of the same colour are the legacy kernel with the same requested decay. The two thin horizontal lines mark the $-5$ and $-35$ dB limits of the fitting range. A room-like response falls roughly along a straight line that reaches $-60$ dB at the requested $T_{60}$; the steeper the line, the shorter the reverberation.

Compare the two families. The solid curves start at the top and fall along a line whose slope depends on the requested $T_{60}$. The dashed curves behave differently: nearly all of their energy sits in the first tap (the cell above prints the share), so each curve makes a large step down at the very start, then decays from that lower level and therefore reaches $-60$ dB before the requested $T_{60}$ instead of at it.

Right panel. Requested $T_{60}$ on the horizontal axis, recovered $T_{60}$ on the vertical axis, with the diagonal as the ideal. The green markers are means over $100$ seeds with $\pm3$ standard-error bars. A marker whose bar overlaps the diagonal agrees with the request within sampling noise.

Takeaway: the RIR operator's stated $T_{60}$ and DRR are properties we can measure, which the legacy kernel could not offer.
        """),

        code(r"""
# S8.11 sanity: every operator output is finite float32 with the input length; recap of measured worst cases from this section (no new random draws)
n_outs = {"gain+12 clipped": n_gain[12.0]["clip"], "noise 10 dB": n_noisy10, "rir 0.3 s": n_rev03,
          "legacy 0.3 s": audio.apply_reverb(n_waveform, sample_rate=n_sr, decay_seconds=0.3)[0]}
for n_nm, n_o in n_outs.items():
    print(f"{n_nm:16s}: dtype {n_o.dtype}, length {n_o.shape[0]}, finite {bool(np.isfinite(n_o).all())}, RMS {np.sqrt(np.mean(n_o.astype(np.float64) ** 2)):.4f}")
check("S8.17", all(o.dtype == np.float32 and o.shape == (n_samples,) and np.isfinite(o).all() for o in n_outs.values()),
      "all operator outputs are finite float32 arrays of the input length")
check("S8.18", n_rev_rec["kind"] == "rir_reverb" and abs(n_rev_rec["measured_drr_db"] - 0.0) < 1e-5 and abs(n_rev_rec["rt60_seconds"] - 0.3) < 1e-12,
      f"record of the figure's reverb condition: kind={n_rev_rec['kind']}, measured DRR {n_rev_rec['measured_drr_db']:.2e} dB, measured RT60 {n_rev_rec['measured_rt60_seconds']:.4f} s")
        """),

        md(r"""
## 9. Noise-injected mixup, a NoisyMix-style analogue without the JSD stability term (source: Erichson et al., AISTATS 2024; q6, q7, q8)

### What NoisyMix is, according to the traces

The paper claims below are restricted to the raw traces q6, q7 and q8.

> q6: the NoisyMix objective combines three ingredients: stochastically augmented images (AugAndMix), noisy feature mixup (NFM), and Jensen-Shannon-divergence (JSD) stability training. The total loss is $\mathcal L_{\text{NoisyMix}}=\mathcal L_{\text{NFM}}+\gamma\,\mathcal L_{\text{stability}}$ (Section 3.1, Eq. 2).

> q6, NFM: $M_{\lambda,\xi}(x,x')=(1+\sigma_1\xi_{\text{mult}})\odot M_\lambda(x,x')+\sigma_2\xi_{\text{add}}$ with $M_\lambda(a,b)=\lambda a+(1-\lambda)b$, $\lambda\sim\mathrm{Beta}(\alpha,\beta)$, and $\xi=(\xi_{\text{add}},\xi_{\text{mult}})$ zero-mean noise with finite first two moments; the labels are mixed the same way, $M_\lambda(y,y')$.

> q6, stability term: the JSD between the predictions on the noisy-mixed clean pair and on the noisy-mixed augmented pair, $\mathcal L_{\text{stability}}=\mathbb E\big[\mathrm{JS}_\pi\big(p(M_{\lambda,\xi}(x,x')),\,p(M_{\lambda,\xi}(x_{\text{am}},x'_{\text{am}}))\big)\big]$.

q6 also reports Theorem 1 (App. B.1): in a small-noise regime, minimising the NFM loss is second-order equivalent to minimising the standard loss plus data-dependent regularisers that penalise the derivatives $\nabla p(x)$ and $\nabla^2p(x)$ scaled by the noise variances, which the paper reads as larger margins and smoother decision boundaries. Theorem 2 (App. B.2) gives the analogous statement for the JSD term. The evaluation in the trace is exclusively on 2D image benchmarks (ImageNet-family and CIFAR-family); q6 and q7 give as an explicit limitation that the method is not directly applicable to time-series tasks.

**Vision-only components (q7).** The AugAndMix operations $C_i$ (rotation, translation, shearing, posterisation, solarisation, autocontrast, equalisation) are pixel-based. q7 states that applying such 2D operations to a log-mel spectrogram destroys harmonic structure, pitch contours and temporal frame order, and that spectrogram axes are heterogeneous (logarithmic frequency against linear time), unlike the isotropic axes of an image. These are q7's statements about audio; we do not test them here.

**Status of the transfer to audio (q8).** q8 places the NFM loss and the JSD stability objective in its "Transferable Hypotheses" list, described as domain-agnostic loss formulations validated on other domains. That is a *hypothesis* in q8's own wording, not a finding.

### What this section actually does

We implement the NFM ingredient on a toy feature-space problem: $\lambda\sim\mathrm{Beta}(1,1)$, multiplicative and additive Gaussian noise, and mixed soft labels. There is **no JSD term and no AugAndMix**, so the regime is called *noise-injected mixup, a NoisyMix-style analogue without the JSD stability term*. It is not NoisyMix and its behaviour says nothing about NoisyMix's reported gains.

The toy is a two-class Gaussian problem with only six informative dimensions out of twenty and a training set of sixty points, chosen so that even the Bayes-optimal rule is well below perfect accuracy and a finite-sample learner is visibly worse than Bayes. At test time we apply an additive-noise shift of standard deviation $s$ to every feature. Because the classes are isotropic Gaussians with means $\pm\mu$ and unit covariance, additive noise inflates the variance to $1+s^2$ and the Bayes rule stays $\mathrm{sign}(\mu^\top x)$, with accuracy $\Phi\big(\lVert\mu\rVert/\sqrt{1+s^2}\big)$. That closed form gives us a ceiling to test against.

**Regimes compared (all use logistic regression with a matched effective penalty per unit of average loss):** clean ERM; clean ERM with a stronger $\ell_2$ penalty (a control, single value $C=0.1$, not tuned); noise-only augmentation (multiplicative and additive noise on resampled training points, no mixing); mixup only ($\sigma=0$); and noise-injected mixup ($\sigma_1=\sigma_2=0.5$, a single pilot value, not tuned). The control matters because q6's Theorem 1 describes NFM as an implicit regulariser, so any gain must be compared with what plain explicit regularisation buys.

**Hypotheses.** A single pilot run informed the choice of $\sigma$, $C$ and the toy dimensions, and the hypotheses below were written after that pilot, so treat them as exploratory rather than pre-registered. (H1) Noise-injected mixup gives better calibrated probabilities than clean ERM under the noise shift, measured by NLL and ECE. (H2) The noise component is what matters for that gain, so it should beat mixup-only. Whether it also beats plain $\ell_2$ regularisation, and whether accuracy improves, are reported as observations, with verdict strings computed from the confidence intervals.
        """),

        code(r"""
# S9.1 toy DGP + regime builders: d=20 (6 informative dims, mean +-0.45), n_train=60, n_test=4000, R=100 reps, noise-shift sd s in {0,.5,1,1.5,2}, seed RNG_SEED+9
from sklearn.linear_model import LogisticRegression
from scipy.stats import norm
from src.metrics import nll, expected_calibration_error, bootstrap_interval

n_d, n_k, n_mu_dim = 20, 6, 0.45
n_mu = np.zeros(n_d); n_mu[:n_k] = n_mu_dim
n_ntrain, n_ntest, n_R9 = 60, 4000, 100
n_svals = [0.0, 0.5, 1.0, 1.5, 2.0]
n_sig_aug = 0.5                                     # sigma_1 = sigma_2, one pilot value, not tuned
n_bayes_analytic = {s: float(norm.cdf(np.linalg.norm(n_mu) / np.sqrt(1 + s ** 2))) for s in n_svals}

def n_draw(rng, n):
    y = rng.integers(0, 2, n)
    return rng.standard_normal((n, n_d)) + np.where(y[:, None] == 1, n_mu, -n_mu), y

def n_fit(X, y, w=None, c=1.0):
    total = len(y) if w is None else float(w.sum())          # keeps the penalty per unit of average loss fixed across regimes
    return LogisticRegression(C=c * n_ntrain / total, max_iter=1000).fit(X, y, sample_weight=w)

def n_nfm(rng, X, y, mixing, noise, m=4):
    # M_{lambda,xi}(x,x') = (1+sigma1*xi_mult) * (lam*x+(1-lam)*x') + sigma2*xi_add ; labels mixed with the same lambda (soft labels -> weighted rows)
    n = len(y); M = m * n
    i, j = rng.integers(0, n, M), rng.integers(0, n, M)
    lam = rng.beta(1.0, 1.0, M)[:, None] if mixing else np.ones((M, 1))
    Xm = lam * X[i] + (1 - lam) * X[j]
    if noise:
        Xm = (1 + n_sig_aug * rng.standard_normal(Xm.shape)) * Xm + n_sig_aug * rng.standard_normal(Xm.shape)
    t = lam[:, 0] * y[i] + (1 - lam[:, 0]) * y[j]
    return np.vstack([Xm, Xm]), np.r_[np.ones(M), np.zeros(M)].astype(int), np.r_[t, 1 - t]

n_regimes = ["ERM (clean)", "ERM + stronger L2 (control)", "noise-only augmentation", "mixup only", "noise-injected mixup (NFM-style)"]
print(f"informative-subspace norm |mu|={np.linalg.norm(n_mu):.3f}; analytic Bayes accuracy by shift s: " + ", ".join(f"s={s}: {v:.4f}" for s, v in n_bayes_analytic.items()))
        """),

        code(r"""
# S9.2 Monte Carlo: R=100 paired reps (same train/test draw for all regimes), seeds RNG_SEED+9000+rep; metrics accuracy / NLL / top-label ECE (10 bins)
n_M = {k: np.zeros((len(n_regimes), len(n_svals), n_R9)) for k in ("acc", "nll", "ece")}
n_bayes_emp = np.zeros((len(n_svals), n_R9))
for n_rep in range(n_R9):
    n_rng9 = np.random.default_rng(RNG_SEED + 9000 + n_rep)
    n_Xtr, n_ytr = n_draw(n_rng9, n_ntrain)
    n_Xte, n_yte = n_draw(n_rng9, n_ntest)
    n_eps = n_rng9.standard_normal(n_Xte.shape)
    n_models = [n_fit(n_Xtr, n_ytr), n_fit(n_Xtr, n_ytr, c=0.1)]
    n_models.append(n_fit(*n_nfm(n_rng9, n_Xtr, n_ytr, mixing=False, noise=True)))
    n_models.append(n_fit(*n_nfm(n_rng9, n_Xtr, n_ytr, mixing=True, noise=False)))
    n_models.append(n_fit(*n_nfm(n_rng9, n_Xtr, n_ytr, mixing=True, noise=True)))
    for n_si, n_s in enumerate(n_svals):
        n_Xs = n_Xte + n_s * n_eps
        n_bayes_emp[n_si, n_rep] = np.mean((n_Xs @ n_mu > 0).astype(int) == n_yte)
        for n_ri, n_mod in enumerate(n_models):
            n_p = n_mod.predict_proba(n_Xs)
            n_M["acc"][n_ri, n_si, n_rep] = np.mean(n_p.argmax(1) == n_yte)
            n_M["nll"][n_ri, n_si, n_rep] = nll(n_p, n_yte)
            n_M["ece"][n_ri, n_si, n_rep] = expected_calibration_error(n_p, n_yte)
print(f"finished {n_R9} paired repetitions x {len(n_regimes)} regimes x {len(n_svals)} shift levels")
        """),

        code(r"""
# S9.3 results at shift s=1.0 (index 2): bootstrap 95% CIs over the 100 Monte Carlo repetitions; paired differences vs ERM; checks
n_si1 = n_svals.index(1.0)
def n_ci(a):
    b = bootstrap_interval(np.asarray(a), seed=RNG_SEED + 9)
    return b.point, b.lower, b.upper
def n_verdict(a, lower_is_better=True):
    p, lo, hi = n_ci(a)
    if hi < 0: return "lower" if lower_is_better else "worse"
    if lo > 0: return "higher" if lower_is_better else "better"
    return "not distinguishable from zero"

print(f"{'regime':36s} {'accuracy [95% CI]':26s} {'NLL [95% CI]':26s} {'ECE [95% CI]':26s}")
for n_ri, n_nm in enumerate(n_regimes):
    print(f"{n_nm:36s} " + " ".join(f"{n_ci(n_M[k][n_ri, n_si1])[0]:.3f} [{n_ci(n_M[k][n_ri, n_si1])[1]:.3f},{n_ci(n_M[k][n_ri, n_si1])[2]:.3f}]".ljust(26) for k in ("acc", "nll", "ece")))
print(f"Bayes accuracy at s=1: analytic {n_bayes_analytic[1.0]:.4f}, empirical mean {n_bayes_emp[n_si1].mean():.4f}")

n_erm, n_l2, n_noise, n_mix, n_nfm_i = 0, 1, 2, 3, 4
n_d_nll = n_M["nll"][n_nfm_i, n_si1] - n_M["nll"][n_erm, n_si1]
n_d_ece = n_M["ece"][n_nfm_i, n_si1] - n_M["ece"][n_erm, n_si1]
n_d_mix = n_M["nll"][n_nfm_i, n_si1] - n_M["nll"][n_mix, n_si1]
n_d_gap = n_M["acc"][n_erm, n_si1] - n_bayes_emp[n_si1]
n_bay_se = n_bayes_emp[:, :].std(axis=1, ddof=1) / np.sqrt(n_R9)
check("S9.1", all(abs(n_bayes_emp[i].mean() - n_bayes_analytic[s]) <= 3 * n_bay_se[i] for i, s in enumerate(n_svals)),
      "empirical Bayes accuracy matches Phi(|mu|/sqrt(1+s^2)) within 3 SE at every shift level (DGP and shift implemented as derived)")
check("S9.2", n_ci(n_d_gap)[2] < 0 and n_ci(n_M["acc"][n_erm, n_si1])[2] < 1.0,
      f"DGP is not saturated: ERM accuracy {n_M['acc'][n_erm, n_si1].mean():.3f} is below Bayes (paired gap CI upper {n_ci(n_d_gap)[2]:.4f} < 0)")
check("S9.3", n_ci(n_d_nll)[2] < 0, f"H1: noise-injected mixup NLL minus ERM NLL at s=1: {n_ci(n_d_nll)[0]:+.3f} CI [{n_ci(n_d_nll)[1]:+.3f},{n_ci(n_d_nll)[2]:+.3f}] (need upper < 0)")
check("S9.4", n_ci(n_d_ece)[2] < 0, f"H1: noise-injected mixup ECE minus ERM ECE at s=1: {n_ci(n_d_ece)[0]:+.3f} CI [{n_ci(n_d_ece)[1]:+.3f},{n_ci(n_d_ece)[2]:+.3f}] (need upper < 0)")
check("S9.5", n_ci(n_d_mix)[2] < 0, f"H2: noise-injected mixup NLL minus mixup-only NLL at s=1: {n_ci(n_d_mix)[0]:+.3f} CI [{n_ci(n_d_mix)[1]:+.3f},{n_ci(n_d_mix)[2]:+.3f}] (need upper < 0)")
        """),

        code(r"""
# S9.4 observations that are reported, not asserted: accuracy, the plain-L2 control, and the clean-shift (s=0) cost; verdict strings come from the CIs
n_d_acc = n_M["acc"][n_nfm_i, n_si1] - n_M["acc"][n_erm, n_si1]
n_d_acc_noise = n_M["acc"][n_noise, n_si1] - n_M["acc"][n_erm, n_si1]
n_d_nll_l2 = n_M["nll"][n_nfm_i, n_si1] - n_M["nll"][n_l2, n_si1]
n_d_acc_l2 = n_M["acc"][n_nfm_i, n_si1] - n_M["acc"][n_l2, n_si1]
n_d_clean_nll = n_M["nll"][n_nfm_i, 0] - n_M["nll"][n_erm, 0]
n_d_clean_acc = n_M["acc"][n_nfm_i, 0] - n_M["acc"][n_erm, 0]
def n_fmt(a): return f"{n_ci(a)[0]:+.4f} CI [{n_ci(a)[1]:+.4f},{n_ci(a)[2]:+.4f}]"
note("S9.6", f"accuracy, noise-injected mixup minus ERM at s=1: {n_fmt(n_d_acc)} -> {n_verdict(n_d_acc, False)}")
note("S9.7", f"accuracy, noise-only augmentation minus ERM at s=1: {n_fmt(n_d_acc_noise)} -> {n_verdict(n_d_acc_noise, False)}")
note("S9.8", f"control: NLL of noise-injected mixup minus the stronger-L2 ERM at s=1: {n_fmt(n_d_nll_l2)} -> NFM-style NLL is {n_verdict(n_d_nll_l2)} than plain L2")
note("S9.9", f"control: accuracy of noise-injected mixup minus the stronger-L2 ERM at s=1: {n_fmt(n_d_acc_l2)} -> {n_verdict(n_d_acc_l2, False)}")
note("S9.10", f"no shift (s=0): NLL of noise-injected mixup minus ERM {n_fmt(n_d_clean_nll)}; accuracy {n_fmt(n_d_clean_acc)}")
n_best_acc = int(np.argmax([n_M["acc"][i, n_si1].mean() for i in range(len(n_regimes))]))
n_best_nll = int(np.argmin([n_M["nll"][i, n_si1].mean() for i in range(len(n_regimes))]))
print(f"best mean accuracy at s=1: {n_regimes[n_best_acc]}; best mean NLL at s=1: {n_regimes[n_best_nll]}")
        """),

        code(r"""
# Figure 4 (S9.5): accuracy and NLL versus test-noise sd s for all five regimes, bootstrap 95% CI bands over R=100 paired repetitions, plus analytic Bayes accuracy
n_fig4, (n_p, n_q) = plt.subplots(1, 2, figsize=(13, 4.6))
n_colors = [PALETTE["reference"], PALETTE["adapted"], PALETTE["mask"], PALETTE["frozen"], PALETTE["shifted"]]
for n_ri, (n_nm, n_c) in enumerate(zip(n_regimes, n_colors)):
    for n_ax_, n_key in ((n_p, "acc"), (n_q, "nll")):
        n_cis = np.array([n_ci(n_M[n_key][n_ri, n_si]) for n_si in range(len(n_svals))])
        n_ax_.plot(n_svals, n_cis[:, 0], marker="o", color=n_c, label=n_nm if n_key == "acc" else None)
        n_ax_.fill_between(n_svals, n_cis[:, 1], n_cis[:, 2], color=n_c, alpha=0.2)
n_p.plot(n_svals, [n_bayes_analytic[s] for s in n_svals], "k--", label="Bayes (analytic)")
n_p.set_xlabel("Test-time noise sd s"); n_p.set_ylabel("Test accuracy"); n_p.set_title("Accuracy under additive-noise shift")
n_q.set_xlabel("Test-time noise sd s"); n_q.set_ylabel("Test NLL (lower is better)"); n_q.set_title("Negative log-likelihood under shift")
n_p.legend(fontsize=7)
plt.tight_layout(); plt.show()
        """),

        md(r"""
### How to read this chart

Left panel: test accuracy against the standard deviation $s$ of the additive test-time noise, one coloured line per training regime, shaded band a bootstrap $95\%$ confidence interval over $100$ paired Monte Carlo repetitions. The dashed black line is the analytic Bayes accuracy $\Phi(\lVert\mu\rVert/\sqrt{1+s^2})$; no learner can sit above it, and the vertical gap between a line and the dashed curve is the price of learning from sixty points. Right panel: negative log-likelihood against $s$ for the same regimes; lower is better and a steeper rise means confidence is degrading faster under the shift.

How to compare. Where two bands do not overlap, the ordering is supported at that shift level; where they overlap, this experiment cannot separate the regimes at that level. Because repetitions are paired, differences are more sharply resolved than the raw bands suggest, and the paired numbers are the ones printed in the cells above and used by the checks. Read the accuracy panel and the NLL panel separately: a regime can improve probability quality without improving the error rate.

What would make the conclusion wrong: a different informative-subspace size, a different training-set size, a tuned penalty for the control, or a different noise strength $\sigma$ could change the ordering. The experiment supports only the statements printed above; it does not show that the gains would carry over to speech features.
        """),

        md(r"""
### Reading the toy honestly

The toy makes three separate kinds of statement, kept apart here.

**Paper claims (q6, q7, q8).** NFM is defined as in the q6 blockquote; Theorem 1 says it is second-order equivalent to standard loss plus derivative-penalising regularisers; the reported empirical results are on image benchmarks; the authors state the method is not directly applicable to time-series tasks.

**Toy observations (printed by the cells above).** They concern a linear model on Gaussian features. They include a **negative finding that we designed the toy to be able to show**: the plain stronger-$\ell_2$ control is a competitor that any claimed benefit of augmentation has to be measured against, and the printed notes state how it compares. Accuracy comparisons are reported with confidence intervals and verdict strings that are computed, not written.

**Hypotheses (not tested here).** (i) The transfer of NFM to speech features (q8 lists it as a transferable hypothesis). (ii) That audio-native perturbations such as reverberation or spectral masking matter more than generic feature noise for speech: this is an open question we do not answer, since neither the toy nor the operators above train a speech model. (iii) That the JSD stability term adds robustness in audio; we did not implement it, so the toy is silent on it.
        """),

        md(r"""
## 10. Synthesis: measure, decide, monitor, and what the sources do not settle (derived here; q3, q8, q12)

### Recap table: steps, source, guarantee form, and status

Status column: **source** = stated in the cited raw trace; **derived** = derived and tested in this notebook; **hypothesis** = an inference of ours or a q8 "transferable hypothesis", not established.

| Step | Where | Guarantee or measurement | Status |
|---|---|---|---|
| Measure gain | `src.audio.apply_gain` (Section 8) | amplitude scale $a=10^{g/20}$, power scale $a^2$; float32 output is not clipped | derived |
| Measure noise | `src.audio.add_white_noise` (Section 8) | requested SNR $\rho$ gives $\sigma_n^2=P_s/10^{\rho/10}$; realised SNR has single-draw SD $\approx4.34\sqrt{2/N}$ dB | derived |
| Measure reverb | `src.audio.apply_rir_reverb` (Section 8) | synthetic exponential-decay RIR; DRR and $T_{60}$ requested and re-measured by impulse recovery | derived |
| Risk control, exchangeable data | Angelopoulos et al., 2208.02814, Theorem 1 (q3) | $\mathbb E[L_{n+1}(\hat\lambda)]\le\alpha$ for monotone bounded losses | source |
| Non-monotone loss | 2208.02814, Proposition 2 and Theorem C.1 (q3, q12) | Prop. 2: for any $\epsilon$ a non-monotone loss with $\mathbb E[L_{n+1}(\hat\lambda)]\ge B-\epsilon$; Theorem C.1: monotonised empirical loss gives asymptotic control | source |
| Covariate shift | 2208.02814, Proposition 3 (q3) | weighted threshold with $w(x)=dP_{\text{test}}/dP_{\text{train}}$ gives exact control | source; use for acoustic shift is a q8 hypothesis |
| General shift | 2208.02814, Proposition 4 (q3, q12) | $\mathbb E[L_{n+1}(\hat\lambda)]\le\alpha+B\sum_i\mathrm{TV}(Z_i,Z_{n+1})$ | source; whether it is informative for reverberation is a hypothesis |
| Noisy feature mixup | Erichson et al., Theorem 1 (q6) | second-order equivalence to standard loss plus derivative penalties | source (vision experiments) |
| Stability loss | Erichson et al., Theorem 2 (q6) | JSD term regularises first and second derivatives on clean and augmented data | source (vision experiments) |
| Noise-injected mixup toy | Section 9 | measured NLL/ECE/accuracy against ERM and a stronger-$\ell_2$ control on a Gaussian toy | derived; transfer to speech is a hypothesis |
        """),

        md(r"""
### What the sources do not promise

This list keeps three categories separate: things the traces state, things they do not address, and our own inferences (marked as hypotheses).

**Conformal risk control (2208.02814), per q12 and q3.**
- Its Section 5 states two primary limitations: the requirement of a monotone loss is difficult to lift, and extensions to non-exchangeable data require knowledge about the form of the shift (q12, quoted below).
- Proposition 2 (q12, Section 2.3): for any $\epsilon$ there is a non-monotone loss for which $\mathbb E[L_{n+1}(\hat\lambda)]\ge B-\epsilon$. Per q3, Theorem C.1 gives asymptotic control for a monotonised empirical loss.
- Proposition 4 gives a total-variation bound for arbitrary shifts (q12, Section 4.1). *Hypothesis, not in q12:* that this bound is uninformative when acoustic shift is large. q12 lists trivial thresholds that force abstention on every command as a possible negative finding to look for, not as a result.

**NoisyMix (Erichson et al., 2024), per q6, q7, q12.**
- Its conclusion states a limitation: it is tailored towards computer vision tasks and not directly applicable to time-series tasks (q6, q7, q12).
- q12 quotes Appendix D.1 on architecture limits in learning high-frequency features in low-dimensional domains, which could imply the model tends toward features similar in frequency to the in-domain task. *Hypothesis of ours, not in q12:* that this bears on channel or bandwidth shift in speech.

**Not measured in this notebook:** real speech-command data (Speech Commands, 1804.03209), trained speech models, pretrained speech backbones, or any real reverberation recording. Everything above is a controlled synthetic experiment.
        """),

        md(r"""
### Limitations and concrete experimental pursuits (q12)

q12 lists three limitations of the sources plus one capture problem. For the first three, the pursuit and the "negative finding" (a result that would count against the approach) are q12's own wording, summarised; the verbatim quotations are marked with quotation marks.

#### 1. Loss monotonicity constraint in risk control

- Quoted (q12, Section 5): *"two primary limitations of our technique remain: firstly, the requirement of a monotone loss is difficult to lift."*
- Pursuit (q12): evaluate a speech-command classifier with a joint abstention-and-error loss that penalises both misclassification and set size, over a range of thresholds.
- Falsifier (q12): applying the threshold estimator to the non-monotone loss gives empirical test risk above the target $\alpha$, or monotonising the empirical loss gives such conservative sets that the system abstains almost always.

#### 2. Knowledge of the shift form

- Quoted (q12, Section 5): *"secondly, extensions to non-exchangeable data require knowledge about the form of the shift. This issue affects most statistical methods, including standard conformal prediction, and ours is no different in this regard."*
- Pursuit (q12): calibrate on clean audio, deploy on reverberant audio, and estimate likelihood ratios with a domain classifier on unlabelled buffers for weighted risk control.
- Falsifier (q12): errors in the estimated density ratio cause the weighted procedure to under-cover, or the total-variation route yields trivial thresholds that abstain on every command.

#### 3. Vision-specific augmentation

- Quoted (q12, Section 5): *"A limitation of NoisyMix is that it is tailored towards computer vision tasks and not directly applicable to natural language processing tasks, or time series tasks."*
- Pursuit (q12): an "Acoustic-NoisyMix" that replaces the 2D visual transformations with acoustic operations (room impulse response, SpecAugment-style masking, additive noise), trained on speech commands and evaluated for calibration and abstention under acoustic shift.
- Falsifier (q12): no statistically significant improvement in out-of-domain accuracy or RMS calibration error over clean training, or degraded clean accuracy from feature noise.

#### 4. Capture gap and a further pursuit of ours

- q12 (item 4) notes that one of the notebook sources captured only a browser-verification page. The selective-classification statements in this notebook come from a separately obtained copy (the source tagged L in Sections 3 and 4), not from that capture.
- *Hypothesis and pursuit of ours, not in q12:* train a speech classifier on full-band audio and test on band-limited audio, to see whether the frequency-bias remark in Appendix D.1 shows up as a channel-mismatch failure. No expected effect size is claimed.
        """),

        md(r"""
### What Notebook 2 will add

Notebook 2 moves from these controlled constructions to real data and trained models. The items below are plans, not results.

1. **Real Speech Commands data** (1804.03209). The dataset's exact size, vocabulary and split will be read from the downloaded files and reported by code, not asserted here.
2. **Convolutional models** on mel-spectrogram inputs. Whether NoisyMix-style regularisation helps them is a hypothesis to test, in the sense of q8's "transferable hypotheses".
3. **A pre-trained self-supervised speech backbone**, frozen or fine-tuned, to compare against hand-crafted features. The specific checkpoint will be fixed in Notebook 2.
4. **Latency and jitter** at batch size one under the acoustic shifts defined here, alongside calibration as the RIR reverberation time is varied.
5. **Held-out-word abstention**: some vocabulary words withheld from training and presented as unknown inputs, to measure the selective risk and coverage trade-off.
6. **Paired statistical comparisons**: exchangeable conformal risk control against a weighted variant using an estimated density ratio on real noise recordings, with bootstrap intervals from `src.metrics.bootstrap_interval`.
        """),

        md(r"""
### References (inline, by arXiv id and q-tag)

- **2208.02814**, Angelopoulos, Bates, Fisch, Lei and Schuster, *Conformal Risk Control*, ICLR 2024 (authors and venue as in q1). Theorem 1, Propositions 2 to 4 and Theorem C.1 are named as in q3; the limitations quoted in Section 10 are from q12.
- **2405.05160**, Liang, Peng and Sun, *Selective Classification Under Distribution Shifts*, TMLR 2024. The statements attributed to it in this notebook rest on the separately re-verified copy tagged L in Sections 3 and 4. The toy in Section 9 does not test selective risk.
- **NoisyMix**, Erichson, Lim, Xu, Utrera, Cao and Mahoney, *NoisyMix: Boosting Model Robustness to Common Corruptions*, AISTATS 2024, PMLR v238 (title, authors and venue as in q1). Equations and Theorems 1 and 2 are as in q6; the limitations are as in q6, q7 and q12; the audio-transfer statements are q8's "transferable hypotheses" and "unsupported extrapolations".
- **1804.03209**, Warden, *Speech Commands: A Dataset for Limited-Vocabulary Speech Recognition*, 2018. Cited only as the planned dataset for Notebook 2; no dataset statistics are quoted here.
        """),

        code(r"""
# S10 summary of what this part measured (all numbers printed from the variables computed above; no new random draws)
n_worst_snr_z = max(abs(n_snr_real[t].mean() - t) / (n_se_db / np.sqrt(n_R_snr)) for t in n_snr_targets)
n_worst_rt_z = max(abs(z) for z in n_z.values())
print("Section 8 (operators):")
print(f"  single-draw SNR SD (theory) {n_se_db:.4f} dB; worst |z| of mean SNR error {n_worst_snr_z:.2f}; worst |z| of mean RT60 {n_worst_rt_z:.2f}")
print(f"  worst recovered-DRR error {n_drr_err:.1e} dB; SDR ceiling at +12 dB clipping {n_sdr12:.2f} dB")
print("Section 9 (toy, s=1):")
for n_ri, n_nm in enumerate(n_regimes):
    print(f"  {n_nm:36s} acc {n_M['acc'][n_ri, n_si1].mean():.3f}  NLL {n_M['nll'][n_ri, n_si1].mean():.3f}  ECE {n_M['ece'][n_ri, n_si1].mean():.3f}")
print(f"  Bayes accuracy (analytic) {n_bayes_analytic[1.0]:.3f}")
        """),

        code(r"""
# Final check summary
check_summary()
        """),
    ]
