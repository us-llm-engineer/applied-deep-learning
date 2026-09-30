"""CPU-safe log-mel features and a small convolutional classifier."""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


def _hz_to_mel(hz: torch.Tensor) -> torch.Tensor:
    return 2595.0 * torch.log10(1.0 + hz / 700.0)


def _mel_to_hz(mel: torch.Tensor) -> torch.Tensor:
    return 700.0 * (torch.pow(10.0, mel / 2595.0) - 1.0)


def _mel_filter(sample_rate: int, n_fft: int, n_mels: int, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Triangular Slaney-style mel bank built in torch (no import-time downloads)."""
    low = torch.tensor(0.0, device=device, dtype=dtype)
    high = torch.tensor(sample_rate / 2.0, device=device, dtype=dtype)
    points = _mel_to_hz(torch.linspace(_hz_to_mel(low), _hz_to_mel(high), n_mels + 2, device=device, dtype=dtype))
    frequencies = torch.linspace(0.0, sample_rate / 2.0, n_fft // 2 + 1, device=device, dtype=dtype)
    left = (frequencies[None, :] - points[:-2, None]) / (points[1:-1, None] - points[:-2, None]).clamp_min(1e-12)
    right = (points[2:, None] - frequencies[None, :]) / (points[2:, None] - points[1:-1, None]).clamp_min(1e-12)
    return torch.minimum(left, right).clamp_min(0.0)


def log_mel(
    waveforms: torch.Tensor,
    *,
    sample_rate: int = 16_000,
    n_mels: int = 40,
    hop_length: int = 160,
    n_fft: int = 400,
    backend: str = "torch",
) -> torch.Tensor:
    """Convert ``(batch, samples)`` waveforms to ``(batch, 1, mel, frames)``.

    The default implementation is pure torch and deterministic on CPU. Set
    ``backend='torchaudio'`` to opt into torchaudio's MelSpectrogram transform.
    """
    if not isinstance(waveforms, torch.Tensor) or waveforms.ndim != 2:
        raise ValueError("waveforms must be a (batch, samples) torch tensor")
    if not waveforms.is_floating_point() or not torch.isfinite(waveforms).all():
        raise ValueError("waveforms must contain finite floating point samples")
    if sample_rate <= 0 or n_mels <= 0 or hop_length <= 0 or n_fft <= 0:
        raise ValueError("sample_rate, n_mels, hop_length, and n_fft must be positive")
    if waveforms.shape[-1] == 0:
        raise ValueError("waveforms must contain at least one sample")
    if backend == "torchaudio":
        try:
            import torchaudio
        except (ImportError, OSError) as exc:
            raise RuntimeError("torchaudio backend requested but torchaudio is unavailable") from exc
        transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate, n_fft=n_fft, hop_length=hop_length,
            n_mels=n_mels, center=True, pad_mode="constant", power=2.0,
        ).to(device=waveforms.device, dtype=waveforms.dtype)
        return torch.log(transform(waveforms).clamp_min(1e-10)).unsqueeze(1)
    if backend != "torch":
        raise ValueError("backend must be 'torch' or 'torchaudio'")

    # Keeping n_fft independent of the signal length supports tiny deterministic fixtures.
    window = torch.hann_window(n_fft, device=waveforms.device, dtype=waveforms.dtype)
    spectrum = torch.stft(
        waveforms, n_fft=n_fft, hop_length=hop_length, win_length=n_fft,
        window=window, center=True, pad_mode="constant", return_complex=True,
    ).abs().square()
    bank = _mel_filter(sample_rate, n_fft, n_mels, device=waveforms.device, dtype=waveforms.dtype)
    energies = torch.einsum("mf,bft->bmt", bank, spectrum)
    return torch.log(energies.clamp_min(1e-10)).unsqueeze(1)


class LogMelCNN(nn.Module):
    """LEGACY (rounds 1-4) baseline: small conv net accepting log-mel tensors from :func:`log_mel`.

    Retained unchanged as the experimental control. Its final ``AdaptiveAvgPool2d((1, 1))``
    collapses both the mel and time axes, so the classifier sees only 32 channel means and no
    temporal order at all -- a structural ceiling on a task whose confusable pairs (``no``/``on``,
    ``go``/``no``, ``up``/``off``) differ largely by phoneme order. The time-preserving models
    below exist to measure how much of the accuracy gap that collapse accounts for; this class
    stays as-is so the comparison has an unchanged reference point.
    """

    def __init__(self, n_mels: int = 40, num_classes: int = 12, width: int = 16) -> None:
        super().__init__()
        if n_mels <= 0 or num_classes <= 0 or width <= 0:
            raise ValueError("n_mels, num_classes, and width must be positive")
        self.n_mels = int(n_mels)
        self.num_classes = int(num_classes)
        self.features = nn.Sequential(
            nn.Conv2d(1, width, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(width), nn.ReLU(inplace=False), nn.MaxPool2d(2),
            nn.Conv2d(width, width * 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(width * 2), nn.ReLU(inplace=False),
            nn.Conv2d(width * 2, width * 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(width * 2), nn.ReLU(inplace=False),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Linear(width * 2, num_classes)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 4 or features.shape[1] != 1:
            raise ValueError("features must have shape (batch, 1, mel, frames)")
        if features.shape[2] != self.n_mels:
            raise ValueError(f"expected {self.n_mels} mel bins")
        return self.classifier(self.features(features).flatten(1))


class TimePoolCNN(nn.Module):
    """LogMelCNN's convolutions with the temporal collapse removed.

    Identical to :class:`LogMelCNN` except the final pool keeps ``time_slots`` temporal
    positions instead of one, so the classifier sees when energy occurred, not just how much.
    Deliberately minimal: it isolates the effect of the pooling change alone.
    """

    def __init__(self, n_mels: int = 40, num_classes: int = 12, width: int = 16, time_slots: int = 8) -> None:
        super().__init__()
        if n_mels <= 0 or num_classes <= 0 or width <= 0 or time_slots <= 0:
            raise ValueError("n_mels, num_classes, width, and time_slots must be positive")
        self.n_mels = int(n_mels)
        self.num_classes = int(num_classes)
        self.features = nn.Sequential(
            nn.Conv2d(1, width, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(width), nn.ReLU(inplace=False), nn.MaxPool2d(2),
            nn.Conv2d(width, width * 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(width * 2), nn.ReLU(inplace=False),
            nn.Conv2d(width * 2, width * 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(width * 2), nn.ReLU(inplace=False),
            nn.AdaptiveAvgPool2d((1, time_slots)),
        )
        self.classifier = nn.Linear(width * 2 * time_slots, num_classes)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 4 or features.shape[1] != 1:
            raise ValueError("features must have shape (batch, 1, mel, frames)")
        return self.classifier(self.features(features).flatten(1))


class WideCNN(nn.Module):
    """Wider and deeper than :class:`TimePoolCNN`, still time-preserving.

    Separates "more capacity" from "kept the time axis": compared against TimePoolCNN it
    measures what widening buys once the temporal bottleneck is already gone.
    """

    def __init__(self, n_mels: int = 40, num_classes: int = 12, width: int = 64, time_slots: int = 8) -> None:
        super().__init__()
        if n_mels <= 0 or num_classes <= 0 or width <= 0 or time_slots <= 0:
            raise ValueError("n_mels, num_classes, width, and time_slots must be positive")
        self.n_mels = int(n_mels)
        self.num_classes = int(num_classes)
        self.features = nn.Sequential(
            nn.Conv2d(1, width, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(width), nn.ReLU(inplace=False), nn.MaxPool2d(2),
            nn.Conv2d(width, width, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(width), nn.ReLU(inplace=False), nn.MaxPool2d(2),
            nn.Conv2d(width, width * 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(width * 2), nn.ReLU(inplace=False),
            nn.Conv2d(width * 2, width * 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(width * 2), nn.ReLU(inplace=False),
            nn.AdaptiveAvgPool2d((1, time_slots)),
        )
        self.classifier = nn.Linear(width * 2 * time_slots, num_classes)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 4 or features.shape[1] != 1:
            raise ValueError("features must have shape (batch, 1, mel, frames)")
        return self.classifier(self.features(features).flatten(1))


class ConvGRU(nn.Module):
    """Convolutions over mel, then a bidirectional GRU over time: an explicit sequence model.

    Pools only the mel axis, so the time axis survives into the recurrence. This is the
    variant that can in principle represent phoneme order rather than a bag of local patterns.
    """

    def __init__(self, n_mels: int = 40, num_classes: int = 12, width: int = 32, hidden: int = 64) -> None:
        super().__init__()
        if n_mels <= 0 or num_classes <= 0 or width <= 0 or hidden <= 0:
            raise ValueError("n_mels, num_classes, width, and hidden must be positive")
        if n_mels % 4 != 0:
            raise ValueError("n_mels must be divisible by 4 (two mel-axis pools of stride 2)")
        self.n_mels = int(n_mels)
        self.num_classes = int(num_classes)
        self.features = nn.Sequential(
            nn.Conv2d(1, width, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(width), nn.ReLU(inplace=False), nn.MaxPool2d((2, 1)),
            nn.Conv2d(width, width * 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(width * 2), nn.ReLU(inplace=False), nn.MaxPool2d((2, 1)),
        )
        self.gru = nn.GRU(width * 2 * (n_mels // 4), hidden, batch_first=True, bidirectional=True)
        self.classifier = nn.Linear(hidden * 2, num_classes)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 4 or features.shape[1] != 1:
            raise ValueError("features must have shape (batch, 1, mel, frames)")
        conv = self.features(features)
        batch, channels, mel, frames = conv.shape
        sequence = conv.permute(0, 3, 1, 2).reshape(batch, frames, channels * mel)
        output, _ = self.gru(sequence)
        return self.classifier(output.mean(dim=1))
