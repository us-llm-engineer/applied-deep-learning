"""Shared builders for the pipeline contract tests (one copy; never duplicate in test files)."""
from __future__ import annotations

import zlib
from pathlib import Path

import numpy as np
from scipy.io import wavfile

from src.dataset import AUXILIARY_WORDS, STUDY_LABELS

COMMANDS = STUDY_LABELS[:10]
CLIP = 16_000


def _rng(*parts: object) -> np.random.Generator:
    return np.random.default_rng(zlib.crc32("|".join(map(str, parts)).encode()))


def write_wav(path: Path, samples: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    wavfile.write(str(path), CLIP, np.asarray(samples, dtype=np.int16))


def clip_samples(word: str, index: int, length: int = CLIP) -> np.ndarray:
    """Unique seeded noise per (word, index): no accidental duplicate checksums."""
    return np.clip(_rng(word, index).normal(0, 3000, length), -32768, 32767).astype(np.int16)


def noise_samples(index: int, length: int) -> np.ndarray:
    return np.clip(_rng("noise", index).normal(0, 2000, length), -32768, 32767).astype(np.int16)


def build_dataset(root: Path, *, command_clips: int = 30, aux_clips: int = 6,
                  noise_files: int = 4, noise_len: int = 160_000,
                  clips_by_word: dict[str, int] | None = None) -> Path:
    """Write dataset_root/<word>/<word>_NNN.wav for 35 words plus _background_noise_."""
    counts = {w: command_clips for w in COMMANDS} | {w: aux_clips for w in AUXILIARY_WORDS}
    counts.update(clips_by_word or {})
    for word, n in counts.items():
        for i in range(n):
            write_wav(root / word / f"{word}_{i:03d}.wav", clip_samples(word, i))
    for k in range(noise_files):
        write_wav(root / "_background_noise_" / f"noise_{k}.wav", noise_samples(k, noise_len))
    return root


def noise_arrays(n_files: int = 4, length: int = 160_000) -> list[np.ndarray]:
    return [noise_samples(k, length) for k in range(n_files)]
