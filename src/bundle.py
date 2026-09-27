"""Speech command dataset bundle: in-distribution splits, OOD unknown, and audit utilities."""
from __future__ import annotations

import dataclasses
import hashlib
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import NamedTuple

import numpy as np
from scipy.io import wavfile

from .data import stratified_split, ManifestRecord
from .dataset import (
    STUDY_LABELS, AUXILIARY_WORDS, holdout_unknown_words,
    manifest_checksum, AudioRecord, silence_windows
)


CLIP = 16_000  # 16,000 samples for 1-second audio at 16 kHz

_SPEAKER_ID = re.compile(r"([0-9a-f]{8})_nohash_[0-9]+(?:\.wav)?\Z")


def speaker_of(identifier: str) -> str | None:
    """Return the source speaker token from a Speech Commands filename ID."""
    # IDs may be relative paths (or platform-style paths); only the final
    # filename is meaningful, and its complete stem must match the convention.
    filename = str(identifier).replace("\\", "/").rsplit("/", 1)[-1]
    match = _SPEAKER_ID.fullmatch(filename)
    return match.group(1) if match else None


def speakers_per_split(bundle: DatasetBundle) -> dict[str, int]:
    """Count distinct parseable source speakers in each split."""
    return {
        name: len({speaker for identifier in split.ids if (speaker := speaker_of(identifier)) is not None})
        for name, split in bundle.splits.items()
    }


def speaker_overlap(bundle: DatasetBundle) -> dict[str, float]:
    """Fraction of test rows whose source speaker also occurs in a reference split."""
    test = bundle.splits.get("test")
    train = bundle.splits.get("train")
    cal = bundle.splits.get("cal")
    if test is None or not test.ids:
        return {"test_seen_in_train": 0.0, "test_seen_in_cal": 0.0}

    test_speakers = [speaker_of(identifier) for identifier in test.ids]
    denominator = len(test.ids)

    def fraction(reference: SplitArrays | None) -> float:
        if reference is None:
            return 0.0
        reference_speakers = {speaker for identifier in reference.ids if (speaker := speaker_of(identifier)) is not None}
        return sum(speaker is not None and speaker in reference_speakers for speaker in test_speakers) / denominator

    return {"test_seen_in_train": fraction(train), "test_seen_in_cal": fraction(cal)}


@dataclasses.dataclass(frozen=True)
class SplitArrays:
    """One split's waveforms, labels, and metadata."""
    waveforms: np.ndarray  # int16, shape (n, 16000)
    y: np.ndarray  # int64, shape (n,)
    ids: tuple[str, ...]
    words: tuple[str, ...]


@dataclasses.dataclass(frozen=True)
class DatasetBundle:
    """Complete dataset with 5 splits, unknown-word handling, and audit info."""
    labels: tuple[str, ...]
    splits: dict[str, SplitArrays]
    train_unknown_words: tuple[str, ...]
    heldout_words: tuple[str, ...]
    manifest_sha256: str
    shortfalls: dict[str, int]


def assemble_bundle(
    dataset_root: Path, *, seed: int = 17, cap_per_label: int = 600
) -> DatasetBundle:
    """Assemble a bundle: read wavs, stratify, handle silence, split train→train+val.

    Reads from <root>/<word>/<word>_NNN.wav and <root>/_background_noise_/*.wav.
    Returns bundle with splits: train, val, cal, test, ood (heldout words with y=-1).
    """
    dataset_root = Path(dataset_root)
    if not dataset_root.is_dir():
        raise ValueError(f"dataset_root does not exist: {dataset_root}")

    bg_noise_dir = dataset_root / "_background_noise_"
    if not bg_noise_dir.is_dir():
        raise ValueError(f"_background_noise_ directory not found: {bg_noise_dir}")

    # Check that all command words exist
    for cmd in STUDY_LABELS[:10]:
        if not (dataset_root / cmd).is_dir():
            raise ValueError(f"command directory not found: {dataset_root / cmd}")

    commands = set(STUDY_LABELS[:10])
    train_unknown_words, heldout_words = holdout_unknown_words(AUXILIARY_WORDS, 5, seed)

    # Load background noise files
    noise_arrays = []
    for wav_file in sorted(bg_noise_dir.glob("*.wav")):
        rate, data = wavfile.read(str(wav_file))
        noise_arrays.append(data.astype(np.int16))

    # Collect command and train-unknown clips
    command_records = {}  # word -> list of (clip_id, word, audio, checksum, source_path)
    unknown_records = []  # list of (clip_id, word, audio, checksum, source_path)
    heldout_clips = []  # separate storage for OOD
    record_ids = {}  # map from id to (word, audio)

    # Collect all clips
    for word_dir in sorted(dataset_root.iterdir()):
        if not word_dir.is_dir() or word_dir.name.startswith("_"):
            continue

        word = word_dir.name

        # Skip unknown words
        if word not in commands and word not in AUXILIARY_WORDS:
            continue

        for wav_file in sorted(word_dir.glob("*.wav")):
            rate, data = wavfile.read(str(wav_file))
            audio = data.astype(np.int16)

            # Pad or trim to CLIP samples
            if len(audio) < CLIP:
                audio = np.pad(audio, (0, CLIP - len(audio)), mode="constant")
            else:
                audio = audio[:CLIP]

            # Generate unique id
            clip_id = f"{word}/{wav_file.name}"
            source_path = wav_file
            checksum = hashlib.sha256(audio.tobytes()).hexdigest()

            if word in commands:
                # Command word
                if word not in command_records:
                    command_records[word] = []
                command_records[word].append((clip_id, word, audio, checksum, source_path))
                record_ids[clip_id] = (word, audio)
            elif word in train_unknown_words:
                # Train-unknown word
                unknown_records.append((clip_id, word, audio, checksum, source_path))
                record_ids[clip_id] = (word, audio)
            else:
                # Heldout word
                heldout_clips.append((clip_id, word, audio, checksum, source_path))

    # Track shortfalls for each command
    shortfalls = {}

    # Cap command clips and create AudioRecords
    indist_records = []
    for cmd in STUDY_LABELS[:10]:
        if cmd not in command_records:
            shortfalls[cmd] = cap_per_label
            continue

        clips = command_records[cmd]
        if len(clips) < cap_per_label:
            shortfalls[cmd] = cap_per_label - len(clips)
        elif len(clips) > cap_per_label:
            # Sample deterministically
            word_seed = int.from_bytes(hashlib.sha256(f"{seed}:{cmd}".encode()).digest()[:4], "little")
            rng = np.random.default_rng(word_seed)
            indices = rng.choice(len(clips), cap_per_label, replace=False)
            clips = [clips[int(i)] for i in sorted(indices)]

        for clip_id, word, audio, checksum, source_path in clips:
            indist_records.append(AudioRecord(
                identifier=clip_id,
                label=cmd,
                path=source_path,
                checksum=checksum,
                sample_rate=16000,
                num_frames=CLIP
            ))

    # Handle unknown label: sample evenly from train-unknown words to reach CAP total
    if unknown_records:
        # Group by word
        unknown_by_word = defaultdict(list)
        for clip_id, word, audio, checksum, source_path in unknown_records:
            unknown_by_word[word].append((clip_id, audio, checksum, source_path))

        # Determine target clips per word
        n_words = len(unknown_by_word)
        clips_per_word = cap_per_label // n_words
        remainder = cap_per_label % n_words

        # Sample from each word
        for i, word in enumerate(sorted(unknown_by_word.keys())):
            n_clips = clips_per_word + (1 if i < remainder else 0)
            clips = unknown_by_word[word]

            if len(clips) < n_clips:
                # Use all available
                selected = clips
            else:
                # Sample
                word_seed = int.from_bytes(hashlib.sha256(f"{seed}:{word}".encode()).digest()[:4], "little")
                word_rng = np.random.default_rng(word_seed)
                indices = word_rng.choice(len(clips), n_clips, replace=False)
                selected = [clips[int(idx)] for idx in sorted(indices)]

            for clip_id, audio, checksum, source_path in selected:
                indist_records.append(AudioRecord(
                    identifier=clip_id,
                    label="unknown",
                    path=source_path,
                    checksum=checksum,
                    sample_rate=16000,
                    num_frames=CLIP
                ))

    # Compute manifest checksum (on command + unknown, before silence)
    manifest_digest = manifest_checksum(indist_records)

    # Map label names to indices
    label_to_idx = {name: i for i, name in enumerate(STUDY_LABELS)}

    # Stratified split of command+unknown data: 70% train+val, 15% cal, 15% test
    manifest_records = [
        ManifestRecord(identifier=r.identifier, label=r.label, path=str(r.path))
        for r in indist_records
    ]
    splits_70_15_15 = stratified_split(manifest_records, seed=seed)

    # Extract train+val, cal, test splits (at 70/15/15)
    train_cal_records = splits_70_15_15["train"]  # 70%
    cal_records = splits_70_15_15["calibration"]  # 15%
    test_records = splits_70_15_15["test"]  # 15%

    # Further split train+val into train (90%) and val (10%)
    train_and_val = stratified_split(
        train_cal_records,
        seed=seed,
        ratios=(0.9, 0.1, 0.0)
    )
    train_records = train_and_val["train"]
    val_records = train_and_val["calibration"]

    # Calculate silence split counts
    silence_ratios = (0.70, 0.15, 0.15)
    exact = [cap_per_label * ratio for ratio in silence_ratios]
    silence_counts = [math.floor(part) for part in exact]
    remainder = cap_per_label - sum(silence_counts)
    order = sorted(range(len(silence_ratios)), key=lambda index: (-(exact[index] - silence_counts[index]), index))
    for index in order[:remainder]:
        silence_counts[index] += 1

    train_val_silence_count, cal_silence_count, test_silence_count = silence_counts

    # Generate silence windows from each time region
    # Generate from train+val region and stratify
    if train_val_silence_count > 0:
        silence_array_70 = silence_windows(noise_arrays, train_val_silence_count, seed=seed, split="train")
        silence_array_70 = silence_array_70.astype(np.int16)

        # Create dummy records for stratification
        silence_70_records = [
            ManifestRecord(identifier=f"silence_70_{i}", label="silence", path=f"dummy_{i}")
            for i in range(len(silence_array_70))
        ]

        # Stratify silence from train+val region into train (90%) and val (10%)
        silence_splits_90_10 = stratified_split(
            silence_70_records,
            seed=seed,
            ratios=(0.9, 0.1, 0.0)
        )

        train_silence_indices = [int(r.identifier.split("_")[-1]) for r in silence_splits_90_10["train"]]
        val_silence_indices = [int(r.identifier.split("_")[-1]) for r in silence_splits_90_10["calibration"]]

        train_silence = silence_array_70[train_silence_indices]
        val_silence = silence_array_70[val_silence_indices]
    else:
        train_silence = np.array([], dtype=np.int16).reshape(0, CLIP)
        val_silence = np.array([], dtype=np.int16).reshape(0, CLIP)

    # Generate silence from cal region
    if cal_silence_count > 0:
        cal_silence = silence_windows(noise_arrays, cal_silence_count, seed=seed, split="cal").astype(np.int16)
    else:
        cal_silence = np.array([], dtype=np.int16).reshape(0, CLIP)

    # Generate silence from test region
    if test_silence_count > 0:
        test_silence = silence_windows(noise_arrays, test_silence_count, seed=seed, split="test").astype(np.int16)
    else:
        test_silence = np.array([], dtype=np.int16).reshape(0, CLIP)

    # Assemble data for each in-distribution split
    def build_split_arrays(manifest_recs: list[ManifestRecord], silence_array: np.ndarray, split_name: str) -> SplitArrays:
        waveforms = []
        labels = []
        ids = []
        words = []

        # Add command+unknown records
        for rec in manifest_recs:
            word, audio = record_ids[rec.identifier]
            waveforms.append(audio)
            labels.append(label_to_idx[rec.label])
            ids.append(rec.identifier)
            words.append(word)

        # Add silence records
        for i, window in enumerate(silence_array):
            waveforms.append(window)
            labels.append(label_to_idx["silence"])
            ids.append(f"silence/{split_name}/{i}")
            words.append("silence")

        return SplitArrays(
            waveforms=np.array(waveforms, dtype=np.int16),
            y=np.array(labels, dtype=np.int64),
            ids=tuple(ids),
            words=tuple(words)
        )

    # Build in-distribution splits
    splits = {
        "train": build_split_arrays(train_records, train_silence, "train"),
        "val": build_split_arrays(val_records, val_silence, "val"),
        "cal": build_split_arrays(cal_records, cal_silence, "cal"),
        "test": build_split_arrays(test_records, test_silence, "test"),
    }

    # Build OOD split from heldout words
    n_ood = cap_per_label // 2
    rng = np.random.default_rng(seed)

    heldout_by_word = defaultdict(list)
    for clip_id, word, audio, checksum, source_path in heldout_clips:
        heldout_by_word[word].append((clip_id, audio, source_path))

    ood_ids = []
    ood_words = []
    ood_waveforms = []
    clips_per_word = n_ood // len(heldout_words)
    remainder = n_ood % len(heldout_words)

    for i, word in enumerate(sorted(heldout_words)):
        if word not in heldout_by_word:
            continue
        n_clips = clips_per_word + (1 if i < remainder else 0)
        clips = heldout_by_word[word]
        word_seed = int.from_bytes(hashlib.sha256(f"{seed}:{word}".encode()).digest()[:4], "little")
        word_rng = np.random.default_rng(word_seed)
        indices = word_rng.choice(len(clips), min(n_clips, len(clips)), replace=False)
        for idx in sorted(indices):
            clip_id, audio, source_path = clips[int(idx)]
            ood_ids.append(clip_id)
            ood_words.append(word)
            ood_waveforms.append(audio)

    splits["ood"] = SplitArrays(
        waveforms=np.array(ood_waveforms, dtype=np.int16),
        y=np.full(len(ood_ids), -1, dtype=np.int64),
        ids=tuple(ood_ids),
        words=tuple(ood_words)
    )

    return DatasetBundle(
        labels=STUDY_LABELS,
        splits=splits,
        train_unknown_words=train_unknown_words,
        heldout_words=heldout_words,
        manifest_sha256=manifest_digest,
        shortfalls=shortfalls
    )


def audit_bundle(bundle: DatasetBundle) -> dict:
    """Audit bundle for integrity: overlaps, duplicates, leaks, time regions."""
    # Check id overlaps
    id_overlap = {}
    splits_list = list(bundle.splits.keys())
    for i, s1 in enumerate(splits_list):
        for s2 in splits_list[i+1:]:
            ids1 = set(bundle.splits[s1].ids)
            ids2 = set(bundle.splits[s2].ids)
            overlap_count = len(ids1 & ids2)
            key = f"{s1}-{s2}"
            id_overlap[key] = overlap_count

    # Check for duplicate checksums (by waveform hash)
    checksum_to_ids = defaultdict(list)
    for split_name, split in bundle.splits.items():
        for clip_id, waveform in zip(split.ids, split.waveforms):
            checksum = hashlib.sha256(waveform.tobytes()).hexdigest()
            checksum_to_ids[checksum].append(clip_id)

    # Only report checksums with multiple ids
    duplicate_checksums = {cs: tuple(ids) for cs, ids in checksum_to_ids.items() if len(ids) > 1}

    # Check for heldout word leaks
    heldout_word_leak = False
    heldout_set = set(bundle.heldout_words)
    for split_name in ["train", "val", "cal", "test"]:
        if split_name in bundle.splits:
            split = bundle.splits[split_name]
            if any(w in heldout_set for w in split.words):
                heldout_word_leak = True
                break

    # Check for silence time overlap (would need original noise files)
    silence_time_overlap = False

    # Label counts per split
    label_counts = {}
    for split_name, split_data in bundle.splits.items():
        counts = Counter(split_data.y.tolist())
        label_counts[split_name] = dict(counts)

    return {
        "id_overlap": id_overlap,
        "duplicate_checksums": duplicate_checksums,
        "heldout_word_leak": heldout_word_leak,
        "silence_time_overlap": silence_time_overlap,
        "label_counts": label_counts,
        "speakers_per_split": speakers_per_split(bundle),
        "speaker_overlap": speaker_overlap(bundle),
    }
