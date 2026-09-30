"""Training functions for acoustic-robust speech command classification."""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.nn.utils import clip_grad_norm_
from torch.optim import Adam

from .audio import add_white_noise, apply_gain, apply_rir_reverb
from .diagnostics import cluster_drift_metrics, sentrycam_alert_epoch
from .experiment import ExperimentConfig


@dataclass(frozen=True)
class TrainResult:
    """Result of training a classifier."""
    model: nn.Module
    history: list[dict]
    best_epoch: int
    epochs_run: int
    optimizer_steps: int
    wall_seconds: float
    grad_norm_history: list[float]
    clip_threshold_history: list[float]
    inter_cluster_distance_history: list[float]
    intra_cluster_variance_history: list[float]
    instability_alert_epoch: int | None
    probe_history: list[dict]


def spec_mask(
    x: np.ndarray,
    *,
    freq_mask: int,
    time_mask: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Mask contiguous frequency and time bands with the array mean.

    Args:
        x: (n_mels, frames) log-mel spectrogram.
        freq_mask: Number of contiguous frequency bins to mask.
        time_mask: Number of contiguous time frames to mask.
        rng: Random generator for deterministic masking positions.

    Returns:
        Masked copy of x with same dtype.

    Raises:
        ValueError: If freq_mask > n_mels or time_mask > frames.
    """
    x = np.asarray(x)
    if freq_mask < 0 or time_mask < 0:
        raise ValueError("freq_mask and time_mask must be non-negative")
    if freq_mask > x.shape[0]:
        raise ValueError(f"freq_mask {freq_mask} exceeds n_mels {x.shape[0]}")
    if time_mask > x.shape[1]:
        raise ValueError(f"time_mask {time_mask} exceeds frames {x.shape[1]}")

    out = x.copy()
    fill_value = float(x.mean(dtype=np.float64))

    # Mask frequency band
    if freq_mask > 0:
        max_start = x.shape[0] - freq_mask
        freq_start = rng.integers(0, max_start + 1)
        out[freq_start:freq_start + freq_mask, :] = fill_value

    # Mask time band
    if time_mask > 0:
        max_start = x.shape[1] - time_mask
        time_start = rng.integers(0, max_start + 1)
        out[:, time_start:time_start + time_mask] = fill_value

    return out.astype(x.dtype)


def apply_condition(
    waveforms: np.ndarray,
    condition: str,
    rng: np.random.Generator,
) -> np.ndarray:
    """Apply data augmentation based on condition.

    Args:
        waveforms: (batch, samples) float32 waveforms.
        condition: One of "clean", "masking", "acoustic".
        rng: Random generator for seeding augmentations.

    Returns:
        Augmented waveforms as (batch, samples) float32, clipped to [-1, 1].

    Raises:
        ValueError: If condition is unknown.
    """
    waveforms = np.asarray(waveforms, dtype=np.float32)

    if condition == "clean":
        return waveforms.copy()
    elif condition == "masking":
        # Masking happens in feature space, not waveform space
        return waveforms.copy()
    elif condition == "acoustic":
        out = np.zeros_like(waveforms)
        sample_rate = 16_000

        for i in range(waveforms.shape[0]):
            w = waveforms[i]
            seed = int(rng.integers(0, 2**31 - 1))
            inner_rng = np.random.default_rng(seed)

            # Random choice: gain, white noise, or reverb
            choice = inner_rng.integers(0, 3)

            if choice == 0:  # Gain
                gain_db = inner_rng.uniform(-12.0, 12.0)
                w, _ = apply_gain(w, gain_db=gain_db)
            elif choice == 1:  # White noise
                snr_db = inner_rng.uniform(5.0, 20.0)
                noise_seed = int(inner_rng.integers(0, 2**31 - 1))
                w, _ = add_white_noise(w, sample_rate=sample_rate, snr_db=snr_db, seed=noise_seed)
            else:  # Reverb
                rt60_seconds = inner_rng.uniform(0.1, 0.6)
                drr_db = inner_rng.uniform(-5.0, 5.0)
                reverb_seed = int(inner_rng.integers(0, 2**31 - 1))
                w, _ = apply_rir_reverb(w, sample_rate=sample_rate, rt60_seconds=rt60_seconds, drr_db=drr_db, seed=reverb_seed)

            # Clip to [-1, 1]
            out[i] = np.clip(w, -1.0, 1.0).astype(np.float32)

        return out
    else:
        raise ValueError(f"unknown condition {condition!r}")


def predict_logits(
    model: nn.Module,
    x: np.ndarray,
    batch_size: int = 256,
    feature_fn=None,
) -> np.ndarray:
    """Generate logits from a model in eval mode without gradients.

    Args:
        model: PyTorch model.
        x: Input array (n_samples, ...).
        batch_size: Batch size for inference.
        feature_fn: Optional function to apply to input before model.

    Returns:
        (n_samples, n_classes) logits as numpy array.
    """
    training_mode = model.training
    logits_list = []

    try:
        model.eval()
        device = next(model.parameters()).device
        with torch.no_grad():
            for i in range(0, len(x), batch_size):
                batch = x[i:i + batch_size]
                batch_tensor = torch.from_numpy(np.asarray(batch, dtype=np.float32)).to(device)

                if feature_fn is not None:
                    batch_tensor = feature_fn(batch_tensor)

                batch_logits = model(batch_tensor)
                logits_list.append(batch_logits.cpu().numpy())
    finally:
        model.train(training_mode)

    return np.vstack(logits_list).astype(np.float32)


def train_classifier(
    model: nn.Module,
    train_x: np.ndarray,
    train_y: np.ndarray,
    val_x: np.ndarray,
    val_y: np.ndarray,
    config: ExperimentConfig,
    augment_fn=None,
    progress: bool = False,
    probe_fn=None,
) -> TrainResult:
    """Train a classifier with gradient accumulation and early stopping.

    Args:
        model: PyTorch model to train.
        train_x: (n_train, n_features) training features.
        train_y: (n_train,) training labels.
        val_x: (n_val, n_features) validation features.
        val_y: (n_val,) validation labels.
        config: ExperimentConfig with optimization settings.
        augment_fn: Optional augmentation function for training batches.
        progress: print a per-epoch progress line. Off by default so the test suite stays quiet;
            long offloaded runs turn it on so the job is never silent for minutes at a time.
        probe_fn: optional ``probe_fn(model, epoch) -> dict`` called once per epoch inside the
            no-grad evaluation block. Whatever it returns is appended to ``TrainResult.probe_history``.
            Kept deliberately generic: paper-specific probes (representation drift, activation
            tensors) live in ``src.diagnostics`` and are selected by the caller, so this loop never
            grows architecture- or methodology-specific code.

    Returns:
        TrainResult with trained model, history, and metadata.
    """
    start_time = time.time()

    # Copy model to avoid mutating input
    model = copy.deepcopy(model)
    device = next(model.parameters()).device

    # Deterministic training
    torch.manual_seed(config.seed)

    # Setup optimizer and loss
    optimizer = Adam(model.parameters(), lr=1e-3, weight_decay=0)
    loss_fn = torch.nn.CrossEntropyLoss()

    # Convert data to tensors
    train_x_tensor = torch.from_numpy(np.asarray(train_x, dtype=np.float32)).to(device)
    train_y_tensor = torch.from_numpy(np.asarray(train_y, dtype=np.int64)).to(device)
    val_x_tensor = torch.from_numpy(np.asarray(val_x, dtype=np.float32)).to(device)
    val_y_tensor = torch.from_numpy(np.asarray(val_y, dtype=np.int64)).to(device)

    model.train()
    history = []
    best_epoch = 0
    best_val_loss = float("inf")
    best_model_state = None
    optimizer_steps = 0
    patience_counter = 0
    grad_norm_history = []
    clip_threshold_history = []
    inter_cluster_distance_history = []
    intra_cluster_variance_history = []
    probe_history = []

    # Create a shuffler generator
    shuffler_rng = torch.Generator()
    shuffler_rng.manual_seed(config.seed)

    for epoch in range(config.max_epochs):
        # Shuffle training data
        indices = torch.randperm(len(train_x_tensor), generator=shuffler_rng)

        epoch_loss = 0.0
        num_batches = 0
        optimizer.zero_grad()

        # Process microbatches with gradient accumulation
        for batch_idx in range(0, len(indices), config.microbatch_size):
            batch_indices = indices[batch_idx:batch_idx + config.microbatch_size].to(device)
            x_batch = train_x_tensor[batch_indices]
            y_batch = train_y_tensor[batch_indices]

            # Apply augmentation if provided
            if augment_fn is not None:
                x_batch_before = x_batch.detach().cpu().numpy()
                augmented_np = augment_fn(x_batch_before, config.condition, np.random.default_rng(config.seed + epoch * 1000 + batch_idx))
                x_batch = torch.from_numpy(np.asarray(augmented_np, dtype=np.float32)).to(device)

            # Forward pass
            logits = model(x_batch)
            loss = loss_fn(logits, y_batch)
            # Report the true cross-entropy, not the accumulation-scaled one: `loss` below is
            # divided by accumulation_steps so gradients accumulate correctly, but history must
            # stay on the same scale as val_loss to be comparable.
            raw_loss = loss.detach()

            # Normalize loss by accumulation steps
            loss = loss / config.accumulation_steps
            loss.backward()

            epoch_loss += raw_loss
            num_batches += 1

            # Optimizer step every accumulation_steps
            if num_batches % config.accumulation_steps == 0:
                threshold = float(np.percentile(grad_norm_history, config.autoclip_percentile)) if grad_norm_history else float("inf")
                pre_clip_norm = float(clip_grad_norm_(model.parameters(), max_norm=threshold))
                grad_norm_history.append(pre_clip_norm)
                clip_threshold_history.append(threshold)
                optimizer.step()
                optimizer.zero_grad()
                optimizer_steps += 1

        # Final step for remaining accumulated gradients
        if num_batches > 0 and num_batches % config.accumulation_steps != 0:
            threshold = float(np.percentile(grad_norm_history, config.autoclip_percentile)) if grad_norm_history else float("inf")
            pre_clip_norm = float(clip_grad_norm_(model.parameters(), max_norm=threshold))
            grad_norm_history.append(pre_clip_norm)
            clip_threshold_history.append(threshold)
            optimizer.step()
            optimizer.zero_grad()
            optimizer_steps += 1

        # Validation
        model.eval()
        with torch.no_grad():
            val_logits = model(val_x_tensor)
            val_loss = float(loss_fn(val_logits, val_y_tensor))
            val_pred = torch.argmax(val_logits, dim=1)
            val_accuracy = float((val_pred == val_y_tensor).float().mean())
            inter_dist, intra_var = cluster_drift_metrics(val_logits.detach().cpu().numpy(), val_y_tensor.cpu().numpy())
            inter_cluster_distance_history.append(inter_dist)
            intra_cluster_variance_history.append(intra_var)
            if probe_fn is not None:
                probe_history.append(probe_fn(model, epoch))
        model.train()

        # Record history
        history.append({
            "epoch": epoch,
            "train_loss": float(epoch_loss / num_batches) if num_batches > 0 else 0.0,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy,
        })
        if progress:
            print(f"  epoch {epoch + 1}/{config.max_epochs} done: "
                  f"train_loss={history[-1]['train_loss']:.4f} "
                  f"val_loss={val_loss:.4f} val_accuracy={val_accuracy:.4f}", flush=True)

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_model_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= config.patience:
            break

    # Restore best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    wall_seconds = time.time() - start_time
    instability_alert_epoch = sentrycam_alert_epoch(inter_cluster_distance_history, intra_cluster_variance_history)

    return TrainResult(
        model=model,
        history=history,
        best_epoch=best_epoch,
        epochs_run=len(history),
        optimizer_steps=optimizer_steps,
        wall_seconds=wall_seconds,
        grad_norm_history=grad_norm_history,
        clip_threshold_history=clip_threshold_history,
        inter_cluster_distance_history=inter_cluster_distance_history,
        intra_cluster_variance_history=intra_cluster_variance_history,
        instability_alert_epoch=instability_alert_epoch,
        probe_history=probe_history,
    )
