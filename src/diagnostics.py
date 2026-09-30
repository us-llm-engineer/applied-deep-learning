"""SentryCam-inspired instability signal and cluster drift metrics (arXiv:2405.15135)."""

from __future__ import annotations

import numpy as np


def cluster_drift_metrics(logits: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    """Returns (inter_cluster_distance, intra_cluster_variance) from per-class centroids in logit space.

    Args:
        logits: (n, classes) float array of logits per sample.
        labels: (n,) int array of class labels, same length as logits' first dimension.

    Returns:
        (inter_cluster_distance, intra_cluster_variance):
        - inter_cluster_distance: mean pairwise Euclidean distance between distinct present class centroids.
          Returns 0.0 if fewer than 2 distinct classes are present.
        - intra_cluster_variance: unweighted mean across classes of each class's intra-class variance
          (mean squared Euclidean distance from rows to their class centroid).

    Raises:
        ValueError: if logits.ndim != 2, labels.ndim != 1, logits.shape[0] != labels.shape[0],
                    or logits.shape[0] == 0.
    """
    logits = np.asarray(logits)
    labels = np.asarray(labels)

    if logits.ndim != 2:
        raise ValueError(f"logits must be 2-dimensional, got {logits.ndim}D")
    if labels.ndim != 1:
        raise ValueError(f"labels must be 1-dimensional, got {labels.ndim}D")
    if logits.shape[0] != labels.shape[0]:
        raise ValueError(f"logits and labels must have the same first dimension, got {logits.shape[0]} and {labels.shape[0]}")
    if logits.shape[0] == 0:
        raise ValueError("logits and labels must be non-empty")

    # Get unique classes present in labels
    unique_classes = np.unique(labels)
    n_classes = len(unique_classes)

    # Compute per-class centroids and intra-class variances
    centroids = []
    intra_variances = []

    for class_idx in unique_classes:
        class_mask = labels == class_idx
        class_logits = logits[class_mask]
        centroid = class_logits.mean(axis=0)
        centroids.append(centroid)

        # Compute intra-class variance: mean squared Euclidean distance to centroid
        distances = np.linalg.norm(class_logits - centroid, axis=1)
        intra_var = np.mean(distances ** 2)
        intra_variances.append(intra_var)

    # Compute intra_cluster_variance as unweighted mean across classes
    intra_cluster_variance = float(np.mean(intra_variances))

    # Compute inter_cluster_distance as mean pairwise Euclidean distance between centroids
    if n_classes < 2:
        inter_cluster_distance = 0.0
    else:
        centroids = np.array(centroids)
        pairwise_distances = []
        for i in range(n_classes):
            for j in range(i + 1, n_classes):
                dist = np.linalg.norm(centroids[i] - centroids[j])
                pairwise_distances.append(dist)
        inter_cluster_distance = float(np.mean(pairwise_distances))

    return inter_cluster_distance, intra_cluster_variance


def sentrycam_alert_epoch(
    distance_history,
    variance_history,
    *,
    k: int = 2,
    alpha: float = 0.25,
    window: int = 10,
) -> int | None:
    """Earliest epoch index satisfying the SentryCam-style 2-condition instability alert, or None.

    Returns the smallest epoch index t such that EITHER the distance or variance condition holds.
    - **Distance condition** at t (requires t >= k): sustained decrease for k consecutive steps,
      plus a significance check that abs(distance_history[t] - distance_history[t-1]) > alpha * sigma.
    - **Variance condition** at t: sustained increase for k consecutive steps,
      plus the same significance check for variance.

    The significance check uses a rolling window of epochs strictly BEFORE t (excluding t itself).
    If the window has fewer than 2 distinct values or sigma == 0, the condition does not hold at t.

    Args:
        distance_history: Sequence of distances, one per epoch (index 0 = first epoch).
        variance_history: Sequence of variances, same length as distance_history.
        k: Minimum consecutive steps required for sustained direction (default 2).
        alpha: Multiplier for significance threshold (default 0.25).
        window: Number of epochs to look back for sigma calculation, excluding t itself (default 10).

    Returns:
        int: earliest epoch index where either condition holds.
        None: if neither condition ever holds.

    Raises:
        ValueError: if lengths of distance_history and variance_history differ,
                    or either is empty.
    """
    distance_history = list(distance_history)
    variance_history = list(variance_history)

    if len(distance_history) != len(variance_history):
        raise ValueError(f"distance_history and variance_history must have the same length, got {len(distance_history)} and {len(variance_history)}")
    if len(distance_history) == 0:
        raise ValueError("distance_history and variance_history must be non-empty")

    n = len(distance_history)

    def check_condition(history, direction: str) -> int | None:
        """Check for sustained direction with significance.
        direction: 'decrease' or 'increase'
        Returns: earliest epoch index where condition holds, or None.
        """
        for t in range(k, n):
            # Check sustained direction: k consecutive steps ending at t
            valid_direction = True
            if direction == "decrease":
                for i in range(t - k, t):
                    if not (history[i] > history[i + 1]):
                        valid_direction = False
                        break
            else:  # increase
                for i in range(t - k, t):
                    if not (history[i] < history[i + 1]):
                        valid_direction = False
                        break

            if not valid_direction:
                continue

            # Check significance
            lo = max(0, t - window)
            window_values = history[lo:t]  # strictly BEFORE t, excluding t itself

            if len(window_values) < 2:
                # Degenerate window: fewer than 2 points
                continue

            sigma = float(np.std(window_values))
            if sigma == 0:
                # No variance in the window: cannot trigger
                continue

            # Check if abs(history[t] - history[t-1]) > alpha * sigma
            if abs(history[t] - history[t - 1]) > alpha * sigma:
                return t

        return None

    # Check both conditions
    distance_epoch = check_condition(distance_history, "decrease")
    variance_epoch = check_condition(variance_history, "increase")

    # Return the earlier one, or None if neither fires
    if distance_epoch is None and variance_epoch is None:
        return None
    elif distance_epoch is None:
        return variance_epoch
    elif variance_epoch is None:
        return distance_epoch
    else:
        return min(distance_epoch, variance_epoch)


# --------------------------------------------------------------------------- in-training probes
#
# SentryCam (arXiv:2405.15135) and MM-PHATE (arXiv:2406.01969) specifics live here, not in
# src/train.py, so the training loop stays free of methodology-specific code. train_classifier
# only knows "call probe_fn(model, epoch), store what it returns".


def penultimate_and_sequence(model, x):
    """Return (penultimate_vector, sequence_activations) for one probe batch.

    penultimate_vector: (n, d) -- the input the classifier head actually sees. Captured with a
        forward hook on ``model.classifier`` so it works for any architecture with that head,
        rather than hardcoding a per-model path. SentryCam probes the penultimate layer
        (l = L-1); earlier rounds of this project probed logits instead, which was a documented
        deviation forced by train_classifier accepting an arbitrary nn.Module.
    sequence_activations: (n, s, m) -- per sequence-step, per unit, for MM-PHATE's 4-way tensor,
        or None if the architecture keeps no time axis (e.g. a global-average-pooled CNN).
    """
    import torch

    captured = {}

    def hook(_module, inputs, _output):
        captured["penultimate"] = inputs[0].detach()

    handle = model.classifier.register_forward_hook(hook)
    seq_handle = None
    seq_module = getattr(model, "gru", None) or getattr(model, "features", None)

    def seq_hook(_module, _inputs, output):
        captured["sequence"] = (output[0] if isinstance(output, tuple) else output).detach()

    if seq_module is not None:
        seq_handle = seq_module.register_forward_hook(seq_hook)
    try:
        with torch.no_grad():
            model(x)
    finally:
        handle.remove()
        if seq_handle is not None:
            seq_handle.remove()

    penult = captured["penultimate"].cpu().numpy()
    seq = captured.get("sequence")
    if seq is None:
        return penult, None
    seq = seq.cpu()
    if seq.ndim == 4:            # conv feature map (n, channels, mel, time) -> (n, time, channels)
        n, c, mel, t = seq.shape
        if mel != 1:             # collapse any surviving mel axis; the time axis is what matters
            seq = seq.mean(dim=2, keepdim=True)
        seq = seq.squeeze(2).permute(0, 2, 1)
    elif seq.ndim != 3:          # (n, s, m) already, anything else has no usable sequence axis
        return penult, None
    if seq.shape[1] < 2:         # a single time step is not a sequence
        return penult, None
    return penult, seq.numpy()


def project_2d(activations):
    """Project (n, d) activations to (n, 2) by PCA on the two leading components.

    NOTE ON FIDELITY: SentryCam projects with a trained parametric autoencoder
    (d -> d/2 -> ... -> 2, GroupNorm+BatchNorm). This is PCA -- cheaper, deterministic, and a
    documented deviation. It must never be described as "SentryCam's projection"; the point it
    preserves is only the paper's reason for projecting at all, namely that its cluster metrics
    are defined in a low-dimensional space because raw high-dimensional distances are distorted.
    """
    x = np.asarray(activations, dtype=np.float64)
    if x.ndim != 2 or x.shape[0] == 0:
        raise ValueError("activations must be a non-empty (n, d) matrix")
    if x.shape[1] < 2:
        raise ValueError("activations need at least 2 dimensions to project to 2D")
    centered = x - x.mean(axis=0, keepdims=True)
    # SVD rather than an eigendecomposition of the covariance: stable for n < d, which happens
    # whenever the probe batch is smaller than the representation width.
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    return (centered @ vt[:2].T).astype(np.float32)


def sentrycam_probe(model, x, labels, keep_sequence=True):
    """One epoch's SentryCam-style record: 2D coordinates, labels, and the two cluster metrics.

    The metrics are computed IN THE 2D SPACE, per SentryCam's Tier-3 definition -- computing them
    in the raw representation space (as this project did in earlier rounds) is the exact thing the
    paper argues against, since high-dimensional distances are noise-dominated.
    """
    penult, seq = penultimate_and_sequence(model, x)
    coords = project_2d(penult)
    inter, intra = cluster_drift_metrics(coords, np.asarray(labels))
    record = {
        "coords_2d": coords,
        "labels": np.asarray(labels),
        "inter_cluster_distance_2d": inter,
        "intra_cluster_variance_2d": intra,
    }
    if keep_sequence and seq is not None:
        # MM-PHATE's 4-way tensor slice for this epoch: (samples, seq-steps, units), z-scored per
        # unit as the paper specifies. The expensive O((nsm)^3) embedding is post-hoc, never here.
        flat = seq.reshape(-1, seq.shape[-1])
        mu, sigma = flat.mean(axis=0), flat.std(axis=0)
        record["sequence_activations"] = ((seq - mu) / np.where(sigma > 0, sigma, 1.0)).astype(np.float32)
    return record
