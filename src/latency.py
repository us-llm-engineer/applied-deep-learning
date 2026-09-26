"""Latency measurement and model size utilities."""
from __future__ import annotations

from dataclasses import dataclass
import time

import numpy as np
import torch
from torch import nn


@dataclass(frozen=True)
class LatencyRecord:
    """Frozen record of latency measurements for a single forward pass with batch size 1."""

    median_ms: float
    p95_ms: float
    mean_ms: float
    repeats: int
    num_threads: int
    device: str


def measure_batch1_latency(
    model: nn.Module,
    input_shape: tuple[int, ...],
    *,
    warmup: int = 3,
    repeats: int = 20,
    seed: int = 0,
) -> LatencyRecord:
    """Measure latency of model forward passes with batch size 1 inputs.

    Times only the repeat passes (excluding warmup). The model is put in eval mode
    with gradients disabled during measurement, and the original training state is
    restored afterwards. The same seeded input is reused for all forward passes.

    Args:
        model: Neural network module to measure.
        input_shape: Shape of input (without batch dimension), must be a tuple.
        warmup: Number of forward passes to run before timing (default 3).
        repeats: Number of forward passes to time (default 20, must be positive).
        seed: Random seed for input generation (default 0).

    Returns:
        LatencyRecord containing median, p95, mean latencies (in ms), along with
        metadata about the measurement (repeats, num_threads, device).

    Raises:
        ValueError: If input_shape is not a tuple, warmup is negative, or repeats is non-positive.
    """
    if not isinstance(input_shape, tuple):
        raise ValueError("input_shape must be a tuple")
    if warmup < 0:
        raise ValueError("warmup must be non-negative")
    if repeats <= 0:
        raise ValueError("repeats must be positive")

    # Save original training state
    original_training = model.training

    # Get device from model's first parameter
    device_obj = next(model.parameters()).device

    # Prepare seeded input (batch size 1)
    rng = np.random.default_rng(seed)
    input_data = torch.from_numpy(rng.standard_normal((1,) + input_shape)).float()
    input_data = input_data.to(device_obj)

    # Timing list for repeats only
    latencies_ms: list[float] = []

    try:
        model.eval()
        with torch.no_grad():
            # Warmup passes (not timed)
            for _ in range(warmup):
                model(input_data)

            # Timed repeats
            for _ in range(repeats):
                if device_obj.type == "cuda":
                    torch.cuda.synchronize()
                t_start = time.perf_counter()
                model(input_data)
                if device_obj.type == "cuda":
                    torch.cuda.synchronize()
                t_end = time.perf_counter()
                latencies_ms.append((t_end - t_start) * 1000.0)
    finally:
        # Restore original training state
        model.train(original_training)

    # Compute statistics from repeats only
    latencies = np.array(latencies_ms)
    median_ms = float(np.median(latencies))
    p95_ms = float(np.percentile(latencies, 95))
    mean_ms = float(np.mean(latencies))
    num_threads = torch.get_num_threads()
    device = str(device_obj)

    return LatencyRecord(
        median_ms=median_ms,
        p95_ms=p95_ms,
        mean_ms=mean_ms,
        repeats=repeats,
        num_threads=num_threads,
        device=device,
    )


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters (requires_grad=True) in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def model_size_bytes(model: nn.Module) -> int:
    """Calculate total size in bytes of all parameters and buffers.

    Includes both trainable and non-trainable parameters, plus all buffers
    (e.g., BatchNorm running statistics). Size is calculated based on dtype.
    """
    total_bytes = 0

    # Add parameters
    for param in model.parameters():
        total_bytes += param.numel() * param.data.element_size()

    # Add buffers
    for buffer in model.buffers():
        total_bytes += buffer.numel() * buffer.data.element_size()

    return total_bytes
