"""Wav2Vec2 encoder interface and adapted block model."""
from __future__ import annotations

import torch
from torch import nn


class EncoderUnavailable(Exception):
    """Raised when torchaudio cannot be imported or weights cannot be downloaded."""
    pass


def load_wav2vec2(device: str = "cpu"):
    """Load Wav2Vec2 base model, raising EncoderUnavailable on any import/download failure.

    Args:
        device: Device to load model on (e.g., 'cpu', 'cuda').

    Returns:
        Loaded Wav2Vec2 model.

    Raises:
        EncoderUnavailable: If torchaudio cannot be imported or weights fail to download.
    """
    try:
        import torchaudio
    except ImportError as e:
        raise EncoderUnavailable(str(e))

    try:
        model = torchaudio.pipelines.WAV2VEC2_BASE.get_model()
        model = model.to(device)
        return model
    except Exception as e:
        raise EncoderUnavailable(str(e))


def mean_pool_features(hidden: torch.Tensor, lengths: torch.Tensor | None = None) -> torch.Tensor:
    """Pool hidden states by averaging across frames, optionally masking by length.

    Args:
        hidden: Tensor of shape (batch, frames, dim).
        lengths: Optional tensor of shape (batch,) indicating valid frame count per example.

    Returns:
        Pooled tensor of shape (batch, dim).
    """
    if lengths is None:
        return hidden.mean(dim=1)

    # Mask padded frames: create (batch, frames, 1) mask
    batch_size, num_frames, dim = hidden.shape
    mask = torch.arange(num_frames, device=hidden.device).unsqueeze(0) < lengths.unsqueeze(1)
    mask = mask.unsqueeze(2).float()  # (batch, frames, 1)

    # Apply mask and sum, then divide by valid frame count per example
    masked = hidden * mask
    sums = masked.sum(dim=1)  # (batch, dim)
    valid_counts = mask.sum(dim=1)  # (batch, 1)

    return sums / valid_counts


def extract_hidden(
    encoder,
    waveforms: torch.Tensor,
    layer_index: int,
    batch_size: int,
) -> torch.Tensor:
    """Extract hidden states from a specific layer of the encoder, processing in batches.

    Args:
        encoder: Encoder with extract_features(waveforms, lengths) -> (layers, lengths).
        waveforms: Tensor of shape (n_examples, n_samples).
        layer_index: Which layer to extract (0-indexed).
        batch_size: Batch size for processing.

    Returns:
        Hidden states tensor of shape (n_examples, frames, dim), non-differentiable.

    Raises:
        ValueError: If layer_index is beyond available layers.
    """
    outputs = []
    device = next(encoder.parameters()).device

    with torch.no_grad():
        for start in range(0, len(waveforms), batch_size):
            end = min(start + batch_size, len(waveforms))
            batch = waveforms[start:end].to(device)

            # encoder.extract_features returns (layers_list, lengths)
            layers, lengths = encoder.extract_features(batch)

            # Check layer index validity on first batch
            if start == 0 and layer_index >= len(layers):
                raise ValueError(f"layer_index {layer_index} >= available layers {len(layers)}")

            outputs.append(layers[layer_index])

    return torch.cat(outputs, dim=0)


def make_linear_head(in_features: int, num_classes: int) -> nn.Module:
    """Create a simple linear classification head.

    Args:
        in_features: Input feature dimension.
        num_classes: Number of output classes.

    Returns:
        Linear layer mapping (batch, in_features) -> (batch, num_classes).
    """
    return nn.Linear(in_features, num_classes)


class AdaptedBlockModel(nn.Module):
    """Transformer block + mean pooling + linear head for classification."""

    def __init__(self, block: nn.Module, num_classes: int, feature_dim: int):
        """Initialize with a transformer block.

        Args:
            block: TransformerEncoderLayer or similar module.
            num_classes: Number of output classes.
            feature_dim: Feature dimension (input to block).
        """
        super().__init__()
        self.block = block
        self.head = make_linear_head(feature_dim, num_classes)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        """Forward pass: block -> mean pool -> linear head.

        Args:
            hidden: Tensor of shape (batch, frames, feature_dim).

        Returns:
            Logits of shape (batch, num_classes).
        """
        x = self.block(hidden)
        x = mean_pool_features(x)
        return self.head(x)
