"""Contract for src.encoder (width: sociable; torchaudio is never required)."""
from __future__ import annotations

import importlib.util

import pytest
import torch
from torch import nn

from src.encoder import (AdaptedBlockModel, EncoderUnavailable, extract_hidden, load_wav2vec2,
                         make_linear_head, mean_pool_features)
from src.latency import count_parameters

FRAMES, DIM, LAYERS = 4, 2, 3


def test_load_wav2vec2_raises_encoder_unavailable_when_torchaudio_is_missing():
    if importlib.util.find_spec("torchaudio") is not None:
        pytest.skip("torchaudio importable; the unavailable path cannot be exercised")
    with pytest.raises(EncoderUnavailable):
        load_wav2vec2("cpu")


def test_mean_pool_without_lengths_averages_all_frames():
    hidden = torch.tensor([[[1., 2.], [3., 4.], [100., 100.]], [[1., 1.], [2., 2.], [3., 3.]]])
    got = mean_pool_features(hidden)
    assert got.shape == (2, 2)
    assert torch.allclose(got, torch.tensor([[104 / 3, 106 / 3], [2., 2.]]))


def test_mean_pool_with_lengths_ignores_padded_frames():
    """Row 0 has 2 valid frames (the 100s are padding): mean = [2, 3]; row 1 uses all 3 frames."""
    hidden = torch.tensor([[[1., 2.], [3., 4.], [100., 100.]], [[1., 1.], [2., 2.], [3., 3.]]])
    got = mean_pool_features(hidden, torch.tensor([2, 3]))
    assert torch.allclose(got, torch.tensor([[2., 3.], [2., 2.]]))


class FakeEncoder:
    """Duck-typed like torchaudio Wav2Vec2Model: extract_features(waveforms, lengths) -> (layers, lengths)."""

    def extract_features(self, waveforms, lengths=None):
        b = waveforms.shape[0]
        base = waveforms.reshape(b, FRAMES, DIM)
        return [base * (k + 1) for k in range(LAYERS)], torch.full((b,), FRAMES)


def _waves(n=7):
    """Multiples of 1/8 so float16 holds them exactly."""
    return (torch.arange(n * FRAMES * DIM, dtype=torch.float32).reshape(n, FRAMES * DIM) % 16) / 8.0


@pytest.mark.parametrize("layer", range(LAYERS))
def test_extract_hidden_returns_the_requested_layer_for_every_example(layer):
    w = _waves()
    got = extract_hidden(FakeEncoder(), w, layer_index=layer, batch_size=3)
    assert got.shape == (7, FRAMES, DIM) and not got.requires_grad
    expected = w.reshape(7, FRAMES, DIM) * (layer + 1)
    assert torch.equal(got.float(), expected)
    assert torch.equal(got.half().float(), expected), "value must survive a float16 cast"


def test_extract_hidden_values_do_not_depend_on_batch_size():
    w = _waves()
    ref = extract_hidden(FakeEncoder(), w, layer_index=1, batch_size=1)
    for bs in (2, 7, 8, 50):
        assert torch.equal(extract_hidden(FakeEncoder(), w, layer_index=1, batch_size=bs), ref), bs


def test_extract_hidden_rejects_a_layer_index_beyond_the_available_layers():
    with pytest.raises(ValueError):
        extract_hidden(FakeEncoder(), _waves(), layer_index=LAYERS, batch_size=4)


def test_linear_head_maps_features_to_class_scores_with_exact_parameter_count():
    head = make_linear_head(8, 3)
    assert head(torch.zeros(5, 8)).shape == (5, 3)
    assert count_parameters(head) == 8 * 3 + 3


def _block():
    return nn.TransformerEncoderLayer(d_model=8, nhead=2, dim_feedforward=16, dropout=0.0, batch_first=True)


def test_adapted_block_model_parameter_count_is_block_plus_head_exactly():
    """Layer: attn 192+24 in, 64+8 out, ffn 144+136, two LayerNorms 16+16 = 600; head 8*3+3 = 27."""
    block = _block()
    assert count_parameters(block) == 600
    model = AdaptedBlockModel(block, 3, 8)
    assert count_parameters(model) == 627
    assert all(p.requires_grad for p in model.parameters())


def test_adapted_block_forward_is_block_then_frame_mean_then_linear():
    torch.manual_seed(0)
    block = _block()
    model = AdaptedBlockModel(block, 3, 8).eval()
    hidden = torch.randn(4, 5, 8)
    block_ids = {id(p) for p in block.parameters()}
    extra = [p for p in model.parameters() if id(p) not in block_ids]
    assert len(extra) == 2, "only a linear head may sit beside the block"
    weight = next(p for p in extra if p.ndim == 2); bias = next(p for p in extra if p.ndim == 1)
    with torch.no_grad():
        expected = block.eval()(hidden).mean(dim=1) @ weight.T + bias
        assert torch.allclose(model(hidden), expected, atol=1e-6)


def test_adapted_block_gradients_reach_block_and_head():
    model = AdaptedBlockModel(_block(), 3, 8)
    model(torch.randn(4, 5, 8)).sum().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
