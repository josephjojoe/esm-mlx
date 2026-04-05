"""Prediction heads for ESM-2.

Includes the masked language model head (RobertaLMHead) and the contact
prediction head that operates on stacked per-layer attention maps.
"""

import mlx.core as mx
import mlx.nn as nn


def symmetrize(x: mx.array) -> mx.array:
    """Average a matrix with its transpose to enforce symmetry."""
    return x + mx.swapaxes(x, -1, -2)


def apc(x: mx.array) -> mx.array:
    """Average Product Correction — removes phylogenetic background signal."""
    a1 = x.sum(-1, keepdims=True)
    a2 = x.sum(-2, keepdims=True)
    a12 = x.sum((-1, -2), keepdims=True)
    return x - (a1 * a2) / a12


class RobertaLMHead(nn.Module):
    """Masked-language-model head with weight tying.

    The final projection reuses the embedding weight matrix (passed in at
    call time) plus a learnable bias vector.
    """

    def __init__(self, embed_dim: int, output_dim: int):
        super().__init__()
        self.dense = nn.Linear(embed_dim, embed_dim)
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.bias = mx.zeros(output_dim)

    def __call__(self, x: mx.array, embed_weight: mx.array) -> mx.array:
        x = self.dense(x)
        x = nn.gelu(x)
        x = self.layer_norm(x)
        x = x @ embed_weight.T + self.bias
        return x


class ContactPredictionHead(nn.Module):
    """Predicts residue–residue contacts from attention maps.

    Takes stacked attention weights from all layers, strips BOS/EOS tokens,
    symmetrizes, applies average product correction (APC), and projects
    through a linear layer with sigmoid activation.
    """

    def __init__(
        self,
        in_features: int,
        prepend_bos: bool,
        append_eos: bool,
        eos_idx: int | None = None,
    ):
        super().__init__()
        self.prepend_bos = prepend_bos
        self.append_eos = append_eos
        self.eos_idx = eos_idx
        self.regression = nn.Linear(in_features, 1)

    def __call__(self, tokens: mx.array, attentions: mx.array) -> mx.array:
        if self.append_eos:
            eos_mask = (tokens != self.eos_idx).astype(attentions.dtype)
            eos_mask = mx.expand_dims(eos_mask, 1) * mx.expand_dims(eos_mask, 2)
            attentions = attentions * eos_mask[:, None, None, :, :]
            attentions = attentions[..., :-1, :-1]

        if self.prepend_bos:
            attentions = attentions[..., 1:, 1:]

        B, L, H, T, _ = attentions.shape
        attentions = attentions.reshape(B, L * H, T, T)
        attentions = apc(symmetrize(attentions))
        attentions = attentions.transpose(0, 2, 3, 1)

        return mx.sigmoid(self.regression(attentions).squeeze(-1))
