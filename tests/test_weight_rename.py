"""Unit tests for the fairseq → MLX weight-name rename.

These tests protect the load path that accepts the HuggingFace-hosted
checkpoints, which are stored in the raw fairseq schema (keys nested under
``encoder.sentence_encoder.`` and ``encoder.lm_head.``).  If the rename
regresses, ``ESM2.from_pretrained`` silently breaks for everyone whose cache
holds the original upload.
"""

from __future__ import annotations

import pytest

from esm_mlx.model import _canonicalise_weights, _rename_fairseq_key


class _Arr:
    """Stand-in for ``mx.array`` so these tests don't require MLX."""

    def __init__(self, shape):
        self.shape = tuple(shape)


@pytest.mark.parametrize(
    "fairseq_key,mlx_key",
    [
        ("encoder.sentence_encoder.embed_tokens.weight", "embed_tokens.weight"),
        (
            "encoder.sentence_encoder.emb_layer_norm_after.weight",
            "emb_layer_norm_after.weight",
        ),
        (
            "encoder.sentence_encoder.layers.0.self_attn.q_proj.weight",
            "layers.0.self_attn.q_proj.weight",
        ),
        (
            "encoder.sentence_encoder.layers.35.final_layer_norm.bias",
            "layers.35.final_layer_norm.bias",
        ),
        ("encoder.lm_head.dense.weight", "lm_head.dense.weight"),
        ("encoder.lm_head.bias", "lm_head.bias"),
        ("encoder.lm_head.layer_norm.bias", "lm_head.layer_norm.bias"),
        # already MLX-native keys pass through unchanged
        ("contact_head.regression.weight", "contact_head.regression.weight"),
        ("contact_head.regression.bias", "contact_head.regression.bias"),
        # idempotence: renaming a renamed key must not mutate it
        ("embed_tokens.weight", "embed_tokens.weight"),
        ("layers.5.self_attn.k_proj.bias", "layers.5.self_attn.k_proj.bias"),
    ],
)
def test_rename_fairseq_key(fairseq_key, mlx_key):
    assert _rename_fairseq_key(fairseq_key) == mlx_key
    # Idempotence.
    assert _rename_fairseq_key(_rename_fairseq_key(fairseq_key)) == mlx_key


@pytest.mark.parametrize(
    "skipped",
    [
        "encoder.lm_head.weight",  # tied with embed_tokens.weight at forward time
        "encoder.sentence_encoder.layers.0.self_attn.rot_emb.inv_freq",
        "encoder.sentence_encoder.layers.10.self_attn.bias_k",
        "encoder.sentence_encoder.layers.10.self_attn.bias_v",
    ],
)
def test_rename_fairseq_key_skips(skipped):
    assert _rename_fairseq_key(skipped) is None


def test_canonicalise_weights_is_a_full_rename_pass():
    """A mixed bag of fairseq + already-MLX + skip keys canonicalises correctly."""
    raw = [
        ("encoder.sentence_encoder.embed_tokens.weight", _Arr((33, 2560))),
        (
            "encoder.sentence_encoder.layers.0.self_attn.q_proj.weight",
            _Arr((2560, 2560)),
        ),
        ("encoder.lm_head.dense.weight", _Arr((2560, 2560))),
        ("encoder.lm_head.bias", _Arr((33,))),
        # skipped:
        ("encoder.lm_head.weight", _Arr((33, 2560))),
        (
            "encoder.sentence_encoder.layers.0.self_attn.rot_emb.inv_freq",
            _Arr((32,)),
        ),
        # already MLX:
        ("contact_head.regression.weight", _Arr((1, 1440))),
    ]
    out = _canonicalise_weights(raw)
    assert set(out) == {
        "embed_tokens.weight",
        "layers.0.self_attn.q_proj.weight",
        "lm_head.dense.weight",
        "lm_head.bias",
        "contact_head.regression.weight",
    }
    # Values are passed through unchanged (identity, same object).
    assert out["embed_tokens.weight"].shape == (33, 2560)


def test_canonicalise_weights_is_idempotent():
    """Running the canonicaliser twice (e.g. on an already-renamed file) is a no-op."""
    raw = [
        ("encoder.sentence_encoder.layers.0.self_attn.v_proj.bias", _Arr((2560,))),
        ("lm_head.bias", _Arr((33,))),
    ]
    once = _canonicalise_weights(raw)
    twice = _canonicalise_weights(once.items())
    assert set(once) == set(twice)
    assert all(once[k] is twice[k] for k in once)
