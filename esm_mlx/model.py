"""ESM-2 protein language model implemented in Apple MLX.

Supports all six ESM-2 architectures (8M through 15B) with optional contact
prediction via stacked attention maps.
"""

import os

import mlx.core as mx
import mlx.nn as nn

HF_REPO_ID = "josephjojoe/esm-mlx"

from .layers import MultiHeadAttention, TransformerLayer
from .heads import RobertaLMHead, ContactPredictionHead

# Architecture configs keyed by the official Facebook Research model names.
# Each maps to (num_layers, embed_dim, attention_heads, alphabet_size).
MODEL_CONFIGS: dict[str, dict] = {
    "esm2_t6_8M_UR50D": {
        "num_layers": 6,
        "embed_dim": 320,
        "attention_heads": 20,
    },
    "esm2_t12_35M_UR50D": {
        "num_layers": 12,
        "embed_dim": 480,
        "attention_heads": 20,
    },
    "esm2_t30_150M_UR50D": {
        "num_layers": 30,
        "embed_dim": 640,
        "attention_heads": 20,
    },
    "esm2_t33_650M_UR50D": {
        "num_layers": 33,
        "embed_dim": 1280,
        "attention_heads": 20,
    },
    "esm2_t36_3B_UR50D": {
        "num_layers": 36,
        "embed_dim": 2560,
        "attention_heads": 40,
    },
    "esm2_t48_15B_UR50D": {
        "num_layers": 48,
        "embed_dim": 5120,
        "attention_heads": 40,
    },
}


class ESM2(nn.Module):
    """ESM-2 protein language model.

    Args:
        num_layers: Number of transformer layers.
        embed_dim: Hidden dimension of the model.
        attention_heads: Number of attention heads per layer.
        alphabet_size: Vocabulary size (default 33 for ESM-2).
        padding_idx: Token index used for padding.
        eos_idx: Token index for end-of-sequence.
    """

    def __init__(
        self,
        num_layers: int = 33,
        embed_dim: int = 1280,
        attention_heads: int = 20,
        alphabet_size: int = 33,
        padding_idx: int = 1,
        eos_idx: int = 2,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.padding_idx = padding_idx

        self.embed_tokens = nn.Embedding(alphabet_size, embed_dim)

        self.layers = [
            TransformerLayer(embed_dim, 4 * embed_dim, attention_heads)
            for _ in range(num_layers)
        ]

        self.emb_layer_norm_after = nn.LayerNorm(embed_dim)

        self.lm_head = RobertaLMHead(embed_dim, alphabet_size)

        self.contact_head = ContactPredictionHead(
            num_layers * attention_heads,
            prepend_bos=True,
            append_eos=True,
            eos_idx=eos_idx,
        )

    def __call__(
        self,
        tokens: mx.array,
        return_contacts: bool = False,
        repr_layers: list[int] | None = None,
    ) -> dict[str, mx.array]:
        """Run a forward pass.

        Args:
            tokens: Integer token array of shape ``(batch, seq_len)``.
            return_contacts: If ``True``, also predict contact maps from
                stacked per-layer attention weights.
            repr_layers: Optional list of layer indices whose hidden states
                to return.  Layer 0 is the embedding output; layers
                1..num_layers are transformer block outputs; layer
                ``num_layers`` is the post-LayerNorm representation (same
                convention as the PyTorch ESM-2 API).

        Returns:
            Dict with ``"logits"``, optionally ``"contacts"``, and
            optionally ``"representations"`` (a dict mapping layer index
            to the corresponding hidden state).
        """
        padding_mask = tokens == self.padding_idx
        mask = padding_mask if padding_mask.any() else None

        repr_set = set(repr_layers) if repr_layers else set()
        representations: dict[int, mx.array] = {}

        # Scale factor compensates for the removal of token dropout (which
        # scales activations by 1-p ≈ 0.88 during training).
        x = self.embed_tokens(tokens) * 0.88

        if mask is not None:
            x = x * (~mx.expand_dims(mask, -1)).astype(x.dtype)

        if 0 in repr_set:
            representations[0] = x

        need_head_weights = return_contacts
        attn_weights = [] if need_head_weights else None

        for i, layer in enumerate(self.layers):
            x, attn = layer(x, mask=mask, need_head_weights=need_head_weights)
            if need_head_weights:
                attn_weights.append(attn)
            if (i + 1) in repr_set:
                representations[i + 1] = x

        x = self.emb_layer_norm_after(x)

        if self.num_layers in repr_set:
            representations[self.num_layers] = x

        logits = self.lm_head(x, self.embed_tokens.weight)

        result = {"logits": logits}

        if repr_layers:
            result["representations"] = representations

        if return_contacts:
            attentions = mx.stack(attn_weights, axis=1)
            if mask is not None:
                attn_mask = (~padding_mask).astype(attentions.dtype)
                attn_mask = mx.expand_dims(attn_mask, 1) * mx.expand_dims(attn_mask, 2)
                attentions = attentions * attn_mask[:, None, None, :, :]
            result["contacts"] = self.contact_head(tokens, attentions)

        return result

    def predict_contacts(self, tokens: mx.array) -> mx.array:
        """Convenience method: return only the contact-map prediction."""
        return self(tokens, return_contacts=True)["contacts"]

    @classmethod
    def from_pretrained(
        cls,
        model_name: str = "esm2_t33_650M_UR50D",
        weights_path: str | None = None,
    ) -> "ESM2":
        """Load a pretrained ESM-2 model.

        Downloads weights from HuggingFace automatically if no local copy
        is found.

        Args:
            model_name: One of the keys in ``MODEL_CONFIGS``.
            weights_path: Explicit path to a ``.safetensors`` file.  When
                ``None``, checks ``weights/<model_name>.safetensors`` locally,
                then falls back to downloading from HuggingFace.

        Returns:
            An ``ESM2`` instance with loaded weights.
        """
        if model_name not in MODEL_CONFIGS:
            raise ValueError(
                f"Unknown model {model_name!r}. "
                f"Choose from: {list(MODEL_CONFIGS)}"
            )

        config = MODEL_CONFIGS[model_name]
        model = cls(**config)

        if weights_path is None:
            local_path = f"weights/{model_name}.safetensors"
            if os.path.exists(local_path):
                weights_path = local_path
            else:
                from huggingface_hub import hf_hub_download

                weights_path = hf_hub_download(
                    repo_id=HF_REPO_ID,
                    filename=f"{model_name}.safetensors",
                )

        weights = mx.load(weights_path)
        model.load_weights(list(weights.items()))
        mx.eval(model.parameters())

        return model
