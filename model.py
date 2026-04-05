import mlx.core as mx
import mlx.nn as nn

from layers import MultiHeadAttention, TransformerLayer
from heads import RobertaLMHead, ContactPredictionHead


class ESM2(nn.Module):
    def __init__(
        self,
        num_layers=33,
        embed_dim=1280,
        attention_heads=20,
        alphabet_size=33,
        padding_idx=1,
        eos_idx=2,
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

        self.lm_head = RobertaLMHead(
            embed_dim, alphabet_size, self.embed_tokens.weight
        )

        self.contact_head = ContactPredictionHead(
            num_layers * attention_heads,
            prepend_bos=True,
            append_eos=True,
            eos_idx=eos_idx,
        )

    def __call__(self, tokens, return_contacts=False):
        padding_mask = tokens == self.padding_idx
        mask = padding_mask if padding_mask.any() else None

        x = self.embed_tokens(tokens)

        if mask is not None:
            x = x * (~mx.expand_dims(mask, -1)).astype(x.dtype)

        need_head_weights = return_contacts
        attn_weights = [] if need_head_weights else None

        for layer in self.layers:
            x, attn = layer(x, mask=mask, need_head_weights=need_head_weights)
            if need_head_weights:
                attn_weights.append(attn)

        x = self.emb_layer_norm_after(x)
        logits = self.lm_head(x)

        result = {"logits": logits}

        if return_contacts:
            attentions = mx.stack(attn_weights, axis=1)
            if mask is not None:
                attn_mask = (~padding_mask).astype(attentions.dtype)
                attn_mask = mx.expand_dims(attn_mask, 1) * mx.expand_dims(attn_mask, 2)
                attentions = attentions * attn_mask[:, None, None, :, :]
            result["contacts"] = self.contact_head(tokens, attentions)

        return result

    def predict_contacts(self, tokens):
        return self(tokens, return_contacts=True)["contacts"]