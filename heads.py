import mlx.core as mx
import mlx.nn as nn


def symmetrize(x):
    return x + mx.swapaxes(x, -1, -2)


def apc(x):
    a1 = x.sum(-1, keepdims=True)
    a2 = x.sum(-2, keepdims=True)
    a12 = x.sum((-1, -2), keepdims=True)
    return x - (a1 * a2) / a12


class RobertaLMHead(nn.Module):
    def __init__(self, embed_dim, output_dim, embed_tokens_weight):
        super().__init__()
        self.dense = nn.Linear(embed_dim, embed_dim)
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.weight = embed_tokens_weight
        self.bias = mx.zeros(output_dim)

    def __call__(self, x):
        x = self.dense(x)
        x = nn.gelu(x)
        x = self.layer_norm(x)
        x = x @ self.weight.T + self.bias
        return x


class ContactPredictionHead(nn.Module):
    def __init__(self, in_features, prepend_bos, append_eos, eos_idx=None):
        super().__init__()
        self.prepend_bos = prepend_bos
        self.append_eos = append_eos
        self.eos_idx = eos_idx
        self.regression = nn.Linear(in_features, 1)

    def __call__(self, tokens, attentions):
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