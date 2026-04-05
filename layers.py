import mlx.core as mx
import mlx.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_dim, num_heads):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads

        self.head_dim = hidden_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.rope = nn.RoPE(self.head_dim, traditional=False, base=10000)

    
    def __call__(self, x, mask=None, need_head_weights=False):
        B, T, _ = x.shape

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = q.reshape(B, T, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(B, T, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(B, T, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)

        q = self.rope(q)
        k = self.rope(k)

        if need_head_weights:
            scores = (q * self.scale) @ k.transpose(0, 1, 3, 2)
            if mask is not None:
                scores = scores + mask.reshape(B, 1, 1, T).astype(scores.dtype) * -1e9
            weights = mx.softmax(scores.astype(mx.float32), axis=-1).astype(scores.dtype)
            out = weights @ v
        else:
            sdpa_mask = None
            if mask is not None:
                sdpa_mask = mask.reshape(B, 1, 1, T).astype(q.dtype) * -1e9
            out = mx.fast.scaled_dot_product_attention(
                q, k, v, scale=self.scale, mask=sdpa_mask,
            )
            weights = None

        out = out.transpose(0, 2, 1, 3).reshape(B, T, -1)
        out = self.out_proj(out)

        return out, weights


class TransformerLayer(nn.Module):
    def __init__(self, embed_dim, ffn_dim, num_heads):
        super().__init__()
        self.self_attn = MultiHeadAttention(embed_dim, num_heads)
        self.self_attn_layer_norm = nn.LayerNorm(embed_dim)
        self.fc1 = nn.Linear(embed_dim, ffn_dim)
        self.fc2 = nn.Linear(ffn_dim, embed_dim)
        self.final_layer_norm = nn.LayerNorm(embed_dim)

    def __call__(self, x, mask=None, need_head_weights=False):
        # Attention block
        residual = x
        x = self.self_attn_layer_norm(x)
        x, attn = self.self_attn(x, mask=mask, need_head_weights=need_head_weights)
        x = residual + x

        # FFN block
        residual = x
        x = self.final_layer_norm(x)
        x = nn.gelu(self.fc1(x))
        x = self.fc2(x)
        x = residual + x

        return x, attn