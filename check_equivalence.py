"""
Check numerical equivalence between PyTorch ESM2 and the MLX port.

Requires PyTorch ESM2 (downloaded via torch.hub) and converted MLX weights.

Usage:
    python check_equivalence.py [--weights esm2_t33_650M_UR50D.safetensors] [--diagnose]
"""

import argparse
import sys

import numpy as np
import torch

import mlx.core as mx

from esm_mlx import ESM2

PADDING_IDX = 1


def load_pytorch_model():
    model, alphabet = torch.hub.load(
        "facebookresearch/esm", "esm2_t33_650M_UR50D"
    )
    model.eval()
    batch_converter = alphabet.get_batch_converter()
    return model, alphabet, batch_converter


def load_mlx_model(weights_path):
    model = ESM2()
    weights = mx.load(weights_path)
    model.load_weights(list(weights.items()))
    return model


def to_np(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().float().numpy()
    return np.array(x, dtype=np.float32)


def topk_agreement(pt_np, mx_np, ks=(1, 5, 10)):
    """Returns {k: fraction} of top-k overlap averaged across positions."""
    pt_ranks = np.argsort(-pt_np, axis=-1)
    mx_ranks = np.argsort(-mx_np, axis=-1)
    results = {}
    for k in ks:
        pt_topk = pt_ranks[..., :k]
        mx_topk = mx_ranks[..., :k]
        match = np.array([
            len(set(p) & set(m)) / k
            for p, m in zip(pt_topk.reshape(-1, k), mx_topk.reshape(-1, k))
        ])
        results[k] = float(np.mean(match))
    return results


def error_distribution(name, pt_np, mx_np):
    diff = np.abs(pt_np - mx_np)
    flat = diff.flatten()
    pcts = [50, 90, 95, 99, 99.9, 100]
    vals = np.percentile(flat, pcts)
    print(f"  {name} error distribution:")
    for p, v in zip(pcts, vals):
        print(f"    p{p:<5} = {v:.6e}")
    rel_diff = diff / (np.abs(pt_np) + 1e-8)
    print(f"    mean relative error = {np.mean(rel_diff):.6e}")


def worst_positions(name, pt_np, mx_np, tokens_np=None, top_k=5):
    diff = np.abs(pt_np - mx_np)
    if diff.ndim == 3:
        pos_max = diff.max(axis=-1)
    elif diff.ndim == 2:
        pos_max = diff
    else:
        return
    print(f"  {name} worst positions (batch, seq_pos) -> max_error:")
    flat_idx = np.argsort(pos_max.flatten())[::-1][:top_k]
    for idx in flat_idx:
        coords = np.unravel_index(idx, pos_max.shape)
        val = pos_max[coords]
        tok = f", token={tokens_np[coords]}" if tokens_np is not None else ""
        print(f"    {coords}{tok}: {val:.6e}")


# ── Layer-by-layer diagnostics ──────────────────────────────────────────────

# WARNING: This reimplements the ESM2 forward pass to capture per-layer outputs.
# Must be kept in sync with ESM2.__call__ in model.py.
def run_mlx_layerwise(mx_model, tokens):
    intermediates = {}
    padding_mask = tokens == mx_model.padding_idx
    mask = padding_mask if padding_mask.any() else None

    x = mx_model.embed_tokens(tokens) * 0.88
    if mask is not None:
        x = x * (~mx.expand_dims(mask, -1)).astype(x.dtype)
    mx.eval(x)
    intermediates[0] = x

    for i, layer in enumerate(mx_model.layers):
        x, _ = layer(x, mask=mask, need_head_weights=False)
        mx.eval(x)
        intermediates[i + 1] = x

    x = mx_model.emb_layer_norm_after(x)
    mx.eval(x)
    intermediates["post_norm"] = x

    logits = mx_model.lm_head(x, mx_model.embed_tokens.weight)
    mx.eval(logits)
    intermediates["logits"] = logits

    return intermediates


def diagnose_layerwise(pt_model, mx_model, batch_tokens):
    print("\n" + "=" * 70)
    print("LAYER-BY-LAYER DIAGNOSTIC")
    print("=" * 70)

    mx_tokens = mx.array(batch_tokens.numpy())
    tokens_np = batch_tokens.numpy()
    non_pad = tokens_np != PADDING_IDX

    all_layers = list(range(pt_model.num_layers + 1))
    with torch.no_grad():
        pt_out = pt_model(batch_tokens, repr_layers=all_layers, need_head_weights=False)
    pt_reprs = pt_out["representations"]
    pt_logits = pt_out["logits"]

    mx_intermediates = run_mlx_layerwise(mx_model, mx_tokens)

    print(f"\n  {'Stage':<16} {'max_diff':>12} {'mean_diff':>12}  {'nonpad_max':>12}")
    print("  " + "-" * 60)

    prev_max = 0
    for layer_idx in range(pt_model.num_layers):
        pt_np = to_np(pt_reprs[layer_idx])
        mx_np = to_np(mx_intermediates[layer_idx])
        diff = np.abs(pt_np - mx_np)
        max_d = float(np.max(diff))
        mean_d = float(np.mean(diff))

        per_pos = diff.max(axis=-1)
        nonpad_max = float(per_pos[non_pad].max()) if non_pad.any() else max_d

        growth = ""
        if prev_max > 0 and max_d > prev_max:
            growth = f" ({max_d / prev_max:.1f}x)"
        prev_max = max_d if max_d > 0 else prev_max

        label = "embed" if layer_idx == 0 else f"layer {layer_idx}"
        print(f"  {label:<16} {max_d:>12.6e} {mean_d:>12.6e}  {nonpad_max:>12.6e}{growth}")

    pt_postnorm = to_np(pt_reprs[pt_model.num_layers])
    mx_postnorm = to_np(mx_intermediates["post_norm"])
    diff = np.abs(pt_postnorm - mx_postnorm)
    max_d, mean_d = float(np.max(diff)), float(np.mean(diff))
    nonpad_max = float(diff.max(axis=-1)[non_pad].max()) if non_pad.any() else max_d
    print(f"  {'post_norm':<16} {max_d:>12.6e} {mean_d:>12.6e}  {nonpad_max:>12.6e}")

    pt_l = to_np(pt_logits)
    mx_l = to_np(mx_intermediates["logits"])
    diff = np.abs(pt_l - mx_l)
    max_d, mean_d = float(np.max(diff)), float(np.mean(diff))
    nonpad_max = float(diff.max(axis=-1)[non_pad].max()) if non_pad.any() else max_d
    print(f"  {'logits':<16} {max_d:>12.6e} {mean_d:>12.6e}  {nonpad_max:>12.6e}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main(weights_path, diagnose=False):
    print("Loading PyTorch model...")
    pt_model, alphabet, batch_converter = load_pytorch_model()

    print("Loading MLX model...")
    mx_model = load_mlx_model(weights_path)

    test_sequences = [
        ("protein1", "MKTAYIAKQRQISFVKSHFSRQLE"),
        ("protein2", "KALTARQQEVFDLIRD"),
    ]

    _, _, batch_tokens = batch_converter(test_sequences)
    print(f"\nInput shape: {batch_tokens.shape}")

    print("\nRunning PyTorch model...")
    with torch.no_grad():
        pt_out = pt_model(batch_tokens, return_contacts=True)

    print("Running MLX model...")
    mx_tokens = mx.array(batch_tokens.numpy())
    mx_out = mx_model(mx_tokens, return_contacts=True)
    mx.eval(mx_out["logits"], mx_out["contacts"])

    pt_logits = to_np(pt_out["logits"])
    mx_logits = to_np(mx_out["logits"])
    pt_contacts = to_np(pt_out["contacts"])
    mx_contacts = to_np(mx_out["contacts"])
    tokens_np = batch_tokens.numpy()
    non_pad = tokens_np != PADDING_IDX

    logit_diff = np.abs(pt_logits - mx_logits)
    per_pos_logit = logit_diff.max(axis=-1)  # (B, T)
    nonpad_logit_max = float(per_pos_logit[non_pad].max())
    nonpad_logit_mean = float(per_pos_logit[non_pad].mean())

    contact_diff = np.abs(pt_contacts - mx_contacts)
    contact_max = float(contact_diff.max())
    contact_mean = float(contact_diff.mean())

    tk = topk_agreement(pt_logits, mx_logits)

    pt_preds = np.argmax(pt_logits, axis=-1)
    mx_preds = np.argmax(mx_logits, axis=-1)
    token_match = float(np.mean(pt_preds[non_pad] == mx_preds[non_pad]))

    print("\nResults (non-padding positions):")

    logits_ok = nonpad_logit_max < 5e-3
    contacts_ok = contact_max < 1e-3
    all_passed = logits_ok and contacts_ok

    status = "PASS" if logits_ok else "FAIL"
    print(f"  [{status}] logits:   max_diff={nonpad_logit_max:.6e}, mean_diff={nonpad_logit_mean:.6e} (atol=5e-3)")
    status = "PASS" if contacts_ok else "FAIL"
    print(f"  [{status}] contacts: max_diff={contact_max:.6e}, mean_diff={contact_mean:.6e} (atol=1e-3)")

    print(f"\n  Token prediction agreement: {token_match * 100:.1f}%")
    for k, v in tk.items():
        print(f"  Top-{k} agreement: {v * 100:.1f}%")

    if all_passed:
        print("\nAll checks passed.")
    else:
        print("\nSome checks failed. Run with --diagnose for layer-by-layer details.")

    if diagnose:
        print("\n" + "=" * 70)
        print("LOGIT ERROR ANALYSIS")
        print("=" * 70)

        error_distribution("logits", pt_logits, mx_logits)
        print()
        error_distribution("contacts", pt_contacts, mx_contacts)
        print()
        worst_positions("logits", pt_logits, mx_logits, tokens_np)

        pad_mask = tokens_np == PADDING_IDX
        pad_logit_max = float(per_pos_logit[pad_mask].max()) if pad_mask.any() else 0
        print(f"\n  Max logit error (padding):     {pad_logit_max:.6e}")
        print(f"  Max logit error (non-padding): {nonpad_logit_max:.6e}")

        diagnose_layerwise(pt_model, mx_model, batch_tokens)
        print()

    return 0 if all_passed else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Check numerical equivalence between PyTorch ESM2 and the MLX port."
    )
    parser.add_argument("--weights", default="esm2_t33_650M_UR50D.safetensors")
    parser.add_argument("--diagnose", action="store_true",
                        help="Run deep layer-by-layer diagnostics")
    args = parser.parse_args()
    sys.exit(main(args.weights, diagnose=args.diagnose))
