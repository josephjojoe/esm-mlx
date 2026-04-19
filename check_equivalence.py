"""
Check numerical equivalence between PyTorch ESM2 and the MLX port.

Requires PyTorch ESM2 (downloaded via torch.hub) and converted MLX weights.

Usage:
    python3 check_equivalence.py [--diagnose]
    python3 check_equivalence.py --dtype float16 [--diagnose]
"""

import argparse
import sys

import numpy as np
import torch

import mlx.core as mx

from esm_mlx import ESM2
from esm_mlx.model import _canonicalise_weights

PADDING_IDX = 1


def load_pytorch_model():
    model, alphabet = torch.hub.load(
        "facebookresearch/esm", "esm2_t33_650M_UR50D"
    )
    model.eval()
    batch_converter = alphabet.get_batch_converter()
    return model, alphabet, batch_converter


def load_mlx_model(weights_path, dtype="float32"):
    from mlx.utils import tree_flatten, tree_map

    model = ESM2()
    raw = mx.load(weights_path)
    weights = _canonicalise_weights(raw.items())
    model.load_weights(list(weights.items()))

    if dtype == "float16":
        casted = tree_map(
            lambda x: x.astype(mx.float16) if isinstance(x, mx.array) else x,
            model.parameters(),
        )
        model.load_weights(list(tree_flatten(casted)))

    mx.eval(model.parameters())
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

    mx_out = mx_model(mx_tokens, repr_layers=all_layers)
    mx.eval(mx_out["logits"])
    mx_reprs = mx_out["representations"]

    header = (f"  {'Stage':<16} {'max_diff':>12} {'mean_diff':>12}"
              f"  {'nonpad_max':>12} {'rel_l2':>10}")
    print(f"\n{header}")
    print("  " + "-" * 68)

    prev_max = 0
    for layer_idx in range(pt_model.num_layers):
        pt_np = to_np(pt_reprs[layer_idx])
        mx_np = to_np(mx_reprs[layer_idx])
        diff = np.abs(pt_np - mx_np)
        max_d = float(np.max(diff))
        mean_d = float(np.mean(diff))

        per_pos = diff.max(axis=-1)
        nonpad_max = float(per_pos[non_pad].max()) if non_pad.any() else max_d

        diff_norm = np.linalg.norm(pt_np - mx_np, axis=-1)
        ref_norm = np.linalg.norm(pt_np, axis=-1)
        rel_l2_per_pos = diff_norm / (ref_norm + 1e-8)
        rel_l2 = float(rel_l2_per_pos[non_pad].max()) if non_pad.any() else float(rel_l2_per_pos.max())

        growth = ""
        if prev_max > 0 and max_d > prev_max:
            growth = f" ({max_d / prev_max:.1f}x)"
        prev_max = max_d if max_d > 0 else prev_max

        label = "embed" if layer_idx == 0 else f"layer {layer_idx}"
        print(f"  {label:<16} {max_d:>12.6e} {mean_d:>12.6e}"
              f"  {nonpad_max:>12.6e} {rel_l2:>9.4e}{growth}")

    for stage_key, stage_label in [(pt_model.num_layers, "post_norm"), ("logits", "logits")]:
        if stage_label == "logits":
            pt_np = to_np(pt_logits)
            mx_np = to_np(mx_out["logits"])
        else:
            pt_np = to_np(pt_reprs[stage_key])
            mx_np = to_np(mx_reprs[stage_key])
        diff = np.abs(pt_np - mx_np)
        max_d, mean_d = float(np.max(diff)), float(np.mean(diff))
        nonpad_max = float(diff.max(axis=-1)[non_pad].max()) if non_pad.any() else max_d
        diff_norm = np.linalg.norm(pt_np - mx_np, axis=-1)
        ref_norm = np.linalg.norm(pt_np, axis=-1)
        rel_l2_per_pos = diff_norm / (ref_norm + 1e-8)
        rel_l2 = float(rel_l2_per_pos[non_pad].max()) if non_pad.any() else float(rel_l2_per_pos.max())
        print(f"  {stage_label:<16} {max_d:>12.6e} {mean_d:>12.6e}"
              f"  {nonpad_max:>12.6e} {rel_l2:>9.4e}")


# ── Main ─────────────────────────────────────────────────────────────────────

TOLERANCES = {
    "float32": {"logits": 5e-3, "contacts": 1e-3},
    "float16": {"logits": 1.0, "contacts": 0.05},
}


def check_logits(pt_logits, mx_logits, non_pad, atol, label):
    """Compare logits and return (passed, max_diff, mean_diff)."""
    diff = np.abs(pt_logits - mx_logits)
    per_pos = diff.max(axis=-1)
    max_d = float(per_pos[non_pad].max())
    mean_d = float(per_pos[non_pad].mean())
    ok = max_d < atol
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] {label}: max_diff={max_d:.6e}, mean_diff={mean_d:.6e} (atol={atol})")
    return ok, per_pos


def main(weights_path, diagnose=False, dtype="float32"):
    print(f"Loading PyTorch model (fp32 reference)...")
    pt_model, alphabet, batch_converter = load_pytorch_model()

    print(f"Loading MLX model ({dtype})...")
    mx_model = load_mlx_model(weights_path, dtype=dtype)

    test_sequences = [
        ("protein1", "MKTAYIAKQRQISFVKSHFSRQLE"),
        ("protein2", "KALTARQQEVFDLIRD"),
    ]

    _, _, batch_tokens = batch_converter(test_sequences)
    mx_tokens = mx.array(batch_tokens.numpy())
    tokens_np = batch_tokens.numpy()
    non_pad = tokens_np != PADDING_IDX

    print(f"\nInput shape: {batch_tokens.shape}")

    atol_logits = TOLERANCES[dtype]["logits"]
    atol_contacts = TOLERANCES[dtype]["contacts"]

    # ── Pass 1: SDPA path (normal inference) ─────────────────────────────
    print("\n--- Pass 1: fused SDPA path (return_contacts=False) ---")

    with torch.no_grad():
        pt_out_sdpa = pt_model(batch_tokens, return_contacts=False)
    mx_out_sdpa = mx_model(mx_tokens, return_contacts=False)
    mx.eval(mx_out_sdpa["logits"])

    pt_logits_sdpa = to_np(pt_out_sdpa["logits"])
    mx_logits_sdpa = to_np(mx_out_sdpa["logits"])

    sdpa_ok, _ = check_logits(
        pt_logits_sdpa, mx_logits_sdpa, non_pad, atol_logits, "logits (sdpa)",
    )

    # ── Pass 2: manual attention path (contact prediction) ───────────────
    print("\n--- Pass 2: manual attention path (return_contacts=True) ---")

    with torch.no_grad():
        pt_out = pt_model(batch_tokens, return_contacts=True)
    mx_out = mx_model(mx_tokens, return_contacts=True)
    mx.eval(mx_out["logits"], mx_out["contacts"])

    pt_logits = to_np(pt_out["logits"])
    mx_logits = to_np(mx_out["logits"])
    pt_contacts = to_np(pt_out["contacts"])
    mx_contacts = to_np(mx_out["contacts"])

    attn_logits_ok, per_pos_logit = check_logits(
        pt_logits, mx_logits, non_pad, atol_logits, "logits (attn)",
    )

    contact_diff = np.abs(pt_contacts - mx_contacts)
    contact_max = float(contact_diff.max())
    contact_mean = float(contact_diff.mean())
    contacts_ok = contact_max < atol_contacts
    status = "PASS" if contacts_ok else "FAIL"
    print(f"  [{status}] contacts: max_diff={contact_max:.6e}, mean_diff={contact_mean:.6e} (atol={atol_contacts})")

    # ── Summary ──────────────────────────────────────────────────────────
    tk = topk_agreement(pt_logits_sdpa, mx_logits_sdpa)
    pt_preds = np.argmax(pt_logits_sdpa, axis=-1)
    mx_preds = np.argmax(mx_logits_sdpa, axis=-1)
    token_match = float(np.mean(pt_preds[non_pad] == mx_preds[non_pad]))

    print(f"\n  Token prediction agreement: {token_match * 100:.1f}%")
    for k, v in tk.items():
        print(f"  Top-{k} agreement: {v * 100:.1f}%")

    all_passed = sdpa_ok and attn_logits_ok and contacts_ok

    if all_passed:
        print("\nAll checks passed.")
    else:
        print("\nSome checks failed. Run with --diagnose for layer-by-layer details.")

    if diagnose:
        print("\n" + "=" * 70)
        print("LOGIT ERROR ANALYSIS")
        print("=" * 70)

        error_distribution("logits (attn path)", pt_logits, mx_logits)
        print()
        error_distribution("contacts", pt_contacts, mx_contacts)
        print()
        worst_positions("logits", pt_logits, mx_logits, tokens_np)

        pad_mask = tokens_np == PADDING_IDX
        pad_logit_max = float(per_pos_logit[pad_mask].max()) if pad_mask.any() else 0
        print(f"\n  Max logit error (padding):     {pad_logit_max:.6e}")
        print(f"  Max logit error (non-padding): {float(per_pos_logit[non_pad].max()):.6e}")

        diagnose_layerwise(pt_model, mx_model, batch_tokens)
        print()

    return 0 if all_passed else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Check numerical equivalence between PyTorch ESM2 and the MLX port."
    )
    parser.add_argument("--weights", default="weights/esm2_t33_650M_UR50D.safetensors")
    parser.add_argument("--dtype", choices=["float32", "float16"], default="float32",
                        help="Data type for MLX model (PyTorch reference always runs fp32)")
    parser.add_argument("--diagnose", action="store_true",
                        help="Run deep layer-by-layer diagnostics")
    args = parser.parse_args()
    sys.exit(main(args.weights, diagnose=args.diagnose, dtype=args.dtype))
