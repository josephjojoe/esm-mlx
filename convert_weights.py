"""Convert PyTorch ESM-2 weights to MLX-compatible safetensors format.

Downloads the official model from ``facebookresearch/esm`` via ``torch.hub``
and saves the parameters as NumPy safetensors, stripping buffers that the
MLX implementation recomputes (RoPE frequencies, bias_k/v).

Usage::

    python convert_weights.py --model esm2_t33_650M_UR50D --out esm2_t33_650M_UR50D.safetensors
"""

import argparse

import numpy as np
import torch
from safetensors.numpy import save_file

from esm_mlx import MODEL_CONFIGS

# Keys to skip: RoPE buffers recomputed by MLX, and unused bias projections.
SKIP_SUFFIXES = (
    "rot_emb.inv_freq",
    "bias_k",
    "bias_v",
)


def load_fair_model(model_name: str):
    """Download and return a PyTorch ESM-2 model from torch.hub."""
    model, alphabet = torch.hub.load("facebookresearch/esm", model_name)
    model.eval()
    return model


def convert(model_name: str, out_path: str) -> None:
    """Convert a PyTorch ESM-2 checkpoint to safetensors."""
    print(f"Loading {model_name}...")
    model = load_fair_model(model_name)

    weights: dict[str, np.ndarray] = {}
    for name, param in model.named_parameters():
        if name.endswith(SKIP_SUFFIXES):
            print(f"  skip: {name}")
            continue

        arr = param.detach().cpu().numpy()
        weights[name] = arr
        print(f"  {name} {arr.shape}")

    print(f"\nSaving {len(weights)} tensors to {out_path}")
    save_file(weights, out_path)
    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert PyTorch ESM-2 weights to MLX safetensors format."
    )
    parser.add_argument(
        "--model",
        default="esm2_t33_650M_UR50D",
        help="Model name from facebookresearch/esm (default: esm2_t33_650M_UR50D)",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Output path (default: <model>.safetensors)",
    )
    args = parser.parse_args()

    if args.model not in MODEL_CONFIGS:
        parser.error(
            f"Unknown model {args.model!r}. "
            f"Choose from: {list(MODEL_CONFIGS)}"
        )

    out_path = args.out or f"{args.model}.safetensors"
    convert(args.model, out_path)
