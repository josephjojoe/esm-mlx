"""Convert ESM-2 PyTorch weights to MLX-compatible safetensors.

Downloads checkpoints directly from Meta's CDN and converts the state dict
without instantiating the full PyTorch model — roughly half the memory of
the previous torch.hub.load approach.  Uses mmap when available (PyTorch
>= 2.1) so tensors are paged in from disk on demand.

Usage::

    python3 convert_weights.py --model esm2_t33_650M_UR50D
    python3 convert_weights.py --all
    python3 convert_weights.py --all --upload josephjojoe/esm-mlx
"""

import argparse
import gc
import os

import numpy as np
import torch
from safetensors.numpy import save_file

from esm_mlx import MODEL_CONFIGS

MODEL_URL = "https://dl.fbaipublicfiles.com/fair-esm/models/{}.pt"
REGRESSION_URL = "https://dl.fbaipublicfiles.com/fair-esm/regression/{}-contact-regression.pt"

SKIP_SUFFIXES = (
    "rot_emb.inv_freq",
    "bias_k",
    "bias_v",
)


def _torch_load(path: str):
    """Load a checkpoint, using mmap for memory efficiency when available."""
    try:
        return torch.load(path, map_location="cpu", mmap=True, weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu", weights_only=False)


def _download(url: str) -> str:
    """Download to the torch hub cache, returning the local path."""
    cache_dir = os.path.join(torch.hub.get_dir(), "checkpoints")
    os.makedirs(cache_dir, exist_ok=True)
    filename = os.path.basename(url)
    local = os.path.join(cache_dir, filename)
    if not os.path.exists(local):
        print(f"  Downloading {filename}...")
        torch.hub.download_url_to_file(url, local)
    else:
        print(f"  Cached: {filename}")
    return local


def convert(model_name: str, out_path: str) -> None:
    """Convert a single ESM-2 checkpoint to safetensors."""
    print(f"\nConverting {model_name}...")

    model_path = _download(MODEL_URL.format(model_name))
    data = _torch_load(model_path)
    state_dict = data.get("model", data)
    del data

    try:
        reg_path = _download(REGRESSION_URL.format(model_name))
        reg_data = _torch_load(reg_path)
        state_dict.update(reg_data.get("model", reg_data))
        del reg_data
    except Exception as e:
        print(f"  No regression weights available: {e}")

    gc.collect()

    weights: dict[str, np.ndarray] = {}
    for name, tensor in state_dict.items():
        if name.endswith(SKIP_SUFFIXES):
            continue
        weights[name] = tensor.numpy()
        print(f"  {name}  {tuple(weights[name].shape)}")

    del state_dict
    gc.collect()

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    save_file(weights, out_path)
    size_mb = os.path.getsize(out_path) / (1024 ** 2)
    print(f"  Saved {len(weights)} tensors -> {out_path} ({size_mb:.0f} MB)")

    del weights
    gc.collect()


def upload(repo_id: str, weights_dir: str = "weights") -> None:
    """Upload all .safetensors files to a single HuggingFace repo."""
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(repo_id, exist_ok=True)

    files = sorted(f for f in os.listdir(weights_dir) if f.endswith(".safetensors"))
    if not files:
        print("No .safetensors files found in", weights_dir)
        return

    for f in files:
        path = os.path.join(weights_dir, f)
        size_mb = os.path.getsize(path) / (1024 ** 2)
        print(f"  Uploading {f} ({size_mb:.0f} MB)...")
        api.upload_file(
            path_or_fileobj=path,
            path_in_repo=f,
            repo_id=repo_id,
        )

    print(f"\nDone — https://huggingface.co/{repo_id}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert PyTorch ESM-2 weights to MLX safetensors.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--model", help="Single model name (e.g. esm2_t33_650M_UR50D)")
    group.add_argument("--all", action="store_true",
                       help="Convert all supported models (smallest first)")
    parser.add_argument("--out", default=None,
                        help="Output path (only with --model)")
    parser.add_argument("--upload", metavar="REPO_ID",
                        help="Upload converted weights to HuggingFace repo")
    args = parser.parse_args()

    if args.all:
        models = list(MODEL_CONFIGS)
    else:
        if args.model not in MODEL_CONFIGS:
            parser.error(
                f"Unknown model {args.model!r}. "
                f"Choose from: {list(MODEL_CONFIGS)}"
            )
        models = [args.model]

    for name in models:
        out_path = (
            args.out if (args.out and not args.all)
            else os.path.join("weights", f"{name}.safetensors")
        )
        if os.path.exists(out_path):
            print(f"\nSkipping {name} (already exists: {out_path})")
            continue
        convert(name, out_path)

    if args.upload:
        upload(args.upload)
