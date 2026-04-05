import argparse
import torch
import numpy as np
from safetensors.numpy import save_file

# Keys to skip: tied weights, RoPE buffers, non-parameters
SKIP_KEYS = set()

SKIP_SUFFIXES = (
    "rot_emb.inv_freq",  # MLX RoPE computes its own
    "bias_k",
    "bias_v",
)


def load_fair_model(model_name):
    model, alphabet = torch.hub.load(
        "facebookresearch/esm", model_name
    )
    model.eval()
    return model


def convert(model_name, out_path):
    print(f"Loading {model_name}...")
    model = load_fair_model(model_name)

    weights = {}
    for name, param in model.named_parameters():
        if name in SKIP_KEYS or name.endswith(SKIP_SUFFIXES):
            print(f"  skip: {name}")
            continue

        arr = param.detach().cpu().numpy()
        weights[name] = arr
        print(f"  {name} {arr.shape}")

    # PyTorch deduplicates tied params; re-add lm_head.weight from embed_tokens
    if "lm_head.weight" not in weights and "embed_tokens.weight" in weights:
        weights["lm_head.weight"] = weights["embed_tokens.weight"].copy()

    print(f"\nSaving {len(weights)} tensors to {out_path}")
    save_file(weights, out_path)
    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="esm2_t33_650M_UR50D")
    parser.add_argument("--out", default="weights.safetensors")
    args = parser.parse_args()
    convert(args.model, args.out)