import argparse
import numpy as np
import torch

import mlx.core as mx

from model import ESM2


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


def compare(name, pt_tensor, mx_array, atol=1e-4):
    pt_np = pt_tensor.detach().cpu().numpy()
    mx_np = np.array(mx_array)
    max_diff = np.max(np.abs(pt_np - mx_np))
    mean_diff = np.mean(np.abs(pt_np - mx_np))
    passed = max_diff < atol
    status = "PASS" if passed else "FAIL"
    print(f"  [{status}] {name}: max_diff={max_diff:.6e}, mean_diff={mean_diff:.6e}")
    return passed


def main(weights_path):
    print("Loading PyTorch model...")
    pt_model, alphabet, batch_converter = load_pytorch_model()

    print("Loading MLX model...")
    mx_model = load_mlx_model(weights_path)

    # Prepare test input
    test_sequences = [
        ("protein1", "MKTAYIAKQRQISFVKSHFSRQLE"),
        ("protein2", "KALTARQQEVFDLIRD"),
    ]

    _, _, batch_tokens = batch_converter(test_sequences)

    print(f"\nInput shape: {batch_tokens.shape}")
    print(f"Tokens: {batch_tokens}")

    # Run PyTorch
    print("\nRunning PyTorch model...")
    with torch.no_grad():
        pt_out = pt_model(batch_tokens, return_contacts=True)

    # Run MLX
    print("Running MLX model...")
    mx_tokens = mx.array(batch_tokens.numpy())
    mx_out = mx_model(mx_tokens, return_contacts=True)
    mx.eval(mx_out["logits"], mx_out["contacts"])

    # Compare logits
    print("\nResults:")
    all_passed = True
    all_passed &= compare("logits", pt_out["logits"], mx_out["logits"])
    all_passed &= compare("contacts", pt_out["contacts"], mx_out["contacts"])

    # Spot check: do the top predicted tokens match?
    pt_preds = pt_out["logits"].argmax(-1)
    mx_preds = np.array(mx.argmax(mx_out["logits"], axis=-1))
    pt_preds_np = pt_preds.numpy()
    token_match = np.mean(pt_preds_np == mx_preds)
    print(f"\n  Token prediction agreement: {token_match * 100:.1f}%")

    if all_passed:
        print("\nAll tests passed!")
    else:
        print("\nSome tests failed — check tolerances or debug layer by layer.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default="weights.safetensors")
    args = parser.parse_args()
    main(args.weights)