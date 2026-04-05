# esm-mlx

ESM-2 protein language model running natively on Apple Silicon via [MLX](https://github.com/ml-explore/mlx).

This is a from-scratch MLX implementation of Meta's [ESM-2](https://github.com/facebookresearch/esm) (Evolutionary Scale Modeling), the state-of-the-art protein language model. It supports masked language modeling and residue–residue contact prediction, and runs **1.2–3.4x faster** than PyTorch on MPS on the same hardware.

## Supported Models

| Model | Layers | Hidden Dim | Heads | Parameters |
|-------|--------|-----------|-------|------------|
| `esm2_t6_8M_UR50D` | 6 | 320 | 20 | 8M |
| `esm2_t12_35M_UR50D` | 12 | 480 | 20 | 35M |
| `esm2_t30_150M_UR50D` | 30 | 640 | 20 | 150M |
| `esm2_t33_650M_UR50D` | 33 | 1280 | 20 | 650M |
| `esm2_t36_3B_UR50D` | 36 | 2560 | 40 | 3B |
| `esm2_t48_15B_UR50D` | 48 | 5120 | 40 | 15B |

## Installation

Requires Python 3.10+ and an Apple Silicon Mac.

```bash
git clone https://github.com/josephjojoe/esm-mlx.git
cd esm-mlx
pip install -e .
```

To also install PyTorch (needed for weight conversion, equivalence checking, and benchmarks):

```bash
pip install -e ".[convert]"
```

## Getting Weights

Convert the official PyTorch weights to MLX-compatible safetensors:

```bash
python3 convert_weights.py --model esm2_t33_650M_UR50D
```

This downloads the model from `torch.hub` and saves `weights/esm2_t33_650M_UR50D.safetensors`. For the 3B model:

```bash
python3 convert_weights.py --model esm2_t36_3B_UR50D
```

## Quick Start

```python
import mlx.core as mx
from esm_mlx import ESM2, Tokenizer

# Load the model
model = ESM2.from_pretrained("esm2_t33_650M_UR50D")

# Tokenize a protein sequence
tok = Tokenizer()
tokens = tok.encode("MKTAYIAKQRQISFVKSHFSRQLE")

# Run inference
out = model(tokens)
logits = out["logits"]  # (1, seq_len, vocab_size)

# Predict contacts
out = model(tokens, return_contacts=True)
contacts = out["contacts"]  # (1, seq_len, seq_len)
```

### Batch Inference

```python
sequences = [
    "MKTAYIAKQRQISFVKSHFSRQLE",
    "KALTARQQEVFDLIRD",
]

tokens = tok.batch_encode(sequences)  # auto-pads shorter sequences
out = model(tokens)
```

### Compiled Inference

For maximum throughput, compile the model forward pass:

```python
model.__call__ = mx.compile(model.__call__, inputs=model.state)
```

## Equivalence Checking

Verify numerical equivalence against the official PyTorch implementation:

```bash
python3 check_equivalence.py
```

Add `--diagnose` for layer-by-layer error analysis:

```bash
python3 check_equivalence.py --diagnose
```

To verify FP16 faithfulness (looser tolerances, comparing fp16 MLX against fp32 PyTorch):

```bash
python3 check_equivalence.py --dtype float16
```

On non-padding positions, typical max error is ~3e-3 for fp32 (normal float32 drift across 33 layers on different backends).

## Benchmarks

All benchmarks: ESM-2 650M on M2 Pro (16 GB), MLX 0.30.6 vs PyTorch 2.10.0 MPS. Median latency over 50 iterations after 10 warmup passes.

### Float32

| Batch | Seq Len | MLX | PyTorch MPS | Speedup |
|-------|---------|-----|-------------|---------|
| 1 | 64 | 43.9 | 43.7 | 1.00x |
| 1 | 128 | 63.0 | 66.9 | 1.06x |
| 1 | 256 | 104.1 | 129.4 | 1.24x |
| 1 | 512 | 190.4 | 242.1 | 1.27x |
| 1 | 1024 | 378.5 | 511.1 | 1.35x |
| 4 | 256 | 340.7 | 462.8 | 1.36x |
| 4 | 512 | 670.0 | 1005.5 | 1.50x |
| 4 | 1024 | 1409.3 | 2439.9 | 1.73x |
| 8 | 256 | 646.6 | 930.1 | 1.44x |
| 8 | 512 | 1305.0 | 2057.8 | 1.58x |
| 8 | 1024 | 2783.9 | 4935.6 | **1.77x** |

### Float16

| Batch | Seq Len | MLX | PyTorch MPS | Speedup |
|-------|---------|-----|-------------|---------|
| 8 | 64 | 149.6 | 218.2 | 1.46x |
| 8 | 128 | 273.3 | 414.8 | 1.52x |
| 8 | 256 | 522.1 | 868.0 | 1.66x |
| 8 | 512 | 1039.9 | 2012.4 | 1.94x |
| 8 | 1024 | 2186.3 | 5266.7 | 2.41x |
| 16 | 256 | 1006.8 | 1738.5 | 1.73x |
| 16 | 512 | 2051.9 | 4123.9 | 2.01x |
| 16 | 1024 | 4349.1 | 14759.5 | **3.39x** |
| 32 | 256 | 1985.4 | 3573.3 | 1.80x |
| 32 | 512 | 4081.2 | 8295.5 | 2.03x |
| 32 | 1024 | 8678.4 | OOM | — |

FP16 widens the gap significantly. The **3.39x** result at batch=16, seq=1024 likely reflects PyTorch MPS thrashing near its memory ceiling — it OOMs entirely one step later at batch=32. MLX's unified-memory allocation avoids this cliff and continues to scale linearly up to batch=192 at seq=1024, sustaining ~3,784 tok/s on 16 GB.

Run your own benchmarks:

```bash
python3 benchmark.py --csv
python3 benchmark.py --dtype float16 --csv
```

See `python3 benchmark.py --help` for options (dtype, batch sizes, sequence lengths, etc.).

## API Reference

### `ESM2`

The main model class.

- **`ESM2(num_layers, embed_dim, attention_heads, ...)`** — Construct with explicit config.
- **`ESM2.from_pretrained(model_name, weights_path=None)`** — Load a pretrained model by name.
- **`model(tokens, return_contacts=False)`** — Forward pass. Returns `{"logits": ...}`, or `{"logits": ..., "contacts": ...}` when `return_contacts=True`.
- **`model.predict_contacts(tokens)`** — Shorthand for contact prediction only.

### `Tokenizer`

Standalone tokenizer (no PyTorch dependency).

- **`tok.encode(sequence)`** — Single sequence to `mx.array` of shape `(1, L+2)`.
- **`tok.batch_encode(sequences)`** — Batch of sequences with padding, shape `(B, max_L+2)`.
- **`tok.decode(indices)`** — Convert token indices back to token strings.

### `MODEL_CONFIGS`

Dict mapping model names to architecture parameters:

```python
from esm_mlx import MODEL_CONFIGS
print(MODEL_CONFIGS["esm2_t33_650M_UR50D"])
# {'num_layers': 33, 'embed_dim': 1280, 'attention_heads': 20}
```

## Architecture

The implementation mirrors the original ESM-2 architecture:

- **Embedding** with 0.88 scale factor (compensates for removed token dropout)
- **Pre-LayerNorm** transformer blocks with GELU FFN
- **Rotary Position Embeddings** (RoPE) via `mlx.nn.RoPE`
- **Fused SDPA** via `mx.fast.scaled_dot_product_attention` (when attention weights aren't needed)
- **RobertaLMHead** with weight-tied output projection
- **ContactPredictionHead** with symmetrization and average product correction (APC)

## Project Structure

```
esm-mlx/
├── esm_mlx/
│   ├── __init__.py           # Public API exports
│   ├── model.py              # ESM2 model class and configs
│   ├── layers.py             # MultiHeadAttention, TransformerLayer
│   ├── heads.py              # RobertaLMHead, ContactPredictionHead
│   └── tokenizer.py          # Standalone ESM-2 tokenizer
├── convert_weights.py        # PyTorch → safetensors converter
├── check_equivalence.py      # Numerical equivalence tests
├── benchmark.py              # MLX vs PyTorch MPS benchmarks
├── pyproject.toml            # Package metadata and dependencies
└── LICENSE                   # MIT license
```

## License

MIT
