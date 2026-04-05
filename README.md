# esm-mlx

ESM-2 protein language model running natively on Apple Silicon via [MLX](https://github.com/ml-explore/mlx).

This is a from-scratch MLX implementation of Meta's [ESM-2](https://github.com/facebookresearch/esm) (Evolutionary Scale Modeling), the state-of-the-art protein language model. It supports masked language modeling and residue–residue contact prediction, and runs **1.2–1.8x faster** than PyTorch on MPS on the same hardware.

## Supported Models

| Model | Layers | Hidden Dim | Heads | Parameters |
|-------|--------|-----------|-------|------------|
| `esm2_t33_650M_UR50D` | 33 | 1280 | 20 | 650M |
| `esm2_t36_3B_UR50D` | 36 | 2560 | 40 | 3B |

## Installation

Requires Python 3.10+ and an Apple Silicon Mac.

```bash
git clone https://github.com/jojoe-ainmo/esm-mlx.git
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
python convert_weights.py --model esm2_t33_650M_UR50D
```

This downloads the model from `torch.hub` and saves `esm2_t33_650M_UR50D.safetensors` in the current directory. For the 3B model:

```bash
python convert_weights.py --model esm2_t36_3B_UR50D
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
python check_equivalence.py
```

Add `--diagnose` for layer-by-layer error analysis:

```bash
python check_equivalence.py --diagnose
```

On non-padding positions, typical max error is ~3e-3 (normal float32 drift across 33 layers on different backends).

## Benchmarks

Inference latency for ESM-2 650M on Apple Silicon (M-series), MLX vs PyTorch MPS, fp32. Each cell shows median latency in ms over 50 iterations after 10 warmup passes.

### Median Latency (ms)

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

The MLX advantage grows with batch size and sequence length. At batch=8, seq=1024 the MLX port is **1.77x faster** than PyTorch on MPS.

Run your own benchmarks:

```bash
python benchmark.py --csv results.csv
```

See `python benchmark.py --help` for options (dtype, batch sizes, sequence lengths, etc.).

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
