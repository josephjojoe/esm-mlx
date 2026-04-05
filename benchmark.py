import argparse
import csv
import gc
import os
import platform
import statistics
import subprocess
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass, field

import numpy as np

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

WARMUP_ITERS = 10
BENCH_ITERS = 50
SEQUENCE_LENGTHS = [64, 128, 256, 512, 1024]
BATCH_SIZES = [1, 4, 8]
BATCH_SIZES_FP16 = [1, 4, 8, 16, 32]
DTYPE_TAGS = {"float32": "fp32", "float16": "fp16"}
VALID_TOKEN_RANGE = (4, 24)  # amino-acid indices; avoids cls/pad/eos/unk/mask


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class BenchResult:
    backend: str
    batch_size: int
    seq_len: int
    n_iters: int
    times_ms: list = field(repr=False)

    @property
    def median_ms(self):
        return statistics.median(self.times_ms)

    @property
    def mean_ms(self):
        return statistics.mean(self.times_ms)

    @property
    def std_ms(self):
        return statistics.stdev(self.times_ms) if len(self.times_ms) > 1 else 0.0

    @property
    def min_ms(self):
        return min(self.times_ms)

    @property
    def max_ms(self):
        return max(self.times_ms)

    @property
    def p90_ms(self):
        return float(np.percentile(self.times_ms, 90))

    @property
    def p99_ms(self):
        return float(np.percentile(self.times_ms, 99))

    @property
    def tokens_per_sec(self):
        total_tokens = self.batch_size * self.seq_len
        return total_tokens / (self.median_ms / 1000)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_random_tokens(batch_size: int, seq_len: int, rng: np.random.Generator):
    """Build (B, seq_len+2) token arrays with CLS/EOS bookends."""
    lo, hi = VALID_TOKEN_RANGE
    body = rng.integers(lo, hi, size=(batch_size, seq_len), dtype=np.int32)
    cls_col = np.zeros((batch_size, 1), dtype=np.int32)       # <cls> = 0
    eos_col = np.full((batch_size, 1), 2, dtype=np.int32)     # <eos> = 2
    return np.concatenate([cls_col, body, eos_col], axis=1)


@contextmanager
def suspend_gc():
    """Disable GC during timed sections to eliminate collector pauses."""
    was_enabled = gc.isenabled()
    gc.disable()
    try:
        yield
    finally:
        if was_enabled:
            gc.enable()


def system_info() -> dict:
    info = {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "python": platform.python_version(),
    }
    try:
        r = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                           capture_output=True, text=True)
        info["cpu"] = r.stdout.strip()
    except Exception:
        pass
    try:
        r = subprocess.run(["sysctl", "-n", "hw.memsize"],
                           capture_output=True, text=True)
        info["ram_gb"] = round(int(r.stdout.strip()) / (1024 ** 3), 1)
    except Exception:
        pass
    try:
        r = subprocess.run(["system_profiler", "SPDisplaysDataType"],
                           capture_output=True, text=True)
        for line in r.stdout.splitlines():
            if "Chipset Model" in line or "Chip" in line:
                info["gpu"] = line.split(":")[-1].strip()
                break
    except Exception:
        pass
    return info


def sanity_check(a_logits, b_logits, label: str, warn_threshold: float = 1.0):
    """Quick numerical sanity check before benchmarking."""
    diff = np.abs(a_logits - b_logits)
    max_diff = float(diff.max())
    mean_diff = float(diff.mean())
    print(f"  Sanity check ({label}): max_diff={max_diff:.4e}, mean_diff={mean_diff:.4e}")
    if max_diff > warn_threshold:
        print("  WARNING: Large numerical divergence — results may not be comparable.")


# ---------------------------------------------------------------------------
# PyTorch / MPS
# ---------------------------------------------------------------------------

def load_pytorch_model(dtype: str = "float32"):
    import torch

    if not torch.backends.mps.is_available():
        print("ERROR: MPS backend not available.")
        sys.exit(1)

    pt_dtype = {"float32": torch.float32, "float16": torch.float16}[dtype]
    model, _alphabet = torch.hub.load("facebookresearch/esm", "esm2_t33_650M_UR50D")
    model.eval()
    model = model.to(dtype=pt_dtype, device="mps")

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  PyTorch params: {n_params / 1e6:.1f}M  dtype={dtype}")
    return model


def bench_pytorch(model, tokens_np: np.ndarray,
                  warmup: int, iters: int,
                  dtype: str = "float32") -> BenchResult:
    import torch

    tokens = torch.from_numpy(tokens_np).to("mps")
    B, full_len = tokens_np.shape
    seq_len = full_len - 2

    with torch.inference_mode():
        for _ in range(warmup):
            model(tokens)
            torch.mps.synchronize()

        times: list[float] = []
        with suspend_gc():
            for _ in range(iters):
                torch.mps.synchronize()
                t0 = time.perf_counter_ns()
                model(tokens)
                torch.mps.synchronize()
                t1 = time.perf_counter_ns()
                times.append((t1 - t0) / 1e6)

    return BenchResult(f"pt_mps_{DTYPE_TAGS[dtype]}", B, seq_len, iters, times)


def pytorch_logits_np(model, tokens_np: np.ndarray) -> np.ndarray:
    import torch
    tokens = torch.from_numpy(tokens_np).to("mps")
    with torch.inference_mode():
        out = model(tokens)
        torch.mps.synchronize()
    return out["logits"].cpu().float().numpy()


# ---------------------------------------------------------------------------
# MLX
# ---------------------------------------------------------------------------

def load_mlx_model(weights_path: str, dtype: str = "float32"):
    import mlx.core as mx
    from mlx.utils import tree_flatten, tree_map
    from model import ESM2

    model = ESM2()
    weights = mx.load(weights_path)
    model.load_weights(list(weights.items()))

    if dtype == "float16":
        casted = tree_map(
            lambda x: x.astype(mx.float16) if isinstance(x, mx.array) else x,
            model.parameters(),
        )
        model.load_weights(list(tree_flatten(casted)))

    mx.eval(model.parameters())

    n_params = sum(v.size for _, v in tree_flatten(model.parameters()))
    print(f"  MLX params:     {n_params / 1e6:.1f}M  dtype={dtype}")

    model.__call__ = mx.compile(model.__call__, inputs=model.state)
    return model


def bench_mlx(model, tokens_np: np.ndarray,
              warmup: int, iters: int,
              dtype: str = "float32") -> BenchResult:
    import mlx.core as mx

    tokens = mx.array(tokens_np)
    B, full_len = tokens_np.shape
    seq_len = full_len - 2

    for _ in range(warmup):
        out = model(tokens)
        mx.eval(out["logits"])

    times: list[float] = []
    with suspend_gc():
        for _ in range(iters):
            t0 = time.perf_counter_ns()
            out = model(tokens)
            mx.eval(out["logits"])
            t1 = time.perf_counter_ns()
            times.append((t1 - t0) / 1e6)

    return BenchResult(f"mlx_{DTYPE_TAGS[dtype]}", B, seq_len, iters, times)


def mlx_logits_np(model, tokens_np: np.ndarray) -> np.ndarray:
    import mlx.core as mx
    tokens = mx.array(tokens_np)
    out = model(tokens)
    mx.eval(out["logits"])
    return np.array(out["logits"], dtype=np.float32)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

COL = "{:<14} {:>3} {:>6} {:>10} {:>10} {:>8} {:>10} {:>10} {:>10} {:>11}"
SEP = "=" * 93

def print_header():
    print(f"\n{SEP}")
    print(COL.format("Backend", "B", "SeqLen",
                      "Median", "Mean", "Std",
                      "Min", "P90", "P99", "Tok/s"))
    print(SEP)


def print_result(r: BenchResult):
    print(COL.format(
        r.backend, r.batch_size, r.seq_len,
        f"{r.median_ms:.1f}ms", f"{r.mean_ms:.1f}ms", f"{r.std_ms:.1f}ms",
        f"{r.min_ms:.1f}ms", f"{r.p90_ms:.1f}ms", f"{r.p99_ms:.1f}ms",
        f"{r.tokens_per_sec:,.0f}",
    ))


def print_comparison(mlx_r: BenchResult, pt_r: BenchResult):
    speedup = pt_r.median_ms / mlx_r.median_ms
    if speedup >= 1:
        print(f"  -> MLX is {speedup:.2f}x faster  "
              f"(B={mlx_r.batch_size}, L={mlx_r.seq_len})")
    else:
        print(f"  -> PyTorch MPS is {1/speedup:.2f}x faster  "
              f"(B={mlx_r.batch_size}, L={mlx_r.seq_len})")


def print_summary(results: list[BenchResult], batch_sizes, seq_lengths,
                  dtype: str = "float32"):
    dtype_tag = DTYPE_TAGS[dtype]
    mlx_backend = f"mlx_{dtype_tag}"
    pt_backend = f"pt_mps_{dtype_tag}"
    print(f"\n{SEP}")
    print(f"SPEEDUP SUMMARY  (PyTorch-MPS median / MLX median)  [{dtype_tag}]")
    print(SEP)
    for bs in batch_sizes:
        for sl in seq_lengths:
            mlx_r = next((r for r in results
                          if r.backend == mlx_backend
                          and r.batch_size == bs and r.seq_len == sl), None)
            pt_r = next((r for r in results
                         if r.backend == pt_backend
                         and r.batch_size == bs and r.seq_len == sl), None)
            if not mlx_r or not pt_r:
                continue
            speedup = pt_r.median_ms / mlx_r.median_ms
            bar_len = int(min(max(speedup, 0.2), 5.0) * 8)
            bar = "█" * bar_len
            tag = "MLX faster" if speedup >= 1 else "PT faster "
            print(f"  B={bs:>2}  L={sl:>4}  │ MLX {mlx_r.median_ms:>8.1f}ms  "
                  f"PT {pt_r.median_ms:>8.1f}ms  │ {speedup:>5.2f}x {tag} {bar}")
    print()


def save_csv(results: list[BenchResult], path: str):
    fields = ["backend", "batch_size", "seq_len", "n_iters",
              "median_ms", "mean_ms", "std_ms", "min_ms", "max_ms",
              "p90_ms", "p99_ms", "tokens_per_sec"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in results:
            w.writerow({
                "backend": r.backend,
                "batch_size": r.batch_size,
                "seq_len": r.seq_len,
                "n_iters": r.n_iters,
                "median_ms": f"{r.median_ms:.2f}",
                "mean_ms": f"{r.mean_ms:.2f}",
                "std_ms": f"{r.std_ms:.2f}",
                "min_ms": f"{r.min_ms:.2f}",
                "max_ms": f"{r.max_ms:.2f}",
                "p90_ms": f"{r.p90_ms:.2f}",
                "p99_ms": f"{r.p99_ms:.2f}",
                "tokens_per_sec": f"{r.tokens_per_sec:.0f}",
            })
    print(f"Results written to {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Benchmark MLX ESM-2 vs PyTorch ESM-2 (MPS) inference speed.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--weights", default="weights.safetensors",
                        help="Path to converted MLX safetensors")
    parser.add_argument("--warmup", type=int, default=WARMUP_ITERS,
                        help="Warmup iterations (not timed)")
    parser.add_argument("--iters", type=int, default=BENCH_ITERS,
                        help="Timed iterations per configuration")
    parser.add_argument("--seq-lengths", type=int, nargs="+",
                        default=SEQUENCE_LENGTHS, metavar="L")
    parser.add_argument("--batch-sizes", type=int, nargs="+",
                        default=None, metavar="B")
    parser.add_argument("--dtype", choices=["float32", "float16"],
                        default="float32",
                        help="Data type for model weights and computation")
    parser.add_argument("--csv", type=str, default=None,
                        help="Write results to CSV")
    parser.add_argument("--mlx-only", action="store_true")
    parser.add_argument("--pytorch-only", action="store_true")
    parser.add_argument("--no-sanity-check", action="store_true",
                        help="Skip numerical sanity check")
    args = parser.parse_args()

    if args.batch_sizes is None:
        args.batch_sizes = (BATCH_SIZES_FP16 if args.dtype == "float16"
                            else BATCH_SIZES)

    run_mlx = not args.pytorch_only
    run_pt = not args.mlx_only

    # -- System info --------------------------------------------------------
    info = system_info()
    print("System info:")
    for k, v in info.items():
        print(f"  {k}: {v}")

    import_versions = []
    if run_mlx:
        import mlx.core as mx
        import_versions.append(f"mlx {mx.__version__}")
    if run_pt:
        import torch
        import_versions.append(f"torch {torch.__version__}")
    print(f"  frameworks: {', '.join(import_versions)}")

    print(f"\nConfig: warmup={args.warmup}, iters={args.iters}, dtype={args.dtype}")
    print(f"  sequence lengths: {args.seq_lengths}")
    print(f"  batch sizes:      {args.batch_sizes}")

    # -- Load models --------------------------------------------------------
    pt_model = mx_model = None
    if run_pt:
        print("\nLoading PyTorch ESM-2 -> MPS...")
        pt_model = load_pytorch_model(dtype=args.dtype)
    if run_mlx:
        print("Loading MLX ESM-2...")
        mx_model = load_mlx_model(args.weights, dtype=args.dtype)

    # -- Optional sanity check ----------------------------------------------
    if not args.no_sanity_check:
        rng_check = np.random.default_rng(0)
        check_tokens = make_random_tokens(1, 64, rng_check)

        if run_mlx and run_pt:
            mx_lg = mlx_logits_np(mx_model, check_tokens)
            pt_lg = pytorch_logits_np(pt_model, check_tokens)
            threshold = 5.0 if args.dtype == "float16" else 1.0
            sanity_check(mx_lg, pt_lg,
                         f"MLX vs PyTorch, B=1 L=64 {args.dtype}",
                         warn_threshold=threshold)

    # -- Benchmark loop -----------------------------------------------------
    rng = np.random.default_rng(42)
    all_results: list[BenchResult] = []

    print_header()

    for bs in args.batch_sizes:
        for sl in args.seq_lengths:
            tokens_np = make_random_tokens(bs, sl, rng)
            mlx_r = pt_r = None

            if run_mlx:
                mlx_r = bench_mlx(mx_model, tokens_np,
                                  warmup=args.warmup, iters=args.iters,
                                  dtype=args.dtype)
                print_result(mlx_r)
                all_results.append(mlx_r)

            if run_pt:
                pt_r = bench_pytorch(pt_model, tokens_np,
                                     warmup=args.warmup, iters=args.iters,
                                     dtype=args.dtype)
                print_result(pt_r)
                all_results.append(pt_r)

            if mlx_r and pt_r:
                print_comparison(mlx_r, pt_r)
            print()

    # -- Summary & export ---------------------------------------------------
    if run_mlx and run_pt:
        print_summary(all_results, args.batch_sizes, args.seq_lengths,
                      dtype=args.dtype)

    if args.csv:
        save_csv(all_results, args.csv)


if __name__ == "__main__":
    main()
