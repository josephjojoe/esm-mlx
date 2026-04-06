"""Generate the benchmark chart used in the README and social posts."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch


LABELS = ["Batch 8", "Batch 16", "Batch 32"]
MLX_MS = [2186.3, 4349.1, 8678.4]
PYTORCH_MPS_MS = [5266.7, 14759.5, None]

MLX_COLOR = "#2563eb"
PYTORCH_COLOR = "#111111"
OOM_COLOR = "#dc2626"
TEXT_BLACK = "#111111"
SUBTITLE_COLOR = "#4b5563"


def make_chart(output_path: Path) -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["SF Pro Display", "Helvetica Neue", "Arial"],
        }
    )

    fig, ax = plt.subplots(figsize=(9, 5))
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#ffffff")

    for spine in ax.spines.values():
        spine.set_visible(False)

    x = np.arange(len(LABELS))
    bar_width = 0.30
    gap = 0.04

    for i, val in enumerate(PYTORCH_MPS_MS):
        x_pos = x[i] - bar_width / 2 - gap / 2
        if val is not None:
            ax.bar(x_pos, val, bar_width, color=PYTORCH_COLOR, edgecolor="none")
            ax.text(
                x_pos,
                val + 180,
                f"{val / 1000:.1f}s",
                ha="center",
                va="bottom",
                fontsize=11.5,
                color=TEXT_BLACK,
                fontweight="medium",
            )
        else:
            ax.text(
                x_pos,
                520,
                "OOM",
                ha="center",
                va="bottom",
                fontsize=15,
                fontweight="bold",
                color=OOM_COLOR,
            )

    for i, val in enumerate(MLX_MS):
        x_pos = x[i] + bar_width / 2 + gap / 2
        ax.bar(x_pos, val, bar_width, color=MLX_COLOR, edgecolor="none")
        ax.text(
            x_pos,
            val + 180,
            f"{val / 1000:.1f}s",
            ha="center",
            va="bottom",
            fontsize=11.5,
            color=MLX_COLOR,
            fontweight="medium",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(LABELS, color=TEXT_BLACK, fontsize=13, fontweight="medium")
    ax.tick_params(axis="x", length=0, pad=10)
    ax.tick_params(axis="y", left=False, labelleft=False)
    ax.set_ylim(0, 16250)

    fig.suptitle(
        "MLX is up to 3.4x faster than PyTorch MPS",
        color=TEXT_BLACK,
        fontsize=18,
        fontweight="bold",
        y=0.965,
    )
    fig.text(
        0.5,
        0.885,
        "ESM-2 650M · FP16 · Sequence Length 1024",
        ha="center",
        va="center",
        fontsize=11.5,
        color=SUBTITLE_COLOR,
    )

    ax.legend(
        handles=[
            Patch(facecolor=MLX_COLOR, label="MLX"),
            Patch(facecolor=PYTORCH_COLOR, label="PyTorch MPS"),
        ],
        loc="upper left",
        fontsize=12,
        frameon=False,
        labelcolor=TEXT_BLACK,
        handlelength=1.0,
        handleheight=0.8,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=[0, 0, 1, 0.89], pad=1.5)
    fig.savefig(
        output_path,
        dpi=220,
        facecolor="#ffffff",
        bbox_inches="tight",
        pad_inches=0.35,
    )
    plt.close(fig)


if __name__ == "__main__":
    make_chart(Path("figures/benchmark_chart_fp16_seq1024.png"))
