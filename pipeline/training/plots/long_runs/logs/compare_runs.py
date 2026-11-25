"""Compare runs side by side: validation loss and image gap per epoch.

usage: python compare_runs.py <out_name> <run> [<run> ...]
e.g.   python compare_runs.py 10k_lr_compare 10k 10k_mlp-lr-1e-4
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

HERE = Path(__file__).parent
OUT, RUNS = sys.argv[1], sys.argv[2:]

LABELS = {
    "10k": "MLP lr 1e-5 (supervisor setting)",
    "10k_mlp-lr-1e-4": "MLP lr 1e-4",
}

SURFACE, TEXT, TEXT_2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984", "#ebeae6"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#6250d6"]  # fixed order, never cycled

plt.rcParams.update({
    "font.family": "Noto Sans", "font.size": 9.5,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": TEXT_2, "axes.labelsize": 9,
    "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False,
    "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID, "grid.linewidth": 1,
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.major.size": 0, "ytick.major.size": 0,
    "xtick.major.pad": 6, "ytick.major.pad": 6,
    "legend.frameon": False, "legend.fontsize": 9, "legend.labelcolor": TEXT_2,
})

data = {r: pd.read_csv(HERE / f"{r}_epochs.csv") for r in RUNS}
max_ep = max(len(d) for d in data.values())

fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
fig.subplots_adjust(left=0.06, right=0.97, top=0.74, bottom=0.14, wspace=0.18)
fig.text(0.06, 0.94, "Effect of the MLP learning rate  ·  10k PixMo, same data and LLM lr",
         fontsize=15, fontweight="bold", color=TEXT)
fig.text(0.06, 0.895, "Each run uses its own cosine schedule, so epoch 5 of a 5-epoch run is the end of its decay.",
         fontsize=9.5, color=MUTED)

panels = [
    ("val_loss", "Validation loss", "lower is better", None),
    ("gap", "Image gap", "wrong-image loss minus right-image loss; higher means more image use", 0),
]
for ax, (col, title, sub, zero) in zip(axes, panels):
    for i, (run, d) in enumerate(data.items()):
        color = SERIES[i]
        ax.plot(d.epoch, d[col], color=color, lw=2, label=LABELS.get(run, run))
        ax.plot(d.epoch, d[col], "o", ms=6, color=color, mec=SURFACE, mew=2, zorder=5)
        best = d[col].idxmin() if col == "val_loss" else d[col].idxmax()
        ax.plot(d.epoch[best], d[col][best], "o", ms=9, color=color, mec=SURFACE, mew=2, zorder=6)
        ax.annotate(f"{d[col][best]:+.3f}" if col == "gap" else f"{d[col][best]:.3f}",
                    (d.epoch[best], d[col][best]), xytext=(0, 11 if col == "gap" else -16),
                    textcoords="offset points", ha="center", fontsize=8.5, color=TEXT)
    if zero is not None:
        ax.axhline(zero, color=MUTED, lw=1)
    ax.set_xticks(range(1, max_ep + 1))
    ax.set_xlim(0.6, max_ep + 0.4)
    ax.set_xlabel("epoch")
    ax.text(0, 1.13, title, transform=ax.transAxes, fontsize=11, fontweight="bold", color=TEXT)
    ax.text(0, 1.05, sub, transform=ax.transAxes, fontsize=8.5, color=MUTED)

axes[1].set_ylim(top=max(d.gap.max() for d in data.values()) * 1.25)
axes[0].legend(loc="upper right", handlelength=1.6)

out = HERE / f"{OUT}.png"
fig.savefig(out, dpi=220)
fig.savefig(out.with_suffix(".pdf"))
print(f"saved {out} and {out.with_suffix('.pdf')}")
