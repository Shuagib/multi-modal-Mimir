"""Train and validation loss for several runs in one graph, on a shared epoch axis.

usage: python compare_loss.py <out_name> <run> [<run> ...]
e.g.   python compare_loss.py 10k_loss_compare 10k 10k_mlp-lr-1e-4

colour = run, line style = split (solid train, dashed + dots validation)
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D

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

steps = {r: pd.read_csv(HERE / f"{r}_steps.csv") for r in RUNS}
epochs = {r: pd.read_csv(HERE / f"{r}_epochs.csv") for r in RUNS}
max_ep = max(len(d) for d in epochs.values())

fig, ax = plt.subplots(figsize=(10, 5.6))
fig.subplots_adjust(left=0.08, right=0.95, top=0.80, bottom=0.11)
fig.text(0.08, 0.94, "Train vs. validation loss  ·  10k PixMo, MLP lr 1e-5 vs 1e-4",
         fontsize=15, fontweight="bold", color=TEXT)
fig.text(0.08, 0.895, "Colour = run. Solid = train (smoothed over 1,000 steps), dashed with dots = validation "
         "(end of epoch).", fontsize=9.5, color=MUTED)

for i, run in enumerate(RUNS):
    c = SERIES[i]
    s, d = steps[run], epochs[run]

    # train
    x = s.global_step / (s.global_step.max() / s.epoch.max())
    ax.plot(x, s.train_loss, color=c, alpha=0.12, lw=1)
    ax.plot(x, s.train_loss.rolling(10, center=True, min_periods=1).mean(), color=c, lw=2)
    end = s.train_loss.tail(10).mean()
    ax.annotate(f"train {end:.2f}", (x.iloc[-1], end), xytext=(6, -2), textcoords="offset points",
                va="center", fontsize=8.5, color=TEXT)

    # validation
    ax.plot(d.epoch, d.val_loss, color=c, lw=2, ls=(0, (4, 2.5)))
    ax.plot(d.epoch, d.val_loss, "o", ms=6, color=c, mec=SURFACE, mew=2, zorder=5)
    b = d.val_loss.idxmin()
    ax.plot(d.epoch[b], d.val_loss[b], "o", ms=9, color=c, mec=SURFACE, mew=2, zorder=6)
    ax.annotate(f"val {d.val_loss[b]:.3f}", (d.epoch[b], d.val_loss[b]), xytext=(0, 12 if i % 2 == 0 else -18),
                textcoords="offset points", ha="center", fontsize=8.5, color=TEXT)

ax.set_xticks(range(1, max_ep + 1))
ax.set_xlim(0, max_ep + 0.9)
ax.set_xlabel("epoch")
ax.set_ylabel("cross-entropy per answer token")

handles = [Line2D([], [], color=SERIES[i], lw=2.5, label=LABELS.get(r, r)) for i, r in enumerate(RUNS)]
handles += [
    Line2D([], [], color=MUTED, lw=2, label="train"),
    Line2D([], [], color=MUTED, lw=2, ls=(0, (4, 2.5)), marker="o", ms=5, mec=SURFACE, label="validation"),
]
ax.legend(handles=handles, loc="upper right", handlelength=2.4, ncol=2, columnspacing=1.6)

out = HERE / f"{OUT}.png"
fig.savefig(out, dpi=220)
fig.savefig(out.with_suffix(".pdf"))
print(f"saved {out} and {out.with_suffix('.pdf')}")
