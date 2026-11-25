"""Overview figure for a long training run: headline numbers + loss, image gap, judge, LR.

usage: python plot_run.py 10k      (reads 10k_steps.csv and 10k_epochs.csv next to this file)
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.gridspec import GridSpec

HERE = Path(__file__).parent
RUN = sys.argv[1] if len(sys.argv) > 1 else "10k"

steps = pd.read_csv(HERE / f"{RUN}_steps.csv")
epochs = pd.read_csv(HERE / f"{RUN}_epochs.csv")

# design tokens (light)
SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_2 = "#52514e"
MUTED = "#8a8984"
GRID = "#ebeae6"
TRAIN = "#2a78d6"   # series 1 - blue
VAL = "#eb6834"     # series 2 - orange
GAP = "#1baf7a"     # aqua
JUDGE = "#6250d6"   # violet
LR = "#52514e"      # neutral

plt.rcParams.update({
    "font.family": "Noto Sans",
    "font.size": 9.5,
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID,
    "axes.linewidth": 1,
    "axes.labelcolor": TEXT_2,
    "axes.labelsize": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.spines.left": False,
    "axes.grid": True,
    "axes.grid.axis": "y",
    "grid.color": GRID,
    "grid.linewidth": 1,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "xtick.major.size": 0,
    "ytick.major.size": 0,
    "xtick.major.pad": 6,
    "ytick.major.pad": 6,
    "legend.frameon": False,
    "legend.fontsize": 8.5,
    "legend.labelcolor": TEXT_2,
    "lines.solid_capstyle": "round",
})

best_val = epochs.loc[epochs.val_loss.idxmin()]
peak_gap = epochs.loc[epochs.gap.idxmax()]
final_train = steps.train_loss.tail(10).mean()
peak_judge = epochs[["judge_acc", "judge_cov"]].max(axis=1).max()
x_end = epochs.global_step.max()


def card_title(ax, title, sub):
    ax.text(0, 1.13, title, transform=ax.transAxes, fontsize=11, fontweight="bold", color=TEXT, va="bottom")
    ax.text(0, 1.04, sub, transform=ax.transAxes, fontsize=8.5, color=MUTED, va="bottom")


def epoch_axis(ax):
    ax.set_xticks(epochs.global_step)
    ax.set_xticklabels(epochs.epoch)
    ax.set_xlim(0, x_end * 1.02)
    ax.set_xlabel("epoch")


def dot(ax, x, y, color, size=7):
    ax.plot(x, y, "o", ms=size, color=color, mec=SURFACE, mew=2, zorder=5)


fig = plt.figure(figsize=(12, 8.6))
gs = GridSpec(3, 2, figure=fig, height_ratios=[0.42, 1, 1], hspace=0.62, wspace=0.18,
              left=0.06, right=0.97, top=0.88, bottom=0.07)

fig.text(0.06, 0.955, f"Mimir × SigLIP 2  ·  full fine-tune on {RUN} PixMo (cap, count, cap-qa)",
         fontsize=16, fontweight="bold", color=TEXT)
fig.text(0.06, 0.925, f"{len(epochs)} epochs  ·  {x_end:,} steps  ·  AdamAtan2, lr 1e-5, cosine to 10%",
         fontsize=9.5, color=MUTED)

# headline numbers
stats = [
    ("Peak image gap", f"+{peak_gap.gap:.3f}", f"epoch {int(peak_gap.epoch)}", GAP),
    ("Best validation loss", f"{best_val.val_loss:.3f}", f"epoch {int(best_val.epoch)}", VAL),
    ("Final train loss", f"{final_train:.2f}", "mean of last 1,000 steps", TRAIN),
    ("Peak judge score", f"{peak_judge:.1f}", "out of 10", JUDGE),
]
top = gs[0, :].subgridspec(1, 4, wspace=0.08)
for i, (label, value, sub, color) in enumerate(stats):
    ax = fig.add_subplot(top[0, i])
    ax.axis("off")
    ax.add_patch(plt.Rectangle((0, 0), 1, 1, transform=ax.transAxes, facecolor="#f3f2ef",
                               edgecolor="none", zorder=0))
    ax.add_patch(plt.Rectangle((0, 0), 0.012, 1, transform=ax.transAxes, facecolor=color,
                               edgecolor="none", zorder=1))
    ax.text(0.07, 0.72, label, transform=ax.transAxes, fontsize=8.5, color=TEXT_2, va="center")
    ax.text(0.07, 0.40, value, transform=ax.transAxes, fontsize=21, fontweight="bold", color=TEXT, va="center")
    ax.text(0.07, 0.13, sub, transform=ax.transAxes, fontsize=8, color=MUTED, va="center")

# loss
ax = fig.add_subplot(gs[1, 0])
smooth = steps.train_loss.rolling(10, center=True, min_periods=1).mean()
ax.plot(steps.global_step, steps.train_loss, color=TRAIN, alpha=0.18, lw=1)
ax.plot(steps.global_step, smooth, color=TRAIN, lw=2, label="train (smoothed)")
ax.plot(epochs.global_step, epochs.val_loss, color=VAL, lw=2, label="validation")
for _, r in epochs.iterrows():
    dot(ax, r.global_step, r.val_loss, VAL, 6)
dot(ax, best_val.global_step, best_val.val_loss, VAL, 9)
ax.annotate(f"best {best_val.val_loss:.3f}", (best_val.global_step, best_val.val_loss),
            xytext=(0, 12), textcoords="offset points", ha="center", fontsize=8.5, color=TEXT)
ax.legend(loc="upper right", handlelength=1.6)
card_title(ax, "Loss", "cross-entropy per answer token; train flattens, validation holds")
epoch_axis(ax)

# image gap
ax = fig.add_subplot(gs[1, 1])
ax.fill_between(epochs.global_step, 0, epochs.gap, color=GAP, alpha=0.12, lw=0)
ax.plot(epochs.global_step, epochs.gap, color=GAP, lw=2)
for _, r in epochs.iterrows():
    dot(ax, r.global_step, r.gap, GAP, 6)
dot(ax, peak_gap.global_step, peak_gap.gap, GAP, 9)
ax.annotate(f"+{peak_gap.gap:.3f}", (peak_gap.global_step, peak_gap.gap),
            xytext=(0, 12), textcoords="offset points", ha="center", fontsize=8.5, color=TEXT)
ax.axhline(0, color=MUTED, lw=1)
ax.set_ylim(min(0, epochs.gap.min()) - 0.01, peak_gap.gap * 1.25)
ax.set_ylabel("Δ loss")
card_title(ax, "Image gap", "loss with a mismatched image minus loss with the right one")
epoch_axis(ax)

# judge
ax = fig.add_subplot(gs[2, 0])
ax.plot(epochs.global_step, epochs.judge_acc, color=JUDGE, lw=2, label="accuracy")
ax.plot(epochs.global_step, epochs.judge_cov, color=JUDGE, lw=1.5, ls=(0, (1, 2.2)), label="coverage")
for _, r in epochs.iterrows():
    dot(ax, r.global_step, r.judge_acc, JUDGE, 6)
ax.set_ylim(0, 10)
ax.set_yticks([0, 2, 4, 6, 8, 10])
ax.set_ylabel("score")
ax.legend(loc="upper left", handlelength=1.6)
card_title(ax, "LLM judge", "Qwen2.5-7B grading against the reference answer, 1–10")
epoch_axis(ax)

# learning rate
ax = fig.add_subplot(gs[2, 1])
ax.fill_between(steps.global_step, 0, steps.lr * 1e6, color=LR, alpha=0.08, lw=0)
ax.plot(steps.global_step, steps.lr * 1e6, color=LR, lw=2)
ax.set_ylim(0, steps.lr.max() * 1e6 * 1.15)
ax.set_ylabel("lr  (×10⁻⁶)")
card_title(ax, "Learning rate", "500-step warmup, then cosine decay to 10% of peak")
epoch_axis(ax)

out = HERE / f"{RUN}_overview.png"
fig.savefig(out, dpi=220)
fig.savefig(out.with_suffix(".pdf"))
print(f"saved {out} and {out.with_suffix('.pdf')}")
