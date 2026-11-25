import re, csv, sys
import matplotlib.pyplot as plt
import numpy as np
R = "/Users/simonholm/GitHub/multi-modal-Mimir/"
L = R + "pipeline/training/logs/"; P = R + "pipeline/training/plots/long_runs/logs/"
row = re.compile(r"│\s*(\d+)\s*│\s*(\d+|end)\s*│\s*([\d.]*)\s*│\s*([\de.+-]*)\s*│\s*([\d.]*)\s*│\s*([+\-\d.]*)\s*│\s*([\d.na]*)\s*│\s*([\d.na]*)")
def parse(path):
    steps, ends, spe = [], [], None
    for line in open(path, encoding="utf-8"):
        m = re.search(r"steps/epoch\s*│\s*(\d+)", line)
        if m: spe = int(m.group(1))
        m = row.search(line)
        if not m: continue
        e, s = int(m.group(1)), m.group(2)
        if s == "end": ends.append((e, float(m.group(5)), float(m.group(6)), m.group(7), m.group(8)))
        elif m.group(3): steps.append((e - 1 + int(s) / spe, float(m.group(3))))
    test = {}
    for k in ["test loss", "test gap", "judge acc", "judge cov"]:
        m = re.findall(rf"{k}\s*│\s*([+\-\d.]+)", open(path, encoding="utf-8").read())
        if m: test[k] = float(m[-1])
    return np.array(steps), ends, test
def smooth(y, k):
    if len(y) < k: return y
    return np.convolve(y, np.ones(k) / k, mode="valid")

runs = {"1k": R + "train-1k.log", "10k": R + "train-10k.log",
        "28k": L + "run_28k_mlp1e-4_llm1e-5_20261125-062827.txt",
        "76k": L + "run_76k_mlp1e-4_llm1e-5_20261125-030936.txt"}
cols = {"1k": "#9aa5b1", "10k": "#2f7ed8", "28k": "#e8743b", "76k": "#2a9d5c"}
data = {k: parse(v) for k, v in runs.items()}
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})

# 1: train/val curves, 3 epochs
fig, ax = plt.subplots(figsize=(9, 5))
for k, (st, ends, _) in data.items():
    if len(st):
        w = max(1, len(st) // 30); ys = smooth(st[:, 1], w)
        ax.plot(st[len(st) - len(ys):, 0], ys, color=cols[k], lw=1.8, label=f"{k} train")
    ax.plot([e[0] for e in ends], [e[1] for e in ends], "--o", color=cols[k], label=f"{k} val")
ax.set(xlabel="epoch", ylabel="cross-entropy per answer token", xticks=[0, 1, 2, 3],
       title="Train vs. validation loss by data size (3 epochs, MLP lr 1e-4, LLM lr 1e-5)")
ax.grid(axis="y", alpha=.3); ax.legend(ncol=4, fontsize=9, frameon=False)
fig.tight_layout(); fig.savefig("scaling_loss.png", dpi=200); fig.savefig("scaling_loss.pdf")

# 2: final metrics vs data size (eval after epoch 3 + held-out test)
sizes = {"1k": 1002, "10k": 10002, "28k": 28002, "76k": 75920}
ks = list(runs); x = [sizes[k] for k in ks]
T = lambda m: [data[k][2][m] for k in ks]
fig, axs = plt.subplots(1, 3, figsize=(12, 3.8))
axs[0].plot(x, [data[k][1][-1][1] for k in ks], "-o", color="#2f7ed8", label="val (eval set)")
axs[0].plot(x, T("test loss"), "--s", color="#2f7ed8", label="test")
axs[0].set(title="Loss \u2193", ylabel="cross-entropy")
axs[1].plot(x, [data[k][1][-1][2] for k in ks], "-o", color="#e8743b", label="val (eval set)")
axs[1].plot(x, T("test gap"), "--s", color="#e8743b", label="test")
axs[1].set(title="Image gap (shuffled \u2212 correct) \u2191")
axs[2].plot(x, T("judge acc"), "-o", color="#2a9d5c", label="accuracy")
axs[2].plot(x, T("judge cov"), "--s", color="#2a9d5c", label="coverage")
axs[2].set(title="Judge score on test (1\u201310) \u2191")
for a in axs:
    a.set_xscale("log"); a.set_xticks(x, ks); a.minorticks_off(); a.set_xlabel("training samples"); a.grid(alpha=.3); a.legend(frameon=False, fontsize=9)
fig.suptitle("Effect of training-set size (3 epochs, MLP lr 1e-4, LLM lr 1e-5)"); fig.tight_layout()
fig.savefig("scaling_metrics.png", dpi=200); fig.savefig("scaling_metrics.pdf")

# 3: 10k LR comparison per epoch
def ep(p): return list(csv.DictReader(open(p)))
lr = {"MLP 1e-5, LLM 1e-5 (10 ep)": (P + "10k_lr1e-5_both/10k_epochs.csv", "#2f7ed8"),
      "MLP 1e-4, LLM 1e-5 (5 ep)": (P + "10k_llm_lr1e-5_mlp_lr1e-4/10k_mlp-lr-1e-4_epochs.csv", "#e8743b"),
      "MLP 1e-4, LLM 1e-5 (3 ep)": (None, "#9aa5b1")}
fig, axs = plt.subplots(1, 2, figsize=(12, 4.2))
for name, (p, c) in lr.items():
    if p: rows = [(int(r["epoch"]), float(r["val_loss"]), float(r["gap"])) for r in ep(p)]
    else: rows = [(e[0], e[1], e[2]) for e in data["10k"][1]]
    e_, v, g = zip(*rows)
    axs[0].plot(e_, v, "-o", color=c, label=name); axs[1].plot(e_, g, "-o", color=c, label=name)
axs[0].set(title="Validation loss ↓", xlabel="epoch", ylabel="loss")
axs[1].set(title="Image gap (shuffled − correct) ↑", xlabel="epoch")
for a in axs: a.set_xticks(range(1, 11)); a.grid(alpha=.3)
axs[0].legend(frameon=False, fontsize=9); fig.suptitle("10k PixMo: learning-rate comparison")
fig.tight_layout(); fig.savefig("lr_10k.png", dpi=200); fig.savefig("lr_10k.pdf")

for k in ks: print(k, data[k][1][-1][:3], data[k][2], len(data[k][0]), "step rows")
