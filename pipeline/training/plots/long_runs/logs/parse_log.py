"""Turn a pasted training table (the │ epoch │ step │ ... │ box from train.py) into CSVs.

usage: python parse_log.py <run>      reads <run>.txt, writes <run>_steps.csv and <run>_epochs.csv
"""
import csv
import sys
from pathlib import Path

HERE = Path(__file__).parent
RUN = sys.argv[1]

step_rows, epoch_rows = [], []
spe = 0  # steps per epoch, from the largest step number seen
for line in (HERE / f"{RUN}.txt").read_text().splitlines():
    if not line.startswith("│"):
        continue  # skips borders, headers, warnings
    cells = [c.strip() for c in line.strip("│").split("│")]
    if len(cells) < 8 or not cells[0].isdigit():
        continue
    epoch, step, train, lr, val, gap, acc, cov = cells[:8]
    if step == "end":
        epoch_rows.append((int(epoch), float(val), float(gap), float(acc), float(cov)))
    else:
        step_rows.append((int(epoch), int(step), float(train), float(lr)))
        spe = max(spe, int(step))

spe += 1  # logged every 100 steps; the epoch ends just after the last logged step

with open(HERE / f"{RUN}_steps.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["epoch", "step_in_epoch", "global_step", "train_loss", "lr"])
    for e, s, loss, lr in step_rows:
        w.writerow([e, s, (e - 1) * spe + s, loss, lr])

with open(HERE / f"{RUN}_epochs.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["epoch", "global_step", "val_loss", "gap", "judge_acc", "judge_cov"])
    for e, val, gap, acc, cov in epoch_rows:
        w.writerow([e, e * spe, val, gap, acc, cov])

print(f"{RUN}: {len(step_rows)} step rows, {len(epoch_rows)} epochs -> {RUN}_steps.csv, {RUN}_epochs.csv")
