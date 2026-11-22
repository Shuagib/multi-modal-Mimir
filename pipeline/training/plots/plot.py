import pandas as pd, matplotlib.pyplot as plt
df = pd.read_csv("checkpoints/projector_mimir/log_....csv")

train = df.dropna(subset=["train_loss"])
val = df.dropna(subset=["val_loss"])

plt.plot(train.index, train.train_loss, alpha=0.4, label="train (per step)")
plt.plot(val.index, val.val_loss, "o-", label="val (per epoch)")
plt.xlabel("step"); plt.ylabel("loss"); plt.legend(); plt.show()