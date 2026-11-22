import pandas as pd
df = pd.read_csv("pipeline/training/logs/log_....csv")
print(df.dropna(subset=["train_loss"]).groupby("epoch").train_loss.mean())