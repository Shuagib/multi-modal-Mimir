import pandas as pd
df = pd.read_csv("pipeline/training/logs/log_20261122-185808.csv")
print(df.dropna(subset=["train_loss"]).groupby("epoch").train_loss.mean())

# epoch
# 1     2.414097
# 2     2.194376
# 3     2.046535
# 4     1.957956
# 5     1.913908
# 6     1.879599
# 7     1.849882
# 8     1.832964
# 9     1.825973
# 10    1.808490
# Name: train_loss, dtype: float64