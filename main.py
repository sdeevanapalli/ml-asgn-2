import pandas as pd

print("baseline metrics:")
df = pd.read_csv("models/metrics_baseline.csv")
print(df.to_string(index=False))

print("\ncontinual learning metrics:")
cl = pd.read_csv("models/metrics_continual.csv")
print(cl.to_string(index=False))