import pandas as pd

print("=== BASELINE METRICS ===")
df = pd.read_csv("models/metrics_baseline.csv")
print(df.to_string(index=False))

print("\n=== CONTINUAL LEARNING METRICS ===")
cl = pd.read_csv("models/metrics_continual.csv")
print(cl.to_string(index=False))