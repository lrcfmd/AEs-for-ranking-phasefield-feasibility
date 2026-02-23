import pandas as pd

df = pd.read_csv("phase_field_dataset.csv")
df = df.iloc[:100]
df.to_csv("PF.csv", index=False)
