import pandas as pd

q = pd.read_csv("DATA/query.csv")
gt = pd.read_csv("DATA/ground_truth.csv")

for pf in q["Phase Field"]:
    for PF in gt["Phase Field"]:
        if all(el in PF.split(" ") for el in pf.split(" ")):
            print(pf)
