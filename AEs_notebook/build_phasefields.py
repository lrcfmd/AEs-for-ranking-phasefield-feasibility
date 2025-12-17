import pandas as pd

def make_fields(x):
    x = "".join(s for s in x if s not in 'x"[ (.)]"' and not s.isdigit())
    elements = str()
    for i,s in enumerate(x):
        if s.isupper() and i != 0:
            elements += " "+s
        else:
            elements += s
    string = [el for el in elements.split(" ")]
    sorted_set = sorted(set(string))
    
    return " ".join(el for el in sorted_set)

def get_unique(PFs):
    uni_PFs = []
    
    for PF in PFs:
        if PF not in uni_PFs:
            uni_PFs.append(PF)
    
    dfs = pd.DataFrame(uni_PFs, columns = ["Phase Field"])

    return dfs
