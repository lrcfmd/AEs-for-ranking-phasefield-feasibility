import numpy as np
import pandas as pd
from DATA.ELEMENTS import ELEMENTS
from DATA.feature_labels import features
from itertools import combinations, permutations
from autoencoders import compress, run_AE
import os

def check_existing(atoms=None,size=None,path=None):

    ground_truth, uni_size = check_for_GT(atoms,size,path)
    
    query = check_for_Q(GT=ground_truth,uni_size=uni_size,atoms=atoms,size=size)
    
    return ground_truth, query

def check_for_GT(atoms=None,size=None,path=None):
    if "ground_truth.csv" in os.listdir("DATA"):
        data = pd.read_csv(f"DATA/ground_truth.csv")
        print("Found existing ground truth dataset...")
    
    else:
        print("Building a new ground truth dataset...")                     
        data = pd.read_csv(path)                          
        print(f"Limiting phase fields to size {size}")
        data = limit_size(data, size)
        print("Checking if atoms are ok")
        data = check_atoms(data, atoms)
        print("Removing phase fields with atomic number > 86")
        data = rm86(data,atoms)
        uni_size = len(data)
        print("Permuting the phase fields")
        data = permute(data)
        data.to_csv("DATA/ground_truth.csv",index=False)
    
    return data, uni_size

def limit_size(data,size):                                                  
    inds = [i for i, PF in enumerate(data["Phase Field"]) if len(PF.split(" ")) == size]
    data = data.iloc[inds]
    data = data.reset_index(drop=True)

    return data

def permute(data):
    dfs = [data]

    for PF in data["Phase Field"]:
        perms = list(permutations(PF.split(" ")))
        perms = perms[1:]
        new_PFs = list()

        for perm in perms:
            new_PF = " ".join(el for el in perm)
            new_PFs.append(new_PF)

        df = pd.DataFrame(); df["Phase Field"] = new_PFs
        dfs.append(df)

    data = pd.concat(dfs,ignore_index=True)
    
    return data

def rm86(data,atoms):
    rminds = []

    for i,PF in enumerate(data["Phase Field"]):
        if any(atoms.index(el)>86 for el in PF.split(" ")):
            rminds.append(i)

    data.drop(rminds,inplace=True)
    data.reset_index(drop=True,inplace=True)
    
    return data

def check_for_Q(GT,uni_size,atoms=None,size=None):
    if "query.csv" in os.listdir("DATA"):                                   
        print("Found existing query dataset...")
        query_df = pd.read_csv(f"DATA/query.csv")

    else:
        print("Building new query dataset...",flush=True)                   
        print("Getting unique elements in the ground truth dataset",flush=True)
        GT = GT.iloc[:uni_size]
        uni_els = unique_elements(GT)
        print("Constructing query phase fields",flush=True)
        query = unique_combinations(uni_els, GT, size)
        query_df = pd.DataFrame({"Phase Field": query})
        query_df = rm86(query_df, atoms)
        print("Permuting query phase fields")
        query_df = permute(query_df)
        query_df.to_csv("DATA/query.csv",index=False)

    return query_df

def unique_elements(data):
    PFs = data["Phase Field"]
    uni_els = []

    for PF in PFs:
        PF = PF.split(" ")
        
        for el in PF:
            if el not in uni_els:
                uni_els.append(el)
    
    return uni_els

def unique_combinations(uni_els, GT, size):
    inds = range(len(uni_els))
    combs = combinations(inds, size)
    query = make_fields(uni_els,combs)
    new_queries = []
    for QPF in query:
        novel = True
        for GTPF in GT["Phase Field"]:
            if all(el in GTPF.split(" ") for el in QPF.split(" ")):
                novel = False
                break
        if novel == True:
            new_queries.append(QPF)

    return new_queries

def check_atoms(data, atoms):
    rminds = []
    for i, PF in enumerate(data["Phase Field"]):
        if not all(el in atoms for el in PF.split(" ")):
            print(PF)
            rminds.append(i)
    
    if rminds:
        print("The following phase fields have been excluded as the elements were not able to be identified from the composition")
        print(data["Phase Field"].loc[rminds])
        data = data.drop(index=rminds).reset_index()
        
    return data 

def make_fields(uni_els,combs):
    query = []
    for c in combs:
        QPF = " ".join(uni_els[i] for i in c)
        query.append(QPF.strip(" "))

    return query

def P2I(atoms, PFs):                                                        
    all_inds = []

    for PF in PFs["Phase Field"]:
        inds = [atoms.index(el) for el in PF.split(" ")]
        all_inds.append(inds)
    
    return all_inds

def get_elemental_features(atoms, features):                                
    features = [f.strip() for f in features]
    feats = []
    dics = make_dics(features)
    
    for el in atoms:
        el_feats = []
        for dic in dics:
            el_feats.append(float(dic[el]))
        feats.append(el_feats)
    
    return feats

def make_dics(features):
    dics = [ {} for f in features]
    symbols = [s.strip() for s in open('DATA/magpie_tables/Abbreviation.table', 'r').readlines()]
    
    for i,f in enumerate(features):
        try:
            table = read_features(f'DATA/magpie_tables/{f}.table')
            dics[i]  = {sym: float(num) for sym, num in zip(symbols, table)}
        
        except:
            pass
    
    return dics

def read_features(f):
    lines = open(f,'r').readlines()
    
    return [float(l.strip()) if l.strip().isdigit else 0 for l in lines]
