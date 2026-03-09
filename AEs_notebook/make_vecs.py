import numpy as np
import pandas as pd
import os
import sys
from DATA.ELEMENTS import ELEMENTS
from DATA.feature_labels import features
from itertools import combinations
from autoencoders import run_AE
import pickle as pkl

def get_elemental_features(atoms, features):                                                                    ### Builds elemental feature vectors
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

def get_latent_vecs(k, GT_atom_inds, Q_atom_inds):
    with open(f"VECS/vectors_L{k}.pkl", "rb") as f:
        el_vecs = pkl.load(f)
    print(el_vecs)
    #GT_vecs = P2V(el_vecs, GT_atom_inds)
    #Q_vecs = P2V(el_vecs, Q_atom_inds)

    return #GT_vecs, Q_vecs

if __name__ == "__main__":
    features = features
    print(features)
    atoms = [s.strip() for s in open('DATA/magpie_tables/Abbreviation.table', 'r').readlines()]
    el_feats = get_elemental_features(atoms, features)
    
