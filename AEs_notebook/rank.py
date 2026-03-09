import numpy as np
import pandas as pd
from DATA.ELEMENTS import ELEMENTS
from DATA.feature_labels import features
from itertools import combinations
from autoencoders import run_AE, run_loaded_AE
import pickle as pkl
import matplotlib.pyplot as plt
from data_handling import get_elemental_features

def P2I(atoms, PFs):                                                                                            ### Converts phase fields to elemental indexes as the compressed 
    all_inds = []                                                                                               ### features are element-wise
    for PF in PFs["Phase Field"]:
        inds = [atoms.index(el) for el in PF.split(" ")]
        all_inds.append(inds)
    return all_inds

def main(ground_truth, query, size, epochs=400, el_feats=None, GT_atom_inds=None, Q_atom_inds=None, kmin=2, kmax=9, plot_his=None):
    GT_vecs = P2V(el_feats, GT_atom_inds)
    Q_vecs = P2V(el_feats, Q_atom_inds)
    print("\n###########################################")
    print(f"### Training Model with Full MP Vectors ###")
    print("###########################################")
    GT_results, Q_results, history = run_AE(GT_vecs, Q_vecs, vector_length=28*size, k=kmax, epochs=epochs)
    GT_results["Phase Field"] = ground_truth["Phase Field"]
    Q_results["Phase Field"] = query["Phase Field"]
    GT_results = GT_results[["Phase Field", "RE"]]
    Q_results = Q_results[["Phase Field", "RE"]]
    GT_results.to_csv("DATA/ground_truth_full_MP_vectors.csv", index=False)
    Q_results.to_csv("DATA/query_full_MP_vectors.csv", index=False)

    for k in range(kmin, kmax):
        print("\n##########################")
        print(f"### Training Model {k} ###")
        print("##########################")
        GT_vecs, Q_vecs = get_latent_vecs(k, GT_atom_inds, Q_atom_inds)
        GT_results, Q_results, history = run_AE(GT_vecs, Q_vecs, vector_length=k*size, k=k, epochs=epochs)
        GT_results["Phase Field"] = ground_truth["Phase Field"]
        Q_results["Phase Field"] = query["Phase Field"]
        GT_results = GT_results[["Phase Field", "RE"]]
        Q_results = Q_results[["Phase Field", "RE"]]
        GT_results.to_csv(f"DATA/ground_truth_ranking_{k}_features.csv", index=False)
        Q_results.to_csv(f"DATA/query_ranking_{k}_features.csv", index=False)
        
        if plot_his:
            plot_history(history, k)

    return

def P2V(el_vecs, atom_inds):
    num_rows = len(atom_inds)
    num_cols = sum(len(el_vecs[i]) for i in atom_inds[0])

    PVs = np.empty((num_rows, num_cols), dtype=np.float32)

    for row, inds in enumerate(atom_inds):
        PVs[row, :] = np.concatenate([el_vecs[i] for i in inds])

    return PVs

def get_latent_vecs(k, GT_atom_inds, Q_atom_inds):
    with open(f"VECS/vectors_L{k}.pkl", "rb") as f:
        el_vecs = pkl.load(f)

    GT_vecs = P2V(el_vecs, GT_atom_inds)
    Q_vecs = P2V(el_vecs, Q_atom_inds)

    return GT_vecs, Q_vecs

def plot_history(history,k):
    loss = np.array(history.history["loss"])
    val_loss = np.array(history.history["val_loss"])
    epochs = [i for i in range(len(loss))]

    fig, ax = plt.subplots()
    line1, = ax.plot(epochs, loss)
    line2, = ax.plot(epochs, val_loss)
    ax.set_title(f"History of ranking using {k} features")
    ax.set_xlabel("epochs")
    ax.set_ylabel("loss")
    ax.legend([line1, line2],["Loss", "Val Loss"])
    ax.set(xlim=(0,len(epochs)))
    plt.show()

    return

def loaded_AE_main(model, loaded_model, k, Q_inds, atoms, features):
    if model == "MP":
        el_feats = get_elemental_features(atoms,features)
        Q_vecs = P2V(el_feats, Q_inds)
    else:
        with open(f"VECS/vectors_L{k}.pkl", "rb") as f:
            el_vecs = pkl.load(f)
        Q_vecs = P2V(el_vecs, Q_inds)

    RE = run_loaded_AE(loaded_model, Q_vecs)

    return RE
