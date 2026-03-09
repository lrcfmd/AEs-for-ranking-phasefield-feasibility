import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from matplotlib import cm
import itertools
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from sklearn import datasets, metrics
from numpy import nan as NaN
from matplotlib.pyplot import figure
from math import ceil,factorial
import os

def plot_MFD(kmin, kmax,L0,L1,del_ax=None):                                                                     ### Plots MFDs for all rankings over a range
    vals = None
    plt.figure(dpi=200)
    fig, axes = plt.subplots(L0,L1, figsize = (20,10), sharey=True)
    axes = axes.flatten()
    vals_df = pd.DataFrame(columns = ["k", "MFD", "MFD Threshold", "TPR", "QPR", "sqrt(MFD*TPR)"])
    if del_ax:
        fig.delaxes(axes[del_ax])
        axes = np.delete(axes, del_ax)

    for k in range(kmin, kmax+1):
        if k == kmax:
            GT = pd.read_csv(os.path.join("DATA","ground_truth_full_MP_vectors.csv"))
            Q = pd.read_csv(os.path.join("DATA","query_full_MP_vectors.csv"))
        else:
            GT =  pd.read_csv(f"DATA/ground_truth_ranking_{k}_features.csv")
            Q = pd.read_csv(f"DATA/query_ranking_{k}_features.csv")
        GT_RE = GT["RE"]
        Q_RE = Q["RE"]
        ax = axes[k-kmin]
        fontsize = 20
        GT_fraction, Q_fraction, single_eval, iter_num = calculate(GT_RE, Q_RE)
        MFD = max(single_eval);MFD_ind = single_eval.index(MFD); MFD_threshold = iter_num[MFD_ind]; TPR = GT_fraction[MFD_ind]; QPR = Q_fraction[MFD_ind]
        vals, vals_df  = save_or_not(k, MFD, MFD_threshold, TPR, QPR, vals_df, vals)
        plot(ax, k, kmin, GT_fraction, Q_fraction, single_eval, iter_num)
    
    vals_df.to_csv("DATA/vals_df.csv", index=False)
    print(vals_df)
    fig.text(0.04, 0.5, "Proportion Below Threshold", va = "center", rotation = "vertical", fontsize = 20)
    fig.text(0.5, 0.04, "Reconstruction Error Threshold", ha = "center", fontsize = 20)
    plt.subplots_adjust(wspace=0.5, hspace = 0.3)
    plt.show()
    
    return vals

def calculate(GT_RE, Q_RE):
    iter_num = np.linspace(0.0,20,num=101)
    GT_fraction = []; Q_fraction = []
    lenGT = len(GT_RE); lenQ = len(Q_RE)
    single_eval = []
    
    for i in iter_num:
            threshold = i
            GT_count_inlier = np.sum(GT_RE[:] <= threshold) 
            Q_count_inlier = np.sum(Q_RE[:] <= threshold) 
            GT_percent = GT_count_inlier/lenGT           
            Q_percent = Q_count_inlier/lenQ
            single = GT_percent - Q_percent
            GT_fraction.append(GT_percent)
            Q_fraction.append(Q_percent)
            single_eval.append(single)

    return GT_fraction, Q_fraction, single_eval, iter_num

def save_or_not(k,MFD,MFD_threshold,TPR,QPR,vals_df,vals=None):
    if vals==None:
        vals = list([k,MFD,MFD_threshold,TPR,QPR,np.sqrt(MFD*TPR)])
        vals_df.loc[len(vals_df)] = vals

    elif vals[5]<MFD*TPR:
        vals = list([k,MFD,MFD_threshold,TPR,QPR,np.sqrt(MFD*TPR)])
        vals_df.loc[len(vals_df)] = vals

    else:
        vals_df.loc[len(vals_df)] = [k,MFD,MFD_threshold,TPR,QPR,np.sqrt(MFD*TPR)]
        vals = vals

    return vals, vals_df

def plot(ax, k, kmin, GT_fraction, Q_fraction, single_eval, iter_num):
    tmp_df = pd.DataFrame(columns=["Reconstruction Error", "Ground Truth Proportion", "Query Proportion", "Difference"])
    ax.plot(iter_num,GT_fraction,color='blue',label='Labelled')
    ax.plot(iter_num,Q_fraction,color='orange',label='Unlabelled')
    ax.plot(iter_num,single_eval,color='gray',label='Fraction Difference')
    single_max = max(single_eval)
    temp = single_eval.index(single_max)
    threshold_max = iter_num[temp]
    ax.plot([0,threshold_max],[single_max,single_max], '--', color="gray")
    ax.plot([threshold_max,threshold_max],[0,single_max], '--', color='gray')
    single_max = round(single_max,2)
    threshold_max = round(threshold_max,3)
    cord =  '('+format(threshold_max, '.2f')+', '+format(single_max,'.2f')+')'
    ax.text(threshold_max, single_max, cord, fontsize=20, fontweight='semibold', c='gray')
    Q_max = Q_fraction[temp]
    Q_max = round(Q_max,3)
    ax.scatter(threshold_max, Q_max, c="orange")
    ax.text(threshold_max, Q_max-0.05, Q_max, fontsize=20,c='orange')
    GT_max = GT_fraction[temp]
    GT_max = round(GT_max,3)
    ax.scatter(threshold_max, GT_max, c="blue")
    ax.text(threshold_max+0.1, GT_max+0.0, GT_max, fontsize=20,c='blue')
    refer_min, refer_max = min(GT_max, Q_max), max(GT_max, Q_max)
    ax.plot([threshold_max,threshold_max],[refer_min,refer_max], '--', color='gray')
    ax.set_xlim(0, 20)
    ax.set_title(f"{k-kmin})", fontsize = 20)
    ax.tick_params(axis="y",labelsize=15)
    ax.tick_params(axis="x",labelsize=15)
    tmp_df["Reconstruction Error"] = iter_num
    tmp_df["Ground Truth Proportion"] = GT_fraction
    tmp_df["Query Proportion"] = Q_fraction
    tmp_df["Difference"] = single_eval
    tmp_df.to_csv(f"DATA/MFD_PLOTS/MFD_{k}.csv", index=False)
    return

def mean_per_PF(k,size,data): 
    data = data["RE"]
    Nperms = factorial(size)
    Nuni = int(len(data)/Nperms)
    print(Nuni)
    means = []
    stds = []
    for i in range(Nuni):
        start = Nuni + i*(Nperms-1); fin = Nuni + (i+1)*(Nperms-1)
        inds = [i for i in range(start, fin)]; inds.append(i)
        df = data.iloc[inds]
        sm = df.sum()
        mean = sm/Nperms; means.append(mean)
        stds.append(np.std(df)) 
    return means, stds, Nuni

def compare_to_threshold(vals,query,size,kmax):
    CFC = pd.DataFrame(columns = ["Phase Field", "Mean RE", "STD"])
    if vals[0] == kmax:
        data = pd.read_csv(f"DATA/query_full_MP_vectors.csv")    
    else:
        data = pd.read_csv(f"DATA/query_ranking_{vals[0]}_features.csv")
    means, stds, Nuni = mean_per_PF(vals[0], size, data)
    CFC["Phase Field"] = query["Phase Field"].iloc[:Nuni]
    CFC["Mean RE"] = means
    CFC["STD"] = stds
    CFC = CFC.loc[CFC["Mean RE"] + CFC["STD"] < vals[2]]
    CFC.sort_values(by="Mean RE", inplace=True, ignore_index=True)
    CFC = add_and_order(CFC, size)
    CFC.to_csv("DATA/chemically_feasible_candidates.csv", index=False)
    return CFC

def add_and_order(CFC, size):
    tmp_df = pd.DataFrame(columns=[f"Atoms {i}" for i in range(size)])
    atoms = [s.strip() for s in open('DATA/magpie_tables/Abbreviation.table', 'r').readlines()]

    for n, PF in enumerate(CFC["Phase Field"]):
        new_PF = " ".join(atoms[i] for i in sorted([atoms.index(el) for el in PF.split(" ")]))
        CFC.at[n, "Phase Field"] = new_PF
        tmp_df.loc[len(tmp_df)] = [str(el) for el in new_PF.split(" ")]
    
    CFC = pd.concat([tmp_df, CFC], axis=1)

    return CFC
        
class ResultsAnalysis:
    def __init__(self, ranking):
        self.ranking = ranking
        self.candidates = self.ranking["Phase Field"]
        self.RE = self.ranking["Mean RE"]
        self.STD = self.ranking["STD"]

    def get_top_n(self, n=10, export=False, name=None):
        if export==True:
            self.ranking.iloc[range(n)].to_csv(f"{name}.csv", index=False)
        return self.ranking.iloc[range(n)]

    def phase_fields_containing(self, atom_pool, export=False, name=None):
        inds = []
        for i,PF in enumerate(self.candidates):
            if any(atom in atom_pool for atom in PF.split(" ")):
                inds.append(i)
        if export==True:
            self.ranking.iloc[inds].to_csv(f"{name}.csv", index=False)
        return self.ranking.iloc[inds]

    def phase_fields_containing_only(self, atom_pool, export=False, name=None):
        inds = []
        for i,PF in enumerate(self.candidates):
            if all(atom in atom_pool for atom in PF.split(" ")):
                inds.append(i)
        if export==True:
            self.ranking.iloc[inds].to_csv(f"{name}.csv", index=False)
        return self.ranking.iloc[inds]

    def get_phase_field(self, PF):
        found = False
        for i, pf in enumerate(self.candidates):
            if all(el in pf.split(" ") for el in PF.split(" ")):
                   return self.ranking.iloc[i]
                   found = True
                   break
        if found == False:
            print("Not found")
        return
