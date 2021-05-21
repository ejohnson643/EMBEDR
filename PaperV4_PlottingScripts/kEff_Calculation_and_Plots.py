"""
###############################################################################
    Procedure: k Effective Calculator and Plots
###############################################################################

    Author: Eric Johnson
    Date Created: Sunday, April 4, 2021
    Email: ericjohnson1.2015@u.northwestern.edu

###############################################################################

    In this script, I want to codify the calculation of kEff and determine
    a reasonable threshold for alpha_nu.

    kEff is meant to indicate the "Effective" number of nearest neighbors that
    a sample has for a given kernel.  To do this, we're going to count the
    number of samples that have an affinity that is a certain fraction of that
    of the closest neighbor.  That is
        k_Eff = | j | p_ij < alpha_nu * p_i,max |.

    For the purposes of this calculation, we should use the full and exact
    affinity matrices (n_neighbors = n_samples - 1).  We should also be careful
    to use unsymmetric affinity matrices!

    To determine the optimal selection of alpha_nu, I think we want to look
    at the effect of reducing P and Q on the embedding quality...

    EDIT:  Since we're going to be using approximate kNN graphs and not the
    full graph of size N, we should just start from that approximate size. That
    is, since in practice we're going to use k_max = 3*perp, let's just roll
    with that.  This wouldn't change anything in practice except that the
    precision will be set differently based on using all the samples vs just
    3*perp samples.

###############################################################################
"""

from embedr.affinity import FixedEntropyAffinity

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

import os
from os import path
import PaperV4_PlottingScripts.plotting_utility as pUtl
import pickle as pkl
import scipy.sparse as sp
import seaborn as sns
from sklearn.metrics import pairwise_distances as pwd
from sklearn.preprocessing import normalize
import time

import warnings

EPSILON = np.finfo(np.float64).eps

data_dir = "./Data"
embed_dir = "./Embeddings/ParameterSweep"

warnings.filterwarnings("ignore", message="divide by zero encountered in log")
warnings.filterwarnings("ignore", message="invalid value encountered in mult")
warnings.filterwarnings("ignore", message="FixedFormatter should only be used")


def calc_DKL(P, Y):

    if Y.ndim == 2:
        Y = Y[np.newaxis, :, :]

    n_embed, n_samples, _ = Y.shape

    EES = np.zeros((n_embed, n_samples))
    for eNo in range(n_embed):
        Q = 1 / (1 + pwd(Y[eNo], metric='sqeuclidean'))

        normalize(Q, norm='l1', axis=1, copy=False)

        EES[eNo] = np.sum(P * np.log((P + EPSILON) / (Q + EPSILON)), axis=1)

    return EES


def calc_DKL_sparse(P, row_idx, col_idx, eY, kEff=None):

    if eY.ndim == 2:
        eY = eY[np.newaxis, :, :]

    n_embed, n_samples, _ = eY.shape
    DKL = np.zeros((n_embed, n_samples))

    for eNo, Y in enumerate(eY):

        Q = 1 / (1 + pwd(Y, metric="sqeuclidean"))
        Q = Q / Q.sum(axis=1)[:, np.newaxis]

        for rowNo, [start, end] in enumerate(zip(row_idx[:-1], row_idx[1:])):
            if kEff is not None:
                end = np.min([end, start + kEff[rowNo]])
            colNos = col_idx[start:end]
            P_row = P[start:end]
            Q_row = Q[rowNo, colNos]

            DKL_row = np.log(P_row + EPSILON) - np.log(Q_row + EPSILON)
            DKL[eNo, rowNo] = np.sum(P_row * DKL_row).squeeze()

    return DKL


def calc_kEff(P, row_idx, alpha_perc=0.02):

    kEff_arr = np.zeros(len(row_idx) - 1)

    for rowNo, [start, end] in enumerate(zip(row_idx[:-1], row_idx[1:])):
        P_row = P[start:end]

        P_row_max = P_row.max() * alpha_perc

        kEff_arr[rowNo] = np.sum(P_row > P_row_max)

    return kEff_arr.astype(int)


if __name__ == "__main__":

    plt.close('all')
    plt.rcParams['svg.fonttype'] = 'none'
    sns.set(color_codes=True)
    sns.set_style('whitegrid')
    matplotlib.rc("font", size=10)
    matplotlib.rc("xtick", labelsize=10)
    matplotlib.rc("ytick", labelsize=10)
    matplotlib.rc("axes", labelsize=12)
    matplotlib.rc("axes", titlesize=16)
    matplotlib.rc("legend", fontsize=10)
    matplotlib.rc("figure", titlesize=12)

    fig_dir  = "./Figures/PresentationFigures/PaperV4/"
    my_dpi = 400

    data_name = "Marrow"
    DR_method = 'tSNE'
    parameter = 'Perplexity'

    print_str  = f"  Calculating kEff for {data_name} Data!  "
    print(f"\n\n" + print_str + "\n" + "=" * len(print_str) + "\n\n")

    file_name_base = f"kEff_Calculation_{data_name}"

    aff_name_base = f"HyperparamSweep_{DR_method}_{parameter}_{data_name}"

    ## Set the name for the full saved output (this is what another script
    ## could load at once to skip all this junk).
    out_name = aff_name_base + "_Output_Dict.pkl"
    out_path = os.path.join(embed_dir, out_name)

    ## Runtime flags
    do_alpha_explore_plots = True
    do_kEff_vs_perp_plot   = False

    ## Runtime parameters
    n_data_embed     = 5
    n_null_embed     = 10
    use_asymmetric_P = True
    pVal_method      = 'average'

    null_gen_seed    = 54321
    affinity_seed    = 12345
    embedding_seed   = 1

    n_param          = 10     ## At most, use this many parameter values
    min_param_val    = 5      ## Minimum parameter value
    max_param_val    = 5000   ## Maximum parameter value

    if DR_method.lower() == 'tsne':
        ## Set some global t-SNE parameters
        n_components     = 2
        n_ee_iter        = 500
        min_ee_iter      = 100
        exaggeration     = 12
        ee_mom           = 0.5
        n_iter           = 1000
        min_iter         = 500
        mom              = 0.8
        n_jobs           = -1
        verbose          = True

        DR_method = 'tSNE'

    ## Set the kEff threshold
    alpha_nu_arr = np.logspace(-5, -1, 9)

    ###########################################################################
    ## Load the data and metadata!
    ###########################################################################
    if True:

        if data_name.lower() in ['marrow', 'diaphragm']:
            data_dir = path.join(data_dir, "TabulaMuris/FACS/")

        X, _ = pUtl.load_data(data_name, data_dir)

        n_samples, n_features = X.shape
        print(f"Loaded {data_name} Data! ({n_samples} x {n_features})")

        PWD_X = pwd(X, metric='euclidean')

        with open(out_path, 'rb') as f:
            out_dict = pkl.load(f)

        param_arr = np.asarray([10, 50, 100, 500, 1100])
        # param_arr = np.sort(list(out_dict['data_Y'].keys()))
        n_param = len(param_arr)

    ###########################################################################
    ## Load the affinity matrices!
    ###########################################################################
    if True:

        n_neib = np.min([int(n_samples - 1), int(3 * max_param_val)])

        aff_params = {"NN_alg": "pynndescent",
                      "n_neighbors": n_neib,
                      "perplexity": param_arr.max(),
                      "normalization": 'pair-wise',
                      "n_jobs": -1,
                      "random_state": affinity_seed,
                      "verbose": 5}

        ##################
        ## Working on X ##
        ##################

        print("\nGetting affinity matrix for data!")

        ## Set filenames and paths
        annoy_data_name = aff_name_base + "_ANNOY_index_data.obj"
        annoy_data_path = os.path.join(embed_dir, annoy_data_name)
        aff_data_name   = aff_name_base + "_base_affinity_data.pkl"
        aff_data_path   = os.path.join(embed_dir, aff_data_name)

        ## First, try and load the affinity matrix and rebuild the ANNOY index
        try:
            print(f"\nTrying to load and rebuild affinity matrix object...")

            with open(aff_data_path, 'rb') as f:
                aff_data_X = pkl.load(f)

            if n_samples > 1000:
                print(f"... affinity matrix loaded, getting kNN index...")

                kNN = aff_data_X.kNN_index._initialize_ANNOY_index(n_features)
                aff_data_X.kNN_index.indices = kNN
                aff_data_X.kNN_index.indices.load(annoy_data_path)

            print(f"... kNN index loaded successfully!")

        ## If it can't be loaded, then recompute.
        except (FileNotFoundError, OSError):
            print(f"... couldn't load!  Recalculating...")

            aff_data_X = FixedEntropyAffinity(**aff_params)
            aff_data_X.fit(X)

            if n_samples > 1000:
                aff_data_X.kNN_index.indices.save(annoy_data_path)

            with open(aff_data_path, 'wb') as f:
                pkl.dump(aff_data_X, f)

            print(f"... done!  Saved to file!")

    ###########################################################################
    ## Load / Calculate kEff for a bunch of alpha_perc
    ###########################################################################
    if do_alpha_explore_plots:

        k_Eff_dict_name = file_name_base + "_SavedDicts.pkl"
        k_Eff_dict_path = path.join(data_dir, k_Eff_dict_name)
        try:
            with open(k_Eff_dict_path, 'rb') as f:
                k_Eff_dict = pkl.load(f)

            data_Y         = k_Eff_dict['Y']
            data_EES       = k_Eff_dict['EES']
            frac_eff_dict  = k_Eff_dict['frac_eff']
            dEES_diff_dict = k_Eff_dict['EES_diff']

            # raise FileNotFoundError

        except (FileNotFoundError, KeyError):

            ## Initialize storage arrays
            data_Y = {}
            data_EES = {}

            frac_eff_dict  = {}
            dEES_diff_dict = {}

            print(f"\nLoading Parameter Sweep!")

            total_sweep_time = 0
            now = time.time()
            for pNo, param in enumerate(param_arr):

                print(f"\nEmbedding data with {DR_method} @ {parameter} ="
                      f" {param} ({pNo + 1}/{n_param})")

                embed_data_name  = aff_name_base + f"_DataEmbedding"
                embed_data_name += f"_Param{param:.0f}_RS{embedding_seed}.pkl"
                embed_data_path = os.path.join(embed_dir, embed_data_name)

                with open(embed_data_path, 'rb') as f:
                    dY, dAff = pkl.load(f)

                ## Extract and save the kernel widths
                taus = dAff.kernel_params['precisions'].reshape(-1, 1)

                ## Get the high-dimensional affinity matrix
                dP = np.sqrt(taus) * np.exp(-dAff.distances**2 * taus / 2)

                n_neib = dAff.n_neighbors
                row_idx = np.arange(0, n_neib * n_samples + 1, n_neib)

                dP = sp.csr_matrix((dP.ravel(), dAff.indices.ravel(), row_idx))

                ## Row-normalize the high-dimensional affinity matrix
                normalize(dP, norm='l1', axis=1, copy=False)

                print(f"\nCalculating EES for data embeddings!")
                dEES = calc_DKL_sparse(dP.data, dP.indptr, dP.indices, dY)

                data_Y[param] = dY[:]
                data_EES[param] = dEES[:]

                dEES_diff = np.zeros((len(alpha_nu_arr), n_data_embed,
                                      n_samples))
                frac_eff = np.zeros((len(alpha_nu_arr), n_samples))

                for aNo, alpha_nu in enumerate(alpha_nu_arr):

                    print(f"Alpha_nu = {alpha_nu:.3e}\t({aNo + 1} /"
                          f" {len(alpha_nu_arr)})")

                    k_Eff = calc_kEff(dP.data, dP.indptr, alpha_perc=alpha_nu)
                    frac_eff[aNo] = k_Eff.astype(float) / n_samples

                    dEES_part = calc_DKL_sparse(dP.data, dP.indptr, dP.indices,
                                                dY, kEff=k_Eff)

                    dEES_diff[aNo] = (dEES_part - dEES) / dEES

                frac_eff_dict[param]  = frac_eff[:]
                dEES_diff_dict[param] = dEES_diff[:]

            k_Eff_dict = {'Y': data_Y.copy(),
                          'EES': data_EES.copy(),
                          'frac_eff': frac_eff_dict.copy(),
                          'EES_diff': dEES_diff_dict.copy()}

            with open(k_Eff_dict_path, 'wb') as f:
                pkl.dump(k_Eff_dict, f)

        del k_Eff_dict

    ###########################################################################
    ## Plot change in EES vs kEff
    ###########################################################################
    if do_alpha_explore_plots:

        plt.close('all')

        box_fliers      = {'marker': ".",
                           # 'markeredgecolor': box_color,
                           'markersize': 2,
                           'alpha': 0.5}
        box_props       = {'alpha': 0.5,
                           # 'color': box_color,
                           'fill': True}
        box_patches     = ['boxes', 'whiskers', 'fliers', 'caps', 'medians']

        fig, [ax1, ax2] = plt.subplots(2, 1, figsize=(12, 9))

        box_handles = []

        for pNo, param in enumerate(param_arr):

            box_color = f"C{pNo}"

            box_fliers['markeredgecolor'] = box_color
            box_props['color'] = box_color

            for aNo, alpha_nu in enumerate(alpha_nu_arr):

                ees_diff = dEES_diff_dict[param][aNo].ravel()
                f_eff    = frac_eff_dict[param][aNo].ravel()

                box1 = ax1.boxplot(np.log10(np.abs(ees_diff)),
                                   widths=0.15,
                                   positions=[aNo + 0.15 * (pNo - 2)],
                                   notch=True,
                                   bootstrap=100,
                                   patch_artist=True,
                                   whis=(1, 99),
                                   boxprops=box_props,
                                   flierprops=box_fliers)

                box2 = ax2.boxplot(f_eff * n_samples / (3 * param),
                                   widths=0.15,
                                   positions=[aNo + 0.15 * (pNo - 2)],
                                   notch=True,
                                   bootstrap=100,
                                   patch_artist=True,
                                   whis=(1, 99),
                                   boxprops=box_props,
                                   flierprops=box_fliers)

                for item in box_patches:
                    plt.setp(box1[item], color=box_color)
                    plt.setp(box2[item], color=box_color)

            box_handles.append([box1, box2])

        ax1.set_xticks(np.arange(len(alpha_nu_arr)))
        ax1.set_xticklabels([f"{an:.1g}" for an in alpha_nu_arr])
        ax1.set_xlabel(r'Fraction of $p_{max}$ Considered "Significant"'
                       r" ($\alpha$)")
        ax1.set_yticklabels([f"{10.**dE:.1g}" for dE in np.arange(-9, 1)])
        ax1.set_ylabel(f"Fractional Change in Quality")

        b_labels = [f"Perp = {param:.0f}" for param in param_arr]
        b_hands1 = [bh[0]['boxes'][0] for bh in box_handles]
        ax1.legend(b_hands1, b_labels)

        ax2.set_xticks(np.arange(len(alpha_nu_arr)))
        ax2.set_xticklabels([f"{an:.1g}" for an in alpha_nu_arr])
        ax2.set_xlabel(r"Fraction of $p_{max}$ Considered Significant"
                       r" ($\alpha$)")
        ax2.set_yticklabels([f"{fr:.0%}" for fr in np.arange(-0.2, 1.1, 0.2)])
        ax2.set_ylabel(f'Fraction of Neighbors\nConsidered "Significant"')

        b_hands2 = [bh[1]['boxes'][0] for bh in box_handles]
        ax2.legend(b_hands1, b_labels)

        fig.tight_layout()

        fig2, ax = plt.subplots(1, 1, figsize=(12, 7))

        for pNo, param in enumerate(param_arr):

            for aNo, alpha_nu in enumerate(alpha_nu_arr):
                box_color = f"C{aNo}"

                ees_diff = np.abs(np.median(dEES_diff_dict[param][aNo],
                                            axis=0))
                ees_diff[ees_diff == 0] = 1.e-6
                f_eff    = frac_eff_dict[param][aNo] * n_samples / (3 * param)

                ax.scatter(np.log10(ees_diff), f_eff,
                           color=box_color, s=2, alpha=0.4)

                ax.scatter([np.log10(np.median(ees_diff))], [np.median(f_eff)],
                           color=box_color, s=100, marker='*so^v'[pNo],
                           edgecolor='k', zorder=20)

            ax.scatter([], [], color='w', s=100, marker='*so^v'[pNo],
                       edgecolor='k', label=f'Perplexity = {param}')

        for aNo, alpha_nu in enumerate(alpha_nu_arr):
            box_color = f"C{aNo}"
            ax.scatter([], [], color=box_color, s=10, alpha=1.0,
                       label=r'$\alpha$ = ' + f'{alpha_nu:.1g}')

        ax.set_xticklabels([f"{10.**dE:.1g}" for dE in np.arange(-7, 1)])
        ax.set_xlabel(f"Fractional Change in Quality")

        ax.set_yticklabels([f"{fr:.0%}" for fr in np.arange(-0.2, 1.1, 0.2)])
        ax.set_ylabel(f'Fraction of Neighbors Considered "Significant"')

        ax.legend()

        fig2.tight_layout()

        fig_base  = "V4SuppFig_" + file_name_base + "_deltaEES_vs_alpha"
        fig_base2 = "V4SuppFig_" + file_name_base + "_deltaEES_vs_fracEff"

        print("\n\nSaving Figure!\n\n")
        pUtl.save_figure(fig,
                         fig_base,
                         fig_dir=fig_dir,
                         dpi=my_dpi)

        print("\n\nSaving Figure!\n\n")
        pUtl.save_figure(fig2,
                         fig_base2,
                         fig_dir=fig_dir,
                         dpi=my_dpi)

    ###########################################################################
    ## Load kEff for a bunch of perps and illustrate relationship
    ###########################################################################
    if do_kEff_vs_perp_plot:
        file_name_base = f"HyperparamSweep_{DR_method}_{parameter}_{data_name}"

        ## Set the name for the full saved output (this is what another script
        ## could load at once to skip all this junk).
        out_name = file_name_base + "_Output_Dict.pkl"
        out_path = os.path.join(embed_dir, out_name)

        with open(out_path, 'rb') as f:
            out_dict = pkl.load(f)

        kEff_arr = out_dict['kEff_arr'][:].astype(int)
        perp_arr = np.sort(list(out_dict['data_Y'].keys()))

        del out_dict

        box_color = 'grey'

        box_fliers      = {'marker': ".",
                           'markeredgecolor': box_color,
                           'markersize': 2,
                           'alpha': 0.5}
        box_props       = {'alpha': 0.5,
                           'color': box_color,
                           'fill': True}
        box_patches     = ['boxes', 'whiskers', 'fliers', 'caps', 'medians']

        fig, ax = plt.subplots(1, 1, figsize=(12, 5))

        for pNo, perp in enumerate(perp_arr):

            box = ax.boxplot(kEff_arr[pNo],
                             widths=0.05,
                             positions=[np.log10(perp)],
                             notch=True,
                             bootstrap=100,
                             patch_artist=True,
                             whis=(1, 99),
                             boxprops=box_props,
                             flierprops=box_fliers)

            for item in box_patches:
                plt.setp(box[item], color=box_color)

        ax.set_yscale('log')

        xtick_idx = np.asarray([0, 3, 8, 13, 18, 23, 29])
        ax.set_yticks([10, 100, 1000, n_samples])
        ax.set_yticklabels([10, 100, 1000, n_samples])
        ax.set_xticks(np.log10(perp_arr[xtick_idx]))
        ax.set_xticklabels(perp_arr[xtick_idx], rotation=-30)

        ax.set_xlabel(f"Perplexity")
        ax.set_ylabel(f'"Effective" Number of Neighbors')

        fig.tight_layout()

        fig_base  = "V4SuppFig_" + file_name_base + "_kEff_vs_Perp"

        print("\n\nSaving Figure!\n\n")
        pUtl.save_figure(fig,
                         fig_base,
                         fig_dir=fig_dir,
                         dpi=my_dpi)
        
    ###########################################################################
    ## SHOW PLOTS
    ###########################################################################
    if True:
        plt.show()
