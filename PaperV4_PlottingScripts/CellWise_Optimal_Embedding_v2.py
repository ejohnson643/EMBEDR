"""
###############################################################################
    Figure: Cell-Wise Optimal Embedding (v2)
###############################################################################

    Author: Eric Johnson
    Date Created: Wednesday, April 7, 2021
    Email: ericjohnson1.2015@u.northwestern.edu

###############################################################################

    This figure demonstrates that when we can assess optimality of
    representation in an embedding on a sample-wise basis that we can adjust
    t-SNE to generate a "cell-wise optimal" embedding!  Specifically, we will
    look for the perplexity at which each sample is best embedded, choosing the
    smallest such perplexity if there are multiple, and generating an affinity
    matrix using such a mixture of kernels.

    Once this is done, we will show the the resulting embedding colored a few
    ways:
        - Unlabeled, so that data density is more obvious
        - Colored by cell ontology class
        - Colored by p-Value coloring
        - Colored by kEff

    Notes:
        - I just had the idea that similar cell ontologies should be colored
          together...  That is, we should visually "detect" the clusters and
          then color all the cell types in each cluster with related colors.
          We should then look into making a confusion table...

    This version uses the newest parameter sweeps as inputs.  These have been
    verified to use an asymmetric P for DKL calculations and to use the
    averaging method for p-values.

###############################################################################
"""

from embedr.affinity import FixedEntropyAffinity
import matplotlib
import matplotlib.gridspec as gs
import matplotlib.pyplot as plt
import matplotlib.transforms as mtf
import numpy as np
import numpy.random as r
from openTSNE import TSNE, TSNEEmbedding
from openTSNE.initialization import random as initRand
import os
import PaperV4_PlottingScripts.plotting_utility as pUtl
import pickle as pkl
import scipy.sparse as sp
import seaborn as sns
from sklearn.cluster import DBSCAN
from sklearn.metrics import pairwise_distances as pwd
from sklearn.preprocessing import normalize
import time
import warnings

EPSILON = np.finfo(np.float64).eps

warnings.filterwarnings("ignore", message="This figure includes Axes that")
warnings.filterwarnings("ignore", message="tight_layout not applied: ")

data_dir = "./Data"
outdict_dir = "./Embeddings/ParameterSweep"
embed_dir = "./Embeddings/CellWise_Optimal"


def load_tSNE_given_affinity(affObj,
                             X,
                             name_base=None,
                             embed_dir="./Embeddings",
                             n_embed=1,
                             n_components=2,
                             early_exag_iter=250,
                             n_iter=750,
                             n_jobs=-1,
                             random_state=1,
                             verbose=True):

    n_samples = affObj.n_samples

    if name_base is not None:
        embed_name = name_base + f"_tSNE.pkl"

    n_2_embed = n_embed

    try:
        if (name_base is None):
            raise ValueError

        if verbose:
            print(f"\nTrying to load {embed_name}")

        with open(os.path.join(embed_dir, embed_name), 'rb') as f:
            Y = pkl.load(f)

        n_embedded = len(Y)

        n_2_embed = n_embed - n_embedded
        if n_embedded < n_embed:
            if verbose:
                print(f"File loaded, but there aren't enough embeddings!"
                      f" ({n_embedded} < {n_embed})")
            raise ValueError

    except (FileNotFoundError, EOFError, ValueError):

        if (n_2_embed == n_embed) and verbose:
            print(f"File couldn't be loaded!")

        tmp_Y = np.zeros((n_2_embed, n_samples, n_components)).astype(float)

        for eNo in range(n_2_embed):
            if verbose:
                print(f"Generating embedding {eNo + 1} / {n_2_embed}")

            init_Y = initRand(X, random_state=random_state + eNo)

            tmp = TSNEEmbedding(init_Y, affObj,
                                n_jobs=n_jobs,
                                verbose=True)

            tmp = tmp.optimize(n_iter=early_exag_iter,
                               exaggeration=12,
                               momentum=0.5)

            tmp = tmp.optimize(n_iter=n_iter,
                               exaggeration=1,
                               momentum=0.8)

            tmp_Y[eNo] = tmp[:]

        if n_2_embed != n_embed:
            Y = np.vstack((Y.reshape(-1, n_samples, n_components), tmp_Y))
        else:
            Y = tmp_Y[:]

        if name_base is not None:
            if verbose:
                print(f"Saving {embed_name} to file!")

            with open(os.path.join(embed_dir, embed_name), 'wb') as f:
                pkl.dump(Y, f)

    return Y[:n_embed].astype(float).squeeze()


def calc_DKL(P, row_idx, col_idx, eY):

    if eY.ndim == 2:
        eY = eY[np.newaxis, :, :]

    n_embed, n_samples, _ = eY.shape
    DKL = np.zeros((n_embed, n_samples))

    for eNo, Y in enumerate(eY):

        Q = 1 / (1 + pwd(Y, metric="sqeuclidean"))
        Q = Q / Q.sum(axis=1)[:, np.newaxis]

        for rowNo, [start, end] in enumerate(zip(row_idx[:-1], row_idx[1:])):
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

    return kEff_arr


###############################################################################
##  RUN THE FILE
###############################################################################
if __name__ == "__main__":

    data_name = "Marrow"
    DR_method = 'tSNE'
    parameter = 'Perplexity'

    print_str = "  Plotting Cell-Wise Optimal Embedding on {data_name} Data!  "
    print(f"\n\n" + print_str + "\n" + len(print_str) * "=" + "\n")

    ## Define figure and file name format
    name_base = f"V4Fig_CellWise_Optimal_Embedding_v2_{data_name}"
    fig_base  = name_base + f"_{DR_method}_{parameter}_{data_name}"

    ## Set the name for the full saved output (this is what another script
    ## could load at once to skip all this junk).
    file_name_base = f"HyperparamSweep_{DR_method}_{parameter}_{data_name}"
    out_name = file_name_base + "_Output_Dict.pkl"
    out_path = os.path.join(outdict_dir, out_name)

    ## Runtime flags
    show_all_axes = True

    ## Runtime parameters
    n_data_embed     = 1
    n_null_embed     = 10
    use_asymmetric_P = True
    pVal_method      = 'average'

    alpha_nu = 0.01

    ## Set random seeds
    null_gen_seed  = 54321
    affinity_seed    = 12345
    embedding_seed = 1  ## value set to 1 for figure!

    ## Minimum number of cells of a cell type before including in label
    label_thresh = 10

    ## Set some global t-SNE parameters
    n_components     = 2
    n_ee_iter        = 250
    exaggeration     = 12
    ee_mom           = 0.5
    n_iter           = 1000
    min_iter         = 250
    mom              = 0.8
    n_jobs           = -1
    verbose          = True

    ###########################################################################
    ## Load the data and metadata!
    ###########################################################################
    if True:
        ## Data file names
        if data_name.lower() in ['marrow', 'diaphragm']:
            data_dir = f"./Data/TabulaMuris/FACS/"

        X, metadata = pUtl.load_data(data_name, data_dir)

        n_samples, n_features = X.shape

        meta_dict = pUtl.parse_metadata(metadata)
        cell_ont_labels = meta_dict['cell_ont_labels']
        cell_ont_map    = meta_dict['cell_ont_map']
        cell_ont_column = metadata['cell_ontology_class']

        if data_name.lower() == 'marrow':
            cell_ont_labels = pUtl.get_nice_marrow_labels()

        ## Load data!
        with open(out_path, 'rb') as f:
            out_dict = pkl.load(f)

        ## Unpack everything
        data_Y = out_dict['data_Y']
        kEff_all = out_dict['kEff_arr']
        pVals = out_dict['pVals']

        del out_dict

        perp_arr = np.sort(list(data_Y.keys()))

        kEff_arr = np.median(kEff_all, axis=1)

        pVals = np.log10(pVals)
        min_pVal = -np.min(pVals)

        print(f"\nGenerating {n_null_embed} Null Data!")
        r.seed(null_gen_seed)
        null_X = np.zeros((n_null_embed, n_samples, n_features))
        for nNo in range(n_null_embed):
            null_X[nNo] = np.asarray([r.choice(col, size=n_samples)
                                      for col in X.T]).T

        best_perps = np.zeros(n_samples).astype(int)
        best_best_pVals = np.zeros(n_samples).astype(float)
        for nn in range(n_samples):
            min_idx = np.argmin(pVals[:, nn])
            if not isinstance(min_idx, np.int64):
                min_idx = min_idx[0]
            best_perps[nn] = perp_arr[min_idx]
            best_best_pVals[nn] = np.min(pVals[:, nn])

    ###########################################################################
    ## Load / Create Affinity Matrix
    ###########################################################################
    if True:
        aff_params = {"n_neighbors": n_samples - 1,
                      "perplexity": best_perps,
                      "normalization": 'pair-wise',
                      "n_jobs": -1,
                      "random_state": affinity_seed,
                      "verbose": 5}

        #######################
        ## Working on DATA X ##
        #######################

        ## Set filenames and paths
        aff_data_name   = name_base + f"_affinity_data.pkl"
        aff_data_path   = os.path.join(embed_dir, aff_data_name)

        ## First, try and load the affinity matrix and rebuild the ANNOY index
        try:
            print(f"\nTrying to load and rebuild affinity matrix object...")

            with open(aff_data_path, 'rb') as f:
                dAff = pkl.load(f)

            print(f"... affinity matrix loaded successfully!")

        ## If it can't be loaded, then recompute.
        except (FileNotFoundError, OSError):
            print(f"... couldn't load!  Recalculating...")

            dAff = FixedEntropyAffinity(**aff_params)
            dAff.fit(X)

            with open(aff_data_path, 'wb') as f:
                pkl.dump(dAff, f)

            print(f"... done!  Saved to file!")

    ###########################################################################
    ## Generate / Load Embeddings
    ###########################################################################
    if True:

        print(f"\nLoading data embedding at cell-wise optimal perplexity!")

        embed_data_name  = name_base + f"_DataEmbedding"
        embed_data_name += f"_RS{embedding_seed}"

        ## Load the data embedding
        data_Y = load_tSNE_given_affinity(dAff, X,
                                          name_base=embed_data_name,
                                          embed_dir=embed_dir,
                                          n_embed=n_data_embed,
                                          n_components=n_components,
                                          early_exag_iter=n_ee_iter,
                                          n_iter=n_iter,
                                          n_jobs=n_jobs,
                                          random_state=embedding_seed,
                                          verbose=True)

    ###########################################################################
    ## Cluster the Embedding
    ###########################################################################
    if True:
        print(f"\nClustering embedding with DBSCAN!")

        PWD_Y = np.triu(pwd(data_Y, metric='euclidean'), k=1)
        n_pwd = n_samples * (n_samples - 1) / 2

        eps = np.percentile(PWD_Y[PWD_Y.nonzero()], 5)

        if data_name.lower() == 'marrow':
            db_min_samp = 50
        elif data_name.lower() == 'diaphragm':
            db_min_samp = 20

        dbObj = DBSCAN(eps=eps, min_samples=db_min_samp)
        dbObj.fit(data_Y)

        db_labels = dbObj.labels_
        unique_labels = np.sort(np.unique(db_labels))
        n_db_labels = len(unique_labels) - 1

        label_idx = {ll: (db_labels == ll).nonzero()[0]
                     for ll in unique_labels}

        cOnt_by_label = {}
        cOnt_ct_label = {}
        cOnt_remap    = {}

        cOnt_and_label_idx = {}

        remap_counter = 0

        for lNo, lab in enumerate(unique_labels):

            if lab == -1:
                continue

            samp_in_label = metadata.iloc[label_idx[lab]]
            samp_ct_label = samp_in_label.groupby('cell_ontology_class')
            samp_ct_label = samp_ct_label.count()

            cOnt = samp_ct_label.index.values
            cnts = samp_ct_label['Unnamed: 0'].values

            good_idx = (cnts > label_thresh).nonzero()[0]

            cnts = cnts[good_idx]
            cOnt = cOnt[good_idx]

            sort_idx = np.argsort(cnts)[::-1]

            cnts = cnts[sort_idx]
            cOnt = cOnt[sort_idx]

            if data_name.lower() == 'marrow':
                if lNo == 3:
                    cnts[1], cnts[2] = cnts[2], cnts[1]
                    cOnt[1], cOnt[2] = cOnt[2], cOnt[1]

            cOnt_by_label[lab] = cOnt
            cOnt_ct_label[lab] = cnts

            ## Make a new mapping of each detected cell ontology into a new
            ## index for the row in a confusion table.
            for ii, cO in enumerate(cOnt):
                if cO not in cOnt_remap:
                    cOnt_remap[cO] = remap_counter
                    remap_counter += 1

            good_idx = [(cell in cOnt) and (dbl == lab)
                        for (cell, dbl) in zip(cell_ont_column, db_labels)]

            cOnt_and_label_idx[lab] = np.asarray(good_idx).nonzero()[0]

        label_cmap = sns.color_palette('tab20')

    ###########################################################################
    ## Set up figure parameters
    ###########################################################################
    if True:

        ## Set matplotlib.rc parameters
        plt.close('all')
        plt.rcParams['svg.fonttype'] = 'none'
        sns.set(color_codes=True)
        sns.set_style('whitegrid')
        matplotlib.rc("font", size=10)
        matplotlib.rc("xtick", labelsize=10)
        matplotlib.rc("ytick", labelsize=10)
        matplotlib.rc("axes", labelsize=12)
        matplotlib.rc("axes", titlesize=12)
        matplotlib.rc("legend", fontsize=10)
        matplotlib.rc("figure", titlesize=12)

        fig_dir  = "./Figures/PresentationFigures/PaperV4/"

        ## Figure Size Parameters
        my_dpi     = 400  ## Pixels per inch.
        fig_width  = 7.5  ## inches (8 inch-wide paper minus margins)
        fig_height = 1. * fig_width
        fig_size   = (fig_width, fig_height)

        ## Figure parameters
        fig_pad          = 0.3   ## Fraction of a fontwidth, iirc
        fig_ppad         = 0.01  ## Percent of fig to leave around edge.

        main_rows        = 2
        main_cols        = 1
        main_hspace      = 0.05
        main_hratios     = [0.8, 2]
        main_wpad        = 0.01
        main_hpad        = 0.01

        top_rows         = 1
        top_cols         = 3
        top_wspace       = 0.025
        top_spine_alpha  = 1
        top_spine_width  = 1

        bot_rows         = 1
        bot_cols         = 2
        bot_wspace       = 0.5
        bot_wratios      = [2, 0.8]
        bot_spine_alpha  = 1
        bot_spine_width  = 1

        bot_title_pad    = -14

        gmp_box_frac     = 0.4
        gmp_box_lw       = 1

        sctr_ylim_pad    = 1.3
        sctr_title_pad   = -14

        ## No color scatterplot params
        no_clr_color     = "gray"
        no_clr_size      = 3
        no_clr_alpha     = 0.3

        ## kEff scatterplot params
        opt_kEff_cmap    = "rocket"
        opt_kEff_size    = 2
        opt_kEff_alpha   = 0.5

        ## EES scatterplot params
        ees_sctr_cmap    = "rocket_r"
        ees_sctr_size    = 2
        ees_sctr_alpha   = 0.5

        ## Colorbar properties
        cbar_pad         = 0.01
        cbar_tickl       = 0
        cbar_tickw       = 0
        cbar_ticksize    = 8
        cbar_tickpad     = 2

        ## Color maps
        base_cmap   = sns.color_palette()
        cblind_cmap = sns.color_palette('colorblind')
        bright_cmap = sns.color_palette('bright')

        ## p-Value Colors
        min_pVal = -pVals.min() + 0.005
        pVal_clr_change = [0, 1, 2, 3, min_pVal]
        [pVal_cmap,
         pVal_cnorm] = pUtl.make_categ_cmap(change_points=pVal_clr_change)

    ###########################################################################
    ## Set up the Gridspec
    ###########################################################################
    if True:

        fig = plt.figure(figsize=fig_size)

        ## If show_background is on, then show all the subplot edges
        spine_alpha = 0
        if show_all_axes:
            spine_alpha = 1

        ## Make main gridspec
        main_gs = fig.add_gridspec(main_rows, main_cols,
                                   height_ratios=main_hratios,
                                   hspace=main_hspace)

        top_gs = main_gs[0].subgridspec(nrows=top_rows,
                                        ncols=top_cols,
                                        wspace=top_wspace)

        top_axes = []
        for ii in range(top_cols):
            ax = fig.add_subplot(top_gs[ii])
            ax = pUtl.make_border_axes(ax, spine_alpha=top_spine_alpha,
                                       spine_width=top_spine_width)
            top_axes.append(ax)

        bot_gs = main_gs[1].subgridspec(nrows=bot_rows,
                                        ncols=bot_cols,
                                        wspace=bot_wspace,
                                        width_ratios=bot_wratios)

        bot_sctr_ax = fig.add_subplot(bot_gs[0])
        bot_sctr_ax = pUtl.make_border_axes(bot_sctr_ax,
                                            spine_alpha=bot_spine_alpha,
                                            spine_width=bot_spine_width)

        bot_conf_ax = fig.add_subplot(bot_gs[1])
        bot_conf_ax = pUtl.make_border_axes(bot_conf_ax,
                                            spine_alpha=bot_spine_alpha,
                                            spine_width=bot_spine_width)

        fig.tight_layout(pad=fig_pad)
        fig.subplots_adjust(left=0 + fig_ppad,
                            right=1 - fig_ppad,
                            bottom=0 + fig_ppad,
                            top=1 - fig_ppad)

    ###########################################################################
    ## TOP-LEFT: Plot without color
    ###########################################################################
    if True:

        ax = top_axes[0]

        hax = ax.scatter(*data_Y.T, color=no_clr_color, s=no_clr_size,
                         alpha=no_clr_alpha, edgecolor='none')

        ylim = ax.get_ylim()
        ax.set_ylim(ylim[0], sctr_ylim_pad * ylim[1])

        ax.set_title(r"Unlabelled Data", pad=sctr_title_pad)

    ###########################################################################
    ## TOP-MIDDLE: Plot with k_Eff
    ###########################################################################
    if True:

        kEff_from_P_name = fig_base + "_optP.pkl"
        try:
            with open(os.path.join(embed_dir, kEff_from_P_name), 'rb') as f:
                opt_kEff = pkl.load(f)
        except FileNotFoundError:
            opt_aff = FixedEntropyAffinity(perplexity=best_perps,
                                           normalization='point-wise',
                                           n_jobs=-1,
                                           random_state=affinity_seed,
                                           verbose=5)
            opt_aff.fit(X)
            closest_neib_P = (alpha_nu * np.max(opt_aff.P, axis=1).toarray())
            closest_neib_P = closest_neib_P.reshape(-1, 1)
            opt_kEff = np.sum(opt_aff.P > closest_neib_P, axis=1).squeeze()

            with open(os.path.join(embed_dir, kEff_from_P_name), 'wb') as f:
                pkl.dump(opt_kEff, f)

        opt_kEff = np.asarray(opt_kEff).squeeze()

        ax = top_axes[1]

        sort_idx = np.argsort(opt_kEff)

        hax = ax.scatter(*data_Y[sort_idx].T, c=np.log10(opt_kEff[sort_idx]),
                         s=opt_kEff_size, alpha=opt_kEff_alpha,
                         cmap=opt_kEff_cmap)

        cax = plt.colorbar(hax, ax=ax, pad=cbar_pad)

        cax.set_ticks([1, 2, 3])
        cax.set_ticklabels([r"$10^1$", r"$10^2$", r"$10^3$", f"{n_samples}"])
        cax.ax.tick_params(length=cbar_tickl, width=cbar_tickl,
                           labelsize=cbar_ticksize, pad=cbar_tickpad)

        ylim = ax.get_ylim()
        ax.set_ylim(ylim[0], sctr_ylim_pad * ylim[1])

        ax.set_title(r"$k_{Eff}$", pad=sctr_title_pad)

    ###########################################################################
    ## TOP-RIGHT: Plot with p-Values
    ###########################################################################
    if True:
        ax = top_axes[2]

        # sort_idx = np.argsort(opt_pVals.squeeze())[::-1]
        sort_idx = np.argsort(best_best_pVals)[::-1]

        # clrs = -np.log10(opt_pVals.squeeze()[sort_idx])
        clrs = -best_best_pVals[sort_idx]

        hax = ax.scatter(*data_Y[sort_idx].T, c=clrs,
                         s=ees_sctr_size, alpha=ees_sctr_alpha,
                         cmap=pVal_cmap, norm=pVal_cnorm)

        cax = plt.colorbar(hax, ax=ax, pad=cbar_pad)
        cax.set_label(r"Poorly-Embedded   $\leftrightarrow$   Well-Embedded",
                      fontsize=8)

        cax.set_ticks([1, 2, 3, 4])
        cax.set_ticklabels([r"$10^{-1}$", r"$10^{-2}$",
                            r"$10^{-3}$", r"$10^{-4}$"])
        cax.ax.tick_params(length=cbar_tickl, width=cbar_tickl,
                           labelsize=cbar_ticksize, pad=cbar_tickpad)

        ylim = ax.get_ylim()
        ax.set_ylim(ylim[0], sctr_ylim_pad * ylim[1])

        ax.set_title(r"EMBEDR $p$-Value", pad=sctr_title_pad)

    ###########################################################################
    ## BOTTOM-LEFT: Color by cell ontology
    ###########################################################################
    if True:

        print(f"\nPlotting embedding by cell type!")

        if data_name.lower() == 'marrow':
            gmp_label = meta_dict['cell_ont_ids'][12]
            gmp_idx = (cell_ont_column == gmp_label).values
            gmp_idx = gmp_idx.nonzero()[0]

            bc_label = meta_dict['cell_ont_ids'][16]
            bc_idx = (cell_ont_column == bc_label).values
            bc_idx = bc_idx.nonzero()[0]

        ax = bot_sctr_ax

        ax.scatter(*data_Y[label_idx[-1]].T, color='grey', s=3, alpha=0.3)

        for lNo, lab in enumerate(unique_labels[1:]):

            color_1 = label_cmap[int(lNo * 2)]
            color_2 = label_cmap[int(lNo * 2) + 1]
            colors = pUtl.make_seq_cmap(color_1, color_2,
                                        n_colors=len(cOnt_by_label[lab]))

            good_idx = cOnt_and_label_idx[lab]

            cOnt = cOnt_by_label[lab]
            cOnt_by_label_cmap = {cO: colors[ii] for ii, cO in enumerate(cOnt)}
            tmp_colors = [cOnt_by_label_cmap[cO]
                          for cO in cell_ont_column[good_idx]]

            hax = ax.scatter(*data_Y[good_idx].T, c=tmp_colors, s=3)

            med_Y = np.median(data_Y[good_idx], axis=0)

            ax.text(*med_Y, f"{lNo + 1}", fontsize=12,
                    ha='center', va='center', fontweight='bold')

            if data_name.lower() == 'marrow':
                if lab == 1:
                    gmp_lab2_idx = np.asarray([idx for idx in gmp_idx
                                               if idx in good_idx])

                if lab == 3:
                    gmp_lab4_idx = np.asarray([idx for idx in gmp_idx
                                               if idx in good_idx])

        if data_name.lower() == 'marrow':

            [gmp2_x1,
             gmp2_y1] = np.percentile(data_Y[gmp_lab2_idx], 95, axis=0)
            [gmp2_x0,
             gmp2_y0] = np.percentile(data_Y[gmp_lab2_idx],  5, axis=0)

            gmp2_dx = gmp2_x1 - gmp2_x0
            gmp2_dy = gmp2_y1 - gmp2_y0

            gmp2_x0 -= gmp2_dx * gmp_box_frac
            gmp2_x1 += gmp2_dx * gmp_box_frac
            gmp2_y0 -= gmp2_dy * gmp_box_frac
            gmp2_y1 += gmp2_dy * gmp_box_frac

            ax.plot([gmp2_x0, gmp2_x1, gmp2_x1, gmp2_x0, gmp2_x0],
                    [gmp2_y0, gmp2_y0, gmp2_y1, gmp2_y1, gmp2_y0], '-k',
                    linewidth=gmp_box_lw)

            gmp4_x1, gmp4_y1 = np.percentile(data_Y[gmp_lab4_idx], 95, axis=0)
            gmp4_x0, gmp4_y0 = np.percentile(data_Y[gmp_lab4_idx],  5, axis=0)

            gmp4_dx = gmp4_x1 - gmp4_x0
            gmp4_dy = gmp4_y1 - gmp4_y0

            gmp4_x0 -= gmp4_dx * gmp_box_frac
            gmp4_x1 += gmp4_dx * gmp_box_frac
            gmp4_y0 -= gmp4_dy * gmp_box_frac
            gmp4_y1 += gmp4_dy * gmp_box_frac

            ax.plot([gmp4_x0, gmp4_x1, gmp4_x1, gmp4_x0, gmp4_x0],
                    [gmp4_y0, gmp4_y0, gmp4_y1, gmp4_y1, gmp4_y0], '-k',
                    linewidth=gmp_box_lw)

            arrow_props = {"fc": label_cmap[4],
                           "ec": label_cmap[4],
                           "lw": 1,
                           "alpha": 0.7,
                           "width": 5,
                           "connectionstyle": 'arc3,rad=-0.2'}
            ax.annotate("", xy=[-15, 27], xycoords=ax.transData,
                        xytext=[-30, -5], textcoords=ax.transData,
                        arrowprops=arrow_props)

        ax.set_title("Cell Type Annotation", pad=bot_title_pad)

        # ax.scatter(*data_Y[bc_idx].T, s=10, edgecolor='k', linewidth=0.2,
        #            facecolor='none')

        fig.tight_layout(pad=fig_pad)
        fig.subplots_adjust(left=0 + fig_ppad,
                            right=1 - fig_ppad,
                            bottom=0 + fig_ppad,
                            top=1 - fig_ppad)

    ###########################################################################
    ## Cell ontology - DBSCAN cluster confusion table
    ###########################################################################
    if True:

        ax = bot_conf_ax

        y_diff = 5

        y_crds = []

        n_cOnt = len(cOnt_remap)

        for lNo, lab in enumerate(unique_labels[1:]):

            color_1 = label_cmap[int(lNo * 2)]
            color_2 = label_cmap[int(lNo * 2) + 1]
            colors = pUtl.make_seq_cmap(color_1, color_2,
                                        n_colors=len(cOnt_by_label[lab]))

            for cNo, cOnt in enumerate(cOnt_by_label[lab]):

                y_crd = y_diff * (n_cOnt - cOnt_remap[cOnt])

                ax.scatter(lNo, y_crd, color=colors[cNo],
                           s=300, edgecolor='0.8')

                ax.text(lNo, y_crd, f"{cOnt_ct_label[lab][cNo]}",
                        ha="center", va='center', fontsize=8)

                y_crds.append(y_crd)

            ax.text(lNo, y_diff * (n_cOnt + 1), f"{lNo + 1}",
                    fontsize=12, ha='center', fontweight='bold')

        if data_name.lower() == 'marrow':
            gmp_y = y_diff * (n_cOnt - cOnt_remap[gmp_label])
            gmp_x0, gmp_x1 = 0.4, 3.6
            gmp_y0, gmp_y1 = gmp_y - y_diff * 0.6, gmp_y + y_diff * 0.6
            ax.plot([gmp_x0, gmp_x1, gmp_x1, gmp_x0, gmp_x0],
                    [gmp_y0, gmp_y0, gmp_y1, gmp_y1, gmp_y0], '-k',
                    linewidth=gmp_box_lw)

        ax.set_xlim(-0.7, len(cOnt_by_label) - 0.3)

        ylim = ax.get_ylim()
        ax.set_ylim(ylim[0], 1.08 * ylim[1])

        ax.set_title("Cluster No.", pad=bot_title_pad)

        fig.tight_layout(pad=fig_pad)
        fig.subplots_adjust(left=0 + fig_ppad,
                            right=1 - fig_ppad,
                            bottom=0 + fig_ppad,
                            top=1 - fig_ppad)

    ###########################################################################
    ## Confusion table annotation rectangles
    ###########################################################################
    if True:

        lax = bot_sctr_ax
        rax = bot_conf_ax

        lax_wnd = lax.get_window_extent()
        rax_wnd = rax.get_window_extent()

        box_wfrac = 0.9
        box_wspace = (1 - box_wfrac) / 2

        w_dist = rax_wnd.x0 - lax_wnd.x1

        box_x0 = rax_wnd.x0 - (w_dist * (box_wfrac + box_wspace))
        box_x1 = rax_wnd.x0 - (w_dist * box_wspace)

        box_x0, _ = rax.transAxes.inverted().transform([box_x0, rax_wnd.y0])
        box_x1, _ = rax.transAxes.inverted().transform([box_x1, rax_wnd.y0])
        box_dx    = box_x1 - box_x0

        new_tf = mtf.blended_transform_factory(rax.transAxes, rax.transData)

        used_cOnt = []
        for lNo, lab in enumerate(unique_labels[1:]):

            color_1 = label_cmap[int(lNo * 2)]

            new_ycrds = []
            for cNo, cOnt in enumerate(cOnt_by_label[lab]):

                if cOnt not in used_cOnt:
                    used_cOnt.append(cOnt)
                else:
                    continue

                y_crd = y_diff * (n_cOnt - cOnt_remap[cOnt])
                new_ycrds.append(y_crd)

                rax.text(box_x0 + box_dx / 2, y_crd,
                         cell_ont_labels[cell_ont_map[cOnt]],
                         ha='center', va='center',
                         fontsize=6, fontweight='bold', transform=new_tf)

            box_y1 = np.max(new_ycrds) + (y_diff / 2)
            box_y0 = np.min(new_ycrds) - (y_diff / 2)
            box_dy = box_y1 - box_y0

            rect = plt.Rectangle((box_x0, box_y0), width=box_dx, height=box_dy,
                                 transform=new_tf, zorder=3,
                                 fill=True, facecolor=color_1,
                                 edgecolor=color_1,
                                 clip_on=False, alpha=0.75)

            rax.add_patch(rect)

    ###########################################################################
    ## Save and Show
    ###########################################################################
    if True:

        pUtl.add_panel_number(top_axes[0], "A",
                              corner=('left', 'top'),
                              edge_pad=10, fontsize=10)

        pUtl.add_panel_number(top_axes[1], "B",
                              corner=('left', 'top'),
                              edge_pad=10, fontsize=10)

        pUtl.add_panel_number(top_axes[2], "C",
                              corner=('left', 'top'),
                              edge_pad=10, fontsize=10)

        pUtl.add_panel_number(bot_sctr_ax, "D",
                              corner=('left', 'top'),
                              edge_pad=10, fontsize=10)

        pUtl.add_panel_number(bot_conf_ax, "E",
                              corner=('left', 'top'),
                              edge_pad=10, fontsize=10,
                              number_loc=(-0.17, None))

        fig.tight_layout(pad=fig_pad)
        fig.subplots_adjust(left=0 + fig_ppad,
                            right=1 - fig_ppad,
                            bottom=0 + fig_ppad,
                            top=1 - fig_ppad)

        print("\n\nSaving Figure!\n\n")
        pUtl.save_figure(fig,
                         fig_base,
                         fig_dir=fig_dir,
                         tight_layout_pad=None,
                         dpi=my_dpi)

        print("Showing Figure!\n\n")
        plt.show()