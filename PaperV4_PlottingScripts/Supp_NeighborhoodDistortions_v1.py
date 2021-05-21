"""
###############################################################################
    Supplemental Figure: Illustration of Neighborhoods and Distortions (v1)
###############################################################################

    Author: Eric Johnson
    Date Created: Sunday, April 4, 2021
    Email: ericjohnson1.2015@u.northwestern.edu

###############################################################################

    In this figure, I want to illustrate how neighborhoods show up in
    different ways in embeddings.  Specifically, I think I want to use sample
    points and then show their neighborhoods and then the change in distance
    between the high and low distributions (PWDY / PWDX).

    I think I'm going to want to show this for a few DRAs and parameters...
     -  t-SNE on MNIST @ kEff = 100
     -  UMAP on MNIST @ k = 100
     -  t-SNE on Marrow @ kEff = 100
     -  UMAP on Marrow @ k = 100

    Since we want to show a couple of sample points each, we might end up with
    a 4 x 4 figure?

    To do this, for each data /parameter set we're going to need the high-dim
    PWD matrix and an embedding.

###############################################################################
"""
import matplotlib
import matplotlib.gridspec as gs
import matplotlib.pyplot as plt
import numpy as np
from openTSNE import TSNEEmbedding
from openTSNE.initialization import random as initRand
from os import path
import PaperV4_PlottingScripts.plotting_utility as pUtl
import pickle as pkl
import seaborn as sns
from sklearn.metrics import pairwise_distances as pwd
from umap import UMAP
# from version_5_0.embedr.
from embedr.affinity import FixedEntropyAffinity
# from version_5_0.embedr.
from embedr.nearest_neighbors import Annoy
import warnings

warnings.filterwarnings("ignore", message="This figure includes Axes that")
warnings.filterwarnings("ignore", message="tight_layout not applied: ")

## Set up directories and files
embed_dir = "./Embeddings/"


def load_tSNE(X,
              name_base=None,
              embed_dir=embed_dir,
              n_embed=1,
              n_components=2,
              perplexity=30,
              early_exag_iter=250,
              n_iter=1000,
              initialization='random',
              n_jobs=-1,
              random_state=1,
              verbose=True):

    if name_base is not None:
        embed_name = name_base + f"_tSNE_{perplexity}_RS{random_state}.pkl"

    n_2_embed = n_embed

    try:
        if (name_base is None):
            raise ValueError

        if verbose:
            print(f"\nTrying to load {embed_name}")

        with open(path.join(embed_dir, embed_name), 'rb') as f:
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

        n_samples, n_features = X.shape

        perplexities = np.asarray(n_samples * [perplexity])
        affObj = FixedEntropyAffinity(perplexity=perplexities,
                                      normalization='pair-wise')
        affObj.fit(X)

        tmp_Y = np.zeros((n_2_embed, n_samples, n_components)).astype(float)

        for eNo in range(n_2_embed):
            if verbose:
                print(f"Generating embedding {eNo + 1} / {n_2_embed}")

            init_Y = initRand(X, random_state=random_state + eNo)

            single_Y = TSNEEmbedding(init_Y, affObj, verbose=True)

            single_Y.optimize(n_iter=early_exag_iter,
                              exaggeration=12,
                              momentum=0.5,
                              inplace=True)
            single_Y.optimize(n_iter=n_iter - early_exag_iter,
                              exaggeration=1.,
                              momentum=0.8,
                              inplace=True)
            tmp_Y[eNo] = single_Y[:]

        if n_2_embed != n_embed:
            Y = np.vstack((Y.reshape(-1, n_samples, n_components), tmp_Y))
        else:
            Y = tmp_Y[:]

        if name_base is not None:
            if verbose:
                print(f"Saving {embed_name} to file!")

            with open(path.join(embed_dir, embed_name), 'wb') as f:
                pkl.dump(Y, f)

    return Y[:n_embed].astype(float).squeeze()


def load_UMAP(X,
              name_base=None,
              embed_dir=embed_dir,
              n_embed=1,
              n_components=2,
              n_neighbors=15,
              min_dist=0.1,
              initialization='random',
              n_jobs=-1,
              random_state=1,
              verbose=True):

    if name_base is not None:
        embed_name = name_base + f"_UMAP_{n_neighbors}_RS{random_state}.pkl"

    n_2_embed = n_embed

    try:
        if (name_base is None):
            raise ValueError

        if verbose:
            print(f"\nTrying to load {embed_name}")

        with open(path.join(embed_dir, embed_name), 'rb') as f:
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

        n_samples, n_features = X.shape

        tmp_Y = np.zeros((n_2_embed, n_samples, n_components)).astype(float)

        for eNo in range(n_2_embed):
            if verbose:
                print(f"Generating embedding {eNo + 1} / {n_2_embed}")

            UMAP_Obj = UMAP(n_components=n_components,
                            n_neighbors=n_neighbors,
                            min_dist=min_dist,
                            init=initialization,
                            n_jobs=n_jobs,
                            random_state=random_state + eNo,
                            verbose=verbose)

            tmp_Y[eNo] = UMAP_Obj.fit_transform(X)[:]

        if n_2_embed != n_embed:
            Y = np.vstack((Y.reshape(-1, n_samples, n_components), tmp_Y))
        else:
            Y = tmp_Y[:]

        if name_base is not None:
            if verbose:
                print(f"Saving {embed_name} to file!")

            with open(path.join(embed_dir, embed_name), 'wb') as f:
                pkl.dump(Y, f)

    return Y[:n_embed].astype(float).squeeze()


###############################################################################
##  RUN THE FILE
###############################################################################
if __name__ == "__main__":

    print_str = "  Plotting PaperV4 Figure: Neighborhood Distortions (v1)  "
    print(f"\n\n" + print_str + "\n" + len(print_str) * "=" + "\n")

    ## Define figure and file name format
    name_base = "V4SuppFig_NeibDistortionsDemo_v1"

    ## Set data and DRA
    data_2_embed = [('Marrow', 'tSNE'),
                    ('Marrow', 'UMAP')]#,
                    # ('MNIST', 'tSNE'),
                    # ('MNIST', 'UMAP')]
    k_NN = 100  ## Number of nearest neighbors (map to perp for t-SNE)

    ## Number of examples to show per condition
    n_2_show = 2

    ## Runtime flags
    show_all_axes = False

    ###########################################################################
    ## Set up Plotting Parameters
    ###########################################################################
    if True:

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

        ## Color maps
        base_cmap   = sns.color_palette()
        cblind_cmap = sns.color_palette('colorblind')
        bright_cmap = sns.color_palette('bright')

        ## Figure-level parameters
        my_dpi      = 400
        fig_width   = 7.5
        fig_height  = 0.6 * fig_width #1.2 * fig_width
        fig_size    = (fig_width, fig_height)
        fig_pad     = 0.2
        fig_ppad    = 0.02

        n_cols = 2 * n_2_show
        n_rows = len(data_2_embed)

        main_hspace = 0.03
        main_wspace = 0.5

        panel_sp_alpha  = 1
        panel_sp_2_show = 'all'

        perc_2_show = [20, 80]

        if show_all_axes:
            spine_alpha = 1
        else:
            spine_alpha = 0

    ###########################################################################
    ## Set up Figure and Gridspec
    ###########################################################################
    if True:

        fig = plt.figure(figsize=fig_size)

        main_gs = fig.add_gridspec(nrows=n_rows, ncols=1, hspace=main_hspace)

        bkgd_axes = []
        row_gridspecs = []
        row_axes = []
        for rowNo in range(n_rows):
            ax = fig.add_subplot(main_gs[rowNo])
            ax = pUtl.make_border_axes(ax, spine_alpha=spine_alpha)

            bkgd_axes.append(ax)

            row_gs = gs.GridSpec(nrows=1, ncols=n_cols, wspace=main_wspace)
            row_gridspecs.append(row_gs)

            row_row_axes = []
            for colNo in range(n_cols):
                ax = fig.add_subplot(row_gs[colNo])
                ax = pUtl.make_border_axes(ax, spine_alpha=panel_sp_alpha,
                                           spines_2_show=panel_sp_2_show)
                row_row_axes.append(ax)
            row_axes.append(row_row_axes)

        fig.tight_layout(pad=fig_pad)
        fig.subplots_adjust(left=0 + fig_ppad,
                            right=1 - fig_ppad,
                            bottom=0 + fig_ppad,
                            top=1 - fig_ppad)

        for rowNo in range(n_rows):
            pUtl.update_tight_bounds(fig, row_gridspecs[rowNo], main_gs[rowNo],
                                     w_pad=main_wspace)

    ###########################################################################
    ## Run the loop (load data, embeds, and plot!)
    ###########################################################################
    if True:

        ## Set other parameters
        n_components   = 2

        tSNE_exag_iter = 250
        tSNE_n_iter    = 1000 - tSNE_exag_iter

        random_seed    = 1
        initialization = 'random'
        n_jobs         = -1

        alpha_nu   = 0.01  ## Fraction of closest neighbor that's 'non-uniform'

        for ii, (data_name, DRA) in enumerate(data_2_embed):
            print(f"\n\nLoading {data_name} to be embedded by {DRA}!")

            if data_name.lower() == 'marrow':
                data_dir = f"./Data/TabulaMuris/FACS/"
                data_file = f"Marrow_PCA_Embeddings.csv"
                data_base = name_base + "_TabulaMuris_Marrow"

                kEff_name = f"TabulaMuris_Marrow_PCs_kEffective.pkl"

            elif data_name.lower() == 'mnist':

                data_dir   = f"./Data/"
                data_file  = f"mnist2500_X.txt"
                data_base = name_base + "_mnist2500"

                kEff_name  = f"mnist2500_kEffective.pkl"

            X, metadata = pUtl.load_data(data_name, data_dir)

            n_samples, n_features = X.shape

            ## Get perp to eff_kNN mapping
            sorted_PWD = pwd(X, metric='sqeuclidean')
            sorted_PWD = np.sort(sorted_PWD, axis=1)[:, 1:]

            kEff_arr, perp_arr = pUtl.get_kEff(file_name=kEff_name,
                                               file_dir=data_dir)

            kNN_name = data_base + "_kNN_graph.pkl"
            try:
                with open(path.join(embed_dir, kNN_name), 'rb') as f:
                    HD_idx, HD_dists = pkl.load(f)
            except FileNotFoundError:
                nn_graph = Annoy(metric='euclidean', n_jobs=n_jobs,
                                 random_state=random_seed, verbose=True)
                HD_idx, HD_dists = nn_graph.fit(X, k_NN=n_samples - 1)

                with open(path.join(embed_dir, kNN_name), 'wb') as f:
                    pkl.dump([HD_idx, HD_dists], f)

            if DRA.lower() == 'tsne':
                perp = pUtl.get_perp_from_kEff(k_NN, kEff_arr, perp_arr)

                Y = load_tSNE(X,
                              name_base=data_base,
                              embed_dir=embed_dir,
                              n_embed=1,
                              n_components=n_components,
                              perplexity=perp,
                              early_exag_iter=tSNE_exag_iter,
                              n_iter=tSNE_n_iter,
                              initialization=initialization,
                              n_jobs=n_jobs,
                              random_state=random_seed,
                              verbose=True)

            if DRA.lower() == 'umap':
                Y = load_UMAP(X,
                              name_base=data_base,
                              embed_dir=embed_dir,
                              n_embed=1,
                              n_components=n_components,
                              n_neighbors=k_NN,
                              min_dist=0.1,
                              initialization=initialization,
                              n_jobs=n_jobs,
                              random_state=random_seed,
                              verbose=True)

            bkgd_ax = bkgd_axes[ii]
            panel_axes = row_axes[ii]

            for eNo in range(n_2_show):
                perc = perc_2_show[eNo]

                Y_perc = np.percentile(Y, perc, axis=0)

                pIdx = np.argmin(np.sum((Y - Y_perc)**2, axis=1))

                Y_perc = Y[pIdx]

                LD_dists = np.sqrt(np.sum((Y - Y_perc)**2, axis=1))

                dist_colors = np.zeros(n_samples)
                dist_colors[HD_idx[pIdx]] = np.log10(HD_dists[pIdx])

                sort_idx = np.argsort(-dist_colors)
                dist_colors[pIdx] = np.nan

                ax1 = panel_axes[eNo * 2]
                ax2 = panel_axes[eNo * 2 + 1]

                ax1.scatter(*Y.T, color='lightgrey', s=3, alpha=0.7)

                ax1.scatter(*Y[sort_idx[:k_NN]].T, c='C0', s=4, alpha=1)

                ax1.scatter(*Y[sort_idx[-k_NN:]].T, c='C1', s=4, alpha=1)

                ax1.scatter(*Y[pIdx].T, c='k', marker='*', s=40, alpha=1)

                h2 = ax2.scatter(*Y[sort_idx].T, c=dist_colors[sort_idx],
                                 cmap='magma_r', s=3, alpha=0.7)

                ax2.scatter(*Y[pIdx].T, c='k', marker='*', s=40, alpha=1)

                ch = fig.colorbar(h2, ax=ax2, pad=0)

                ch.set_label("Log-Dist to Select Point", fontsize=8)
                ch.ax.tick_params(labelsize=6)

            bkgd_ax.set_title(f"{data_name} Embedded by {DRA} @ k = {k_NN}",
                              fontsize=12, pad=-5)

    ###########################################################################
    ## Save and Show!
    ###########################################################################
    if True:

        fig.tight_layout(pad=fig_pad)
        for rowNo in range(n_rows):
            pUtl.update_tight_bounds(fig, row_gridspecs[rowNo], main_gs[rowNo],
                                     w_pad=main_wspace)

        fig.tight_layout(pad=fig_pad)
        for rowNo in range(n_rows):
            pUtl.update_tight_bounds(fig, row_gridspecs[rowNo], main_gs[rowNo],
                                     w_pad=main_wspace)

        # SAVE FIGURE HERE
        pUtl.save_figure(fig,
                         name_base,
                         fig_dir=fig_dir,
                         tight_layout_pad=fig_pad,
                         dpi=my_dpi)

        plt.show()
