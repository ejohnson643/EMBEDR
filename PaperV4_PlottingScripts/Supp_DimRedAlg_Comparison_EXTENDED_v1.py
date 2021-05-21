"""
###############################################################################
    Figure: Dimensionality Reduction Algorithm Comparison (v1) EXTENDED
###############################################################################

    Author: Eric Johnson
    Date Created: Thursday, April 22, 2021
    Email: ericjohnson1.2015@u.northwestern.edu

###############################################################################

    This figure will show EMBEDR applied to several other dimensionality
    reduction algorithms.  In particular, we'll look at:
     -  PCA
     -  LLE
     -  Modified LLE
     -  MDS
     -  Isomap
     -  Spectral Embedding (Laplacian Eigenmap)

    These will all be run at default parameters and will embed into 2D.

    To run this, we need to compute a high-dimensional affinity matrix.  We'll
    use perplexity = 250, since that corresponds to a minimum in the global
    sweep.

###############################################################################
"""

from embedr.affinity import FixedEntropyAffinity
import matplotlib
import matplotlib.gridspec as gs
import matplotlib.pyplot as plt
import numpy as np
import numpy.random as r
import os
import PaperV4_PlottingScripts.plotting_utility as pUtl
import pickle as pkl
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import LocallyLinearEmbedding as LLE
from sklearn.manifold import Isomap
from sklearn.manifold import SpectralEmbedding as SE
from sklearn.metrics import pairwise_distances as pwd
from sklearn.random_projection import SparseRandomProjection as SRP
import time
import warnings

EPSILON = np.finfo(np.float64).eps

warnings.filterwarnings("ignore", message="This figure includes Axes that")
warnings.filterwarnings("ignore", message="tight_layout not applied: ")


def calc_DKL(P, eY):

    P_data, row_idx, col_idx = P.data, P.indptr, P.indices

    if eY.ndim == 2:
        eY = eY[np.newaxis, :, :]

    n_embed, n_samples, _ = eY.shape
    DKL = np.zeros((n_embed, n_samples))

    for eNo, Y in enumerate(eY):

        Q = 1 / (1 + pwd(Y, metric="sqeuclidean"))
        Q = Q / Q.sum(axis=1)[:, np.newaxis]

        for rowNo, [start, end] in enumerate(zip(row_idx[:-1], row_idx[1:])):
            colNos = col_idx[start:end]
            P_row = P_data[start:end]
            Q_row = Q[rowNo, colNos]

            DKL_row = np.log(P_row + EPSILON) - np.log(Q_row + EPSILON)
            DKL[eNo, rowNo] = np.sum(P_row * DKL_row).squeeze()

    return DKL


###############################################################################
##  RUN THE FILE
###############################################################################
if __name__ == "__main__":

    print_str = "  Plotting PaperV4 Figure: DRA Extended Comparison (v1)  "
    print(f"\n\n" + print_str + "\n" + len(print_str) * "=" + "\n")

    ## Define figure and file name format
    name_base = "V4SuppFig_DimRedAlg_Comparison_Extended_v1"

    ## Select which data to use
    seq_type = "FACS"
    tissue = "Marrow"

    ## Dimension into which we embed!
    n_components = 2

    ## How many nulls to use
    n_null_embed = 10

    ## Set random seeds
    aff_seed   = 54321
    null_seed  = 12345
    embed_seed = 1

    ## Which DR algorithms to use.  The values are pre-initialized objects.
    DR_methods = {'PCA': PCA(n_components=n_components),
                  # 'LLE': LLE(method='standard'),
                  # 'Modified LLE': LLE(method='modified'),
                  'Isomap': Isomap(n_components=n_components),
                  'Random Projection': SRP(n_components=n_components,
                                           random_state=embed_seed),
                  'Spectral Embedding': SE(n_components=n_components)}

    ## Runtime flags
    show_all_axes = False

    ###########################################################################
    ## Load the data and metadata!
    ###########################################################################
    if True:
        ## Set up directories and files
        embed_dir = "./Embeddings/"

        if tissue == "MNIST":
            data_dir   = f"./Data/"
            data_file  = f"mnist2500_X.txt"
            label_file = f"mnist2500_labels.txt"
            kEff_name  = f"mnist2500_kEffective.pkl"
            fig_base   = name_base + f"_MNIST"

        else:
            data_dir = f"./Data/TabulaMuris/{seq_type}/"
            data_file = f"{tissue}_PCA_Embeddings.csv"
            metadata_file = f"{tissue}_Metadata.csv"
            kEff_name = f"TabulaMuris_{tissue}_PCs_kEffective.pkl"
            fig_base   = name_base + f"_TabulaMuris_{tissue}_{seq_type}"

        X, metadata = pUtl.load_data(tissue, data_dir)

        n_samples, n_features = X.shape

        ## Generate Null Data Resample
        r.seed(null_seed)
        null_X = np.zeros((n_null_embed, n_samples, n_features))
        for nNo in range(n_null_embed):
            null_X[nNo] = np.asarray([r.choice(col, size=n_samples)
                                      for col in X.T]).T

        alpha_nu   = 0.01  ## Fraction of closest neibthat 'non-uniform'

        kEff_arr, perp_arr = pUtl.get_kEff(file_name=kEff_name,
                                           file_dir=data_dir,
                                           verbose=True)

    ###########################################################################
    ## Generate / Load Affinity Matrices
    ###########################################################################
    if True:

        aff_perp = 250

        aff_params = {'perplexity': aff_perp,
                      'normalization': 'point-wise',
                      'symmetrize': False,
                      'n_jobs': -1,
                      'n_neighbors': n_samples - 1,
                      'random_state': aff_seed,
                      'verbose': 5}

        ## Data affinity matrix
        aff_data_name = name_base + f"_data_affinity.pkl"
        aff_data_path = os.path.join(embed_dir, aff_data_name)

        try:
            print(f"\nTrying to load affinity matrix...")

            with open(aff_data_path, 'rb') as f:
                dAff = pkl.load(f)

            print(f"... affinity matrix loaded successfully!")

        except:
            print(f"... affinity matrix couldn't be loaded... Recalculating!")

            dAff = FixedEntropyAffinity(**aff_params)
            dAff.fit(X)

            with open(aff_data_path, 'wb') as f:
                pkl.dump(dAff, f)

            print(f"... Done!  Affinity matrix saved to file!")


        nAff_dict = {}
        for nNo in range(n_null_embed):

            print(f"\nGetting affinity matrix for null!"
                  f" {nNo + 1}/{n_null_embed}")

            aff_null_name = name_base + f"_null_affinity_{nNo}.pkl"
            aff_null_path = os.path.join(embed_dir, aff_null_name)

            try:
                print(f"\nTrying to load affinity matrix...")

                with open(aff_null_path, 'rb') as f:
                    nAff = pkl.load(f)

                print(f"... affinity matrix loaded successfully!")

            except (FileNotFoundError):
                print(f"... affinity matrix couldn't be loaded..."
                      f" Recalculating!")

                nAff = FixedEntropyAffinity(**aff_params)
                nAff.fit(null_X[nNo])

                with open(aff_null_path, 'wb') as f:
                    pkl.dump(nAff, f)

                print(f"... Done!  Affinity matrix saved to file!")

            nAff_dict[nNo] = nAff

    ###########################################################################
    ## Generate / Load Embeddings
    ###########################################################################
    if True:

        data_Y = {}
        null_Y = {}

        for algNo, DRA in enumerate(DR_methods.keys()):

            print(f"\nGenerating embeddings with {DRA}!")
            DRAObj = DR_methods[DRA]

            ## Data embedding
            emb_data_name = name_base + f"_{DRA}_embed_data_Y.pkl"
            emb_data_path = os.path.join(embed_dir, emb_data_name)

            try:
                print(f"\nTrying to load DATA embedding...")
                with open(emb_data_path, 'rb') as f:
                    dY = pkl.load(f)

            except FileNotFoundError:

                start = time.time()
                dY = DRAObj.fit_transform(X)
                end = time.time()

                print(f"\nFitting DATA with {DRA} took"
                      f" {end - start:.2g} seconds!")

                with open(emb_data_path, 'wb') as f:
                    pkl.dump(dY, f)

            data_Y[DRA] = dY.copy()

            ## Null embeddings
            nY = np.zeros((n_null_embed, n_samples, n_components))

            for nNo in range(n_null_embed):
                print(f"\nTrying to load NULL {nNo+1}/{n_null_embed} embed...")
                emb_null_name = name_base + f"_{DRA}_embed_null_Y_{nNo}.pkl"
                emb_null_path = os.path.join(embed_dir, emb_null_name)

                try:
                    with open(emb_null_path, 'rb') as f:
                        nY[nNo] = pkl.load(f)

                except FileNotFoundError:

                    start = time.time()
                    nY[nNo] = DRAObj.fit_transform(null_X[nNo])
                    end = time.time()

                    print(f"\nFitting NULL {nNo + 1} with {DRA} took"
                          f" {end - start:.2g} seconds!")

                    with open(emb_null_path, 'wb') as f:
                        pkl.dump(nY[nNo], f)

            null_Y[DRA] = nY.copy()

    ###########################################################################
    ## Generate / Load EES and p-Values
    ###########################################################################
    if True:

        data_EES = {}
        null_EES = {}
        pVal_dict = {}

        for algNo, DRA in enumerate(DR_methods.keys()):
            print(f"\nCalculating EES for {DRA}!")

            ## Load data EES
            EES_data_name = name_base + f"_{DRA}_EES_data_Y.pkl"
            EES_data_path = os.path.join(embed_dir, EES_data_name)

            try:
                print(f"\nTrying to load data EES...")

                with open(EES_data_path, 'rb') as f:
                    dEES = pkl.load(f)

                print(f"... EES loaded succesfully!")

            except FileNotFoundError:

                print(f"... EES couldn't be loaded!  Recalculating!")

                dEES = calc_DKL(dAff.P, data_Y[DRA])

                with open(EES_data_path, 'wb') as f:
                    pkl.dump(dEES, f)

            data_EES[DRA] = dEES.copy()

            ## Load null EES
            nEES = np.zeros((n_null_embed, n_samples))

            for nNo in range(n_null_embed):
                print(f"\nTrying to load NULL {nNo+1}/{n_null_embed} EES...")
                EES_null_name = name_base + f"_{DRA}_EES_null_Y_{nNo}.pkl"
                EES_null_path = os.path.join(embed_dir, EES_null_name)

                try:
                    print(f"\nTrying to load null EES...")

                    with open(EES_null_path, 'rb') as f:
                        nEES[nNo] = pkl.load(f)

                    print(f"... EES loaded succesfully!")

                except FileNotFoundError:

                    print(f"... EES couldn't be loaded!  Recalculating!")

                    nP = nAff_dict[nNo].P
                    nEES[nNo] = calc_DKL(nP, null_Y[DRA][nNo]).squeeze()

                    with open(EES_null_path, 'wb') as f:
                        pkl.dump(nEES[nNo], f)

            null_EES[DRA] = nEES.copy()

            pVals = pUtl.calc_emp_pVals(dEES, nEES, summary_method='average')

            pVal_dict[DRA] = pVals[1].copy()

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
        fig_height = 0.8 * fig_width
        fig_size   = (fig_width, fig_height)

        ## Figure parameters
        fig_pad          = 0.3   ## Fraction of a fontwidth, iirc
        fig_ppad         = 0.01  ## Percent of fig to leave around edge.

        main_rows        = 2
        main_cols        = 2
        main_wratios     = main_cols * [1.] + [0.1, 0.12]
        main_wpad        = 0.01
        main_hpad        = 0.01
        main_spine_alpha = 1.

        ## Colorbar properties
        cbar_pad         = 0.5
        cbar_wpad        = 0.2
        cbar_hpad        = 0.7
        cbar_tickl       = 0
        cbar_tickw       = 0
        cbar_ticksize    = 8
        cbar_tickpad     = 2

        sctr_size        = 2
        sctr_alpha       = 0.8
        sctr_ylim_pad    = 0.12
        sctr_title_pad   = -14

        ## Text box parameters
        text_y           = 0.011
        text_fs          = 8
        text_fw          = 'bold'
        text_rect_pad    = 3
        disp_dpi         = 72

        ## p-Value Colors
        min_pVal = -np.log10(1 / (n_samples * n_null_embed)) + 0.005
        pVal_clr_change = [0, 1, 2, 3, min_pVal]
        pVal_clr_idx = [4, 0, 3, 2]
        [pVal_cmap,
         pVal_cnorm] = pUtl.make_categ_cmap(change_points=pVal_clr_change,
                                            cmap_idx=pVal_clr_idx,
                                            max_diverge=0.01)

        ## Color maps
        base_cmap   = sns.color_palette()
        cblind_cmap = sns.color_palette('colorblind')
        bright_cmap = sns.color_palette('bright')

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
        main_gs = fig.add_gridspec(main_rows, main_cols + 2,
                                   width_ratios=main_wratios)

        main_axes = []
        for rowNo in range(main_rows):
            main_ax_row = []
            for colNo in range(main_cols):
                ax = fig.add_subplot(main_gs[rowNo, colNo])
                ax = pUtl.make_border_axes(ax, spine_alpha=main_spine_alpha)

                main_ax_row.append(ax)
            main_axes.append(main_ax_row)

        # for rowNo in range(main_rows):
        #     ax = = fig.add_subplot(main_gs[rowNo, -1])
        #     ax = pUtl.make_border_axes(ax, spine_color=spine_alpha)

        colorbar_gs = main_gs[:, -2].subgridspec(nrows=2, ncols=1)

        cax_bkgd1 = fig.add_subplot(colorbar_gs[0])
        cax_bkgd1 = pUtl.make_border_axes(cax_bkgd1, spine_alpha=spine_alpha)

        cax_gs1 = gs.GridSpec(nrows=1, ncols=1)

        cax_ax1 = fig.add_subplot(cax_gs1[0])
        cax_ax1 = pUtl.make_border_axes(cax_ax1, spine_alpha=1,
                                        spine_color='k')

        cax_bkgd2 = fig.add_subplot(colorbar_gs[1])
        cax_bkgd2 = pUtl.make_border_axes(cax_bkgd2, spine_alpha=spine_alpha)

        cax_gs2 = gs.GridSpec(nrows=1, ncols=1)

        cax_ax2 = fig.add_subplot(cax_gs2[0])
        cax_ax2 = pUtl.make_border_axes(cax_ax2, spine_alpha=spine_alpha)

        def update_bounds():

            fig.tight_layout(pad=fig_pad)
            fig.subplots_adjust(left=0 + fig_ppad,
                                right=1 - fig_ppad,
                                bottom=0 + fig_ppad,
                                top=1 - fig_ppad)

            pUtl.update_tight_bounds(fig, cax_gs1, colorbar_gs[0],
                                     fig_pad=fig_pad,
                                     inner_pad=cbar_pad,
                                     w_pad=cbar_wpad,
                                     h_pad=cbar_hpad)

            pUtl.update_tight_bounds(fig, cax_gs2, colorbar_gs[1],
                                     fig_pad=fig_pad,
                                     inner_pad=cbar_pad,
                                     w_pad=cbar_wpad,
                                     h_pad=cbar_hpad)

        update_bounds()

    ###########################################################################
    ## PLOT
    ###########################################################################
    if True:

        for algNo, DRA in enumerate(DR_methods.keys()):
            rowNo = int(algNo / main_cols)
            colNo = int(algNo % main_cols)

            ax = main_axes[rowNo][colNo]

            pVals = pVal_dict[DRA]
            sort_idx = np.argsort(pVals)[::-1]

            clrs = -np.log10(pVals)[sort_idx]

            dY = data_Y[DRA]

            hax = ax.scatter(*dY[sort_idx].T, c=clrs, s=sctr_size,
                             alpha=sctr_alpha, cmap=pVal_cmap, norm=pVal_cnorm)

            ax.set_title(f"{DRA}", pad=sctr_title_pad)

            if colNo == (main_cols - 2):
                if rowNo == 0:
                    cax = cax_ax1
                else:
                    cax = cax_ax2

                hcb = plt.colorbar(hax, cax=cax)

                hcb.set_ticks([1, 2, 3, 4])
                hcb.set_ticklabels([r"$0.1$", r"$0.01$",
                                    r"$10^{-3}$", r"$10^{-4}$"])
                hcb.ax.tick_params(length=cbar_tickl, width=cbar_tickl,
                                   labelsize=cbar_ticksize, pad=cbar_tickpad)

                hcb.set_label(r"EMBEDR $p$-Value")

            ylim = ax.get_ylim()
            ylim_extend = sctr_ylim_pad * (ylim[1] - ylim[0])
            ax.set_ylim(ylim[0] - ylim_extend, ylim[1] + ylim_extend)

            ax_width = ax.get_window_extent().width
            ax_height = ax.get_window_extent().height / fig.dpi

            n_boxes = len(pVal_clr_change) - 1

            n_good = np.zeros(n_boxes).astype(int)
            for ii in range(n_boxes):
                good_idx  = -np.log10(pVals) >= pVal_clr_change[ii]
                good_idx *= -np.log10(pVals) < pVal_clr_change[ii + 1]
                n_good[ii] = np.sum(good_idx)
            best_ii = np.argmax(n_good)

            rect_width = 1 / n_boxes
            for ii in range(n_boxes):
                text_x = rect_width / 2 + (ii * rect_width)
                rect_x = ii * rect_width

                text = f"{n_good[ii]:,} ({n_good[ii] / n_samples:.0%})"

                if ii == best_ii:
                    text_fw = 'bold'
                else:
                    text_fw = 'normal'

                text_h = ax.text(text_x, text_y, text, ha='center',
                                 va='bottom', fontsize=text_fs,
                                 fontweight=text_fw, transform=ax.transAxes,
                                 color='w')

                rect_height = text_h.get_size() + 2 * text_rect_pad
                rect_height /= (disp_dpi * ax_height)

                rect = plt.Rectangle((rect_x, 0),
                                     width=rect_width,
                                     height=rect_height,
                                     transform=ax.transAxes,
                                     fill=True,
                                     facecolor=cblind_cmap[pVal_clr_idx[ii]],
                                     clip_on=False,
                                     edgecolor='w')

                ax.add_patch(rect)

    ###########################################################################
    ## Save and Show!
    ###########################################################################
    if True:

        update_bounds()

        fig_name = "V4SuppFig_DimRedAlg_Comparison_Extended_v1"

        # SAVE FIGURE HERE
        pUtl.save_figure(fig,
                         fig_base,
                         fig_dir=fig_dir,
                         tight_layout_pad=fig_pad,
                         dpi=my_dpi)

        plt.show()

