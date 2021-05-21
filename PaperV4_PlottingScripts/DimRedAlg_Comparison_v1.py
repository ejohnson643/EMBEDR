"""
###############################################################################
    Figure: Dimensionality Reduction Algorithm Comparison (v1)
###############################################################################

    Author: Eric Johnson
    Date Created: Sunday, March 28, 2021
    Email: ericjohnson1.2015@u.northwestern.edu

###############################################################################

    This figure is meant to demonstrate that EMBEDR can be used to compare
    dimensionality reduction algorithms.  We're going to run the method on
    t-SNE and UMAP at a couple of perplexties, showing the embeddings colored
    by p-value with EES distributions below them.

    Specifically, we're going to do

     t-SNE @ default  |||  UMAP @ default
    ------------------|||-----------------
     t-SNE @ 1200     |||  UMAP @ k=1600

    Each panel will have the embedding, along with the EES distributions below.
    For the EES distributions, we should just show the KDE and the AUC.

###############################################################################
"""

from collections import Counter
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
from sklearn.metrics import pairwise_distances as pwd
from sklearn.metrics import auc
from sklearn.preprocessing import normalize
import warnings

EPSILON = np.finfo(np.float64).eps

warnings.filterwarnings("ignore", message="This figure includes Axes that")
warnings.filterwarnings("ignore", message="tight_layout not applied: ")


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



def get_eCDF(data, extend=False):
    counts = Counter(data.ravel())
    vals = np.msort(list(counts.keys()))
    CDF = np.cumsum([counts[val] for val in vals])
    CDF = CDF / CDF[-1]

    if extend:
        data_range = vals[-1] - vals[0]
        vals = [vals[0] - (0.01 * data_range)] + list(vals)
        vals = np.asarray(vals + [vals[-1] + (0.01 * data_range)])
        CDF = np.asarray([0] + list(CDF) + [1])

    return vals, CDF


def get_QQ_vals(data1, data2):

    vals1, CDF1 = get_eCDF(data1, extend=True)
    vals2, CDF2 = get_eCDF(data2, extend=True)

    joint_vals = np.msort(np.unique(np.hstack((vals1, vals2))))

    joint_CDF1 = np.zeros_like(joint_vals)
    joint_CDF2 = np.zeros_like(joint_vals)

    id1, id2 = 0, 0
    for ii, val in enumerate(joint_vals):

        joint_CDF1[ii] = CDF1[id1]
        if (val in vals1) and (id1 + 1 < len(vals1)):
            id1 += 1

        joint_CDF2[ii] = CDF2[id2]
        if (val in vals2) and (id2 + 1 < len(vals2)):
            id2 += 1

    return joint_vals, joint_CDF1, joint_CDF2


###############################################################################
##  RUN THE FILE
###############################################################################
if __name__ == "__main__":

    print_str = "  Plotting PaperV4 Figure: DimRed Alg Comparison (v1)  "
    print(f"\n\n" + print_str + "\n" + len(print_str) * "=" + "\n")

    ## Define figure and file name format
    name_base = "V4Fig_DimRed_Algorithm_Comparison_v1"

    ## Set parameters at which to plot data
    DR_params = [('tSNE', 30),
                 ('UMAP', 15),
                 ('tSNE', 250),
                 ('UMAP', 400)]

    ## Set random seeds
    global_seed = 12345
    random_seed = 1
    null_seed   = 100

    ## P-Valuye calculation methods
    pVal_method = "average"

    ## Runtime flags
    show_all_axes = False

    ###########################################################################
    ## Load the data and metadata!
    ###########################################################################
    if True:

        ## Set the tissue and seq-type
        seq_type = "FACS"
        tissue   = "Marrow"

        ## Set the embedding directory
        embed_dir     = f"./Embeddings/Data/"

        ## Data file names
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

        meta_dict = pUtl.parse_metadata(metadata)

        n_samples, n_features = X.shape

        ## Generate Null Data Resample
        r.seed(global_seed)
        null_X = np.array([r.choice(col, size=n_samples) for col in X.T]).T

        ## Get perp to eff_kNN mapping
        sorted_PWD = pwd(X, metric='sqeuclidean')
        sorted_PWD = np.sort(sorted_PWD, axis=1)[:, 1:]
        print(f"SORTED PWD IS SQUARED EUCLIDEAN")

        ## Set perplexity array!
        # perp_arr = np.logspace(1, np.log10(5037), 31).astype(int)
        # perp_arr = sorted(perp_arr)
        # perp_arr = np.hstack((perp_arr[:3], perp_arr[4:]))

        alpha_nu = 0.01
        kEff_arr, perp_arr = pUtl.get_kEff(file_name=kEff_name,
                                           file_dir=data_dir,
                                           alpha_nu=alpha_nu)

    ###########################################################################
    ## Generate / Load Embeddings
    ###########################################################################
    if True:

        ## Set other parameters
        n_components     = 2
        n_data_embed     = 5
        n_null_embed     = 10
        n_jobs           = -1
        initialization   = 'random'

        ## t-SNE runtime parameters
        early_exag_iter = 250
        n_iter = 750
        momentum = 0.8

        data_Y = {}
        null_Y = {}

        for algNo, (alg, param) in enumerate(DR_params):
            print(f"\nLoading / Generating {alg} embedding (param = {param})")

            if alg.lower() == 'tsne':

                dY = pUtl.load_tSNE(X,
                                    name_base=fig_base + "_DataEmbed",
                                    embed_dir=embed_dir,
                                    n_embed=n_data_embed,
                                    n_components=n_components,
                                    perplexity=param,
                                    early_exag_iter=early_exag_iter,
                                    n_iter=n_iter,
                                    initialization=initialization,
                                    n_jobs=n_jobs,
                                    random_state=random_seed,
                                    verbose=True)

                nY = pUtl.load_tSNE(null_X,
                                    name_base=fig_base + "_NullEmbed",
                                    embed_dir=embed_dir,
                                    n_embed=n_null_embed,
                                    n_components=n_components,
                                    perplexity=param,
                                    early_exag_iter=early_exag_iter,
                                    n_iter=n_iter,
                                    initialization=initialization,
                                    n_jobs=n_jobs,
                                    random_state=null_seed,
                                    verbose=True)

            if alg.lower() == 'umap':

                dY = pUtl.load_UMAP(X,
                                    name_base=fig_base + "_DataEmbed",
                                    embed_dir=embed_dir,
                                    n_embed=n_data_embed,
                                    min_dist=0.1,
                                    n_components=n_components,
                                    n_neighbors=param,
                                    initialization=initialization,
                                    n_jobs=n_jobs,
                                    random_state=random_seed,
                                    verbose=True)

                nY = pUtl.load_UMAP(null_X,
                                    name_base=fig_base + "_NullEmbed",
                                    embed_dir=embed_dir,
                                    n_embed=n_null_embed,
                                    min_dist=0.1,
                                    n_components=n_components,
                                    n_neighbors=param,
                                    initialization=initialization,
                                    n_jobs=n_jobs,
                                    random_state=random_seed,
                                    verbose=True)

            data_Y[(alg, param)] = dY[:]
            null_Y[(alg, param)] = nY[:]

            del dY
            del nY

    ###########################################################################
    ## Load / Create Affinity Matrix
    ###########################################################################
    if True:

        data_P = {}
        null_P = {}

        for algNo, (alg, param) in enumerate(DR_params):
            print(f"\nLoading / Generating Affinity Matrix (param = {param})")

            if alg.lower() == 'umap':

                perp = pUtl.get_perp_from_kEff(param, kEff_arr, perp_arr)
                perp = int(pUtl.human_round(perp)[0])

            else:
                perp = int(param)

            ## Set filenames and paths
            x_str = f"_data_X_perp{perp}"
            annoy_data_name = fig_base + f"_ANNOY_index{x_str}.obj"
            annoy_data_path = os.path.join(embed_dir, annoy_data_name)
            aff_data_name   = fig_base + f"_base_affinity{x_str}.pkl"
            aff_data_path   = os.path.join(embed_dir, aff_data_name)

            try:
                print(f"\nTrying to load affinity matrix object...")

                ## First, try and load the affinity matrix
                with open(aff_data_path, 'rb') as f:
                    aff_dX = pkl.load(f)

                print(f"... aff mat loaded, getting kNN index...")

                ## Then try and fix the ANNOY index
                kNN = aff_dX.kNN_index._initialize_ANNOY_index(n_features)
                aff_dX.kNN_index.indices = kNN
                aff_dX.kNN_index.indices.load(annoy_data_path)

                print(f"... kNN index loaded successfully!")

            ## If it can't be loaded, then recompute.
            except (FileNotFoundError, OSError):
                print(f"... couldn't load!  Recalculating...")

                aff_dX = FixedEntropyAffinity(perplexity=perp,
                                              normalization='point-wise',
                                              n_jobs=-1,
                                              random_state=random_seed,
                                              verbose=5)
                aff_dX.fit(X)
                aff_dX.kNN_index.indices.save(annoy_data_path)

                with open(aff_data_path, 'wb') as f:
                    pkl.dump(aff_dX, f)

                print(f"... done!  Saved to file!")

            ## Set filenames and paths
            x_str = f"_null_X_perp{perp}"
            annoy_null_name = fig_base + f"_ANNOY_index{x_str}.obj"
            annoy_null_path = os.path.join(embed_dir, annoy_null_name)
            aff_null_name   = fig_base + f"_base_affinity{x_str}.pkl"
            aff_null_path   = os.path.join(embed_dir, aff_null_name)

            try:
                print(f"\nTrying to load affinity matrix object...")

                ## First, try and load the affinity matrix
                with open(aff_null_path, 'rb') as f:
                    aff_nX = pkl.load(f)

                print(f"... aff mat loaded, getting kNN index...")

                ## Then try and fix the ANNOY index
                kNN = aff_nX.kNN_index._initialize_ANNOY_index(n_features)
                aff_nX.kNN_index.indices = kNN
                aff_nX.kNN_index.indices.load(annoy_null_path)

                print(f"... kNN index loaded successfully!")

            ## If it can't be loaded, then recompute.
            except (FileNotFoundError, OSError):
                print(f"... couldn't load!  Recalculating...")

                aff_nX = FixedEntropyAffinity(perplexity=perp,
                                              normalization='point-wise',
                                              n_jobs=-1,
                                              random_state=null_seed,
                                              verbose=5)
                aff_nX.fit(null_X)
                aff_nX.kNN_index.indices.save(annoy_null_path)

                with open(aff_null_path, 'wb') as f:
                    pkl.dump(aff_nX, f)

                print(f"... done!  Saved to file!")

            data_P[(alg, param)] = aff_dX.P.copy()
            null_P[(alg, param)] = aff_nX.P.copy()

            del aff_dX
            del aff_nX

    ###########################################################################
    ## Load / Calculate p-Values
    ###########################################################################
    if True:

        data_EES = {}
        null_EES = {}
        pValues  = {}

        for algNo, (alg, param) in enumerate(DR_params):
            print(f"\nLoading / Calculating p-Values for ({alg}, {param})")

            pVal_name = fig_base + f"_{alg}_{param}_pVal_dict.pkl"
            pVal_path = os.path.join(embed_dir, pVal_name)
            try:
                with open(pVal_path, 'rb') as f:
                    pVal_dict = pkl.load(f)

                data_EES[(alg, param)] = pVal_dict['data_EES']
                null_EES[(alg, param)] = pVal_dict['null_EES']
                pValues[(alg, param)]  = pVal_dict['pValues']

                del pVal_dict

            except (FileNotFoundError, KeyError):

                print(f"Couldn't be loaded!  Recalculating!")

                dP = data_P[(alg, param)]
                normalize(dP, norm='l1', axis=1, copy=False)

                dY = data_Y[(alg, param)]
                dEES = calc_DKL(dP.data, dP.indptr, dP.indices, dY)

                nP = null_P[(alg, param)]
                normalize(nP, norm='l1', axis=1, copy=False)

                nY = null_Y[(alg, param)]
                nEES = calc_DKL(nP.data, nP.indptr, nP.indices, nY)

                data_EES[(alg, param)] = dEES.copy()
                null_EES[(alg, param)] = nEES.copy()

                _, pVs = pUtl.calc_emp_pVals(dEES, nEES,
                                             summary_method=pVal_method)

                pValues[(alg, param)] = pVs.copy()

                with open(pVal_path, 'wb') as f:
                    pkl.dump({'data_EES': dEES,
                              'null_EES': nEES,
                              'pValues':  pVs}, f)

    ###########################################################################
    ## Plotting Parameters
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
        matplotlib.rc("axes", titlesize=12)
        matplotlib.rc("legend", fontsize=10)
        matplotlib.rc("figure", titlesize=12)

        fig_dir  = "./Figures/PresentationFigures/PaperV4/"

        ## Figure Size Parameters
        my_dpi     = 400  ## Pixels per inch.
        fig_width  = 7.5  ## inches (8 inch-wide paper minus margins)
        fig_height = 0.75 * fig_width
        fig_size   = (fig_width, fig_height)

        ## Figure parameters
        fig_pad          = 1.0   ## Fraction of a fontwidth, iirc
        fig_ppad         = 0.2   ## Percent of fig to leave around edge.

        main_rows        = 2
        main_cols        = 3
        main_hspace      = 0.0
        main_wspace      = 0.0
        main_wratios     = [1, 1, 0.1, 0.12]

        panel_rows       = 1
        panel_cols       = 1
        panel_hspace     = 0.2
        panel_hratios    = [1, 0.3]
        panel_pad        = 0.5
        panel_wpad       = 0.2
        panel_hpad       = 0.7

        ## Scatterplot parameters
        sctr_size        = 3
        sctr_alpha       = 0.8
        sctr_ylim_pad    = 0.12
        sctr_sp_alpha    = 1

        ## Text box parameters
        text_y           = 0.011
        text_fs          = 9
        text_fw          = 'bold'
        text_rect_pad    = 3
        disp_dpi         = 72

        ## KDE plot parameters
        kde_sp_alpha     = 1
        kde_sp_2_show    = ['left', 'bottom']

        ## Colorbar properties
        cbar_tickl       = 0
        cbar_tickw       = 0
        cbar_ticksize    = 8
        cbar_tickpad     = 2

        ## Color maps
        base_cmap   = sns.color_palette()
        cblind_cmap = sns.color_palette('colorblind')
        bright_cmap = sns.color_palette('bright')

        data_color = bright_cmap[3]
        null_color = bright_cmap[4]

        ## p-Value Colors
        min_pVal = np.min([pVals.min() for pVals in pValues.values()])
        min_pVal = -np.log10(min_pVal) + 0.005
        pVal_clr_change = [0, 1, 2, 3, min_pVal]
        pVal_clr_idx = [4, 0, 3, 2]  ##[3, 4, 0, 2]
        [pVal_cmap,
         pVal_cnorm] = pUtl.make_categ_cmap(change_points=pVal_clr_change,
                                            cmap_idx=pVal_clr_idx,
                                            max_diverge=0.01)

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
        main_gs = fig.add_gridspec(main_rows, main_cols + 1,
                                   hspace=main_hspace,
                                   wspace=main_wspace,
                                   width_ratios=main_wratios)

        ## Set up background axes
        main_bkgd_axes  = []
        panel_gridspecs = []
        panel_axes      = []
        kde_gridspecs   = []
        for ii in range(main_rows):
            panel_gs_row = []
            panel_ax_row = []
            kde_gs_row   = []
            for jj in range(main_cols - 1):
                ax = fig.add_subplot(main_gs[ii, jj])
                ax = pUtl.make_border_axes(ax, spine_alpha=spine_alpha)
                main_bkgd_axes.append(ax)

                pGS = gs.GridSpec(nrows=panel_rows, ncols=panel_cols)
                panel_gs_row.append(pGS)

                panel_ax_col = []
                for kk in range(panel_rows):
                    if kk == 0:
                        ax = fig.add_subplot(pGS[kk])
                        ax = pUtl.make_border_axes(ax,
                                                   spine_alpha=sctr_sp_alpha,
                                                   spine_color='k')
                    else:
                        # kde_gs = gs.GridSpec(nrows=1, ncols=1)
                        # ax = fig.add_subplot(kde_gs[0])
                        ax = fig.add_subplot(pGS[kk])
                        ax = pUtl.make_border_axes(ax,
                                                   spine_alpha=kde_sp_alpha,
                                                   spines_2_show=kde_sp_2_show,
                                                   spine_color='k',
                                                   xticks=None,
                                                   yticks=None,
                                                   xticklabels=None,
                                                   yticklabels=None)
                    panel_ax_col.append(ax)
                panel_ax_row.append(panel_ax_col)
                # kde_gs_row.append(kde_gs)

            panel_gridspecs.append(panel_gs_row)
            panel_axes.append(panel_ax_row)
            kde_gridspecs.append(kde_gs_row)

        colorbar_gs = main_gs[:, 2].subgridspec(nrows=2, ncols=1,
                                                hspace=0)

        cax_bkgd1 = fig.add_subplot(colorbar_gs[0])
        cax_bkgd1 = pUtl.make_border_axes(cax_bkgd1, spine_alpha=spine_alpha)

        cax_gs1 = gs.GridSpec(nrows=panel_rows, ncols=panel_cols,)

        cax_ax1 = fig.add_subplot(cax_gs1[0])
        cax_ax1 = pUtl.make_border_axes(cax_ax1, spine_alpha=1,
                                        spine_color='k')

        cax_bkgd2 = fig.add_subplot(colorbar_gs[1])
        cax_bkgd2 = pUtl.make_border_axes(cax_bkgd2, spine_alpha=spine_alpha)

        cax_gs2 = gs.GridSpec(nrows=panel_rows, ncols=panel_cols,)

        cax_ax2 = fig.add_subplot(cax_gs2[0])
        cax_ax2 = pUtl.make_border_axes(cax_ax2, spine_alpha=spine_alpha)

        def update_bounds():

            fig.tight_layout(pad=fig_pad)
            fig.subplots_adjust(left=0 + fig_ppad,
                                right=1 - fig_ppad,
                                bottom=0 + fig_ppad,
                                top=1 - fig_ppad)

            for ii in range(main_rows):
                for jj in range(main_cols - 1):
                    pUtl.update_tight_bounds(fig, panel_gridspecs[ii][jj],
                                             main_gs[ii, jj],
                                             fig_pad=fig_pad,
                                             inner_pad=panel_pad,
                                             w_pad=panel_wpad,
                                             h_pad=panel_hpad)

            pUtl.update_tight_bounds(fig, cax_gs1, colorbar_gs[0],
                                     fig_pad=fig_pad,
                                     inner_pad=panel_pad,
                                     w_pad=panel_wpad,
                                     h_pad=panel_hpad)

            pUtl.update_tight_bounds(fig, cax_gs2, colorbar_gs[1],
                                     fig_pad=fig_pad,
                                     inner_pad=panel_pad,
                                     w_pad=panel_wpad,
                                     h_pad=panel_hpad)

        update_bounds()

    ###########################################################################
    ## Generate the scatter plots!
    ###########################################################################
    if True:
        for algNo, (alg, param) in enumerate(DR_params):
            print(f"\nPlotting {alg} embedding (param = {param})")

            rowNo = int(algNo / (main_cols - 1))
            colNo = int(algNo % (main_cols - 1))

            ax = panel_axes[rowNo][colNo][0]

            dY = data_Y[(alg, param)][0][:]

            pVs = pValues[(alg, param)][:].squeeze()

            sort_idx = np.argsort(pVs)[::-1]

            clrs = -np.log10(pVs[sort_idx])
            hax = ax.scatter(*dY.T, c=clrs, s=sctr_size, alpha=sctr_alpha,
                             cmap=pVal_cmap, norm=pVal_cnorm)

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
                good_idx  = -np.log10(pVs) >= pVal_clr_change[ii]
                good_idx *= -np.log10(pVs) < pVal_clr_change[ii + 1]
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

            if alg.lower() == 'tsne':
                kEff = pUtl.get_kEff_from_perp(param, kEff_arr, perp_arr)
                kEff = int(pUtl.human_round(kEff)[0])
                title = r"t-SNE at $k_{Eff} \approx $" + f"{kEff}"
            else:
                kNN = int(pUtl.human_round(param)[0])
                title = r"UMAP at $k = $" + f"{kNN}"
            ax.set_title(title, pad=-15)

    ###########################################################################
    ## Generate the EES plots!
    ###########################################################################
    if False:

        for algNo, (alg, param) in enumerate(DR_params):
            print(f"\nPlotting {alg} embedding (param = {param})")

            rowNo = int(algNo / (main_cols - 1))
            colNo = int(algNo % (main_cols - 1))

            ax = panel_axes[rowNo][colNo][0]

            dEES = data_EES[(alg, param)][:]
            nEES = null_EES[(alg, param)][:]

            # sns.kdeplot(dEES.ravel(), color=data_color, fill=0.3, ax=ax)

            # sns.kdeplot(nEES.ravel(), color=null_color, fill=0.3, ax=ax)

            jVals, jCDF1, jCDF2 = get_QQ_vals(dEES.ravel(), nEES.ravel())

            auc_val = auc(jCDF2, jCDF1)

            text_x = 0.85
            if algNo == 1:
                text_x = 0.15
            ax.text(text_x, 0.85, f"AUC = {auc_val:.2f}",
                    va='center', ha='center', fontsize=10,
                    transform=ax.transAxes,)# bbox={'alpha': 0.2,
                                             #     'edgecolor': '0.8',
                                              #    'facecolor': '0.8'})

            # xmaxlim = np.max([np.percentile(dEES, 97),
            #                   np.percentile(nEES, 97)])
            # xminlim = np.min([np.percentile(dEES, 0.1),
            #                   np.percentile(nEES, 0.1)])

            # ax.set_xlim(0, 5.25)

            # ax.set_xlabel(r"Quality Score ($EES$)", fontsize=8)
            # ax.set_ylabel("")

            # ax.set_xticks([])
            # ax.set_yticks([])
            # ax.set_xticklabels([])
            # ax.set_yticklabels([])

    ###########################################################################
    ## Save and Show
    ###########################################################################
    if True:

        update_bounds()

        pUtl.add_panel_number(panel_axes[0][0][0], "A",
                              corner=('left', 'top'),
                              edge_pad=10, fontsize=10)

        pUtl.add_panel_number(panel_axes[0][1][0], "B",
                              corner=('left', 'top'),
                              edge_pad=10, fontsize=10)

        pUtl.add_panel_number(panel_axes[1][0][0], "C",
                              corner=('left', 'top'),
                              edge_pad=10, fontsize=10)

        pUtl.add_panel_number(panel_axes[1][1][0], "D",
                              corner=('left', 'top'),
                              edge_pad=10, fontsize=10)

        print("\n\nSaving Figure!\n\n")
        pUtl.save_figure(fig,
                         fig_base,
                         fig_dir=fig_dir,
                         tight_layout_pad=None,
                         dpi=my_dpi)

        print("Showing Figure!\n\n")
        plt.show()