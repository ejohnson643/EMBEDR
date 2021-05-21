"""
###############################################################################
    Figure: An Illustration of Marginal Resampling (v1)
###############################################################################

    Author: Eric Johnson
    Date Created: Tuesday, March 16, 2021
    Email: ericjohnson1.2015@u.northwestern.edu

###############################################################################

    The idea here is to illustrate how the marginal resampling affects the
    embedding.  So we're going to show 3/4 sub-features:
    1.  Gene expression heatmap of data and null
    2.  Then joint + marginal distributions of two interesting genes (PCs)
    3.  Then data vs null embeddings. (Maybe at two perpelexities!?)

    In this version, we're going to use UMAP instead of t-SNE and make some
    other aesthetic changes.

###############################################################################
"""

import matplotlib
import matplotlib.gridspec as gs
import matplotlib.pyplot as plt
import numpy as np
import numpy.random as r
from os import path
import PaperV4_PlottingScripts.plotting_utility as pUtl
import pickle as pkl
import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering as AggClust
from sklearn.metrics import pairwise_distances as pwd
import seaborn as sns
import warnings

warnings.filterwarnings("ignore", message="This figure includes Axes that")
warnings.filterwarnings("ignore", message="tight_layout not applied: ")


def make_linkage_matrix(model):
    """Given a sklearn clustering model, create the linkage matrix

    The model contains the different parts of the linkage matrix as different
    attributes.  In particular, the model doesn't count the number of original
    data samples that are under each node, which is useful for the dendrogram
    function.

    This function is adapted from the example here:
    https://scikit-learn.org/stable/auto_examples/cluster/
        plot_agglomerative_dendrogram.html
    """

    counts = np.zeros(model.children_.shape[0])
    N_samples = len(model.labels_)

    for ii, merge in enumerate(model.children_):
        counter = 0
        for child_idx in merge:
            if child_idx < N_samples:  ## If we're merging a leaf node
                counter += 1
            else:                      ## Otherwise add previous counts
                counter += counts[child_idx - N_samples]
        counts[ii] = counter

    out = np.column_stack([model.children_, model.distances_, counts])
    out = out.astype(float)

    return out


###############################################################################
##  RUN THE FILE
###############################################################################
if __name__ == "__main__":

    print_str = "  Plotting PaperV4 Figure: Marginal Resampling (v1)  "
    print(f"\n\n" + print_str + "\n" + len(print_str) * "=" + "\n")

    ## Define figure and file name format
    name_base = "V4Fig_Marginal_Resampling_v1"

    ## Select which data to use
    seq_type = "FACS"
    tissue = "Marrow"

    ## Set the DR alg and parameter
    DR_algs = [('UMAP', 15),
               ('tSNE', 30)]
               # ('UMAP', 150)]

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

        print(f"Trying to load {data_file} from {data_dir}...")

        X, metadata = pUtl.load_data(tissue, data_dir)

        n_samples, n_features = X.shape

        ## Set other parameters
        n_components   = 2
        n_embed        = 1
        n_null_embed   = 1
        n_jobs         = -1
        initialization = 'random'
        random_seed    = 1
        null_seed      = 100

        ## t-SNE runtime parameters
        early_exag_iter = 250
        n_iter = 750
        momentum = 0.8

        ## UMAP runtime parameters
        min_dist = 0.8

        alpha_nu   = 0.01  ## Fraction of closest neighbor that's 'non-uniform'
        kEff_arr, perp_arr = pUtl.get_kEff(file_name=kEff_name,
                                           file_dir=data_dir,
                                           alpha_nu=alpha_nu)

        ## Use hierarchical clustering to sort the data.
        print(f"\nHierarchically Clustering Data!")
        clust_model = AggClust(distance_threshold=0, n_clusters=None).fit(X)
        link_mat = make_linkage_matrix(clust_model)
        dendro = sch.dendrogram(link_mat, no_plot=True)
        sort_idx = dendro['leaves']
        X = X[sort_idx]

        ## Generate Null Data Resample
        r.seed(random_seed + null_seed)
        null_X = np.array([r.choice(col, size=n_samples) for col in X.T]).T

    ###########################################################################
    ## Generate / Load Embeddings
    ###########################################################################
    if True:

        data_Y, null_Y = {}, {}

        for alg, param in DR_algs:

            if alg.lower() == 'tsne':
                ## Perplexity
                perplexity = param

                ## Load the data embedding
                dY = pUtl.load_tSNE(X,
                                    name_base=fig_base,
                                    embed_dir=embed_dir,
                                    n_embed=n_embed,
                                    n_components=n_components,
                                    perplexity=perplexity,
                                    early_exag_iter=early_exag_iter,
                                    n_iter=n_iter,
                                    initialization=initialization,
                                    n_jobs=n_jobs,
                                    random_state=random_seed,
                                    verbose=True)

                ## Load the null embedding
                nY = pUtl.load_tSNE(null_X,
                                    name_base=fig_base + "_Null",
                                    embed_dir=embed_dir,
                                    n_embed=n_null_embed,
                                    n_components=n_components,
                                    perplexity=perplexity,
                                    early_exag_iter=early_exag_iter,
                                    n_iter=n_iter,
                                    initialization=initialization,
                                    n_jobs=n_jobs,
                                    random_state=random_seed + null_seed,
                                    verbose=True)

            ## If requested, do UMAP!
            elif alg.lower() == 'umap':
                ## k Nearest Neighbors
                n_neighbors = param

                dY = pUtl.load_UMAP(X,
                                    name_base=name_base,
                                    embed_dir=embed_dir,
                                    n_embed=n_embed,
                                    n_components=n_components,
                                    n_neighbors=n_neighbors,
                                    min_dist=min_dist,
                                    initialization=initialization,
                                    n_jobs=n_jobs,
                                    random_state=random_seed,
                                    verbose=True)

                nY = pUtl.load_UMAP(null_X,
                                    name_base=name_base + "_Null",
                                    embed_dir=embed_dir,
                                    n_embed=n_null_embed,
                                    n_components=n_components,
                                    n_neighbors=n_neighbors,
                                    min_dist=min_dist,
                                    initialization=initialization,
                                    n_jobs=n_jobs,
                                    random_state=random_seed + null_seed,
                                    verbose=True)

            data_Y[(alg, param)] = dY[:]
            null_Y[(alg, param)] = nY[:]

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
        matplotlib.rc("axes", titlesize=16)
        matplotlib.rc("legend", fontsize=10)
        matplotlib.rc("figure", titlesize=12)

        fig_dir  = "./Figures/PresentationFigures/PaperV4/"

        ## Figure-level parameters
        my_dpi              = 400
        fig_wid             = 7.5  ## inches (8 inch-wide paper minus margins)
        fig_hgt             = 1.0 * fig_wid
        fig_size            = (fig_wid, fig_hgt)
        fig_pad             = 0.5
        fig_ppad            = 0.01  ## Percent of fig to leave around edge.

        spine_alpha = 0
        if show_all_axes:
            spine_alpha = 1

        ## Set gridspec parameters
        main_hspace       = 0.1
        main_hratios      = [1.8, 1.0]

        top_wspace        = 0.05
        top_wratios       = [1.0, 2.5]

        bot_wspace        = 0.04
        bot_wratios       = [1.0, 1.0, 0.1, 1.0, 1.0]

        ## Heatmap gridspec parameters
        heat_n_rows       = 700
        heat_row_pad      = 0.5
        heat_h2w_frac     = 0.8
        heat_cmap         = 'viridis'
        heat_interp       = 'None'

        ## Joint distribution parameters
        jdist_w_pad       = 0.25
        jdist_wspace      = 0.01
        jdist_h_pad       = 0.25
        jdist_hspace      = 0.01
        jdist_wratios     = [3.0, 1.0]
        jdist_hratios     = [1.0, 3.0]
        jdist_sp_alpha    = 1
        marg1_sp_2_show   = ["right", "bottom"]
        marg2_sp_2_show   = ["left", "top"]

        jdist_sctr_s      = 10
        jdist_sctr_dAlpha = 0.3
        jdist_sctr_nAlpha = 0.3
        jdist_sctr_lw     = 0.2
        marg_n_bins       = 31
        marg_type         = 'density'
        marg_data_alpha   = 0.7
        marg_null_alpha   = 0.7
        kde_data_lw       = 1.0
        kde_null_lw       = 3.0
        kde_bw_adjust     = 0.5
        kde_data_alpha    = 0.9
        kde_null_alpha    = 0.9

        ## Embedding parameters
        embed_sp_alpha    = 1
        embed_sp_lw       = 2
        embed_lab_alpha   = 1
        embed_data_spac   = 0.1
        embed_sctr_s      = 3
        embed_sctr_lw     = 0.2
        embed_sctr_dAlpha = 0.3
        embed_sctr_nAlpha = 0.3
        embed_bbox_lw     = 1
        embed_bbox_style  = 'round'
        embed_bbox_alpha  = 0.8
        embed_bbox_y      = 0.96
        embed_bbox_x      = 0.05
        embed_bbox_va     = 'top'

        ## Color maps
        base_cmap   = sns.color_palette()
        cblind_cmap = sns.color_palette('colorblind')
        bright_cmap = sns.color_palette('bright')

        ## Useful colors
        data_color  = bright_cmap[3]  ## Bright Red
        null_color  = bright_cmap[4]  ## Bright Purple

    ###########################################################################
    ## Set up figure and gridspec
    ###########################################################################
    if True:

        ## Create the figure
        fig = plt.figure(figsize=fig_size)

        ## Set up top / bottom subplots
        main_gs = fig.add_gridspec(2, 1,
                                   hspace=main_hspace,
                                   height_ratios=main_hratios)

        top_behind_ax = fig.add_subplot(main_gs[0])
        top_behind_ax = pUtl.make_border_axes(top_behind_ax,
                                              spine_alpha=spine_alpha)

        bot_behind_ax = fig.add_subplot(main_gs[1])
        bot_behind_ax = pUtl.make_border_axes(bot_behind_ax,
                                              spine_alpha=spine_alpha)

        ## Set up two parts of top gridspec: heatmap and joint dist.
        top_gs = main_gs[0].subgridspec(nrows=1, ncols=2,
                                        wspace=top_wspace,
                                        width_ratios=top_wratios)

        heatmap_behind_ax = fig.add_subplot(top_gs[0])
        heatmap_behind_ax = pUtl.make_border_axes(heatmap_behind_ax,
                                                  spine_alpha=spine_alpha)

        jointdist_behind_ax = fig.add_subplot(top_gs[1])
        jointdist_behind_ax = pUtl.make_border_axes(jointdist_behind_ax,
                                                    spine_alpha=spine_alpha)

        ## Set up the heatmap gridspec: data and null
        heatmap_gs = gs.GridSpec(nrows=2, ncols=1)

        data_hmap_ax = pUtl.make_border_axes(fig.add_subplot(heatmap_gs[0]),
                                             spine_alpha=spine_alpha)
        null_hmap_ax = pUtl.make_border_axes(fig.add_subplot(heatmap_gs[1]),
                                             spine_alpha=spine_alpha)

        ## Set up the joint dist gridspec: joint and 2 marginals.
        jointdist_gs = gs.GridSpec(nrows=2,
                                   ncols=2,
                                   wspace=jdist_wspace,
                                   hspace=jdist_hspace,
                                   width_ratios=jdist_wratios,
                                   height_ratios=jdist_hratios)

        jdist_ax = fig.add_subplot(jointdist_gs[1, 0])
        jdist_ax = pUtl.make_border_axes(jdist_ax,
                                         spine_alpha=jdist_sp_alpha)
        marg1_ax = fig.add_subplot(jointdist_gs[0, 0])
        marg1_ax = pUtl.make_border_axes(marg1_ax,
                                         spine_alpha=jdist_sp_alpha,
                                         spines_2_show=marg1_sp_2_show)
        marg2_ax = fig.add_subplot(jointdist_gs[1, 1])
        marg2_ax = pUtl.make_border_axes(marg2_ax,
                                         spine_alpha=jdist_sp_alpha,
                                         spines_2_show=marg2_sp_2_show)

        ## Set up the data and null embedding axes.
        bot_gs = main_gs[1].subgridspec(nrows=1,
                                        ncols=5,
                                        wspace=bot_wspace,
                                        width_ratios=bot_wratios)

        bot_left_ax  = fig.add_subplot(bot_gs[0:2])
        bot_left_ax  = pUtl.make_border_axes(bot_left_ax,
                                             spine_alpha=spine_alpha)
        bot_right_ax = fig.add_subplot(bot_gs[-2:])
        bot_right_ax = pUtl.make_border_axes(bot_right_ax,
                                             spine_alpha=spine_alpha)

        bot_ldata_ax = fig.add_subplot(bot_gs[0])
        bot_ldata_ax = pUtl.make_border_axes(bot_ldata_ax,
                                             spine_alpha=embed_sp_alpha,
                                             spine_width=embed_sp_lw,
                                             spine_color=data_color)

        bot_lnull_ax = fig.add_subplot(bot_gs[1])
        bot_lnull_ax = pUtl.make_border_axes(bot_lnull_ax,
                                             spine_alpha=embed_sp_alpha,
                                             spine_width=embed_sp_lw,
                                             spine_color=null_color)

        bot_rdata_ax = fig.add_subplot(bot_gs[-2])
        bot_rdata_ax = pUtl.make_border_axes(bot_rdata_ax,
                                             spine_alpha=embed_sp_alpha,
                                             spine_width=embed_sp_lw,
                                             spine_color=data_color)

        bot_rnull_ax = fig.add_subplot(bot_gs[-1])
        bot_rnull_ax = pUtl.make_border_axes(bot_rnull_ax,
                                             spine_alpha=embed_sp_alpha,
                                             spine_width=embed_sp_lw,
                                             spine_color=null_color)

        def update_grid():
            # fig.tight_layout()

            ## Tight-layout for heatmap subspec
            # hmap_crds = top_gs[0].get_position(fig).bounds
            # heatmap_gs.tight_layout(fig,
            #                         rect=[hmap_crds[0], hmap_crds[1],
            #                               hmap_crds[0] + hmap_crds[2],
            #                               hmap_crds[1] + hmap_crds[3]])

            pUtl.update_tight_bounds(fig, heatmap_gs, top_gs[0],
                                     fig_pad=fig_pad)

            ## Tight-layout for joint dist subspec
            # jdist_crds = top_gs[1].get_position(fig).bounds
            # jointdist_gs.tight_layout(fig,
            #                           rect=[jdist_crds[0], jdist_crds[1],
            #                                 jdist_crds[0] + jdist_crds[2],
            #                                 jdist_crds[1] + jdist_crds[3]])

            pUtl.update_tight_bounds(fig, jointdist_gs, top_gs[1],
                                     w_pad=jdist_w_pad, h_pad=jdist_h_pad,
                                     fig_pad=fig_pad)

        update_grid()

        update_grid()

    ###########################################################################
    ## Plot Heatmaps!
    ###########################################################################
    if True:
        ax = data_hmap_ax

        ax_bds = ax.get_position(original=True).bounds
        ax_h2w_ratio = ax_bds[3] / ax_bds[2]

        heat_n_cols = heat_n_rows * ax_h2w_ratio * heat_h2w_frac
        heat_col_pad = heat_row_pad * ax_h2w_ratio

        extent = [-heat_row_pad, heat_n_rows + heat_row_pad,
                  -heat_col_pad, heat_n_cols + heat_col_pad]

        tmp_X = (X[:heat_n_rows] - X.mean(axis=0)) / X.std(axis=0)

        ax.imshow(tmp_X, extent=extent, cmap=heat_cmap,
                  interpolation=heat_interp)

        ax.set_xlabel(r"$N_{genes}$ (or $N_{PCs}$)")
        ax.set_ylabel(r"$N_{cells}$", labelpad=0)
        ax.set_title("Data")

        ax = null_hmap_ax
        tmp_null_X = null_X[:heat_n_rows] - null_X.mean(axis=0)
        tmp_null_X = tmp_null_X / null_X.std(axis=0)
        ax.imshow(tmp_null_X, extent=extent, cmap=heat_cmap,
                  interpolation=heat_interp)

        ax.set_xlabel(r"$N_{genes}$ (or $N_{PCs}$)")
        ax.set_ylabel(r"$N_{cells}$", labelpad=0)
        ax.set_title("Null")

    ###########################################################################
    ## Plot Joint Distribution and Marginals!
    ###########################################################################
    if True:

        marg1_bins = np.linspace(np.min(np.hstack((X[:, 0], null_X[:, 0]))),
                                 np.max(np.hstack((X[:, 0], null_X[:, 0]))),
                                 marg_n_bins)
        marg2_bins = np.linspace(np.min(np.hstack((X[:, 1], null_X[:, 1]))),
                                 np.max(np.hstack((X[:, 1], null_X[:, 1]))),
                                 marg_n_bins)

        jdist_ax.scatter(*X[:, :2].T,
                         c=[data_color],
                         s=jdist_sctr_s,
                         alpha=jdist_sctr_dAlpha,
                         edgecolor=data_color,
                         label='Data',
                         lw=jdist_sctr_lw,
                         zorder=3,)

        jdist_ax.scatter(*null_X[:, :2].T,
                         c=[null_color],
                         s=jdist_sctr_s,
                         alpha=jdist_sctr_nAlpha,
                         edgecolor=null_color,
                         label='Null',
                         lw=jdist_sctr_lw,
                         zorder=2,)

        xlim = jdist_ax.get_xlim()
        ylim = jdist_ax.get_ylim()

        marg1_ax.set_ylabel("Density", labelpad=-7, rotation=-90)
        marg1_ax.yaxis.set_label_position('right')
        marg2_ax.set_xlabel("Density", labelpad=-13)
        marg2_ax.xaxis.set_label_position('top')

        sns.histplot(null_X[:, 0],
                     ax=marg1_ax,
                     bins=marg1_bins,
                     stat=marg_type,
                     color=null_color,
                     alpha=marg_null_alpha,
                     label='Null',
                     zorder=1)

        sns.histplot(X[:, 0],
                     ax=marg1_ax,
                     bins=marg1_bins,
                     stat=marg_type,
                     color=data_color,
                     alpha=marg_data_alpha,
                     label='Data',
                     zorder=2)

        sns.kdeplot(null_X[:, 0],
                    ax=marg1_ax,
                    color=null_color,
                    lw=kde_null_lw,
                    alpha=kde_null_alpha,
                    bw_adjust=kde_bw_adjust)

        sns.kdeplot(X[:, 0],
                    ax=marg1_ax,
                    color=data_color,
                    lw=kde_data_lw,
                    alpha=kde_data_alpha,
                    bw_adjust=kde_bw_adjust)

        marg1_ax.legend(loc=2)

        sns.histplot(y=null_X[:, 1],
                     ax=marg2_ax,
                     bins=marg2_bins,
                     stat=marg_type,
                     color=null_color,
                     alpha=marg_null_alpha)

        sns.histplot(y=X[:, 1],
                     ax=marg2_ax,
                     bins=marg2_bins,
                     stat=marg_type,
                     color=data_color,
                     alpha=marg_data_alpha)

        sns.kdeplot(y=null_X[:, 1],
                    ax=marg2_ax,
                    color=null_color,
                    lw=kde_null_lw,
                    alpha=kde_null_alpha,
                    bw_adjust=kde_bw_adjust)

        sns.kdeplot(y=X[:, 1],
                    ax=marg2_ax,
                    color=data_color,
                    lw=kde_data_lw,
                    alpha=kde_data_alpha,
                    bw_adjust=kde_bw_adjust)

        marg1_ax.set_xlim(xlim)
        marg2_ax.set_ylim(ylim)

        jdist_ax.set_xlabel("Feature 1")
        jdist_ax.set_ylabel("Feature 2", rotation=-90, labelpad=13)

        jdist_ax.xaxis.set_label_position('top')
        jdist_ax.yaxis.set_label_position('right')

        jdist_ax.legend()

    ###########################################################################
    ## Plot Data and Null Embeddings
    ###########################################################################
    if True:

        data_bbox = {'boxstyle': embed_bbox_style,
                     'ec': data_color,
                     'lw': embed_bbox_lw,
                     'fc': data_color,
                     'alpha': embed_bbox_alpha}

        null_bbox = {'boxstyle': embed_bbox_style,
                     'ec': null_color,
                     'lw': embed_bbox_lw,
                     'fc': null_color,
                     'alpha': embed_bbox_alpha}

        dY1 = data_Y[DR_algs[0]]
        bot_ldata_ax.scatter(-dY1[:, 0], -dY1[:, 1],
                             color=data_color,
                             s=embed_sctr_s,
                             alpha=embed_sctr_dAlpha,
                             lw=embed_sctr_lw)

        bot_ldata_ax.text(1 - embed_bbox_x, embed_bbox_y, "Data",
                          transform=bot_ldata_ax.transAxes,
                          ha='right', va=embed_bbox_va,
                          bbox=data_bbox)

        nY1 = null_Y[DR_algs[0]]
        bot_lnull_ax.scatter(*nY1.T,
                             color=null_color,
                             s=embed_sctr_s,
                             alpha=embed_sctr_nAlpha,
                             lw=embed_sctr_lw)

        bot_lnull_ax.text(embed_bbox_x, embed_bbox_y, "Null",
                          transform=bot_lnull_ax.transAxes,
                          ha='left', va=embed_bbox_va,
                          bbox=null_bbox)

        alg, param = DR_algs[0]
        if alg.lower() == 'tsne':
            kEff = int(pUtl.get_kEff_from_perp(param, kEff_arr, perp_arr))
            title = f"{alg}: " + r"$k_{Eff} =$" + f"{kEff}"
        elif alg.lower() == 'umap':
            title = f"{alg}: " + r"$k =$" + f"{param}"
        bot_left_ax.set_title(title, pad=4)

        dY2 = data_Y[DR_algs[1]]
        bot_rdata_ax.scatter(-dY2[:, 0], dY2[:, 1],
                             color=data_color,
                             s=embed_sctr_s,
                             alpha=embed_sctr_dAlpha,
                             lw=embed_sctr_lw)

        bot_rdata_ax.text(1 - embed_bbox_x, embed_bbox_y, "Data",
                          transform=bot_rdata_ax.transAxes,
                          ha='right', va=embed_bbox_va,
                          bbox=data_bbox)

        nY2 = null_Y[DR_algs[1]]
        bot_rnull_ax.scatter(*nY2.T,
                             color=null_color,
                             s=embed_sctr_s,
                             alpha=embed_sctr_nAlpha,
                             lw=embed_sctr_lw)

        bot_rnull_ax.text(embed_bbox_x, embed_bbox_y, "Null",
                          transform=bot_rnull_ax.transAxes,
                          ha='left', va=embed_bbox_va,
                          bbox=null_bbox)

        alg, param = DR_algs[1]
        if alg.lower() == 'tsne':
            kEff = pUtl.get_kEff_from_perp(param, kEff_arr, perp_arr)
            kEff = int(pUtl.human_round(kEff))
            title = f"{alg}: " + r"$k_{Eff} \approx$" + f"{kEff}"
        elif alg.lower() == 'umap':
            title = f"{alg}: " + r"$k =$" + f"{param}"
        bot_right_ax.set_title(title, pad=4)

    ###########################################################################
    ## Plot Annotation Arrows
    ###########################################################################
    if True:

        update_grid()

        update_grid()

        arrow_props = {"fc": data_color,
                       "ec": data_color,
                       "lw": 1,
                       "alpha": 0.7,
                       "width": 5}
        data_arrow = arrow_props.copy()
        data_arrow.update({"connectionstyle": 'arc3,rad=0.2'})

        jdist_ax.annotate("",
                          xy=[0.05, 0.53],
                          xycoords=jdist_ax.transAxes,
                          xytext=[1.01, 0.0],
                          textcoords=data_hmap_ax.transAxes,
                          arrowprops=data_arrow)

        null_arrow = arrow_props.copy()
        null_arrow.update({'fc': null_color,
                           'ec': null_color,
                           "connectionstyle": 'arc3,rad=-0.2'})

        jdist_ax.annotate("",
                          xy=[0.2, 0.2],
                          xycoords=jdist_ax.transAxes,
                          xytext=[1.01, 0.5],
                          textcoords=null_hmap_ax.transAxes,
                          arrowprops=null_arrow)

    ###########################################################################
    ## Save and Show!
    ###########################################################################
    if True:

        pUtl.add_panel_number(heatmap_behind_ax, "A",
                              corner=('left', 'top'),
                              number_loc=(None, None),
                              edge_pad=10, fontsize=10)
        pUtl.add_panel_number(jointdist_behind_ax, "B",
                              corner=('right', 'top'),
                              number_loc=(None, None),
                              edge_pad=10, fontsize=10)
        pUtl.add_panel_number(bot_left_ax,  "C",
                              corner=('left', 'top'),
                              number_loc=(None, None),
                              edge_pad=10, fontsize=10)
        pUtl.add_panel_number(bot_right_ax, "D",
                              corner=('right', 'top'),
                              number_loc=(None, None),
                              edge_pad=10, fontsize=10)

        pUtl.save_figure(fig,
                         fig_base,
                         fig_dir=fig_dir,
                         tight_layout_pad=fig_pad,
                         dpi=my_dpi)

        print("Showing Figure!\n\n")
        plt.show()
