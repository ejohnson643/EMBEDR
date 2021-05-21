"""
###############################################################################
    Figure: An Overview of the EMBEDR Algorithm (v1)
###############################################################################

    Author: Eric Johnson
    Date Created: Monday, March 15, 2021
    Email: ericjohnson1.2015@u.northwestern.edu

###############################################################################

    This is an eight attempt at the algorithm overview figure

    The idea here is to illustrate how the method moves from multiple
    instantiations of the algorithm to generate a statistical ensemble of D_KLs
    to using a reshuffled data set to generate a null distribution.  These get
    compared to give p-values.  Then we can examine the embedding using this
    coloring.

    +-----------------------------------------+
    |                  DATA                   |
    +-----------------------------------------+
    |   GENE HEATMAP ====> Affinity BAR CHART |\\
    +-----------------------------------------+->> EES ---> EES_DISTB
    |   EMBEDDING    ====> Affinity BAR CHART |//
    +-----------------------------------------+
    |                  NULL                   |
    +-----------------------------------------+
    |   GENE HEATMAP ====> Affinity BAR CHART |\\
    +-----------------------------------------+->> EES ---> EES_DISTB
    |   EMBEDDING    ====> Affinity BAR CHART |//
    +-----------------------------------------+
                                              |    EMBEDR p-VALUE SCATTER

    In this version, we're going to maintain the last structure, but we're
    going to use UMAP instead of t-SNE and we're going to make some other
    aesthetic adjustments.

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
import seaborn as sns
from sklearn.cluster import AgglomerativeClustering as AggClust
from sklearn.metrics import pairwise_distances as pwd
from version_5_0.embedr.affinity import FixedEntropyAffinity
import warnings

warnings.filterwarnings("ignore", message="This figure includes Axes that")
warnings.filterwarnings("ignore", message="tight_layout not applied: ")

EPSILON = np.finfo(np.float64).eps

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

    print_str = "  Plotting PaperV4 Figure: EMBEDR Overview (v1)  "
    print(f"\n\n" + print_str + "\n" + len(print_str) * "=" + "\n")

    ## Define figure and file name format
    name_base = "V4Fig_EMBEDR_Overview_v1"

    ## Select which data to use
    seq_type = "FACS"
    tissue = "Marrow"

    ## Determine how many embeddings and nulls to show
    n_embed_to_show      = 3
    n_null_embed_to_show = 3

    ## Set the DR alg and parameter
    DR_alg   = 'UMAP'
    DR_param = 100

    ## p-Value method
    pVal_method = "average"

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
            kEff_name = f"TabulaMuris_{tissue}_PCs_kEffective_v3.pkl"
            fig_base   = name_base + f"_TabulaMuris_{tissue}_{seq_type}"

        print(f"Trying to load {data_file} from {data_dir}...")

        X, metadata = pUtl.load_data(tissue, data_dir)

        n_samples, n_features = X.shape

        ## Set other parameters
        n_components   = 2
        n_embed = np.max([n_embed_to_show, 3])
        n_null_embed = np.max([n_null_embed_to_show, 3])
        n_jobs         = -1
        initialization = 'random'
        random_seed    = 1
        null_seed      = 100

        if DR_alg.lower() == "tsne":
            ## Perplexity
            perplexity = DR_param

            ## t-SNE runtime parameters
            early_exaggeration_iter = 250
            early_exaggeration = 12
            early_exaggeration_mom = 0.5
            n_iter = 750
            momentum = 0.8

        elif DR_alg.lower() == "umap":
            ## k Nearest Neighbors
            n_neighbors = DR_param

            ## UMAP runtime parameters
            min_dist = 0.5

        else:
            err_str = f"Unknown DR alg '{DR_alg}'!"
            raise ValueError(err_str)

        ## Get perp to eff_kNN mapping
        sorted_PWD = pwd(X, metric='sqeuclidean')
        sorted_PWD = np.sort(sorted_PWD, axis=1)[:, 1:]
        print(f"SORTED PWD IS SQUARED EUCLIDEAN")

        alpha_nu   = 0.01  ## Fraction of closest neighbor that's 'non-uniform'

        kEff_arr, perp_arr = pUtl.get_kEff(file_name=kEff_name,
                                           file_dir=data_dir,)

        if DR_alg.lower() == 'tsne':
            perp_eff = perplexity
            k_eff    = int(pUtl.get_kEff_from_perp(perp_eff,
                                                   kEff_arr,
                                                   perp_arr))
        else:
            k_eff    = n_neighbors
            perp_eff = int(pUtl.get_perp_from_kEff(k_eff,
                                                   kEff_arr,
                                                   perp_arr))

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

        ## If requested, do t-SNE!
        if DR_alg.lower() == 'tsne':
            data_Y = pUtl.load_tSNE(X,
                                    name_base=fig_base,
                                    embed_dir=embed_dir,
                                    n_embed=n_embed,
                                    n_components=n_components,
                                    perplexity=perplexity,
                                    early_exag_iter=early_exaggeration_iter,
                                    n_iter=n_iter,
                                    initialization=initialization,
                                    n_jobs=n_jobs,
                                    random_state=random_seed,
                                    verbose=True)

            null_Y = pUtl.load_tSNE(null_X,
                                    name_base=fig_base + "_Null",
                                    embed_dir=embed_dir,
                                    n_embed=n_null_embed,
                                    n_components=n_components,
                                    perplexity=perplexity,
                                    early_exag_iter=early_exaggeration_iter,
                                    n_iter=n_iter,
                                    initialization=initialization,
                                    n_jobs=n_jobs,
                                    random_state=random_seed + null_seed,
                                    verbose=True)

        ## If requested, do UMAP!
        elif DR_alg.lower() == 'umap':
            data_Y = pUtl.load_UMAP(X,
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

            null_Y = pUtl.load_UMAP(null_X,
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

        ## Get the median point in the first embedding
        med_Y = np.mean(data_Y[1], axis=0)
        med_idx = np.argmin(np.sum((data_Y[1] - med_Y)**2, axis=1))

    ###########################################################################
    ## Generate / Load EES and p-Values
    ###########################################################################
    if True:

        ## Initialize EES arrays
        data_EES = np.zeros((n_embed, n_samples))
        null_EES = np.zeros((n_null_embed, n_samples))

        ## Calculate EES for data.
        try:
            dEES_name = fig_base + f"_dataEES_Perp{perp_eff}"
            dEES_name += f"_RS{random_seed}.pkl"
            print(f"\nTrying to load data EES...")
            with open(path.join(embed_dir, dEES_name), 'rb') as f:
                data_dict = pkl.load(f)
            data_EES = data_dict['EES']
            data_P   = data_dict['P']
            print(f"... matrix loaded succesfully!")
        except (FileNotFoundError, IndexError, KeyError):
            print(f"... matrix could not be loaded! Recalculating...")

            affmat = FixedEntropyAffinity(perplexity=perp_eff,
                                          n_jobs=n_jobs,
                                          random_state=random_seed,
                                          verbose=5)
            affmat.fit(X)

            data_P = affmat.P.toarray().copy()
            del affmat
            data_P = data_P / data_P.sum(axis=1)[:, np.newaxis]

            ## Calculate the KL-Divergence
            for eNo in range(n_embed):
                data_Q = 1 / (1 + pwd(data_Y[eNo], metric='sqeuclidean'))
                data_Q = data_Q / np.sum(data_Q, axis=1)[:, np.newaxis]

                tmp_log = np.log((data_P + EPSILON) / (data_Q + EPSILON))
                data_EES[eNo] = np.sum(data_P * tmp_log, axis=1).squeeze()
            print(f"... data EES calculated!")

            data_dict = {'EES': data_EES.copy(),
                         'P':   data_P.copy()}

            with open(path.join(embed_dir, dEES_name), 'wb') as f:
                pkl.dump(data_dict, f)
            print(f"... Saved to file!")

        del data_dict

        ## Calculate EES for null.
        try:
            nEES_name = fig_base + f"_nullEES_Perp{perp_eff}"
            nEES_name += f"_RS{random_seed}.pkl"
            print(f"\nTrying to load null EES...")
            with open(path.join(embed_dir, nEES_name), 'rb') as f:
                null_dict = pkl.load(f)
            null_EES = null_dict['EES']
            null_P   = null_dict['P']
            print(f"... matrix loaded succesfully!")
        except (FileNotFoundError, IndexError, KeyError):
            print(f"... matrix could not be loaded! Recalculating...")

            affmat = FixedEntropyAffinity(perplexity=perp_eff,
                                          n_jobs=n_jobs,
                                          random_state=random_seed + null_seed,
                                          verbose=5)
            affmat.fit(null_X)

            null_P = affmat.P.toarray().copy()
            del affmat
            null_P = null_P / null_P.sum(axis=1)[:, np.newaxis]

            ## Calculate the KL-Divergence
            for nNo in range(n_null_embed):
                null_Q = 1 / (1 + pwd(null_Y[nNo], metric='sqeuclidean'))
                null_Q = null_Q / np.sum(null_Q, axis=1)[:, np.newaxis]

                tmp_log = np.log((null_P + EPSILON) / (null_Q + EPSILON))
                null_EES[nNo] = np.sum(null_P * tmp_log, axis=1).squeeze()
            print(f"... null EES calculated!")

            null_dict = {'EES': null_EES.copy(),
                         'P':   null_P.copy()}

            with open(path.join(embed_dir, nEES_name), 'wb') as f:
                pkl.dump(null_dict, f)
            print(f"... Saved to file!")

        del null_dict

        try:
            pVal_name = fig_base + f"_pValues_Perp{perp_eff}"
            pVal_name += f"_RS{random_seed}.pkl"
            print(f"\nTrying to load p-Values...")
            with open(path.join(embed_dir, pVal_name), 'rb') as f:
                pVal_dict = pkl.load(f)
            pVal_arr, pVals = pVal_dict['all_pVals'], pVal_dict['summ_pVals']
            print(f"... p-Values loaded succesfully!")
        except (FileNotFoundError, KeyError):
            print(f"... p-Values could not be loaded! Recalculating...")

            [pVal_arr, pVals] = pUtl.calc_emp_pVals(data_EES,
                                                    null_EES,
                                                    summary_method=pVal_method)
            print(f"... p-Values calculated!")

            pVal_dict = {'all_pVals': pVal_arr.copy(),
                         'summ_pVals': pVals.copy()}

            with open(path.join(embed_dir, pVal_name), 'wb') as f:
                pkl.dump(pVal_dict, f)

        del pVal_dict

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

        ## Main gridspec parameters
        main_wspace         = 0
        main_wratios        = [2.0, 1.5]

        ## Background parameters
        background_lw       = 3
        background_color    = 'k'
        background_sp_alpha = 1

        ## Left gridspec parameters
        left_hspace         = 0
        left_hratios        = [1.0, 0.03, 1.0]

        ## Right gridpspec parameters
        right_hspace        = 0.02
        right_hratios       = [1.5, 2.0]

        ## Embed (data or null) gridspec parameters
        embed_hspace         = 0
        embed_wspace         = 0
        embed_hratios        = [1.0, 1.2]
        embed_wratios        = [3.0, 0.8]
        embed_patch_alpha    = 0.7

        emb_overlap_ax_wsp   = -0.9
        emb_overlap_ax_hsp   = -0.9

        emb_sctr_spine_alpha = 1
        emb_sctr_show_frac   = 0.2
        emb_sctr_alpha       = 0.3
        emb_sctr_arrow_off   = 0.005  ## Figure fraction?
        emb_sctr_null_rot    = 45
        emb_sctr_star_size   = 250

        aff_axes_wspace      = 0.0
        aff_axes_wratios     = [1.0, 0.1, 1.0]
        aff_axes_wpad        = 0.0
        aff_axes_hpad        = 0.1
        aff_axes_pad         = 0.4
        aff_spine_alpha      = 1
        aff_spine_toshow     = ['bottom', 'left']

        ## Affinity/Distance Bar chart parameters
        aff_n_bars           = 20
        aff_bar_P_alpha      = 1.0
        aff_bar_Q_alpha      = 0.5
        aff_bar_Q_alpha_bnd  = 0.2
        aff_bar_align        = 'center'
        aff_bar_ec           = 'w'
        aff_bar_offset       = 0.5
        aff_bar_width        = 0.8

        ## EES Distribution Panel Parameters
        dist_spine_alpha     = 1
        dist_spine_2show     = ['bottom', 'left']
        dist_wpad            = 0
        dist_hpad            = 0
        dist_pad             = 1.0

        ## Plot colored by p-Value parameters
        pVal_spine_alpha     = 0
        pVal_spine_2show     = 'all'
        pVal_wpad            = 0
        pVal_hpad            = 0
        pVal_pad             = 1.0
        pVal_star_size       = 100
        pVal_star_alpha      = 0.8

        ## Gene heatmap parameters
        n_heatmap_rows       = 500
        heatmap_row_pad      = 0.5
        ax_h2w_frac          = 1.2

        ## Color maps
        base_cmap   = sns.color_palette()
        cblind_cmap = sns.color_palette('colorblind')
        bright_cmap = sns.color_palette('bright')

        ## Useful colors
        data_color  = bright_cmap[3]  ## Bright Red
        null_color  = bright_cmap[4]  ## Bright Purple

        ## p-Value Colors
        pVal_clr_change = [0, 1, 1.301, 2, 4.405]
        [pVal_cmap,
         pVal_cnorm] = pUtl.make_categ_cmap(change_points=[0, 4.405],
                                            categ_cmap=bright_cmap,
                                            cmap_idx=[4],
                                            reverse_last_interval=False)

    ###########################################################################
    ## Set up figure and gridspec
    ###########################################################################
    if True:

        ## Create the figure
        fig = plt.figure(figsize=fig_size)

        ## Set up main gridspec
        main_gs = fig.add_gridspec(1, 2, wspace=main_wspace,
                                   width_ratios=main_wratios)

        ## Set up background axes
        left_behind_ax = fig.add_subplot(main_gs[0])
        left_behind_ax = pUtl.make_border_axes(left_behind_ax,
                                               spine_alpha=spine_alpha)

        right_behind_ax = fig.add_subplot(main_gs[1])
        right_behind_ax = pUtl.make_border_axes(right_behind_ax,
                                                spine_alpha=spine_alpha)

        ## Set up data/null background subplots (LEFT HALF)
        left_gs = main_gs[0].subgridspec(3, 1,
                                         hspace=left_hspace,
                                         height_ratios=left_hratios)

        ## Set up distb / p-val background subplots (RIGHT HALF)
        right_gs = main_gs[1].subgridspec(2, 1,
                                          hspace=right_hspace,
                                          height_ratios=right_hratios)

        ## Set up background / hidden LEFT subplots
        data_behind_ax = fig.add_subplot(left_gs[0])
        data_behind_ax = pUtl.make_border_axes(data_behind_ax,
                                               spine_alpha=background_sp_alpha,
                                               spine_width=background_lw,
                                               spine_color=background_color)

        data_null_buffer_ax = fig.add_subplot(left_gs[1])
        data_null_buffer_ax = pUtl.make_border_axes(data_null_buffer_ax,
                                                    spine_alpha=spine_alpha)

        null_behind_ax = fig.add_subplot(left_gs[2])
        null_behind_ax = pUtl.make_border_axes(null_behind_ax,
                                               spine_alpha=background_sp_alpha,
                                               spine_width=background_lw,
                                               spine_color=background_color)

        if True:
            ## Set up data subplots (TOP-LEFT PART)
            data_gs = left_gs[0].subgridspec(2, 2,
                                             hspace=embed_hspace,
                                             wspace=embed_wspace,
                                             height_ratios=embed_hratios,
                                             width_ratios=embed_wratios)

            ## Hidden axes behind high-dim data.
            data_X_behind_ax = fig.add_subplot(data_gs[0, 0])
            data_X_behind_ax = pUtl.make_border_axes(data_X_behind_ax,
                                                     spine_alpha=spine_alpha)

            ## High-dim data gridspec
            data_X_gs = gs.GridSpec(nrows=1,
                                    ncols=3,
                                    wspace=aff_axes_wspace,
                                    width_ratios=aff_axes_wratios)

            ## Axis for X heatmap
            data_vis_ax = fig.add_subplot(data_X_gs[0])
            data_vis_ax = pUtl.make_border_axes(data_vis_ax,
                                                spine_alpha=emb_sctr_alpha,
                                                spine_color=data_color)

            ## Axis for heatmap -> distance distb arrow.
            data_X_arrow_ax = fig.add_subplot(data_X_gs[1])
            data_X_arrow_ax = pUtl.make_border_axes(data_X_arrow_ax,
                                                    spine_alpha=spine_alpha)

            ## Axis for P bar chart (distance distb)
            affinity_P_ax = fig.add_subplot(data_X_gs[2])
            sa = aff_spine_toshow
            affinity_P_ax = pUtl.make_border_axes(affinity_P_ax,
                                                  spine_alpha=aff_spine_alpha,
                                                  spines_2_show=sa)

            ## Axis for low-dim data (hidden)
            data_Y_behind_ax = fig.add_subplot(data_gs[1, 0])
            data_Y_behind_ax = pUtl.make_border_axes(data_Y_behind_ax,
                                                     spine_alpha=spine_alpha)

            ## Grid spec for low-dim data.
            bot_left_gs = data_gs[1, 0]
            data_Y_gs = bot_left_gs.subgridspec(nrows=n_embed_to_show,
                                                ncols=n_embed_to_show,
                                                wspace=emb_overlap_ax_wsp,
                                                hspace=emb_overlap_ax_hsp)

            ## For each embedding to show, make some axes.
            data_Y_behind_axes = []
            data_Y_gridspecs = []
            data_Y_axes = []
            for eNo in range(n_embed_to_show):
                tmp_ax = fig.add_subplot(data_Y_gs[eNo, eNo])
                sa = spine_alpha
                data_Y_behind_axes = pUtl.make_border_axes(tmp_ax,
                                                           spine_alpha=sa)

                data_Y_subgs = gs.GridSpec(nrows=1,
                                           ncols=3,
                                           wspace=aff_axes_wspace,
                                           width_ratios=aff_axes_wratios)
                data_Y_gridspecs.append(data_Y_subgs)

                sa = emb_sctr_spine_alpha
                emb_alpha = embed_patch_alpha
                embed_vis_ax = fig.add_subplot(data_Y_subgs[0])
                embed_vis_ax = pUtl.make_border_axes(embed_vis_ax,
                                                     patch_alpha=emb_alpha,
                                                     spine_alpha=sa,
                                                     spine_color=data_color)

                affinity_Q_ax = fig.add_subplot(data_Y_subgs[2])
                tmp = aff_spine_toshow
                sa = aff_spine_alpha
                emb_alpha = embed_patch_alpha
                affinity_Q_ax = pUtl.make_border_axes(affinity_Q_ax,
                                                      patch_alpha=emb_alpha,
                                                      spine_alpha=sa,
                                                      spines_2_show=tmp)

                data_Y_axes.append([embed_vis_ax, affinity_Q_ax])

            ## Make the axis to go to the right of high/low dim embeds where we
            ## "connect" the two ideas into an EES.
            data_EES_behind_ax = fig.add_subplot(data_gs[:, 1])
            data_EES_behind_ax = pUtl.make_border_axes(data_EES_behind_ax,
                                                       spine_alpha=spine_alpha)

        if True:
            ## Set up null subplots (TOP-RIGHT PART)
            null_gs = left_gs[2].subgridspec(2, 2,
                                             hspace=embed_hspace,
                                             wspace=embed_wspace,
                                             height_ratios=embed_hratios,
                                             width_ratios=embed_wratios)

            ## Hidden axes behind high-dim null.
            null_X_behind_ax = fig.add_subplot(null_gs[0, 0])
            null_X_behind_ax = pUtl.make_border_axes(null_X_behind_ax,
                                                     spine_alpha=spine_alpha)

            ## High-dim null gridspec
            null_X_gs = gs.GridSpec(nrows=1,
                                    ncols=3,
                                    wspace=aff_axes_wspace,
                                    width_ratios=aff_axes_wratios)

            ## Axis for null_X heatmap
            null_vis_ax = fig.add_subplot(null_X_gs[0])
            null_vis_ax = pUtl.make_border_axes(null_vis_ax,
                                                spine_alpha=emb_sctr_alpha,
                                                spine_color=null_color)

            ## Axis for heatmap -> distance distb arrow.
            null_X_arrow_ax = fig.add_subplot(null_X_gs[1])
            null_X_arrow_ax = pUtl.make_border_axes(null_X_arrow_ax,
                                                    spine_alpha=spine_alpha)

            ## Axis for P bar chart (distance distb)
            aff_null_P_ax = fig.add_subplot(null_X_gs[2])
            sa = aff_spine_toshow
            aff_null_P_ax = pUtl.make_border_axes(aff_null_P_ax,
                                                  spine_alpha=aff_spine_alpha,
                                                  spines_2_show=sa)

            ## Axis for low-dim null (hidden)
            null_Y_behind_ax = fig.add_subplot(null_gs[1, 0])
            null_Y_behind_ax = pUtl.make_border_axes(null_Y_behind_ax,
                                                     spine_alpha=spine_alpha)

            ## Grid spec for low-dim null.
            bot_left_gs = null_gs[1, 0]
            null_Y_gs = bot_left_gs.subgridspec(nrows=n_null_embed_to_show,
                                                ncols=n_null_embed_to_show,
                                                wspace=emb_overlap_ax_wsp,
                                                hspace=emb_overlap_ax_hsp)

            ## For each embedding to show, make some axes.
            null_Y_behind_axes = []
            null_Y_gridspecs = []
            null_Y_axes = []
            for nNo in range(n_null_embed_to_show):
                tmp_ax = fig.add_subplot(null_Y_gs[nNo, nNo])
                sa = spine_alpha
                null_Y_behind_axes = pUtl.make_border_axes(tmp_ax,
                                                           spine_alpha=sa)

                null_Y_subgs = gs.GridSpec(nrows=1,
                                           ncols=3,
                                           wspace=aff_axes_wspace,
                                           width_ratios=aff_axes_wratios)
                null_Y_gridspecs.append(null_Y_subgs)

                embed_vis_ax = fig.add_subplot(null_Y_subgs[0])
                sa = emb_sctr_spine_alpha
                emb_alpha = embed_patch_alpha
                embed_vis_ax = pUtl.make_border_axes(embed_vis_ax,
                                                     patch_alpha=emb_alpha,
                                                     spine_alpha=sa,
                                                     spine_color=null_color)

                affinity_Q_ax = fig.add_subplot(null_Y_subgs[2])
                tmp = aff_spine_toshow
                sa = aff_spine_alpha
                emb_alpha = embed_patch_alpha
                affinity_Q_ax = pUtl.make_border_axes(affinity_Q_ax,
                                                      patch_alpha=emb_alpha,
                                                      spine_alpha=sa,
                                                      spines_2_show=tmp)

                null_Y_axes.append([embed_vis_ax, affinity_Q_ax])

            ## Make the axis to go to the right of high/low dim embeds where we
            ## "connect" the two ideas into an EES.
            null_EES_behind_ax = fig.add_subplot(null_gs[:, 1])
            null_EES_behind_ax = pUtl.make_border_axes(null_EES_behind_ax,
                                                       spine_alpha=spine_alpha)

        if True:

            # Make right axes
            EES_behind_ax = pUtl.make_border_axes(fig.add_subplot(right_gs[0]),
                                                  spine_alpha=spine_alpha)

            EES_dist_gs = gs.GridSpec(nrows=1,
                                      ncols=1)

            EES_dist_ax = fig.add_subplot(EES_dist_gs[0])
            EES_dist_ax = pUtl.make_border_axes(EES_dist_ax,
                                                xticks=None,
                                                yticks=None,
                                                xticklabels=None,
                                                yticklabels=None,
                                                spine_alpha=dist_spine_alpha,
                                                spines_2_show=dist_spine_2show)
        if True:

            # Make right axes
            pVal_behind_ax = fig.add_subplot(right_gs[1])
            pVal_behind_ax = pUtl.make_border_axes(pVal_behind_ax,
                                                   spine_alpha=spine_alpha)

            pVal_gs = gs.GridSpec(nrows=1,
                                  ncols=1)

            pVal_ax = fig.add_subplot(pVal_gs[0])
            pVal_ax = pUtl.make_border_axes(pVal_ax,
                                            xticks=None,
                                            yticks=None,
                                            spine_alpha=pVal_spine_alpha,
                                            spines_2_show=pVal_spine_2show)

        def update_grid():
            ## FIX THE BOUNDS
            pUtl.update_tight_bounds(fig, data_X_gs, data_gs[0, 0],
                                     w_pad=aff_axes_wpad, fig_pad=aff_axes_pad)

            pUtl.update_tight_bounds(fig, null_X_gs, null_gs[0, 0],
                                     w_pad=aff_axes_wpad, fig_pad=aff_axes_pad)

            pUtl.update_tight_bounds(fig, EES_dist_gs, right_gs[0],
                                     w_pad=dist_wpad, h_pad=dist_hpad,
                                     fig_pad=dist_pad)

            pUtl.update_tight_bounds(fig, pVal_gs, right_gs[1],
                                     w_pad=pVal_wpad, h_pad=pVal_hpad,
                                     fig_pad=pVal_pad)

            for eNo in range(n_embed_to_show):
                pUtl.update_tight_bounds(fig,
                                         data_Y_gridspecs[eNo],
                                         data_Y_gs[eNo, eNo],
                                         w_pad=aff_axes_wpad,
                                         fig_pad=aff_axes_pad)

                pUtl.update_tight_bounds(fig,
                                         null_Y_gridspecs[eNo],
                                         null_Y_gs[eNo, eNo],
                                         w_pad=aff_axes_wpad,
                                         fig_pad=aff_axes_pad)

        update_grid()

    ###########################################################################
    ## Show Data X Heatmap
    ###########################################################################
    if True:
        ax = data_vis_ax

        ax_bds = ax.get_position(original=True).bounds
        ax_h2w_ratio = ax_bds[3] / ax_bds[2]

        n_heatmap_cols  = n_heatmap_rows * ax_h2w_ratio * ax_h2w_frac
        heatmap_col_pad = heatmap_row_pad * ax_h2w_ratio

        extent = [-heatmap_row_pad, n_heatmap_rows + heatmap_row_pad,
                  -heatmap_col_pad, n_heatmap_cols + heatmap_col_pad]

        ax.imshow(X[:n_heatmap_rows], extent=extent, cmap='viridis')

        ax.set_xlabel(r"$N_{genes}$")
        ax.set_ylabel(r"$N_{cells}$", labelpad=-0.2)

    ###########################################################################
    ## Show Data X Affinity Bars
    ###########################################################################
    if True:
        ax = affinity_P_ax

        P_row        = data_P[0]
        P_nonzero    = P_row.nonzero()[0]
        P_row        = P_row[P_nonzero]
        P_sorted_row = np.sort(P_row)[::-1]

        bar_handle = ax.bar(np.arange(aff_n_bars) + aff_bar_offset,
                            height=P_sorted_row[:aff_n_bars],
                            color=data_color,
                            width=aff_bar_width,
                            align=aff_bar_align,
                            edgeColor=aff_bar_ec,
                            alpha=aff_bar_P_alpha)

        ax.set_xlabel("Neighbors")
        ax.set_ylabel(r"Distance")

        bbox = {'boxstyle': 'round',
                'ec': data_color,
                'fc': data_color,
                'lw': 1.3,
                'alpha': 0.3}

        ax.text(0.5, 0.8, r"$\vec{x}_i\rightarrow\vec{x}_j$",
                ha='center', va='center', fontsize=12,
                bbox=bbox, transform=ax.transAxes)

    ###########################################################################
    ## Show Data Y Embeddings
    ###########################################################################
    if True:

        arrow_props = {"fc": 'k',
                       "ec": 'k',
                       "lw": 0,
                       "alpha": 1,
                       "width": 2,
                       "headwidth": 7
                       }

        x_min, x_max = 0.05, 0.9
        y_min, y_max = 0.05, 0.9

        for eNo in range(n_embed_to_show):
            ax = data_Y_axes[eNo][0]

            dY = data_Y[eNo]

            if eNo < (n_embed_to_show - 1):
                sctr_color = 'lightgrey'
            else:
                sctr_color = 'grey'

            ax.scatter(*dY.T, s=2, color=sctr_color)

            ax.scatter(*dY[med_idx],
                       s=emb_sctr_star_size,
                       marker='*',
                       color=data_color,
                       alpha=0.7,
                       edgecolor='k',
                       zorder=20)

    ###########################################################################
    ## Show Data Y Affinity Bars
    ###########################################################################
    if True:
        P_row = data_P[0]
        P_nonzero = P_row.nonzero()[0]
        P_row = P_row[P_nonzero]
        P_sorted_idx = np.argsort(P_row)[::-1]

        bbox = {'boxstyle': 'round',
                'ec': data_color,
                'fc': data_color,
                'lw': 1.3,
                'alpha': 0.3}

        alt_bbox = {'boxstyle': 'round',
                    'ec': data_color,
                    'fc': data_color,
                    'lw': 0.7,
                    'alpha': 0.2}

        for eNo in range(n_embed_to_show):
            ax = data_Y_axes[eNo][1]

            dY = data_Y[eNo]
            PWD = np.sum((dY[0] - dY)**2, axis=1)

            Q_row = 1 / (1 + PWD)
            Q_row = Q_row[Q_row.nonzero()]
            Q_sorted_row = np.sort(Q_row)[::-1][:aff_n_bars]
            Q_max, Q_min = Q_sorted_row[0], Q_sorted_row[-1]
            Q_sorted_row = (Q_sorted_row - 0.9 * Q_min) / (Q_max - Q_min)

            if eNo < (n_embed_to_show  - 1):
                bar_alpha = aff_bar_Q_alpha_bnd
            else:
                bar_alpha = aff_bar_Q_alpha

            bar_handle = ax.bar(np.arange(aff_n_bars) + aff_bar_offset,
                                height=Q_sorted_row,
                                color=data_color,
                                width=aff_bar_width,
                                align=aff_bar_align,
                                edgeColor=aff_bar_ec,
                                alpha=bar_alpha)

            ylim = ax.get_ylim()
            ax.set_ylim(ylim[0], 1.3 * ylim[1])

            if eNo == (n_embed_to_show - 1):
                ax.set_xlabel("Neighbors")
                ax.set_ylabel(r"Distance")
                ax.text(0.5, 0.85,
                        r"$\vec{y}_{i,n}\rightarrow\vec{y}_{j,n}$",
                        ha='center',
                        va='center',
                        fontsize=8,
                        bbox=bbox,
                        transform=ax.transAxes)

            else:
                ax.set_xlabel(" ")
                ax.set_ylabel(" ")

    ###########################################################################
    ## Show comparison EES
    ###########################################################################
    if True:
        ax = data_EES_behind_ax

        bbox = {'boxstyle': 'round',
                'ec': data_color,
                'fc': data_color,
                'lw': 2,
                'alpha': 0.5}

        EES_text = f"Comparing\nDistances\n" + r"$\Rightarrow EES_{i,n}$"
        ax.text(0.35, 0.5, EES_text, bbox=bbox, fontsize=12, ha='center',
                va='center')

        arrow_props = {"fc": data_color,
                       "ec": data_color,
                       "lw": 0,
                       "alpha": 1,
                       "connectionstyle": 'arc3,rad=0.2'}

        ax.annotate(s="",
                    xy=(0.4, 0.4),
                    xycoords=ax.transAxes,
                    xytext=(-0.1, 0.2),
                    textcoords=ax.transAxes,
                    arrowprops=arrow_props)

        arrow_props = {"fc": data_color,
                       "ec": data_color,
                       "lw": 0,
                       "alpha": 1,
                       "connectionstyle": 'arc3,rad=-0.2'}

        ax.annotate(s="",
                    xy=(0.4, 0.6),
                    xycoords=ax.transAxes,
                    xytext=(-0.1, 0.8),
                    textcoords=ax.transAxes,
                    arrowprops=arrow_props)

    ###########################################################################
    ## Show Null X Heatmap
    ###########################################################################
    if True:
        ax = null_vis_ax

        ax_bds = ax.get_position(original=True).bounds
        ax_h2w_ratio = ax_bds[3] / ax_bds[2]

        n_heatmap_cols  = n_heatmap_rows * ax_h2w_ratio * ax_h2w_frac
        heatmap_col_pad = heatmap_row_pad * ax_h2w_ratio

        extent = [-heatmap_row_pad, n_heatmap_rows + heatmap_row_pad,
                  -heatmap_col_pad, n_heatmap_cols + heatmap_col_pad]

        ax.imshow(null_X[:n_heatmap_rows], extent=extent, cmap='viridis')

        bbox = {'ec': 'k',
                'fc': 'w',
                'lw': 0,
                'alpha': 0.1}
        ax.text(0.5, 0.5, 'Null\nData', bbox=bbox, fontsize=14, ha='center',
                va='center', fontweight='bold', transform=ax.transAxes)

        ax.set_xlabel(r"$N_{genes}$")
        # ax.xaxis.set_label_position('top')
        ax.set_ylabel(r"$N_{cells}$", labelpad=-0.2)

    ###########################################################################
    ## Show Null X Affinity Bars
    ###########################################################################
    if True:
        ax = aff_null_P_ax

        P_row = null_P[0]
        P_nonzero = P_row.nonzero()[0]
        P_row = P_row[P_nonzero]
        P_sorted_row = np.sort(P_row)[::-1]

        bar_handle = ax.bar(np.arange(aff_n_bars) + aff_bar_offset,
                            height=P_sorted_row[:aff_n_bars],
                            color=null_color,
                            width=aff_bar_width,
                            align=aff_bar_align,
                            edgeColor=aff_bar_ec,
                            alpha=aff_bar_P_alpha)

        ax.set_xlabel("Neighbors")
        ax.set_ylabel(r"Distance")

        bbox = {'boxstyle': 'round',
                'ec': null_color,
                'fc': null_color,
                'lw': 1.3,
                'alpha': 0.3}

        ax.text(0.5, 0.8, r"$\vec{x}^*_i\rightarrow\vec{x}^*_j$",
                ha='center', va='center', fontsize=12,
                bbox=bbox, transform=ax.transAxes)

    ###########################################################################
    ## Show Null Y Embeddings
    ###########################################################################
    if True:

        arrow_props = {"fc": 'k',
                       "ec": 'k',
                       "lw": 0,
                       "alpha": 1,
                       "width": 2,
                       "headwidth": 7
                       }

        x_min, x_max = 0.05, 0.9
        y_min, y_max = 0.05, 0.9

        for nNo in range(n_null_embed_to_show):
            ax = null_Y_axes[nNo][0]

            nY = null_Y[nNo]
            rot_mat = np.array([[np.cos(emb_sctr_null_rot * nNo),
                                 np.sin(emb_sctr_null_rot * nNo)],
                                [-np.sin(emb_sctr_null_rot * nNo),
                                 +np.cos(emb_sctr_null_rot * nNo)]])
            nY = np.dot(nY, rot_mat)

            if nNo < (n_null_embed_to_show - 1):
                sctr_color = 'lightgrey'
            else:
                sctr_color = 'grey'

            ax.scatter(*nY.T, s=2, color=sctr_color)

            ax.scatter(*nY[med_idx + 40],
                       s=emb_sctr_star_size,
                       marker='*',
                       color=null_color,
                       alpha=0.7,
                       edgecolor='k',
                       zorder=20)

    ###########################################################################
    ## Show Null Y Affinity Bars
    ###########################################################################
    if True:

        P_row = null_P[0]
        P_nonzero = P_row.nonzero()[0]
        P_row = P_row[P_nonzero]
        P_sorted_idx = np.argsort(P_row)[::-1]

        bbox = {'boxstyle': 'round',
                'ec': null_color,
                'fc': null_color,
                'lw': 1.3,
                'alpha': 0.3}

        alt_bbox = {'boxstyle': 'round',
                    'ec': null_color,
                    'fc': null_color,
                    'lw': 0.7,
                    'alpha': 0.2}

        for nNo in range(n_null_embed_to_show):
            ax = null_Y_axes[nNo][1]

            nY = null_Y[nNo]
            PWD = np.sum((nY[0] - nY)**2, axis=1)

            Q_row = 1 / (1 + PWD)

            Q_row = Q_row[Q_row.nonzero()]
            Q_sorted_row = np.sort(Q_row)[::-1][:aff_n_bars]
            Q_max, Q_min = Q_sorted_row[0], Q_sorted_row[-1]
            Q_sorted_row = (Q_sorted_row - 0.9 * Q_min) / (Q_max - Q_min)

            if nNo < (n_null_embed_to_show  - 1):
                bar_alpha = aff_bar_Q_alpha_bnd
            else:
                bar_alpha = aff_bar_Q_alpha

            bar_handle = ax.bar(np.arange(aff_n_bars) + aff_bar_offset,
                                height=Q_sorted_row,
                                color=null_color,
                                width=aff_bar_width,
                                align=aff_bar_align,
                                edgeColor=aff_bar_ec,
                                alpha=bar_alpha)

            ylim = ax.get_ylim()
            ax.set_ylim(ylim[0], 1.3 * ylim[1])

            if nNo == (n_null_embed_to_show - 1):
                ax.set_xlabel("Neighbors")
                ax.set_ylabel(r"Distance")
                ax.text(0.5, 0.85,
                        r"$\vec{y}^*_{i,n}\rightarrow\vec{y}^*_{j,n}$",
                        ha='center',
                        va='center',
                        fontsize=8,
                        bbox=bbox,
                        transform=ax.transAxes)

            else:
                ax.set_xlabel(" ")
                ax.set_ylabel(" ")

    ###########################################################################
    ## Show Null Comparison EES
    ###########################################################################
    if True:
        ax = null_EES_behind_ax

        bbox = {'boxstyle': 'round',
                'ec': null_color,
                'fc': null_color,
                'lw': 2,
                'alpha': 0.5}

        EES_text = f"Comparing\nDistances\n" + r"$\Rightarrow EES^*_{i,n}$"
        ax.text(0.35, 0.5, EES_text, bbox=bbox, fontsize=12, ha='center',
                va='center')

        arrow_props = {"fc": null_color,
                       "ec": null_color,
                       "lw": 0,
                       "alpha": 1,
                       "width": 4,
                       "headwidth": 12,
                       "connectionstyle": 'arc3,rad=0.2'}

        ax.annotate(s="",
                    xy=(0.4, 0.4),
                    xycoords=ax.transAxes,
                    xytext=(-0.1, 0.2),
                    textcoords=ax.transAxes,
                    arrowprops=arrow_props)

        arrow_props = {"fc": null_color,
                       "ec": null_color,
                       "lw": 0,
                       "alpha": 1,
                       "width": 4,
                       "headwidth": 12,
                       "connectionstyle": 'arc3,rad=-0.2'}

        ax.annotate(s="",
                    xy=(0.4, 0.6),
                    xycoords=ax.transAxes,
                    xytext=(-0.1, 0.8),
                    textcoords=ax.transAxes,
                    arrowprops=arrow_props)

    ###########################################################################
    ## SHOW EES Distribution
    ###########################################################################
    if True:

        print("\nPlotting DKL Distributions!\n")

        null_bins = np.linspace(null_EES.min(),
                                np.percentile(null_EES, 99.5),
                                51)

        match_EES = np.mean(data_EES[:, med_idx])  #1.8  #data_EES[0][med_idx]
        match_EES = null_bins[np.argmin(np.abs(null_bins - match_EES))]

        left_null = null_EES[(null_EES < match_EES).nonzero()].ravel()
        right_null = null_EES[(null_EES >= match_EES).nonzero()].ravel()

        pVal = float(len(left_null)) / float(len(left_null) + len(right_null))

        d1 = sns.histplot(left_null,
                          ax=EES_dist_ax,
                          bins=null_bins,
                          color=null_color,
                          alpha=1.0)

        d2 = sns.histplot(right_null,
                          ax=EES_dist_ax,
                          bins=null_bins,
                          color=null_color,
                          label='Null',
                          alpha=0.5)

        maxCount = np.max([p.get_height() for p in d1.patches])

        EES_dist_ax.axvline(match_EES, linewidth=3, color=data_color)

        bbox = {
            'ec': data_color,
            'fc': data_color,
            'lw': 2,
            'alpha': 0.7,
            'boxstyle': 'round'
        }

        pVal_txt = f"EMBEDR p-Value:\n" + r"$p_{i,n}\,=\,$" + f"{pVal:.2f}"
        pVal_txt_coords = [match_EES + 0.3, 500]
        pVal_txt_h = EES_dist_ax.text(*pVal_txt_coords,
                                      pVal_txt,
                                      ha='left',
                                      va='bottom',
                                      fontsize=10,
                                      bbox=bbox,)

        EES_dist_ax.set_xlabel(r"$EES^*$")#, labelpad=-7)
        EES_dist_ax.set_ylabel("Frequency")#, labelpad=-7)
        EES_dist_ax.set_xticks([])
        EES_dist_ax.set_yticks([])

    ###########################################################################
    ## Scatter Plot with p-Values
    ###########################################################################
    if True:

        dY = data_Y[-1]

        sort_idx = np.argsort(np.log10(pVals))[::-1]
        sort_pVals = -np.log10(pVals)[sort_idx]

        ax = pVal_ax

        hax = ax.scatter(*dY[sort_idx].T,
                         s=20,
                         c=sort_pVals,
                         alpha=1.,
                         cmap=pVal_cmap,
                         norm=pVal_cnorm,
                         edgecolor='k',
                         linewidths=0.2)

        ax.scatter(*dY[med_idx],
                   s=pVal_star_size,
                   marker='*',
                   color=data_color,
                   alpha=pVal_star_alpha,
                   edgecolor='k',
                   zorder=20)

        ylim = ax.get_ylim()
        ax.set_ylim(ylim[0], 1.2 * ylim[1])

        ax.set_title(f"{DR_alg} Embedding", pad=-7)

        h_cbar = fig.colorbar(hax, orientation='horizontal', pad=0.1)

        ticklabels =  r"Well-Embedded    $\leftarrow$  QUALITY  "
        ticklabels += r"$\rightarrow$     Noise-like"

        h_cbar.set_ticks([2.2])
        h_cbar.ax.set_xticklabels([ticklabels], fontsize=8)
        h_cbar.ax.invert_xaxis()
        h_cbar.ax.tick_params(length=0)
        h_cbar.set_label(r"EMBEDR $p$-Value")

    ###########################################################################
    ## Set the layout and draw.
    ###########################################################################
    if True:

        update_grid()

        fig.canvas.draw()

        update_grid()

        fig.canvas.draw()

        fig.tight_layout()

    ###########################################################################
    ## Show Data X Annotation Arrow
    ###########################################################################
    if True:

        ax = data_X_arrow_ax

        arrow_props = {"fc": 'k',
                       "ec": 'k',
                       "lw": 0,
                       "alpha": 1,
                       "width": 4,
                       "headwidth": 12,
                       }

        ## We need to get the renderer so we know how many pixels everything
        ## is going to take.
        renderer = fig.canvas.get_renderer()

        ## Get the pixel to figure fraction transform
        transFunc = fig.transFigure.inverted().transform

        ## Get the bounding box of the axes to point to:
        point_box = transFunc(affinity_P_ax.get_tightbbox(renderer))

        ## Get the left coordinate of the axes
        point_x = point_box[0, 0]

        ## Get the bounding box of the axes to point from:
        text_box  = data_vis_ax.get_position(fig).bounds

        ## Get the right coordinate of the axes (plus a little fudge)
        text_x  = text_box[0] + text_box[2] + emb_sctr_arrow_off

        ## Annotate this arrow!
        ax.annotate(s="",
                    xy=[point_x, 0.5],      ## Arrow Point
                    xycoords=(fig.transFigure, affinity_P_ax.transAxes),
                    xytext=[text_x, 0.5],   ## Arrow Base
                    textcoords=(fig.transFigure, data_vis_ax.transAxes),
                    arrowprops=arrow_props)

    ###########################################################################
    ## Show Data Y Annotation Arrow
    ###########################################################################
    if True:

        ax = fig.add_subplot(data_Y_gridspecs[-1][1])
        ax = pUtl.make_border_axes(ax, spine_alpha=spine_alpha)

        arrow_props = {"fc": 'k',
                       "ec": 'k',
                       "lw": 0,
                       "alpha": 1,
                       "width": 4,
                       "headwidth": 12,
                       }

        ## We need to get the renderer so we know how many pixels everything
        ## is going to take.
        renderer = fig.canvas.get_renderer()

        ## Get the pixel to figure fraction transform
        transFunc = fig.transFigure.inverted().transform

        ## Get the bounding box of the axes to point to:
        point_box = transFunc(data_Y_axes[-1][1].get_tightbbox(renderer))

        ## Get the left coordinate of the axes
        point_x = point_box[0, 0]

        ## Get the bounding box of the axes to point from:
        text_box  = data_Y_axes[-1][0].get_position(fig).bounds

        ## Get the right coordinate of the axes (plus a little fudge)
        text_x  = text_box[0] + text_box[2] + emb_sctr_arrow_off

        ax.annotate(s="",
                    xy=[point_x, 0.5],
                    xycoords=(fig.transFigure,
                              data_Y_axes[-1][1].transAxes),
                    xytext=[text_x, 0.5],
                    textcoords=(fig.transFigure,
                                data_Y_axes[-1][0].transAxes),
                    arrowprops=arrow_props)

    ###########################################################################
    ## Show Data X -> Y Annotation Arrows
    ###########################################################################
    if True:
        ax1 = data_vis_ax
        ax2_list = data_Y_axes

        arrow_props = {"fc": 'k',
                       "ec": 'k',
                       "lw": 0,
                       "alpha": 1,
                       "width": 2,
                       "headwidth": 8,
                       "connectionstyle": 'arc3,rad=-0.2',
                       }

        for eNo in np.arange(n_embed_to_show)[::-1]:

            ax1.annotate(s="",
                         xy=(0.82, 1.03),
                         xycoords=ax2_list[eNo][0].transAxes,
                         xytext=(0.9, -0.03),
                         textcoords=ax1.transAxes,
                         arrowprops=arrow_props,
                         zorder=100)

    ###########################################################################
    ## Show Null X Annotation Arrow
    ###########################################################################
    if True:

        ax = null_X_arrow_ax

        arrow_props = {"fc": 'k',
                       "ec": 'k',
                       "lw": 0,
                       "alpha": 1,
                       "width": 4,
                       "headwidth": 12,
                       }

        ## We need to get the renderer so we know how many pixels everything
        ## is going to take.
        renderer = fig.canvas.get_renderer()

        ## Get the pixel to figure fraction transform
        transFunc = fig.transFigure.inverted().transform

        ## Get the bounding box of the axes to point to:
        point_box = transFunc(aff_null_P_ax.get_tightbbox(renderer))

        ## Get the left coordinate of the axes
        point_x = point_box[0, 0]

        ## Get the bounding box of the axes to point from:
        text_box  = null_vis_ax.get_position(fig).bounds

        ## Get the right coordinate of the axes (plus a little fudge)
        text_x  = text_box[0] + text_box[2] + emb_sctr_arrow_off

        ## Annotate this arrow!
        ax.annotate(s="",
                    xy=[point_x, 0.5],      ## Arrow Point
                    xycoords=(fig.transFigure, aff_null_P_ax.transAxes),
                    xytext=[text_x, 0.5],   ## Arrow Base
                    textcoords=(fig.transFigure, null_vis_ax.transAxes),
                    arrowprops=arrow_props)

    ###########################################################################
    ## Show Null Y Annotation Arrow
    ###########################################################################
    if True:

        ax = fig.add_subplot(null_Y_gridspecs[-1][1])
        ax = pUtl.make_border_axes(ax, spine_alpha=spine_alpha)

        arrow_props = {"fc": 'k',
                       "ec": 'k',
                       "lw": 0,
                       "alpha": 1,
                       "width": 4,
                       "headwidth": 12,
                       }

        ## We need to get the renderer so we know how many pixels everything
        ## is going to take.
        renderer = fig.canvas.get_renderer()

        ## Get the pixel to figure fraction transform
        transFunc = fig.transFigure.inverted().transform

        ## Get the bounding box of the axes to point to:
        point_box = transFunc(null_Y_axes[-1][1].get_tightbbox(renderer))

        ## Get the left coordinate of the axes
        point_x = point_box[0, 0]

        ## Get the bounding box of the axes to point from:
        text_box  = null_Y_axes[-1][0].get_position(fig).bounds

        ## Get the right coordinate of the axes (plus a little fudge)
        text_x  = text_box[0] + text_box[2] + emb_sctr_arrow_off

        ax.annotate(s="",
                    xy=[point_x, 0.5],
                    xycoords=(fig.transFigure,
                              null_Y_axes[-1][1].transAxes),
                    xytext=[text_x, 0.5],
                    textcoords=(fig.transFigure,
                                null_Y_axes[-1][0].transAxes),
                    arrowprops=arrow_props)

    ###########################################################################
    ## Show Null X -> Y Annotation Arrows
    ###########################################################################
    if True:
        ax1 = null_vis_ax
        ax2_list = null_Y_axes

        arrow_props = {"fc": 'k',
                       "ec": 'k',
                       "lw": 0,
                       "alpha": 1,
                       "width": 2,
                       "headwidth": 8,
                       "connectionstyle": 'arc3,rad=-0.2',
                       }

        for nNo in np.arange(n_embed_to_show)[::-1]:

            ax1.annotate(s="",
                         xy=(0.82, 1.03),
                         xycoords=ax2_list[nNo][0].transAxes,
                         xytext=(0.9, -0.03),
                         textcoords=ax1.transAxes,
                         arrowprops=arrow_props,
                         zorder=100)

    ###########################################################################
    ## Show EES -> Distribution Annotation Arrows
    ###########################################################################
    if True:

        ax_top = data_behind_ax
        ax_bot = null_behind_ax

        ax = EES_dist_ax

        arrow_props = {"fc": data_color,
                       "ec": data_color,
                       "lw": 0,
                       "alpha": 1,
                       "connectionstyle": 'arc3,rad=-0.2',
                       }

        ax.annotate(s="",
                    xy=(match_EES * 0.9, maxCount),
                    xycoords=ax.transData,
                    xytext=[0.95, 0.6],
                    textcoords=ax_top.transAxes,
                    arrowprops=arrow_props)

        arrow_props = {"fc": null_color,
                       "ec": null_color,
                       "lw": 0,
                       "alpha": 1,
                       "connectionstyle": 'arc3,rad=-0.1',
                       }

        ax.annotate(s="",
                    xy=(match_EES * 0.7, -20),
                    xycoords=ax.transData,
                    xytext=[0.95, 0.6],
                    textcoords=ax_bot.transAxes,
                    arrowprops=arrow_props)

    ###########################################################################
    ## Save and Show!
    ###########################################################################
    if True:

        pUtl.add_panel_number(data_behind_ax, "A",
                              corner=('right', 'top'),
                              edge_pad=10, fontsize=10)
        pUtl.add_panel_number(null_behind_ax, "B",
                              corner=('right', 'top'),
                              edge_pad=10, fontsize=10)
        pUtl.add_panel_number(EES_behind_ax,  "C",
                              corner=('right', 'top'),
                              edge_pad=10, fontsize=10)
        pUtl.add_panel_number(pVal_behind_ax, "D",
                              corner=('right', 'top'),
                              edge_pad=10, fontsize=10)

        pUtl.save_figure(fig,
                         fig_base,
                         fig_dir=fig_dir,
                         tight_layout_pad=fig_pad,
                         dpi=my_dpi)

        print("Showing Figure!\n\n")
        plt.show()
