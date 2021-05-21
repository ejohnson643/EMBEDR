"""
###############################################################################
    Figure: A Zoology of Distortive Effects (v1)
###############################################################################

    Author: Eric Johnson
    Date Created: Monday, March 8, 2021
    Email: ericjohnson1.2015@u.northwestern.edu

###############################################################################

    In this figure, we want to show how different cell types get distorted as
    we change algorithm and algorithmic parameters.  We also want to illustrate
    actual underlying variation in basic properties of data in a DR embedding.

    Specifically, in this Figure, I want to make a 4x4 panel figure, showing
    t-SNE and UMAP embeddings with select clusters annotated.  These panels
    will also contain insets showing the #NN within 3sigma (for t-SNE) or
    distance to kth NN (for UMAP).

###############################################################################
"""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import PaperV4_PlottingScripts.plotting_utility as pUtl
import seaborn as sns
from sklearn.metrics import pairwise_distances as pwd
import warnings

warnings.filterwarnings("ignore", message="This figure includes Axes that")
warnings.filterwarnings("ignore", message="tight_layout not applied: ")

###############################################################################
##  RUN THE FILE
###############################################################################
if __name__ == "__main__":

    print_str = "  Plotting PaperV4 Figure: Dim. Red. Zoology (v1)  "
    print(f"\n\n" + print_str + "\n" + len(print_str) * "=" + "\n")

    ## Define figure and file name format
    name_base = "V4Fig_DimRedZoology_v1"

    ## Select which data to use
    seq_type = "FACS"
    tissue = "Marrow"

    ## Set parameters at which to plot data
    DR_params = [('tSNE', 7),
                 ('UMAP', 15),
                 ('tSNE', 250),
                 ('UMAP', 400)]

    ## Select clusters to label
    clust_2_label = [0, 10,      ## Shows merging of two clusters
                     4, 8,    ## Shows connection of three clusters
                     6, 11,         ## Shows distortion of single cluster
                     5, 12, 15]  ## Shows mixing of three clusters
    # clust_2_label = [1, 4, 5, 7, 8, 9, 14, 17, 18, 20] ## <- Alpha cOnt
    clust_2_label = [1, 10, 0, 8, 6, 4, 9, 7, 2, 3]  ## By size cOnt
    # clust_2_label = [1]
    # clust_2_label = [1, 6, 11, 18, 19, 20, 21]
    clust_2_label = sorted(clust_2_label)

    ## Runtime flags
    show_all_axes = False
    color_by_cluster = True
    color_by_variability = False

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

        ## Set other parameters
        n_components   = 2

        tSNE_exag_iter = 250
        tSNE_n_iter    = 1000 - tSNE_exag_iter

        random_seed    = 1
        initialization = 'random'
        n_jobs         = -1

        ## Get perp to eff_kNN mapping
        sorted_PWD = pwd(X, metric='sqeuclidean')
        sorted_PWD = np.sort(sorted_PWD, axis=1)[:, 1:]
        print(f"SORTED PWD IS SQUARED EUCLIDEAN")

        alpha_nu   = 0.01  ## Fraction of closest neibthat 'non-uniform'

        kEff_arr, perp_arr = pUtl.get_kEff(file_name=kEff_name,
                                           file_dir=data_dir,
                                           verbose=True)

    ###########################################################################
    ## Set up Colormaps and Cell Labels
    ###########################################################################
    if True:

        ## Color maps
        base_cmap   = sns.color_palette()
        cblind_cmap = sns.color_palette('colorblind')
        bright_cmap = sns.color_palette('bright')

        if tissue == "MNIST":
            label_colors = [cblind_cmap[ll] if ll in clust_2_label
                            else 'lightgrey'for ll in metadata]
            labels = metadata

        else:

            parsed_meta = pUtl.parse_metadata(metadata)

            cOnt_labels = parsed_meta['cell_ont_labels']
            cOnt_map    = parsed_meta['cell_ont_map']
            cOnt_map_rev = {val: key for key, val in cOnt_map.items()}

            # labels = metadata['cluster.ids'].values
            # label_colors = [cblind_cmap[cIdMap[ll]] if (ll in clust_2_label)
            #                 else 'lightgrey' for ll in labels]

            labels = [cOnt_map[ll] for ll in metadata['cell_ontology_class']]
            labels = np.asarray(labels).squeeze()

            l2cl = {cl: (ii + 3) % 10 for ii, cl in enumerate(clust_2_label)}
            label_colors = [cblind_cmap[l2cl[ll]] if (ll in clust_2_label)
                            else 'lightgrey' for ll in labels]

            # with open("./Embeddings/temp_labels.pkl", 'rb') as f:
            #     labels = pkl.load(f)
            # labels[labels == 21] = 11
            # l2cl = {cl: ii for ii, cl in enumerate(clust_2_label)}
            # label_colors = [cblind_cmap[l2cl[ll]] if (ll in clust_2_label)
            #                 else 'lightgrey' for ll in labels]

            label_sizes = [3 if (ll in clust_2_label) else 1 for ll in labels]

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
        matplotlib.rc("axes", titlesize=16)
        matplotlib.rc("legend", fontsize=10)
        matplotlib.rc("figure", titlesize=12)

        fig_dir  = "./Figures/PresentationFigures/PaperV4/"

        my_dpi = 400
        fig_wid = 7.2  ## inches (8 inch-wide paper minus margins)
        fig_hgt = 0.8 * fig_wid

        n_cols = 2
        n_rows = int(np.ceil(len(DR_params) / n_cols))

        if color_by_cluster:
            main_wspace    = 0.005
            fig_pad        = 0.5
        else:
            main_wspace    = 0.15
            fig_pad        = 3
        main_hspace        = 0.01
        main_spns_2_show   = 'all'
        main_spn_alpha     = 0.5
        main_spn_width     = 1.0
        main_height_ratios = [1, 1]

        if show_all_axes:
            spine_alpha = 1
        else:
            spine_alpha = 0

        fig = plt.figure(figsize=(fig_wid, fig_hgt))

        back_axis = pUtl.make_border_axes(fig.add_subplot(111),
                                          spine_alpha=spine_alpha)

        main_gs = fig.add_gridspec(nrows=n_rows,
                                   ncols=n_cols,
                                   wspace=main_wspace,
                                   hspace=main_hspace)

        main_axes = []
        for rowNo in range(n_rows):
            axes_row = []
            for colNo in range(n_cols):
                ax = fig.add_subplot(main_gs[rowNo, colNo])
                ax = pUtl.make_border_axes(ax,
                                           spines_2_show=main_spns_2_show,
                                           spine_alpha=main_spn_alpha,
                                           spine_width=main_spn_width)
                axes_row.append(ax)
            main_axes.append(axes_row)

        # fig.tight_layout()

    ###########################################################################
    ## Generate the plots!
    ###########################################################################
    if True:

        for algNo, (alg, param) in enumerate(DR_params):
            print(f"\nPlotting {alg} embedding (param = {param})")

            if alg.lower() == 'tsne':
                Y = pUtl.load_tSNE(X,
                                   name_base=name_base,
                                   embed_dir=embed_dir,
                                   n_embed=1,
                                   n_components=n_components,
                                   perplexity=param,
                                   early_exag_iter=tSNE_exag_iter,
                                   n_iter=tSNE_n_iter,
                                   initialization=initialization,
                                   n_jobs=n_jobs,
                                   random_state=random_seed,
                                   verbose=True)

            if alg.lower() == 'umap':
                Y = pUtl.load_UMAP(X,
                                   name_base=name_base,
                                   embed_dir=embed_dir,
                                   n_embed=1,
                                   n_components=n_components,
                                   n_neighbors=param,
                                   min_dist=0.1,
                                   initialization=initialization,
                                   n_jobs=n_jobs,
                                   random_state=random_seed,
                                   verbose=True)

            rowNo = int(algNo / n_cols)
            colNo = int(algNo % n_cols)
            ax = main_axes[rowNo][colNo]

            if color_by_cluster:
                ax.scatter(*Y.T,
                           c=label_colors,
                           s=label_sizes,
                           alpha=0.2)

                text_off = 0
                text_h   = 0
                for cNo, lab in enumerate(clust_2_label):
                    good_idx = (labels == lab)

                    med_Y = np.median(Y[good_idx], axis=0)

                    ax.text(*med_Y, f"{cNo}",
                            fontsize=12,
                            fontweight='bold',
                            va='center', ha='center')

                    ax_width = back_axis.get_window_extent().width
                    ax_height = back_axis.get_window_extent().height / fig.dpi

                    pad = 3

                    rect_width = ax.get_window_extent().width / ax_width

                    if cNo < 5:
                        x_loc = rect_width / 2
                        rect_x = 0
                    else:
                        x_loc = 1 - (rect_width / 2)
                        rect_x = 1 - rect_width

                    cLab = cOnt_labels[lab].title()
                    if "Slamf1-Negative" in cLab:
                        cLab = " ".join(cLab.split(" Multipotent "))
                    cLab = f"{cNo}: " + cLab

                    bb = ax.text(x_loc, -0.013 - (cNo % 5) * text_h,
                                 cLab,
                                 # r"$N=$" + f"{sum(good_idx)}",
                                 ha='center', va='top',
                                 fontsize=10,
                                 transform=back_axis.transAxes)

                    text_h = (bb.get_size() + 2 * pad) / 72. / ax_height

                    # if (cNo % 2) == 1:
                    #     text_off -= text_h

                    # rect_y = (int(cNo / 2) + 1) * text_h
                    rect_y = ((cNo % 5) + 1) * text_h

                    rect = plt.Rectangle((rect_x, -rect_y),
                                         width=rect_width,
                                         height=text_h,
                                         transform=back_axis.transAxes,
                                         zorder=3,
                                         fill=True,
                                         facecolor=cblind_cmap[l2cl[lab]],
                                         clip_on=False,
                                         alpha=0.8,
                                         edgecolor='0.8')

                    back_axis.add_patch(rect)

            elif color_by_variability:

                if alg.lower() == 'tsne':

                    var_arr, _, _ = pUtl.calc_N_nonuniform(sorted_PWD,
                                                           perp_arr=[param])

                elif alg.lower() == 'umap':

                    var_arr = np.log10(np.sqrt(sorted_PWD[:, param]))

                var_arr = var_arr.astype(float).squeeze()

                hax = ax.scatter(*Y.T,
                                 c=var_arr,
                                 s=3,
                                 alpha=0.2,
                                 cmap='magma')

                cax = fig.colorbar(hax, ax=ax, cmap='magma', drawedges=False)

                if alg.lower() == 'tsne':
                    cax.set_label(r"Effective Nearest Neighbors, $k_{Eff}$")
                if alg.lower() == 'umap':
                    cax.set_label(r"Distance to $k^{th}$ Neighbor")

                cax.solids.set_edgecolor('face')
                fig.canvas.draw()

            if alg.lower() == 'tsne':
                kEff = pUtl.get_kEff_from_perp(param,
                                               kEff_arr,
                                               perp_arr)
                kEff = pUtl.human_round(kEff)[0]
                title = f"t-SNE: " + r"$k_{Eff} \approx $" + f"{kEff:.0f}"
            if alg.lower() == 'umap':
                title = f"UMAP: " + r"$k = $" + f"{param:.0f}"

            ax.set_title(title, fontsize=14, pad=-15)
            ylim = ax.get_ylim()
            ax.set_ylim(ylim[0], ylim[1] + 0.1 * (ylim[1] - ylim[0]))

    ###########################################################################
    ## Save and Show!
    ###########################################################################
    if True:

        _ = pUtl.add_panel_number(main_axes[0][0], "A", edge_pad=10)
        _ = pUtl.add_panel_number(main_axes[0][1], "B", edge_pad=10)
        _ = pUtl.add_panel_number(main_axes[1][0], "C", edge_pad=10)
        _ = pUtl.add_panel_number(main_axes[1][1], "D", edge_pad=10)

        if color_by_cluster:
            fig_base += "_ColoredByCellOnt"
        elif color_by_variability:
            fig_base += "_ColoredByVariableStat"
            fig_base = "V4Supp" + fig_base[2:]

        fig.tight_layout()

        ## SAVE FIGURE HERE
        pUtl.save_figure(fig,
                         fig_base,
                         fig_dir=fig_dir,
                         tight_layout_pad=fig_pad,
                         dpi=my_dpi)

        plt.show()
