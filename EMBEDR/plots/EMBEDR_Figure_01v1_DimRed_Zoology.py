"""
###############################################################################
    Figure: A Zoology of Distortive Effects (v1)
###############################################################################

    Author: Eric Johnson
    Date Created: Monday, March 8, 2021
    Date Edited: Thursday, October 28, 2021
    Email: ericjohnson1.2015@u.northwestern.edu

###############################################################################

    In this figure, we want to show how different cell types get distorted as
    we change algorithm and algorithmic parameters.  We also want to illustrate
    actual underlying variation in basic properties of data in a DR embedding.

    Specifically, in this Figure, I want to make a 4x4 panel figure, showing
    t-SNE and UMAP embeddings with select clusters annotated.  These panels
    will also contain insets showing the number of effective nearest neighbors
    (for t-SNE) or the distance to the kth neighbor (for UMAP).

    In this edit I want to formulate this more as a set of functions that can
    be run to quickly generate this figure for any data set.

###############################################################################
"""
from EMBEDR.embedr import EMBEDR, EMBEDR_sweep
import EMBEDR.plotting_utility as putl

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import scanpy as sc
import seaborn as sns
import warnings

warnings.filterwarnings("ignore", message="This figure includes Axes that")
warnings.filterwarnings("ignore", message="tight_layout not applied: ")


def make_figure(X, cluster_labels, clusters_2_label=None, label_colors=None,
                label_sizes=None, DRAs=None, grid_params={}, project_name=None,
                project_dir=None, EMBEDR_params={}, n_rows=None, n_cols=2):

    if clusters_2_label is None:
        clusters_2_label = [1, 10, 0, 8, 6, 4, 9, 7, 2, 3]  ## By size cOnt
        clusters_2_label = sorted(clusters_2_label)

    if label_colors is None:
        cblind_cmap = sns.color_palette('colorblind')
        l2cl = {cl: (ii + 3) % 10
                for ii, cl in enumerate(clusters_2_label)}
        label_colors = [cblind_cmap[l2cl[ll]] if (ll in clusters_2_label)
                        else 'lightgrey' for ll in cluster_labels.squeeze()]
        label_colors = np.asarray(label_colors)

    if label_sizes is None:
        label_sizes = [3 if (ll in clusters_2_label) else 1
                       for ll in cluster_labels]
        label_sizes = np.asarray(label_sizes)

    if DRAs is None:
        ## Set parameters at which to plot data
        DRAs = [('tSNE', 9),
                ('UMAP', 15),
                ('tSNE', 350),
                ('UMAP', 400)]

    if project_name is None:
        project_name = "EMBEDR_Figure_01v1_DimRedZoology"
    if project_dir is None:
        project_dir = "./"

    if n_rows is None:
        n_rows = int(np.ceil(len(DRAs) / n_cols))

    fig, back_axis, main_gs, main_axes = set_main_grid(n_rows=n_rows,
                                                       n_cols=n_cols,
                                                       **grid_params)

    for algNo, (alg, param) in enumerate(DRAs):
        print(f"\nPlotting {alg} embedding (param = {param})")

        if alg.lower() in ['tsne', 't-sne']:
            embObj = EMBEDR(X=X,
                            perplexity=param,
                            DRA='tsne',
                            n_data_embed=1,
                            n_jobs=-1,
                            project_name=project_name,
                            project_dir=project_dir,
                            **EMBEDR_params)
            Y, _ = embObj.get_tSNE_embedding(X)

        if alg.lower() in ['umap']:
            embObj = EMBEDR(X=X,
                            n_neighbors=param,
                            DRA='umap',
                            n_data_embed=1,
                            n_jobs=-1,
                            project_name=project_name,
                            project_dir=project_dir,
                            **EMBEDR_params)
            Y, _ = embObj.get_UMAP_embedding(X)

        rowNo = int(algNo / n_cols)
        colNo = int(algNo % n_cols)
        ax = main_axes[rowNo][colNo]

        add_plot_color_by_cluster(Y[0], cluster_labels, ax, label_colors,
                                  label_sizes, clusters_2_label)
    return


def set_main_grid(fig_wid=7.2, fig_hgt=5.76, n_rows=2, n_cols=2,
                  spine_alpha=0, main_wspace=0.005, main_hspace=0.01,
                  main_spns_2_show='all', main_spn_alpha=0.5,
                  main_spn_width=1.0):

    fig = plt.figure(figsize=(fig_wid, fig_hgt))

    back_axis = putl.make_border_axes(fig.add_subplot(111),
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
            ax = putl.make_border_axes(ax,
                                       spines_2_show=main_spns_2_show,
                                       spine_alpha=main_spn_alpha,
                                       spine_width=main_spn_width)
            axes_row.append(ax)
        main_axes.append(axes_row)

    return fig, back_axis, main_gs, main_axes


def add_plot_color_by_cluster(Y, cluster_labels, ax, label_colors, label_sizes,
                              clusters_2_label, scatter_alpha=0.2):

    ax.scatter(*Y.T,
               c=label_colors,
               s=label_sizes,
               alpha=scatter_alpha)

    for cNo, cluster in enumerate(clusters_2_label):
        good_idx = (cluster_labels == cluster).squeeze()

        cluster_median = np.median(Y[good_idx], axis=0)

        cluster_number = cNo + 1
        ax.text(*cluster_median,
                f"{cluster_number}",
                fontsize=12,
                fontweight='bold',
                va='center', ha='center')

                # text_off = 0
                # text_h   = 0
                # for cNo, lab in enumerate(clust_2_label):
                #     good_idx = (labels == lab)

                #     med_Y = np.median(Y[good_idx], axis=0)

                #     ax.text(*med_Y, f"{cNo}",
                #             fontsize=12,
                #             fontweight='bold',
                #             va='center', ha='center')

                #     ax_width = back_axis.get_window_extent().width
                #     ax_height = back_axis.get_window_extent().height / fig.dpi

                #     pad = 3

                #     rect_width = ax.get_window_extent().width / ax_width

                #     if cNo < 5:
                #         x_loc = rect_width / 2
                #         rect_x = 0
                #     else:
                #         x_loc = 1 - (rect_width / 2)
                #         rect_x = 1 - rect_width

                #     cLab = cOnt_labels[lab].title()
                #     if "Slamf1-Negative" in cLab:
                #         cLab = " ".join(cLab.split(" Multipotent "))
                #     cLab = f"{cNo}: " + cLab

                #     bb = ax.text(x_loc, -0.013 - (cNo % 5) * text_h,
                #                  cLab,
                #                  # r"$N=$" + f"{sum(good_idx)}",
                #                  ha='center', va='top',
                #                  fontsize=10,
                #                  transform=back_axis.transAxes)

                #     text_h = (bb.get_size() + 2 * pad) / 72. / ax_height

                #     # if (cNo % 2) == 1:
                #     #     text_off -= text_h

                #     # rect_y = (int(cNo / 2) + 1) * text_h
                #     rect_y = ((cNo % 5) + 1) * text_h

                #     rect = plt.Rectangle((rect_x, -rect_y),
                #                          width=rect_width,
                #                          height=text_h,
                #                          transform=back_axis.transAxes,
                #                          zorder=3,
                #                          fill=True,
                #                          facecolor=cblind_cmap[l2cl[lab]],
                #                          clip_on=False,
                #                          alpha=0.8,
                #                          edgecolor='0.8')

                #     back_axis.add_patch(rect)


###############################################################################
##  RUN THE FILE
###############################################################################
if __name__ == "__main__":

    print("\n\n" + 66 * "=")
    print(f"\n\tGenerating EMBEDR Figure 01v1 (Dim. Red. Zoology)\n")
    print(66 * "=" + "\n\n")

    ###########################################################################
    ##  Set Runtime Parameters
    ###########################################################################
    if True:
        ## Set the figure base name
        name_base = "EMBEDR_Figure_01v1_DimRedZoology"

        ## Select which data to use
        seq_type = "FACS"
        dataset  = "Marrow"

        ## Set parameters at which to plot data
        DR_params = [('tSNE', 7),
                     ('UMAP', 15),
                     ('tSNE', 250),
                     ('UMAP', 400)]

        ## Set other parameters
        n_components   = 2

        tSNE_exag_iter = 250
        tSNE_n_iter    = 1000 - tSNE_exag_iter

        random_seed    = 1
        initialization = 'random'
        n_jobs         = -1

        ## Data directory
        # data_dir = f"../../data/tabula-muris/04_facs_processed_data/"
        data_dir = f"../../data/TabulaMuris/"

        ## Figure directory
        fig_dir = f"./"

        ## Runtime flags
        show_all_axes = False         ## Show ALL axes borders
        color_by_cluster = True       ## Color plot by cell type annotations
        color_by_variability = False  ## Color plot by kEff / dist to NN

    ###########################################################################
    ##  Set Figure Parameters
    ###########################################################################
    if True:
        ## Select clusters to label
        clust_2_label = [1, 10, 0, 8, 6, 4, 9, 7, 2, 3]  ## By size cOnt
        clust_2_label = sorted(clust_2_label)

        ## Environment-wide parameters.
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

        ## Figure size and gridspec size
        my_dpi = 400
        fig_wid = 7.2  ## inches (8 inch-wide paper minus margins)
        fig_hgt = 0.8 * fig_wid

        ## Automatically set n_rows based on conditions and n_cols.
        n_cols = 2
        n_rows = int(np.ceil(len(DR_params) / n_cols))

        ## Main gridspec parameters.
        if color_by_cluster:  ## We need less space if coloring by cluster.
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

        ## Toggle for axes borders.
        if show_all_axes:
            spine_alpha = 1
        else:
            spine_alpha = 0

    ###########################################################################
    ##  Set Coloration Parameters
    ###########################################################################
    if True:

        if dataset.lower() == "mnist":
            label_colors = [cblind_cmap[ll] if ll in clust_2_label
                            else 'lightgrey'for ll in metadata]
            labels = metadata

        elif dataset.lower() in ['marrow']:

            data_path = f"{seq_type}/Processed_{dataset.title()}.h5ad"
            data = sc.read_h5ad(os.path.join(data_dir, data_path))

            ## CELL ONTOLOGY ANNOTATIONS
            cell_ont_meta = data.obs['cell_ontology_class'].values
            cell_ont_ids = np.sort(cell_ont_meta.unique()).squeeze()

            cell_ont_counts = data.obs.groupby('cell_ontology_class')
            cell_ont_counts = cell_ont_counts['cell_ontology_class'].count()

            cell_ont_ids = sorted(cell_ont_ids,
                                  key=lambda cO: -cell_ont_counts[cO])

            cell_ont_labels = np.asarray([f"{cO} (N = {cell_ont_counts[cO]})"
                                          for cO in cell_ont_ids])

            cell_ont_cmap = sns.color_palette('husl', len(cell_ont_ids))

            cell_ont_map = {cO: ii for ii, cO in enumerate(cell_ont_ids)}

            cell_ont_alpha_map = {cO: ii for ii, cO 
                                  in enumerate(np.sort(cell_ont_ids))}

            cell_ont_colors = [cell_ont_cmap[cell_ont_map[cO]]
                               for cO in cell_ont_meta]

            cell_ont_alpha_colors = [cell_ont_cmap[cell_ont_alpha_map[cO]]
                                     for cO in cell_ont_meta]

    ###########################################################################
    ##  Do The Plotting!
    ###########################################################################
    if True:

        pass



            # parsed_meta = pUtl.parse_metadata(metadata)

            # cOnt_labels = parsed_meta['cell_ont_labels']
            # cOnt_map    = parsed_meta['cell_ont_map']
            # cOnt_map_rev = {val: key for key, val in cOnt_map.items()}

            # # labels = metadata['cluster.ids'].values
            # # label_colors = [cblind_cmap[cIdMap[ll]] if (ll in clust_2_label)
            # #                 else 'lightgrey' for ll in labels]

            # labels = [cOnt_map[ll] for ll in metadata['cell_ontology_class']]
            # labels = np.asarray(labels).squeeze()

            # l2cl = {cl: (ii + 3) % 10 for ii, cl in enumerate(clust_2_label)}
            # label_colors = [cblind_cmap[l2cl[ll]] if (ll in clust_2_label)
            #                 else 'lightgrey' for ll in labels]

            # # with open("./Embeddings/temp_labels.pkl", 'rb') as f:
            # #     labels = pkl.load(f)
            # # labels[labels == 21] = 11
            # # l2cl = {cl: ii for ii, cl in enumerate(clust_2_label)}
            # # label_colors = [cblind_cmap[l2cl[ll]] if (ll in clust_2_label)
            # #                 else 'lightgrey' for ll in labels]

            # label_sizes = [3 if (ll in clust_2_label) else 1 for ll in labels]