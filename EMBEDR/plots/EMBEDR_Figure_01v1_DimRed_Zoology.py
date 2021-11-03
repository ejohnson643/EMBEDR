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
from EMBEDR.human_round import human_round
import EMBEDR.plotting_utility as putl
import EMBEDR.utility as utl

import anndata as ad
import matplotlib
import matplotlib.gridspec as gs
import matplotlib.pyplot as plt
import numpy as np
import os
import scanpy as sc
import seaborn as sns
import warnings

warnings.filterwarnings("ignore", message="This figure includes Axes that")
warnings.filterwarnings("ignore", message="tight_layout not applied: ")
warnings.filterwarnings("ignore", message="Creating an ndarray from ragged")


def _make_figure_grid(fig_size=(7.2, 5.76), 
                      n_rows=2,
                      n_cols=2,
                      show_all_borders=False,
                      wspace=0.005,
                      hspace=0.01,
                      spines_2_show='all',
                      spine_alpha=0.5,
                      spine_width=1.0):

    back_spine_alpha = 0
    if show_all_borders:
        back_spine_alpha = 1

    fig = plt.figure(figsize=fig_size)

    back_axis = fig.add_subplot(111)
    back_axis = putl.make_border_axes(back_axis, spine_alpha=back_spine_alpha)

    main_gs = fig.add_gridspec(nrows=n_rows, ncols=n_cols,
                               wspace=wspace, hspace=hspace)

    main_axes = []
    for rowNo in range(n_rows):
        axes_row = []
        for colNo in range(n_cols):
            ax = fig.add_subplot(main_gs[rowNo, colNo])
            ax = putl.make_border_axes(ax, spines_2_show=spines_2_show,
                                       spine_alpha=spine_alpha,
                                       spine_width=spine_width)
            axes_row.append(ax)
        main_axes.append(axes_row)

    return fig, back_axis, main_gs, main_axes


def _add_plot_colored_by_cluster(Y,
                                 labels,
                                 axis,
                                 colors,
                                 sizes,
                                 labels_2_hl,
                                 scatter_alpha=0.2):

    hax = axis.scatter(*Y.T, c=colors, s=sizes, alpha=scatter_alpha)

    for lNo, lab in enumerate(labels_2_hl):
        good_idx = (labels == lab).squeeze()

        label_median = np.median(Y[good_idx], axis=0)

        axis.text(*label_median, "{}".format(lNo + 1), fontsize=12,
                  fontweight='bold', va='center', ha='center')

    return hax


def _add_plot_colored_by_var(Y,
                             labels,
                             axis,
                             sizes,
                             scatter_alpha=0.2,
                             reverse_label=False):

    sort_idx = np.argsort(labels)
    if reverse_label:
        sort_idx = sort_idx[::-1]

    hax = axis.scatter(*Y[sort_idx].T, c=labels[sort_idx], s=sizes[sort_idx],
                       alpha=scatter_alpha)

    return hax


    


def EMBEDR_Figure_01(X,
                     metadata=None,
                     data_dir=None,
                     embedding_params=None,
                     EMBEDR_params=None,
                     project_dir="./",
                     project_name="EMBEDR_Figure_01v1_DimRedZoology",
                     color_by_cluster=True,
                     label_name="cell_ontology_class",
                     labels_2_hl=None,
                     label_colors=None,
                     label_sizes=None,
                     label_params=None,
                     grid_params=None,
                     n_rows=2,
                     n_cols=2,
                     scatter_alpha=0.2,
                     title_size=14,
                     title_pad=-15,
                     add_panel_numbers=False,
                     fig_dir="./",
                     fig_pad=None):

    if metadata is None:
        load_metadata = True

    data_name = ""
    if isinstance(X, str):
        data_name = X.title()
        if load_metadata:
            X, metadata = utl.load_data(X, data_dir=data_dir)
        else:
            X = utl.load_data(X, data_dir=data_dir, load_metadata=False)

    if metadata is None:
        err_str  = f"Metadata must be either loadable with `utl.load_data`"
        err_str += f" or provided.  Metadata is currently `None`..."
        raise ValueError(err_str)

    if embedding_params is None:
        embedding_params = [('tSNE', 9),   ('UMAP', 15),
                            ('tSNE', 350), ('UMAP', 400)]

    if EMBEDR_params is None:
        EMBEDR_params = {}

    if color_by_cluster:
        if label_params is None:
            label_params = {}

        [labels,
         label_counts,
         long_labels,
         lab_2_idx_map,
         label_cmap] = putl.process_categorical_label(metadata,
                                                      label_name,
                                                      **label_params)

        if labels_2_hl is None:
            labels_2_hl = label_counts.index.values[:10]

        if label_colors is None:
            label_colors = [label_cmap[lab_2_idx_map[ll]] if ll in labels_2_hl
                            else 'lightgrey'for ll in labels]
        
        if label_sizes is None:
            label_sizes = [3 if ll in labels_2_hl else 1 for ll in labels]

    elif label_sizes is None:
        label_sizes = 3 * np.ones((len(X)))

    if grid_params is None:
        grid_params = {}

    [fig,
     back_axis,
     main_gs,
     main_axes] = _make_figure_grid(n_rows=n_rows, n_cols=n_cols,
                                    **grid_params)

    if EMBEDR_params is None:
        EMBEDR_params = {'verbose': 1}

    for algNo, (alg, param) in enumerate(embedding_params):
        print(f"Plotting data embedded by {alg} (param = {param})")

        if alg.lower() in ['tsne', 't-sne']:
            embObj = EMBEDR(X=X,
                            perplexity=param,
                            DRA='tsne',
                            n_data_embed=1,
                            n_jobs=-1,
                            project_name=project_name,
                            project_dir=project_dir)
            Y, _ = embObj.get_tSNE_embedding(X)
            kEff = human_round(embObj.kEff)
            title = f"t-SNE: " + r"$k_{Eff} \approx $" + f"{kEff:.0f}"

            if not color_by_cluster:
                labels = np.log10(embObj._kEff)

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
            title = f"UMAP: " + r"$k = $" + f"{param:.0f}"

            if not color_by_cluster:
                kNN_graph = embObj.get_kNN_graph(X)
                labels = np.log10(kNN_graph.kNN_dst[:, param - 1])

        rowNo = int(algNo / n_cols)
        colNo = int(algNo % n_cols)
        axis = main_axes[rowNo][colNo]

        if color_by_cluster:
            hax = _add_plot_colored_by_cluster(Y[0], labels, axis,
                                               label_colors, label_sizes,
                                               labels_2_hl,
                                               scatter_alpha=scatter_alpha)

        else:
            hax = _add_plot_colored_by_var(Y[0], labels, axis, label_sizes,
                                           scatter_alpha=scatter_alpha)

            cax = fig.colorbar(hax, ax=axis, pad=-0.002,
                               drawedges=False)

            c_ticks = cax.get_ticks()
            cax.set_ticks(c_ticks, )
            c_ticklabels = [f"{int(human_round(10**tck))}" for tck in c_ticks]
            cax.set_ticklabels(c_ticklabels)
            cax.ax.yaxis.set_tick_params(pad=-0.5)

            if alg.lower() == 'tsne':
                cax.set_label(r"Effective Nearest Neighbors, $k_{Eff}$",
                              labelpad=-3)
            if alg.lower() == 'umap':
                cax.set_label(r"Distance to $k^{th}$ Neighbor",
                              labelpad=-3)

            cax.solids.set_edgecolor('face')
            fig.canvas.draw()

        axis.set_title(title, fontsize=title_size, pad=title_pad)
        ylim = axis.get_ylim()
        axis.set_ylim(ylim[0], ylim[1] + 0.1 * (ylim[1] - ylim[0]))

    if color_by_cluster:
        text_off = 0
        text_h   = 0

        ax_width = back_axis.get_window_extent().width
        ax_height = back_axis.get_window_extent().height / fig.dpi

        pad = 3

        rect_width = axis.get_window_extent().width / ax_width

        for lNo, lab in enumerate(labels_2_hl):

            if lNo < 5:
                x_loc = rect_width / 2
                rect_x = 0
            else:
                x_loc = 1 - (rect_width / 2)
                rect_x = 1 - rect_width

            if "Slamf1-Negative" in lab:
                lab = " ".join(lab.split(" Multipotent "))
            label_str = f"{lNo + 1}: " + lab.title()

            bb = axis.text(x_loc, -0.013 - (lNo % 5) * text_h,
                           label_str, ha='center', va='top', fontsize=10,
                           transform=back_axis.transAxes)

            text_h = (bb.get_size() + 2 * pad) / 72. / ax_height

            # if (cNo % 2) == 1:
            #     text_off -= text_h

            # rect_y = (int(cNo / 2) + 1) * text_h
            rect_y = ((lNo % 5) + 1) * text_h

            label_color = label_cmap[lab_2_idx_map[lab]]

            rect = plt.Rectangle((rect_x, -rect_y),
                                 width=rect_width,
                                 height=text_h,
                                 transform=back_axis.transAxes,
                                 zorder=3,
                                 fill=True,
                                 facecolor=label_color,
                                 clip_on=False,
                                 alpha=0.8,
                                 edgecolor='0.8')

            back_axis.add_patch(rect)

    fig.tight_layout()

    if add_panel_numbers:

        letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        for rowNo in range(n_rows):
            for colNo in range(n_cols):

                axis = main_axes[rowNo][colNo]
                letter = letters[rowNo * n_cols + colNo]

                _ = putl.add_panel_number(axis, letter, edge_pad=10)

    fig.tight_layout()

    if color_by_cluster:
        fig_base = project_name + f"_{data_name}" + "_ColoredByCluster"
        fig_pad = 0.5 if fig_pad is not None else fig_pad
    else:
        fig_base = project_name + f"_{data_name}" + "_ColoredByVariable"
        fig_pad = 3 if fig_pad is not None else fig_pad

    print(fig_base)
    ## SAVE FIGURE HERE
    putl.save_figure(fig,
                     fig_base,
                     fig_dir=fig_dir,
                     tight_layout_pad=fig_pad)

    return fig































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