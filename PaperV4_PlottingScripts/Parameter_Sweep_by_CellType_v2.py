"""
###############################################################################
    Figure: Perplexity Sweep by Cell Type (v2)
###############################################################################

    Author: Eric Johnson
    Date Created: Wednesday, April 7, 2021
    Email: ericjohnson1.2015@u.northwestern.edu

###############################################################################

    This figure is meant to illustrate the results of a sweep over the
    perplexity hyperparameter.  To do this, we're going to isolate some parts
    of the previous figure to show how the sweeps change when we consider only
    one cell type at a time.

    This version uses the newest parameter sweeps as inputs.  These have been
    verified to use an asymmetric P for DKL calculations and to use the
    averaging method for p-values.

###############################################################################
"""

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
import warnings

warnings.filterwarnings("ignore", message="This figure includes Axes that")
warnings.filterwarnings("ignore", message="tight_layout not applied: ")

EPSILON = np.finfo(np.float64).eps

data_dir = "./Data"
embed_dir = "./Embeddings/ParameterSweep"

if __name__ == "__main__":

    data_name = "Diaphragm"
    DR_method = 'tSNE'
    parameter = 'Perplexity'

    print_str  = f"  Plotting Cell-Type {DR_method}-{parameter} Sweep on"
    print_str += f" {data_name} Data!  "
    print(f"\n\n" + print_str + "\n" + "=" * len(print_str) + "\n\n")

    file_name_base = f"HyperparamSweep_{DR_method}_{parameter}_{data_name}"
    ## Define figure and file name format
    name_base = "V4Fig_CellType_Parameter_Sweep_v2"
    fig_base  = name_base + f"_{DR_method}_{parameter}_{data_name}"

    ## Set the name for the full saved output (this is what another script
    ## could load at once to skip all this junk).
    out_name = file_name_base + "_Output_Dict.pkl"
    out_path = os.path.join(embed_dir, out_name)

    ## Which perps to inset
    if data_name.lower() == 'marrow':
        perp_2_hl    = [250, 1300]
    elif data_name.lower() == 'diaphragm':
        perp_2_hl    = [200, 600]

    ## Which cell ontology classes to show!
    cOnt_rows, cOnt_cols = 2, 3
    if data_name.lower() == 'marrow':
        cOnt_2_show = [['granulocyte',
                        'macrophage',
                        'granulocytopoietic cell'
                        ],
                       ['immature B cell',
                        'common lymphoid progenitor',
                        'Slamf1-negative multipotent progenitor cell'
                        ]]

    elif data_name.lower() == 'diaphragm':
        cOnt_2_show = [['skeletal muscle satellite stem cell',
                        'mesenchymal stem cell',
                        'lymphocyte'
                        ],
                       ['endothelial cell',
                        'macrophage',
                        ""
                        ]]

    if data_name.lower() == 'marrow':
        thresh = 3
    elif data_name.lower() == 'diaphragm':
        thresh = 1

    ## Runtime flags
    show_all_axes = False

    ###########################################################################
    ## Load the data and metadata!
    ###########################################################################
    if True:
        ## Data file names
        if data_name.lower() in ['marrow', 'diaphragm']:
            data_dir = f"./Data/TabulaMuris/FACS/"

        X, metadata = pUtl.load_data(data_name, data_dir)

        meta_dict = pUtl.parse_metadata(metadata)
        cell_ont_labels = meta_dict['cell_ont_labels']
        cell_ont_map    = meta_dict['cell_ont_map']
        cell_ont_column = metadata['cell_ontology_class']

        ## Load data!
        with open(out_path, 'rb') as f:
            out_dict = pkl.load(f)

        ## Unpack everything
        data_Y = out_dict['data_Y']
        kEff_all = out_dict['kEff_arr']
        pVals = out_dict['pVals']

        del out_dict

        perp_arr = np.sort(list(data_Y.keys()))
        perpIdx_2_show = np.array([ii for ii, p in enumerate(perp_arr)
                                   if p in perp_2_hl])

        n_samples, n_components = data_Y[perp_arr[0]][0].shape

        kEff_arr = np.median(kEff_all, axis=1)

        pVals = np.log10(pVals)
        min_pVal = -np.min(pVals)

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

        ## Figure Size Parameters
        my_dpi     = 400  ## Pixels per inch.
        fig_width  = 7.5  ## inches (8 inch-wide paper minus margins)
        fig_height = 1.0 * fig_width
        fig_size   = (fig_width, fig_height)

        ## Figure parameters
        fig_pad             = 0.0
        fig_ppad            = 0.01  ## Percent of fig to leave around edge.

        main_hspace         = 0.0
        main_hratios        = [1.1, 1.0]

        top_nrows           = cOnt_rows
        top_ncols           = cOnt_cols
        top_wspace          = 0.5
        top_hspace          = 1.0
        top_wpad            = top_wspace
        top_hpad            = top_hspace
        top_spine_alpha     = 1
        top_pVal_hl_alpha   = 0.3

        bot_ncols           = len(perp_2_hl)
        bot_nrows           = 1
        bot_hspace          = 0
        bot_hpad            = bot_hspace
        bot_wspace          = 0.5
        bot_wpad            = bot_wspace
        bot_pval_wratios    = len(perp_2_hl) * [1]
        bot_spine_alpha     = 1.0
        bot_spine_lw        = 0.5

        ## Which perps to show on x-axis
        perp_2_hl_idx = [np.argmin(np.abs(perp_arr - perp))
                         for perp in perp_2_hl]
        xtick_idx   = [0] + perp_2_hl_idx + [len(perp_arr) - 1]
        if data_name.lower() == 'marrow':
            xticklabels = [perp_arr[0]] + perp_2_hl + [perp_arr[-1]]
        elif data_name.lower() == 'diaphragm':
            xticklabels = [perp_arr[0]] + perp_2_hl
        xticklabels = pUtl.get_kEff_from_perp(xticklabels, kEff_arr, perp_arr)
        xticklabels = pUtl.human_round(xticklabels).astype(int)

        ## Color maps
        base_cmap   = sns.color_palette()
        cblind_cmap = sns.color_palette('colorblind')
        bright_cmap = sns.color_palette('bright')

        ## Colors to use
        col_idx = [0, 6, 1, 2, 8, 3]
        hl_colors = [cblind_cmap[ll] for ll in col_idx]
        null_color = bright_cmap[4]

        ## p-Value Colors
        min_pVal = -pVals.min() + 0.005
        pVal_clr_change = [0, 1, 2, 3, min_pVal]
        [pVal_cmap,
         pVal_cnorm] = pUtl.make_categ_cmap(change_points=[0, min_pVal],
                                            categ_cmap=bright_cmap,
                                            cmap_idx=[4],
                                            reverse_last_interval=False)

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
        main_gs = fig.add_gridspec(2, 1,
                                   hspace=main_hspace,
                                   height_ratios=main_hratios)

        top_bkgd_ax = fig.add_subplot(main_gs[0])
        top_bkgd_ax = pUtl.make_border_axes(top_bkgd_ax,
                                            spine_alpha=spine_alpha)

        bot_bkgd_ax = fig.add_subplot(main_gs[1])
        bot_bkgd_ax = pUtl.make_border_axes(bot_bkgd_ax,
                                            spine_alpha=spine_alpha)

        top_gs = gs.GridSpec(nrows=top_nrows,
                             ncols=top_ncols,
                             wspace=top_wspace,
                             hspace=top_hspace)

        top_kws = {'spine_alpha': top_spine_alpha,
                   'spines_2_show': ['left', 'bottom'],
                   'xticks': None,
                   'xticklabels': None,
                   'yticks': None,
                   'yticklabels': None}

        top_axes = []
        for ii in range(top_nrows):
            tmp = []
            for jj in range(top_ncols):
                ax = fig.add_subplot(top_gs[ii, jj])
                ax = pUtl.make_border_axes(ax, **top_kws)
                tmp.append(ax)
            top_axes.append(tmp)

        bot_gs = gs.GridSpec(nrows=bot_nrows,
                             ncols=bot_ncols,
                             wspace=bot_wspace,
                             hspace=bot_hspace,
                             width_ratios=bot_pval_wratios)

        bot_axes = []
        for ii in range(bot_ncols):
            ax = fig.add_subplot(bot_gs[ii])
            ax = pUtl.make_border_axes(ax, spine_alpha=bot_spine_alpha,
                                       spine_width=bot_spine_lw)

            bot_axes.append(ax)

        fig.tight_layout(pad=fig_pad)

        pUtl.update_tight_bounds(fig, top_gs, main_gs[0], w_pad=top_wpad,
                                 h_pad=top_hpad, fig_pad=fig_pad)

        pUtl.update_tight_bounds(fig, bot_gs, main_gs[1], w_pad=bot_wpad,
                                 h_pad=bot_hpad, fig_pad=fig_pad)

        fig.subplots_adjust(left=0 + fig_ppad,
                            right=1 - fig_ppad,
                            bottom=0 + fig_ppad,
                            top=1 - fig_ppad)

        pUtl.update_tight_bounds(fig, top_gs, main_gs[0], w_pad=top_wpad,
                                 h_pad=top_hpad, fig_pad=fig_pad)

        pUtl.update_tight_bounds(fig, bot_gs, main_gs[1], w_pad=bot_wpad,
                                 h_pad=bot_hpad, fig_pad=fig_pad)

    ###########################################################################
    ## Plot the Cell-Type p-Value Curves
    ###########################################################################
    if True:
        print_str  = "\n" + 40 * "=" + "\n\nPlotting Summary p-Val vs"
        print_str += " Perplexity!\n\n" + 40 * "=" + "\n"
        print(print_str)

        for ii in range(cOnt_rows):
            for jj in range(cOnt_cols):
                cOnt = cOnt_2_show[ii][jj]

                ax = top_axes[ii][jj]

                if len(cOnt) < 1:
                    ax.set_xticks([])
                    ax.set_yticks([])
                    continue

                print(f"\nPlotting {cOnt}!")

                good_idx = (metadata['cell_ontology_class'] == cOnt).values
                good_pVals = pVals[:, good_idx]

                cOnt_color = hl_colors[ii * cOnt_cols + jj]

                ax.axhline(-thresh, color='0.8', lw=2)

                ax.plot(np.arange(len(perp_arr)),
                        good_pVals,
                        "-",
                        color=cOnt_color,
                        lw=0.5,
                        alpha=20 / sum(good_idx) + 0.05)

                ax.plot(np.arange(len(perp_arr)),
                        np.percentile(good_pVals,
                                      [90],  #[10, 90],
                                      axis=1).T,
                        "-",
                        color=cOnt_color,
                        lw=1.3,
                        alpha=1)

                ax.plot(np.arange(len(perp_arr)),
                        np.median(good_pVals, axis=1),
                        "-s",
                        color=cOnt_color,
                        lw=1.,
                        alpha=1,
                        markersize=5,
                        markerfacecolor=cOnt_color,
                        markeredgecolor='w',
                        markeredgewidth=0.8)

                ax.set_xticks(xtick_idx)
                ax.grid(which='major', axis='x', color='w', alpha=0)
                ax.grid(which='major', axis='y', color='w', alpha=0)
                ax.tick_params(axis='x', size=4, width=1, color='0.8',
                               tick1On=True, pad=2)
                if ii == (cOnt_rows - 1):
                    ax.set_xticklabels(xticklabels)
                    ax.set_xlabel(r"    $k_{Eff}$", labelpad=0)
                else:
                    ax.set_xticklabels([])

                ax.set_yticks(-np.sort(pVal_clr_change[:-1] + [4]))
                if jj == 0:
                    ax.set_yticklabels(["1", "0.1", r"$10^{-2}$",
                                        r"$10^{-3}$", r"$10^{-4}$"])
                    ax.set_ylabel(r"EMBEDR $p$-Value")
                else:
                    ax.set_yticklabels([])

                ylim = ax.get_ylim()
                ax.set_ylim(ylim[0], 0.08)

                ax.set_xlim(0, len(perp_arr) - 1)

                title = cell_ont_labels[cell_ont_map[cOnt]]
                title = title.title().split(" (N")
                if "Common" in title[0]:
                    title = "Common Lymphoid\nProgenitor (N" + title[1]
                elif "Basophil" in title[0]:
                    title = " (N".join(title)
                elif "Slamf1" in title[0]:
                    title = "Slamf1-Negative\nProgenitor (N" + title[1]
                else:
                    title = "\n(N".join(title)
                print(title)

                ax.set_title(title, fontsize=12)

    ###########################################################################
    ## Plot the Highlighted Embeddings at Different Perplexities
    ###########################################################################
    if True:
        print_str  = "\n" + 50 * "=" + "\n\nPlotting Embeddings with"
        print_str += " Highlighted Cell Types!\n\n" + 50 * "=" + "\n"
        print(print_str)

        for ii, pId in enumerate(perp_2_hl_idx):

            perp = perp_2_hl[ii]

            dY = data_Y[perp][0]

            ax = bot_axes[ii]

            tmp_pVals = -pVals[pId]
            sort_idx = np.argsort(tmp_pVals)

            bad_idx = np.array([cO not in cOnt_2_show[0] + cOnt_2_show[1]
                                for cO in cell_ont_column])

            ax.scatter(*dY[sort_idx[bad_idx]].T,
                       s=2,
                       c='lightgrey',
                       alpha=0.2)


            kE = pUtl.get_kEff_from_perp(perp_2_hl[ii],
                                         kEff_arr,
                                         perp_arr)
            kE = pUtl.human_round(kE)[0]
            title  = r"$k_{Eff} \approx $" + f"{int(kE)}"
            # title += r"; $p_i<10^{-" + f"{int(thresh)}" + r"}$"
            ax.set_title(title)

            ax_width = ax.get_window_extent().width

            for jj in range(cOnt_rows):
                for kk in range(cOnt_cols):
                    cOnt = cOnt_2_show[jj][kk]

                    cOnt_idx = (cell_ont_column == cOnt).values

                    cOnt_color = hl_colors[jj * cOnt_cols + kk]

                    ax.scatter(*dY[cOnt_idx].T,
                               s=2,
                               color=cOnt_color,
                               alpha=0.05)
                    good_idx = tmp_pVals[cOnt_idx] >= thresh
                    good_Y   = dY[cOnt_idx][good_idx]

                    ax.scatter(*good_Y.T,
                               s=3,
                               color=cOnt_color,
                               alpha=0.5)

                    ax_frac = 1 / (cOnt_rows * cOnt_cols)

                    x_loc = ax_frac * (jj * cOnt_cols + kk) + ax_frac / 2

                    if len(cOnt) < 1:
                        txt = ""
                    else:
                        txt = f"{sum(good_idx) / sum(cOnt_idx):.1%}"
                    bb = ax.text(x_loc, -0.015, txt,
                                 ha='center', va='top',
                                 fontsize=10, fontweight='bold',
                                 transform=ax.transAxes)
                    ax_height = ax.get_window_extent().height / fig.dpi
                    pad = 3
                    text_h = (bb.get_size() + 2 * pad) / 72. / ax_height
                    txt_xy = (ax_frac * (jj * cOnt_cols + kk), -text_h)
                    rect = plt.Rectangle(txt_xy, width=ax_frac, height=text_h,
                                         transform=ax.transAxes, zorder=3,
                                         fill=True, facecolor=cOnt_color,
                                         clip_on=False, alpha=1.0,
                                         edgecolor='0.8')

                    ax.add_patch(rect)

            ylim = ax.get_ylim()
            ax.set_ylim(1.2 * ylim[0], ylim[1])

            txt  = r"Fraction of cells with"
            txt += r" $p_i<10^{-" + f"{int(thresh)}" + r"}$"
            bb = ax.text(0.5, 0.002, txt,
                         ha='center', va='bottom', fontsize=10,
                         transform=ax.transAxes)
            pad = 3
            text_h = (bb.get_size() + 2 * pad) / 72. / ax_height
            rect = plt.Rectangle((0, 0), width=1, height=text_h,
                                 transform=ax.transAxes, zorder=3,
                                 fill=True, facecolor='lightgrey',
                                 clip_on=False, alpha=0.2, edgecolor='0.8')

            ax.add_patch(rect)

    ###########################################################################
    ## Save and Show
    ###########################################################################
    if True:
        pUtl.update_tight_bounds(fig, top_gs, main_gs[0], w_pad=top_wpad,
                                 h_pad=top_hpad, fig_pad=fig_pad)

        pUtl.update_tight_bounds(fig, bot_gs, main_gs[1], w_pad=bot_wpad,
                                 h_pad=bot_hpad, fig_pad=fig_pad)

        pUtl.add_panel_number(top_bkgd_ax, "A", corner=('left', 'top'),
                              number_loc=(None, .01), edge_pad=20, fontsize=12)

        pUtl.add_panel_number(bot_axes[0], "B",
                              corner=('left', 'top'),
                              edge_pad=10, fontsize=12)

        pUtl.add_panel_number(bot_axes[1], "C",
                              corner=('right', 'top'),
                              edge_pad=10, fontsize=12)

        print("\n\nSaving Figure!\n\n")
        pUtl.save_figure(fig,
                         fig_base,
                         fig_dir=fig_dir,
                         tight_layout_pad=None,
                         dpi=my_dpi)

        print("Showing Figure!\n\n")
        plt.show()