"""
###############################################################################
    Figure: Global Perplexity Sweep (v2)
###############################################################################

    Author: Eric Johnson
    Date Created: Wednesday, April 7, 2021
    Email: ericjohnson1.2015@u.northwestern.edu

###############################################################################

    This figure is meant to illustrate the results of a sweep over the
    perplexity hyperparameter.  To do this, we're going to first show the
    p-Value results at 3 "interesting" perplexities.  We're then going to plot
    the p-Values at a higher resolution across perplexity (k_eff).

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

    data_name = "Marrow"
    DR_method = 'tSNE'
    parameter = 'Perplexity'

    print_str  = f"  Plotting Global {DR_method}-{parameter} Sweep on"
    print_str += f" {data_name} Data!  "
    print(f"\n\n" + print_str + "\n" + "=" * len(print_str) + "\n\n")

    file_name_base = f"HyperparamSweep_{DR_method}_{parameter}_{data_name}"
    ## Define figure and file name format
    name_base = "V4Fig_Global_Parameter_Sweep_v2"
    fig_base  = name_base + f"_{DR_method}_{parameter}_{data_name}"

    ## Set the name for the full saved output (this is what another script
    ## could load at once to skip all this junk).
    out_name = file_name_base + "_Output_Dict.pkl"
    out_path = os.path.join(embed_dir, out_name)

    ## Which perps to inset
    if data_name.lower() == 'marrow':
        perp_2_show    = np.array([30, 250, 1300])
    elif data_name.lower() == 'diaphragm':
        perp_2_show    = np.array([40, 200, 600])
    elif data_name.lower() == 'mnist':
        perp_2_show    = np.array([10, 100, 1000])

    perpLab_2_show = pUtl.human_round(perp_2_show)

    ## Runtime flags
    show_all_axes = False

    ###########################################################################
    ## Load the data and metadata!
    ###########################################################################
    if True:

        ## Load data!
        with open(out_path, 'rb') as f:
            out_dict = pkl.load(f)

        ## Unpack everything
        data_Y = out_dict['data_Y']
        null_Y = out_dict['null_Y']
        data_EES = out_dict['data_EES']
        null_EES = out_dict['null_EES']
        kEff_all = out_dict['kEff_arr']
        pVals = out_dict['pVals']

        del out_dict

        perp_arr = np.sort(list(data_Y.keys()))
        perpIdx_2_show = np.array([ii for ii, p in enumerate(perp_arr)
                                   if p in perp_2_show])

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

        ## Colormaps
        cblind_cmap = sns.color_palette('colorblind')
        box_color = 'grey'  #cblind_cmap[0]

        ## p-Value Colors
        clr_changes = [0, 1, 2, 3, min_pVal + 0.005]
        [pVal_cmap,
         pVal_cnorm] = pUtl.make_categ_cmap(change_points=clr_changes)

        color_bounds = np.linspace(clr_changes[0], clr_changes[-1],
                                   pVal_cmap.N)

        pVals_2_show  = np.sort(clr_changes[:-1] + [4])

        ## Figure-level parameters
        my_dpi              = 400
        fig_wid             = 7.5  ## inches (8 inch-wide paper minus margins)
        fig_hgt             = 1.0 * fig_wid
        fig_size            = (fig_wid, fig_hgt)
        fig_pad             = 0.4
        fig_ppad            = 0.01  ## Percent of fig to leave around edge.

        spine_alpha = 0
        if show_all_axes:
            spine_alpha = 1

        ## Set gridspec parameters
        main_hspace       = 0.02
        main_hratios      = [1.0, 1.3]

        top_wspace          = 0.01

        bot_wpad            = 0.0
        bot_hpad            = 0.0
        bot_labelsize       = 16
        bot_xlim = [-1, len(perp_arr)]

        box_fliers      = {'marker': ".",
                           'markeredgecolor': box_color,
                           'markersize': 2,
                           'alpha': 0.5}
        box_props       = {'alpha': 0.5,
                           'color': box_color,
                           'fill': True}
        box_widths      = 0.8
        box_notch       = True
        box_bootstrap   = 100
        box_patch_art   = True
        box_whiskers    = (1, 99)
        box_patches     = ['boxes', 'whiskers', 'fliers', 'caps', 'medians']
        box_show_alpha  = 0.9
        box_alpha       = 0.5

        ## Which perps to show on x-axis
        if data_name.lower() == 'marrow':
            xLab_idx  = np.array([0, 5, 10, 19, 24, 29])
        elif data_name.lower() == 'diaphragm':
            xLab_idx  = np.array([0, 4, 8, 15, 19, 23])
        elif data_name.lower() == 'mnist':
            xLab_idx  = np.array([0, 5, 10, 19, 24, 29])

        ## Embedding parameters
        emb_s          = 2
        emb_alpha      = 1
        emb_xlabel_pad = -12
        emb_ylim_pad   = 1.1

        ## Colorbar parameters
        cax_ticklabels = ["1", "0.1", r"$10^{-2}$", r"$10^{-3}$", r"$10^{-4}$"]
        cax_width_frac = 1.3
        cax_w2h_ratio  = 0.1

    ###########################################################################
    ## Set up figure and gridspec
    ###########################################################################
    if True:

        ## Create the figure!
        fig = plt.figure(figsize=fig_size)

        ## Set up top - bottom subplots
        main_gs = fig.add_gridspec(2, 1,
                                   hspace=main_hspace,
                                   height_ratios=main_hratios)

        ## Set up top axes
        top_behind_ax = fig.add_subplot(main_gs[0])
        top_behind_ax = pUtl.make_border_axes(top_behind_ax,
                                              spine_alpha=spine_alpha)

        ## Gridspec for select embeddings
        top_gs = main_gs[0].subgridspec(nrows=1,
                                        ncols=len(perp_2_show),
                                        wspace=top_wspace)

        ## Axes for select embeddings
        top_axes = []
        for ii in range(len(perp_2_show)):
            ax = fig.add_subplot(top_gs[ii])
            ax = pUtl.make_border_axes(ax, spine_alpha=1, spine_width=0.5)
            top_axes.append(ax)

        ## Set up bottom axis
        bot_behind_ax = fig.add_subplot(main_gs[1])
        bot_behind_ax = pUtl.make_border_axes(bot_behind_ax,
                                              spine_alpha=spine_alpha)

        ## Set up floating bottom gridspec
        bot_gs = gs.GridSpec(nrows=1, ncols=1,
                             wspace=bot_wpad, hspace=bot_hpad)

        bot_ax = pUtl.make_border_axes(fig.add_subplot(bot_gs[0]),
                                       spine_alpha=1)

        pUtl.update_tight_bounds(fig, bot_gs, main_gs[1], w_pad=bot_wpad,
                                 h_pad=bot_hpad, fig_pad=fig_pad)

        fig.tight_layout(pad=fig_pad)

    ###########################################################################
    ## Plot the Global p-Value Curve
    ###########################################################################
    if True:
        print_str  = "\n" + 40 * "=" + "\n\n  Plotting Summary p-Val vs"
        print_str += " Perplexity!\n\n" + 40 * "=" + "\n"
        print(print_str)

        perp_boxes = {}
        for ii, perp in enumerate(perp_arr):

            if perp in perp_2_show:
                box_props['alpha'] = box_show_alpha
            else:
                box_props['alpha'] = box_alpha

            box = bot_ax.boxplot(pVals[ii],
                                 widths=box_widths,
                                 positions=[ii],
                                 notch=box_notch,
                                 bootstrap=box_bootstrap,
                                 patch_artist=box_patch_art,
                                 whis=box_whiskers,
                                 boxprops=box_props,
                                 flierprops=box_fliers)

            for item in box_patches:
                plt.setp(box[item], color=box_color)

            if perp in perp_2_show:
                perp_boxes[perp] = box['boxes'][0]

        bot_ax.set_xticks(np.arange(len(perp_arr))[xLab_idx])
        xticks = [f"{int(kE)}" for kE in pUtl.human_round(kEff_arr[xLab_idx])]
        bot_ax.grid(which='major', axis='x', alpha=0)
        bot_ax.set_xticklabels(xticks)
        bot_ax.set_yticks(-np.sort(clr_changes[:-1] + [4]))
        bot_ax.set_yticklabels([])

        bot_ax.set_xlabel(r"$    k_{Eff}$", fontsize=bot_labelsize,
                          labelpad=-8)
        bot_ax.set_xlim(*bot_xlim)
        bot_ax.tick_params(pad=-3)

        pUtl.update_tight_bounds(fig, bot_gs, main_gs[1], w_pad=bot_wpad,
                                 h_pad=bot_hpad, fig_pad=fig_pad)

    ###########################################################################
    ## Plot the Specified Embeddings
    ###########################################################################
    if True:
        print_str  = "\n" + 40 * "=" + "\n\n  Plotting Specific Embeddings!"
        print_str += "\n\n" + 40 * "=" + "\n"
        print(print_str)

        for pNo, perp in enumerate(perp_2_show):
            pIdx = perpIdx_2_show[pNo]

            dY = data_Y[perp][0]

            tmp_pVals = -pVals[pIdx]

            sort_idx = np.argsort(tmp_pVals)

            ax = top_axes[pNo]
            hax = ax.scatter(*dY[sort_idx].T,
                             s=emb_s,
                             c=tmp_pVals[sort_idx],
                             cmap=pVal_cmap,
                             norm=pVal_cnorm,
                             alpha=emb_alpha)
            ylim = ax.get_ylim()
            ax.set_ylim(ylim[0], emb_ylim_pad * ylim[1])

            kE = pUtl.get_kEff_from_perp(perp_2_show[pNo], kEff_arr, perp_arr)
            kE = int(pUtl.human_round(kE))
            ax.set_xlabel(r"$k_{Eff} \approx $" + f"{kE}",
                          labelpad=emb_xlabel_pad)
            ax.xaxis.set_label_position('top')

    ###########################################################################
    ## Plot the Colorbar...
    ###########################################################################
    if True:
        print_str  = "\n" + 40 * "=" + "\n\n  Plotting the Colorbar!"
        print_str += "\n\n" + 40 * "=" + "\n"
        print(print_str)

        ## Update the figure again...
        pUtl.update_tight_bounds(fig, bot_gs, main_gs[1], w_pad=bot_wpad,
                                 h_pad=bot_hpad, fig_pad=fig_pad)

        inv_ax_trans = bot_ax.transAxes.inverted()
        fig_trans    = fig.transFigure

        ## Convert from data to display
        min_pVal_crds = bot_ax.transData.transform([bot_xlim[0], pVals.min()])
        max_pVal_crds = bot_ax.transData.transform([bot_xlim[0], pVals.max()])

        # print(f"min_pVal_crds: {min_pVal_crds}")
        # print(f"max_pVal_crds: {max_pVal_crds}")

        ## Convert from display to figure coordinates
        cFigX0, cFigY0 = fig.transFigure.inverted().transform(min_pVal_crds)
        cFigX1, cFigY1 = fig.transFigure.inverted().transform(max_pVal_crds)

        # print(f"cFig0: {cFigX0:.4f}, {cFigY0:.4f}")
        # print(f"cFig1: {cFigX1:.4f}, {cFigY1:.4f}")

        cFig_height = np.abs(cFigY1 - cFigY0)
        cFig_width  = cax_w2h_ratio * cFig_height

        # print(f"The color bar will be {cFig_width:.4f} x {cFig_height:.4f}")

        cAxX0, cAxY0 = cFigX0 - cax_width_frac * cFig_width, cFigY0
        cAxX1, cAxY1 = cAxX0 + cFig_width, cFigY0 + cFig_height

        ## Convert from Figure back into Axes
        [cAxX0,
         cAxY0] = inv_ax_trans.transform(fig_trans.transform([cAxX0, cAxY0]))
        [cAxX1,
         cAxY1] = inv_ax_trans.transform(fig_trans.transform([cAxX1, cAxY1]))

        # print(f"cAx0: {cAxX0:.4f}, {cAxY0:.4f}")
        # print(f"cAx1: {cAxX1:.4f}, {cAxY1:.4f}")

        cAx_height = np.abs(cAxY1 - cAxY0)
        cAx_width  = np.abs(cAxX1 - cAxX0)

        # print(f"The color bar will be {cAx_width:.4f} x {cAx_height:.4f}")

        caxIns = bot_ax.inset_axes([cAxX0, cAxY0, cAx_width, cAx_height])
        caxIns = pUtl.make_border_axes(caxIns, spine_alpha=0)

        cAx = fig.colorbar(hax, cax=caxIns, boundaries=color_bounds, ticks=[])
        cAx.ax.invert_yaxis()

        cAx.set_ticks(pVals_2_show)
        cAx.set_ticklabels(cax_ticklabels)
        cAx.ax.tick_params(length=0)
        cAx.ax.yaxis.set_ticks_position('left')

        cAx.ax.set_ylabel(r"EMBEDR $p$-Value",
                          fontsize=bot_labelsize,
                          labelpad=2)
        cAx.ax.yaxis.set_label_position('left')

        pUtl.update_tight_bounds(fig, bot_gs, main_gs[1], w_pad=bot_wpad,
                                 h_pad=bot_hpad, fig_pad=fig_pad)

        fig.tight_layout(pad=fig_pad)

        tform = bot_ax.transAxes.inverted()

        for pNo, perp in enumerate(perp_2_show):

            bX0, bY0, bdX, bdY = perp_boxes[perp].get_window_extent().bounds
            tX = bX0 + (bdX / 2.)
            tY = bY0 + (bdY / 2.)
            tX, tY = tform.transform([tX, tY])
            kE = pUtl.get_kEff_from_perp(perp, kEff_arr, perp_arr)
            kE = int(pUtl.human_round(kE))
            bot_ax.text(tX, tY, r"$k_{Eff} \approx $" + f"{kE}",
                        va='center', ha='center', fontweight='bold',
                        color='w',
                        rotation=-90, fontsize=8,
                        transform=bot_ax.transAxes)

    ###########################################################################
    ## Save and Show
    ###########################################################################
    if True:
        pUtl.update_tight_bounds(fig, bot_gs, main_gs[1], w_pad=bot_wpad,
                                 h_pad=bot_hpad, fig_pad=fig_pad)

        pUtl.add_panel_number(bot_behind_ax, "D", edge_pad=10, fontsize=10)
        pUtl.add_panel_number(top_axes[0], "A", edge_pad=10,
                              fontsize=10)
        pUtl.add_panel_number(top_axes[1], "B", edge_pad=10,
                              fontsize=10)
        pUtl.add_panel_number(top_axes[2], "C", edge_pad=10,
                              fontsize=10)

        print("\n\nSaving Figure!\n\n")
        pUtl.save_figure(fig,
                         fig_base,
                         fig_dir=fig_dir,
                         tight_layout_pad=fig_pad,
                         dpi=my_dpi)

        print("Showing Figure!\n\n")
        plt.show()