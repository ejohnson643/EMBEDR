"""
###############################################################################
    Figure: Global Perplexity Sweep (v1)
###############################################################################

    Author: Eric Johnson
    Date Created: Tuesday, March 16, 2021
    Email: ericjohnson1.2015@u.northwestern.edu

###############################################################################

    This figure is meant to illustrate the results of a sweep over the
    perplexity hyperparameter.  To do this, we're going to first show the
    p-Value results at 3 "interesting" perplexities.  We're then going to plot
    the p-Values at a higher resolution across perplexity (k_eff).

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

###############################################################################
##  RUN THE FILE
###############################################################################
if __name__ == "__main__":

    print_str = "  Plotting PaperV4 Figure: Global Parameter Sweep (v1)  "
    print(f"\n\n" + print_str + "\n" + len(print_str) * "=" + "\n")

    ## Define figure and file name format
    name_base = "V4Fig_Global_Parameter_Sweep_v1"

    ## Set perplexity array!
    perp_arr = np.logspace(1, np.log10(5037), 31).astype(int)
    perp_arr = sorted(perp_arr)
    perp_arr = np.hstack((perp_arr[:3], perp_arr[4:]))

    ## Which perps to inset
    perp_2_show    = np.array([28, 1179, 4093])
    perpIdx_2_show = np.array([ii for ii, p in enumerate(perp_arr)
                               if p in perp_2_show])
    perpLab_2_show = np.array([30, 1200, 4000])

    ## p-Value method
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

        ## Data file names
        data_dir      = f"./Data/TabulaMuris/{seq_type}/"
        embed_dir     = f"./Embeddings/Data/"
        data_file     = f"{tissue}_PCA_Embeddings.csv"
        metadata_file = f"{tissue}_Metadata.csv"
        data_name_fmt = "TabulaMurisMarrow_PerpSweep_20PCs_"
        kEff_name = f"TabulaMuris_{tissue}_PCs_kEffective_v2.pkl"

        tmp_file_name = f"{data_name_fmt}tSNE_Embed_00"
        embed_files = [f for f in os.listdir(embed_dir) if tmp_file_name in f]

        fig_base      = name_base + f"_TabulaMuris_{tissue}_{seq_type}_PCs"

        X, metadata = pUtl.load_data(tissue, data_dir)

        n_samples, n_features = X.shape

        ## Set other parameters
        n_components     = 2
        n_embed_expected = 1
        n_jobs         = -1
        initialization = 'random'
        random_seed      = 1
        null_seed        = 100

        ## t-SNE runtime parameters
        early_exag_iter = 250
        n_iter = 750
        momentum = 0.8

        ## Because this is slow and expensive, we want to load the DKLs and
        ## p-Values rather than recomputing.
        print("\nLOADING ALL THE OTHER STUFF!\n")
        try:
            # raise ValueError
            big_dict_file_name = f"{data_name_fmt}_Output_Full.pkl"
            with open("./Embeddings/" + big_dict_file_name, 'rb') as f:
                big_dict = pkl.load(f)
        except FileNotFoundError:
            with open(f"./Embeddings/{data_name_fmt}_Output.pkl", 'rb') as f:
                big_dict = pkl.load(f)
            del big_dict['PArr']
            del big_dict['QArr']
            del big_dict['PNull']
            del big_dict['QNull']

        data_EES = big_dict['DKL']
        null_EES = big_dict['DKLNull']

        try:
            pVal_dict = big_dict['DKLPVals']

            n_embed = len(pVal_dict[perp_arr[1]])

        except KeyError:
            print(f"\nCalculating EMBEDR p-Values!\n")

            pVal_dict = {}
            for ii, perp in enumerate(perp_arr):

                print(f"\nperplexity = {perp}!\n")

                dEES = data_EES[perp].copy()
                nEES = null_EES[perp].copy()

                pVal_arr = pUtl.calc_emp_pVals(dEES, nEES)

                pVal_dict[perp] = pVal_arr.copy()

            big_dict['DKLPVals'] = pVal_dict

            with open("./Embeddings/" + big_dict_file_name, 'wb') as f:
                pkl.dump(big_dict, f)

            del big_dict

        pVals = np.zeros((len(perp_arr), n_samples))

        if pVal_method.lower() == 'simes':
            print(f"Calculating Summary p-Values using Simes' Method!\n")
            simes_mult = n_embed / np.arange(1, n_embed + 1).reshape(-1, 1)

            for ii, perp in enumerate(perp_arr):
                tmp_pVals = np.sort(pVal_dict[perp], axis=0)
                pVals[ii] = np.min(tmp_pVals * simes_mult, axis=0)

        elif pVal_method.lower() == 'average':
            print(f"Calculating Summary p-Values using Averaging Method!\n")

            for ii, perp in enumerate(perp_arr):
                pVals[ii] = np.mean(pVal_dict[perp], axis=0)

        else:
            err_str = f"Unknown p-Value Summary Method '{pVal_method}'"
            raise ValueError(err_str)

        pVals = np.log10(pVals)

        ## Get perp to eff_kNN mapping
        sorted_PWD = pwd(X, metric='sqeuclidean')
        sorted_PWD = np.sort(sorted_PWD, axis=1)[:, 1:]
        print(f"SORTED PWD IS SQUARED EUCLIDEAN")

        alpha_nu = 0.01
        kEff_arr, perp_arr = pUtl.get_kEff(file_name=kEff_name,
                                           file_dir=data_dir,)

    ###########################################################################
    ## Generate / Load Embeddings
    ###########################################################################
    if True:

        data_Y = {}

        for ii, perp in enumerate(perp_2_show):

            print(f"\nLoading embedding for perplexity = {perp}")

            ## Load the data embedding
            dY = pUtl.load_tSNE(X,
                                name_base=fig_base,
                                embed_dir=embed_dir,
                                n_embed=n_embed_expected,
                                n_components=n_components,
                                perplexity=perp,
                                early_exag_iter=early_exag_iter,
                                n_iter=n_iter,
                                initialization=initialization,
                                n_jobs=n_jobs,
                                random_state=random_seed,
                                verbose=True)

            data_Y[perp] = dY[:]

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
        box_color = cblind_cmap[0]

        ## p-Value Colors
        clr_changes = [0, 1, 2, 3, 4.405]
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
        xLab_idx  = np.array([0, 6, 11, 19, 24, 29])
        xLab_aprx = [10, 40, 120, 625, 1800, 5000]

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

            dY = data_Y[perp]

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
                        rotation=-90, fontsize=10,
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
