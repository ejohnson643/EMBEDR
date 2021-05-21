"""
###############################################################################
    Paper Version 4 Plotting Utility
###############################################################################

    Author: Eric Johnson
    Date Created: Monday, March 8, 2021
    Email: ericjohnson1.2015@u.northwestern.edu

###############################################################################

    For generating all the figures, there have been established a whole suite
    of functions that are useful.  I will aggregate them here with some
    documentation so that I am not constantly reinventing the wheel.

###############################################################################
"""

from collections import Counter
from embedr._tsne import compute_fixedEntropy_GaussianAffinity as calcAff
import matplotlib.colors as mcl
import numpy as np
from openTSNE import TSNE, TSNEEmbedding
from openTSNE.initialization import random as initRand
import os
from os import path
import pandas as pd
import pickle as pkl
import seaborn as sns
import time
print(f"Importing UMAP, this can take a while...")
from umap import UMAP
from version_5_0.embedr.affinity import FixedEntropyAffinity

## Default directories
data_dir  = "../Data/"
fig_dir   = "../Figures/PresentationFigures/PaperV4/"
embed_dir = "../Embeddings/"

###############################################################################
##  Functions for Plotting
###############################################################################


def save_figure(fig,
                fig_name,
                fig_dir=fig_dir,
                formats=['pdf', 'png', 'svg'],
                tight_layout_pad=None,
                bbox_inches=None,
                transparent=True,
                dpi=600,
                verbose=True):
    """Wrapper to save a given figure using several formats"""

    if verbose:
        fmt_str = ", ".join([f"{fmt}" for fmt in formats])
        print(f"\nSaving {fig_name} to file (" + fmt_str + ")")

    if tight_layout_pad is not None:
        fig.tight_layout(pad=tight_layout_pad)

    fig_path = path.join(fig_dir, fig_name)

    for fmt in formats:
        fig_fmt_path = fig_path + "." + fmt
        if verbose:
            print(f"Saving {fig_fmt_path}")

        fig.savefig(fig_fmt_path,
                    format=fmt,
                    bbox_inches=bbox_inches,
                    transparent=transparent,
                    dpi=dpi)
    return


def make_border_axes(axis,
                     patch_alpha=0,
                     xticks=[],
                     yticks=[],
                     xticklabels=[],
                     yticklabels=[],
                     spine_width=1.25,
                     spine_color='0.8',
                     spine_alpha=1.0,
                     spines_2_show='all',
                     visible=True):

    valid_spines = ['top', 'left', 'bottom', 'right']
    if spines_2_show == 'all':
        spines_2_show = valid_spines

    for spine in spines_2_show:
        if spine not in valid_spines:
            err_str =  f"Spine '{spine}' is not a valid spine label! ('top',"
            err_str += f" 'left', 'bottom', 'right')"
            raise ValueError(err_str)

    for spine in axis.spines:
        if spine in spines_2_show:
            axis.spines[spine].set_alpha(spine_alpha)
            axis.spines[spine].set_linewidth(spine_width)
            axis.spines[spine].set_color(spine_color)
        else:
            axis.spines[spine].set_alpha(0)
            axis.spines[spine].set_linewidth(0)
            axis.spines[spine].set_color('w')

    axis.patch.set_alpha(patch_alpha)

    if xticks is not None:
        axis.set_xticks(xticks)

    if xticklabels is not None:
        axis.set_xticks(xticklabels)

    if yticks is not None:
        axis.set_yticks(yticks)

    if yticklabels is not None:
        axis.set_yticks(yticklabels)

    axis.set_visible(visible)

    return axis


def add_panel_number(axis,
                     number,
                     edge_pad=15,
                     number_loc=(None, None),
                     hw_ratio=None,
                     corner=('left', 'top'),
                     **text_kws):

    default_kws = {'ha': 'left',
                   'va': 'bottom',
                   'fontsize': 10,
                   'bbox': {'boxstyle': 'round',
                            'facecolor': 'w',
                            'alpha': 0.5,
                            'edgecolor': 'k'},
                   'fontweight': 'bold'}

    ## Get axes position in figure fraction
    ax_X0, ax_Y0, ax_wid, ax_hgt = axis.get_position().bounds

    ## Get the axis' figure
    fig = axis.get_figure()

    ## Get the coordinates of the axes in DISPLAY UNITS
    disp_X0, disp_Y0 = fig.transFigure.transform([ax_X0, ax_Y0])

    ## Set the height and width of the panel number box using DISPLAY UNITS
    width = edge_pad
    if hw_ratio is None:
        height = width
    else:
        height = hw_ratio * width

    ## Get the position in AXES FRACTION
    p_X0, p_Y0 = axis.transAxes.inverted().transform([width  + disp_X0,
                                                      height + disp_Y0])

    ## Depending on the corner, adjust the position and text alignment
    if corner[0] == 'right':
        p_X0 = 1 - p_X0
        default_kws['ha'] = 'right'
    elif corner[0] != 'left':
        raise ValueError("corner[0] must be 'left' or 'right'!")

    if corner[1] == 'top':
        p_Y0 = 1 - p_Y0
        default_kws['va'] = 'top'
    elif corner[1] != 'bottom':
        raise ValueError("corner[0] must be 'top' or 'bottom'!")

    ## Get the actual text coordinates
    x, y = number_loc
    if x is None:
        x = 0
    if y is None:
        y = 0

    p_X0 += x
    p_Y0 += y

    default_kws.update(text_kws)

    axis.text(p_X0, p_Y0, number, transform=axis.transAxes, **default_kws)

    return axis


def update_tight_bounds(fig,
                        inner_gs,
                        outer_gs,
                        rect=None,
                        inner_pad=1.08,
                        w_pad=0,
                        h_pad=0,
                        fig_pad=0):

    fig.tight_layout(pad=fig_pad)

    out_crds = outer_gs.get_position(fig).bounds

    if rect is None:
        rect = [out_crds[0],
                out_crds[1],
                out_crds[0] + out_crds[2],
                out_crds[1] + out_crds[3]]

    inner_gs.tight_layout(fig,
                          rect=rect,
                          pad=inner_pad,
                          w_pad=w_pad,
                          h_pad=h_pad)

    return

###############################################################################
##  Functions for Dealing with Tabula Muris metadata and coloring
###############################################################################


def parse_metadata(metadata):

    out = {}

    ## CELL CLUSTERING LABELS
    cluster_ids = np.sort(metadata['cluster.ids'].unique()).squeeze()
    out['cluster_ids'] = cluster_ids

    cId_counts  = metadata.groupby('cluster.ids')
    cId_counts  = cId_counts['cluster.ids'].count()
    out['cluster_counts'] = cId_counts

    cId_labels  = [f"Cluster {cId + 1} (N = {cId_counts[cId]})"
                   for cId in cluster_ids]
    out['cluster_labels'] = cId_labels

    cId_cmap    = sns.color_palette('husl', len(cluster_ids))
    out['cluster_cmap'] = cId_cmap

    cId_colors  = [cId_cmap[cId] for cId in metadata['cluster.ids']]
    out['cluster_colors'] = cId_colors

    ## CELL ONTOLOGY ANNOTATIONS
    cell_ont_meta   = metadata['cell_ontology_class']

    cell_ont_ids    = np.sort(cell_ont_meta.unique()).squeeze()

    cell_ont_counts = metadata.groupby('cell_ontology_class')
    cell_ont_counts = cell_ont_counts['cell_ontology_class'].count()
    out['cell_ont_counts'] = cell_ont_counts

    cell_ont_ids    = sorted(cell_ont_ids, key=lambda cO: -cell_ont_counts[cO])
    out['cell_ont_ids'] = cell_ont_ids

    cell_ont_labels = [f"{cO} (N = {cell_ont_counts[cO]})"
                       for cO in cell_ont_ids]
    out['cell_ont_labels'] = cell_ont_labels

    cell_ont_cmap   = sns.color_palette('husl', len(cell_ont_ids))
    out['cell_ont_cmap'] = cell_ont_cmap

    cell_ont_map    = {cO: ii for ii, cO in enumerate(cell_ont_ids)}
    out['cell_ont_map'] = cell_ont_map

    cell_ont_alpha_map = {cO: ii
                          for ii, cO in enumerate(np.sort(cell_ont_ids))}
    out['cell_ont_alpha_map'] = cell_ont_alpha_map

    cell_ont_colors = [cell_ont_cmap[cell_ont_map[cO]] for cO in cell_ont_meta]
    out['cell_ont_colors'] = cell_ont_colors

    cell_ont_alpha_colors = [cell_ont_cmap[cell_ont_alpha_map[cO]]
                             for cO in cell_ont_meta]
    out['cell_ont_alpha_colors'] = cell_ont_alpha_colors

    return out


def make_categ_cmap(change_points=None,
                    categ_cmap=None,
                    cmap_idx=None,
                    cmap_dx=0.001,
                    reverse_last_interval=True,
                    max_diverge=0.75):
    """Make a categorical colormap that fades between colors at specific values

    This function takes in a list of end points + interior points to set as
    the edge of regions on a colormap.  The function then returns a new
    continuous colormap that transitions between these regions (fades to white,
    then changes colors).
    """

    ## Set the points at which the p-value color will change.
    if change_points is None:
        change_points = [0, 1, 2, 3, 4]  ## -log10 of p-Values.

    ## Set the categorical colormap
    if categ_cmap is None:
        categ_cmap = sns.color_palette('colorblind')

    ## Set the list of indices to use from the colormap
    if cmap_idx is None:
        cmap_idx = [4, 0, 3, 2] + list(range(4, len(change_points) - 1))
        cmap_idx = (np.asarray(cmap_idx) % 10).astype(int)

    ## Set the base colors for regions of the colormap
    colors = [categ_cmap[idx] for idx in cmap_idx]

    ## Make an appropriate grid of points on which to set colors.
    color_grid = []
    for intNo, end in enumerate(change_points[1:]):
        color_grid += list(np.arange(change_points[intNo], end, cmap_dx))
    color_grid += [change_points[-1]]
    color_grid = np.sort(np.unique(np.asarray(color_grid)).squeeze())

    ## Initialize the RGB+ array.
    cmap_colors = np.ones((len(color_grid), 4))

    ## Iterate through the grid, setting interpolated colors for each region.
    start_idx = 0
    for intNo, start in enumerate(change_points[:-1]):

        ## Get the number of grid points in this interval
        N_ticks = int((change_points[intNo + 1] - start) / cmap_dx)
        ## If it's the last interval, add an extra.
        if intNo == (len(change_points) - 2):
            N_ticks += 1

        ## Iterate through each of the RGB values.
        for jj in range(3):

            ## Base color for each interval
            base_color = colors[intNo][jj]

            ## Maximum divergence from the base color.
            upper_bound = max_diverge * (1 - base_color) + base_color

            ## Interpolated grid for the interval.
            intv_color_grid = np.linspace(base_color, upper_bound, N_ticks)

            ## If we're in the last interval, can reverse the direction
            if (intNo == (len(change_points) - 2)) and reverse_last_interval:
                intv_color_grid = intv_color_grid[::-1]

            ## Set the colors!
            cmap_colors[start_idx:start_idx + N_ticks, jj] = intv_color_grid

        start_idx += N_ticks

    ## Convert the grids and colors to matplotlib colormaps.
    cmap = mcl.ListedColormap(cmap_colors)
    cnorm = mcl.BoundaryNorm(color_grid, cmap.N)

    return cmap, cnorm


def make_seq_cmap(color_1, color_2, n_colors=10):

    out_colors = np.zeros((n_colors, 3))
    for ii in range(3):
        out_colors[:, ii] = np.linspace(color_1[ii], color_2[ii], n_colors)

    return out_colors


def cyclic_perm(a):
    n = len(a)
    b = [[a[i - j] for i in range(n)] for j in range(n)]
    return b


def get_nice_marrow_labels():
    nice_labels = ['Granulocyte (N = 761)',
                   'Slamf1-Negative Multipotent\nProgenitor Cell (N = 710)',
                   'Naive B Cell (N = 697)',
                   'Precursor B Cell (N = 517)',
                   'Immature B Cell (N = 344)',
                   'Late Pro-B Cell (N = 301)',
                   'Hematopoietic Precursor\nCell (N = 268)',
                   'Monocyte (N = 266)',
                   'Granulocytopoietic\nCell (N = 221)',
                   'Macrophage (N = 173)',
                   'Common Lymphoid\nProgenitor (N = 156)',
                   'Slamf1-Positive Multipotent\nProgenitor Cell (N = 135)',
                   'Granulocyte Monocyte\nProgenitor Cell (N = 134)',
                   'Immature T Cell (N = 66)',
                   'Megakaryocyte-Erythroid\nProgenitor Cell (N = 54)',
                   'Mature Natural Killer\nCell (N = 48)',
                   'B Cell (N = 45)',
                   'Immature Nk T Cell (N = 41)',
                   'Immature Natural Killer\nCell (N = 37)',
                   'Basophil (N = 25)',
                   'Pre-Natural Killer (N = 22)',
                   'Regulatory T Cell (N = 16)']

    # nice_labels = ["Granulocyte",
    #                'Slamf1-Negative\n' +
    #                'Multipotent Progenitor',
    #                'Naive B Cell',
    #                'Precursor B Cell',
    #                'Immature B Cell',
    #                'Late Pro-B Cell',
    #                'Hematopoietic\n' +
    #                'Precursor Cell',
    #                'Monocyte',
    #                'Granulocytopoietic Cell',
    #                'Macrophage',
    #                'Common Lymphoid\n' +
    #                'Progenitor',
    #                'Slamf1-Positive\n' +
    #                'Multipotent Progenitor',
    #                'Granulocyte Monocyte\n' +
    #                'Progenitor Cell',
    #                'Immature T Cell',
    #                'Megakaryocyte-\n' +
    #                'Erythroid Progenitor',
    #                'Mature Natural\n' +
    #                'Killer Cell',
    #                'B Cell',
    #                'Immature Nk T Cell',
    #                'Immature Natural\n' +
    #                'Killer Cell',
    #                'Basophil',
    #                'Pre-Natural Killer',
    #                'Regulatory T Cell']

    return nice_labels


def wrap_strings(str_arr, line_len=26):

    out_list = []
    for line in str_arr:

        total_char = len(line)

        opt_breaks = int(total_char / line_len)
        # pot_breaks = line.count("-") + line.count(" ")
        words = line.split(" ")
        alt_words = [word.split("-") for word in words]

        words = [wrd for word in alt_words for wrd in word]

        front_sum = np.cumsum([len(word) for word in words])

        new_words = []
        for ii, word in enumerate(words[:-1]):
            new_words.append(word + line[front_sum[ii] + ii])
        words = new_words + [words[-1]]

        word_lens = [len(word.strip()) for word in words]
        if np.all([wl > line_len for wl in word_lens]):
            out_list.append("\n".join(words))

        out_lines = []

        for fudgeNo in range(opt_breaks + 1):
            out_line = ""
            curr_line = ""
            break_counter = 0
            for idx, word in enumerate(words):

                if break_counter < opt_breaks:

                    if (len(word.strip()) + len(curr_line)) <= line_len:
                        curr_line += word
                        out_line += word

                    elif break_counter == fudgeNo - 1:
                        curr_line += word
                        out_line += word
                        fudgeNo = -1

                    else:
                        out_line = out_line.strip() + "\n" + word
                        curr_line = word
                        break_counter += 1

                else:
                    out_line += word

            if len(out_line.split("\n")) < (opt_breaks + 1):
                out_line += (opt_breaks + 1 - len(out_line.split("\n"))) * "\n"

            out_lines.append(out_line)

        words[-1] += " "
        for fudgeNo in range(opt_breaks + 1):
            out_line = ""
            curr_line = ""
            break_counter = 0
            for idx, word in enumerate(words[::-1]):

                word = word[::-1]

                if break_counter < opt_breaks:

                    if (len(word.strip()) + len(curr_line)) <= line_len:
                        curr_line += word
                        out_line += word

                    elif break_counter == fudgeNo:
                        curr_line += word
                        out_line += word
                        fudgeNo = -1

                    else:
                        out_line = out_line.strip() + "\n" + word
                        curr_line = word
                        break_counter += 1

                else:
                    out_line += word

            if len(out_line.split("\n")) < (opt_breaks + 1):
                out_line += (opt_breaks + 1 - len(out_line.split("\n"))) * "\n"

            out_line = out_line[::-1]

            out_line = "\n".join([ll.strip() for ll in out_line.split("\n")])

            # split_line = out_line.split("\n")
            # split_line = [line.split(" ") for line in split_line]
            # print(split_line)

            out_lines.append(out_line)

        print(out_lines)

        out_lines = np.asarray(out_lines)
        line_lens = np.asarray([[len(word) for word in out_line.split('\n')]
                                for out_line in out_lines])
        # line_lens = np.asarray()

        print(line_lens, line_len)

        all_good = np.asarray([np.all(ll < line_len) for ll in line_lens])

        if np.sum(all_good) == 1:
            out_list.append(out_lines[all_good][0])

        elif np.any(all_good):
            len_col = np.ones((sum(all_good), 1)) * line_len
            SSR = np.sum((line_lens[all_good] - len_col)**2, axis=1)
            best_idx = np.argmin(SSR)

            out_list.append(out_lines[all_good][best_idx])

        else:
            len_col = np.ones((len(out_lines), 1)) * line_len
            SSR = np.sum((line_lens - len_col)**2, axis=1)
            best_idx = np.argmin(SSR)

            out_list.append(out_lines[best_idx])


        # front_sum = np.cumsum([len(word) for word in words])[:-1]
        # back_sum = np.cumsum([len(word) for word in words[::-1]])[:-1][::-1]

        # front_break_lens = []
        # back_break_lens = []

        # brk_f_len = line_len
        # brk_b_len = line_len

        # front_lines = []
        # back_lines = []

        # f_idx = 0
        # b_idx = len(words)

        # for bNo in range(opt_breaks):

        #     print(brk_b_len, b_idx, back_lines)

        #     best_front_break = (front_sum <= brk_f_len).nonzero()[0][-1]
        #     front_break_lens.append(front_sum[best_front_break])
        #     brk_f_len = front_break_lens[-1] + line_len

        #     front_lines.append(words[f_idx:best_front_break + 1])
        #     f_idx = best_front_break + 1

        #     best_back_break = (back_sum <= brk_b_len).nonzero()[0][0]
        #     back_break_lens.append(back_sum[best_back_break])
        #     brk_b_len = back_break_lens[-1] + line_len

        #     print(best_back_break, back_break_lens)

        #     back_lines.append(words[best_back_break + 1:b_idx])
        #     b_idx = best_back_break + 1

        # front_break_lens.append(total_char)
        # back_break_lens.append(total_char)

        # front_lines.append(words[f_idx:])
        # back_lines.append(words[:b_idx])
        # back_lines = back_lines[::-1]

        # front_diff = np.diff(front_break_lens, prepend=0)
        # back_diff  = np.diff(back_break_lens, prepend=0)

        # front_short = np.sum((line_len - front_diff)**2)
        # back_short = np.sum((line_len - back_diff)**2)

        # if np.any(front_diff > line_len) and np.any(back_diff > line_len):
        #     if front_short < back_short:
        #         out_break = front_break_lens
        #     else:







        # start_line = True
        # for word in words:
        #     if "-" not in word:
        #         if start_line:
        #             front_count = len(word)
        #             front_lines = word
        #             start_line  = False
        #         elif (front_count + len(word)) <= line_len:
        #             front_count += len(word) + 1
        #             front_lines += " " + word
        #         else:
        #             front_count = len(word)
        #             front_lines += "\n" + word

        #     else:
        #         for idx, wrd in enumerate(word.split("-")):
        #             if start_line:
        #                 front_count = len(wrd)
        #                 front_lines = wrd
        #                 start_line  = False
        #             elif (front_count + len(wrd)) <= line_len:
        #                 front_count += len(wrd) + 1
        #                 if idx == 0:
        #                     front_lines += " " + wrd
        #                 else:
        #                     front_lines += "-" + wrd
        #             else:
        #                 front_count = len(wrd)
        #                 if idx == 0:
        #                     front_lines += "\n" + wrd
        #                 else:
        #                     front_lines += "-\n" + wrd

        # start_line = True
        # for word in words[::-1]:
        #     if "-" not in word:
        #         if start_line:
        #             back_count = len(word)
        #             back_lines = word
        #             start_line  = False
        #         elif (back_count + len(word)) <= line_len:
        #             back_count += len(word) + 1
        #             back_lines = word + " " + back_lines
        #         else:
        #             back_count = len(word)
        #             back_lines = word + "\n" + back_lines

        #     else:
        #         for idx, wrd in enumerate(word.split("-")[::-1]):
        #             if start_line:
        #                 back_count = len(wrd)
        #                 back_lines = wrd
        #                 start_line  = False
        #             elif (back_count + len(wrd)) <= line_len:
        #                 back_count += len(wrd) + 1
        #                 if idx == 0:
        #                     back_lines = wrd + " " + back_lines
        #                 else:
        #                     back_lines = wrd + "-" + back_lines
        #             else:
        #                 back_count = len(wrd)
        #                 if idx == 0:
        #                     back_lines = wrd + "\n" + back_lines
        #                 else:
        #                     back_lines = wrd + "-\n" + back_lines

        # print("Front:", front_lines)
        # print("Back:", back_lines)
        # out_list.append(front_lines)






        # wrapLine = ""
        # start = True

        # words = line.split(" ")

        # for word in line.split(" "):

        #     if len(word.split("-")) == 1:

        #         if start:
        #             wrapLine = word.capitalize()
        #             lineCount = len(word)
        #             start = False
        #         elif (lineCount + len(word)) <= line_len:
        #             lineCount += len(word) + 1
        #             wrapLine += " " + word.capitalize()
        #         else:
        #             lineCount = len(word)
        #             wrapLine += "\n" + word.capitalize()
        #             start = False

        #     else:
        #         for wrdNo, wrd in enumerate(word.split("-")):
        #             if start:
        #                 wrapLine = wrd.capitalize()
        #                 lineCount = len(wrd)
        #                 start = False
        #             elif wrdNo == 0:
        #                 if (lineCount + len(wrd)) <= line_len:
        #                     lineCount += len(wrd) + 1
        #                     wrapLine += " " + wrd.capitalize()
        #                 else:
        #                     lineCount = len(wrd)
        #                     wrapLine += "\n" + wrd.capitalize()
        #                     start = False
        #             elif (lineCount + len(wrd)) <= line_len:
        #                 lineCount += len(wrd) + 1
        #                 wrapLine += "-" + wrd.capitalize()
        #             else:
        #                 lineCount = len(wrd)
        #                 wrapLine += "-\n" + wrd.capitalize()
        #                 start = False

        # out_list.append(wrapLine)

    return out_list


def scatter_by_marker(gene): 
    ## THIS FUNCTION ASSUMES YOU'VE DONE SOMETHING LIKE RUN
    ## %run CellWise_Optimal_Embedding.py
    ## vargenes_X = pd.read_csv("./Data/TabulaMuris/FACS/tmp_Marrow_scaledata
    ##                           _v2.csv", index_col=0)
    ## BEFORE USING!
    ## It explicitly needs: vargenes_X, which is just scaledata, data_Y, which
    ## is an embedding of Marrow data, and db_labels, which is the cluster 
    ## labels for each cell in db_labels. Oh and unique_labels, which are the
    ## unique labels in db_labels.
    ##
    ## The threshold of 0.25 is based on what it seemed that Seurat does...

    if gene in vargenes_X.index.values:
        gene_levels = vargenes_X.loc[gene].values

    else:
        print(f"\nCouldn't find gene '{gene}'!")
        return None, None

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    sort_idx = np.argsort(gene_levels)

    hax = ax.scatter(*data_Y[sort_idx].T,
                     c=gene_levels[sort_idx],
                     s=3, alpha=0.7, cmap='rocket')

    cax = plt.colorbar(hax, ax=ax)

    for ll in unique_labels:
        if ll == -1:
            continue
        lab_idx = (ll == db_labels).nonzero()[0]
        frac_over_thresh = np.sum(gene_levels[lab_idx] > 0.25)
        frac_over_thresh /= len(lab_idx)
        ax.scatter([], [], color=sns.color_palette('tab20')[int(ll * 2)],
                   s=100, label=f"{ll}: {frac_over_thresh:.2%}")
    ax.set_title(gene)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend()
    fig.tight_layout()
    return fig, ax


###############################################################################
##  Functions for Loading or Finding Data
###############################################################################


def load_data(data_name,
              data_dir=data_dir,
              dtype=float,
              load_metadata=True,
              load_PCA=True):

    if data_name.lower() == 'mnist':
        data_path = path.join(data_dir, "mnist2500_X.txt")

        X = np.loadtxt(data_path)

        if load_metadata:
            metadata_path = path.join(data_dir, "mnist2500_labels.txt")
            metadata = np.loadtxt(metadata_path).astype(int)

    elif data_name.lower() in ["marrow", "diaphragm"]:

        if load_PCA:
            data_path = path.join(data_dir, f"{data_name}_PCA_Embeddings.csv")
            X = pd.read_csv(data_path).values[:, 1:].copy()

        else:
            data_path = path.join(data_dir, f"{data_name}_ScaleData.csv")
            X = pd.read_csv(data_path).values[:, 1:].copy().T

        if load_metadata:
            metadata_path = path.join(data_dir, f"{data_name}_Metadata.csv")
            metadata = pd.read_csv(metadata_path)

    elif data_name.lower() == 'mnist_42000':
        data_path = path.join(data_dir, "mnist_large.csv")

        X = pd.read_csv(data_path)

        if load_metadata:
            metadata = X['label'].values

        X = X.values[:, 1:].copy()

    else:
        X = pd.read_csv(path.join(data_dir, data_name)).values

    if not load_metadata:
        metadata = None

    return X.astype(dtype), metadata


def load_tSNE(X,
              name_base=None,
              embed_dir=embed_dir,
              n_embed=1,
              n_components=2,
              perplexity=30,
              early_exag_iter=250,
              n_iter=1000,
              initialization='random',
              n_jobs=-1,
              random_state=1,
              verbose=True):

    if name_base is not None:
        embed_name = name_base + f"_tSNE_{perplexity}_RS{random_state}.pkl"

    n_2_embed = n_embed

    try:
        if (name_base is None):
            raise ValueError

        if verbose:
            print(f"\nTrying to load {embed_name}")

        with open(path.join(embed_dir, embed_name), 'rb') as f:
            Y = pkl.load(f)

        n_embedded = len(Y)
        n_2_embed = n_embed - n_embedded

        if n_embedded < n_embed:
            if verbose:
                print(f"File loaded, but there aren't enough embeddings!"
                      f" ({n_embedded} < {n_embed})")
            raise ValueError

    except (FileNotFoundError, EOFError, ValueError):

        if (n_2_embed == n_embed) and verbose:
            print(f"File couldn't be loaded!")

        n_samples, n_features = X.shape

        perplexities = np.asarray(n_samples * [perplexity])
        affObj = FixedEntropyAffinity(perplexity=perplexities,
                                      normalization='pair-wise')
        affObj.fit(X)

        tmp_Y = np.zeros((n_2_embed, n_samples, n_components)).astype(float)

        for eNo in range(n_2_embed):
            if verbose:
                print(f"Generating embedding {eNo + 1} / {n_2_embed}")

            init_Y = initRand(X, random_state=random_state + eNo)

            single_Y = TSNEEmbedding(init_Y, affObj, verbose=True)

            single_Y.optimize(n_iter=early_exag_iter,
                              exaggeration=12,
                              momentum=0.5,
                              inplace=True)
            single_Y.optimize(n_iter=n_iter - early_exag_iter,
                              exaggeration=1.,
                              momentum=0.8,
                              inplace=True)

            # tSNE_Obj = TSNE(n_components=n_components,
            #                 perplexity=perplexity,
            #                 early_exaggeration_iter=early_exag_iter,
            #                 n_iter=n_iter,
            #                 initialization=initialization,
            #                 n_jobs=n_jobs,
            #                 random_state=random_state + eNo,
            #                 verbose=verbose)

            # tmp_Y[eNo] = tSNE_Obj.fit(X)[:]
            tmp_Y[eNo] = single_Y[:]

        if n_2_embed != n_embed:
            Y = np.vstack((Y.reshape(-1, n_samples, n_components), tmp_Y))
        else:
            Y = tmp_Y[:]

        if name_base is not None:
            if verbose:
                print(f"Saving {embed_name} to file!")

            with open(path.join(embed_dir, embed_name), 'wb') as f:
                pkl.dump(Y, f)

    return Y[:n_embed].astype(float).squeeze()


def load_UMAP(X,
              name_base=None,
              embed_dir=embed_dir,
              n_embed=1,
              n_components=2,
              n_neighbors=15,
              min_dist=0.1,
              initialization='random',
              n_jobs=-1,
              random_state=1,
              verbose=True):

    if name_base is not None:
        embed_name = name_base + f"_UMAP_{n_neighbors}_RS{random_state}.pkl"

    n_2_embed = n_embed

    try:
        if (name_base is None):
            raise ValueError

        if verbose:
            print(f"\nTrying to load {embed_name}")

        with open(path.join(embed_dir, embed_name), 'rb') as f:
            Y = pkl.load(f)

        n_embedded = len(Y)
        n_2_embed = n_embed - n_embedded

        if n_embedded < n_embed:
            if verbose:
                print(f"File loaded, but there aren't enough embeddings!"
                      f" ({n_embedded} < {n_embed})")
            raise ValueError

    except (FileNotFoundError, EOFError, ValueError):

        if (n_2_embed == n_embed) and verbose:
            print(f"File couldn't be loaded!")

        n_samples, n_features = X.shape

        tmp_Y = np.zeros((n_2_embed, n_samples, n_components)).astype(float)

        for eNo in range(n_2_embed):
            if verbose:
                print(f"Generating embedding {eNo + 1} / {n_2_embed}")

            UMAP_Obj = UMAP(n_components=n_components,
                            n_neighbors=n_neighbors,
                            min_dist=min_dist,
                            init=initialization,
                            n_jobs=n_jobs,
                            random_state=random_state + eNo,
                            verbose=verbose)

            tmp_Y[eNo] = UMAP_Obj.fit_transform(X)[:]

        if n_2_embed != n_embed:
            Y = np.vstack((Y.reshape(-1, n_samples, n_components), tmp_Y))
        else:
            Y = tmp_Y[:]

        if name_base is not None:
            if verbose:
                print(f"Saving {embed_name} to file!")

            with open(path.join(embed_dir, embed_name), 'wb') as f:
                pkl.dump(Y, f)

    return Y[:n_embed].astype(float).squeeze()


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


def calc_emp_pVals(alt_data, null_data, summary_method=None):

    nVals, nCDF = get_eCDF(null_data)

    ## If there's only one embedding, reshape into 2D array
    if alt_data.ndim == 1:
        alt_data = alt_data.reshape(1, -1)
    n_embed, n_samples = alt_data.shape

    ## Initialize the p-Value array
    pVal_arr = np.ones_like(alt_data)

    max_null = null_data.max()

    for eNo in range(n_embed):
        idx = (alt_data[eNo] <= max_null).nonzero()

        ## Get the cumulative probability of null < data
        pVal_arr[eNo][idx] = np.array([nCDF[np.searchsorted(nVals, data)]
                                       for data in alt_data[eNo][idx]])

    if summary_method is None:
        return pVal_arr

    elif summary_method.lower() == 'simes':
        simes_mult = n_embed / np.arange(1, n_embed + 1).reshape(-1, 1)
        pVals = np.sort(pVal_arr, axis=0)
        pVals = np.min(pVals * simes_mult, axis=0)

        return pVal_arr, pVals

    elif summary_method.lower() == 'average':
        return pVal_arr, np.mean(pVal_arr, axis=0)

    else:
        err_str = f"Unknown p-value summarizer, '{summary_method}'"
        raise ValueError(err_str)


###############################################################################
##  Functions for Loading or and Calculating the Effective Nearest Neighbors
###############################################################################

def is_iterable(x):

    try:
        _ = [_ for _ in x]
        return True
    except TypeError:
        return False

round_levels = {  5:   5,
                 20:  10,
                100:  50,
                500: 100}

def human_round(x, round_levels=round_levels, inplace=False, round_dir='none'):

    if not is_iterable(x):
        x = np.asarray([x])

    if not inplace:
        x = x.copy()

    levels = np.asarray(list(round_levels.keys()))
    values = np.asarray(list(round_levels.values()))
    for ii in range(len(levels)):
        lev, val = levels[ii], values[ii]
        if ii == (len(levels) - 1):
            next_lev = x.max()
        else:
            next_lev = levels[ii + 1]

        at_level = (x > lev) & (x <= next_lev)

        if round_dir.lower() == 'up':
            x[at_level] = (np.ceil(x[at_level] / val) * val).astype(int)
        elif round_dir.lower() == 'down':
            x[at_level] = (np.floor(x[at_level] / val) * val).astype(int)
        else:
            x[at_level] = (np.round(x[at_level] / val) * val).astype(int)

    return x


def generate_rounded_log_arr(n_els, n_max, round_levels=round_levels,
                             lower_bound=None):

    arr = np.logspace(0, np.log10(n_max), int(n_els)).astype(int)
    arr = np.unique(human_round(arr, round_levels=round_levels).astype(int))

    too_big = arr >= n_max
    if np.any(too_big):
        n_s_arr = (n_max - 1) * np.ones(too_big.sum())
        arr[too_big] = human_round(n_s_arr, round_levels={0: 10},
                                   round_dir='down')

    if lower_bound is not None:
        arr = arr[arr >= lower_bound]

    return np.sort(arr)


def generate_perp_arr(n_samples, n_perp=30, round_levels=round_levels,
                      lower_bound=None):

    max_arr = generate_rounded_log_arr(n_samples, n_samples,
                                       round_levels=round_levels,
                                       lower_bound=lower_bound)
    max_n_possible = len(max_arr)

    if n_perp >= max_n_possible:
        print(f"Requested number of perplexities ({n_perp}) is too large."
              f"  Can only generate {max_n_possible}.")
        n_perp = max_n_possible

    len_perp_arr = 0
    test_n_perp = n_perp

    prev_min_n = 0
    prev_max_n = np.inf

    counter = 0

    while len_perp_arr != n_perp:
        perp_arr = generate_rounded_log_arr(test_n_perp, n_samples,
                                            round_levels=round_levels,
                                            lower_bound=lower_bound)

        len_perp_arr = len(perp_arr)

        if len_perp_arr < n_perp:
            prev_min_n = test_n_perp
            if np.isinf(prev_max_n):
                test_n_perp *= 2
            else:
                test_n_perp = (prev_max_n + test_n_perp) / 2
        elif len_perp_arr > n_perp:
            prev_max_n = test_n_perp
            test_n_perp = (prev_min_n + test_n_perp) / 2

        if prev_max_n - prev_min_n < 1:
            break

        # print(len_perp_arr, n_perp, test_n_perp, prev_max_n, prev_min_n)

        counter += 1
        if counter >= 50:
            break

    return perp_arr


def calc_N_nonuniform(sorted_PWD,
                      perp_arr=None,
                      alpha_nu=0.02,
                      verbose=True):

    if verbose:
        print(f"Calculating median number of non-uniform neighbors!")

    n_samples = len(sorted_PWD)

    if perp_arr is None:
        perp_arr = generate_perp_arr(n_samples)
    perp_arr = np.sort(np.unique(perp_arr))

    n_nonunif_arr = np.zeros((len(perp_arr), n_samples)).astype(int)
    tau_arr = np.zeros((len(perp_arr), n_samples)).astype(float)

    if verbose:
        start = time.time()
    for pNo, perp in enumerate(perp_arr[::-1]):
        if verbose:
            print_str  = f"Evaluating perp = {perp} ({pNo + 1}/{len(perp_arr)}"
            print_str += f") {time.time() - start:.2f} seconds elapsed..."
            print(print_str)

        rev_pNo = len(perp_arr) - pNo - 1

        n_neibs = np.min([n_samples - 1, int(3 * perp)])

        tmp_PWD = np.sqrt(sorted_PWD[:, :n_neibs])

        perps = np.asarray(n_samples * [perp])
        tmp_P, tau_arr[rev_pNo] = calcAff(tmp_PWD, perplexities=perps)

        closest_P = alpha_nu * tmp_P[:, 0].reshape(-1, 1)
        n_nonunif_arr[rev_pNo] = np.sum(tmp_P > closest_P, axis=1)

        print(np.median(n_nonunif_arr[rev_pNo]))

    return n_nonunif_arr, tau_arr, perp_arr


def get_kEff(sorted_PWD=None,
             file_name=None,
             file_dir=data_dir,
             perp_arr=None,
             alpha_nu=0.01,
             verbose=True):

    if (sorted_PWD is None) and (file_name is None):
        err_str  = f"Need either a PWD or a filename to get the median"
        err_str += f" non-uniform samples!"
        raise ValueError(err_str)

    try:
        if file_name is None:
            raise FileNotFoundError
        file_path = os.path.join(file_dir, file_name)
        if verbose:
            print(f"Trying to load {file_name}!")

        with open(file_path, 'rb') as f:
            out_dict = pkl.load(f)

        n_nu_med = out_dict['N_nonunif']
        perp_arr = out_dict['perp_arr']

        if verbose:
            print("Everything loaded!")

    except(FileNotFoundError, KeyError, EOFError):

        if verbose:
            print(f"Couldn't be loaded, recalculating!")

        n_nu_arr, _, perp_arr = calc_N_nonuniform(sorted_PWD,
                                                  perp_arr=perp_arr,
                                                  alpha_nu=alpha_nu,
                                                  verbose=verbose)

        print(n_nu_arr)

        n_nu_med = np.median(n_nu_arr, axis=1).squeeze()

        if file_name is not None:
            with open(os.path.join(file_dir, file_name), 'wb') as f:
                pkl.dump({'N_nonunif': n_nu_med.copy(),
                          'perp_arr': perp_arr.copy()}, f)

    return n_nu_med, perp_arr


def interpolate(x_coords, y_coords, x_arr):

    x_out = np.zeros_like(x_arr).astype(float)
    for ii, x in enumerate(x_arr):
        if (x < x_coords[0]) or (x > x_coords[-1]):
            err_str = f"Value {x} is outside interpolation regime!"
            raise ValueError(err_str)

        if np.any(np.isclose(x, x_coords)):
            idx = np.where(np.isclose(x, x_coords))[0]
            if len(idx) > 1:
                idx = idx[0]
            x_out[ii] = (y_coords[idx]).squeeze()
            continue

        grid_id = np.searchsorted(x_coords, x, side='right')
        x_diff = (x_coords[grid_id] - x_coords[grid_id - 1])
        frac = (x - x_coords[grid_id - 1])
        slope = (y_coords[grid_id] - y_coords[grid_id - 1]) / x_diff
        x_out[ii] = (y_coords[grid_id - 1] + slope * frac).squeeze()

    return np.asarray(x_out, dtype=float).squeeze()


def get_kEff_from_perp(perp, kEff_arr, perp_arr):

    sort_idx = np.argsort(perp_arr)
    x_coords = perp_arr[sort_idx]
    y_coords = kEff_arr[sort_idx]

    try:
        _ = (p for p in perp)
    except TypeError:
        perp = [perp]

    return interpolate(x_coords, y_coords, np.asarray(perp)).squeeze()


def get_perp_from_kEff(nn, kEff_arr, perp_arr):

    sort_idx = np.argsort(perp_arr)
    y_coords = perp_arr[sort_idx]
    x_coords = kEff_arr[sort_idx]

    try:
        _ = (n for n in nn)
    except TypeError:
        nn = [nn]

    return interpolate(x_coords, y_coords, np.asarray(nn)).squeeze()
