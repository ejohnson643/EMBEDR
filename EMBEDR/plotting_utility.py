"""
###############################################################################
    Plotting Utility
###############################################################################

    Author: Eric Johnson
    Date Created: Monday, March 8, 2021
    Date Edited: Wednesday, October 27 2021
    Email: ericjohnson1.2015@u.northwestern.edu

###############################################################################

    This file contains a collection of functions that facilitate the
    construction of figures for the EMBEDR manuscript.

###############################################################################
"""
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcl
import numpy as np
import os
from os import path
import pandas as pd
import pickle as pkl
import seaborn as sns
import time


###############################################################################
##  Functions for Figure Objects
###############################################################################


def save_figure(fig,
                fig_name,
                fig_dir="./",
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


def update_tight_bounds(fig,
                        inner_gs,
                        outer_gs,
                        rect=None,
                        inner_pad=1.08,
                        w_pad=0,
                        h_pad=0,
                        fig_pad=0):
    """Update the figure layout. Layout a gridspec inside another gridspec."""

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
##  Functions for Axes Objects
###############################################################################


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
    """Initialize axes with a border."""

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
    """Add a panel number to the corner of an axis object."""

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


###############################################################################
##  Functions for Colorbars
###############################################################################


def make_categ_cmap(change_points=[0, 2, 3, 4, 5],
                    categorical_cmap=None,
                    cmap_idx=None,
                    cmap_dx=0.001,
                    reverse_last_interval=True):
    """Make categorical colormap that fades between colors at specific values.

    This function takes in a list of end points + interior points to set as
    the edge of regions on a colormap.  The function then returns a new
    continuous colormap that transitions between these regions (fades to white,
    then changes colors).

    Parameters
    ----------
    change_points: Iterable (optional)
        The values at which to change between categories.  The end-points (the
        max and min values to be shown on the colormap) must be supplied so
        that if 4 categories are desired, `change_points` must contain 4 + 1
        values.

    categorical_cmap: Seaborn colormap object (optional)
        A categorical colormap (list of tuples) to which the intervals between
        `change_points` will be mapped.

    cmap_idx: Iterable (optional)
        A list of indices that maps the colors in the colormap to the correct
        interval. This allows for preset colormaps to be remapped by changing
        `cmap_idx` from [0, 1, 2, 3] to [2, 3, 1, 0], for example.

    cmap_dx: float (optional, default=0.001)
        Interval at which to interpolate colors.  Smaller will make the
        colormap seem more continuous, but may have trouble rendering on some
        computers.

    reverse_last_interval: bool (optional, default=True)
        Flag indicating whether to reverse the interpolation direction on the
        last interval.  This can be useful to set up a maximal contrast in one
        part of the colormap.
    """

    if categorical_cmap is None:
        import seaborn as sns
        categorical_cmap = sns.color_palette('colorblind')

    ## Set the list of indices to use from the colormap
    if cmap_idx is None:
        cmap_idx = [4, 0, 3, 2] + list(range(4, len(change_points) - 1))

    ## Set the base colors for regions of the colormap
    colors = [categorical_cmap[idx] for idx in cmap_idx]

    ## Make an appropriate grid of points on which to set colors.
    color_grid = []
    for intNo, end in enumerate(change_points[1:]):
        color_grid += list(np.arange(change_points[intNo], end, cmap_dx))
    color_grid += [change_points[-1]]
    color_grid = np.sort(np.unique(np.asarray(color_grid)).squeeze())

    ## Initialize the RGB+ array.
    out_colors = np.ones((len(color_grid), 4))

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
            upper_bound = 0.75 * (1 - base_color) + base_color

            ## Interpolated grid for the interval.
            intv_color_grid = np.linspace(base_color, upper_bound, N_ticks)

            ## If we're in the last interval, can reverse the direction
            if (intNo == (len(change_points) - 2)) and reverse_last_interval:
                intv_color_grid = intv_color_grid[::-1]

            ## Set the colors!
            out_colors[start_idx:start_idx + N_ticks, jj] = intv_color_grid

        start_idx += N_ticks

    ## Convert the grids and colors to matplotlib colormaps.
    import matplotlib.colors as mcl
    out_cmap = mcl.ListedColormap(out_colors)
    out_cnorm = mcl.BoundaryNorm(color_grid, out_cmap.N)

    return out_cmap, out_cnorm


def make_seq_cmap(color_1, color_2, n_colors=10):
    """Make a sequential colormap between two arbitrary colors."""

    out_colors = np.zeros((n_colors, 3))
    for ii in range(3):
        out_colors[:, ii] = np.linspace(color_1[ii], color_2[ii], n_colors)

    return out_colors


def cyclic_perm(a):
    n = len(a)
    b = [[a[i - j] for i in range(n)] for j in range(n)]
    return b


def wrap_strings(str_arr, line_len=26):
    """Automatically break strings into multiple lines."""

    out_list = []
    for line in str_arr:

        total_char = len(line)

        opt_breaks = int(total_char / line_len)

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

            out_lines.append(out_line)

        print(out_lines)

        out_lines = np.asarray(out_lines)
        line_lens = np.asarray([[len(word) for word in out_line.split('\n')]
                                for out_line in out_lines])

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

    return out_list

