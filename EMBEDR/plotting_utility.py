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
        axis.set_xticklabels(xticklabels)

    if yticks is not None:
        axis.set_yticks(yticks)

    if yticklabels is not None:
        axis.set_yticklabels(yticklabels)

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
##  Functions for 3D Figures
###############################################################################

def generate_rotations(fig,
                       axis,
                       angles=range(360),
                       file_name="3DAnimation_",
                       elevation=15):

    files = []
    for ii, angle in enumerate(angles):
        if ii % 30 == 0:
            print(f"{ii}: Angle = {angle}")
        ax.view_init(elev = elevation, azim=angle)
        fname = f'{file_name}_{ii}.jpeg'
        fig.savefig(fname)
        files.append(fname)

    return files


def animate_gif(files, output, delay=3, repeat=True):
    """
    Uses imageMagick to produce an animated .gif from a list of
    picture files.
    """
    loop = -1 if repeat else 0
    file_list = " ".join(files)
    os.system(f'convert -delay {delay} -loop {loop} {file_list} {output}')


###############################################################################
##  Functions for Figure Aesthetics
###############################################################################


class CategoricalFadingCMap(object):
    """Make categorical colormap that fades between colors at specific values.

    This class creates a blended categorical-continuous colormap in which
    designated colors are faded to black or white in descrete regions.  As an
    example, this class is used to create the p-value colorbars in the EMBEDR
    plotting functions, where different levels of p-values are given different
    colors, but within each level, the color also fades as a p-value goes from
    one end of the category to the other.  This is useful for situations in
    which values have discrete bins into which they can be mapped, but we still
    want to see the individual variation in points.

    The default arguments are set to that used by EMBEDR for p-value colorbars.

    Parameters
    ----------
    change_points: Iterable (optional, default=[0, 2, 3, 4, 5])
        The values at which to change between categories.  The end-points (the
        max and min values to be shown on the colormap) must be supplied so
        that if 4 categories are desired, `change_points` must contain 4 + 1
        values.  These values are in units of the measurement being used to
        assign colors to points, i.e. if height is being used to color points,
        then change_points might be [1ft, 2ft, 4ft, 6ft], so that there are 3
        height categories: 1-2ft, 2-4ft, and 4-6ft.  All values outside this
        range will be mapped to the minimum and maximum color of the range
        (i.e. a 7ft person would have the same color as a 6ft person).

    base_cmap: Union[str, Iterable of tuples] (optional, default='colorblind')
        A categorical colormap (list of tuples) or the name of a Seaborn 
        colormap to which the intervals between `change_points` will be mapped.

    cmap_idx: Iterable (optional, default=[4, 0, 3, 2])
        A list of indices that maps the colors in the colormap to the correct
        interval. This allows for preset colormaps to be remapped by changing
        `cmap_idx` from [0, 1, 2, 3] to [2, 3, 1, 0], for example.

    cmap_dx: float (optional, default=0.001)
        Interval at which to interpolate colors.  Smaller will make the
        colormap seem more continuous, but may have trouble rendering on some
        computers.  If `cmap_dx` < 1, this will be interpreted as an interval
        size in the units of `change_points`.  If `cmap_dx` > 1, this will
        be interpreted as the number of interpolation intervals to calculate
        across `change_points`.

    cmap_kwds: dict (optional, default={})
        Other keywords to pass to `matplotlib.colors.ListedColormap` object.

    max_divergence: float (optional, default=0.75)
        Maximal distance between white/black and the category color to allow
        in each region.  Setting to 0 will keep the colors constant in each
        category, while setting to 1 will allow the colors to fade entirely to
        black or white.

    reverse_last_interval: bool (optional, default=True)
        Flag indicating whether to reverse the interpolation direction on the
        last interval.  This can be useful to set up a maximal contrast in one
        part of the colormap.

    fade_to_white: bool (optional, default=True)
        Flag indicating whether the colors should fade to white or black within
        a category.
    """

    def __init__(self,
                 change_points=[0, 2, 3, 4, 5],
                 base_cmap='colorblind',
                 cmap_idx=None,
                 cmap_dx=0.001,
                 cmap_kwds=None,
                 max_divergence=0.75,
                 reverse_last_interval=True,
                 fade_to_white=True):

        self.change_points  = change_points
        self.base_cmap      = base_cmap
        self.cmap_idx       = cmap_idx
        self.cmap_dx        = cmap_dx
        self.cmap_kwds      = cmap_kwds
        self.max_divergence = max_divergence

        ## Optional flags
        self.reverse_last_interval = reverse_last_interval
        self.fade_to_white         = fade_to_white

        self._validate_parameters()

        self.cmap, self.cnorm = self.make_cmap()

    def _validate_parameters(self):

        try:
            self.change_points = np.unique([el for el in self.change_points])
            self.change_points = np.sort(self.change_points).squeeze()
            self.n_categ = len(self.change_points) - 1
        except TypeError as te:
            err_str = "Input argument `change_points` is not iterable!"
            raise TypeError(err_str)

        if isinstance(self.base_cmap, str):
            self.base_cmap = sns.color_palette(self.base_cmap)
        else:
            try:
                _ = self.base_cmap[0]
            except TypeError as err:
                err_str  = err.args[0] + f"\n\n\t    Input `base_cmap` could"
                err_str += f" not be indexed (_ = cmap[0] failed).  Make sure"
                err_str += f" `base_cmap` is either a subscriptable colormap"
                err_str += f" or an iterable containing colors from which to"
                err_str += f" create the categorical colormap."
                raise TypeError(err_str)
        self._n_base_colors = len(self.base_cmap)

        if self.cmap_idx is None:
            self.cmap_idx = [4, 0, 3, 2] + list(range(4, self.n_categ))

        try:
            [el for el in self.cmap_idx]
            assert len(self.cmap_idx) == self.n_categ
        except TypeError as te:
            err_str = "Input argument `change_points` is not iterable!"
            raise TypeError(err_str)
        except AssertionError as ae:
            err_str  = f"Input size of `cmap_idx` does not map the number of"
            err_str += f" categories indicated by `change_points`"
            err_str += f" ({self.n_categ} != {len(self.cmap_idx)}). There must"
            err_str += f" be one index in `cmap_idx` for each category."
            raise ValueError(err_str)

        try:
            self.cmap_dx = float(self.cmap_dx)
            assert self.cmap_dx > 0
        except (AssertionError, ValueError) as err:
            err_str = f"Input argument `cmap_dx` must be a positive float."
            raise ValueError(err_str)

        if self.cmap_dx > 1:
            self.cmap_dx = ((self.change_points[-1] - self.change_points[0])
                            / self.cmap_dx)

        if self.cmap_kwds is None:
            self.cmap_kwds = {'name': "EMBEDR p-Values (-log10)"}
        err_str = f"Input argument `cmap_kwds` must be a dictionary!"
        assert isinstance(self.cmap_kwds, dict), err_str
        self.cmap_kwds = self.cmap_kwds.copy()

        try:
            self.max_divergence = float(self.max_divergence)
            assert 1 > self.max_divergence > 0
        except (AssertionError, ValueError) as err:
            err_str = f"Input argument `max_divergence` must be in [0, 1]."
            raise ValueError(err_str)

        self.reverse_last_interval = bool(self.reverse_last_interval)
        self.fade_to_white = bool(self.fade_to_white)

    def make_cmap(self):

        ## Get the base colors of the categories.
        self.base_colors = [self.base_cmap[idx] for idx in self.cmap_idx]

        ## Make an appropriate grid of points on which to set colors.
        color_grid = []
        for start, end in zip(self.change_points[:-1], self.change_points[1:]):
            color_grid += list(np.arange(start, end, self.cmap_dx))
        color_grid += [self.change_points[-1]]
        ## This checks that we didn't double up any grid points.
        color_grid = np.sort(np.unique(np.asarray(color_grid)).squeeze())

        ## Initialize the RGB+ array.
        final_colors = np.ones((len(color_grid), 4))

        ## Iterate through the category boundaries, setting interpolated colors
        ## for each category.
        cat_idx = 0
        for catNo, [start, end] in enumerate(zip(self.change_points[:-1],
                                                 self.change_points[1:])):
            ## Get the number of grid points in this interval
            n_ticks = int((end - start) / self.cmap_dx)
            ## If it's the last interval, add an extra.
            if end == self.change_points[-1]:
                n_ticks += 1

            ## Iterate through each of the RGB values.
            for jj in range(3):

                ## Base color for each interval
                base_color = self.base_colors[catNo][jj]

                ## Maximum divergence from the base color.
                top = 1 if self.fade_to_white else 0
                color_diff = self.max_divergence * (top - base_color)
                upper_bound = color_diff + base_color

                ## Interpolated grid for the interval.
                intv_color_grid = np.linspace(base_color, upper_bound, n_ticks)

                ## If we're in the last interval, can reverse the direction
                if catNo == (self.n_categ - 1):
                    if self.reverse_last_interval:
                        intv_color_grid = intv_color_grid[::-1]

                ## Set the colors!
                final_colors[cat_idx:cat_idx + n_ticks, jj] = intv_color_grid

            cat_idx += n_ticks

        ## Convert the grids and colors to matplotlib colormaps.
        cmap = mcl.ListedColormap(final_colors, **self.cmap_kwds)
        cmap.set_extremes(bad='lightgrey',
                          under=self.base_colors[0],
                          over=self.base_colors[-1])
        cnorm = mcl.BoundaryNorm(color_grid, cmap.N)

        return cmap, cnorm


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


def process_categorical_label(metadata, label, cmap='colorblind',
                              alphabetical_sort=False):

    ## Extract the raw labels
    raw_labels = metadata[label].values.copy()

    ## Get the unique labels and their counts
    label_counts = metadata[label].value_counts()
    unique_labels = label_counts.index.values

    if alphabetical_sort:
        str_labels = unique_labels.astype(str)
        unique_labels = unique_labels[np.argsort(str_labels)]
    label_counts = label_counts.reindex(unique_labels)

    ## Make some nice long labels.
    long_labels = np.asarray([f"{ll} (N = {label_counts.loc[ll]:d})"
                              for ll in unique_labels])

    ## Make a colormap
    if isinstance(cmap, str):
        label_cmap = sns.color_palette(cmap, len(unique_labels))
    else:
        label_cmap = cmap

    lab_2_idx_map = {ll: ii for ii, ll in enumerate(unique_labels)}

    return raw_labels, label_counts, long_labels, lab_2_idx_map, label_cmap

