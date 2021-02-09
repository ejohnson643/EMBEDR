"""
###############################################################################
   EMBEDR Utilities
###############################################################################

    Author: Eric Johnson
    Date Created: February 4, 2021
    Email: ericjohnson1.2015@u.northwestern.edu

###############################################################################

    This file contains useful utility functions for the EMBEDR module.

###############################################################################
"""
from collections import Counter
import matplotlib.colors as mcl
import numpy as np
import numpy.random as r
import os
import seaborn as sns


def generate_nulls(X, seed=None, n_null=1, seed_mult=1):

    N, K = X.shape

    null_X = np.zeros((n_null, N, K))

    for nn in range(n_null):

        if seed is not None:
            r.seed(seed + nn * seed_mult)

            null_X[nn] = np.asarray([r.choice(col, size=N)
                                     for col in X.T]).T.copy()

    return null_X


def calculate_eCDF(X, extend=False, pad=0.02):

    counts = Counter(X.ravel())
    vals = np.array(np.msort(list(counts.keys()))).squeeze()
    CDF = np.cumsum([counts[val] for val in vals])

    CDF = CDF / CDF[-1]

    if extend:
        pad = pad * (vals[-1] - vals[0])

        vals = np.array([vals[0] - pad] + list(vals) + [vals[-1] + pad])
        CDF = np.array([0] + list(CDF.ravel()) + [1])

    return vals, CDF


def calculate_EMBEDR_pValues(data_EES, null_EES):

    ## Get the null EES distribution
    nVals, nCDF = calculate_eCDF(null_EES)

    ## Copy the array and reshape if needed
    data_EES = data_EES.copy().squeeze()
    if data_EES.ndim == 1:
        data_EES = data_EES.reshape(1, -1)

    n_data_embeds, n_samples = data_EES.shape

    ## Initialize the p-Values
    pValues = np.ones_like(data_EES)

    for eNo in range(n_data_embeds):

        ## Get EES values that are smaller than the largest null EES.
        ## (Everything to the right of the null distribution will be 1!)
        good_idx = (data_EES[eNo] <= nVals.max()).nonzero()

        ## Get the cumulative probability of nEES < dEES
        pValues[eNo][good_idx] = np.array([nCDF[np.searchsorted(nVals, EES)]
                                           for EES in data_EES[eNo][good_idx]])

    return pValues


def calculate_Simes_pValues(pValues):

    ## Copy the array and check the shape
    pValues = pValues.copy().squeeze()
    if pValues.ndim == 1:
        return pValues

    n_data_embeds, n_samples = pValues.shape

    simes_mult = n_data_embeds / (np.arange(n_data_embeds).reshape(-1, 1) + 1)

    return np.min(np.sort(pValues, axis=0) * simes_mult, axis=0)


def make_pVal_cmap(change_points=None,
                   categ_cmap=None,
                   cmap_idx=None,
                   cmap_dx=0.001,
                   reverse_last_interval=True):
    """Make a categorical colormap that fades between colors at specific p-Vals

    This function takes in a list of end points + interior points to set as
    the edge of regions on a colormap.  The function then returns a new
    continuous colormap that transitions between these regions (fades to white,
    then changes colors).
    """

    ## Set the points at which the p-value color will change.
    if change_points is None:
        change_points = [0, 2, 3, 4, 5]  ## -log10 of p-Values.

    ## Set the categorical colormap
    if categ_cmap is None:
        categ_cmap = sns.color_palette('colorblind')

    ## Set the list of indices to use from the colormap
    if cmap_idx is None:
        cmap_idx = [4, 0, 3, 2] + list(range(4, len(change_points) - 1))

    ## Set the base colors for regions of the colormap
    colors = [categ_cmap[idx] for idx in cmap_idx]

    ## Make an appropriate grid of points on which to set colors.
    color_grid = []
    for intNo, end in enumerate(change_points[1:]):
        color_grid += list(np.arange(change_points[intNo], end, cmap_dx))
    color_grid += [change_points[-1]]
    color_grid = np.sort(np.unique(np.asarray(color_grid)).squeeze())

    ## Initialize the RGB+ array.
    pVal_colors = np.ones((len(color_grid), 4))

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
            pVal_colors[start_idx:start_idx + N_ticks, jj] = intv_color_grid

        start_idx += N_ticks

    ## Convert the grids and colors to matplotlib colormaps.
    pVal_cmap = mcl.ListedColormap(pVal_colors)
    pVal_cnorm = mcl.BoundaryNorm(color_grid, pVal_cmap.N)

    return pVal_cmap, pVal_cnorm


def save_figure(fig,
                fig_name,
                fig_dir="./Figures",
                formats=['pdf', 'png', 'svg'],
                do_tight_layout=True,
                bbox_inches=None,
                transparent=True,
                dpi=300):

    if do_tight_layout:
        fig.tight_layout()

    fig_path = os.path.join(fig_dir, fig_name)

    for fmt in formats:
        fig.savefig(fig_path + "." + fmt,
                    format=fmt,
                    bbox_inches=bbox_inches,
                    transparent=transparent,
                    dpi=dpi)
