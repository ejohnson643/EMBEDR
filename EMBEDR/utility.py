from collections import Counter
import numpy as np
from time import time


def unique_logspace(lb, ub, n, dtype=int):
    """Return array of up to `n` unique log-spaced ints from `lb` to `ub`."""
    lb, ub = np.log10(lb), np.log10(ub)
    return np.unique(np.logspace(lb, ub, n).astype(dtype))


class Timer:
    """Wrapper for timing methods with a message."""

    def __init__(self, message, verbose=0):
        self.message = message
        self.start_time = time()
        self.verbose = verbose

    def __enter__(self):
        if self.verbose >= 0:
            print("--->", self.message)

    def __exit__(self):
        end = time()
        dt = end - self.start_time
        if self.verbose >= 0:
            print(f"---> Time Elapsed: {dt:.2g} seconds!")


def calculate_eCDF(data, extend=False):
    """Calculate the x- and y-coordinates of an empirical CDF curve.

    This function finds the unique values within a dataset, `data`, and
    calculates the likelihood that a random data point within the set is
    less than or equal to each of those unique values.  The `extend` option
    creates extra values outside the range of `data` corresponding to
    P(X <= x) = 0 and 1, which are useful for plotting eCDFs."""

    ## Get the unique values in `data` and their counts (the histogram).
    counts = Counter(data.ravel())
    ## Sort the unique values
    vals = np.msort(list(counts.keys()))
    ## Calculate the cumulative number of counts, then divide by the total.
    CDF = np.cumsum([counts[val] for val in vals])
    CDF = CDF / CDF[-1]

    ## If `extend`, add points to `vals` and `CDF`
    if extend:
        data_range = vals[-1] - vals[0]
        vals = [vals[0] - (0.01 * data_range)] + list(vals)
        vals = np.asarray(vals + [vals[-1] + (0.01 * data_range)])
        CDF = np.asarray([0] + list(CDF) + [1])

    return vals, CDF


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
