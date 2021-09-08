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

