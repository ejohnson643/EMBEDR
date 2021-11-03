###############################################################################
##  Functions for Loading or and Calculating the Effective Nearest Neighbors
###############################################################################

import numpy as np

## The thresholds at which to modify rounding in `human_round`
round_levels = {  5:   5,
                 20:  10,
                100:  50,
                500: 100}


def is_iterable(x):
    """Check if a variable is iterable"""

    try:
        _ = [_ for _ in x]
        return True
    except TypeError:
        return False


def unique_logspace(lb, ub, n, dtype=int):
    """Return array of up to `n` unique log-spaced ints from `lb` to `ub`."""
    lb, ub = np.log10(lb), np.log10(ub)
    return np.unique(np.logspace(lb, ub, n).astype(dtype))


def human_round(x,
                round_levels=round_levels,
                inplace=False,
                round_dir='none'):
    """Round an input like a human.

    This function takes a numeric input and rounds the values to human-scale.
    For example, rather than 17, we human_round to 15, and 327 is rounded to
    350.

    Arguments
    ---------
    x: number or numeric array
        The number(s) to be human-rounded.

    round_levels: dictionary (optional, default=`round_levels`)
        Dictionary where the `key` specifies a threshold above which we round
        to the nearest `value`.

    inplace: bool (optional, default=False)
        Flag indicating whether to modify `x` inplace or to generate a copy.

    round_dir: string (optional, default='none')
        Flag indicating whether to round up, down, or in either direction.
        Options are 'up', 'down', and 'none'.
    """

    x = np.asarray([x]).squeeze().astype(float)

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


def generate_rounded_log_arr(n_els, n_min, n_max, round_levels=round_levels):
    """Generate a human-rounded log-spaced array.

    Since log-spaced arrays of integers are often non-unique, this function
    wraps up the clean-up of these arrays.

    Arguments
    ---------
    n_els: integer
        The maximum number of elements in the array.  This is an upper bound on
        the array size because some elements may be eliminated when the array
        is reduced to its unique elements.

    n_min: integer
        The minimum value in the array.

    n_max: integer
        The maximum value in the array.

    round_levels: dictionary (optional)
        The levels and rounding amounts used by `human_round`.

    lower_bound: float (optional, default=None)
    """

    ## Generate a log-spaced array of integers.
    arr = np.logspace(np.log10(n_min), np.log10(n_max), int(n_els)).astype(int)
    ## Round the array, then reduce to unique elements.
    arr = np.unique(human_round(arr, round_levels=round_levels).astype(int))

    ## If rounding makes any elements too large, replace them with values
    ## slightly below the max value.
    too_big = arr >= n_max
    if np.any(too_big):
        n_s_arr = (n_max - 1) * np.ones(too_big.sum())
        arr[too_big] = human_round(n_s_arr, round_levels={0: 10},
                                   round_dir='down')

    return np.sort(arr)


def generate_perp_arr(n_samples, n_perp=30, round_levels=round_levels,
                      lower_bound=None):
    """Generate a human-rounded log-spaced array with exactly N elements.
    
    Because of the rounding and uniqueness issues, `generate_rounded_log_arr`
    doesn't always return elements of the desired length.  We can get around
    this by trying to generate arrays with different lengths, where sometimes
    the log-spacing and rounding will fall to give us an array of the desired
    length (`n_perp`).  There is an obvious upper bound, which is if we convert
    np.arange(n_samples), corresponding to all possible numbers being rounded.
    This upper bound is reported if it is below the array length requested.

    Parameters
    ----------
    n_samples: integer
        The number of samples in the data set.  For a perplexity array, this is
        the maximum value that the perplexity can have.

    n_perp: integer (optional, default=30)
        The desired number of elements in the perplexity array.

    round_levels: dictionary (optional)
        The levels and rounding amounts used by `human_round`.

    lower_bound: float (optional, default=None)
        The smallest perplexity allowed in the array.  By default this is set
        to 1.
    """

    max_arr = generate_rounded_log_arr(n_samples,
                                       lower_bound,
                                       n_samples,
                                       round_levels=round_levels)
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
        perp_arr = generate_rounded_log_arr(test_n_perp,
                                            lower_bound,
                                            n_samples,
                                            round_levels=round_levels)

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

        counter += 1
        if counter >= 50:
            break

    return perp_arr
