"""
################################################################################
    Affinity Kernel Functions
################################################################################

    Author: Eric Johnson
    Date Created: Friday, July 2, 2021
    Email: ericjohnson1.2015@u.northwestern.edu

################################################################################

    In this file, we create numba implementations of several affinity kernel
    calculations for speed.

    TO-DO:
     - Affinity calculator should work with scipy sparse arrays.  So that
       `distances` doesn't need to have the same number of neighbors for each
       sample.

################################################################################
"""
from numba import jit
import numpy as np

EPSILON = np.finfo(np.float64).eps


@jit(nopython=True)
def fit_Gaussian_toEntropy(distances,
                           perplexities,
                           perp_tol=1.e-8,
                           max_iter=200,
                           tau_init=1,
                           tau_min=-1,
                           tau_max=-1):
    """Fit sample-wise Gaussian kernels to have a fixed entropy.

    Perform a binary search over the precision of a Gaussian kernel, tau, such
    that that kernel has an entropy = log(perplexity).  This fitting is done
    separately for each sample so that the size of the kernel for each affinity
    is different.

    Parameters
    ----------
    distances: np.ndarray (N x K)
        The SORTED pairwise euclidean distance matrix between a sample and its
        K nearest neighbors.  Sorting should be so that the distance to the
        nearest neighbors are in column 0 and the distance to the kth neighbor
        is in `distances[:, K-1]`.

    perplexities: np.ndarray (N)
        Array of perplexities - one perplexity per sample.  If you want to use
        one perplexity, provide an array of size N with the same value.

    perp_tol: float
        Tolerance to which the entropy should be fit.

    max_iter: int
        Maximum number of iterations to run the binary search before quitting.

    tau_init: float
        Initial guess for the kernel precision.  Currently this operates
        globally - each sample uses the same initialization.

    tau_min: float
        Minimum allowed precision.  See notes below for rationale behind
        defaults.

    tau_max: float
        Maximum allowed precision.  See notes below for rationale behind
        defaults.

    Returns:
    --------
    P: np.ndarray
        Affinity matrix (row-normalized, but unsymmetrized).

    tau_arr: np.ndarray
        Array of fitted precisions.

    """

    N, K = distances.shape

    if tau_min < 0:
        ## The kernel shouldn't be so wide that the furthest samples are in
        ## high probability regions, as this means that the samples are
        ## effectively uniformly weighted.  The 0.126 comes from the z-score
        ## needed to contain 10% of a normal distribution.
        tau_min_init = (0.126 ** 2) / (distances[:, -1].max() ** 2)
    else:
        tau_min_init = tau_min

    if tau_max < 0:
        ## The kernel shouldn't be so peaked that the nearest neighbors are
        ## outside of the meaningful part of the kernel.  The 2.33 comes from
        ## the z-score corresponding to a 98% probability region of a standard
        ## normal distribution.
        tau_max_init = (2.33 ** 2) / ((distances[:, 0].min() + EPSILON) ** 2)
    else:
        tau_max_init = tau_max

    P = np.zeros_like(distances)
    tau_arr = tau_init * np.ones(N)
    row_sums = np.ones(N)

    desired_entropy = np.log(perplexities)

    for nn in range(N):
        tau_min, tau_max = tau_min_init, tau_max_init

        tau = tau_arr[nn]

        for ii in range(max_iter):
            rowSum = 0
            rowPartSum = 0

            for kk in range(K):
                sqDij = distances[nn, kk]**2
                P[nn, kk] = np.sqrt(tau) * np.exp(-sqDij * tau / 2.)
                rowSum += P[nn, kk]

            rowSum += EPSILON

            for kk in range(K):
                sqDij = distances[nn, kk]**2
                rowPartSum += sqDij * P[nn, kk] / rowSum

            rowEnt = tau / 2 * rowPartSum + np.log(rowSum) - np.log(tau) / 2

            dEnt = rowEnt - desired_entropy[nn]

            if np.abs(dEnt) <= perp_tol:
                break

            if dEnt > 0:
                tau_min = tau
                if np.isinf(tau_max):
                    tau *= 2
                else:
                    tau = (tau + tau_max) / 2.
            else:
                tau_max = tau
                if np.isinf(tau_min):
                    tau /= 2
                else:
                    tau = (tau_min + tau) / 2.

        tau_arr[nn] = tau
        row_sums[nn] = rowSum

        for kk in range(K):
            P[nn, kk] = P[nn, kk] / rowSum

    return P, tau_arr, row_sums


@jit(nopython=True)
def GaussianAff_fromPrec(distances,
                         taus,
                         row_sums=None):
    """Calculate Gaussian affinities from input precisions.

    Parameters
    ----------
    distances: np.ndarray (N x K)
        The pairwise euclidean distance matrix between N samples and their K
        nearest neighbors.

    taus: np.ndarray(N)
        The precisions (width) of each sample's Gaussian kernel.  A precision
        must be given for each of the N samples in the dataset.

    row_sums: np.ndarray(N) (optional, default=None)
        The normalization factors needed to make each sample's affinities with
        their K nearest neighbors a probability distribution.  Providing this
        will speed up the calculation significantly!


    Returns:
    --------
    P: np.ndarray
        Affinity matrix (row-normalized, but unsymmetrized).
    """

    (N, K) = distances.shape

    P = np.zeros_like(distances)

    if row_sums is None:
        for nn in range(N):

            row_sum = 0

            for kk in range(K):
                sqDij = distances[nn, kk]**2
                P[nn, kk] = np.sqrt(taus[nn]) * np.exp(-sqDij * taus[nn] / 2.)
                row_sum += P[nn, kk]

            for kk in range(K):
                P[nn, kk] = P[nn, kk] / row_sum

    else:
        for nn in range(N):
            for kk in range(K):
                sqDij = distances[nn, kk]**2
                P[nn, kk] = np.sqrt(taus[nn]) * np.exp(-sqDij * taus[nn] / 2.)
                P[nn, kk] = P[nn, kk] / row_sums[nn]

    return P



