"""
###############################################################################
    Empirical Embedding Statistic
###############################################################################

    Author: Eric Johnson
    Date Created: Tuesday, August 31, 2021
    Email: ericjohnson1.2015@u.northwestern.edu

###############################################################################

    In this file, I want to store methods for calculating embedding quality
    statistics.  At this point, we only have the DKL method implemented, but
    going forward, we should support:
        DKL                 -> Needs AFFMATS
        Spearman’s Rho      -> Needs PWD / Ranked Mat
        (Kendall’s Tau)     -> Needs PWD / Ranked Mat
        L&V’s QNX           -> Needs PWD / Ranked Mat
        (T&C)               -> Needs PWD?? CORANKING MATRIX
        (LCMC)              -> Needs PWD?? CORANKING MATRIX


###############################################################################
"""
from numba import jit
import numpy as np

EPSILON = np.finfo(np.float64).eps


@jit(nopython=True)
def calculate_DKL(P_indices, P_indptr, P_data, Y, t_dof=1.):
    """Calculate the Kullback-Liebler Divergence between 2 affinity matrices.
    """

    n_embeds, n_samples, n_components = Y.shape

    DKL = np.zeros((n_embeds, n_samples))

    for eNo in range(n_embeds):

        for ii in range(n_samples):

            P_row = 0
            Q_row = 0
            for kk in range(P_indptr[ii], P_indptr[ii + 1]):
                jj = P_indices[kk]
                P_ij = P_data[kk] + EPSILON
                P_row = P_row + P_ij

                d_ij = 0
                for dd in range(n_components):
                    d_ij += (Y[eNo, ii, dd] - Y[eNo, jj, dd]) ** 2

                Q_ij = t_dof / (t_dof + d_ij)
                Q_row += Q_ij

                dkl_i = np.log(P_ij + EPSILON) - np.log(Q_ij + EPSILON)
                DKL[eNo, ii] += P_ij * dkl_i

            DKL[eNo, ii] += P_row * np.log(Q_row)

    return DKL

