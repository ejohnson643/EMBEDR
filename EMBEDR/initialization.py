"""
###############################################################################
    Embedding Initialization Schemes
###############################################################################

    Author: Eric Johnson
    Date Created: Thursday, July 15, 2021
    Email: ericjohnson1.2015@u.northwestern.edu

###############################################################################

    In this file, I want to include some quick initialization schemes for
    setting up dimensionally-reduced embeddings.  Specifically, we'll have a
    random initialization, a PCA initialization, and a spectral initialization.

    Some parts of this code are adapted from the openTSNE package by Pavlin
    Poličar under the BSD 3-Clause License.

###############################################################################
"""

import EMBEDR.utility as utl

import scipy.sparse as sp
from sklearn.decomposition import PCA
from sklearn.manifold import MDS
from sklearn.utils import check_random_state, check_array
import numpy as np

_init_scaling=10000.

def rescale(X, scaling=_init_scaling):
    return X / (np.std(X, axis=0) * scaling)


def random_init(X,
                n_components=2,
                scaling=_init_scaling,
                random_state=None,
                verbose=1):

    if verbose >= 2:
        print(f"Generating random initial embedding with same size as X.")

    ## Set the seed, if supplied.
    rs = check_random_state(random_state)

    ## Get the size of the data.
    n_samples = X.shape[0]

    ## Generate a randomized set of points of the correct size.
    Y = rs.normal(0, 1, (n_samples, n_components))

    return rescale(Y, scaling=scaling)


def pca_init(X,
             n_components=2,
             scaling=_init_scaling,
             random_state=None,
             verbose=1):

    if verbose >= 2:
        timer = utl.Timer("Generating PCA initialization!")
        timer.__enter__()

    pca = PCA(n_components=n_components,
              random_state=random_state)
    Y = pca.fit_transform(X)

    if verbose >= 2:
        timer.__exit()

    return rescale(Y, scaling=scaling)


def spectral_init(P,
                  n_components=2,
                  scaling=_init_scaling,
                  tol=1.e-4,
                  max_iter=None,
                  random_state=None,
                  verbose=1):
    """Generate an initial embedding based on spectral decomp of the kNN graph.

    As noted in openTSNE, this method treats `P` as a hopping probability map
    and then diffuses samples based on their affinities.

    """

    if verbose >= 2:
        timer = utl.Timer("Generating spectral initialization!")
        timer.__enter__()

    P = check_array(P, accept_sparse=True, ensure_2d=True)
    if P.shape[0] != P.shape[1]:
        err_str = f"The graph adjacency matrix (affinity matrix, `P`) must be"
        raise ValueError(err_str + f" square!")

    ## Get the row sums as a diagonal matrix
    row_sums = sp.diags(np.ravel(np.sum(P, axis=1)))

    ## Get the leading eigenvectors
    v0 = np.ones(P.shape[0]) / np.sqrt(P.shape[0])
    evals, evecs = sp.linalg.eigsh(P,
                                   M=row_sums,
                                   k=n_components + 1,
                                   tol=tol,
                                   maxiter=max_iter,
                                   which='LM',
                                   v0=v0)

    ## Make sure the eigenvalues are sorted
    sort_idx = np.argsort(evals)[::-1]
    evecs = evecs[:, sort_idx]

    ## Multiply the eigenvectors by their eigenvalues
    evecs *= evals[sort_idx]

    ## Drop the leading eigenvector
    embedding = evecs[:, 1:]

    if verbose >= 2:
        timer.__exit()

    return rescale(embedding, scaling=scaling)

