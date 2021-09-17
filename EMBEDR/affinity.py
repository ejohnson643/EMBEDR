"""
###############################################################################
    Affinity Matrix Calculators
###############################################################################

    Author: Eric Johnson
    Date Created: Friday, July 2, 2021
    Email: ericjohnson1.2015@u.northwestern.edu

###############################################################################

    Adapted from the openTSNE package by Pavlin Poličar under the BSD 3-Clause
    License.

    In this file, the AffMat class is defined.  This class gives a structure to
    the process of calculating affinity matrices based on an input data matrix
    or kNN graph.

    For now, we're just going to focus on fixed-entropy Gaussian kernels. We're
    going to have these calculator classes take a kNN graph as input, because
    the previous version was a bit redundant with the nearest_neighbors module.
    As a result, we won't need all the kNN index building and metric checking
    functions.  This also means that when we implement locally specified
    numbers of nearest neighbors it should be completely contained in the
    `nearest_neighbors` module.

###############################################################################
"""

from EMBEDR._affinity import fit_Gaussian_toEntropy, GaussianAff_fromPrec
import EMBEDR.nearest_neighbors as nn
import EMBEDR.utility as utl

import numpy as np
import scipy.sparse as sp
from sklearn.neighbors import NearestNeighbors as NNB
from sklearn.utils import check_array, check_random_state

EPSILON = np.finfo(np.float64).eps


class AffinityMatrix(object):
    """Calculate an 'affinity' matrix based on distances between samples

    It is often useful to generate an "affinity" matrix, :math:`P`, which
    contains measures of similarity between samples in a data set.  In the case
    of several dimensionality reduction algorithms, like t-SNE and UMAP, this
    affinity matrix is actually the *input*, not the raw data.

    Parameters
    ----------
    kernel_params: dict (optional, default={})
        Additional keyword arguments for the kernel function that is used to
        calculate the affinity between samples in the input data set.

    symmetrize: bool (optional, default=True)
        Flag indicating whether to symmetrize the affinity matrix or not.

    normalization: str (optional, default="local")
        String indicating which type of normalization to apply to affinity
        matrices, if needed.  Options are "local", which will normalize each
        sample's affinity distribution (row-normalize), "global", which will
        make it so that sum(P) = 1, or None, which will not do any normalizing.

    n_jobs: int (optional, default=1)
        Number of processors to use when calculating the affinities.  This
        follows the scikit-learn convention where ``-1`` means to use all
        processors, ``-2`` means to use all but one, etc.

    random_state: Union[int, RandomState] (optional, default=None)
        State for the random number generator.  If `random_state` is an
        integer, then the input is interpreted as a seed for the RNG.  A
        RandomState instance can also be provided, which will then be used as
        the RNG.  Default is `None`, which uses the `RandomState` provided by
        np.random.

    verbose: numeric (optional, default=0)
        Flag indicating the level of verbosity to use in the outputs.  Setting
        to -1 (or smaller) will suppress all output.
    """

    def __init__(self,
                 kernel_params={},
                 symmetrize=True,
                 normalization='local',
                 precomputed=False,
                 kNN_params={},
                 n_jobs=1,
                 random_state=1,
                 verbose=1):

        self.kernel_params = kernel_params

        self.symmetrize = bool(symmetrize)

        if isinstance(normalization, str):
            self.normalization = normalization.lower()
        elif normalization is None:
            self.normalization = normalization
        else:
            raise TypeError(f"Invalid type {type(normalization)} for keyword"
                            f" argument `normalization`.")
        assert self.normalization in ['local', 'global', None]

        self.precomputed = bool(precomputed)

        default_kNN_params = {'n_neighbors': None}
        default_kNN_params.update(kNN_params)
        self.kNN_params = default_kNN_params.copy()
        self.n_neighbors = self.kNN_params['n_neighbors']
        del self.kNN_params['n_neighbors']

        ## Generic runtime parameters
        self.n_jobs = int(n_jobs)
        self.random_state = check_random_state(random_state)
        self.verbose = verbose

        return

    def kernel(self, kNNIndex, **kwargs):
        return

    def calculate_affinities(self, kNNIndex, recalc=False):

        ## Calculate raw (unsymmetrized, unnormed) affinity matrix
        if recalc:
            P = self._recalculate_P(kNNIndex)
        else:
            P = self.kernel(kNNIndex)

        ## Convert to a sparse matrix by default
        if not sp.issparse(P):
            P = sp.csr_matrix(
                (P.ravel(),
                 kNNIndex.kNN_idx[:, :self.n_neighbors].ravel(),
                 np.arange(0, self.n_samples * self.n_neighbors + 1,
                           self.n_neighbors)),
                shape=(self.n_samples, self.n_samples))

        ## If desired, symmetrize the affinity matrix
        if self.symmetrize:
            P = (P + P.T) / 2

        ## Normalize the affinity matrix
        if self.normalization == "global":
            P /= np.sum(P)
        elif self.normalization == "local":
            P = sp.diags(np.asarray(1 / P.sum(axis=1)).ravel()) @ P

        return P

    def _convert_to_kNN(self, X):
        err_str =  f"If `precomputed`=True, then input `X` must be an NxN"
        err_str += f" distance matrix.  X.shape = {X.shape}"
        assert X.shape[1] == self.n_samples, err_str

        tmp_params = self.kNN_params.copy()
        tmp_params['NN_alg':'balltree']
        kNNIndex = nn._initialize_kNN_index(X, **tmp_params)

        tmp_nn = NNB(metric='precomputed', n_jobs=self.n_jobs)
        tmp_nn.fit(X)

        kNN_dst, kNN_idx = tmp_nn.kneighbors(X, self.n_neighbors + 1)
        kNNIndex.kNN_idx = kNN_idx[:, 1:]
        kNNIndex.kNN_dst = kNN_dst[:, 1:]

        return kNNIndex

    def _check_X(self, X):
        return

    def _recalculate_P(self, kNNIndex):
        return

    def fit(self, X):

        X = self._check_X(X)

        self.P = self.calculate_affinities(X)

        return self.P


class FixedEntropyAffinity(AffinityMatrix):

    def __init__(self,
                 kernel_params={},
                 symmetrize=True,
                 normalization='local',
                 precomputed=False,
                 kNN_params={},
                 n_jobs=1,
                 random_state=1,
                 verbose=1,
                 perplexity=None):

        super().__init__(kernel_params, symmetrize, normalization, precomputed,
                         kNN_params, n_jobs, random_state, verbose)

        def_kernel_params = dict(perp_tol=1.e-8, max_iter=200, tau_init=1,
                                 tau_min=-1, tau_max=-1)
        self.kernel_params.update(def_kernel_params)

        self.perplexity = perplexity

    def _check_perplexity(self, perplexity):
        if perplexity is None:
            perplexity = np.clip(self.n_samples / 10., 1, self.n_samples)

        if np.isscalar(perplexity):
            perplexity = float(perplexity)
            if (perplexity <= 1) or (perplexity >= self.n_samples):
                err_str = f"Perplexity must be between 1 and {self.n_samples}!"
                raise ValueError(err_str)

            perp_arr = np.ones((self.n_samples)) * perplexity

        else:
            perp_arr = np.array(perplexity).astype(float).squeeze()

            if perp_arr.ndim == 0:
                perp_arr = np.ones((self.n_samples)) * perplexity

            elif perp_arr.ndim == 1:
                err_str =  f"Perplexity array must have length = len(data) ="
                err_str += f" {self.n_samples}."
                assert len(perp_arr) == self.n_samples, err_str

            else:
                err_str =  f"Perplexity must be either a scalar or a 1D array."
                raise ValueError(err_str)

            perp_arr = np.clip(perp_arr, 1, self.n_samples)

        if not hasattr(self, '_perp_arr'):
            self._perp_arr = perp_arr[:]
        return self._perp_arr.max()

    def _check_X(self, X):
        if isinstance(X, nn.kNNIndex):
            if self.verbose >= 3:
                print(f"A kNN graph has been input!")

            ## Check that X is a valid array.
            self.n_samples, n_neighbors = X.kNN_idx.shape

            ## Check perplexity
            self.perplexity = self._check_perplexity(self.perplexity)

            ## Set the number of nearest neighbors
            if self.n_neighbors is None:
                self.n_neighbors = int(np.clip(3 * self.perplexity,
                                               1, self.n_samples - 1))
            else:
                self.n_neighbors = np.clip(n_neighbors, 1, self.n_samples - 1)

        elif self.precomputed:
            if self.verbose >= 2:
                print(f"A precomputed distance matrix has been input!"
                      f" Converting to a kNN graph...")

            ## Check that X is a valid array.
            X = check_array(X, accept_sparse=True, ensure_2d=True)
            self.n_samples = X.shape[0]

            ## Check perplexity
            self.perplexity = self._check_perplexity(self.perplexity)

            ## Set the number of nearest neighbors
            if self.n_neighbors is None:
                self.n_neighbors = self.n_samples - 1
            else:
                self.n_neighbors = np.clip(self.n_neighbors, 1,
                                           self.n_samples - 1)

            ## Convert X to a kNN_index.
            X = self._convert_to_kNN(X)

        else:
            if self.verbose >= 3:
                print(f"Data matrix has been input!  Computing kNN graph...")

            X = check_array(X, accept_sparse=True, ensure_2d=True)
            self.n_samples = X.shape[0]

            ## Check perplexity
            self.perplexity = self._check_perplexity(self.perplexity)

            ## Set the number of nearest neighbors
            if self.n_neighbors is None:
                self.n_neighbors = int(np.clip(3 * self.perplexity,
                                               1, self.n_samples - 1))
            else:
                self.n_neighbors = np.clip(self.n_neighbors, 1,
                                           self.n_samples - 1)

            kNNIndex = nn._initialize_kNN_index(X, **self.kNN_params)
            kNNIndex.fit(X, self.n_neighbors)
            X = kNNIndex

        return X

    def kernel(self, kNNIndex):

        timer_str = f"Calculating fixed-entropy Gaussian affinity matrix!"
        timer = utl.Timer(timer_str, verbose=self.verbose)
        timer.__enter__()

        tmp_kernel_params = self.kernel_params.copy()
        good_keys = ['perp_tol', 'max_iter', 'tau_init', 'tau_min', 'tau_max']
        tmp_kernel_params = {k: tmp_kernel_params[k] for k in good_keys}

        k_NN = self.n_neighbors
        P, taus, rows = fit_Gaussian_toEntropy(kNNIndex.kNN_dst[:, :k_NN],
                                               self._perp_arr,
                                               **tmp_kernel_params)
        timer.__exit__()

        self.kernel_params['precisions'] = taus.copy()
        self.kernel_params['row_sums'] = rows.copy()

        return P

    def _recalculate_P(self, kNNIndex):

        if 'precisions' not in self.kernel_params:
            err_str =  f"Affinity matrix must have been fit before we can"
            err_str += f" recalculate from parameters."
            raise AttributeError(err_str)

        taus = self.kernel_params['precisions']

        row_sums = None
        if 'row_sums' in self.kernel_params:
            row_sums = self.kernel_params['row_sums']

        k_NN = self.n_neighbors
        return GaussianAff_fromPrec(kNNIndex.kNN_dst[:, :k_NN],
                                    taus=taus,
                                    row_sums=row_sums)


valid_kernels = {'fixed_entropy_gauss': FixedEntropyAffinity}


def _initialize_affinity_matrix(X,
                                aff_type='fixed_entropy_gauss',
                                perplexity=None,
                                n_neighbors=None,
                                kernel_params={},
                                symmetrize=True,
                                normalization='local',
                                precomputed=False,
                                kNN_params={},
                                n_jobs=1,
                                random_state=None,
                                verbose=0,
                                **kwargs):

    if aff_type not in valid_kernels:
        err_str  = f"Unknown type of affinity matrix '{aff_type}'."
        err_str += f" Allowed types are: "
        err_str += f", ".join([f"'{k}'" for k in valid_kernels]) + f"."
        raise ValueError(err_str)

    aff_class = valid_kernels[aff_type]

    tmp_kNN_params = {'n_neighbors': n_neighbors}
    tmp_kNN_params.update(kNN_params)

    aff_obj = aff_class(kernel_params=kernel_params,
                        symmetrize=symmetrize,
                        normalization=normalization,
                        precomputed=precomputed,
                        kNN_params=tmp_kNN_params,
                        n_jobs=n_jobs,
                        random_state=random_state,
                        verbose=verbose,
                        perplexity=perplexity,
                        **kwargs)

    kNN_index = aff_obj._check_X(X)

    return aff_obj, kNN_index





