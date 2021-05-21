"""
###############################################################################
   EMBEDR: Statistical Quality Assessment of Dimensionality Reduction
###############################################################################

    Author: Eric Johnson
    Date Created: February 4, 2021
    Email: ericjohnson1.2015@u.northwestern.edu

###############################################################################

    EMBEDR (Empirical Marginal resampling Better Evaluates Dimensionality
    Reduction) is an analysis procedure that can be used to statistically
    assess the quality with which a high-dimensional sample is embedded in a
    low-dimensional space by a dimensionality reduction algorithm.  The code
    provided here is a sort of a quickstart to use EMBEDR, but a more efficient
    implementation is currently being developed.

###############################################################################
"""
import matplotlib.pyplot as plt
import numpy as np

from openTSNE import TSNEEmbedding
from openTSNE.affinity import PerplexityBasedNN
from openTSNE.initialization import random as initRand

import os
import pickle as pkl
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances as pwd
from sklearn.utils import check_random_state
import time

import utility as utl

EPSILON = np.finfo(np.float64).eps


class EMBEDR:
    """Implement the EMBEDR algorithm on a DRA

    Parameters
    ----------
    n_components: int
        Dimensionality of the embedding space.  Default is 2.

    perplexity: float
        Similar to the perplexity parameter from van der Maaten (2008); sets
        the scale of the affinity kernel used to measure embedding quality.
        NOTE: In the EMBEDR algorithm, this parameter is used EVEN WHEN NOT
        USING t-SNE!  Default is 30.

    dimred_alg: str
        Dimensionality reduction algorithm to use.  Currently only t-SNE, UMAP,
        and PCA are accepted.

    dimred_params: dict
        Parameters to pass to the dimensionality reduction algorithms.  For
        t-SNE, the fields 'n_iter', 'exaggeration', 'early_exag_iter',
        'exag_mom', and 'momentum' are directly used and other parameters that
        can be sent to openTSNE's `TSNEEmbedding` class should be set as a
        dictionary under the key 'openTSNE_params'.  For UMAP and PCA, the
        dictionary is fed directly to the UMAP and PCA classes, respectively.
        (This is to improve the efficiency of t-SNE by reducing re-calculations
        of certain parameters.)  To see the default structure of these
        parameters, run the quickstart example and examine
        embedr_obj.tSNE_params.

    n_data_embed: int
        The number of times to embed the data.  Default is 1.  This parameter
        has no effect for deterministic DRAs (PCA).

    n_null_embed: int
        The number of null data sets to generate and embed.  Default is 1.
        See Johnson, Kath, and Mani (2020) for recommendations on setting this
        parameter.

    random_state: Union[int, RandomState]
        If the value is an integer, then the input `random_state` is used as a
        seed to create a RandomState instance. If the input value is a
        `RandomState` instance, then it will be used as the RNG. If the input
        value is None, then the RNG is the `RandomState` instance provided by
        `np.random`.

    cache_results: bool
        A flag indicating whether to store calculated embeddings and affinity
        matrices for repeated use.  Default is True.  If set to False, the
        EMBEDR object will always calculate affinity matrices, embeddings, and
        p-values from scratch.

    project_dir: str
        Path to folder for caching results.  Not used if `cache_results` set to
        `False`.  Default is "./Embeddings".

    project_name: str
        Name of project.  Not used if `cache_results` is set to `False`.
        Default is "default_project".

    n_jobs: int
        Number of threads to use when finding nearest neighbors. This follows 
        the scikit-learn convention: ``-1`` means to use all processors, ``-2``
        indicates that all but one processor should be used, etc.

    verbose: int
        Integer flag indicating level of verbosity to use in output. Setting to
        -1 will suppress all output.
    """

    valid_DRAs = ['t-sne', 'tsne', 'umap', 'pca']

    def __init__(self,
                 n_components=2,
                 perplexity=30,
                 dimred_alg='t-SNE',
                 dimred_params={},
                 n_data_embed=1,
                 n_null_embed=1,
                 random_state=None,
                 cache_results=True,
                 project_dir='./Embeddings',
                 project_name="default_project",
                 n_jobs=-1,
                 verbose=0):

        self.verbose = float(verbose)

        if self.verbose > 0:
            print(f"\n\nInitializing EMBEDR Object!\n\n")

        if int(n_components) > 0:
            self.n_components = int(n_components)
        else:
            err_str  = f"Dimensionality of the embedding space must be a"
            err_str += f" positive integer. (n_components > 0)"
            raise ValueError(err_str)

        if float(perplexity) > 0:
            self.perplexity = float(perplexity)
        else:
            err_str  = f"Dimensionality of the embedding space must be a"
            err_str += f" positive integer. (n_components > 0)"
            raise ValueError(err_str)

        if dimred_alg.lower() in self.valid_DRAs:
            self.DRA = dimred_alg.lower()
        else:
            err_str  = f"Unknown dimensionality reduction algorithm:"
            err_str += f" {dimred_alg}.  Accepted algorithms: t-SNE, UMAP, PCA"
            raise ValueError(err_str)

        self.DRA_params = dimred_params

        if int(n_data_embed) > 0:
            self.n_data_embed = n_data_embed
        else:
            err_str  = f"Number of times to embed the data must be > 0!"
            raise ValueError(err_str)

        if int(n_null_embed) > 0:
            self.n_null_embed = n_null_embed
        else:
            err_str  = f"Number of times to embed the null must be > 0!"
            raise ValueError(err_str)

        self.random_state = check_random_state(random_state)

        self.project_name = os.path.join(project_dir, project_name)

        self.n_jobs = n_jobs
        self.do_cache = cache_results

    def fit(self, X):
        """Embeds data `X` in lower-dimensional space using indicated DRA.

        This method also automatically generates the required null embeddings
        and quality statistics in order to assign each sample in `X` a
        "p-value" indicating the likelihood that a random sample would be
        embedded that well.

        Attributes
        ----------
        n_samples: int
            Number of samples in supplied data `X`.

        n_features: int
            Number of features in supplied data `X`.

        data_Y: (n_data_embed x n_samples x n_components) array
            Data `X` embedded `n_data_embed` times by `dimred_alg`.

        null_Y: (n_null_embed x n_samples x n_components) array
            Null data embedded `n_null_embed` times by `dimred_alg`.  To
            recover the high-dimensional null data, use
            `utility.generate_nulls(X)`.

        EES_data: (n_data_embed x n_samples) array
            Quality statistics for each sample in the original data set
            (for each) embedding.

        EES_null: (n_null_embed x n_samples) array
            Quality statistics for each sample in each null data set.

        pVals: (n_data_embed x n_samples) array
            EMBEDR p-values. Summarized across `N_data_embed` embeddings using
            Simes' method.

        Example
        -------
        > from embedr import EMBEDR
        > import numpy as np
        > X = np.loadtxt("./Data/mnist2500_X.txt")
        >
        > embedr_obj = EMBEDR(verbose=1)
        > embedr_obj.fit(X)

        EMBEDR: Fitting 2500 x 784 data!

        ===> Finding 90 nearest neighbors using Annoy approximate search using
        euclidean distance...
           --> Time elapsed: 1.08 seconds
        ===> Calculating affinity matrix...
           --> Time elapsed: 0.14 seconds

        EMBEDR: Performing dimensionality reduction with t-SNE!

        Making embedding 1/1
        ===> Running optimization with exaggeration=12.00, lr=208.33 for 250
        iterations...
        Iteration   50, KL divergence 4.2395, 50 iterations in 1.2785 sec
        Iteration  100, KL divergence 3.8564, 50 iterations in 1.3624 sec
        Iteration  150, KL divergence 3.7929, 50 iterations in 1.3644 sec
        Iteration  200, KL divergence 3.7916, 50 iterations in 1.3373 sec
        Iteration  250, KL divergence 3.7915, 50 iterations in 1.3015 sec
           --> Time elapsed: 6.64 seconds
        ===> Running optimization with exaggeration=1.00, lr=208.33 for 750
        iterations...
        Iteration   50, KL divergence 1.8392, 50 iterations in 1.2951 sec
        Iteration  100, KL divergence 1.5378, 50 iterations in 1.3905 sec
        Iteration  150, KL divergence 1.4119, 50 iterations in 1.3123 sec
        Iteration  200, KL divergence 1.3416, 50 iterations in 1.8416 sec
        Iteration  250, KL divergence 1.3009, 50 iterations in 2.8216 sec
        Iteration  300, KL divergence 1.2757, 50 iterations in 3.3737 sec
        Iteration  350, KL divergence 1.2605, 50 iterations in 4.6760 sec
        Iteration  400, KL divergence 1.2502, 50 iterations in 4.8709 sec
        Iteration  450, KL divergence 1.2412, 50 iterations in 5.4483 sec
        Iteration  500, KL divergence 1.2357, 50 iterations in 5.5746 sec
        Iteration  550, KL divergence 1.2308, 50 iterations in 5.9486 sec
        Iteration  600, KL divergence 1.2251, 50 iterations in 6.5560 sec
        Iteration  650, KL divergence 1.2220, 50 iterations in 7.8521 sec
        Iteration  700, KL divergence 1.2186, 50 iterations in 8.0716 sec
        Iteration  750, KL divergence 1.2148, 50 iterations in 7.9768 sec
           --> Time elapsed: 69.01 seconds

        Generating null embeddings!

        Making embedding 1/1
        ===> Finding 90 nearest neighbors using Annoy approximate search using
        euclidean distance...
           --> Time elapsed: 1.52 seconds
        ===> Calculating affinity matrix...
           --> Time elapsed: 0.17 seconds
        ===> Running optimization with exaggeration=12.00, lr=208.33 for 250
        iterations...
        Iteration   50, KL divergence 3.6849, 50 iterations in 1.4261 sec
        Iteration  100, KL divergence 3.6849, 50 iterations in 1.4006 sec
        Iteration  150, KL divergence 3.6849, 50 iterations in 1.3852 sec
        Iteration  200, KL divergence 3.6849, 50 iterations in 1.4197 sec
        Iteration  250, KL divergence 3.6849, 50 iterations in 1.3978 sec
           --> Time elapsed: 7.03 seconds
        ===> Running optimization with exaggeration=1.00, lr=208.33 for 750
        iterations...
        Iteration   50, KL divergence 3.6849, 50 iterations in 1.3969 sec
        Iteration  100, KL divergence 3.5462, 50 iterations in 1.4457 sec
        Iteration  150, KL divergence 3.4705, 50 iterations in 1.4580 sec
        Iteration  200, KL divergence 3.4616, 50 iterations in 1.4063 sec
        Iteration  250, KL divergence 3.4572, 50 iterations in 1.4360 sec
        Iteration  300, KL divergence 3.4546, 50 iterations in 1.4415 sec
        Iteration  350, KL divergence 3.4529, 50 iterations in 1.4424 sec
        Iteration  400, KL divergence 3.4521, 50 iterations in 1.4172 sec
        Iteration  450, KL divergence 3.4517, 50 iterations in 1.4487 sec
        Iteration  500, KL divergence 3.4512, 50 iterations in 1.5098 sec
        Iteration  550, KL divergence 3.4508, 50 iterations in 1.4553 sec
        Iteration  600, KL divergence 3.4507, 50 iterations in 1.4576 sec
        Iteration  650, KL divergence 3.4505, 50 iterations in 1.4777 sec
        Iteration  700, KL divergence 3.4505, 50 iterations in 1.4248 sec
        Iteration  750, KL divergence 3.4502, 50 iterations in 1.4421 sec
           --> Time elapsed: 21.66 seconds

        Calculating EES using 1 embeddings and 1 nulls!

        """

        err_str = "Input data must be an N_samples x N_features numpy array."
        assert isinstance(X, np.ndarray), err_str
        assert X.ndim == 2, err_str

        n_samp, n_feat = X.shape
        self.n_samples, self.n_features = X.shape

        if self.verbose >= 0:
            print(f"\nEMBEDR: Fitting {n_samp} x {n_feat} data!\n")

        r_seed = self.random_state.get_state()[1][0] % (2**31)

        if self.verbose > 1:
            print(f"Generating high-dim affinity matrix (P)")

        aff_verbose = True if self.verbose > 0 else False
        self.aff_name = self.project_name + "_affinity_matrix.pkl"

        try:
            if self.do_cache:
                if self.verbose > 1:
                    print(f"\nTrying to load {self.aff_name}!")

                with open(self.aff_name, 'rb') as f:
                    self._affmat = pkl.load(f)

                if self.verbose > 1:
                    print(f"{self.aff_name} loaded!")

            else:
                raise FileNotFoundError

        except FileNotFoundError:

            self._affmat = PerplexityBasedNN(X,
                                             perplexity=self.perplexity,
                                             n_jobs=self.n_jobs,
                                             random_state=r_seed,
                                             verbose=aff_verbose)

            if self.do_cache:
                if self.verbose > 1:
                    print(f"\nSaving {self.aff_name} to file!")

                del self._affmat.knn_index

                with open(self.aff_name, 'wb') as f:
                    pkl.dump(self._affmat, f)

        if self.DRA in ['t-sne', 'tsne']:
            self._fit_tSNE(X)

        elif self.DRA == 'umap':
            self._fit_UMAP(X)

        else:
            self._fit_PCA(X)

        self._calc_EES()

    def fit_transform(self, X):
        """Embeds data `X` in lower-dimensional space using indicated DRA.

        This method also automatically generates the required null embeddings
        and quality statistics in order to assign each sample in `X` a
        "p-value" indicating the likelihood that a random sample would be
        embedded that well.

        Returns:
        --------
        data_Y: (n_data_embed x n_samples x n_components) array
            Embedded data in `n_components` space.

        Attributes
        ----------
        n_samples: int
            Number of samples in supplied data `X`.

        n_features: int
            Number of features in supplied data `X`.

        data_Y: (n_data_embed x n_samples x n_components) array
            Data `X` embedded `n_data_embed` times by `dimred_alg`.

        null_Y: (n_null_embed x n_samples x n_components) array
            Null data embedded `n_null_embed` times by `dimred_alg`.  To
            recover the high-dimensional null data, use
            `utility.generate_nulls(X)`.

        EES_data: (n_data_embed x n_samples) array
            Quality statistics for each sample in the original data set
            (for each) embedding.

        EES_null: (n_null_embed x n_samples) array
            Quality statistics for each sample in each null data set.

        pVals: (n_data_embed x n_samples) array
            EMBEDR p-values. Summarized across `N_data_embed` embeddings using
            Simes' method.

        Example
        -------
        > from embedr import EMBEDR
        > import numpy as np
        > X = np.loadtxt("./Data/mnist2500_X.txt")
        >
        > embedr_obj = EMBEDR()
        > data_Y = embedr_obj.fit_transform(X)

        EMBEDR: Fitting 2500 x 784 data!

        """
        self.fit(X)

        return self.data_Y

    def _fit_tSNE(self, X):

        if self.verbose > 0:
            print(f"\nEMBEDR: Performing dimensionality reduction with t-SNE!")

        aff_verbose = True if self.verbose > 0 else False
        tSNE_verbose = True if self.verbose > 0 else False

        r_seed = self.random_state.get_state()[1][0] % (2**31)

        tSNE_defaults = {'n_iter': 750,
                         'exaggeration': 12,
                         'early_exag_iter': 250,
                         'exag_mom': 0.5,
                         'momentum': 0.8,
                         'openTSNE_params': {'learning_rate': 'auto'}}
        tSNE_defaults.update(self.DRA_params)

        self.tSNE_params = tSNE_defaults

        self.dY_name = self.project_name + "_tSNE_data_embeds.pkl"
        self.nP_name = self.project_name + "_tSNE_null_affinities.pkl"
        self.nY_name = self.project_name + "_tSNE_null_embeds.pkl"

        try:
            if self.do_cache:
                if self.verbose > 1:
                    print(f"\nTrying to load {self.dY_name} from file!")

                with open(self.dY_name, 'rb') as f:
                    self.data_Y = pkl.load(f)

                if self.verbose > 1:
                    print(f"{self.dY_name} loaded!")

            else:
                raise FileNotFoundError

        except FileNotFoundError:

            data_Y = np.zeros((self.n_data_embed,
                               self.n_samples,
                               self.n_components))

            for eNo in range(self.n_data_embed):
                if self.verbose > 0:
                    print(f"\nMaking embedding {eNo + 1}/{self.n_data_embed}")

                init_Y = initRand(X,
                                  n_components=self.n_components,
                                  random_state=r_seed + eNo,
                                  verbose=tSNE_verbose)

                tSNE_dY = TSNEEmbedding(init_Y,
                                        n_jobs=-1,
                                        affinities=self._affmat,
                                        verbose=tSNE_verbose,
                                        **tSNE_defaults['openTSNE_params'])

                ## Early exaggeration phase
                tSNE_dY.optimize(n_iter=tSNE_defaults['early_exag_iter'],
                                 exaggeration=tSNE_defaults['exaggeration'],
                                 momentum=tSNE_defaults['exag_mom'],
                                 inplace=True,
                                 verbose=tSNE_verbose)

                ## Early exaggeration phase
                tSNE_dY.optimize(n_iter=tSNE_defaults['n_iter'],
                                 exaggeration=1,
                                 momentum=tSNE_defaults['momentum'],
                                 inplace=True,
                                 verbose=tSNE_verbose)

                data_Y[eNo] = tSNE_dY[:]

            self.data_Y = data_Y

            del tSNE_dY

            if self.do_cache:
                if self.verbose > 1:
                    print(f"\nSaving {self.dY_name} to file!")

                with open(self.dY_name, 'wb') as f:
                    pkl.dump(self.data_Y, f)

        try:
            if self.do_cache:
                if self.verbose > 1:
                    print(f"\nTrying to load {self.nP_name} from file!")

                with open(self.nP_name, 'rb') as f:
                    self._null_affmat = pkl.load(f)

                if self.verbose > 1:
                    print(f"{self.nP_name} loaded!")

                if self.verbose > 1:
                    print(f"\nTrying to load {self.nY_name} from file!")

                with open(self.nY_name, 'rb') as f:
                    self.null_Y = pkl.load(f)

                if self.verbose > 1:
                    print(f"{self.nY_name} loaded!")

            else:
                raise FileNotFoundError

        except FileNotFoundError:

            if self.verbose > 0:
                print(f"\nGenerating null embeddings!")

            null_P = {}

            null_Y = np.zeros((self.n_null_embed,
                               self.n_samples,
                               self.n_components))

            for nNo in range(self.n_null_embed):
                if self.verbose > 0:
                    print(f"\nMaking embedding {nNo + 1}/{self.n_null_embed}")

                null_X = utl.generate_nulls(X, seed=r_seed + nNo).squeeze()

                nP = PerplexityBasedNN(null_X,
                                       perplexity=self.perplexity,
                                       n_jobs=self.n_jobs,
                                       random_state=r_seed,
                                       verbose=aff_verbose)

                del nP.knn_index

                null_P[nNo] = nP

                init_Y = initRand(null_X,
                                  n_components=self.n_components,
                                  random_state=r_seed + nNo,
                                  verbose=tSNE_verbose)

                tSNE_nY = TSNEEmbedding(init_Y,
                                        n_jobs=-1,
                                        affinities=nP,
                                        verbose=tSNE_verbose,
                                        **tSNE_defaults['openTSNE_params'])

                ## Early exaggeration phase
                tSNE_nY.optimize(n_iter=tSNE_defaults['early_exag_iter'],
                                 exaggeration=tSNE_defaults['exaggeration'],
                                 momentum=tSNE_defaults['exag_mom'],
                                 inplace=True,
                                 verbose=tSNE_verbose)

                ## Early exaggeration phase
                tSNE_nY.optimize(n_iter=tSNE_defaults['n_iter'],
                                 exaggeration=1,
                                 momentum=tSNE_defaults['momentum'],
                                 inplace=True,
                                 verbose=tSNE_verbose)

                null_Y[nNo] = tSNE_nY[:]

            self._null_affmat = null_P

            self.null_Y = null_Y

            if self.do_cache:
                if self.verbose > 1:
                    print(f"Trying to write {self.nP_name} to file!")

                with open(self.nP_name, 'wb') as f:
                    pkl.dump(null_P, f)

                if self.verbose > 1:
                    print(f"Trying to write {self.nY_name} to file!")

                with open(self.nY_name, 'wb') as f:
                    pkl.dump(null_Y, f)

    def _fit_UMAP(self, X):

        if self.verbose > 0:
            print(f"\nEMBEDR: Performing dimensionality reduction with UMAP!")

        start = time.time()

        aff_verbose = True if self.verbose > 0 else False
        UMAP_verbose = True if self.verbose > 0 else False

        r_seed = self.random_state.get_state()[1][0] % (2**31)

        UMAP_defaults = {'min_dist': 0.1,
                         'n_neighbors': 30}
        UMAP_defaults.update(self.DRA_params)

        self.UMAP_params = UMAP_defaults

        self.dY_name = self.project_name + "_UMAP_data_embeds.pkl"
        self.nP_name = self.project_name + "_UMAP_null_affinities.pkl"
        self.nY_name = self.project_name + "_UMAP_null_embeds.pkl"

        try:
            if self.do_cache:
                if self.verbose > 1:
                    print(f"\nTrying to load {self.dY_name} from file!")

                with open(self.dY_name, 'rb') as f:
                    self.data_Y = pkl.load(f)

                if self.verbose > 1:
                    print(f"{self.dY_name} loaded!")

            else:
                raise FileNotFoundError

        except FileNotFoundError:

            from umap import UMAP

            data_Y = np.zeros((self.n_data_embed,
                               self.n_samples,
                               self.n_components))

            for eNo in range(self.n_data_embed):
                if self.verbose > 0:
                    print(f"\nMaking embedding {eNo + 1}/{self.n_data_embed}")

                UMAP_obj = UMAP(n_components=self.n_components,
                                random_state=r_seed + eNo,
                                verbose=UMAP_verbose,
                                **self.UMAP_params)

                UMAP_dY = UMAP_obj.fit_transform(X)

                data_Y[eNo] = UMAP_dY[:]

            del UMAP_dY

            self.data_Y = data_Y

            if self.do_cache:
                if self.verbose > 1:
                    print(f"\nSaving {self.dY_name} to file!")

                with open(self.dY_name, 'wb') as f:
                    pkl.dump(self.data_Y, f)

        try:
            if self.do_cache:
                if self.verbose > 1:
                    print(f"\nTrying to load {self.nP_name} from file!")

                with open(self.nP_name, 'rb') as f:
                    self._null_affmat = pkl.load(f)

                if self.verbose > 1:
                    print(f"{self.nP_name} loaded!")

                if self.verbose > 1:
                    print(f"\nTrying to load {self.nY_name} from file!")

                with open(self.nY_name, 'rb') as f:
                    self.null_Y = pkl.load(f)

                if self.verbose > 1:
                    print(f"{self.nY_name} loaded!")

            else:
                raise FileNotFoundError

        except FileNotFoundError:

            from umap import UMAP

            if self.verbose > 0:
                print(f"\nGenerating null embeddings!")

            null_P = {}

            null_Y = np.zeros((self.n_null_embed,
                               self.n_samples,
                               self.n_components))

            for nNo in range(self.n_null_embed):
                if self.verbose > 0:
                    print(f"\nMaking embedding {nNo + 1}/{self.n_null_embed}")

                null_X = utl.generate_nulls(X, seed=r_seed + nNo).squeeze()

                nP = PerplexityBasedNN(null_X,
                                       perplexity=self.perplexity,
                                       n_jobs=self.n_jobs,
                                       random_state=r_seed,
                                       verbose=aff_verbose)

                del nP.knn_index

                null_P[nNo] = nP

                UMAP_obj = UMAP(n_components=self.n_components,
                                random_state=r_seed + nNo,
                                verbose=UMAP_verbose,
                                **self.UMAP_params)

                UMAP_nY = UMAP_obj.fit_transform(null_X)

                null_Y[nNo] = UMAP_nY[:]

            self._null_affmat = null_P

            self.null_Y = null_Y

            if self.do_cache:
                if self.verbose > 1:
                    print(f"Trying to write {self.nP_name} to file!")

                with open(self.nP_name, 'wb') as f:
                    pkl.dump(null_P, f)

                if self.verbose > 1:
                    print(f"Trying to write {self.nY_name} to file!")

                with open(self.nY_name, 'wb') as f:
                    pkl.dump(null_Y, f)

    def _fit_PCA(self, X):

        if self.verbose > 0:
            print(f"\nEMBEDR: Performing dimensionality reduction with PCA!")

        start = time.time()

        aff_verbose = True if self.verbose > 0 else False
        PCA_verbose = True if self.verbose > 0 else False

        r_seed = self.random_state.get_state()[1][0] % (2**31)

        PCA_defaults = {}
        PCA_defaults.update(self.DRA_params)

        self.PCA_params = PCA_defaults

        self.dY_name = self.project_name + "_PCA_data_embeds.pkl"
        self.nP_name = self.project_name + "_PCA_null_affinities.pkl"
        self.nY_name = self.project_name + "_PCA_null_embeds.pkl"

        try:
            if self.do_cache:
                if self.verbose > 1:
                    print(f"\nTrying to load {self.dY_name} from file!")

                with open(self.dY_name, 'rb') as f:
                    self.data_Y = pkl.load(f)

                if self.verbose > 1:
                    print(f"{self.dY_name} loaded!")

            else:
                raise FileNotFoundError

        except FileNotFoundError:

            PCA_obj = PCA(n_components=self.n_components,
                          **self.PCA_params)

            self.data_Y = PCA_obj.fit_transform(X)[np.newaxis, :]

            if self.do_cache:
                if self.verbose > 1:
                    print(f"\nSaving {self.dY_name} to file!")

                with open(self.dY_name, 'wb') as f:
                    pkl.dump(self.data_Y, f)

        try:
            if self.do_cache:
                if self.verbose > 1:
                    print(f"\nTrying to load {self.nP_name} from file!")

                with open(self.nP_name, 'rb') as f:
                    self._null_affmat = pkl.load(f)

                if self.verbose > 1:
                    print(f"{self.nP_name} loaded!")

                if self.verbose > 1:
                    print(f"\nTrying to load {self.nY_name} from file!")

                with open(self.nY_name, 'rb') as f:
                    self.null_Y = pkl.load(f)

                if self.verbose > 1:
                    print(f"{self.nY_name} loaded!")

            else:
                raise FileNotFoundError

        except FileNotFoundError:

            if self.verbose > 0:
                print(f"\nGenerating null embeddings!")

            null_P = {}

            null_Y = np.zeros((self.n_null_embed,
                               self.n_samples,
                               self.n_components))

            for nNo in range(self.n_null_embed):
                if self.verbose > 0:
                    print(f"\nMaking embedding {nNo + 1}/{self.n_null_embed}")

                null_X = utl.generate_nulls(X, seed=r_seed + nNo).squeeze()

                nP = PerplexityBasedNN(null_X,
                                       perplexity=self.perplexity,
                                       n_jobs=self.n_jobs,
                                       random_state=r_seed,
                                       verbose=aff_verbose)

                del nP.knn_index

                null_P[nNo] = nP

                PCA_obj = PCA(n_components=self.n_components,
                              **self.PCA_params)

                null_Y[nNo] = PCA_obj.fit_transform(null_X)

            self._null_affmat = null_P

            self.null_Y = null_Y

            if self.do_cache:
                if self.verbose > 1:
                    print(f"Trying to write {self.nP_name} to file!")

                with open(self.nP_name, 'wb') as f:
                    pkl.dump(null_P, f)

                if self.verbose > 1:
                    print(f"Trying to write {self.nY_name} to file!")

                with open(self.nY_name, 'wb') as f:
                    pkl.dump(null_Y, f)

    def _calc_EES(self):

        if self.verbose > 0:
            print(f"\nCalculating EES using {self.n_data_embed} embeddings"
                  f" and {self.n_null_embed} nulls!")

        self.EES_name = self.project_name + "_EES.pkl"

        try:
            if self.do_cache:
                if self.verbose > 1:
                    print(f"\nTrying to load {self.EES_name} from file!")

                with open(self.EES_name, 'rb') as f:
                    EES_dict = pkl.load(f)

                self.EES_data  = EES_dict['data']
                self.EES_null  = EES_dict['null']
                self.EES_pVals = EES_dict['all p-values']
                self.pVals     = EES_dict['p-values']

                if self.verbose > 1:
                    print(f"{self.EES_name} loaded successfully!")
            else:
                raise FileNotFoundError

        except (KeyError, FileNotFoundError):

            dP = self._affmat.P.toarray()
            dP = dP / dP.sum(axis=1)[:, np.newaxis]

            EES_data = np.zeros((self.n_data_embed, self.n_samples))

            for eNo in range(self.n_data_embed):
                EES_data[eNo] = self._calc_DKL(dP, self.data_Y[eNo])

            self.EES_data = EES_data[:]
            del EES_data

            EES_null = np.zeros((self.n_null_embed, self.n_samples))

            for nNo in range(self.n_null_embed):
                nP = self._null_affmat[nNo].P.toarray()
                nP = nP / np.sum(nP, axis=1)

                EES_null[nNo] = self._calc_DKL(nP, self.null_Y[nNo])

            self.EES_null = EES_null[:]
            del EES_null

            self._all_pVals = utl.calculate_EMBEDR_pValues(self.EES_data,
                                                           self.EES_null)

            self.pVals = utl.calculate_Simes_pValues(self._all_pVals)

            if self.do_cache:

                EES_dict = {'data': self.EES_data,
                            'null': self.EES_null,
                            'all p-values': self._all_pVals,
                            'p-values': self.pVals}

                if self.verbose > 1:
                    print(f"\nSaving EES and p-values to file!")

                with open(self.EES_name, 'wb') as f:
                    pkl.dump(EES_dict, f)

    def _calc_DKL(self, P, Y, metric="sqeuclidean"):

        Q = 1 / (1 + pwd(Y, metric=metric))
        Q = Q / Q.sum(axis=1)[:, np.newaxis]

        DKL = np.log(P + EPSILON) - np.log(Q + EPSILON)
        DKL = np.sum(P * DKL, axis=1).squeeze()

        if np.any(DKL < 0) and (self.verbose > 0):
            print("WARNING: Illegal values detected for DKL!")
            if self.verbose > 2:
                print((DKL < 0).nonzero())

        return DKL

    def plot(self,
             ax=None,
             cax=None,
             show_cbar=True,
             embed_2_show=0,
             plot_data=True,
             cbar_ticks=None,
             cbar_ticklabels=None,
             pVal_clr_change=[0, 1, 2, 3, 4],
             scatter_s=5,
             scatter_alpha=0.4,
             scatter_kwds={},
             text_kwds={},
             cbar_kwds={},
             cite_EMBEDR=True):
        """Generates scatter plot of embedded data colored by EMBEDR p-value

        Parameters
        ----------
        ax: matplotlib.axes
            Axis on which to place the scatter plot.  If None, axes will be
            generated.  Default is None.

        cax: matplotlib.axes
            Axis on which to place the colorbar.  If None, axes will be
            generated.  Default is None.

        show_cbar: bool
            Flag indicating whether to show the colorbar.  Default is True.

        embed_2_show: int
            Index in range 0:`n_data_embed` indicating which embedding to plot.
            Default is 0.

        plot_data: bool
            Flag indicating whether to plot data or null embeddings.

        cbar_ticks: array-like
            If not None, values to use to set the colorbar ticks.  Default is
            None, which results in [0, 1, 2, 3, 4].

        cbar_ticklabels: array-like
            If not `None`, values to use to label the colorbar ticks. Default
            is None.

        pVal_clr_change: array-like
            -log10(pVals) at which to break up the categorical color bar.
            Default is [0, 1, 2, 3, 4].

        scatter_s: float
            markersize parameter for `plt.scatter`. Default is 5.

        scatter_alpha: float in [0, 1]
            "alpha" parameter for `plt.scatter`.  Default is 0.4.

        scatter_kwds: dict
            Other keywords for `matplotlib.pyplot.scatter`.  Default is {}.

        text_kwds: dict
            Other keywords for `matplotlib.pyplot.text`.  Default is {}.

        cbar_kwds: dict
            Other keywords for `fig.colorbar`.  Default is {}.

        cite_EMBEDR: bool
            Flag indicating whether or not to place EMBEDR citation on the
            plot.  Default is `True`.

        Returns
        -------
        ax: matplotlib.axes
            Axis on which the data have been plotted.

        Example
        -------
        > import matplotlib.pyplot as plt
        > fig, ax = plt.subplots(1, 1)
        > ax = embedr_obj.plot(ax=ax)
        """

        fig = plt.gcf()
        if ax is None:
            ax = fig.gca()

        if plot_data:
            Y = self.data_Y[embed_2_show]
        else:
            Y = self.null_Y[embed_2_show]

        [pVal_cmap,
         pVal_cnorm] = utl.make_pVal_cmap(change_points=pVal_clr_change)

        color_bounds = np.linspace(pVal_clr_change[0],
                                   pVal_clr_change[-1],
                                   pVal_cmap.N)

        pVals = -np.log10(self.pVals)

        sort_idx = np.argsort(pVals)

        h_ax = ax.scatter(*Y[sort_idx].T,
                          s=scatter_s,
                          c=pVals[sort_idx],
                          cmap=pVal_cmap,
                          norm=pVal_cnorm,
                          alpha=scatter_alpha,
                          **scatter_kwds)

        if cite_EMBEDR:
            _ = ax.text(0.02, 0.02,
                        "Made with the EMBEDR package.",
                        fontsize=6,
                        transform=ax.transAxes,
                        ha='left',
                        va='bottom',
                        **text_kwds)

        if show_cbar:
            cbar_ax = fig.colorbar(h_ax,
                                   cax=cax,
                                   boundaries=color_bounds,
                                   ticks=[],
                                   **cbar_kwds)
            cbar_ax.ax.invert_yaxis()

            if cbar_ticks is None:
                cbar_ticks = pVal_clr_change
            cbar_ax.set_ticks(cbar_ticks)

            if cbar_ticklabels is None:
                cbar_ticklabels = ['1',
                                   '0.1',
                                   r"$10^{-2}$",
                                   r"$10^{-3}$",
                                   r"$10^{-4}$"]
            cbar_ax.set_ticklabels(cbar_ticklabels)

            cbar_ax.ax.tick_params(length=0)
            cbar_ax.ax.set_ylabel("EMBEDR p-Value")

        return ax


if __name__ == "__main__":

    plt.close('all')

    X = np.loadtxt("./Data/mnist2500_X.txt")

    tSNE_embed = EMBEDR(random_state=1, verbose=5,
                        n_data_embed=3,
                        n_null_embed=5,
                        project_name='tSNE_test')
    tSNE_Y = tSNE_embed.fit_transform(X)

    UMAP_embed = EMBEDR(dimred_alg='UMAP', random_state=1, verbose=5,
                        n_data_embed=7,
                        n_null_embed=12,
                        project_name='UMAP_test')
    UMAP_Y = UMAP_embed.fit_transform(X)

    PCA_embed = EMBEDR(dimred_alg='PCA', random_state=1, verbose=5,
                       n_null_embed=10,
                       project_name='PCA_test')
    PCA_Y = PCA_embed.fit_transform(X)

    fig, [ax1, ax2, ax3] = plt.subplots(1, 3, figsize=(12, 5))

    pVal_clr_change = [0, 1, -np.log10(0.05), 2, 4.402]

    ax1 = tSNE_embed.plot(ax=ax1, pVal_clr_change=pVal_clr_change,
                          cite_EMBEDR=False)
    ax2 = UMAP_embed.plot(ax=ax2, show_cbar=False,
                          pVal_clr_change=pVal_clr_change,
                          cite_EMBEDR=False)
    ax3 = PCA_embed.plot(ax=ax3, show_cbar=False,
                         pVal_clr_change=pVal_clr_change,
                         cite_EMBEDR=False)

    ax1.set_xticklabels([])
    ax1.set_yticklabels([])
    ax1.set_xlabel("t-SNE 1", fontsize=12)
    ax1.set_ylabel("t-SNE 2", fontsize=12)
    ax1.set_title("t-SNE")

    ax2.set_xticklabels([])
    ax2.set_yticklabels([])
    ax2.set_xlabel("UMAP 1", fontsize=12)
    ax2.set_ylabel("UMAP 2", fontsize=12)
    ax2.set_title("UMAP")

    ax3.set_xticklabels([])
    ax3.set_yticklabels([])
    ax3.set_xlabel("P.C. 1", fontsize=12)
    ax3.set_ylabel("P.C. 2", fontsize=12)
    ax3.set_title("PCA")

    utl.save_figure(fig, "EMBEDR_test_figure")

    plt.show()
