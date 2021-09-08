"""
===============================================================================
===============================================================================

    EMBEDR:
    Empirical Marginal resampling Better Evaluates Dimensionality Reduction

===============================================================================
===============================================================================

Author: Eric Johnson
Date Created: July 1, 2021
Email: eric.johnson643@gmail.com

===============================================================================
===============================================================================

The EMBEDR algorithm is a method for statistically evaluating the quality of
dimensionality reduction algorithms' projections of data into lower dimensional
spaces.  The method relies on the idea that embedding quality can be assessed
statistically by comparing to an appropriately constructed null embedding.
Once this comparison has been performed, significant features of the data can
be extracted and examined with rigor, facilitating data-driven discoveries in
high-dimensional data.

===============================================================================
"""

from EMBEDR._affinity import GaussianAff_fromPrec
import EMBEDR.affinity as aff
import EMBEDR.nearest_neighbors as nn
import EMBEDR.utility as utl

import numpy as np
import numpy.random as r
import os
from os import path
import pickle as pkl
import scipy.sparse as sp
from sklearn.utils import check_array, check_random_state
import time
import warnings


class EMBEDR(object):
    """Empirical Marginal resampling Better Evaluates Dim. Red. (EMBEDR) Class

    This object implements the EMBEDR pipeline as specified in Johnson, Kath,
    and Mani (2020) https://www.biorxiv.org/content/10.1101/2020.11.18.389031v1

    This frameworks seeks to provide quantitative estimates of dimensionality
    reduction quality by directly constructing a null hypothesis for the
    quality of embeddings due to random chance and then comparing the quality
    of the embedded data to that null expectation.

    Parameters
    ----------

    n_jobs: int
        Number of threads to use when finding nearest neighbors. This follows
        the scikit-learn convention: ``-1`` means to use all processors, ``-2``
        indicates that all but one processor should be used, etc.

    Attributes
    ----------

    Methods
    -------

    Notes
    -----

    Examples
    --------
    """

    def __init__(self,
                 project_name="EMBEDR_project",
                 embed_dir="./data/embeddings",
                 dra='tSNE',
                 dra_params={},
                 param_type="perplexity",  ## Or n_neighbors
                 param_lb=5,    ## This is a lower bound on perp/kNN
                 param_ub=0.5,  ## UB is FRACTION of n_samples
                 n_param=1,     ## Number of params to sweep (from ub to lb).
                 kNN_params={},
                 aff_params={},
                 n_components=2,
                 n_data_embed=1,
                 n_null_embed=1,
                 n_jobs=1,
                 random_state=1,
                 verbose=5):

    ## CODE FOLDING

        self.verbose = float(verbose)
        if self.verbose >= 1:
            print(f"\nInitializing EMBEDR object!\n\n")

        self.project_name = project_name

        if not os.path.isdir(embed_dir):
            raise OSError(f"Embedding directory '{embed_dir}' couldn't be"
                          f" found!")
        self.embed_dir = embed_dir

        self.dra = dra.lower()
        self.dra_params = dra_params

        self.param_type = param_type.lower()

        self.param_lb = int(param_lb)
        assert self.param_lb > 0

        self.param_ub = float(param_ub)
        assert self.param_ub > 0
        assert self.param_ub <= 1

        self.n_param = int(n_param)
        assert self.n_param >= 1

        self.kNN_params = kNN_params
        self.aff_params = aff_params

        self.n_components = int(n_components)
        assert self.n_components >= 1

        if self.dra.lower() == 'pca':
            self.n_data_embed = 1
        else:
            self.n_data_embed = int(n_data_embed)
        assert self.n_data_embed >= 1

        self.n_null_embed = int(n_null_embed)
        assert self.n_null_embed >= 1

        self.n_jobs = int(n_jobs)

        self.random_state = check_random_state(random_state)

        return

    def _set_perplexity_kNN(self):

        ## Set upper bound on parameter (perp/kNN) based on n_samples.
        self.param_ub = int(self.n_samples * self.param_ub)

        self.param_vals = utl.unique_logspace(self.param_lb,
                                              self.param_ub,
                                              self.n_param)

        if self.param_type == 'perplexity':
            self.n_neighbors_arr = np.clip(3 * self.param_vals.copy() + 1,
                                           0, self.n_samples)
            self.perplexity = self.param_vals[-1]
        else:
            self.n_neighbors_arr = self.param_vals.copy()
        self.n_neighbors = self.n_neighbors_arr[-1]

    def _initialize_kNN_index(self, data=True):
        if data:
            return self._initialize_data_kNN_index()
        else:
            return self._initialize_null_kNN_index()

    def _initialize_data_kNN_index(self):
        index = nn._initialize_kNN_index(self.data_X,
                                         n_jobs=self.n_jobs,
                                         random_state=self.random_state,
                                         verbose=self.verbose,
                                         **self.kNN_params)

        index_params = {'metric': index.metric,
                        'metric_params': index.metric_params}
        self.kNN_params.update(index_params)

        return index

    def fit_affinity(self, perplexity, data=True):

        ## REMOVE THIS LATER!  WE DON'T NEED TO STORE THIS!
        ## Also introduce ability to do different affinities somewhere.
        ## Also can omit random_state since we're supplying kNN index.
        tmp_aff = aff.FixedEntropyAffinity(perplexity=perplexity,
                                           kNN_params=self.kNN_params,
                                           n_jobs=self.n_jobs,
                                           verbose=self.verbose,
                                           **self.aff_params)
        tmp_aff.n_neighbors = self.n_neighbors

        if data:
            tmp_aff.fit(self.data_kNN)
            ## We only want to save precisions and probably ROW SUMS!
            return {'precisions': tmp_aff.kernel_params['precisions'],
                    'row_sums': tmp_aff.kernel_params['row_sums']}

    def fit_embedding(self, P, perplexity, data=True):

        if self.dra == 'tsne':
            return self._fit_tSNE(P, perplexity, data=data)

    def _fit_tSNE(self, P, perplexity, data=True):

        if data:
            return self._fit_tSNE_data(P, perplexity)

    def _fit_tSNE_data(self, P, perplexity):
        return

    def fit(self, X):

        ## Check that the data is a 2D array
        self.data_X = check_array(X, accept_sparse=True, ensure_2d=True)

        ## Get the data shape
        self.n_samples, self.n_features = self.data_X.shape

        ## Initialize the kNN graph (this doesn't fit the graph!)
        self.data_kNN = self._initialize_kNN_index(data=True)

        ## Set parameter upper bound and get n_neighbors
        self._set_perplexity_kNN()

        ## Fit the kNN graph to the data
        self.data_kNN.fit(self.data_X, self.n_neighbors)

        ## At this point we need to split.  If we're using n_neighbors as our
        ## parameter to vary, then we need to determine the corresponding
        ## perplexity.  This can be done LATER using our kEff interpolator.
        ## Assuming that we want to iterate over perplexity...
        self.data_aff = {}
        for pNo, p in enumerate(self.param_vals):

            if self.verbose >= 1:
                print(f"\nParameter `{self.param_type}` = {p}"
                      f" ({pNo + 1} / {self.n_param})\n")

            if self.param_type == 'perplexity':
                self.perplexity = p
                self.n_neighbors = self.n_neighbors_arr[pNo]

            self.data_aff[p] = self.fit_affinity(perplexity=p, data=True)

            P = GaussianAff_fromPrec(self.data_kNN.kNN_dst,
                                     self.data_aff[p]['precisions'],
                                     self.data_aff[p]['row_sums'])

            print(P.shape)

            ## LATER: Write a calculator that gets kEff from P.

            ## Once we have P, we can embed.
            self.data_Y[p] = self.fit_embedding(P, perplexity=p, data=True)

            ## Once we have Y, we can get Q and therefore EES
            ## numba calculator for EES given PWD, tau, and Y.

            ## Repeat with nulls

            ## Get p-values.

        return






