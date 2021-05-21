"""
###############################################################################
    Procedure: t-SNE Perplexity Sweep
###############################################################################

    Author: Eric Johnson
    Date Created: Sunday, April 4, 2021
    Email: ericjohnson1.2015@u.northwestern.edu

###############################################################################

    In this script, I want to make a more general template for doing a
    perplexity sweep that I can copy and build off of.  I also want to set up
    a script for re-doing the MNIST, Marrow, and Diaphragm sweeps.

    The idea here is to do as much caching along the way as possible. The
    general order of operations is:
     -  Set up data
         -  Set up perplexity array
         -  Set up naming scheme
         -  Set up null data (should we do more than one, yes, let's be safe.)
     -  Set up largest-scale (largest-perplexity) affinity matrix.
         -  We probably don't need to save the kNN index for this, but we will
            anyways.
         -  Repeat for nulls!
     - for perp in perp_arr:
         -  Embed the data
             -  Recalculate the affinity matrix based on the new perplexity.
             -  Save the tau and kEff arrays!
             -  Calculate EES right away!
         -  Embed the nulls
             -  Recalculate the affinity matrix based on the new perplexity.
             -  Calculate the EES right away!
             -  We don't need to save anything except the embeddings and EES.
         - Calculate p-Values
             - Probably don't need to save all the intermediate p-values...

         - Make sure to cache at EACH iteration!

    Note, we're going to use the most up-to-date parameters here. Specifically:
     -  n_data_embed = 5
     -  n_null_embed = 10
     -  EES calculated with an asymmetric P
     -  p-Values calculated by averaging

    To save (potentially a lot) of time, we're going to also employ callbacks
    to quit t-SNE early, using the scheme proposed in Belkina et al. (2019).

###############################################################################
"""

from embedr.affinity import FixedEntropyAffinity

import numpy as np
import numpy.random as r

from openTSNE import TSNEEmbedding
from openTSNE.callbacks import Callback
from openTSNE.initialization import random as initRand

import os
from os import path
import pandas as pd
import PaperV4_PlottingScripts.plotting_utility as pUtl
import pickle as pkl
import scipy.sparse as sp
from sklearn.metrics import pairwise_distances as pwd
from sklearn.preprocessing import normalize
import time

EPSILON = np.finfo(np.float64).eps

data_dir = "./Data"
embed_dir = "./Embeddings/ParameterSweep"


class QuitEarlyExaggeration(Callback):
    """Use the Belkina et al. (2019) method from 2019 to stop EE

    Specifically, track the relative change in DKL vs iteration and quit EE
    when the relative DKL reaches a peak. (Quit as soon as it starts going
    down.)
    """

    def __init__(self, max_iter=500, min_iter=100, verbose=False):
        self.iter_count = 0
        self.last_log_time = None
        self.DKL = []
        self.max_rel_err = 0
        self.max_iter = max_iter
        self.min_iter = min_iter
        self.verbose = verbose

    def optimization_about_to_start(self):
        self.last_log_time = time.time()
        self.iter_count = 0
        self.DKL = []
        self.max_rel_err = 0

    def __call__(self, iteration, error, embedding):
        now = time.time()
        duration = now - self.last_log_time
        self.last_log_time = now

        self.iter_count = iteration

        ## Calculate the relative change in error
        if len(self.DKL) > 0:
            rel_err = (self.DKL[-1] - error) / self.DKL[-1]
        else:
            rel_err = 0

        self.DKL.append(error)

        if self.verbose:
            print(f"Iteration {iteration:04d}: D_KL = {error:6.4f}"
                  f" (Rel Change = {rel_err:6.2%} in {duration:.4f} sec)")

        ## If the relative change in error isn't decreasing
        if (rel_err - self.max_rel_err) >= -1.e-4:
            ## If it's *increasing* update the max relative error
            if rel_err > self.max_rel_err:
                self.max_rel_err = rel_err
        ## If it's decreasing, quit!
        else:
            if self.iter_count >= self.min_iter:
                return True

        ## If we've gone `max_iter` and there's been no change, quit.
        if self.iter_count > self.max_iter:
            if np.abs(self.DKL[-1] - self.DKL[0]) < 1.e-6:
                return True


class QuitWhenEmbedded(Callback):
    """Use the Belkina et al. (2019) method from 2019 to stop t-SNE

    Specifically, track the relative change in DKL vs iteration and quit the
    gradient descent when the relative changes in DKL are less than 0.0001.
    """

    def __init__(self, rel_err_tol=1.e-4, min_iter=250, verbose=False):
        self.iter_count = 0
        self.last_log_time = None
        self.DKL = []
        self.rel_err_tol = rel_err_tol
        self.verbose = verbose
        self.min_iter = min_iter

    def optimization_about_to_start(self):
        self.last_log_time = time.time()
        self.iter_count = 0
        self.DKL = []

    def __call__(self, iteration, error, embedding):
        now = time.time()
        duration = now - self.last_log_time
        self.last_log_time = now

        self.iter_count = iteration

        ## Calculate the relative change in error
        if len(self.DKL) < 1:
            rel_err = np.inf
        else:
            rel_err = (self.DKL[-1] - error) / error

        self.DKL.append(error)

        if self.verbose:
            print(f"Iteration {iteration:04d}: D_KL = {error:6.4f}"
                  f" (Rel Change = {rel_err:6.2%} in {duration:.4f} sec)")

        ## If enough iterations have elapsed...
        if self.iter_count > self.min_iter:
            ## If there has been *any* change in the DKL...
            if np.abs(self.DKL[-1] - self.DKL[0]) > 1.e-6:
                ## If the relative error decreased below tolerance, quit!
                if rel_err < self.rel_err_tol:
                    return True


def load_tSNE(X,
              precalc_kNN=None,
              file_name="Default_File_Name.pkl",
              file_path=embed_dir,
              perplexity=30,
              random_seed=1,
              n_embed=1,
              affinity_params={},
              n_components=2,
              n_ee_iter=500,
              min_ee_iter=100,
              exaggeration=12,
              ee_mom=0.5,
              n_iter=1000,
              min_iter=250,
              mom=0.8,
              n_jobs=-1,
              verbose=True):

    ## Get the data shape
    n_samples, n_features = X.shape

    n_embeds_made = 0
    try:
        print(f"\nTrying to load {file_name} from file...")
        with open(file_path, 'rb') as f:
            [Y, affinity_matrix] = pkl.load(f)

        n_embeds_made = len(Y)  ## How many embeddings have already been made?
        print(f"... file loaded, checking size...")
        assert n_embeds_made >= n_embed
        print(f"... there are enough embeddings! ({n_embeds_made} >= "
              f"{n_embed})")

    except (AssertionError, FileNotFoundError):
        if n_embeds_made == 0:
            print(f"... couldn't load file!")
        elif n_embeds_made < n_embed:
            print(f"... there aren't enough embeddings!"
                  f" ({n_embeds_made} < {n_embed})")

        ## Set up affinity matrix... initialize the perplexity and kNN.
        affinity_params['perplexity'] = perplexity
        n_neighbors = np.min([n_samples - 1, int(3 * perplexity)])
        affinity_params['n_neighbors'] = n_neighbors

        affinity_matrix = FixedEntropyAffinity(**aff_params)
        affinity_matrix._check_data(X)
        [affinity_matrix.perplexity,
         affinity_matrix.perp_arr
         ] = affinity_matrix._check_perplexity(perplexity)

        if precalc_kNN is None:
            affinity_matrix.fit(X)
        else:
            affinity_matrix.kNN_index = precalc_kNN.kNN_index
            affinity_matrix.indices   = precalc_kNN.indices[:, :n_neighbors]
            affinity_matrix.distances = precalc_kNN.distances[:, :n_neighbors]

            if precalc_kNN.perplexity != perplexity:
                print(f"\nRecalculating affinity matrix from kNN graph...")
                now = time.time()
                affinity_matrix.P = affinity_matrix._compute_affinities()
                print(f"... finished in {time.time() - now:.2f} seconds!")
            else:
                affinity_matrix.P = precalc_kNN.P.copy()
                affinity_matrix.kernel_params = precalc_kNN.kernel_params

        n_2_make = n_embed - n_embeds_made
        tmp_Y = np.zeros((n_2_make, n_samples, n_components))
        for eNo in range(n_2_make):
            print(f"\nGenerating embedding {eNo + 1 + n_embeds_made}/"
                  f"{n_embed}")

            tmp_rs = random_seed + eNo + n_embeds_made

            now = time.time()

            init_Y = initRand(X, n_components=n_components,
                              random_state=tmp_rs)

            cb1 = QuitEarlyExaggeration(max_iter=n_ee_iter,
                                        min_iter=min_ee_iter)
            eY = TSNEEmbedding(init_Y, affinity_matrix, verbose=verbose,
                               callbacks=cb1, callbacks_every_iters=2,
                               n_jobs=n_jobs)

            eY = eY.optimize(n_iter=n_ee_iter,
                             exaggeration=exaggeration,
                             momentum=ee_mom)

            duration = time.time() - now
            print(f"===> Early exaggeration quit after  {cb1.iter_count}"
                  f" iterations! (Total: {duration:.2f} sec)")

            cb2 = QuitWhenEmbedded(min_iter=min_iter)
            eY = TSNEEmbedding(eY, affinity_matrix, verbose=verbose,
                               callbacks=cb2, callbacks_every_iters=10,
                               n_jobs=n_jobs)

            eY = eY.optimize(n_iter=n_iter,
                             exaggeration=1,
                             momentum=mom)
            duration = time.time() - now
            print(f"===> Optimization finished after "
                  f"{cb1.iter_count + cb2.iter_count} iterations!"
                  f" (Total: {duration:.2f} sec)")

            tmp_Y[eNo] = eY[:]

        if n_embeds_made > 0:
            Y = np.vstack((Y, tmp_Y))
        else:
            Y = tmp_Y[:]

        with open(file_path, 'wb') as f:
            pkl.dump([Y, affinity_matrix], f)
        print(f"There are {n_embed} embeddings... saved to file!")

    return Y, affinity_matrix


def calc_DKL(P, row_idx, col_idx, eY):

    if eY.ndim == 2:
        eY = eY[np.newaxis, :, :]

    n_embed, n_samples, _ = eY.shape
    DKL = np.zeros((n_embed, n_samples))

    for eNo, Y in enumerate(eY):

        Q = 1 / (1 + pwd(Y, metric="sqeuclidean"))
        Q = Q / Q.sum(axis=1)[:, np.newaxis]

        for rowNo, [start, end] in enumerate(zip(row_idx[:-1], row_idx[1:])):
            colNos = col_idx[start:end]
            P_row = P[start:end]
            Q_row = Q[rowNo, colNos]

            DKL_row = np.log(P_row + EPSILON) - np.log(Q_row + EPSILON)
            DKL[eNo, rowNo] = np.sum(P_row * DKL_row).squeeze()

    return DKL


def calc_kEff(P, row_idx, alpha_perc=0.02):

    kEff_arr = np.zeros(len(row_idx) - 1)

    for rowNo, [start, end] in enumerate(zip(row_idx[:-1], row_idx[1:])):
        P_row = P[start:end]

        P_row_max = P_row.max() * alpha_perc

        kEff_arr[rowNo] = np.sum(P_row > P_row_max)

    return kEff_arr


if __name__ == "__main__":

    data_name = "Diaphragm"
    DR_method = 'tSNE'
    parameter = 'Perplexity'

    print_str  = f"  Performing {DR_method}-{parameter} Sweep on"
    print_str += f" {data_name} Data!  "
    print(f"\n\n" + print_str + "\n" + "=" * len(print_str) + "\n\n")

    file_name_base = f"HyperparamSweep_{DR_method}_{parameter}_{data_name}"

    ## Set the name for the full saved output (this is what another script
    ## could load at once to skip all this junk).
    out_name = file_name_base + "_Output_Dict.pkl"
    out_path = os.path.join(embed_dir, out_name)

    ## Runtime parameters
    n_data_embed     = 5
    n_null_embed     = 10
    use_asymmetric_P = True
    pVal_method      = 'average'

    null_gen_seed    = 54321
    affinity_seed    = 12345
    embedding_seed   = 1

    n_param          = 30      ## At most, use this many parameter values
    min_param_val    = 5       ## Minimum parameter value
    max_param_val    = 10000   ## Maximum parameter value

    if DR_method.lower() == 'tsne':
        ## Set some global t-SNE parameters
        n_components     = 2
        n_ee_iter        = 500
        min_ee_iter      = 100
        exaggeration     = 12
        ee_mom           = 0.5
        n_iter           = 1000
        min_iter         = 500
        mom              = 0.8
        n_jobs           = -1
        verbose          = True

        DR_method = 'tSNE'

    ## Set the kEff threshold
    alpha_nu = 0.01

    ###########################################################################
    ## Load the data and metadata!
    ###########################################################################
    if True:

        if data_name.lower() in ['marrow', 'diaphragm']:
            data_dir = path.join(data_dir, "TabulaMuris/FACS/")

        X, _ = pUtl.load_data(data_name, data_dir)

        n_samples, n_features = X.shape
        print(f"Loaded {data_name} Data! ({n_samples} x {n_features})")

        if parameter.lower() == 'perplexity':
            ## Set perplexities to sweep
            param_arr = pUtl.generate_perp_arr(np.min([n_samples,
                                                       max_param_val + 1]),
                                               lower_bound=min_param_val,
                                               n_perp=n_param)
            # param_arr = param_arr[param_arr < ]
            param_arr = np.sort(param_arr)[::-1]
            n_param = len(param_arr)

        print(f"\nGenerating {n_null_embed} Null Data!")
        r.seed(null_gen_seed)
        null_X = np.zeros((n_null_embed, n_samples, n_features))
        for nNo in range(n_null_embed):
            null_X[nNo] = np.asarray([r.choice(col, size=n_samples)
                                      for col in X.T]).T

    ###########################################################################
    ## Load the affinity matrices!
    ###########################################################################
    if True:

        n_neib = np.min([int(n_samples - 1), int(3 * max_param_val)])

        aff_params = {"NN_alg": "pynndescent",
                      "n_neighbors": n_neib,
                      "perplexity": param_arr.max(),
                      "normalization": 'pair-wise',
                      "n_jobs": -1,
                      "random_state": affinity_seed,
                      "verbose": 5}

        ##################
        ## Working on X ##
        ##################

        print("\nGetting affinity matrix for data!")

        ## Set filenames and paths
        annoy_data_name = file_name_base + "_ANNOY_index_data.obj"
        annoy_data_path = os.path.join(embed_dir, annoy_data_name)
        aff_data_name   = file_name_base + "_base_affinity_data.pkl"
        aff_data_path   = os.path.join(embed_dir, aff_data_name)

        ## First, try and load the affinity matrix and rebuild the ANNOY index
        try:
            print(f"\nTrying to load and rebuild affinity matrix object...")

            with open(aff_data_path, 'rb') as f:
                aff_data_X = pkl.load(f)

            if n_samples > 1000:
                print(f"... affinity matrix loaded, getting kNN index...")

                kNN = aff_data_X.kNN_index._initialize_ANNOY_index(n_features)
                aff_data_X.kNN_index.indices = kNN
                aff_data_X.kNN_index.indices.load(annoy_data_path)

            print(f"... kNN index loaded successfully!")

        ## If it can't be loaded, then recompute.
        except (FileNotFoundError, OSError):
            print(f"... couldn't load!  Recalculating...")

            aff_data_X = FixedEntropyAffinity(**aff_params)
            aff_data_X.fit(X)

            if n_samples > 1000:
                aff_data_X.kNN_index.indices.save(annoy_data_path)

            with open(aff_data_path, 'wb') as f:
                pkl.dump(aff_data_X, f)

            print(f"... done!  Saved to file!")

        #######################
        ## Working on NULL X ##
        #######################

        null_affinities = {}
        for nNo in range(n_null_embed):

            print(f"\nGetting affinity matrix for null!"
                  f" {nNo + 1}/{n_null_embed}")

            ## Set filenames and paths
            annoy_null_name = file_name_base + f"_ANNOY_index_null_{nNo}.obj"
            annoy_null_path = os.path.join(embed_dir, annoy_null_name)
            aff_null_name = file_name_base + f"_base_affinity_null_{nNo}.pkl"
            aff_null_path = os.path.join(embed_dir, aff_null_name)

            ## First, try to load the affinities and rebuild the ANNOY index
            try:
                print(f"\nTrying to load / rebuild affinity matrix object...")

                with open(aff_null_path, 'rb') as f:
                    aff_null_X = pkl.load(f)

                if n_samples > 1000:
                    print(f"... affinity matrix loaded, getting kNN index...")
                    kNN = aff_null_X.kNN_index._initialize_ANNOY_index(n_features)
                    aff_null_X.kNN_index.indices = kNN
                    aff_null_X.kNN_index.indices.load(annoy_null_path)
                print(f"... kNN index loaded successfully!")

            ## If it can't be loaded, then recompute.
            except (FileNotFoundError, OSError):
                print(f"... couldn't load!  Recalculating...")

                aff_null_X = FixedEntropyAffinity(**aff_params)
                aff_null_X.fit(null_X[nNo])

                if n_samples > 1000:
                    aff_null_X.kNN_index.indices.save(annoy_null_path)

                with open(aff_null_path, 'wb') as f:
                    pkl.dump(aff_null_X, f)

                print(f"... done!  Saved to file!")

            null_affinities[nNo] = aff_null_X

    ###########################################################################
    ## Start the parameter sweep!
    ###########################################################################
    if True:

        ## Initialize storage arrays (although we will cache these regularly!)
        data_Y = {}
        null_Y = {}

        tau_arr  = np.zeros((n_param, n_samples))
        kEff_arr = np.zeros((n_param, n_samples))

        data_EES = {}
        null_EES = {}

        pValues  = np.zeros((n_param, n_samples))

        print(f"\nStarting Parameter Sweep!")

        total_sweep_time = 0
        now = time.time()
        for pNo, param in enumerate(param_arr):
            store_idx = n_param - pNo - 1

        #######################################################################
        ## Embedding DATA
        #######################################################################
            print(f"\n\nEmbedding data with {DR_method} @ {parameter} ="
                  f" {param} ({pNo + 1}/{n_param})")

            embed_data_name  = file_name_base + f"_DataEmbedding"
            embed_data_name += f"_Param{param:.0f}_RS{embedding_seed}.pkl"
            embed_data_path = os.path.join(embed_dir, embed_data_name)

            ## Generate embeddings
            dY, dAff = load_tSNE(X,
                                 precalc_kNN=aff_data_X,
                                 file_name=embed_data_name,
                                 file_path=embed_data_path,
                                 perplexity=param,
                                 random_seed=embedding_seed,
                                 n_embed=n_data_embed,
                                 affinity_params=aff_params,
                                 n_components=n_components,
                                 n_ee_iter=n_ee_iter,
                                 min_ee_iter=min_ee_iter,
                                 exaggeration=exaggeration,
                                 ee_mom=ee_mom,
                                 n_iter=n_iter,
                                 min_iter=min_iter,
                                 mom=mom,
                                 n_jobs=n_jobs,
                                 verbose=True)

            ## Extract and save the kernel widths
            taus = dAff.kernel_params['precisions'].reshape(-1, 1)
            tau_arr[store_idx] = taus.ravel().copy()

            ## Get the high-dim affinity matrix
            if use_asymmetric_P:
                dP = np.sqrt(taus) * np.exp(-dAff.distances**2 * taus / 2)

                n_neib = dAff.n_neighbors
                row_idx = np.arange(0, n_neib * n_samples + 1, n_neib)

                dP = sp.csr_matrix((dP.ravel(), dAff.indices.ravel(), row_idx))

            else:
                dP = dAff.P.copy()

            ## Row-normalize the high-dimensional affinity matrix
            normalize(dP, norm='l1', axis=1, copy=False)

            print(f"\nCalculating EES for data embeddings!")
            dEES = calc_DKL(dP.data, dP.indptr, dP.indices, dY)

            print(f"\nCalculating kEff!")
            kEff_arr[store_idx] = calc_kEff(dP.data, dP.indptr,
                                            alpha_perc=alpha_nu)

        #######################################################################
        ## Embedding NULL
        #######################################################################
            print(f"\n\nEmbedding null with {DR_method} @ {parameter} ="
                  f" {param} ({pNo + 1}/{n_param})")

            nY = np.zeros((n_null_embed, n_samples, n_components))
            nEES = np.zeros((n_null_embed, n_samples))
            for nNo in range(n_null_embed):

                print(f"\nEmbedding null {nNo + 1} / {n_null_embed}")

                embed_null_name  = file_name_base + f"_NullEmbedding{nNo}"
                embed_null_name += f"_Param{param:.0f}_RS{embedding_seed}.pkl"
                embed_null_path = os.path.join(embed_dir, embed_null_name)

                null_seed = embedding_seed + nNo

                ## Generate embeddings
                nY[nNo], nAff = load_tSNE(null_X[nNo],
                                          precalc_kNN=null_affinities[nNo],
                                          file_name=embed_null_name,
                                          file_path=embed_null_path,
                                          perplexity=param,
                                          random_seed=null_seed,
                                          n_embed=1,
                                          affinity_params=aff_params,
                                          n_components=n_components,
                                          n_ee_iter=n_ee_iter,
                                          min_ee_iter=min_ee_iter,
                                          exaggeration=exaggeration,
                                          ee_mom=ee_mom,
                                          n_iter=n_iter,
                                          min_iter=min_iter,
                                          mom=mom,
                                          n_jobs=n_jobs,
                                          verbose=True)

                ## Extract and save the kernel widths
                taus = nAff.kernel_params['precisions'].reshape(-1, 1)

                ## Get the high-dim affinity matrix
                if use_asymmetric_P:
                    nP = np.sqrt(taus) * np.exp(-nAff.distances**2 * taus / 2)

                    n_neib = nAff.n_neighbors
                    row_idx = np.arange(0, n_neib * n_samples + 1, n_neib)

                    nP = sp.csr_matrix((nP.ravel(),
                                        nAff.indices.ravel(),
                                        row_idx))
                else:
                    nP = nAff.P.copy()

                ## Row-normalize the high-dimensional affinity matrix
                normalize(nP, norm='l1', axis=1, copy=False)

                print(f"\nCalculating EES for null embeddings!")
                nEES[nNo] = calc_DKL(nP.data, nP.indptr, nP.indices, nY[nNo])

            print(f"\nCalculating p-Values!")
            [_, pValues[store_idx]
             ] = pUtl.calc_emp_pVals(dEES, nEES, summary_method=pVal_method)

            data_Y[param] = dY[:]
            null_Y[param] = nY[:]

            data_EES[param] = dEES[:]
            null_EES[param] = nEES[:]

            param_duration = time.time() - now
            now = time.time()
            total_sweep_time += param_duration
            print(f"\nIteration took {param_duration:.2f} seconds"
                  f" (In total: {total_sweep_time:.2f} seconds or"
                  f" {total_sweep_time/3600.:.2f} hours)")

        out_dict = {'data_Y': data_Y.copy(),
                    'null_Y': null_Y.copy(),
                    'data_EES': data_EES.copy(),
                    'null_EES': null_EES.copy(),
                    'kEff_arr': kEff_arr.copy(),
                    'tau_arr': tau_arr.copy(),
                    'pVals': pValues.copy()}

        print(f"\nSaving out_dict! ({out_path})")
        with open(out_path, 'wb') as f:
            pkl.dump(out_dict, f)

        del out_dict







