"""
###############################################################################
    t-Distributed Stochastic Neighborhood Embedding (t-SNE) Implementation
###############################################################################

    Author: Eric Johnson
    Date Created: Wednesday, July 6, 2021
    Email: ericjohnson1.2015@u.northwestern.edu

###############################################################################

    In this file, I want to define a class for implementing the t-SNE algorithm
    for the dimensionality reduction of a high-dimensional dataset into 1 or 2
    dimensions.

    Unlike other implementations, here we will attempt to streamline some
    important details that will make the algorithm more appropriate for the
    EMBEDR algorithm; namely, we eliminate the calculation of the affinity
    matrix and we make the method iterable to make several embeddings.  By
    default we will use the FIt-SNE implementation by Linderman et al (2019).

    Some parts of this code are adapted from the openTSNE package by Pavlin
    Poličar under the BSD 3-Clause License.

###############################################################################
"""
from collections import Iterable

import EMBEDR.affinity as aff
import EMBEDR.callbacks as cb
import EMBEDR.initialization as initial
from EMBEDR import _tsne
from EMBEDR.quad_tree import QuadTree
import EMBEDR.utility as utl

import logging
import multiprocessing
import numpy as np
from sklearn.utils import check_array, check_random_state
from time import time

EPSILON = np.finfo(np.float64).eps

log = logging.getLogger(__name__)


class OptimizationInterrupt(InterruptedError):
    """Error indicating that the optimization was interrupted by a callback.

    Attributes
    ----------
    error: float
        The "error" of the current embedding.  In the case of t-SNE, this is
        the Kullback-Liebler Divergence between the high dimensional data and
        the lower-dimensional embedding.

    final_embedding: np.ndarray
        The final embedding generated by the optimization before interruptions.
    """

    def __init__(self, error, final_embedding):
        super().__init__()
        self.error = error
        self.final_embedding = final_embedding


class tSNE_Embed:

    def __init__(self,
                 n_components=2,            ## Either 1 or 2 **
                 perplexity=None,           ## Float > 0, only used if no aff
                                            ## ** keep track of affmat used!?
                 initialization='random',   ## Either 'random' or 'pca' **
                 learning_rate=None,        ## Float > 1, autoset if None **
                 n_early_iter=250,          ## Int < n_iter **
                 early_exag=12,             ## Float > 0, warning if small
                                            ## ** If cbs, keep track of nIter
                 early_mom=0.5,             ## Float > 0, warning if big **
                 n_iter=1000,               ## Int > 0
                                            ## ** If cbs, keep track of nIter
                 exag=1.,                   ## Float > 0, warning if != 1 **
                 mom=0.8,                   ## Float > 0, warning if big **
                 max_grad_norm=None,        ## Float > 0, default is None **
                 max_step_norm=5,           ## Float > 0 **
                 t_dof=1.,                  ## Float > 0, warning if !=1 **
                 neg_grad_method="fft",     ## 'fft' or 'bh' **
                 bh_theta=0.5,              ## 0 < float < 1 **
                 FI_n_interp_pts=3,         ## Int >= 1 **
                 FI_min_n_interv=50,        ## Int >= 1 **
                 FI_ints_per_interv=1,      ## Int >= 1 **
                 callbacks=None,            ## ** track cbs, warn if different
                 propagate_interrupt=False,  ## Bool
                 iter_per_callback=10,      ## Int > 0
                 iter_per_log=50,           ## Int > 0
                 n_jobs=1,                  ## Int > 0
                 random_state=None,         ## Check method already exists
                 verbose=1):                ## Float

        self.n_components = int(n_components)

        if isinstance(initialization, np.ndarray):
            self.initialization = check_array(initialization,
                                              accept_sparse=False,
                                              ensure_2d=True)
        elif isinstance(initialization, str):
            assert initialization.lower() in ['random', 'pca', 'spectral']
            self.initialization = initialization.lower()
        else:
            err_str =  f"`initialization` must be either 'random', 'pca',"
            err_str += f" 'spectral', or an array of shape `n_samples` x"
            err_str += f" `n_components`."
            raise ValueError(err_str)

        self.perplexity = perplexity

        self.learning_rate = learning_rate

        self.n_early_iter = int(n_early_iter)
        self.early_exag = early_exag
        assert early_mom > 0
        self.early_mom = float(early_mom)

        self.n_iter = int(n_iter)
        assert exag > 0
        self.exag = float(exag)
        assert mom > 0
        self.mom = float(mom)

        self.max_grad_norm = max_grad_norm
        self._gradient = None
        self.max_step_norm = max_step_norm
        self._gains = None

        assert t_dof > 0
        self.t_dof = float(t_dof)

        self.neg_grad_method = neg_grad_method

        assert (bh_theta >= 0) and (bh_theta <= 1)
        self.bh_theta = float(bh_theta)

        self.FI_n_interp_pts = int(FI_n_interp_pts)
        self.FI_min_n_interv = int(FI_min_n_interv)
        self.FI_ints_per_interv = int(FI_ints_per_interv)

        if callbacks is None:
            callbacks = dict(early_exag=cb.QuitEarlyExaggerationPhase(),
                             no_exag=cb.QuitNoExaggerationPhase())
        err_str = f"All elements of `callbacks` must be callable objects!"
        for cb_type in ['early_exag', 'no_exag']:
            if isinstance(callbacks[cb_type], Iterable):
                if any(not callable(cb) for cb in callbacks[cb_type]):
                    raise ValueError(err_str)

            elif callable(callbacks[cb_type]):
                callbacks[cb_type] = (callbacks[cb_type],)

            else:
                raise ValueError(err_str)

        self.callbacks = callbacks
        self.propagate_interrupt = bool(propagate_interrupt)
        self.iter_per_callback = iter_per_callback
        self.iter_per_log = iter_per_log

        self._errors = []

        ## Check the number of jobs compared to the number of cores available.
        if n_jobs < 0:
            n_cores = multiprocessing.cpu_count()
            ## Add negative number of n_jobs to the number of cores, but
            ## increment by one because -1 indicates using all cores, -2 all
            ## except one, and so on...
            n_jobs = n_cores + n_jobs + 1
        ## If the number of jobs, after this correction is still <= 0, then the
        ## user probably thought they had more cores, so we'll default to 1.
        if n_jobs <= 0:
            log.warning(f"`n_jobs` receieved value {n_jobs} but only {n_cores}"
                        f" cores are available. Defaulting to single job.")
            n_jobs = 1
        self.n_jobs = n_jobs

        self.rs = check_random_state(random_state)
        self.verbose = float(verbose)

        if self.verbose >= 1:
            print(f"\nInitialized EMBEDR.tSNE_Embed object!")

    def fit(self, X, **aff_kwargs):

        if self.verbose >= 1:
            print(f"\nGenerating {self.n_components}-dimensional embedding"
                  f" with t-SNE!")

        P = self.initialize_embedding(X, **aff_kwargs)

        ## Optimize Early Exaggeration Phase
        try:
            self._callbacks = self.callbacks['early_exag']
            self.optimize(P,
                          self.n_early_iter,
                          self.early_exag,
                          self.early_mom)
        except OptimizationInterrupt as opt_int:
            if self.verbose >= 1:
                print(f"Optimization interrupted by callback")
            if self.propagate_interrupt:
                raise opt_int

        ## Optimize regular descent phase
        try:
            self._callbacks = self.callbacks['no_exag']
            self.optimize(P,
                          self.n_iter,
                          self.exag,
                          self.mom)
        except OptimizationInterrupt as opt_int:
            if self.verbose >= 1:
                print(f"Optimization interrupted by callback")
            if self.propagate_interrupt:
                raise opt_int

        return

    def initialize_embedding(self, X, **aff_kwargs):
        """Given input data, generate initial embedding and affinity matrix

        More specifically, if an affinity matrix has been provided, then we
        can generate some types of initial embeddings.  On the other hand, if
        "raw" data has been provided, we need to also generate an affinity
        matrix for t-SNE.

        NOTE: The properties of affinity matrices generated in this routine
        will not be stored, so it is recommended to generate the affinities
        beforehand and pass them to be fit here.  This saves computation time
        and improves reproducibility.
        """

        ## If the input data is not an affinity matrix, then we need to
        ## generate an affinity matrix.
        if not isinstance(X, aff.AffinityMatrix):
            if self.verbose > 2:
                print(f"Data to be fit (`X`) are not an affinity matrix..."
                      f" Generating P!")

            ## If PCA initialization was requested, set initial embed as PCs.
            if isinstance(self.initialization, np.ndarray):
                pass
            elif self.initialization == 'pca':
                self.initialization = initial.pca_init(X,
                                                       self.n_components,
                                                       random_state=self.rs,
                                                       verbose=self.verbose)

            ## We can then initialize an affinity matrix object
            out = aff._initialize_affinity_matrix(X,
                                                  perplexity=self.perplexity,
                                                  n_jobs=self.n_jobs,
                                                  random_state=self.rs,
                                                  verbose=self.verbose,
                                                  **aff_kwargs)
            affmat, kNN_index = out

            ## We can then fit the affinity matrix object.
            affmat.fit(kNN_index)

            ## For t-SNE, the affinity matrix must be a probability dist,
            ## even if we've previously row-normalized. (Except during the
            ## early exaggeration phase.)
            P = affmat.P / affmat.P.sum()

            ## We then get the shape of the data from the affinity matrix.
            self.n_samples = affmat.n_samples

        ## If we did supply an affinity matrix then we just need to norm P.
        else:
            if self.verbose > 2:
                print(f"Data to be fit (`X`) are an affinity matrix!")

            ## For t-SNE, the affinity matrix must be a probability dist,
            ## even if we've previously row-normalized. (Except during the
            ## early exaggeration phase.)
            P = X.P / X.P.sum()

            self.n_samples = X.n_samples

        ## If an array was provided, ensure that it has the correct shape.
        if isinstance(self.initialization, np.ndarray):
            Y = check_array(self.initialization,
                            accept_sparse=False,
                            ensure_2d=True)

            ## Check that the initial embedding has the correct shape.
            err_str =  f"Invalid shape for initialization array. Expected"
            err_str += f" ({self.n_samples}, {self.n_components}), got"
            err_str += f" {Y.shape}..."
            assert Y.shape == (self.n_samples, self.n_components), err_str

            ## Check that the spread of the initial embed is not too large...
            if np.any(np.std(Y, axis=0) > 1.e-2):
                log.warning(f"Variance of initial embedding component is >"
                            f"1.e-4. Initial embeddings with high variance"
                            f" may converge slowly or not at all! (Try "
                            f"EMBEDR.initialization.rescale.)")

            self.embedding = Y.copy()

        ## If we've requested random intialization, do that here.
        elif self.initialization == 'random':
            self.embedding = initial.random_init(P,
                                                 self.n_components,
                                                 random_state=self.rs,
                                                 verbose=self.verbose)

        ## If we want to do PCA initialization, it requires raw data and not
        ## an affinity matrix.
        elif self.initialization == 'pca':
            err_str  = f"Cannot generate PCA initialization because input data"
            err_str += f", `X`, is an affinity matrix, not a samples x"
            err_str += f" features matrix."
            raise ValueError(err_str)

        ## If we wanted a spectral initialization, do that here.
        elif self.initialization == 'spectral':
            self.embedding = initial.spectral_init(P,
                                                   self.n_components,
                                                   random_state=self.rs,
                                                   verbose=self.verbose)

        ## If an array was provided, ensure that it has the correct shape.
        elif isinstance(self.initialization, np.ndarray):
            Y = check_array(self.initialization,
                            accept_sparse=False,
                            ensure_2d=True)

            ## Check that the initial embedding has the correct shape.
            err_str =  f"Invalid shape for initialization array. Expected"
            err_str += f" ({self.n_samples}, {self.n_components}), got"
            err_str += f" {Y.shape}..."
            assert Y.shape == (self.n_samples, self.n_components), err_str

            ## Check that the spread of the initial embed is not too large...
            if np.any(np.std(Y, axis=0) > 1.e-2):
                log.warning(f"Variance of initial embedding component is >"
                            f"1.e-4. Initial embeddings with high variance"
                            f" may converge slowly or not at all! (Try "
                            f"EMBEDR.initialization.rescale.)")

            self.embedding = Y.copy()

        ## Set the learning rate if it isn't supplied.
        if self.learning_rate is None:
            self.learning_rate = max(200, self.n_samples / 12.)

        return P

    def optimize(self,
                 P,
                 n_iter,
                 exaggeration,
                 momentum):

        ## If there is any exaggeration, modify `P`.
        if exaggeration != 1:
            P *= exaggeration

        ## Initialize the gradient and gains arrays.
        self._gradient = np.zeros_like(self.embedding,
                                       dtype=np.float64,
                                       order='C')
        if self._gains is None:
            self._gains = np.ones_like(self.embedding,
                                       dtype=np.float64,
                                       order='C')

        ## Initialize the update array to look like the gradient.
        update = np.zeros_like(self._gradient)

        ## Callbacks can have an initialization method, call that here.
        do_callbacks = True
        if isinstance(self._callbacks, Iterable):
            for cb in self._callbacks:
                getattr(cb, "optimization_about_to_start", lambda: ...)()
        else:
            do_callbacks = False

        ## Start the timer if we're worried about printing information.
        if self.verbose >= 1:
            timer_str  = f"Fitting t-SNE for up to {n_iter} iterations with"
            timer_str += f" exaggeration = {exaggeration:.1f} and learning"
            timer_str += f" rate = {self.learning_rate:.1f}."
            timer = utl.Timer(timer_str)
            timer.__enter__()

        start_time = time()

        ## START THE LOOP!
        for ii in range(n_iter):

            ## Determine whether to do callbacks in this iteration.
            do_cbs_now = do_callbacks and \
                ((ii + 1) % self.iter_per_callback == 0)
            ## Determine whether to calculate the error (D_KL) in this iter.
            calc_error = do_cbs_now or ((ii + 1) % self.iter_per_log == 0)

            ## Calculate the gradient and error
            if self.neg_grad_method.lower() in ['bh', 'barnes-hut']:
                d_kl = self._fit_bh(P, return_DKL=calc_error)

            elif self.neg_grad_method.lower() in ['fft', 'fit-sne', 'fitsne']:
                d_kl = self._fit_fft(P, return_DKL=calc_error)

            else:
                err_str  = f"Currently, only Barnes-Hut and FIt-SNE methods"
                err_str += f" are supported for calculating t-SNE gradients."
                err_str += f" (`neg_grad_method` = 'BH' or 'FIt-SNE'.)"
                raise ValueError(err_str)

            ## If we are applying exaggeration, adjust the error (D_KL).
            if calc_error:
                if exaggeration != 1:
                    d_kl = d_kl / exaggeration - np.log(exaggeration)

                self._errors.append([ii, d_kl])

            ## To avoid samples flying off, we clip the gradient.
            if self.max_grad_norm is not None:
                norm  = np.linalg.norm(self._gradient, axis=1)
                coeff = self.max_grad_norm / (norm + 1.e-6)
                mask  = coeff < 1  ## Anywhere that the norm > max_grad_norm...
                self._gradient[mask] *= coeff[mask, None]

            ## If it's a callback iteration...
            if do_cbs_now:

                ## Do all the callbacks.

                cb_out = np.array([cb(ii + 1, d_kl, self.embedding)
                                   for cb in self._callbacks]).astype(bool)

                ## If any of the callbacks say to stop...
                if np.any(cb_out):
                    ## ... fix the affinity matrix...
                    if exaggeration != 1:
                        P /= exaggeration
                    ## ... and quit the loop!
                    raise OptimizationInterrupt(error=d_kl,
                                                final_embedding=self.embedding)

            ## Get where the last update and current gradient have diff signs
            grad_dir_flip = np.sign(update) != np.sign(self._gradient)
            grad_dir_same = np.invert(grad_dir_flip)
            self._gains[grad_dir_flip] += 0.2
            self._gains[grad_dir_same]  = self._gains[grad_dir_same] * 0.8
            self._gains[grad_dir_same] += 0.01  ## Minimum gain

            ## Get the update
            update  = momentum * update
            update -= self.learning_rate * self._gains * self._gradient

            ## To avoid samples flying off, we clip the update.
            if self.max_step_norm is not None:
                update_norm  = np.linalg.norm(update, axis=1, keepdims=True)
                mask = update_norm.squeeze() > self.max_step_norm
                update[mask] /= update_norm[mask]
                update[mask] *= self.max_step_norm

            ## Update the embedding!
            self.embedding += update

            ## Recenter the embedding
            self.embedding -= np.mean(self.embedding, axis=0)

            ## Display progress!
            if (self.verbose >= 1) and ((ii + 1) % self.iter_per_log == 0):
                stop_time = time()
                dt = stop_time - start_time
                print(f"Itr {ii + 1:4d}, DKL {d_kl:6.4f},\t"
                      f"{self.iter_per_log} iterations in {dt:.4f} sec")
                start_time = time()

        if self.verbose >= 1:
            timer.__exit__()

        ## Before returning, fix the affinity matrix for future optimizations.
        if exaggeration != 1:
            P /= exaggeration

        ## We also need to calculate the error one more time for the last loop.
        d_kl = self._fit_bh(P, return_DKL=True)

        self._errors.append([ii, d_kl])

        return

    def _fit_bh(self,
                P,
                return_DKL=False,
                pairwise_normalization=True):

        ## Make sure the gradient is initialized to zero
        self._gradient[:] = 0

        ## Fit the quadtree representation of the embedding
        qtree = QuadTree(self.embedding)

        ## Calculate the negative gradient
        sum_Qi = _tsne.estimate_negative_gradient_bh(
            qtree,
            self.embedding,
            self._gradient,
            theta=self.bh_theta,
            dof=self.t_dof,
            num_threads=self.n_jobs,
            pairwise_normalization=pairwise_normalization,
        )
        sum_Q = np.sum(sum_Qi)

        # print(sum_Qi[:])
        # print(sum_Q)

        ## Delete the quadtree
        del qtree

        # Calculate the positive gradient
        sum_P, d_kl = _tsne.estimate_positive_gradient_nn(
            P.indices,
            P.indptr,
            P.data,
            self.embedding,
            self._gradient,
            t_dof=self.t_dof,
            num_threads=self.n_jobs,
            calc_error=return_DKL,
        )

        ## The positive gradient uses unnormalized qij, so we need to
        ## normalize, but only if we're calculating the DKL (error)
        if return_DKL:
            d_kl += sum_P * np.log(sum_Q + EPSILON)

        return d_kl

    def _fit_fft(self,
                 P,
                 return_DKL=False):

        self._gradient[:] = 0

        ## If the embedding is 1D:
        if self.n_components == 1:
            sum_Q = _tsne.estimate_negative_gradient_fft_1d(
                self.embedding.ravel(),
                self._gradient.ravel(),
                n_interpolation_points=self.FI_n_interp_pts,
                min_num_intervals=self.FI_min_n_interv,
                ints_in_interval=self.FI_ints_per_interv,
                dof=self.t_dof,
            )
        elif self.n_components == 2:
            sum_Q = _tsne.estimate_negative_gradient_fft_2d(
                self.embedding,
                self._gradient,
                n_interpolation_points=self.FI_n_interp_pts,
                min_num_intervals=self.FI_min_n_interv,
                ints_in_interval=self.FI_ints_per_interv,
                dof=self.t_dof,
            )
        else:
            err_str  = f"Interpolation-based t-SNE (FIt-SNE) is not supported"
            err_str += f" for more than 2 dimensions."
            raise RuntimeError(err_str)

        ## Compute positive gradient.
        sum_P, d_kl = _tsne.estimate_positive_gradient_nn(
            P.indices,
            P.indptr,
            P.data,
            self.embedding,
            self._gradient,
            t_dof=self.t_dof,
            num_threads=self.n_jobs,
            calc_error=return_DKL,
        )

        ## The positive gradient uses unnormalized qij, so we need to
        ## normalize, but only if we're calculating the DKL (error)
        if return_DKL:
            d_kl += sum_P * np.log(sum_Q + EPSILON)

        return d_kl


