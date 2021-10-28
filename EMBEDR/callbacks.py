"""
###############################################################################
    Gradient Descent Callbacks
###############################################################################

    Author: Eric Johnson
    Date Created: Wednesday, October 20, 2021
    Email: ericjohnson1.2015@u.northwestern.edu

###############################################################################

    In this file, I want to implement a few callback functions that will allow
    for us to monitor and interact with the t-SNE gradient descent without
    introducing a bunch of overhead.  In particular, I want to use the Belkina
    et al. (2019) methods for stopping the early exaggeration and descent
    phases of t-SNE and introduce a method to track the t-SNE positions at
    each step of the descent for animation.

    The structure of this code is adapted from the openTSNE package by Pavlin
    Poličar under the BSD 3-Clause License.

###############################################################################
"""

import logging
import time
import warnings
from functools import partial

import numpy as np
import scipy.sparse as sp

class Callback(object):
    def optimization_about_to_start(self):
        """This is called at the beginning of the optimization procedure."""

    def __call__(self, iteration, error, embedding):
        """This is the main method called from the optimization.

        Parameters
        ----------
        iteration: int
            The current iteration number.

        error: float
            The current KL divergence of the given embedding.

        embedding: TSNEEmbedding
            The current t-SNE embedding.

        Returns
        -------
        stop_optimization: bool
            If this value is set to ``True``, the optimization will be
            interrupted.
        """

class QuitEarlyExaggerationPhase(Callback):
    """Use the Belkina et al. (2019) method from 2019 to stop EE

    Specifically, track the relative change in DKL vs iteration and quit EE
    when the relative DKL reaches a peak. (Quit as soon as the relative change
    in starts decreasing.)
    """

    def __init__(self,
                 max_iter=500,
                 min_iter=100,
                 verbose=False):
        self.iter_count = 0
        self.last_log_time = None
        self.init_DKL = -1
        self.DKL = -1
        self.max_rel_err = 0
        self.max_iter = int(max_iter)
        self.min_iter = int(min_iter)
        self.verbose  = bool(verbose)

    def optimization_about_to_start(self):
        self.last_log_time = time.time()
        self.iter_count = 0
        self.init_DKL = -1
        self.DKL = -1
        self.max_rel_err = 0

    def __call__(self, iteration, error, embedding):

        ## Get the current time and the time since the last callback.
        now = time.time()
        duration = now - self.last_log_time
        self.last_log_time = now

        ## Store the iteration number.
        self.iter_count = iteration

        ## Calculate the relative change in error
        if self.DKL >= 0:
            rel_err = (self.DKL - error) / self.DKL
        else:
            rel_err = 0
            self.init_DKL = error

        self.DKL = error

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
            if np.abs(self.DKL - self.init_DKL) < 1.e-6:
                return True

class QuitNoExaggerationPhase(Callback):
    """Use the Belkina et al. (2019) method from 2019 to stop t-SNE

    Specifically, track the relative change in DKL vs iteration and quit the
    gradient descent when the relative changes in DKL are less than a
    specified tolerance.
    """
    def __init__(self, rel_err_tol=1.e-6, min_iter=250, verbose=False):
        self.iter_count = 0
        self.last_log_time = None
        self.init_DKL = -1.0
        self.DKL = -1.0
        self.rel_err_tol = rel_err_tol
        self.verbose = verbose
        self.min_iter = min_iter

    def optimization_about_to_start(self):
        self.last_log_time = time.time()
        self.iter_count = 0
        self.init_DKL = -1.0
        self.DKL = -1.0

    def __call__(self, iteration, error, embedding):
        now = time.time()
        duration = now - self.last_log_time
        self.last_log_time = now

        self.iter_count = iteration

        ## Calculate the relative change in error
        if self.DKL < 0:
            self.init_DKL = error
            rel_err = np.inf
        else:
            rel_err = (self.DKL - error) / error

        self.DKL = error

        if self.verbose:
            print(f"Iteration {iteration:04d}: D_KL = {error:6.4f}"
                  f" (Rel Change = {rel_err:6.2%} in {duration:.4f} sec)")

        ## If enough iterations have elapsed...
        if self.iter_count > self.min_iter:
            ## If there has been *any* change in the DKL...
            if np.abs(self.DKL - self.init_DKL) > 1.e-6:
                ## If the relative error decreased below tolerance, quit!
                if rel_err < self.rel_err_tol:
                    return True
