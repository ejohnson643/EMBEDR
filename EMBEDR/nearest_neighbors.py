"""
###############################################################################
    k Nearest Neighbors Calculators
###############################################################################

    Author: Eric Johnson
    Date Created: Thursday, July 2, 2020
    Date Edited: Thursday, June 3, 2021
    Email: ericjohnson1.2015@u.northwestern.edu

###############################################################################

    Adapted from the openTSNE package by Pavlin Poličar under the BSD 3-Clause
    License.

    In this file, a k-Nearest Neighbors class is defined.  This class gives a
    structure to the process of finding and indexing the nearest neighbors to
    samples in a data set according to given metrics.  This class is subclassed
    to use 3 kNN algorithms:

    1. BallTree, which uses a Ball-Tree algorithm (Slow, but more accurate)
    2. Annoy, which uses the Spotify ANNOY algorithm (fast, but approximate)
    3. NNDescent, which uses Leland McInnes' pynndescent algorithm

    Each of these objects is initialized with a distance metric and some
    runtime parameters (verbosity, number of jobs, etc.) and then the `build`
    method is used to build the nearest neighbors index.  New points can be
    queried relative to this index using the `query` method.

###############################################################################
"""

import numpy as np
import os
import scipy.sparse as sp
from sklearn import neighbors
from sklearn.exceptions import NotFittedError
from sklearn.utils import check_array, check_random_state
import warnings

import EMBEDR.utility as utl


class kNNIndex(object):
    """Find the k nearest neighbors between samples in a data set

    Parameters
    ----------
    metric: Union[str, Callable]
        Metric used to calculate distances between samples in the data set.

    metric_params: dict (optional, default=None)
        If needed, parameters for the supplied distance metric, `metric`.
        Default is `None`.

    n_jobs: int (optional, default=1)
        Number of processors to use when finding the nearest neighbors.  This
        follows the scikit-learn convention where ``-1`` means to use all
        processors, ``-2`` means to use all but one, etc.

    random_state: Union[int, RandomState] (optional, default=None)
        State for the random number generator. If `random_state` is an
        integer, then the input is interpreted as a seed for the RNG.  A
        RandomState instance can also be provided, which will then be used as
        the RNG.  Default is `None`, which uses the `RandomState` provided by
        np.random.

    verbose: numeric (optional, default=0)
        Flag indicating the level of verbosity to use in the outputs.  Setting
        to -1 (or smaller) will suppress all output.

    Attributes
    ----------
    indices: nearest neighbors object
        Depending on the kNN algorithm, different objects for storing and
        querying the nearest neighbor indices of samples in a data set.
    """

    VALID_METRICS = []  ## Different kNN methods permit different metrics.

    def __init__(self,
                 metric,
                 metric_params=None,
                 n_jobs=1,
                 random_state=None,
                 verbose=0,
                 **kwargs):

        self.metric = self._check_metric(metric)
        if metric_params is not None:
            self.metric_params = metric_params.copy()
        else:
            self.metric_params = None
        self.n_jobs = int(n_jobs)
        self.random_state = check_random_state(random_state)
        self.verbose = float(verbose)

    def _check_metric(self, metric):
        """Check that the given metric is supported by the kNNIndex instance"""
        if callable(metric):
            pass

        elif metric not in self.VALID_METRICS:
            err_str  = f"`{self.__class__.__name__} does not support the"
            err_str += f" `{metric}` metric. Please use one of the following"
            err_str += f" metrics:\n\t" + "\n\t".join(self.VALID_METRICS)
            raise ValueError(err_str)

        return metric

    def fit(self, X, k_NN):
        """Build the k-nearest neighbor graph on the input data.

        Finds and returns the indices of the k-nearest neighbors for each
        sample in the input data.

        Parameters
        ----------
        X: array_like
            Input data.  (n_samples x n_features)
        k_NN: integer
            Number of nearest neighbors to find for each sample in `X`. Must
            be smaller than n_samples, e.g. k_NN <= n_samples - 1.

        Returns:
        --------
        NN_idx: np.ndarray
            The (row) indices of the k nearest neighbors for each sample in
            `X`. This matrix will be size (n_samples x k_NN).
        distances: np.ndarray
            Distance to k-nearest neighbors for each sample in `X`. This
            matrix will be size (n_samples x k_NN).
        """

    def query(self, query, k_NN):
        """Find the k-nearest neighbors to queried data in original data, `X`.

        Using a previously built kNN graph, finds the nearest neighbors to a
        set of query data.  If the kNN graph has not been built, raises an
        error.

        Parameters
        ----------
        query: array_like
            Query data whose neighbors are to be located in the original data.
        k_NN: integer
            Number of nearest neighbors to find for each sample in `query`.
            Must be smaller than n_samples, e.g. k_NN <= n_samples - 1, where
            n_samples is the number of samples in the original NN graph.

        Returns:
        --------
        NN_idx: np.ndarray
            The (row) indices of k-nearest neighbors in the original data for
            each sample in `query`. This matrix will be size (n_samples x
            k_NN).
        distances: np.ndarray
            Distance to k-nearest neighbors in the original data for each
            sample in `query`. This matrix will be size (n_samples x k_NN).
        """

    def _check_k(self, k_NN, n_samples):
        warn_str =  f"`k_NN` = {k_NN} is too large! Must be less than"
        warn_str += f"`n_samples` = {n_samples}. Resetting to {n_samples - 1}."
        if k_NN < n_samples:
            return k_NN
        else:
            warnings.warn(warn_str)
            return n_samples - 1


class BallTree(kNNIndex):
    """Use the scikit-learn.neighbors Ball-Tree algorithm to find NNs.

    This is also known as the ``exact'' algorithm in the context of this
    package.

    Parameters
    ----------
    metric: Union[str, Callable]
        Metric used to calculate distances between samples in the data set.

    metric_params: dict (optional, default=None)
        If needed, parameters for the supplied distance metric, `metric`.
        Default is `None`.

    n_jobs: int (optional, default=1)
        Number of processors to use when finding the nearest neighbors.  This
        follows the scikit-learn convention where ``-1`` means to use all
        processors, ``-2`` means to use all but one, etc.  Default is 1.

    verbose: numeric (optional, default=0)
        Flag indicating the level of verbosity to use in the outputs.  Setting
        to -1 (or smaller) will suppress all output.

    Attributes
    ----------
    indices: sklearn.neighbors.unsupervised.NearestNeighbors
        This is sklearn's NN object which efficiently stores and queries NNs.

    """

    VALID_METRICS = neighbors.BallTree.valid_metrics

    def __init__(self,
                 metric,
                 metric_params=None,
                 n_jobs=1,
                 random_state=None,
                 verbose=0,
                 **kwargs):

        ## BallTree is a deterministic algorithm, so we always set the
        ## random_state to None.
        super().__init__(metric, metric_params, n_jobs, None, verbose)

        ## Initialize the sklearn NN object.
        self.index = neighbors.NearestNeighbors(
            algorithm='ball_tree',
            metric=self.metric,
            metric_params=self.metric_params,
            n_jobs=self.n_jobs
        )

    def fit(self, X, k_NN):

        if self.verbose:
            timer_str =  f"Finding {k_NN} nearest neighbors using an exact"
            timer_str += f" search and the {self.metric} metric..."
            timer = utl.Timer(timer_str, verbose=self.verbose)
            timer.__enter__()

        ## Get the data shape
        self.n_samples, self.n_features = X.shape[0], X.shape[1]

        ## Check k_NN
        k_NN = self._check_k(k_NN, self.n_samples)

        ## "Fit" the indices of the kNN to data
        self.index.fit(X)

        ## Return the indices and distances of the k_NN nearest neighbors.
        distances, NN_idx = self.index.kneighbors(n_neighbors=k_NN)

        if self.verbose:
            timer.__exit__()

        self.kNN_idx = NN_idx[:, :]
        self.kNN_dst = distances[:, :]

        ## Return the indices of the nearest neighbors and the distances
        ## to those neighbors.
        return self.kNN_idx.copy(), self.kNN_dst.copy()

    def query(self, query, k_NN):

        ## Check if the index has already been built
        try:
            _ = self.index.kneighbors()
        except NotFittedError as NFE:
            err_str =  f"Cannot query the kNN graph because it has not been"
            err_str += f" built! (Run kNNIndex.fit first!)"
            NFE.args[0] = err_str + "\n\n" + NFE.args[0]
            raise NFE

        if self.verbose:
            timer_str =  f"Finding {k_NN} nearest neighbors in an existing kNN"
            timer_str += f" graph using an exact search and the {self.metric}"
            timer_str += f" metric..."
            timer = utl.Timer(timer_str, verbose=self.verbose)
            timer.__enter__()

        ## Find the indices and distances to the nearest neighbors of the
        ## queried points
        distances, NN_idx = self.index.kneighbors(query, n_neighbors=k_NN)

        ## Stop the watch
        if self.verbose:
            timer.__exit__()

        ## Return the indices of the nearest neighbors to the queried points
        ## *in the original graph* and the distances to those points.
        return NN_idx[:, :k_NN], distances[:, :k_NN]


class Annoy(kNNIndex):
    """Use the ANNOY approximate nearest neighbor algorithm to find NNs.

    Parameters
    ----------
    metric: Union[str, Callable]
        Metric used to calculate distances between samples in the data set.

    metric_params: dict (optional, default=None)
        If needed, parameters for the supplied distance metric, `metric`.
        Default is `None`.

    n_jobs: int (optional, default=1)
        Number of processors to use when finding the nearest neighbors.  This
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

    n_trees: int (optional, default=50)
        Number of randomly-projected trees to use when building the ANN graph.
        As indicated in the ANNOY documentation, this parameter determines the
        rough accuracy of the generated graph, where more trees will be more
        accurate, but will take longer to build and occupy more memory.

    Attributes
    ----------
    indices: annoy.ANNOY object.
        This is the ANNOY library's NN object which efficiently stores and
        queries NNs.

    Notes
    -----
    Annoy objects don't support pickling, (see Issue #367 on the Annoy Github)
    so we offer a workaround for now.  Specifically, an index will be saved
    to the cwd if you try to simply pickle this object.  If you want to save
    the index to specific location, set the `pickle_name` attribute after
    instantiation.
    """

    VALID_METRICS = [
        "angular", "cosine",
        "dot",
        "euclidean", "l2",
        "hamming",
        "manhattan", "l1", "taxicab"
    ]

    def __init__(self,
                 metric,
                 metric_params=None,
                 n_jobs=1,
                 random_state=None,
                 verbose=0,
                 ## Subclass-specific keywords
                 n_trees=50):

        super().__init__(metric,
                         metric_params,
                         n_jobs,
                         random_state,
                         verbose)

        self.n_trees = n_trees

        ## Set the name of the metric to the name that the Annoy module likes.
        annoy_metric = self.metric
        annoy_aliases = {
            "cosine": "angular",
            "l1": "manhattan",
            "l2": "euclidean",
            "taxicab": "manhattan",
        }
        if annoy_metric in annoy_aliases:
            annoy_metric = annoy_aliases[annoy_metric]
            self.metric = annoy_metric

    def fit(self, X, k_NN):

        if self.verbose:
            timer_str =  f"Finding {k_NN} nearest neighbors using an"
            timer_str += f" approximate search and the {self.metric} metric..."
            timer = utl.Timer(timer_str, verbose=self.verbose)
            timer.__enter__()

        X = check_array(X, accept_sparse=True, ensure_2d=True)

        ## Get the data shape
        self.n_samples, self.n_features = X.shape[0], X.shape[1]

        ## Initialize the tree.
        self.index = self._initialize_ANNOY_index(self.n_features)

        ## Set the random seed.
        ## ANNOY uses only a 32-bit integer as a random seed.
        seed = self.random_state.get_state()[1][0] % (2**31)
        self.index.set_seed(seed)

        ## Add the data to the tree
        for ii in range(self.n_samples):
            self.index.add_item(ii, X[ii])

        ## Build the requested number of trees.  Default is 50.
        self.index.build(self.n_trees)

        ## Initialize output: NN indices and distances
        NN_idx = np.zeros((self.n_samples, k_NN)).astype(int)
        distances = np.zeros((self.n_samples, k_NN))

        ## Define helper function to get NNs
        def getnns(ii):
            ## Annoy returns the query point as the first element, so we ask
            ## for k_NN + 1 neighbors.
            [aa, bb] = self.index.get_nns_by_item(ii, k_NN + 1,
                                                  include_distances=True)

            ## Don't save the closest neighbor (the query point itself)
            NN_idx[ii] = aa[1:]
            distances[ii] = bb[1:]

        if self.n_jobs == 1:
            for ii in range(self.n_samples):
                getnns(ii)
        else:
            from joblib import Parallel, delayed

            Parallel(n_jobs=self.n_jobs, require="sharedmem")(
                delayed(getnns)(ii) for ii in range(self.n_samples))

        if self.verbose:
            timer.__exit__()

        self.kNN_idx = NN_idx[:, :]
        self.kNN_dst = distances[:, :]

        ## Return the indices of the nearest neighbors and the distances
        ## to those neighbors.
        return self.kNN_idx.copy(), self.kNN_dst.copy()
        # return NN_idx, distances

    def _initialize_ANNOY_index(self, f):
        ## This is only imported if we need it
        from .dependencies.annoy import AnnoyIndex
        return AnnoyIndex(f, self.metric)

    def query(self, query, k_NN):

        if self.index is None:
            err_str =  f"Cannot 'query' the kNN graph because it has not been"
            err_str += f" constructed!  (Run kNNIndex.fit(X, k_NN))"
            raise ValueError(err_str)

        if self.verbose:
            timer_str = f"Finding {k_NN} nearest neighbors to query points in"
            timer_str += f" existing kNN graph using an approximate search and"
            timer_str += f" the '{self.metric}'' metric..."
            timer = utl.Timer(timer_str, verbose=self.verbose)
            timer.__enter__()

        ## Check query shape, if 1D array, reshape.
        if query.ndim == 1:
            query = query.copy().reshape(1, -1)
        elif query.ndim != 2:
            err_str =  f"Input argument 'query' has an invalid shape: expected"
            err_str += f"2-D array, got {query.shape}."
            raise ValueError(err_str)

        ## Get number of query points.
        n_query = query.shape[0]

        ## Initialize NN indices and distances output
        NN_idx = np.zeros((n_query, k_NN)).astype(int)
        distances = np.zeros((n_query, k_NN))

        ## Define helper function to get NNs
        def getnns(ii):
            NN_idx_ii, distances_ii = self.index.get_nns_by_vector(
                query[ii], k_NN, include_distances=True)

            NN_idx[ii] = NN_idx_ii[:]
            distances[ii] = distances_ii[:]

        ## Loop over query points
        ## If only one processor...
        if self.n_jobs == 1:
            for ii in range(n_query):
                getnns(ii)
        ## If more than one processor...
        else:
            from joblib import Parallel, delayed

            Parallel(n_jobs=self.n_jobs, require='sharedmem')(
                delayed(getnns)(ii) for ii in range(n_query)
            )

        if self.verbose:
            timer.__exit__()

        return NN_idx, distances

    def __getstate__(self):

        if self.pickle_name is None:
            self.pickle_name = f"Annoy_index_{np.random.randint(1e8)}.obj"

            warn_str =  f"No file name/path specified (set the `pickle_name`"
            warn_str += f" attribute) so Annoy index will be saved to cwd as"
            warn_str += f" {self.pickle_name}."

            warnings.warn(warn_str)

        err_str = f"Annoy index could not be saved!"
        assert self.index.save(self.pickle_name), err_str

        state_dict = self.__dict__
        del state_dict['index']

        return state_dict

    def __setstate__(self, state):

        self.__dict__.update(state)

        if state['pickle_name']:
            self._load_annoy_index(state['pickle_name'])
        else:
            warn_str =  f"Annoy index's file location was not saved. The index"
            warn_str += f"will need to be rebuilt before querying the graph."

    def _load_annoy_index(self, path):
        if os.path.exists(path):
            if self.verbose >= 3:
                print(f"Loading Annoy index from {path}")

            self.index = self._initialize_ANNOY_index(self.n_features)
            self.index.load(path)

        else:
            warn_str =  f"Annoy index at {path} could not be found or loaded."
            warn_str += f" Index will need to be rebuilt before querying"
            warnings.warn(warn_str + " the graph.")

            self.index = None

        return


class NNDescent(kNNIndex):
    """Use the pynndescent approximate NN algorithm to find NNs.

    Parameters
    ----------
    metric: Union[str, Callable]
        Metric used to calculate distances between samples in the data set.

    metric_params: dict (optional, default=None)
        If needed, parameters for the supplied distance metric, `metric`.
        Default is `None`.

    n_trees: int (optional, default=None)
        (From the API guide at readthedocs) This implementation uses random
        projection forests for initializing the index build process. This
        parameter controls the number of trees in that forest. A larger number
        will result in more accurate neighbor computation at the cost of
        performance. The default of None means a value will be chosen based on
        the size of the graph_data (X).

    n_iters: int (optional, default=None)
        (From the API guide at readthedocs) The maximum number of NN-descent
        iterations to perform. The NN-descent algorithm can abort early if
        limited progress is being made, so this only controls the worst case.
        Don’t tweak this value unless you know what you’re doing. The default
        of None means a value will be chosen based on the size of the
        graph_data (X).

    n_jobs: int (optional, default=1)
        Number of processors to use when finding the nearest neighbors.  This
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

    Other keyword arguments will be passed to the pynndescent.NNDescent object
    when it is initialized.  See https://pynndescent.readthedocs.io/ for the
    complete API for this tool.

    Attributes
    ----------
    indices: pynndescent.NNDescent object.
        This is the pynndescent library's NN object which efficiently stores
        and queries NNs.
    """

    VALID_METRICS = [
        ## Minkowski-Type Distances
        "euclidean", "l2", "sqeuclidean",
        "manhattan", "taxicab", "l1",
        "chebyshev", "linfinity", "linfty", "linf",
        "minkowski",
        ## Standardized / Weighted Distances
        "seuclidean", "standardised_euclidean",
        "wminkowski", "weighted_minkowski",
        "mahalanobis",
        ## Other Distances
        "braycurtis",
        "canberra",
        "correlation",
        "cosine",
        "dot",
        "haversine",
        "hellinger",
        "kantorovich",
        "spearmanr",
        "wasserstein",
        ## Binary Distances
        "dice",
        "hamming",
        "jaccard",
        "kulsinski",
        "matching",
        "rogerstanimoto",
        "russellrao",
        "sokalmichener",
        "sokalsneath",
        "yule"]

    def __init__(self,
                 metric,
                 metric_params=None,
                 n_jobs=1,
                 random_state=None,
                 verbose=0,
                 ## Subclass-specific keywords.
                 n_trees=None,
                 n_iters=None,
                 **pynnd_kws):
        try:
            import pynndescent
        except ImportError:
            err_str =  f"Please install pynndescent: `conda install -c"
            err_str += f" conda-forge pynndescent` or `pip install"
            err_str += f" pynndescent`."
            raise ImportError(err_str)
        super().__init__(metric, metric_params, n_jobs, random_state, verbose)

        self.n_trees = n_trees
        self.n_iters = n_iters
        self.pynnd_kws = pynnd_kws

    def check_metric(self, metric):
        import pynndescent

        set_named = set(list(pynndescent.distances.named_distances))
        if not (set_named == set(self.VALID_METRICS)):
            err_str =  f"`pynndescent` has recently changed which distance"
            err_str += f" metrics are supported and this package has not been"
            err_str += f" updated. Please notify the developers."
            warnings.warn(err_str)

        if callable(metric):
            from numba.core.registry import CPUDispatcher

            if not isinstance(metric, CPUDispatcher):
                warn_str =  f"`pynndescent` requires callable metrics to be"
                warn_str += f" compiled with `numba`, but `{metric.__name__}`"
                warn_str += f" is not compiled. `EMBEDR` will attempt to"
                warn_str += f" compile the function. If this results in an"
                warn_str += f" error, then the function may not be compatible"
                warn_str += f" with `numba.njit` and should be rewritten."
                warn_str += f" Otherwise, set `NN_alg`='exact' to use"
                warn_str += f" scikit-learn to calculate nearest neighbors."
                warnings.warn(warn_str)
                from numba import njit

                metric = njit(fast_math=True)(metric)

        return super().check_metric(metric)

    def fit(self, X, k_NN):

        if self.verbose:
            timer_str =  f"Finding {k_NN} approximate nearest neighbors using"
            timer_str += f" NNDescent and the '{self.metric}' metric..."
            timer = utl.Timer(timer_str, verbose=self.verbose)
            timer.__enter__()

        ## Get the data shape
        self.n_samples, self.n_features = X.shape[0], X.shape[1]

        k_NN = self._check_k(k_NN, self.n_samples)

        ## > These values were taken from UMAP, which we assume to be sensible
        ## > defaults [because the UMAP and pynndescent authors are the same.]
        ## - Pavlin Policar
        if self.n_trees is None:
            self.n_trees = 5 + int(round((self.n_samples ** 0.5) / 20))
        if self.n_iters is None:
            self.n_iters = max(5, int(round(np.log2(self.n_samples))))

        ## If `k_NN` > 15, use just the first 15 NN to build the approximate
        ## NN graph, then use query() to the rest of the desired neighbors.
        if k_NN <= 15:
            k_build = k_NN + 1
        else:
            k_build = 15

        import pynndescent
        self.index = pynndescent.NNDescent(
            X,
            n_neighbors=k_build,
            metric=self.metric,
            metric_kwds=self.metric_params,
            random_state=self.random_state,
            n_trees=self.n_trees,
            n_iters=self.n_iters,
            n_jobs=self.n_jobs,
            verbose=self.verbose,
            **self.pynnd_kws)

        ## If k_NN <= 15, we're in the clear!
        NN_idx, distances = self.index.neighbor_graph

        ## ... Except when pynndescent fails, then it puts a -1 in the index.
        n_failures = np.sum(NN_idx == -1)

        ## If k_NN > 15, use query() to get the indices and distances
        if k_NN > 15:
            self.index.prepare()
            NN_idx, distances = self.index.query(X, k=k_NN + 1)

        ## If pynndescent fails to find neighbors for some points, raise ERROR.
        if n_failures > 0:
            err_str = "WARNING: `pynndescent` failed to find neighbors for all"
            err_str += " points in the data."

            if self.verbose >= 4:
                print_opt = np.get_print_options()
                np.set_print_options(threshold=np.inf)
                err_str += " The indices of the failed points are: "
                err_str += f"\n{np.where(np.sum(NN_idx == -1, axis=1))[0]}"
                np.set_print_options(**print_opt)
            else:
                err_str += " Set verbose >= 4 to see the indices of the"
                err_str += " failed points."

            raise ValueError(err_str)

        if self.verbose:
            timer.__exit__()

        # return NN_idx[:, 1:], distances[:, 1:]
        self.kNN_idx = NN_idx[:, 1:]
        self.kNN_dst = distances[:, 1:]

        ## Return the indices of the nearest neighbors and the distances
        ## to those neighbors.
        return self.kNN_idx.copy(), self.kNN_dst.copy()

    def query(self, query, k_NN):

        if self.index is None:
            err_str =  f"Cannot 'query' the kNN graph because it has not been"
            err_str += f" constructed!  (Run kNNIndex.fit(X, k_NN))"
            raise ValueError(err_str)

        if self.verbose:
            timer_str =  f"Finding {k_NN} approximate nearest neighbors to"
            timer_str += f" query points in the existing NN graph using"
            timer_str += f" `pynndescent` and the '{self.metric}' metric..."
            timer = utl.Timer(timer_str, verbose=self.verbose)
            timer.__enter__()

        NN_idx, distances = self.index.query(query, k=k_NN)

        if self.verbose:
            timer.__exit__()

        return NN_idx, distances


def _initialize_kNN_index(X,
                          NN_alg='auto',
                          metric='l2',
                          metric_params=None,
                          n_jobs=1,
                          random_state=None,
                          verbose=0,
                          **kwargs):
    """Initialize default kNNIndex subclass based on data.

    This method will return an appropriate kNNIndex subclass (BallTree, ANNOY,
    or NNDescent) based on the type and shape of the data, X, and related
    keywords.  In general, as long as the data are large enough and an
    appropriate metric is selected, an ANNOY object will be returned.

    Parameters
    ----------
    X: array-like
        Data used to initialize a kNNIndex object.  Depending on the size and
        type of the data, a different kNN algorithm will be initialized if
        `NN_alg`='auto'.

    NN_alg: Union[str, kNNIndex] (optional, default='auto')
        Name of NN algorithm to use.  If 'auto' is set, the preferred algorithm
        will be set based on X.  If len(X) < 1000, then an exact BallTree
        method will be set.  If len(X) > 1000 and X is a dense matrix, then
        the Annoy method will be set.  Otherwise, NNDescent will be used. Other
        options are: 'exact', 'approx', 'balltree', 'annoy', and 'pynndescent'.

    metric: Union[str, Callable] (optional, default='l2')
        Metric used to calculate distances between samples in the data set.

    metric_params: dict (optional, default=None)
        If needed, parameters for the supplied distance metric, `metric`.
        Default is `None`.

    n_jobs: int (optional, default=1)
        Number of processors to use when finding the nearest neighbors.  This
        follows the scikit-learn convention where ``-1`` means to use all
        processors, ``-2`` means to use all but one, etc.

    random_state: Union[int, RandomState] (optional, default=None)
        State for the random number generator. If `random_state` is an
        integer, then the input is interpreted as a seed for the RNG.  A
        RandomState instance can also be provided, which will then be used as
        the RNG.  Default is `None`, which uses the `RandomState` provided by
        np.random.

    verbose: numeric (optional, default=0)
        Flag indicating the level of verbosity to use in the outputs.  Setting
        to -1 (or smaller) will suppress all output.

    Other keyword arguments will be passed to the kNNIndex subclass at
    initialization.  See BallTree, Annoy, or NNDescent subclasses for more
    information.

    Returns
    -------
    kNN_index: kNNIndex object
        Initialized NN indexer.  Specific type of indexer will depend on data
        type and shape.
    """

    ## If the user supplied an initialized kNNIndex object, then we're good...
    if isinstance(NN_alg, kNNIndex):
        return NN_alg

    ## Check the data type, size, and shape.
    X = check_array(X, accept_sparse=True, ensure_2d=True)

    ## If the data are not sparse and the metric is ok, use ANNOY. (Init ANNOY
    ## just to get the list of valid metrics.)
    ANNOY_metrics = Annoy('l2').VALID_METRICS
    if not (sp.issparse(X)) and (metric.lower() in ANNOY_metrics):
        preferred_approx_method = Annoy
    else:
        preferred_approx_method = NNDescent

    ## If we've requested 'ANNOY' but the data are sparse, raise an error.
    if sp.issparse(X) and (NN_alg.lower() == 'annoy'):
        err_str = f"Incompatible kNN algorithm 'ANNOY' for sparse input data."
        raise ValueError(err_str)

    ## If the data have fewer than 1000 samples, just use the exact method.
    if X.shape[0] < 1000:
        preferred_method = BallTree
    else:
        preferred_method = preferred_approx_method

    ## Set up a quick string to class converter
    methods = {
        "exact": BallTree,
        "auto": preferred_method,
        "approx": preferred_approx_method,
        "balltree": BallTree,
        "annoy": Annoy,
        "pynndescent": NNDescent
    }

    ## Check that NN_alg is a valid method.
    if NN_alg.lower() not in methods:
        err_str =  f"Unrecognized nearest neighbor algorithm '{NN_alg}'. Valid"
        err_str += f" methods are 'exact', 'auto', 'approx', 'balltree',"
        err_str += f"'annoy', and 'pynndescent'."
        raise ValueError(err_str)

    ## Instantiate the kNNIndex.
    kNN_index = methods[NN_alg.lower()](metric, metric_params, n_jobs,
                                        random_state, verbose, **kwargs)

    return kNN_index
