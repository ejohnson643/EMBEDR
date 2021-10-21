
import EMBEDR.affinity as aff
import EMBEDR.callbacks as cb
import EMBEDR.ees as ees
import EMBEDR.nearest_neighbors as nn
from EMBEDR.tsne import tSNE_Embed
import EMBEDR.utility as utl
from EMBEDR.version import __version__ as ev

import hashlib
import json
import numpy as np
import os
from os import path
import pickle as pkl
import scipy.sparse as sp
import scipy.stats as st
from sklearn.utils import check_array, check_random_state


class EMBEDR(object):

    def __init__(self,
                 X=None,
                 # Important hyperparameters
                 perplexity=None,
                 n_neighbors=None,
                 # kNN graph parameters
                 kNN_metric='euclidean',
                 kNN_alg='auto',
                 kNN_params={},
                 # Affinity matrix parameters
                 aff_type="fixed_entropy_gauss",
                 aff_params={},
                 # Dimensionality reduction parameters
                 n_components=2,
                 DRA='tsne',
                 DRA_params={},
                 # Embedding statistic parameters
                 EES_type="dkl",
                 EES_params={},
                 pVal_type="average",
                 # Runtime parameters
                 n_data_embed=1,
                 n_null_embed=1,
                 n_jobs=1,
                 allow_sparse=True,
                 keep_affmats=False,
                 random_state=1,
                 verbose=1,
                 # File I/O parameters
                 do_cache=True,
                 project_name="EMBEDR_project",
                 project_dir="./projects/"):

        # Important hyperparameters
        self.perplexity = perplexity
        self.n_neighbors = n_neighbors

        ## kNN graph parameters
        self.kNN_metric = kNN_metric
        self.kNN_alg = kNN_alg.lower()
        self.kNN_params = kNN_params.copy()

        ## Affinity matrix parameters
        self.aff_type = aff_type.lower()
        self.aff_params = aff_params.copy()

        ## Dimensionality reduction parameters
        self.n_components = int(n_components)
        self.DRA = DRA.lower()
        self.DRA_params = DRA_params.copy()

        ## Embedding statistic parameters
        self.EES_type = EES_type.lower()
        self.EES_params = EES_params.copy()
        self.pVal_type = pVal_type.lower()

        ## Runtime parameters
        err_str = "Number of data embeddings must be > 0."
        assert int(n_data_embed) > 0, err_str
        self.n_data_embed = int(n_data_embed)

        err_str = "Number of null embeddings must be >= 0."
        assert int(n_null_embed) >= 0, err_str
        self.n_null_embed = int(n_null_embed)

        self.n_jobs = int(n_jobs)
        self.rs = check_random_state(random_state)
        self._seed = self.rs.get_state()[1][0]
        self.verbose = float(verbose)

        self._allow_sparse = bool(allow_sparse)
        self._keep_affmats = bool(keep_affmats)

        ## File I/O parameters
        self.do_cache = bool(do_cache)
        self.project_name = project_name
        if self.do_cache:
            if path.isdir(project_dir):
                self.project_dir = project_dir
            else:
                err_str  = f"Couldn't find project directory `{project_dir}`."
                err_str += f" Please make sure this is a valid directory or"
                err_str += f" turn off file caching (set `do_cache=False`)."
                raise OSError(err_str)

            if not path.isdir(path.join(project_dir, project_name)):
                if self.verbose >= 5:
                    print(f"Creating project directory "
                          f"{path.join(project_dir, project_name)}")
                os.mkdir(path.join(project_dir, project_name))

        ## Amount of data to use when creating unique hash.
        self.chunksize = 100

        if X is not None:
            self._init_with_data(X)

        print(self)

    def __str__(self):
        out_str = ""
        if self.verbose > 1:
            out_str += f"\n\n\tEMBEDR Class v{ev}\n" + 35 * "=" + "\n\n"

        if self.verbose > 1:
            if self.do_cache:
                out_str += f"\tIntermediate results for the"
                out_str += f" {self.project_name} project will be cached in"
                out_str += f" {self.project_dir}!"
            else:
                out_str += f"\tIntermediate results for the"
                out_str += f" {self.project_name} project will not be cached!"

        if self.verbose > 2:
            out_str += f"\n\tkNN algorithm is {self.kNN_alg}"
            out_str += f"\n\tAffinity type is {self.aff_type}"
            out_str += f"\n\tDim.Red. Algorithm is {self.DRA}"
            out_str += f"\n\tn_components = {self.n_components}"
            out_str += f"\n\tn_data_embed = {self.n_data_embed}"
            out_str += f"\n\tn_null_embed = {self.n_null_embed}"

        return out_str

    def fit(self, X):

        #####################
        ## Fit to the DATA ##
        #####################

        self._init_with_data(X)

        ## Finally, we can do the computations!
        self._fit(null_fit=False)

        #####################
        ## Fit to the NULL ##
        #####################

        self._fit(null_fit=True)

        ## Get p-Values
        self.calculate_pValues()

    def _init_with_data(self, X):

        ## Check that the data is a 2D array
        self.data_X = check_array(X, accept_sparse=True, ensure_2d=True)
        ## If the data aren't sparse but could be, sparsify!
        if self._allow_sparse and (not sp.issparse(self.data_X)):
            if np.mean(self.data_X == 0) > 0.1:
                self.data_X = sp.csr_matrix(self.data_X)

        ## Get the data shape
        self.n_samples, self.n_features = self.data_X.shape

        if (self.verbose >= 1) and self._keep_affmats:
            if self.n_samples > 5000:
                print(f"WARNING: Saving affinity matrices may result in"
                      f" high memory usage!  (`keep_affmats` = True)")

        ## Get the object hash
        self.hash = self._get_hash()

        ## Create project directory
        self.project_hdr = self._get_project_hdr()

        ## Set up n_neighbors and perplexity hyperparameters
        self._set_hyperparameters()

        return

    def _set_hyperparameters(self):
        """Set perplexity and number of neighbors (k) based on input data.

        The user can specify either perplexity (if using a Gaussian affinity
        for t-SNE or the DKL metric, for example) or the number of nearest
        neighbors to a sample that are considered a sample's neighborhood as a
        hyperparameter in the EMBEDR method (or generically for a DRA).  By
        default, we try to set them based on the data automatically if one or
        both are not supplied.  We also leave the option for perplexity and
        n_neighbors to be set on a sample-wise basis by allowing the user to
        provide an array of values for these hyperparameters.  However, to
        simplify some calculations, if an array of nearest neighbors values are
        provided, the largest value is used to initialize kNN graphs and
        affinity matrices.  Perplexity and `n_neighbors` are strongly
        correlated for most data sets, so that if an overly large kNN graph is
        specified, a small kNN will still only use a smaller, "effective",
        number of nearest neighbors.  This effect is detailed in the EMBEDR
        publication where the "effective" nearest neighbors are discussed.
        """

        ## If neither perplexity nor n_neighbors are set...
        if (self.perplexity is None) and (self.n_neighbors is None):
            ## Set perplexity to 10% of the sample size
            self.perplexity = np.clip(self.n_samples / 10., 1, self.n_samples)
            ## Set n_neighbors based on perplexity
            self.n_neighbors = int(np.clip(int(3 * self.perplexity + 1), 1,
                                           self.n_samples - 1))

        ## If perplexity is not set and n_neighbors is...
        elif self.perplexity is None:
            ## Set perplexity to be 1/3 of the number of neighbors.  Note,
            ## it doesn't matter whether `n_neighbors` is a scalar or an array.
            self.perplexity = np.clip(self.n_neighbors / 3., 1, self.n_samples)

        ## If n_neighbors is not set and perplexity is...
        elif self.n_neighbors is None:
            ## Set n_neighbors based on perplexity. Note,
            ## it doesn't matter whether `perplexity` is a scalar or an array.
            self.n_neighbors = np.clip(3 * self.perplexity + 1,
                                       1, self.n_samples - 1)

        ## Check the values in the perplexity array (and set `perp_arr`)
        if np.isscalar(self.perplexity):
            if (self.perplexity <= 1) or (self.perplexity >= self.n_samples):
                err_str = f"Perplexity must be between 0 and {self.n_samples}!"
                raise ValueError(err_str)

            perp_arr = np.ones((self.n_samples)) * self.perplexity

        else:
            perp_arr = np.array(self.perplexity).astype(float).squeeze()

            if perp_arr.ndim == 0:
                perp_arr = np.ones((self.n_samples)) * self.perplexity

            elif perp_arr.ndim == 1:
                err_str =  f"Perplexity array must have length = len(data) ="
                err_str += f" {self.n_samples}."
                assert len(perp_arr) == self.n_samples, err_str

            else:
                err_str =  f"Perplexity must be either a scalar or a 1D array."
                raise ValueError(err_str)

            perp_arr = np.clip(perp_arr, 1, self.n_samples)

        self._perp_arr = perp_arr[:]

        ## Check the values in the nearest neighbors array (and set `perp_arr`)
        if np.isscalar(self.n_neighbors):

            self.n_neighbors = int(self.n_neighbors)

            if not (1 <= self.n_neighbors <= self.n_samples - 1):
                err_str  = f"Perplexity must be between 1 and"
                err_str += f"{self.n_samples - 1}!"
                raise ValueError(err_str)

            nn_arr = (np.ones((self.n_samples)) * self.n_neighbors).astype(int)

        else:
            nn_arr = np.array(self.n_neighbors).astype(int).squeeze()

            if nn_arr.ndim == 0:
                nn_arr = np.ones((self.n_samples)) * self.n_neighbors

            elif nn_arr.ndim == 1:
                err_str =  f"Nearest neighbors array must have length ="
                err_str += f" len(data) = {self.n_samples}."
                assert len(nn_arr) == self.n_samples, err_str

            else:
                err_str =  f"`n_neighbors` must be either a scalar or a"
                raise ValueError(err_str + " 1D array.")

            nn_arr = np.clip(nn_arr, 1, self.n_samples - 1)
            self.n_neighbors = nn_arr[:]

        self._nn_arr = nn_arr[:]
        self._max_nn = int(self._nn_arr.max())

    def _get_hash(self):

        ## If we already have a hash then skip this step...
        if hasattr(self, 'hash'):
            return self.hash

        # Hash should depend on data, (size) and random state.

        # Having it depend on size is an easy way to get a new hash from
        # adding or removing samples.
        hash_str = f"{self.n_samples}{self.n_features}"

        # Make sure the hash depends on the EMBEDR object-level random state
        hash_str += f"{self._seed}"

        # Convert the string to bytes
        hash_str = bytes(hash_str, 'ascii')

        # Having the hash depend on data is hopefully a safe way to make
        # unique identifiers even for datasets with identical shapes.
        if isinstance(self.data_X, np.ndarray):
            data_chunk = self.data_X[:self.chunksize, :self.chunksize]
        elif sp.issparse(self.data_X):
            data_chunk = self.data_X[:self.chunksize, :self.chunksize]
            data_chunk = data_chunk.toarray()
        else:
            err_str = f"{self.data_X.__class__.__name__} objects are not"
            err_str += f" supported by EMBEDR at this time."
            raise TypeError(err_str)

        ## Force float data types to be float64...
        if data_chunk.dtype == 'float32':
            data_chunk = data_chunk.astype('float64')

        # Add the data chunks that's been converted to bytes.
        hash_str += data_chunk.tobytes()

        # Digest the bytes into a hash!
        return hashlib.md5(hash_str).hexdigest()

    def _get_project_hdr(self):

        # If we're not caching, then just skip everything.
        if not self.do_cache:
            return None

        # First we set up the subfolder for this data set (hash)
        project_subdir = path.join(self.project_name, self.hash)
        project_path = path.join(self.project_dir, project_subdir)
        if not path.isdir(project_path):
            os.mkdir(project_path)
        self.project_subdir = project_subdir
        self.project_path = project_path

        if self.verbose >= 5:
            print(f"Attempting to load project header in {self.project_path}!")

        # Get the expected project header file
        proj_hdr_name = self.project_name + "_header.json"
        proj_hdr_path = path.join(self.project_path, proj_hdr_name)

        # If it already exists, load it up!
        if path.isfile(proj_hdr_path):
            with open(proj_hdr_path, 'r') as f:
                proj_hdr = json.load(f)

            if self.verbose >= 5:
                print(f"Project header loaded successfully!")

        # Otherwise, initialize a new header.
        else:
            if self.verbose >= 5:
                print(f"Project header couldn't be found...")

            proj_hdr = {'project_name': self.project_name,
                        'project_subdir': self.project_subdir}
            self._set_project_hdr(proj_hdr)

        return proj_hdr

    def _set_project_hdr(self, hdr):

        if self.verbose >= 5:
            print(f"(Re-)Setting project header!")

        # Get the expected project header file
        proj_hdr_name = self.project_name + "_header.json"
        proj_hdr_path = path.join(self.project_path, proj_hdr_name)

        with open(proj_hdr_path, 'w') as f:
            return json.dump(hdr, f, indent=4)

    def _fit(self, null_fit=False):

        if not null_fit:

            ## Basically, no matter what, we need the kNN graph
            self.data_kNN = self.get_kNN_graph(self.data_X)

            ## If we're using t-SNE to embed or DKL as the EES, we need an
            ## affinity matrix.
            if (self.DRA in ['tsne', 't-sne']) or (self.EES_type == 'dkl'):
                self.data_P = self.get_affinity_matrix(X=self.data_X,
                                                       kNN_graph=self.data_kNN)

            ## We then need to get the requested embeddings.
            if (self.DRA in ['tsne', 't-sne']):
                dY, dEES = self.get_tSNE_embedding(X=self.data_X,
                                                   kNN_graph=self.data_kNN,
                                                   aff_mat=self.data_P)

            elif (self.DRA in ['umap']):
                print(f"WARNING: UMAP has not been implemented!")
                # dY, dEES = self._get_UMAP_embedding(self.data_P, null_fit)

            if not self._keep_affmats:
                if self.verbose >= 5:
                    print(f"Deleting data affinity matrix! (Use `obj.data_P."
                          f"calculate_affinities(obj.data_kNN, recalc=True)`"
                          f" to recalculate.)")
                del self.data_P.P

            self.data_Y = dY.copy()
            self.data_EES = dEES.copy()

        else:
            ## Because each null will have its own kNN graph, we need to loop
            ## over the entire fitting process.

            self.null_kNN = {}
            self.null_P   = {}
            self.null_Y   = np.zeros((self.n_null_embed,
                                      self.n_samples,
                                      self.n_components))
            self.null_EES = np.zeros((self.n_null_embed, self.n_samples))

            self._null_seed = self._seed

            for nNo in range(self.n_null_embed):

                if self.verbose >= 1:
                    print(f"\nGenerating null {nNo + 1} / {self.n_null_embed}")

                null_X = self.get_null(seed_offset=self._null_seed)

                ## Generate a kNN graph
                nKNN = self.get_kNN_graph(null_X,
                                          null_fit=null_fit)
                self.null_kNN[nNo] = nKNN

                ## If we need an affinity matrix...
                if (self.DRA in ['tsne', 't-sne']) or (self.EES_type == 'dkl'):
                    nP = self.get_affinity_matrix(null_X,
                                                  kNN_graph=nKNN,
                                                  null_fit=null_fit)
                    self.null_P[nNo] = nP

                ## We then need to get the requested embeddings.
                if (self.DRA in ['tsne', 't-sne']):
                    nY, nEES = self.get_tSNE_embedding(null_X,
                                                       kNN_graph=nKNN,
                                                       aff_mat=nP,
                                                       null_fit=null_fit)

                elif (self.DRA in ['umap']):
                    print(f"WARNING: UMAP has not been implemented!")
                    # nY, nEES = self._get_UMAP_embedding(nP, null_fit)

                if not self._keep_affmats:
                    if self.verbose >= 5:
                        print(f"Deleting null affinity matrix! (Use `obj."
                              f"null_P[{nNo}].calculate_affinities(obj."
                              f"null_kNN[{nNo}], recalc=True)`"
                              f" to recalculate.)")
                    del self.null_P[nNo].P

                self.null_Y[nNo] = nY.copy()
                self.null_EES[nNo] = nEES.copy()

                self._null_seed += 100

    def get_kNN_graph(self, X, null_fit=False):

        ## Initialize the kNN graph using the data and seed.
        seed = self._null_seed if null_fit else self._seed
        kNNObj = self._initialize_kNN_index(X, seed)

        ## If we're doing file caching, first we want to try and load the graph
        if self.do_cache:
            loaded_kNN = self.load_kNN_graph(X,
                                             kNNObj=kNNObj,
                                             seed=seed,
                                             null_fit=null_fit,
                                             raise_error=False)

            ## If we found a matching kNN graph, then we're done!
            if loaded_kNN is not None:
                return loaded_kNN

        ## If we're not caching, no kNNs have been made, or a matching kNN
        ## couldn't be found, then we need to fit a new one!
        if self.verbose >= 1:
            print(f"\nFitting kNN graph to data with k={self._max_nn}"
                  f" neighbors...")

        kNNObj.fit(X, self._max_nn)

        ## Finally, if we're caching and a kNN graph has just been fit, then
        ## add it to the header and save it to the cache.
        if self.do_cache:

            ## These are the header attributes we use to distinguish kNN graphs
            kNN_subhdr = {'kNN_alg': kNNObj.__class__.__name__,
                          'metric': kNNObj.metric,
                          'params': self.kNN_params,
                          'seed': int(seed)}

            ## If there are no saved kNN's yet, make that field in the header.
            if 'kNN' not in self.project_hdr:
                self.project_hdr['kNN'] = dict(Data={}, Null={})

            ## Get the new file name.
            data_type = 'Null' if null_fit else 'Data'
            kNN_hdr = self.project_hdr['kNN'][data_type]
            n_kNN_made = len(kNN_hdr)  ## Increment the filename by 1
            kNN_name = f"{data_type}_kNN_{n_kNN_made:04d}.knn"

            kNNObj._filename = kNN_name

            ## If the kNN graph was generated with the Annoy algorithm, then
            ## we need a separate filename to cache the index.
            if isinstance(kNNObj, nn.Annoy):
                pkl_name = kNN_name[:-4] + ".ann"
                kNNObj.pickle_name = path.join(self.project_path, pkl_name)

            ## Save the graph to file.
            if self.verbose >= 2:
                print(f"Caching {kNN_name} to file!")

            ## Dump the kNN object.
            kNN_path = path.join(self.project_path, kNN_name)
            with open(kNN_path, 'wb') as f:
                pkl.dump(kNNObj, f)

            ## Update the project header.
            kNN_subpath = path.join(self.project_subdir, kNN_name)
            kNN_hdr[kNN_subpath] = kNN_subhdr
            self._set_project_hdr(self.project_hdr)

        return kNNObj

    def _initialize_kNN_index(self, X, seed):

        if self.verbose >= 3:
            print(f"\nInitializing kNN index...")

        index = nn._initialize_kNN_index(X,
                                         NN_alg=self.kNN_alg,
                                         metric=self.kNN_metric,
                                         n_jobs=self.n_jobs,
                                         random_state=seed,
                                         verbose=self.verbose,
                                         **self.kNN_params)

        index_params = {'metric_params': index.metric_params}
        if isinstance(index, (nn.Annoy, nn.NNDescent)):
            index_params['n_trees'] = index.n_trees
        if isinstance(index, nn.NNDescent):
            index_params['n_iters'] = index.n_iters

        self.kNN_params.update(index_params)

        return index

    def load_kNN_graph(self,
                       X,
                       kNNObj=None,
                       seed=None,
                       null_fit=False,
                       raise_error=True):

        ## Set the seed if it is not provided.
        if seed is None:
            seed = self._null_seed if null_fit else self._seed

        ## Initialize the kNN graph using the data and seed.
        if kNNObj is None:
            kNNObj = self._initialize_kNN_index(X, seed)

        ## Check for the existence of loaded kNN graphs.
        if 'kNN' not in self.project_hdr:

            err_str  = f"Error loading kNN graph: no kNN graphs have yet been"
            err_str += f" made for this data and/or seed!"

            ## If we're running this independent of `fit`, then raise an error!
            if raise_error:
                raise FileNotFoundError(err_str)
            ## Otherwise, warn that nothing could be loaded.
            elif self.verbose >= 2:
                print(err_str)

            return  ## We can't get past here with this function!

        ## Get the kNN header.
        data_type = 'Null' if null_fit else 'Data'
        if self.verbose >= 3:
            print(f"Searching for matching kNN graph in {data_type} cache...")
        kNN_hdr = self.project_hdr['kNN'][data_type]

        ## Look in the cache for a matching graph.
        kNN_path = self._match_kNN_graph(kNNObj, kNN_hdr, seed)

        ## If no matching graph is found then return None.
        if kNN_path is None:
            return None

        if self.verbose >= 3:
            print(f"Attempting to load kNN graph...")

        ## If a path has been found to a matching kNN graph load it!
        with open(kNN_path, 'rb') as f:
            kNNObj = pkl.load(f)

        ## If the kNN is an ANNOY object, try to load the ANNOY index using the
        ## current project directory.
        if isinstance(kNNObj, nn.Annoy):
            if kNNObj.index is None:
                ann_name = kNNObj.pickle_name.split("/")[-1]
                ann_path = os.path.join(self.project_dir,
                                        self.project_subdir,
                                        ann_name)
                kNNObj._load_annoy_index(ann_path)

        ## Check that the requested number of neighbors are present.
        if self.verbose >= 3:
            print_str = f"kNN graph loaded!  Checking shape ... "

        ## If there are too few neighbors, query for more.
        if kNNObj.kNN_dst.shape[1] < self._max_nn:
            if self.verbose >= 3:
                print_str += f"... not enough neighbors, querying for more!"
                print(print_str)

            idx, dst = out.query(X, self._max_nn + 1)
            kNNObj.kNN_dst = dst[:, 1:]
            kNNObj.kNN_idx = idx[:, 1:]

        ## If there are too many neighbors, report that information.
        elif kNNObj.kNN_dst.shape[1] > self._max_nn:
            if self.verbose >= 3:
                print(print_str + f"Too many neighbors, clipping kNN graph...")

        elif self.verbose >= 3:
            print(print_str + f"... shape is correct!")

        ## Always clip to the number of nearest neighbors required!
        kNNObj.kNN_dst = kNNObj.kNN_dst[:, :self._max_nn]
        kNNObj.kNN_idx = kNNObj.kNN_idx[:, :self._max_nn]

        ## Return the loaded kNN graph.
        return kNNObj

    def _match_kNN_graph(self, kNNObj, kNN_hdr, seed):

        matching_kNN = None
        for kNN_name, kNN_subhdr in kNN_hdr.items():

            if self.verbose >= 5:
                print(f"Checking kNN graph {kNN_name}...")

            ## Check that the algorithm (subclass) matches
            if kNNObj.__class__.__name__ != kNN_subhdr['kNN_alg']:
                if self.verbose >= 5:
                    print(f"...kNN algorithm didn't match!")
                continue

            ## Check that the metric matches
            if kNNObj.metric != kNN_subhdr['metric']:
                if self.verbose >= 5:
                    print(f"...distance metric didn't match!")
                continue

            ## Check that the other parameters match (see
            ## EMBEDR.nearest_neighbors for a full list of arguments.)
            if self.kNN_params != kNN_subhdr['params']:
                if self.verbose >= 5:
                    print(f"...other kNN parameters didn't match!")
                continue

            ## Check that the random seed matches
            if seed != kNN_subhdr['seed']:
                if self.verbose >= 5:
                    print(f"...random seed didn't match!")
                continue

            ## If all checks are passed, save the corresponding path!
            if self.verbose >= 3:
                print(f"kNN graph at {kNN_name} passed all checks!")

            matching_kNN = kNN_name
            break

        if matching_kNN is None:
            return None

        ## In EMBEDR >v2.1 we want to be able to modify the project dir so that
        ## it can be specified as a relative path. In previous versions the
        ## header was keyed by a relative path at instantiation, which made it
        ## impossible to load objects from different working dirs.
        kNN_name = matching_kNN.split("/")[-1]
        return os.path.join(self.project_dir, self.project_subdir, kNN_name)

    def get_affinity_matrix(self, X, kNN_graph=None, null_fit=False):

        ## Initialize affinity matrix object.
        if kNN_graph is None:
            kNN_graph = self.get_kNN_graph(X, null_fit=null_fit)
        affObj = self._initialize_affinity_matrix(kNN_graph)[0]
        init_kernel_params = affObj.kernel_params.copy()

        ## If we're doing file caching...
        if self.do_cache:
            loaded_affObj = self.load_affinity_matrix(X,
                                                      kNN_graph=kNN_graph,
                                                      affObj=affObj,
                                                      null_fit=null_fit,
                                                      raise_error=False)

            ## If we found a matching affinity matrix, then we're done!
            if loaded_affObj is not None:
                return loaded_affObj

        ## If we're not caching, no affinities have been made, or a matching
        ## affinity matrix couldn't be found, then we need to fit a new one!
        affObj.fit(kNN_graph)

        ## Finally, if we're caching, then add the fitted affinity matrix to
        ## the header and save the affinity matrix to the cache.
        if self.do_cache:

            ## If there are no saved kNN's yet, make that field in the header.
            if 'affinity_matrix' not in self.project_hdr:
                self.project_hdr['affinity_matrix'] = dict(Data={}, Null={})

            ## Get the new file name.
            data_type = 'Null' if null_fit else 'Data'
            aff_hdr = self.project_hdr['affinity_matrix'][data_type]
            n_aff_made = len(aff_hdr)
            aff_name = f"{data_type}_AffMat_{n_aff_made:04d}.aff"

            affObj._filename = aff_name

            if self.verbose >= 2:
                print(f"Caching {aff_name} to file!")

            ## To save space when caching, we don't save the calculated
            ## affinity matrix, P, but the info we need to quickly regenerate
            ## it instead.
            if hasattr(affObj, 'P'):
                P = affObj.P.copy()
                del affObj.P

            aff_path = path.join(self.project_path, aff_name)
            with open(aff_path, 'wb') as f:
                pkl.dump(affObj, f)

            ## Within the object however, we keep the matrix.
            affObj.P = P.copy()

            ## Get the kernel parameters and convert any arrays to lists.
            fit_kernel_params = affObj.kernel_params.copy()
            for k in init_kernel_params:
                if isinstance(fit_kernel_params[k], np.ndarray):
                    init_kernel_params[k] = fit_kernel_params[k].tolist()

            ## These are the parameters used to match affinity matrices
            aff_subhdr = dict(aff_type=affObj.__class__.__name__,
                              n_neighbors=int(affObj.n_neighbors),
                              kernel_params=init_kernel_params,
                              kNN_alg=kNN_graph.__class__.__name__,
                              kNN_params=affObj.kNN_params,
                              kNN_filename=kNN_graph._filename,
                              normalization=affObj.normalization,
                              symmetrize=affObj.symmetrize)
            ## Don't match perplexity if it isn't set.
            if self.perplexity is not None:
                aff_subhdr['perp_arr'] = affObj._perp_arr.tolist()

            ## Update the project header
            aff_subpath = path.join(self.project_subdir, aff_name)
            aff_hdr[aff_subpath] = aff_subhdr
            self._set_project_hdr(self.project_hdr)

        return affObj

    def _initialize_affinity_matrix(self, X):

        if self.verbose >= 3:
            print(f"\nInitializing affinity matrix...")

        kNN_params = {'metric': self.kNN_metric}
        kNN_params.update(self.kNN_params)

        aff_params = self.aff_params.copy()
        if 'kernel_params' in self.aff_params:
            kernel_params = self.aff_params['kernel_params']
            del aff_params['kernel_params']
        else:
            kernel_params = {}

        [affObj,
         kNNObj] = aff._initialize_affinity_matrix(X,
                                                   aff_type=self.aff_type,
                                                   perplexity=self._perp_arr,
                                                   n_neighbors=self._max_nn,
                                                   kNN_params=kNN_params,
                                                   n_jobs=self.n_jobs,
                                                   random_state=self.rs,
                                                   verbose=self.verbose,
                                                   kernel_params=kernel_params,
                                                   **self.aff_params)

        return affObj, kNNObj

    def load_affinity_matrix(self,
                             X,
                             kNN_graph=None,
                             affObj=None,
                             null_fit=False,
                             raise_error=True):

        if kNN_graph is None:
            kNN_graph = self.load_kNN_graph(X, null_fit=null_fit,
                                            raise_error=raise_error)

        if affObj is None:
            affObj = self._initialize_affinity_matrix(kNN_graph)[0]

        ## Check for the existence of loaded kNN graphs.
        if "affinity_matrix" not in self.project_hdr:

            err_str  = f"Error loading affinity matrix: no affinity matrices"
            err_str += f" have yet been made for this data and/or seed!"

            ## If we're running this independent of `fit`, then raise an error!
            if raise_error:
                raise FileNotFoundError(err_str)
            ## Otherwise, warn that nothing could be loaded.
            elif self.verbose >= 2:
                print(err_str)

            return  ## We can't get past here with this function!

        ## Get the affinity matrix header.
        data_type = 'Null' if null_fit else 'Data'
        if self.verbose >= 3:
            print(f"Searching for matching affinity matrix in {data_type}"
                  " cache...")
        aff_hdr = self.project_hdr['affinity_matrix'][data_type]

        ## Look for matching affinity matrix
        aff_path = self._match_affinity_matrix(affObj, kNN_graph, aff_hdr)

        ## If no matching matrix is found then return None.
        if aff_path is None:
            return None

        if self.verbose >= 3:
            print(f"Attempting to load affinity matrix...")

        ## If a path has been found to a matching kNN graph load it!
        with open(aff_path, 'rb') as f:
            affObj = pkl.load(f)

        if self.verbose >= 3:
            print(f"Affinity matrix successfully loaded!")

        return affObj

    def _match_affinity_matrix(self, affObj, kNNObj, aff_hdr):

        ## Get the kernel parameters and convert any arrays to lists.
        init_kernel_params = affObj.kernel_params.copy()
        for k in init_kernel_params:
            if isinstance(init_kernel_params[k], np.ndarray):
                init_kernel_params[k] = init_kernel_params[k].tolist()

        ## If we have made affinity matrices(s) for this project, cycle
        ## through them until we get matching parameters.  We want to
        ## then load from the corresponding filepath.
        matching_aff = None
        for aff_name, aff_subhdr in aff_hdr.items():

            if self.verbose >= 5:
                print(f"Checking affinity matrix at {aff_name}...")

            ## Check that... the affinity matrix type matches
            if affObj.__class__.__name__ != aff_subhdr['aff_type']:
                if self.verbose >= 5:
                    print(f"...affinity type doesn't match!")
                continue

            if affObj.n_neighbors != aff_subhdr['n_neighbors']:
                if self.verbose >= 5:
                    print(f"...number of neighbors doesn't match!")
                continue

            ## Check that... the kernel parameters match
            if np.any([val != aff_subhdr['kernel_params'][key]
                       for key, val in init_kernel_params.items()]):
                if self.verbose >= 5:
                    print(f"...kernel parameters don't match!")
                continue

            ## Check that... the type of kNN kernel matches
            if kNNObj.__class__.__name__ != aff_subhdr['kNN_alg']:
                if self.verbose >= 5:
                    print(f"...kNN algorithm doesn't match!")
                continue

            ## Check that... the kNN parameters upon which the matrix
            ## is built match.
            if affObj.kNN_params != aff_subhdr['kNN_params']:
                if self.verbose >= 5:
                    print(f"...kNN parameters don't match!")
                continue

            ## Check that... the kNN graph's name is correct!
            if kNNObj._filename != aff_subhdr['kNN_filename']:
                if self.verbose >= 5:
                    print(f"...kNN graph (name) doesn't match!")
                continue

            ## Check that... the matrix normalization matches
            if affObj.normalization != aff_subhdr['normalization']:
                if self.verbose >= 5:
                    print(f"...affinity normalization doesn't match!")
                continue

            ## Check that... the matrix symmetry matches
            if affObj.symmetrize != aff_subhdr['symmetrize']:
                if self.verbose >= 5:
                    print(f"...affinity symmetry doesn't match!")
                continue

            if self.perplexity is not None:
                if np.any(affObj._perp_arr != aff_subhdr['perp_arr']):
                    if self.verbose >= 5:
                        print(f"...perplexities don't match!")
                    continue

            # If all checks are passed, save the corresponding path!
            if self.verbose >= 2:
                print(f"Affinity matrix at {aff_name} passed all"
                      f" checks!")

            matching_aff = aff_name
            break

        if matching_aff is None:
            return None

        ## In EMBEDR >v2.1 we want to be able to modify the project dir so that
        ## it can be specified as a relative path. In previous versions the
        ## header was keyed by a relative path at instantiation, which made it
        ## impossible to load objects from different working dirs.
        aff_name = matching_aff.split("/")[-1]
        return os.path.join(self.project_dir, self.project_subdir, aff_name)

    def get_tSNE_embedding(self,
                           X,
                           kNN_graph=None,
                           aff_mat=None,
                           null_fit=False,
                           return_tSNE_objects=False):

        ## Get the number of requested embeddings
        n_embeds_made = 0
        n_embeds_reqd = 1 if null_fit else self.n_data_embed
        n_embeds_2_make = n_embeds_reqd

        ## Initialize an embedding using the input arguments.
        if kNN_graph is None:
            kNN_graph = self.get_kNN_graph(X, null_fit=null_fit)

        if aff_mat is None:
            aff_mat = self.get_affinity_matrix(X, kNN_graph=kNN_graph,
                                               null_fit=null_fit)

        # Make sure that an affinity matrix is loaded...
        if not hasattr(aff_mat, "P"):
            try:
                aff_mat.P = aff_mat.calculate_affinities(kNN_graph,
                                                         recalc=True)
            except AttributeError:
                aff_mat.P = aff_mat.calculate_affinities(kNN_graph,
                                                         recalc=False)

        embObj = self._initialize_tSNE_embed(aff_mat)

        ## If we're doing file caching...
        if self.do_cache:

            [emb_Y,
             emb_EES,
             emb_path] = self.load_tSNE_embedding(X,
                                                  kNN_graph=kNN_graph,
                                                  aff_mat=aff_mat,
                                                  embObj=embObj,
                                                  null_fit=null_fit,
                                                  raise_error=False)

            ## If anything was loaded, check it's shape!
            if emb_Y is not None:
                n_embeds_made = emb_Y.shape[0]

            ## If we don't need any more embeddings, then quit!
            n_embeds_2_make = n_embeds_reqd - n_embeds_made
            if n_embeds_2_make == 0:
                return emb_Y, emb_EES

        ## If there aren't enough embeddings, we fit them here!
        tmp_embed_arr = np.zeros((n_embeds_2_make,
                                  self.n_samples,
                                  self.n_components))

        ## Create the required number of embeddings!
        for ii in range(n_embeds_2_make):
            if self.verbose >= 1:
                print(f"\nFitting embedding {ii + 1}/{n_embeds_2_make}"
                      f" ({n_embeds_reqd} total requested).")

            ## We make sure to increment the random seed so that the
            ## intitializations are offset in each iteration.
            seed_offset = n_embeds_made + ii

            embObj = tSNE_Embed(n_components=self.n_components,
                                perplexity=self.perplexity,
                                n_jobs=self.n_jobs,
                                random_state=self._seed + seed_offset,
                                verbose=self.verbose,
                                **self.DRA_params)
            embObj.fit(aff_mat)

            tmp_embed_arr[ii] = embObj.embedding[:]

        ## If the current affinity matrix is not locally normalized...
        if aff_mat.normalization != 'local':

            ## Make copy of current affinity parameters
            old_aff_params = {}
            if self.aff_params is not None:
                old_aff_params = self.aff_params.copy()
            ## Force current normalization to be 'local'
            self.aff_params.update({"normalization": 'local'})

            ## Load the correct affinity matrix...
            local_aff_mat = self.get_affinity_matrix(X,
                                                     kNN_graph=kNN_graph,
                                                     null_fit=null_fit)

            # Make sure that an affinity matrix is loaded...
            if not hasattr(local_aff_mat, "P"):
                try:
                    P = local_aff_mat.calculate_affinities(kNN_graph,
                                                           recalc=True)
                except AttributeError:
                    P = local_aff_mat.calculate_affinities(kNN_graph,
                                                           recalc=False)
                local_aff_mat.P = P

            ## Return the old affinity matrix parameters
            self.aff_params = old_aff_params.copy()

        ## ... if it is locally normalized...
        else:
            local_aff_mat = aff_mat

        ## Calculate the EES
        tmp_EES_arr = self.calculate_EES(local_aff_mat.P, tmp_embed_arr)

        ## Append the new results to those already loaded.
        if n_embeds_made > 0:
            emb_Y   = np.vstack((emb_Y, tmp_embed_arr))
            emb_EES = np.vstack((emb_EES, tmp_EES_arr))
        else:
            emb_Y = tmp_embed_arr.copy()
            emb_EES = tmp_EES_arr.copy()

        if self.verbose >= 4:
            print(f"Checking that the `emb_Y` has size {emb_Y.shape}")
            print(f"Checking that the `emb_EES` has size {emb_EES.shape}")

        ## Then if we re caching files, we need to save these embeddings!
        if self.do_cache:

            ## If there are no saved kNN's yet, make that field in the header.
            if 'Embed_tSNE' not in self.project_hdr:
                self.project_hdr['Embed_tSNE'] = dict(Data={}, Null={})

            ## Get the embeddings sub-header.
            data_type = "Null" if null_fit else "Data"
            emb_hdr = self.project_hdr['Embed_tSNE'][data_type]

            ## If no embeddings were loaded, make a new file.
            if emb_path is None:
                ## Get the embedding filename.
                n_times_made = len(emb_hdr)
                emb_name = f"{data_type}_tSNE_Embed_{n_times_made:04d}.emb"

                if self.verbose >= 2:
                    print(f"Caching {emb_name} to file!")

            ## If we already loaded some embeddings, add them to the file.
            else:
                emb_name = emb_path.split("/")[-1]
                if self.verbose >= 2:
                    print(f"Appending to cache {emb_name}!")

            ## Get the embedding filepath
            emb_path = path.join(self.project_path, emb_name)

            with open(emb_path, 'wb') as f:
                pkl.dump([emb_Y, emb_EES], f)

            tSNE_params = self._get_matchable_tSNE_params(embObj)

            emb_subhdr = dict(DRA_params=tSNE_params,
                              affmat_filename=aff_mat._filename)

            ## Reset the project header json.
            emb_subpath = path.join(self.project_subdir, emb_name)
            emb_hdr[emb_path] = emb_subhdr
            self._set_project_hdr(self.project_hdr)

        return emb_Y, emb_EES

    def _initialize_tSNE_embed(self, affObj):

        if self.verbose >= 3:
            print(f"\nInitializing t-SNE Embedding...")

        embObj = tSNE_Embed(n_components=self.n_components,
                            perplexity=self._perp_arr,
                            n_jobs=self.n_jobs,
                            random_state=self._seed,
                            verbose=self.verbose,
                            **self.DRA_params)

        embObj.initialize_embedding(affObj)

        return embObj

    def load_tSNE_embedding(self,
                            X,
                            kNN_graph=None,
                            aff_mat=None,
                            embObj=None,
                            null_fit=False,
                            raise_error=True):

        ## If a kNN_graph has not been provided, load one!
        if kNN_graph is None:
            kNN_graph = self.load_kNN_graph(X, null_fit=null_fit,
                                            raise_error=raise_error)
        ## If an affinity matrix has not been provided, load one!
        if aff_mat is None:
            aff_mat = self.load_affinity_matrix(X, kNN_graph=kNN_graph,
                                                null_fit=null_fit,
                                                raise_error=raise_error)

        ## Initialize the embedding object.
        if embObj is None:
            embObj = self._initialize_tSNE_embed(aff_mat)

        ## Check for the existence of loaded kNN graphs.
        if "Embed_tSNE" not in self.project_hdr:

            err_str  = f"Error loading t-SNE embeddings: no t-SNE embeds"
            err_str += f" have yet been made for this data and/or seed!"

            ## If we're running this independent of `fit`, then raise an error!
            if raise_error:
                raise FileNotFoundError(err_str)
            ## Otherwise, warn that nothing could be loaded.
            elif self.verbose >= 2:
                print(err_str)

            ## We can't get past here with this function!
            return [None, None, None]

        ## Get the embedding header
        data_type = "Null" if null_fit else "Data"
        if self.verbose >= 3:
            print(f"Looking for matching t-SNE embeddings in"
                  f"the '{data_type}' cache.")
        emb_hdr = self.project_hdr['Embed_tSNE'][data_type]

        ## Look in the cache for a matching embedding.
        emb_path = self._match_tSNE_embeds(embObj, aff_mat, emb_hdr)

        ## If there is no matching embedding, return None.
        if emb_path is None:
            return [None, None, None]

        ## Load it!
        if self.verbose >= 3:
            print(f"Attempting to load t-SNE embeddings...")

        with open(emb_path, 'rb') as f:
            emb_Y, emb_EES = pkl.load(f)

        ## Get the number of requested embeddings
        n_embeds_reqd = 1 if null_fit else self.n_data_embed

        ## Load up to the requested number of embeddings!
        emb_Y   = emb_Y[:n_embeds_reqd]
        emb_EES = emb_EES[:n_embeds_reqd]

        ## Check that there are the required number of embeddings
        n_embeds_made = emb_Y.shape[0]
        if self.verbose >= 3:
            print(f"{n_embeds_made} t-SNE embeddings loaded!"
                  f" ({n_embeds_reqd} requested)")

        return emb_Y, emb_EES, emb_path

    def _get_matchable_tSNE_params(self, embObj):

        ## Get the t-SNE parameters and convert any arrays to lists.
        matchable_params = ['initialization', 'learning_rate', 'n_early_iter',
                            'early_exag', 'early_mom', 'n_iter', 'exag', 'mom',
                            'max_grad_norm', 'max_step_norm', 't_dof',
                            'neg_grad_method', 'bh_theta', 'FI_n_interp_pts',
                            'FI_min_n_interv', 'FI_ints_per_interv']
        tSNE_params = {}
        for key, val in embObj.__dict__.items():
            if key not in matchable_params:
                continue
            if isinstance(val, np.ndarray):
                tSNE_params[key] = val.tolist()
            else:
                tSNE_params[key] = val

        return tSNE_params

    def _match_tSNE_embeds(self, embObj, affObj, emb_hdr):

        ## Get the matchable t-SNE embedding parameters
        tmp_tSNE_params = self._get_matchable_tSNE_params(embObj)

        matching_embed = None

        ## We want to cycle through the available embeddings and see if they
        ## match the given parameters.
        for emb_name, emb_subhdr in emb_hdr.items():

            if self.verbose >= 5:
                print(f"Checking t-SNE embedding at {emb_name}...")

            ## Match the t-SNE runtime parameters
            if emb_subhdr['DRA_params'] != tmp_tSNE_params:
                continue

            ## Match the affinity matrix.
            if affObj._filename != emb_subhdr['affmat_filename']:
                continue

            # If all checks are passed, save the corresponding path!
            if self.verbose >= 2:
                print(f"t-SNE embedding at {emb_name} passed all checks!")

            matching_embed = emb_name
            break

        if matching_embed is None:
            return None

        ## In EMBEDR >v2.1 we want to be able to modify the project dir so that
        ## it can be specified as a relative path. In previous versions the
        ## header was keyed by a relative path at instantiation, which made it
        ## impossible to load objects from different working dirs.
        emb_name = matching_embed.split("/")[-1]
        return os.path.join(self.project_dir, self.project_subdir, emb_name)

    def calculate_EES(self, P, Y):

        if self.EES_type == 'dkl':
            if self.verbose >= 2:
                print(f"Calculating D_KL as EES!")

            EES = self._calculate_EES_DKL(P, Y)

        else:
            err_str  = f"WARNING: Other empirical embedding statistics have"
            err_str += f" not yet been implemented!"
            raise ValueError(err_str)

        return EES

    def _calculate_EES_DKL(self, P, Y):

        if Y.ndim != 3:
            tmpY = Y[np.newaxis, :, :]
        else:
            tmpY = Y

        P_rowsums = np.asarray(P.sum(axis=1))
        tmp_P = sp.csr_matrix(P / P_rowsums)

        DKL = ees.calculate_DKL(tmp_P.indices, tmp_P.indptr, tmp_P.data, tmpY)

        if np.any(DKL < 0) and (self.verbose > 0):
            print("WARNING: Illegal values detected for DKL!")
            if self.verbose >= 4:
                print((DKL < 0).nonzero())

        return DKL

    def get_null(self, seed_offset=0):

        rng = np.random.RandomState(self._seed + seed_offset)

        if sp.issparse(self.data_X):
            out = sp.csr_matrix([rng.choice(col, size=self.n_samples)
                                 for col in self.data_X.toarray().T]).T

        else:
            out = np.asarray([rng.choice(col, size=self.n_samples)
                              for col in self.data_X.T]).T

        return out

    def calculate_pValues(self):

        if self.verbose >= 2:
            print(f"\nCalulating EES p-Values!")

        nVals, nCDF = utl.calculate_eCDF(self.null_EES)

        n_embeds, n_samples = self.data_EES.shape
        pVals = np.ones_like(self.data_EES)

        max_nEES = nVals[-1]

        for eNo in range(n_embeds):
            ## If the data are smaller than the largest null...
            idx = (self.data_EES[eNo] <= max_nEES).nonzero()
            ## ... get the corresponding value from the null CDF.
            pVals[eNo][idx] = np.asarray([nCDF[np.searchsorted(nVals, data)]
                                          for data in self.data_EES[eNo][idx]])

        if self.pVal_type is None:
            if self.verbose >= 3:
                print(f"No summary method indicated, returning all p-Values!")

            self._pValues = pVals[:]
            self.pValues = None

            return self.pValues

        elif self.pVal_type == 'simes':
            if self.verbose >= 3:
                print(f"Returning summary p-Values using Simes' method!")

            simes_mult = n_embeds / np.arange(1, n_embeds + 1).reshape(-1, 1)
            pVal_idx = np.argsort(pVals, axis=0)
            summ_pVals = np.min(pVals[pVal_idx] * simes_mult, axis=0)

            self._pValues = pVals[:]
            self.pValues = summ_pVals[:]

            return self.pValues

        elif self.pVal_type in ['average', 'avg']:
            if self.verbose >= 3:
                print(f"Returning summary p-Values using averaging method!")

            self._pValues = pVals[:]
            self.pValues = np.mean(pVals, axis=0)

            return self.pValues

        else:
            err_str = f"Unknown p-Value summary method '{self.pVal_type}'..."
            raise ValueError(err_str)

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
        import matplotlib.pyplot as plt

        fig = plt.gcf()
        if ax is None:
            ax = fig.gca()

        if plot_data:
            Y = self.data_Y[embed_2_show]
        else:
            Y = self.null_Y[embed_2_show]

        [pVal_cmap,
         pVal_cnorm] = utl.make_categ_cmap(change_points=pVal_clr_change)

        color_bounds = np.linspace(pVal_clr_change[0],
                                   pVal_clr_change[-1],
                                   pVal_cmap.N)

        pVals = -np.log10(self.pValues)

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
                                   ax=ax,
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


class EMBEDR_sweep(object):

    valid_hyperparameters = ['perplexity', 'n_neighbors']

    def __init__(self,
                 # Set hyperparameters to sweep
                 sweep_values=None,
                 sweep_type='perplexity',
                 n_sweep_values=1,
                 min_sweep_value=0.005,
                 max_sweep_value=0.5,
                 # kNN graph parameters
                 kNN_metric='euclidean',
                 kNN_alg='auto',
                 kNN_params={},
                 # Affinity matrix parameters
                 aff_type="fixed_entropy_gauss",
                 aff_params={},
                 # Dimensionality reduction parameters
                 n_components=2,
                 DRA='tsne',
                 DRA_params={},
                 # Embedding statistic parameters
                 EES_type="dkl",
                 EES_params={},
                 pVal_type="average",
                 use_t_stat_for_opt=True,
                 use_log_t_stat=False,
                 # Runtime parameters
                 n_data_embed=1,
                 n_null_embed=1,
                 n_jobs=1,
                 allow_sparse=True,
                 keep_affmats=False,
                 random_state=1,
                 verbose=1,
                 # File I/O parameters
                 do_cache=True,
                 project_name="EMBEDR_project",
                 project_dir="./projects/"):

        ## Set the verbosity for output level
        self.verbose = float(verbose)
        if self.verbose >= 1:
            print(f"\nInitializing EMBEDR hyperparameter sweep!")

        ## Check that a valid hyperparameter has been set for sweeping.
        if sweep_type.lower() not in self.valid_hyperparameters:
            err_str  = f"Unknown hyperparameter {sweep_type.lower()}..."
            err_str += f" Accepted hyperparameters:"
            raise ValueError(err_str + f" {self.valid_hyperparameters}.")
        self.sweep_type = sweep_type.lower()

        ## Do pre-checking of hyperparameter values and bounds...
        ## ... If an array of hyperparameter values are supplied, check it!
        if sweep_values is not None:
            tmp = check_array(sweep_values,
                              accept_sparse=False,   ## Don't accept sparse
                              ensure_2d=False,       ## Don't enforce 2D array
                              allow_nd=False,        ## Don't allow ndim > 2
                              ensure_min_samples=1)  ## Requre 1 value in array

            ## If an array is supplied, then we reset `n_sweep_values`,
            ## `min/max_sweep_value` based on the array.  Defaults or other
            ## inputs are ignored.
            if self.verbose >= 3:
                print(f"Array of hyperparameter values provided, ignoring"
                      f" input values for `n_sweep_values`, `min_sweep_value,"
                      f" and `max_sweep_value`.")

            ## The sweep parameters can either be >1 (up to inf for now) or
            ## between 0 and 1.  In the former case, the values are interpreted
            ## as real hyperparameter values (providing [10, 20] means that
            ## a `perplexity`/`n_neighbors` of 10 and 20 will be set). The
            ## latter case will be interpreted to set the hyperparameters as a
            ## fraction of the number of samples in the data.  We don't want
            ## to mix these cases, so we insist the array is completely in
            ## [0, 1] or [1, inf].
            if np.any(tmp < 0):
                err_str = f"Hyperparameter values must be > 0."
                raise ValueError(err_str)
            if np.all(tmp < 1):
                if self.verbose >= 3:
                    print(f"Hyperparameter values specified as fractions of"
                          f" the number of samples!")
            elif np.all(tmp > 1):
                if self.verbose >= 3:
                    print(f"Hyperparameter values specified absolutely!")
            else:
                err_str = f"Hyperparameter values must be specified absolutely"
                err_str += f" (values > 1) or as fractions of the sample size"
                err_str += f" (values < 1)."
                raise ValueError(err_str)

            ## Copy the values and squeeze the array.
            self.sweep_values = tmp.squeeze()[:]
            ## Make sure it's a 2D array, where the rows correspond to sweeps.
            if self.sweep_values.ndim == 1:
                self.sweep_values = self.sweep_values.reshape(-1, 1)

            ## Set these based on the array.
            self.n_sweep_values = len(self.sweep_values)
            self.min_sweep_value = np.min(self.sweep_values)
            self.max_sweep_value = np.max(self.sweep_values)

        ## ... If an array is not supplied, check the values of other inputs.
        else:
            self.sweep_values = None

            ## The number of sweeps must be at least 1!
            n_sweep_values = int(n_sweep_values)
            err_str  = f"Number of hyperparameter values in sweep must be > 1!"
            assert n_sweep_values >= 1, err_str
            self.n_sweep_values = n_sweep_values

            ## The minimum hyperparameter value must be greater than 0.
            min_sweep_value = float(min_sweep_value)
            err_str  = f"Minimum hyperparameter value must be > 0!"
            assert min_sweep_value > 0, err_str
            self.min_sweep_value = min_sweep_value

            ## The maximum hyperparameter value must be greater than 0.
            max_sweep_value = float(max_sweep_value)
            err_str  = f"Maximum hyperparameter value must be > 0!"
            assert max_sweep_value > 0, err_str
            self.max_sweep_value = max_sweep_value

        if self.verbose >= 1:
            print(f"\nSweeping over {self.n_sweep_values} values of the"
                  f" '{self.sweep_type}' parameter!")

        ## kNN graph parameters
        self.kNN_metric = kNN_metric
        self.kNN_alg = kNN_alg.lower()
        self.kNN_params = kNN_params

        ## Affinity matrix parameters
        self.aff_type = aff_type.lower()
        self.aff_params = aff_params

        ## Dimensionality reduction parameters
        self.n_components = int(n_components)
        self.DRA = DRA.lower()
        self.DRA_params = DRA_params

        ## Embedding statistic parameters
        self.EES_type = EES_type.lower()
        self.EES_params = EES_params
        self.pVal_type = pVal_type.lower()

        self._use_t_stat = bool(use_t_stat_for_opt)
        self._log_t = bool(use_log_t_stat)

        ## Runtime parameters
        err_str = "Number of data embeddings must be > 0."
        assert int(n_data_embed) > 0, err_str
        self.n_data_embed = int(n_data_embed)

        err_str = "Number of null embeddings must be >= 0."
        assert int(n_null_embed) >= 0, err_str
        self.n_null_embed = int(n_null_embed)

        self.n_jobs = int(n_jobs)
        self._allow_sparse = bool(allow_sparse)
        self._keep_affmats = bool(keep_affmats)
        self.rs = check_random_state(random_state)
        self._seed = self.rs.get_state()[1][0]

        ## File I/O parameters
        self.do_cache = bool(do_cache)
        self.project_name = project_name
        self.project_dir = project_dir
        if self.do_cache:
            if path.isdir(project_dir):
                self.project_dir = project_dir
            else:
                err_str  = f"Couldn't find project directory `{project_dir}`."
                err_str += f" Please make sure this is a valid directory or"
                err_str += f" turn off file caching (set `do_cache=False`)."
                raise OSError(err_str)

            if not path.isdir(path.join(project_dir, project_name)):
                if self.verbose >= 5:
                    print(f"Creating project directory "
                          f"{path.join(project_dir, project_name)}")
                os.mkdir(path.join(project_dir, project_name))
        else:
            if self.verbose >= 0:
                warn_str  = f"\nWARNING: caching has been set to `False`. It"
                warn_str += f" is strongly recommended that file caching be"
                warn_str += f" used (`do_cache=True`) to save on time/memory!"
                print(warn_str)
        return

    def fit(self, X):

        ## Check that the data is a 2D array
        self.data_X = check_array(X, accept_sparse=True, ensure_2d=True)
        ## If the data aren't sparse but could be, sparcify!
        if not sp.issparse(self.data_X) and (np.mean(self.data_X == 0) > 0.1):
            self.data_X = sp.csr_matrix(self.data_X)

        ## Get the data shape
        self.n_samples, self.n_features = self.data_X.shape

        if self.verbose >= 1:
            print(f"\nFitting '{self.sweep_type}' sweep!")

        ## Check hyperparameter values.
        self._check_hyperparameters()

        self.embeddings = {}
        self.pValues    = {}
        self.data_EES   = {}
        self.null_EES   = {}

        ## Loop over the values of the hyperparameters!
        for ii, hp in enumerate(self.sweep_values):
            if self.verbose >= 1:
                print(f"\nFitting data with '{self.sweep_type}' = {hp}"
                      f" ({ii + 1} / {self.n_sweep_values})")

            perp = None
            knn = None
            if self.sweep_type == 'perplexity':
                perp = hp
            else:
                knn = hp
            embObj = EMBEDR(perplexity=perp,
                            n_neighbors=knn,
                            kNN_metric=self.kNN_metric,
                            kNN_alg=self.kNN_alg,
                            kNN_params=self.kNN_params,
                            aff_type=self.aff_type,
                            aff_params=self.aff_params,
                            n_components=self.n_components,
                            DRA=self.DRA,
                            DRA_params=self.DRA_params,
                            EES_type=self.EES_type,
                            EES_params=self.EES_params,
                            pVal_type=self.pVal_type,
                            n_data_embed=self.n_data_embed,
                            n_null_embed=self.n_null_embed,
                            n_jobs=self.n_jobs,
                            allow_sparse=self._allow_sparse,
                            keep_affmats=self._keep_affmats,
                            random_state=self.rs,
                            verbose=self.verbose,
                            do_cache=self.do_cache,
                            project_name=self.project_name,
                            project_dir=self.project_dir)

            if hasattr(self, 'hash'):
                embObj.hash = self.hash

            embObj.fit(self.data_X)

            self.embeddings[hp] = embObj.data_Y.copy()
            self.data_EES[hp]   = embObj.data_EES.copy()
            self.null_EES[hp]   = embObj.null_EES.copy()
            self.pValues[hp]    = embObj.pValues.copy()

        return

    def _check_hyperparameters(self):

        ## If values for the sweep weren't provided, generate an array based on
        ## the min/max and n_sweep parameters.
        if self.sweep_values is None:

            if self.verbose >= 3:
                print(f"No sweep values provided, generating array"
                      f" automatically!")

            ## If the minimum sweep value is < 1, interpret it relative to
            ## the number of samples.
            if self.min_sweep_value < 1:

                ## Floats are ok for `perplexity`
                if self.sweep_type == 'perplexity':
                    min_val = float(self.n_samples * self.min_sweep_value)
                ## `n_neighbors` must be an integer.
                elif self.sweep_type == 'n_neighbors':
                    min_val = int(self.n_samples * self.min_sweep_value)
                    min_val = np.max([1, min_val])

            ## Check that the minimum sweep value is < `n_samples`.
            elif self.min_sweep_value > self.n_samples:
                warn_str  = f"Warning: `min_sweep_value`="
                warn_str += f"{self.min_sweep_value} is larger than"
                warn_str += f" `n_samples`={self.n_samples}. Clipping"
                warn_str += f" values to be in [1, {self.n_samples}."

                min_val = self.n_samples
                print(warn_str)

            else:
                min_val = self.min_sweep_value

            if self.verbose >= 5:
                print(f"Min sweep value is {self.min_sweep_value}, which"
                      f" corresponds to '{self.sweep_type}' = {min_val}.")

            ## If the maximum sweep value is < 1, interpret it relative to
            ## the number of samples.
            if self.max_sweep_value < 1:
                if self.sweep_type == 'perplexity':
                    max_val = float(self.n_samples * self.max_sweep_value)
                elif self.sweep_type == 'n_neighbors':
                    max_val = int(self.n_samples * self.max_sweep_value)
                    max_val = np.max([1, max_val])

            ## Check that the maximum sweep value is < `n_samples`.
            elif self.max_sweep_value > self.n_samples:
                warn_str  = f"Warning: `max_sweep_value`="
                warn_str += f"{self.max_sweep_value} is larger than"
                warn_str += f" `n_samples`={self.n_samples}. Clipping"
                warn_str += f" values to be in [1, {self.n_samples}."

                max_val = self.n_samples
                print(warn_str)

            else:
                max_val = self.max_sweep_value

            if self.verbose >= 5:
                print(f"Max sweep value is {self.max_sweep_value}, which"
                      f" corresponds to '{self.sweep_type}' = {max_val}.")

            if min_val > max_val:
                err_str =  f"Invalid values for `min_sweep_value` and"
                err_str += f" `max_sweep_value`... Minimum value must be less"
                err_str += f" than the maximum value, not {min_val}>{max_val}."
                raise ValueError(err_str)

            out = np.logspace(np.log10(min_val), np.log10(max_val),
                              self.n_sweep_values)[::-1]

        ## If values for the sweep *were* provided, we need to make sure they
        ## are legal with the size of the data.
        else:
            ## If the sweep values are provided as relative values...
            if np.all(self.sweep_values < 1):
                if self.verbose >= 3:
                    print(f"All hyperparameter values provided as fractions"
                          f" of the number of samples!")
                out = np.clip(self.n_samples * self.sweep_values, 1,
                              self.n_samples)

            ## If the sweep values are provided as absolute values...
            elif np.all(self.sweep_values >= 1):
                if self.verbose >= 3:
                    print(f"All hyperparameter values provided absolutely!")
                out = np.clip(self.sweep_values, 1, self.n_samples)

            else:
                raise ValueError(f"Cannot mix relative and absolute"
                                 f" specification of `sweep_values`.")

        out = np.round(out, 2)

        if self.sweep_type == 'n_neighbors':
            out = out.astype(int)

        self.sweep_values = np.sort(np.unique(out)).squeeze()[::-1]

        if len(self.sweep_values) < self.n_sweep_values:

            if self.verbose >= 3:
                print(f"Warning: after removing duplicate values, the number"
                      f" of hyperparameter values in the sweep is "
                      f"{len(self.sweep_values)} (not {self.n_sweep_values}).")

            self.n_sweep_values = len(self.sweep_values)

        if self.verbose >= 5:
            print(f"Hyperparameter array has been set as:")
            print(self.sweep_values)

    def get_optimal_hyperparameters(self):

        pVal_arr = np.asarray([self.pValues[hp] for hp in self.sweep_values])

        if self._use_t_stat:

            if self.verbose >= 3 and not self._log_t:
                print(f"Setting optimal {self.sweep_type} using normal fits.")
            elif self.verbose >= 3:
                print(f"Setting optimal {self.sweep_type} using log-normal"
                      f" fits.")

            t_stats = np.zeros((self.n_sweep_values, self.n_samples))

            all_dEES = np.asarray([self.data_EES[hp]
                                   for hp in self.sweep_values])
            all_nEES = np.asarray([self.null_EES[hp]
                                   for hp in self.sweep_values])
            all_nEES = all_nEES.reshape(self.n_sweep_values, -1)

            if self._log_t:
                all_dEES = np.log(all_dEES)
                all_nEES = np.log(all_nEES)

            dEES_means = all_dEES.mean(axis=1)
            dEES_stds  = all_dEES.std(axis=1)
            dEES_N     = self.n_data_embed
            nEES_means = all_nEES.mean(axis=1)
            nEES_stds  = all_nEES.std(axis=1)
            nEES_N     = self.n_null_embed * self.n_samples

            for hpNo in range(self.n_sweep_values):
                t_res = st.ttest_ind_from_stats(dEES_means[hpNo],
                                                dEES_stds[hpNo],
                                                dEES_N,
                                                nEES_means[hpNo],
                                                nEES_stds[hpNo],
                                                nEES_N,
                                                equal_var=False)
                t_stats[hpNo] = t_res.statistic

            opt_sweep_values = np.zeros((self.n_samples))

            for ii in range(self.n_samples):
                ## Get the p-values for sample ii at each hyperparam value
                pVal_row = pVal_arr[:, ii]
                ## Locate the minimal p-values
                min_pVal_idx = (pVal_row == pVal_row.min()).nonzero()[0]
                ## Locate the t-stats corresponding to those minimal p-values
                min_pVal_t = t_stats[min_pVal_idx, ii]
                ## Locate the minimal t-stat
                min_pVal_t_idx = min_pVal_t.argmin()
                ## Set the optimal hyperparam to be the min-t / min-hp value
                opt_val = self.sweep_values[min_pVal_idx[min_pVal_t_idx]]
                opt_sweep_values[ii] = opt_val

        # if self._use_min_EES:
            # dEES_arr = np.asarray([np.median(self.data_EES[hp], axis=0)
            #                        for hp in self.sweep_values])

            # opt_sweep_values = np.zeros((self.n_samples))

            # for ii in range(self.n_samples):
            #     pVal_row = pVal_arr[:, ii]
            #     min_pVal_idx = (pVal_row == pVal_row.min()).nonzero()[0]
            #     min_pVal_EES = dEES_arr[min_pVal_idx, ii]

            #     min_pVal_EES_idx = min_pVal_EES.argmin()
            #     opt_val = self.sweep_values[min_pVal_idx[min_pVal_EES_idx]]

            #     opt_sweep_values[ii] = opt_val

        else:
            if self.verbose >= 3:
                print(f"Setting optimal {self.sweep_type} using minimum.")

            opt_hp_idx = np.argmin(pVal_arr[::-1], axis=0)
            opt_sweep_values = self.sweep_values[::-1][opt_hp_idx].squeeze()

        if self.verbose >= 5:
            print(f"Returning optimal '{self.sweep_type}' values!")

        return opt_sweep_values

    def fit_samplewise_optimal(self):

        try:
            opt_hp_vals = self.get_optimal_hyperparameters()
        except AttributeError:
            err_str  = f"A hyperparameter sweep must be run before a sample"
            err_str += f"-wise optimal embedding can be generated!"
            raise AttributeError(err_str)

        perp = knn = None
        if self.sweep_type == 'perplexity':
            perp = opt_hp_vals
        else:
            knn = opt_hp_vals
        optObj = EMBEDR(perplexity=perp,
                        n_neighbors=knn,
                        kNN_metric=self.kNN_metric,
                        kNN_alg=self.kNN_alg,
                        kNN_params=self.kNN_params,
                        aff_type=self.aff_type,
                        aff_params=self.aff_params,
                        n_components=self.n_components,
                        DRA=self.DRA,
                        DRA_params=self.DRA_params,
                        EES_type=self.EES_type,
                        EES_params=self.EES_params,
                        pVal_type=self.pVal_type,
                        n_data_embed=self.n_data_embed,
                        n_null_embed=self.n_null_embed,
                        n_jobs=self.n_jobs,
                        random_state=self.rs,
                        verbose=self.verbose,
                        do_cache=self.do_cache,
                        project_name=self.project_name,
                        project_dir=self.project_dir)

        optObj.fit(self.data_X)



        self.opt_obj            = optObj
        self.opt_embed          = optObj.data_Y[:]
        self.opt_embed_data_EES = optObj.data_EES[:]
        self.opt_embed_null_EES = optObj.null_EES[:]
        self.opt_embed_pValues  = optObj.pValues[:]
