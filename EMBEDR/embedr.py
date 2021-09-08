
import EMBEDR.affinity as aff
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
from sklearn.utils import check_array, check_random_state


class EMBEDR(object):

    def __init__(self,
                 # Important hyperparameters
                 perplexity=None,
                 n_neighbors=None,
                 # kNN graph parameters
                 metric='euclidean',
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
        self.kNN_metric = metric
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

        ## File I/O parameters
        self.do_cache = bool(do_cache)
        self.project_name = project_name
        if self.do_cache:
            if path.isdir(project_dir):
                self.project_dir = project_dir
            else:
                err_str = f"Couldn't find project directory `{project_dir}`."
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

        print(self)

    def __str__(self):
        out_str = ""
        if self.verbose > 0:
            out_str += f"\n\n\tEMBEDR Class v{ev}\n" + 35 * "=" + "\n\n"
        return out_str

    def fit(self, X):

        #####################
        ## Fit to the DATA ##
        #####################

        ## Check that the data is a 2D array
        self.data_X = check_array(X, accept_sparse=True, ensure_2d=True)
        ## If the data aren't sparse but could be, sparcify!
        if not sp.issparse(self.data_X) and (np.mean(self.data_X == 0) > 0.1):
            self.data_X = sp.csr_matrix(self.data_X)

        ## Get the data shape
        self.n_samples, self.n_features = self.data_X.shape

        ## Get the object hash
        self.hash = self.get_hash()

        ## Create project directory
        self.project_hdr = self.get_project_hdr()

        ## Set up n_neighbors and perplexity hyperparameters
        self._set_hyperparameters()

        ## Finally, we can do the computations!
        self._fit(null_fit=False)

        #####################
        ## Fit to the NULL ##
        #####################

        self._fit(null_fit=True)

        ## Get p-Values
        self.calculate_pValues()

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

            nn_arr = np.ones((self.n_samples)) * self.n_neighbors

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
        self._max_nn = self._nn_arr.max()

    def get_hash(self):

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

        # Add the data chunks that's been converted to bytes.
        hash_str += data_chunk.tobytes()

        # Digest the bytes into a hash!
        return hashlib.md5(hash_str).hexdigest()

    def get_project_hdr(self):

        # If we're not caching, then just skip everything.
        if not self.do_cache:
            return None

        ## First we 

        # First we set up the subfolder for this data set (hash)
        project_path = path.join(self.project_dir,
                                 self.project_name,
                                 self.hash)
        if not path.isdir(project_path):
            os.mkdir(project_path)
        self.project_path = project_path

        # Get the expected project header file
        proj_hdr_name = self.project_name + "_header.json"
        proj_hdr_path = path.join(self.project_path, proj_hdr_name)

        # If it already exists, load it up!
        if path.isfile(proj_hdr_path):
            with open(proj_hdr_path, 'r') as f:
                proj_hdr = json.load(f)

        # Otherwise, initialize a new header.
        else:
            proj_hdr = {'project_name': self.project_name,
                        'project_path': self.project_path}

            self.set_project_hdr(proj_hdr)

        return proj_hdr

    def set_project_hdr(self, hdr):

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
            self.data_kNN = self._get_kNN_graph(self.data_X)

            ## If we're using t-SNE to embed or DKL as the EES, we need an
            ## affinity matrix.
            if (self.DRA in ['tsne', 't-sne']) or (self.EES_type == 'dkl'):
                self.data_P = self._get_affinity_matrix(self.data_kNN)

            ## We then need to get the requested embeddings.
            if (self.DRA in ['tsne', 't-sne']):
                dY, dEES = self._get_tSNE_embedding(self.data_P)

            elif (self.DRA in ['umap']):
                print(f"WARNING: UMAP has not been implemented!")
                # dY, dEES = self._get_UMAP_embedding(self.data_P, null_fit)

            self.data_Y = dY[:]
            self.data_EES = dEES[:]

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
                nKNN = self._get_kNN_graph(null_X, null_fit)
                self.null_kNN[nNo] = nKNN

                ## If we need an affinity matrix...
                if (self.DRA in ['tsne', 't-sne']) or (self.EES_type == 'dkl'):
                    nP = self._get_affinity_matrix(nKNN, null_fit)
                    self.null_P[nNo] = nP

                ## We then need to get the requested embeddings.
                if (self.DRA in ['tsne', 't-sne']):
                    nY, nEES = self._get_tSNE_embedding(nP, null_fit)

                elif (self.DRA in ['umap']):
                    print(f"WARNING: UMAP has not been implemented!")
                    # nY, nEES = self._get_UMAP_embedding(nP, null_fit)

                self.null_Y[nNo] = nY
                self.null_EES[nNo] = nEES

                self._null_seed += 100

    def _get_kNN_graph(self, X, null_fit=False):

        ## First we initialize a kNN graph with the input parameters.
        seed = self._null_seed if null_fit else self._seed
        tmp_kNN = self._initialize_kNN_index(X, seed)

        ## If we're doing file caching, first we want to try and load the graph
        if self.do_cache:

            ## Check if *any* kNN graphs have been made.
            if 'kNN' not in self.project_hdr:
                if self.verbose >= 2:
                    print(f"No kNN graphs have been made yet for this data"
                          f" and/or seed.")

                self.project_hdr['kNN'] = dict(Data={}, Null={})

            ## If kNN graphs have been fit...
            else:

                if null_fit:
                    if self.verbose >= 3:
                        print(f"Looking for matching kNN graph in"
                              f" null cache.")

                    kNN_hdr = self.project_hdr['kNN']['Null']
                else:
                    if self.verbose >= 3:
                        print(f"Looking for matching kNN graph in"
                              f" data cache.")

                    kNN_hdr = self.project_hdr['kNN']['Data']

                ## Look in the cache for a matching graph.
                kNN_path = self._match_kNN_graph(tmp_kNN, kNN_hdr, seed)

                # If a path has been found to a matching kNN graph load it!
                if kNN_path is not None:
                    if self.verbose >= 3:
                        print(f"Attempting to load kNN graph...")

                    with open(kNN_path, 'rb') as f:
                        out = pkl.load(f)

                    if self.verbose >= 3:
                        print(f"kNN graph loaded!  Checking shape...")

                    if out.kNN_dst.shape[1] < self._max_nn:
                        print(f"Not enough neighbors, querying for more!")
                        idx, dst = out.query(X, self._max_nn + 1)
                        out.kNN_dst = dst[:, 1:]
                        out.kNN_idx = idx[:, 1:]

                    ## Return the loaded kNN graph.
                    return out

        ## If we're not caching, no kNNs have been made, or a matching kNN
        ## couldn't be found, then we need to fit a new one!
        if self.verbose >= 1:
            print(f"\nFitting kNN graph to data with k={self._max_nn}"
                  f" neighbors!")

        tmp_kNN.fit(X, self._max_nn)

        ## Finally, if we're caching and a kNN graph has just been fit, then
        ## add it to the header and save it to the cache.
        if self.do_cache:

            ## These are the header attributes we use to distinguish kNN graphs
            kNN_subhdr = {'kNN_alg': tmp_kNN.__class__.__name__,
                          'metric': tmp_kNN.metric,
                          'params': self.kNN_params,
                          'seed': int(seed)}

            ## Increment the filename by 1
            if null_fit:
                kNN_hdr = self.project_hdr['kNN']['Null']
                n_kNN_made = len(kNN_hdr)
                kNN_name = f"Null_kNN_{n_kNN_made:04d}.knn"
            else:
                kNN_hdr = self.project_hdr['kNN']['Data']
                n_kNN_made = len(kNN_hdr)
                kNN_name = f"Data_kNN_{n_kNN_made:04d}.knn"

            tmp_kNN._filename = kNN_name

            ## If the kNN graph was generated with the Annoy algorithm, then
            ## we need a separate filename to cache the index.
            if isinstance(tmp_kNN, nn.Annoy):
                pkl_name = kNN_name[:-4] + ".ann"
                tmp_kNN.pickle_name = path.join(self.project_path, pkl_name)

            ## Save the graph to file.
            if self.verbose >= 2:
                print(f"Caching {kNN_name} to file!")

            kNN_path = path.join(self.project_path, kNN_name)
            with open(kNN_path, 'wb') as f:
                pkl.dump(tmp_kNN, f)

            ## Update the project header.
            kNN_hdr[kNN_path] = kNN_subhdr
            self.set_project_hdr(self.project_hdr)

        return tmp_kNN

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

    def _match_kNN_graph(self, tmp_kNN, kNN_hdr, seed):
        kNN_path = None

        for tmp_path, kNN_subhdr in kNN_hdr.items():

            if self.verbose >= 5:
                print(f"Checking kNN graph at {tmp_path}...")

            ## Check that the algorithm (subclass) matches
            if tmp_kNN.__class__.__name__ != kNN_subhdr['kNN_alg']:
                if self.verbose >= 5:
                    print(f"...kNN algorithm didn't match!")
                continue
            ## Check that the metric matches
            if tmp_kNN.metric != kNN_subhdr['metric']:
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
            if self.verbose >= 2:
                print(f"kNN graph at {tmp_path} passed all checks!")

            kNN_path = tmp_path
            break

        return kNN_path

    def _get_affinity_matrix(self, X, null_fit=False):

        ## Initialize an affinity matrix based on the given parameters.
        tmp_aff = self._initialize_affinity_matrix(X)

        ## If we're doing file caching...
        if self.do_cache:

            ## Check if *any* affinity matrices have been made...
            if "affinity_matrix" not in self.project_hdr:
                if self.verbose >= 2:
                    print(f"No affinity matrices have been made yet for this"
                          f" data and/or seed.")

                self.project_hdr['affinity_matrix'] = dict(Data={}, Null={})

            ## If affinity matrices have been cached...
            else:

                if null_fit:
                    if self.verbose >= 3:
                        print(f"\nLooking for matching affinity matrix in"
                              f" null cache")

                    aff_hdr = self.project_hdr['affinity_matrix']['Null']

                else:
                    if self.verbose >= 3:
                        print(f"\nLooking for matching affinity matrix in"
                              f" data cache")

                    aff_hdr = self.project_hdr['affinity_matrix']['Data']

                aff_path = self._match_affinity_matrix(tmp_aff, X, aff_hdr)

                ## If a path has been found to a matching affinity matrix,
                ## load it!
                if aff_path is not None:
                    if self.verbose >= 3:
                        print(f"Attempting to load affinity matrix...")

                    with open(aff_path, 'rb') as f:
                        out = pkl.load(f)

                    ## Check if the actual matrix, P, was saved...
                    try:
                        _ = out.P
                    ## If P wasn't saved, recalculate it!
                    except AttributeError:
                        out.P = out.calculate_affinities(self.data_kNN,
                                                         recalc=True)

                    if self.verbose >= 3:
                        print(f"Affinity matrix successfully loaded!")

                    return out

        ## If we're not caching, no affinities have been made, or a matching
        ## affinity matrix couldn't be found, then we need to fit a new one!
        tmp_aff.fit(X)

        ## Finally, if we're caching, then add the fitted affinity matrix to
        ## the header and save the affinity matrix to the cache.
        if self.do_cache:

            if null_fit:
                aff_hdr = self.project_hdr['affinity_matrix']['Null']
                n_aff_made = len(aff_hdr)
                aff_name = f"Null_AffMat_{n_aff_made:04d}.aff"
            else:
                aff_hdr = self.project_hdr['affinity_matrix']['Data']
                n_aff_made = len(aff_hdr)
                aff_name = f"Data_AffMat_{n_aff_made:04d}.aff"

            tmp_aff._filename = aff_name

            if self.verbose >= 2:
                print(f"Caching {aff_name} to file!")

            ## To save space when caching, we don't save the calculated
            ## affinity matrix, P, but the info we need to quickly regenerate
            ## it instead.
            P = tmp_aff.P.copy()
            del tmp_aff.P

            aff_path = path.join(self.project_path, aff_name)
            with open(aff_path, 'wb') as f:
                pkl.dump(tmp_aff, f)

            ## Within the object however, we keep the matrix.
            tmp_aff.P = P.copy()

            ## Get the kernel parameters and convert any arrays to lists.
            tmp_kernel_params = tmp_aff.kernel_params.copy()
            for k in tmp_kernel_params:
                if isinstance(tmp_kernel_params[k], np.ndarray):
                    tmp_kernel_params[k] = tmp_kernel_params[k].tolist()

            ## These are the parameters used to match affinity matrices
            aff_subhdr = dict(aff_type=tmp_aff.__class__.__name__,
                              n_neighbors=int(tmp_aff.n_neighbors),
                              kernel_params=tmp_kernel_params,
                              kNN_alg=X.__class__.__name__,
                              kNN_params=tmp_aff.kNN_params,
                              kNN_filename=X._filename,
                              normalization=tmp_aff.normalization,
                              symmetrize=tmp_aff.symmetrize)
            ## Don't match perplexity if it isn't set.
            if self.perplexity is not None:
                aff_subhdr['perp_arr'] = tmp_aff._perp_arr.tolist()

            ## Reset the project header json.
            aff_hdr[aff_path] = aff_subhdr
            self.set_project_hdr(self.project_hdr)

        return tmp_aff

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

        out = aff._initialize_affinity_matrix(X,
                                              aff_type=self.aff_type,
                                              perplexity=self._perp_arr,
                                              n_neighbors=self._max_nn,
                                              kNN_params=kNN_params,
                                              n_jobs=self.n_jobs,
                                              random_state=self.rs,
                                              verbose=self.verbose,
                                              kernel_params=kernel_params,
                                              **self.aff_params)

        # self.aff_params.update({'kernel_params': out.kernel_params})

        return out[0]

    def _match_affinity_matrix(self, tmp_aff, X, aff_hdr):

        ## Get the kernel parameters and convert any arrays to lists.
        tmp_kernel_params = tmp_aff.kernel_params.copy()
        for k in tmp_kernel_params:
            if isinstance(tmp_kernel_params[k], np.ndarray):
                tmp_kernel_params[k] = tmp_kernel_params[k].tolist()

        ## If we have made affinity matrices(s) for this project, cycle
        ## through them until we get matching parameters.  We want to
        ## then load from the corresponding filepath.
        aff_path = None
        for tmp_path, aff_subhdr in aff_hdr.items():

            if self.verbose >= 5:
                print(f"Checking affinity matrix at {tmp_path}...")

            ## Check that... the affinity matrix type matches
            if tmp_aff.__class__.__name__ != aff_subhdr['aff_type']:
                if self.verbose >= 5:
                    print(f"...affinity type doesn't match!")
                continue

            if tmp_aff.n_neighbors != aff_subhdr['n_neighbors']:
                if self.verbose >= 5:
                    print(f"...number of neighbors doesn't match!")
                continue

            ## Check that... the kernel parameters match
            if np.any([val != aff_subhdr['kernel_params'][key]
                       for key, val in tmp_kernel_params.items()]):
                if self.verbose >= 5:
                    print(f"...kernel parameters don't match!")
                continue

            ## Check that... the type of kNN kernel matches
            if X.__class__.__name__ != aff_subhdr['kNN_alg']:
                if self.verbose >= 5:
                    print(f"...kNN algorithm doesn't match!")
                continue

            ## Check that... the kNN parameters upon which the matrix
            ## is built match.
            if tmp_aff.kNN_params != aff_subhdr['kNN_params']:
                if self.verbose >= 5:
                    print(f"...kNN parameters don't match!")
                continue

            if X._filename != aff_subhdr['kNN_filename']:
                if self.verbose >= 5:
                    print(f"...kNN graph (name) doesn't match!")
                continue

            ## Check that... the matrix normalization matches
            if tmp_aff.normalization != aff_subhdr['normalization']:
                if self.verbose >= 5:
                    print(f"...affinity normalization doesn't match!")
                continue

            ## Check that... the matrix symmetry matches
            if tmp_aff.symmetrize != aff_subhdr['symmetrize']:
                if self.verbose >= 5:
                    print(f"...affinity symmetry doesn't match!")
                continue

            if self.perplexity is not None:
                if np.any(tmp_aff._perp_arr != aff_subhdr['perp_arr']):
                    if self.verbose >= 5:
                        print(f"...perplexities don't match!")
                    continue

            # If all checks are passed, save the corresponding path!
            if self.verbose >= 2:
                print(f"Affinity matrix at {tmp_path} passed all"
                      f" checks!")

            aff_path = tmp_path
            break

        return aff_path

    def _get_tSNE_embedding(self, P, null_fit=False):

        ## Get the number of requested embeddings
        n_embeds_made = 0
        n_embeds_reqd = 1 if null_fit else self.n_data_embed
        n_embeds_2_make = n_embeds_reqd

        ## Initialize an embedding using the input arguments.
        tmp_Y = self._initialize_tSNE_embed(P)

        ## If we're doing file caching...
        if self.do_cache:

            ## Check to see if any embeddings have ever been made...
            if 'Embed_tSNE' not in self.project_hdr:

                if self.verbose >= 3:
                    print(f"\nNo t-SNE embeddings have been made for this"
                          f" data and/or seed!")

                self.project_hdr['Embed_tSNE'] = dict(Data={}, Null={})

                dra_path = None

            ## If embeddings have been made...
            else:

                if null_fit:
                    if self.verbose >= 3:
                        print(f"Looking for matching t-SNE embeddings in"
                              f" null cache.")

                    dra_hdr = self.project_hdr['Embed_tSNE']['Null']
                else:
                    if self.verbose >= 3:
                        print(f"Looking for matching t-SNE embeddings in"
                              f" data cache.")

                    dra_hdr = self.project_hdr['Embed_tSNE']['Data']

                ## Look in the cache for a matching graph.
                dra_path = self._match_tSNE_embeds(tmp_Y, P, dra_hdr)

                ## If a matching embedding exists...
                if dra_path is not None:

                    ## Load it!
                    if self.verbose >= 3:
                        print(f"Attempting to load t-SNE embeddings...")

                    with open(dra_path, 'rb') as f:
                        tmp_Y, tmp_EES = pkl.load(f)

                    ## Load up to the requested number of embeddings!
                    tmp_Y = tmp_Y[:n_embeds_reqd]

                    ## Check that there are the required number of embeddings
                    n_embeds_made = tmp_Y.shape[0]

                    if self.verbose >= 3:
                        print(f"{n_embeds_made} t-SNE embeddings loaded!"
                              f" ({n_embeds_reqd} requested)")

                    ## If there are not enough embeddings, then we'll make more
                    n_embeds_2_make = n_embeds_reqd - n_embeds_made

                    ## If there are enough embeddings, return them!
                    if n_embeds_2_make == 0:
                        return tmp_Y, tmp_EES

        ## If there aren't enough embeddings, we fit them here!
        tmp_embed_arr = np.zeros((n_embeds_2_make,
                                  self.n_samples,
                                  self.n_components))
        for ii in range(n_embeds_2_make):
            if self.verbose >= 1:
                print(f"\nFitting embedding {ii + 1}/{n_embeds_2_make}"
                      f" ({n_embeds_reqd} total requested).")

            ## We make sure to increment the random seed so that the
            ## intitializations are offset in each iteration.
            seed_offset = n_embeds_made + ii

            tmp_embed = tSNE_Embed(n_components=self.n_components,
                                   perplexity=self.perplexity,
                                   n_jobs=self.n_jobs,
                                   random_state=self._seed + seed_offset,
                                   verbose=self.verbose,
                                   **self.DRA_params)

            tmp_embed.fit(P)

            tmp_embed_arr[ii] = tmp_embed.embedding[:]

        ## Calculate the EES
        tmp_EES = self.calculate_EES(P.P, tmp_embed_arr)

        if n_embeds_made > 0:
            tmp_embed_arr = np.vstack((tmp_Y, tmp_embed_arr))

        if self.verbose >= 4:
            print(f"Checking that the `tmp_embed_arr` has size"
                  f"{tmp_embed_arr.shape}")

        ## Then if we re caching files, we need to save these embeddings!
        if self.do_cache:

            if dra_path is None:
                if null_fit:
                    dra_hdr = self.project_hdr['Embed_tSNE']['Null']
                    n_times_made = len(dra_hdr)
                    dra_name = f"Null_tSNE_Embed_{n_times_made:04d}.emb"
                else:
                    dra_hdr = self.project_hdr['Embed_tSNE']['Data']
                    n_times_made = len(dra_hdr)
                    dra_name = f"Data_tSNE_Embed_{n_times_made:04d}.emb"

                if self.verbose >= 2:
                    print(f"Caching {dra_name} to file!")

                dra_path = path.join(self.project_path, dra_name)

            else:
                dra_name = dra_path.split("/")[-1]
                if self.verbose >= 2:
                    print(f"Appending to cache {dra_name}!")

            with open(dra_path, 'wb') as f:
                pkl.dump([tmp_embed_arr, tmp_EES], f)

            tmp_tSNE_params = self._get_matchable_tSNE_params(tmp_embed)

            dra_subhdr = dict(DRA_params=tmp_tSNE_params,
                              affmat_filename=P._filename)

            ## Reset the project header json.
            dra_hdr[dra_path] = dra_subhdr
            self.set_project_hdr(self.project_hdr)

        return tmp_embed_arr, tmp_EES

    def _initialize_tSNE_embed(self, P):

        if self.verbose >= 3:
            print(f"\nInitializing t-SNE Embedding...")

        out = tSNE_Embed(n_components=self.n_components,
                         perplexity=self._perp_arr,
                         n_jobs=self.n_jobs,
                         random_state=self._seed,
                         verbose=self.verbose,
                         **self.DRA_params)

        out.initialize_embedding(P)

        # keep_params = ['initialization',
        #                'learning_rate',
        #                'n_early_iter',
        #                'early_exag',
        #                'early_mom',
        #                'n_iter',
        #                'exag',
        #                'mom',
        #                'max_grad_norm',
        #                'max_step_norm',
        #                't_dof',
        #                'neg_grad_method',
        #                'bh_theta',
        #                'FI_n_interp_pts',
        #                'FI_min_n_interv',
        #                'FI_ints_per_interv']
        # init_params = {}
        # for key, val in out.__dict__.items():
        #     if key not in keep_params:
        #         continue
        #     if isinstance(val, np.ndarray):
        #         init_params[key] = val.tolist()
        #     else:
        #         init_params[key] = val

        # init_params = self._get_matchable_tSNE_params(out)

        # self.DRA_params.update(self._get_matchable_tSNE_params(out))

        # ##    n_components=2,
        # ##    perplexity=None,
        # initialization='random',
        # learning_rate=None,
        # n_early_iter=250,
        # early_exag=None,
        # early_mom=0.5,
        # n_iter=1000,
        # exag=1.,
        # mom=0.8,
        # max_grad_norm=None,
        # max_step_norm=5,
        # t_dof=1.,
        # neg_grad_method="fft",
        # bh_theta=0.5,
        # FI_n_interp_pts=3,
        # FI_min_n_interv=50,
        # FI_ints_per_interv=1,
        # callbacks=None,
        # iter_per_callback=50,
        # iter_per_log=50,
        # ##    n_jobs=1,
        # ##    random_state=None,
        # ##    verbose=1)

        return out

    def _get_matchable_tSNE_params(self, obj):

        ## Get the t-SNE parameters and convert any arrays to lists.
        matchable_params = ['initialization', 'learning_rate', 'n_early_iter',
                            'early_exag', 'early_mom', 'n_iter', 'exag', 'mom',
                            'max_grad_norm', 'max_step_norm', 't_dof',
                            'neg_grad_method', 'bh_theta', 'FI_n_interp_pts',
                            'FI_min_n_interv', 'FI_ints_per_interv']
        tSNE_params = {}
        for key, val in obj.__dict__.items():
            if key not in matchable_params:
                continue
            if isinstance(val, np.ndarray):
                tSNE_params[key] = val.tolist()
            else:
                tSNE_params[key] = val

        return tSNE_params

    def _match_tSNE_embeds(self, tmp_Y, P, dra_hdr):

        ## Get the matchable t-SNE embedding parameters
        tmp_tSNE_params = self._get_matchable_tSNE_params(tmp_Y)

        dra_path = None

        ## We want to cycle through the available embeddings and see if they
        ## match the given parameters.
        for tmp_path, dra_subhdr in dra_hdr.items():

            ## Match the t-SNE runtime parameters
            if dra_subhdr['DRA_params'] != tmp_tSNE_params:
                continue

            ## Match the affinity matrix.
            if P._filename != dra_subhdr['affmat_filename']:
                continue

            # If all checks are passed, save the corresponding path!
            if self.verbose >= 2:
                print(f"t-SNE embedding at {tmp_path} passed all checks!")
            dra_path = tmp_path
            break

        return dra_path

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

        DKL = ees.calculate_DKL(P.indices, P.indptr, P.data, tmpY)

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


class EMBEDR_sweep(object):

    def __init__(self,):
        return
