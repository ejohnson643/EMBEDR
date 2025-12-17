from collections import Counter
import numpy as np
import os
from time import time

###############################################################################
##  Output logging utilities.
###############################################################################


class Timer:
    """Wrapper for timing methods with a message."""

    def __init__(self, message, verbose=0):
        self.message = message
        self.start_time = time()
        self.verbose = verbose

    def __enter__(self):
        if self.verbose >= 0:
            print("--->", self.message)

    def __exit__(self):
        end = time()
        dt = end - self.start_time
        if self.verbose >= 0:
            print(f"---> Time Elapsed: {dt:.2g} seconds!")


###############################################################################
##  Data I/O Functions
###############################################################################
tabula_muris_tissues = ['marrow', 'diaphragm', 'brain_non-myeloid']


def load_data(data_name,
              data_dir=None,
              dtype=float,
              load_metadata=True):

    import scanpy as sc
    import pandas as pd

    metadata = None
    if data_name.lower() == 'mnist':

        if data_dir is None:
            data_dir = "./data/"

        data_path = os.path.join(data_dir, "mnist2500_X.txt")

        X = np.loadtxt(data_path).astype(dtype)

        if load_metadata:
            metadata_path = os.path.join(data_dir, "mnist2500_labels.txt")
            metadata = np.loadtxt(metadata_path).astype(int)
            metadata = pd.DataFrame(metadata, columns=['label'])

    elif data_name.lower() in tabula_muris_tissues:

        if data_dir is None:
            data_dir = "./data/tabula-muris/04_facs_processed_data/FACS/"

        data_file = f"Processed_{data_name.title()}.h5ad"
        data_path = os.path.join(data_dir, data_file)

        X = sc.read_h5ad(data_path)
        metadata = X.obs.copy()

        X = X.obsm['X_pca']

    elif data_name.lower() == "ATAC":

        if data_dir is None:
            data_dir = "./data/10kPBMC_scATAC/02_processed_data/"

        data_file = f"atac_pbmc_10k_nextgem_preprocessed_data.h5ad"
        data_path = os.path.join(data_dir, data_file)

        X = sc.read_h5ad(data_path)
        metadata = X.obs.copy()

        X = X.obsm['lsi']

    if load_metadata:
        return X, metadata
    else:
        return X


###############################################################################
##  Functions for eCDFs
###############################################################################


def calculate_eCDF(data, extend=False):
    """Calculate the x- and y-coordinates of an empirical CDF curve.

    This function finds the unique values within a dataset, `data`, and
    calculates the likelihood that a random data point within the set is
    less than or equal to each of those unique values.  The `extend` option
    creates extra values outside the range of `data` corresponding to
    P(X <= x) = 0 and 1, which are useful for plotting eCDFs."""

    ## Get the unique values in `data` and their counts (the histogram).
    counts = Counter(data.ravel())
    ## Sort the unique values
    vals = np.sort(list(counts.keys()))
    ## Calculate the cumulative number of counts, then divide by the total.
    CDF = np.cumsum([counts[val] for val in vals])
    CDF = CDF / CDF[-1]

    ## If `extend`, add points to `vals` and `CDF`
    if extend:
        data_range = vals[-1] - vals[0]
        vals = [vals[0] - (0.01 * data_range)] + list(vals)
        vals = np.asarray(vals + [vals[-1] + (0.01 * data_range)])
        CDF = np.asarray([0] + list(CDF) + [1])

    return vals, CDF


def get_QQ_vals(data1, data2):
    """Align 2 datasets' eCDFs for plotting on QQ-plot."""

    vals1, CDF1 = get_eCDF(data1, extend=True)
    vals2, CDF2 = get_eCDF(data2, extend=True)

    joint_vals = np.sort(np.unique(np.hstack((vals1, vals2))), axis=0)

    joint_CDF1 = np.zeros_like(joint_vals)
    joint_CDF2 = np.zeros_like(joint_vals)

    id1, id2 = 0, 0
    for ii, val in enumerate(joint_vals):

        joint_CDF1[ii] = CDF1[id1]
        if (val in vals1) and (id1 + 1 < len(vals1)):
            id1 += 1

        joint_CDF2[ii] = CDF2[id2]
        if (val in vals2) and (id2 + 1 < len(vals2)):
            id2 += 1

    return joint_vals, joint_CDF1, joint_CDF2


def interpolate(x_coords, y_coords, x_arr):
    """Interpolate between coordinates to find the y-values at `x_arr`."""

    x_out = np.zeros_like(x_arr).astype(float)
    for ii, x in enumerate(x_arr):
        if (x < x_coords[0]) or (x > x_coords[-1]):
            err_str = f"Value {x} is outside interpolation regime!"
            raise ValueError(err_str)

        if np.any(np.isclose(x, x_coords)):
            idx = np.where(np.isclose(x, x_coords))[0]
            if len(idx) > 1:
                idx = idx[0]
            x_out[ii] = (y_coords[idx]).squeeze()
            continue

        grid_id = np.searchsorted(x_coords, x, side='right')
        x_diff = (x_coords[grid_id] - x_coords[grid_id - 1])
        frac = (x - x_coords[grid_id - 1])
        slope = (y_coords[grid_id] - y_coords[grid_id - 1]) / x_diff
        x_out[ii] = (y_coords[grid_id - 1] + slope * frac).squeeze()

    return np.asarray(x_out, dtype=float).squeeze()
