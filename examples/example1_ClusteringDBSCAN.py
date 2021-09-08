"""
###############################################################################
    Example 1: Clustering an Embedding with DBSCAN
###############################################################################

    Author: Eric Johnson
    Date Created: Wednesday, September 8, 2021
    Email: ericjohnson1.2015@u.northwestern.edu

###############################################################################

    Once an embedding has been generated, it can be interested to see how
    clusters show up in the new representation of the data.  In particular, if
    we have applied sample-wise optimal hyperparameters, we may be able to
    apply such unsupervised clustering techniques with more confidence!

###############################################################################
"""

from EMBEDR import EMBEDR
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.cluster import DBSCAN
from sklearn.metrics import pairwise_distances as pwd

if __name__ == "__main__":

    ## Set matplotlib.rc parameters
    plt.close('all')
    plt.rcParams['svg.fonttype'] = 'none'
    sns.set(color_codes=True)
    sns.set_style('whitegrid')
    matplotlib.rc("font", size=10)
    matplotlib.rc("xtick", labelsize=10)
    matplotlib.rc("ytick", labelsize=10)
    matplotlib.rc("axes", labelsize=12)
    matplotlib.rc("axes", titlesize=12)
    matplotlib.rc("legend", fontsize=10)
    matplotlib.rc("figure", titlesize=12)

    X = np.loadtxt("./data/mnist2500_X.txt").astype(float)
    labels = np.loadtxt("./data/mnist2500_labels.txt").astype(int)

    n_samples, n_features = X.shape

    embedr_obj = EMBEDR(perplexity=100, random_state=1, verbose=1,
                        project_name='Exmp1_DBSCAN_Cluster')

    ## Generate a t-SNE embedding and EMBEDR p-Values.
    embedr_obj.fit(X)

    ## Set the DBSCAN parameters based on the data.  If these don't work, 
    PWD_Y = np.triu(pwd(embedr_obj.data_Y[0], metric='euclidean'), k=1)
    n_pwd = n_samples * (n_samples - 1) / 2
    eps = np.percentile(PWD_Y[PWD_Y.nonzero()], 1.5)

    db_min_samp = int(n_samples / 100)
    # db_min_samp = 100

    print(f"DBSCAN parameters set at eps={eps:.2f} and minimum {db_min_samp}"
          f" samples per anchor point.")

    ## Cluster the resulting embedding
    dbObj = DBSCAN(eps=eps, min_samples=db_min_samp)
    dbObj.fit(embedr_obj.data_Y[0])

    db_labels = dbObj.labels_
    colors = [(0,0,0) if dbl == -1 else sns.color_palette()[dbl]
              for dbl in db_labels]
    unique_labels = np.sort(np.unique(db_labels))
    n_db_labels = len(unique_labels) - 1

    print(f"After clustering, {n_db_labels} clusters were detected!")

    ## Generate a figure
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    handle = ax.scatter(*embedr_obj.data_Y[0].T, s=2, c=colors)

    for lbl in unique_labels:
        if lbl == -1:
            _ = ax.scatter([], [], s=2, color='k', label='No Cluster')
        else:
            _ = ax.scatter([], [], s=2, color=sns.color_palette()[lbl],
                           label=f'Cluster {lbl}')

    _ = ax.legend()

    fig.tight_layout()

    plt.show()