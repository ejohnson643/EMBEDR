from embedr import EMBEDR
import matplotlib.pyplot as plt
import numpy as np
from openTSNE.affinity import PerplexityBasedNN
import utility as utl

if __name__ == "__main__":

    X = np.loadtxt("./Data/mnist2500_X.txt")

    old_perp = 30
    new_perp = 100

    n_neighbors = 40

    n_jobs = -1
    seed = 1
    verbose = 5

    n_data_embed = 1
    n_null_embed = 2

    fig, [ax1, ax2] = plt.subplots(1, 2, figsize=(12, 5))

    ## Initialize and fit the data like normal
    UMAP_embed = EMBEDR(perplexity=old_perp,
                        dimred_params={'n_neighbors': n_neighbors},
                        # cache_results=False,  ## Turn off file caching.
                        dimred_alg="UMAP",
                        n_jobs=n_jobs,
                        random_state=seed,
                        verbose=verbose,
                        n_data_embed=n_data_embed,
                        n_null_embed=n_null_embed,
                        project_name='changing_perplexity_test')
    UMAP_embed.fit(X)

    ## Let's see the results!
    UMAP_embed.plot(ax=ax1, show_cbar=False)

    ## Calculate a new affinity matrix at the new perplexity
    new_aff_mat = PerplexityBasedNN(X,
                                    perplexity=new_perp,
                                    n_jobs=n_jobs,
                                    random_state=seed,
                                    verbose=verbose)

    ## Calculate null affinity matrices at the new perplexity
    new_null_mat = {}
    for nNo in range(n_null_embed):

        null_X = utl.generate_nulls(X, seed=seed + nNo).squeeze()
        nP = PerplexityBasedNN(null_X,
                               perplexity=new_perp,
                               n_jobs=n_jobs,
                               random_state=seed,
                               verbose=verbose)

        new_null_mat[nNo] = nP

    ## Reset the affinity matrices in the method
    UMAP_embed._affmat = new_aff_mat
    UMAP_embed._null_affmat = new_null_mat

    ## Recalculate the p-Values and quality scores.
    UMAP_embed.do_cache = False  ## Need to turn off file caching to force the
                                 ## method to recalculate.
    UMAP_embed._calc_EES()

    ## Let's see the results!
    UMAP_embed.plot(ax=ax2)

    ax1.set_title(f"Affinity Perplexity = {old_perp}")
    ax2.set_title(f"Affinity Perplexity = {new_perp}")
    ax1.set_xticklabels([])
    ax1.set_yticklabels([])
    ax2.set_xticklabels([])
    ax2.set_yticklabels([])

    fig.tight_layout()

    plt.show()
