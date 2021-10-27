"""
###############################################################################
    Uniform Manifold Approximation and Projection (UMAP) Implementation
###############################################################################

    Author: Eric Johnson
    Date Created: Thursday, October 21, 2021
    Email: ericjohnson1.2015@u.northwestern.edu

###############################################################################

    In this file, I want to define a class to wrap around the UMAP class in
    umap-learn so as to work best with the EMBEDR workflow.  In the future,
    I would like to restructure the algorithm so that different numbers of
    nearest neighbors can be supplied for each sample in the dataset, but this
    seems very complicated at the moment.

    ALSO: They literally just rolled out a version that allows for
    precomputed_kNN!  However, this isn't part of an official release yet, so
    I won't rely on it staying there.  Too bad.

###############################################################################
"""

from umap import UMAP

def _initialize_UMAP_embed(n_neighbors=15,
                           n_components=2,
                           metric='euclidean',
                           metric_kwds=None,
                           output_metric='euclidean',
                           output_metric_kwds=None,
                           n_epochs=None,
                           learning_rate=1.0,
                           init='random',
                           min_dist=0.1,
                           spread=1.0,
                           low_memory=True,
                           n_jobs=-1,
                           set_op_mix_ratio=1.0,
                           local_connectivity=1.0,
                           repulsion_strength=1.0,
                           negative_sample_rate=5,
                           transform_queue_size=4.0,
                           a=None, b=None,
                           random_state=None,
                           angular_rp_forest=False,
                           target_n_neighbors=-1,
                           target_metric='categorical',
                           target_metric_kwds=None,
                           target_weight=0.5,
                           transform_seed=42,
                           transform_mode='embedding',
                           force_approximation_algorithm=False,
                           verbose=False,
                           unique=False,
                           densmap=False,
                           dens_lambda=2.0,
                           dens_frac=0.3,
                           dens_var_shift=0.1,
                           output_dens=False,
                           disconnection_distance=None):

    return UMAP(n_neighbors=n_neighbors,
                n_components=n_components,
                metric=metric,
                metric_kwds=metric_kwds,
                output_metric=output_metric,
                output_metric_kwds=output_metric_kwds,
                n_epochs=n_epochs,
                learning_rate=learning_rate,
                init=init,
                min_dist=min_dist,
                spread=spread,
                low_memory=low_memory,
                n_jobs=n_jobs,
                set_op_mix_ratio=set_op_mix_ratio,
                local_connectivity=local_connectivity,
                repulsion_strength=repulsion_strength,
                negative_sample_rate=negative_sample_rate,
                transform_queue_size=transform_queue_size,
                a=a, b=b,
                random_state=random_state,
                angular_rp_forest=angular_rp_forest,
                target_n_neighbors=target_n_neighbors,
                target_metric=target_metric,
                target_metric_kwds=target_metric_kwds,
                target_weight=target_weight,
                transform_seed=transform_seed,
                transform_mode=transform_mode,
                force_approximation_algorithm=force_approximation_algorithm,
                verbose=bool(verbose),
                unique=unique,
                densmap=densmap,
                dens_lambda=dens_lambda,
                dens_frac=dens_frac,
                dens_var_shift=dens_var_shift,
                output_dens=output_dens,
                disconnection_distance=disconnection_distance)
