# EMBEDR v1.0

Author: Eric Johnson
Date Created: July 1, 2021
Email: eric.johnson643@gmail.com

## Overview

In this package, I want to create a minimal working version of the EMBEDR
algorithm.  It will only support a few dimensionality reduction algorithms
and will be designed to simply get me the figures I need for revisions as soon
as possible.

To do this, I will need:
- kNN algorithms.  I don't need everything available to the user at the top.
- Fixed Entropy affinity calculator
- unwrapped t-SNE and UMAP implementations.
- EES calculators

The workflow will be:
- Input data and perplexities / k_NNs.
    - Initialize PWD matrix if possible.  If not, we'll figure something out.
        - Ideas: downsample (landmark samples??) and extrapolate affinity
    - Calculate P
        - Maybe don't store, but keep precisions and recalculate when needed
- Generate null data
    - Calculate P_null
- Embed data / null
    - Calculate EES
- Calculate p-Values

Storage:
- Data
- One PWD for data and each null
- Precisions for affinities for each data/null at each perp/k_NN
- Embeddings for each data and null at each perp/k_NN
- EES for each data and null at each perp/k_NN
- p-Values for each data/null at each perp/k_NN

## To-Do

MAKE SURE THIS CAN BE INSTALLED ON QUEST!  So far so good...