# Geometric-Topic-Modeling

This is a Python 2 implementation of Geometric Dirichlet Means algorithm for topic inference (M. Yurochkin, X. Nguyen NIPS 2016) and Conic Scan Coverage algorithms for nonparametric topic modeling (M. Yurochkin, A. Guha, X. Nguyen to appear in NIPS 2017). Code written by Mikhail Yurochkin.

## Overview

This is a simple demonstration of GDM, CSC and Gibbs sampler (from lda package) on simulated data. More extensive guide is in preparation.

all_func.py Implements data simulation according to LDA model, GDM algorithm and projection estimate of topic proportions $\theta$

geom_tm.py Implements CSC algorithm for sparse document-term matrix and wraps it as scikit-learn class

tester_CSC.py contains a simulated example

Implementation is designed to be used in the interactive mode (e.g. Python IDE like Spyder).

## Usage guide for GDM algorithm

```
gdm(wdfn, K, ncores=-1)
```

wdfn: $M \times V$ matrix of normalized document-term counts

K: number of topics to fit

ncores: CPUs to use for k-means

Returns: topic estimates

## Usage guide for CSC algorithm

```
geom_tm(delta=0.4, prop_discard=0.5, prop_n=0.01, verbose=False)
```

Parameters:

delta: cosine cone radius $\omega$

prop_discard: quantile to compute $\mathcal{R}$

prop_n: proportion of data to be used as outlier threshold $\lambda$

verbose: if True, plots as in Figure 2 will be printed


Methods:
```
fit_a(data, cent)
```

data: sparse $M \times V$ matrix of normalized document-term counts

cent: data mean $\hat C_p$ 

Returns:
a_betas_: topic estimates from Algorithm 2 without spherical k-means step
K_: estimated number of topics

```
fit_sph(data, cent, init=None, it=10)
```

data: sparse $M \times V$ matrix of normalized document-term counts

cent: data mean $\hat C_p$

init, it: if None and fit_a was run, will complete Algorithm 2 with \emph{it} spherical k-means iterations

Returns:
sph_betas_: updated topics
sph_clust_: cluster assignments

```
fit_all(data, cent, it=5)
```

Full run of Algorithm 2 with \emph{it} spherical k-means post processing iterations
