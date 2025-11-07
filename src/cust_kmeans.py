## Part of this code is from the implementation of https://gist.github.com/denis-bz ##

from __future__ import division
import random
import numpy as np
from scipy.spatial.distance import cdist
from scipy.sparse import issparse


def kmeans(X, centres, delta=.001, maxiter=10, metric="euclidean", p=2, verbose=1):
    """ centres, Xtocentre, distances = kmeans( X, initial centres ... )
    in:
        X N x dim  may be sparse
        centres k x dim: initial centres, e.g. random.sample( X, k )
        delta: relative error, iterate until the average distance to centres
            is within delta of the previous average distance
        maxiter
        metric: any of the 20-odd in scipy.spatial.distance
            "chebyshev" = max, "cityblock" = L1, "minkowski" with p=
            or a function( Xvec, centrevec ), e.g. Lqmetric below
        p: for minkowski metric -- local mod cdist for 0 < p < 1 too
        verbose: 0 silent, 2 prints running distances
    out:
        centres, k x dim
        Xtocentre: each X -> its nearest centre, ints N -> k
        distances, N
    see also: kmeanssample below, class Kmeans below.
    """
    if not issparse(X):
        X = np.asanyarray(X)  # ?
    centres = centres.todense() if issparse(centres) \
        else centres.copy()
    N, dim = X.shape
    k, cdim = centres.shape
    if dim != cdim:
        raise ValueError("kmeans: X %s and centres %s must have the same number of columns" % (
            X.shape, centres.shape))
    # if verbose:
    #    print(X.shape, centres.shape, delta, maxiter, metric)

    allx = np.arange(N)
    prevdist = 0
    for jiter in range(1, maxiter + 1):
        D = cdist_sparse(X, centres, metric=metric)  # , p=p )  # |X| x |centres|
        xtoc = D.argmin(axis=1)  # X -> nearest centre
        distances = D[allx, xtoc]
        avdist = distances.mean()  # median ?
        # if verbose >= 2:
        #    print("kmeans: av |X - nearest centre| = %.4g" % avdist)
        if (1 - delta) * prevdist <= avdist <= prevdist \
                or jiter == maxiter:
            break
        prevdist = avdist
        for jc in range(k):  # (1 pass in C)
            c = np.where(xtoc == jc)[0]
            if len(c) > 0:
                centres[jc] = X[c].mean(axis=0)
    # if verbose:
    #    print("kmeans: %d iterations  cluster sizes:" % jiter, np.bincount(xtoc))
    if verbose >= 2:
        r50 = np.zeros(k)
        r90 = np.zeros(k)
        for j in range(k):
            dist = distances[xtoc == j]
            if len(dist) > 0:
                r50[j], r90[j] = np.percentile(dist, (50, 90))
        # print("kmeans: cluster 50 % radius", r50.astype(int))
        # print("kmeans: cluster 90 % radius", r90.astype(int))
        # scale L1 / dim, L2 / sqrt(dim) ?
    # Compute inertia (sum of squared distances to the closest center)
    inertia = distances.sum()

    return centres, xtoc, distances, inertia
    # return centres, xtoc, distances


def kmeans_with_n_init(X, nclusters, n_init=10, delta=0.001, maxiter=10, metric="cosine", verbose=1):
    """ Run k-means n_init times and return the best clustering result
    in:
        X: data points, N x dim
        nclusters: number of clusters
        n_init: number of times to run k-means with different initial centroids
        delta: stopping criterion for convergence
        maxiter: maximum iterations for k-means
        metric: distance metric to use (default is 'cosine')
    out:
        best_centres: the best centres found in all initializations
        best_xtocentre: data points assigned to nearest centers
        best_distances: distances of each point to its assigned center
        best_inertia: the best inertia (sum of squared distances)
    """
    best_inertia = float('inf')  # Initialize to a very large value
    best_centres = None
    best_xtocentre = None
    best_distances = None

    for i in range(n_init):
        # Initialize centroids randomly
        initial_centres = randomsample(X, nclusters)

        # Run kmeans
        centres, xtocentre, distances, inertia = kmeans(
            X,
            initial_centres,
            delta=delta,
            maxiter=maxiter,
            metric=metric,
            verbose=verbose
        )

        # if verbose:
        #    print(f"Run {i+1}/{n_init}, Inertia: {inertia}")

        # Keep track of the best result
        if inertia < best_inertia:
            best_inertia = inertia
            best_centres = centres
            best_xtocentre = xtocentre
            best_distances = distances

    # if verbose:
    #    print(f"Best inertia after {n_init} runs: {best_inertia}")

    return best_centres, best_xtocentre, best_distances, best_inertia


def kmeanssample(X, k, nsample=0, **kwargs):
    """ 2-pass kmeans, fast for large N:
        1) kmeans a random sample of nsample ~ sqrt(N) from X
        2) full kmeans, starting from those centres
    """
    # merge w kmeans ? mttiw
    # v large N: sample N^1/2, N^1/2 of that
    # seed like sklearn ?
    N, dim = X.shape
    if nsample == 0:
        nsample = max(2 * np.sqrt(N), 10 * k)
    Xsample = randomsample(X, int(nsample))
    pass1centres = randomsample(X, int(k))
    samplecentres = kmeans(Xsample, pass1centres, **kwargs)[0]
    return kmeans(X, samplecentres, **kwargs)


def cdist_sparse(X, Y, **kwargs):
    """ -> |X| x |Y| cdist array, any cdist metric
        X or Y may be sparse -- best csr
    """
    # todense row at a time, v slow if both v sparse
    sxy = 2 * issparse(X) + issparse(Y)
    if sxy == 0:
        return cdist(X, Y, **kwargs)
    d = np.empty((X.shape[0], Y.shape[0]), np.float64)
    if sxy == 2:
        for j, x in enumerate(X):
            d[j] = cdist(x.todense(), Y, **kwargs)[0]
    elif sxy == 1:
        for k, y in enumerate(Y):
            d[:, k] = cdist(X, y.todense(), **kwargs)[0]
    else:
        for j, x in enumerate(X):
            for k, y in enumerate(Y):
                d[j, k] = cdist(x.todense(), y.todense(), **kwargs)[0]
    return d


def randomsample(X, n):
    """ random.sample of the rows of X
        X may be sparse -- best csr
    """
    sampleix = random.sample(range(X.shape[0]), int(n))
    return X[sampleix]


def nearestcentres(X, centres, metric="euclidean", p=2):
    """ each X -> nearest centre, any metric
            euclidean2 (~ withinss) is more sensitive to outliers,
            cityblock (manhattan, L1) less sensitive
    """
    D = cdist(X, centres, metric=metric)  # , p=p )  # |X| x |centres|
    return D.argmin(axis=1)


def Lqmetric(x, y=None, q=.5):
    # yes a metric, may increase weight of near matches; see ...
    return (np.abs(x - y) ** q).mean() if y is not None \
        else (np.abs(x) ** q).mean()


def kmeans_with_n_init_withDebugging(X, nclusters, n_init=10, delta=0.001, maxiter=10, metric="cosine", verbose=1):
    """ Run k-means n_init times and return the best clustering result
    in:
        X: data points, N x dim
        nclusters: number of clusters
        n_init: number of times to run k-means with different initial centroids
        delta: stopping criterion for convergence
        maxiter: maximum iterations for k-means
        metric: distance metric to use (default is 'cosine')
    out:
        best_centres: the best centres found in all initializations
        best_xtocentre: data points assigned to nearest centers
        best_distances: distances of each point to its assigned center
        best_inertia: the best inertia (sum of squared distances)
    """
    best_inertia = float('inf')  # Initialize to a very large value
    best_centres = None
    best_xtocentre = None
    best_distances = None

    for i in range(n_init):
        # Initialize centroids randomly
        initial_centres = randomsample(X, nclusters)

        # Identify zero rows (rows where all values are zero)
        zero_rows = (X == 0).all(axis=1)
        zero_rows_indices = np.where(zero_rows)[0]

        # Assign zero rows to cluster 0
        xtocenter = np.zeros(X.shape[0], dtype=int)  # Initialize xtocenter with zeros
        xtocenter[zero_rows_indices] = 0  # Assign zero rows to cluster 0

        # Remove zero rows from X for normal k-means clustering
        X_nonzero = X[~zero_rows]  # Remove zero rows

        # Run k-means only on the non-zero rows
        if X_nonzero.shape[0] > 0:
            centres, xtoc, distances, inertia = kmeans(
                X_nonzero,
                initial_centres,
                delta=delta,
                maxiter=maxiter,
                metric=metric,
                verbose=verbose
            )

            # Assign the cluster labels of the non-zero rows back to xtocenter array
            xtocenter[~zero_rows] = xtoc
        else:
            # If there are no non-zero rows, we skip k-means and keep all rows in cluster 0
            xtocenter = np.zeros(X.shape[0], dtype=int)

        # Keep track of the best result
        if inertia < best_inertia:
            best_inertia = inertia
            best_centres = centres
            best_xtocentre = xtocenter
            best_distances = distances

        if verbose:
            print(f"Run {i+1}/{n_init}, Inertia: {best_inertia}")

    return best_centres, best_xtocentre, best_distances, best_inertia


def kmeans_corr(X, n_clusters, n_inits, max_iter, tol, random_state=0):
    """
    K-means with correlation distance (1 - Pearson r), single-function, vectorized.
    Returns labels of shape (n_samples,).
    ! Caution ! it modifies X in place to save memory space. Pass a copy to discard later when the labels are obtained.
    """

    n_samples, n_dims = X.shape

    # Row-wise mean-center and L2-normalize (so correlation distance = 1 - dot)
    X -= X.mean(axis=1, keepdims=True)
    nrms = np.linalg.norm(X, axis=1, ord=2, keepdims=True)
    # handle zero-variance rows
    np.maximum(nrms, 1e-12, out=nrms)
    X /= nrms

    best_labels = None
    best_inertia = np.inf
    rng = np.random.default_rng(random_state)

    # workspace
    sim = np.empty((n_samples, n_clusters), dtype=np.float64)
    labels = np.empty(n_samples, dtype=np.int32)

    for _ in range(int(n_inits)):
        # init centers by sampling distinct points
        idx = rng.choice(n_samples, size=n_clusters, replace=False)
        C = X[idx].copy() # shape (n_clusters, n_dims)

        prev_labels = None
        for iter_id in range(int(max_iter)):
            # assign (maximize similarity == minimize 1 - similarity)
            sim[:] = X @ C.T
            np.argmax(sim, axis=1, out=labels)

            # check convergence by label stability
            if prev_labels is not None:
                changed = np.count_nonzero(labels != prev_labels)
                if changed / n_samples <= tol:
                    break
            prev_labels = labels.copy()

            # update centers: mean then renormalize; handle empty clusters
            for k in range(n_clusters):
                mask = (labels == k)
                if not np.any(mask):
                    # re-seed empty cluster with a random point
                    j = rng.integers(0, n_samples)
                    C[k] = X[j]
                else:
                    C[k] = X[mask].mean(axis=0)
                # renormalize (row is already zero-mean because mean of zero-mean rows)
                nk = np.linalg.norm(C[k])
                if nk < 1e-12:
                    # fallback if degenerate
                    j = rng.integers(0, n_samples)
                    C[k] = X[j]
                else:
                    C[k] /= nk

        # compute inertia (sum correlation distance)
        # distance = 1 - similarity of assigned center
        sim[:] = X @ C.T
        inertia = np.sum(1.0 - sim[np.arange(n_samples), labels])

        if inertia < best_inertia:
            best_inertia = inertia
            best_labels = labels.copy()

    return best_labels