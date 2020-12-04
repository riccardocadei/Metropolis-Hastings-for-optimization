"""
Some helpful functions to run markov chain simulations.
"""
import scipy.stats as st
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import copy
import time


def vect_to_S(x):
    """
    Compute the subset S of cities corresponding to the vector encoding

    Parameters
    ----------
    x: ndarray of shape (n,)

    Returns
    -------
    S: ndarray, the subset of corresponding cities
    """

    return np.nonzero(x)[0]

def f(S, params):
    """
    Compute the objective function (that we want to maximize).

    Parameters
    ----------
    S: subset of P({0,..., n-1}) as an array of shape (k,) with k the size of S
    params: tuple (lambda_, n, coords, pop), where
                lambda_ is the fixed value of the deployment cost,
                n is the fixed number of cities,
                coords is the (n,2) array containing the coordinates of each cities
                pop is the (n,) array containing the (normalized) population of each cities

    Returns
    -------
    f: The evaluated objective function.
    """
    lambda_, n, coords, pop = params

    # consider only coordinates of cities in S
    coords_S = coords[S]
    pop_S = pop[S]
    max_dist_S = np.max(sp.spatial.distance.cdist(coords_S, coords_S, 'euclidean')) if len(coords_S) > 1 else 0
    f = np.sum(pop_S) - (1/4) * lambda_ * n * np.pi * (max_dist_S)**2

    return f

def max_distance(x, coords, max_cities=False):
    """
    Compute the pairwise distance between cities represented by x
    and return the maximum distance among all of them.

    Parameters
    ----------
    x: ndarray of shape (n,)
    coords: (n, 2) array containing the coordinates of each cities
    max_cities: boolean, whether to return the city couples realising the argmax

    Return
    ------
    max_dist: float, the maximum distance among all pairwise distances between cities represented by x
    city_maxs: list of tuples (returned if max_cities == True), the couple of cities that intervene in the maximum distance
    """

    S_x = vect_to_S(x)
    if len(S_x) <= 1:
        return 0

    coords_S = coords[S_x]
    dists = sp.spatial.distance.cdist(coords_S, coords_S, 'euclidean')
    max_dist = np.max(dists)

    if not max_cities:
        return max_dist

    ind_maxs = np.argwhere(dists == max_dist) # indices in dist matrice of the max distance
    city_maxs = np.zeros(ind_maxs.shape[0], dtype=tuple) # change the indices to the city number
    for n, (i, j) in enumerate(ind_maxs):
        city_maxs[n] = (S_x[i], S_x[j])

    return max_dist, city_maxs
