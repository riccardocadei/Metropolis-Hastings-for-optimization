"""
Some helpful functions to run markov chain simulations.
"""
import scipy.stats as st
import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt
import copy
import time
from DatasetGenerator import *


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

def f(S, lambda_, data):
    """
    Compute the objective function (that we want to maximize).

    Parameters
    ----------
    S: subset of P({0,..., n-1}) as an array of shape (k,) with k the size of S
    lambda_: float, the fixed value of the deployment cost
    data: Dataset

    Returns
    -------
    f: The evaluated objective function.
    """
    n, coords, pop, dists = data.N, data.x, data.v, data.d

    # consider only coordinates of cities in S
    pop_S = pop[S]
    max_dist_S = np.max(dists[S,:][:,S]) if len(S) > 1 else 0
    f = np.sum(pop_S) - (1/4) * lambda_ * n * np.pi * (max_dist_S)**2

    return f

def max_distance(x, dists, max_cities=False):
    """
    Compute the pairwise distance between cities represented by x
    and return the maximum distance among all of them.

    Parameters
    ----------
    x: ndarray of shape (n,)
    dists: ndarray, the matrix distance between all cities in the working dataset
    max_cities: boolean, whether to return the city couples realising the argmax

    Return
    ------
    max_dist: float, the maximum distance among all pairwise distances between cities represented by x
    city_maxs: list of tuples (returned if max_cities == True), the couple of cities that intervene in the maximum distance
    """

    S_x = vect_to_S(x)
    max_dist = np.max(dists[S_x,:][:,S_x]) if len(S_x) > 1 else 0

    if not max_cities:
        return max_dist

    # also return the couple of cities realising the maximum distance
    ind_maxs = np.argwhere(dists[S_x,:][:,S_x] == max_dist if len(S_x) > 1 else 0)  # indices in dist matrice of the max distance
    city_maxs = np.zeros(ind_maxs.shape[0], dtype=tuple)                            # change the indices to the city number
    for n, (i, j) in enumerate(ind_maxs):
        city_maxs[n] = (S_x[i], S_x[j])

    return max_dist, city_maxs


### Function for the competition:

def preprocessing_data(csv_file):
    """
    Function used to load the data from the csv file to panda
    """

    #Importing the csv file as Dataframe
    df = pd.read_csv(csv_file, index_col="city id")

    #Computing starting state
    starting_state = np.zeros(len(df))
    starting_state[df['normalized population'] == df['normalized population'].max()] = 1

    #Converting the df to a Dataset object
    data = Dataset_competition(N=len(df))
    data.x = df[['position x', 'position y']].to_numpy()
    data.v = df['normalized population'].to_numpy()
    data.d = sp.spatial.distance.cdist(data.x, data.x, 'euclidean')

    return starting_state, data

def submission(csv_file, Sapprox, save_file_name):

    #Importing the csv file as Dataframe
    df_submission = pd.read_csv(csv_file, usecols = ["city id"])

    df_submission['1/0 variable'] = 0
    df_submission['1/0 variable'][Sapprox] = 1
    
    df_submission.to_csv(save_file_name, index=False)
