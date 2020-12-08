"""
Functions used to run the Markov chain simulation.
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st
import scipy as sp
import copy
from time import time

from helpers import *
from DatasetGenerator import *


def Delta_computation(x, y, k, lambda_, data):
    """
    Compute the difference -(f(y)-f(x)), for x and y neighbors that differentiate only for the city k.

    To do this efficiently, we first handle the cases where at least one of the distances is null.
    That is if one of the vector contains 0 or 1 city.
    We then compute the maximum distance for the vector with the most cities. If the city k doesn't
    intervene in the maximum distance, then both x and y have the same maximum distance and so their
    difference is null. Otherwise, we need to compute the maximum distance of the other vector to
    compute the difference.

    Parameters
    ----------
    x: ndarray of shape (n,)
    y: ndarray of shape (n,), the proposed move
    k: int, the index of the digit different in x and y
    lambda_: float, the fixed value of the deployment cost
    data: Dataset

    Returns
    -------
    Delta: minus the difference between f(y) and f(x)
    """
    n, coords, pop, dists = data.N, data.x, data.v, data.d
    num_ones_y = np.count_nonzero(y)
    num_ones_x = np.count_nonzero(x)
    S_x = vect_to_S(x)
    S_y = vect_to_S(y)
    assert num_ones_y - num_ones_x == 1 or num_ones_x - num_ones_y == 1, 'x and y should be neighbors'
    assert x[k] != y[k], 'x and y should differ at index k'

    #Compute first part of the delta
    Delta = pop[k] * (x[k] - y[k])

    # both vectors contain either 0 or 1 city, so both distances are null
    if num_ones_x == 0 or num_ones_y == 0:
        return Delta

    # one of the vector has only one city, so its distance is null : the other vector's distance has to be computed
    if num_ones_x == 1 or num_ones_y == 1:
        y_dist_null = 1 if num_ones_y == 1 else -1
        max_dist = max_distance(x, dists) if y_dist_null == 1 else max_distance(y, dists)
        Delta = Delta - (1/4) * lambda_ * n * np.pi * y_dist_null * max_dist**2
        return Delta

    # compute the maximum distance only for the vector with the most cities, i.e. where city k is added
    biggest, smallest, x_is_biggest = (x, y, 1) if num_ones_x > num_ones_y else (y, x, -1)
    max_dist_biggest, city_maxs = max_distance(biggest, dists, max_cities=True)

    # adding k changes the maximum distance, so max_distance of x is different of max_distance of y
    if all(k in cities for cities in city_maxs):
        max_dist_smallest = max_distance(smallest, dists)
        Delta = Delta - (1/4) * lambda_ * n * np.pi * x_is_biggest * ((max_dist_biggest)**2 - (max_dist_smallest)**2)
        return Delta

    # adding k doesn't change the maximum distance, so both max distance of x and y are equal
    return Delta

def forward(beta, x, lambda_, data):
    """
    Apply one step of the Metropolis_Hastings algorithm, that is
    choose one step in the base chain, compute the associates acceptance
    probability and decide whether to accept the move in the new chain.

    Parameters
    ----------
    beta: float, the distribution parameter
    x: ndarray of shape (n,)
    lambda_: float, the fixed value of the deployment cost
    data: Dataset

    Returns
    -------
    y: one of the neighbours of x if the move is accepted, x otherwise
    """

    # propose a move on the Symmetric Random Walk on hypercube of dimension n: switch one digit at random
    k = np.random.randint(0, data.N)
    y = x.copy()
    y[k] = (y[k] + 1)%2

    # compute Delta
    Delta = Delta_computation(x, y, k, lambda_, data)

    # compute acceptance probability
    a_xy = 1 if Delta <= 0 else np.exp(- beta * Delta)

    # accept the move to y with probability a_xy and reject it with probability 1 - a_xy
    move = np.random.choice([True, False], p=[a_xy, 1 - a_xy])

    return (y, Delta) if move else (x, 0)

def metropolis_hastings(beta, x, n_iter, best_x, lambda_, data, plot=False):
    """
    Apply the Metropolis_Hastings algorithm for n_iter steps, starting at state x.
    If given, enters the size of the selected cities in the plot ax_size
    and the objective function evaluation in the plot ax_obj.

    Parameters
    ----------
    beta: float, the distribution parameter
    x: ndarray of shape (n,)
    n_iter: int, the number of iterations to run the Markov chain
    best_x: ndarray of shape (n,), the best state visited so far
    lambda_: float, the fixed value of the deployment cost
    data: Dataset
    axs: tuple of size 2, the subplot axis in which to plot the evolution of the objectif function
        and the number of cities selected at each step ; if None, no plot is created

    Returns
    -------
    x: the final state of the Markov chain
    best_x: the best state visited so far
    """
    N = list(range(n_iter + 1))
    # if axs:
    #     axs[0].set_title("Evolution of the approximate maximum when beta = {:.3f}".format(beta))
    if plot:
        nb_cities = [np.count_nonzero(x)]
        objs = [f(vect_to_S(x), lambda_, data)]

    max_f_visited = f(vect_to_S(best_x), lambda_, data)
    f_x = f(vect_to_S(x), lambda_, data)

    # run the Markov chain for n_iter steps
    for _ in range(n_iter):
        x, Delta = forward(beta, x, lambda_, data)

        # memorize the best visited state x
        f_x = f_x-Delta
        if f_x > max_f_visited:
            max_f_visited = f_x
            best_x = x

        if plot:
            nb_cities.append(np.count_nonzero(x))
            objs.append(f(vect_to_S(x), lambda_, data))

    if plot:
        return x, best_x, nb_cities, objs
    # if axs:
    #     ax, ax2 = axs
    #     ax.plot(N, objs, 'o', color='blue')
    #     ax2.plot(N, nb_cities, 'o', color='red')

    #     ax.set_xlabel("Iteration")
    #     ax.set_ylabel("Objective function", color='blue')
    #     ax2.set_ylabel("Number of cities", color='red')

    return x, best_x, None, None


def simulated_annealing(starting_state, betas, n_iter, lambda_, data, verbose=False, plot=False):
    """
    Runs the Metropolis-Hastings algorithm for each beta in the list betas.

    Parameters
    ----------
    starting_state: ndarray of shape (n,), the state to start the Markov chain
    betas: list of increasing beta (floats)
    n_iter: int, number of iteration for each beta (temperature)
    lambda_: float, the fixed value of the deployment cost
    data: Dataset
    verbose: boolean, whether to print the running time of each Metropolis-Hastings algorithm
    plot: boolean, whether to plot the evolution of the the objectif function
        and the number of cities selected for each beta

    Returns
    -------
    S_approx: the approximation of the optimizing set.
    """
    x = starting_state
    best_x = x

    if plot:
        nb_cities, objs = [], []

    # run Metropolis-Hastings algorithm for each beta
    for k, beta in enumerate(betas):

        start = time.time()
        x, best_x, nb_cities_beta, objs_beta = metropolis_hastings(beta, x, n_iter, best_x, lambda_, data, plot)
        end = time.time()

        if plot:
            nb_cities += nb_cities_beta
            objs += objs_beta

        if verbose:
            print("[step {}/{}] Time spent on beta = {:.3f} : {:.3f} sec"
                  .format(k + 1, len(betas), beta, end - start))

    # keep only the best state
    if f(vect_to_S(best_x), lambda_, data) > f(vect_to_S(x), lambda_, data):
        x = best_x

    S_approx = vect_to_S(x)

    if plot:
        fig, axs = plt.subplots(2, figsize=(20, 10), constrained_layout=True)
        # fig.suptitle('Evolution of the state visited by the Markov chain (for lambda={})'.format(lambda_), fontsize=20)
        total_steps = list(range(len(objs)))
        betas_changes = [i * n_iter for i in range(len(betas) + 1)]

        axs[0].plot(total_steps, objs, 'o', color='blue')
        for b in betas_changes:
            axs[0].axvline(b, color='blue')
        axs[0].set_xlabel("Iterations", fontsize=25)
        axs[0].set_ylabel("Objective function", fontsize=25)
        axs[0].tick_params(labelsize=17)

        axs[1].plot(total_steps, nb_cities, 'o', color='red')
        for b in betas_changes:
            axs[1].axvline(b, color='red')
        axs[1].set_xlabel("Iterations", fontsize=25)
        axs[1].set_ylabel("Number of cities", fontsize=25)
        axs[1].tick_params(labelsize=17)

        data_name = "G1" if isinstance(data, G1) else "G2"
        figtitle = 'plots\global_evol_{}_{}.png'.format(lambda_, data_name)
        fig.savefig(figtitle)

    return S_approx
