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
    params: tuple (lambda_, n, coords, pop), where
                lambda_ is the fixed value of the deployment cost,
                n is the fixed number of cities,
                coords is the (n,2) array containing the coordinates of each cities
                pop is the (n,) array containing the (normalized) population of each cities

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
    params: tuple (lambda_, n, coords, pop), where
                lambda_ is the fixed value of the deployment cost,
                n is the fixed number of cities,
                coords is the (n,2) array containing the coordinates of each cities
                pop is the (n,) array containing the (normalized) population of each cities

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

def metropolis_hastings(beta, x, n_iter, best_x_visited, lambda_, data, ax_size=None, ax_obj=None):
    """
    Apply the Metropolis_Hastings algorithm for n_iter steps, starting at state x.
    If given, enters the size of the selected cities in the plot ax_size
    and the objective function evaluation in the plot ax_obj.

    Parameters
    ----------
    beta: float, the distribution parameter
    x: ndarray of shape (n,)
    n_iter: int, the number of iterations to run the Markov chain
    params: tuple (lambda_, n, coords, pop), where
                lambda_ is the fixed value of the deployment cost,
                n is the fixed number of cities,
                coords is the (n,2) array containing the coordinates of each cities
                pop is the (n,) array containing the (normalized) population of each cities
    ax_size: the subplot axis in which to plot the evolution of the number of cities selected at each step
             if None, no plot is created
    ax_obj: the subplot axis in which to plot the evolution of the objectif function at each step
             if None, no plot is created

    Returns
    -------
    x: the final state of the Markov chain
    """
    N = list(range(n_iter + 1))
    if ax_size:
        nb_cities = [np.count_nonzero(x)]
    if ax_obj:
        costs = [f(vect_to_S(x), lambda_, data)]

    max_f_visited = f(vect_to_S(best_x_visited), lambda_, data)
    f_x = f(vect_to_S(x), lambda_, data)

    # run the Markov chain for n_iter steps
    for _ in range(n_iter):
        x, Delta = forward(beta, x, lambda_, data)
        # Delta = f(x)-f(y)
        f_x = f_x-Delta
        if f_x > max_f_visited:
            max_f_visited = f_x
            best_x_visited = x

        if ax_size:
            nb_cities.append(np.count_nonzero(x))
        if ax_obj:
            costs.append(f(vect_to_S(x), lambda_, data))

    if ax_size:
        ax_size.plot(N, nb_cities, 'o')
        ax_size.set_title("Evolution of the number of cities throughout {} iterations when beta = {:.3f}".format(n_iter, beta))
        ax_size.set_xlabel("Iteration")
        ax_size.set_ylabel("Number of cities")
    if ax_obj:
        ax_obj.plot(N, costs, 'o')
        ax_obj.set_title("Evolution of the objective function throughout {} iterations when beta = {:.3f}".format(n_iter, beta))
        ax_obj.set_xlabel("Iteration")
        ax_obj.set_ylabel("Objective function f")

    return x, best_x_visited


def simulated_annealing(betas, n_iter, lambda_, data, verbose=False, plot_size=False, plot_obj=False):
    """
    Runs the Metropolis-Hastings algorithm for each beta in the list betas. For the first run, choose
    the starting state x at random, then start from the previous ending state.

    Parameters
    ----------
    betas: list of increasing beta (floats)
    n_iter: int, number of iteration for each beta (temperature)
    params: tuple (lambda_, n, coords, pop), where
                lambda_ is the fixed value of the deployment cost,
                n is the fixed number of cities,
                coords is the (n,2) array containing the coordinates of each cities
                pop is the (n,) array containing the (normalized) population of each cities
    verbose: boolean, whether to print the running time of each Metropolis-Hastings algorithm
    plot_size: boolean, whether to plot the evolution of the number of cities selected for each beta
    plot_obj: boolean, whether to plot the evolution of the objectif function for each beta

    Returns
    -------
    S_star_approx: the approximation of the optimizing set.
    """

    if plot_size:
        fig_size, axs_size = plt.subplots(len(betas), figsize=(10, 30))
        fig_size.suptitle('Evolution of the number of cities (lambda={})'.format(lambda_))
    if plot_obj:
        fig_obj, axs_obj = plt.subplots(len(betas), figsize=(10,30))
        fig_obj.suptitle('Evolution of objectif function (lambda={})'.format(lambda_))

    # start by picking a random state on the hypercube of dimension n
    x = np.random.randint(low=0, high=2, size=data.N)

    # Riccardo: fix the starting point
    #x = np.zeros(data.N)
    #x = np.ones(data.N)

    best_x_visited = x
    # run Metropolis-Hastings algorithm for each beta
    for k, beta in enumerate(betas):
        ax_size = axs_size[k] if plot_size else None
        ax_obj = axs_obj[k] if plot_obj else None

        start = time.time()
        x, best_x_visited = metropolis_hastings(beta, x, n_iter, best_x_visited, lambda_, data, ax_size, ax_obj)
        end = time.time()

        if verbose:
            print("[step {}/{}] Time spent on beta = {:.3f} : {:.3f} sec"
                  .format(k + 1, len(betas), beta, end - start))

    if f(vect_to_S(best_x_visited), lambda_, data) > f(vect_to_S(x), lambda_, data):
        x = best_x_visited

    S_star_approx = vect_to_S(x)

    if plot_size:
        fig_size.tight_layout()
        figtitle = 'size_{}.pdf'.format(lambda_)
        fig_size.savefig(figtitle)
    if plot_obj:
        fig_obj.tight_layout()
        figtitle = 'obj_{}.pdf'.format(lambda_)
        fig_obj.savefig(figtitle)

    return S_star_approx
