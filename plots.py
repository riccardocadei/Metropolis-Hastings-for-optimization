import scipy.stats as st
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import copy
import time


def best_S(G, betas, lambda_, n_iter, nb_instances, verbose=False):
    """
    Compute approximation of S* for nb_instances different instances of G.
    The approximation follows the simulated annealing algorithm.

    Returns
    -------
    S_approxs: list of S approximated by solving many times the problem over multiple instances of G at lambda_ fixed
    list_params: list of parameters used
    """
    list_S_approx = []
    list_params = []

    for k in range(nb_instances):
        g = G()
        n, coords, pop = g1.N, g1.x, g1.v
        params = (lambda_, n, coords, pop)
        x = np.random.randint(0, 2, n)

        # run the Metropolis-Hastings algorithm for each beta == simulated annealing
        for beta in betas:
            x = metropolis_hastings(beta, x, n_iter, params)

        list_S_approx.append(vect_to_S(x))
        list_params.append(params)

        if verbose:
            print("[lambda={} : {}/{}]".format(lambda_, k + 1, nb_instances))

    return list_S_approx, list_params

def avg_size_S(G, betas, lambda_, n_iter, nb_instances, verbose=False):
    """
    Compute the average size of selected cities for the (approximated) best set S.
    """
    S_approxs, _ = best_S(G, betas, lambda_, n_iter, nb_instances, verbose)
    return np.sum(list(map(len, S_approxs))) / nb_instances

def avg_obj_S(G, betas, lambda_, n_iter, nb_instances, verbose=False):
    """
    Compute the average value of the objective function for the (approximated) best set S.
    """
    S_approxs, list_params = best_S(G, betas, lambda_, n_iter, nb_instances, verbose)
    return np.sum(list(map(f, S_approxs, list_params))) / nb_instances

def plot_avg_size(G, lambdas, betas, n_iter, nb_instances, verbose=False):
    E = [avg_size_S(G, betas, lambda_, n_iter, nb_instances, verbose) for lambda_ in lambdas]

    plt.plot(lambdas, E, 'o')
    plt.title("Expectation of the number of cities over multiple instances of G1 for different lambdas")
    plt.xlabel("Lambda")
    plt.ylabel("Expectation")
    plt.savefig('avg_size_{}to{}.pdf'.format(lambdas[0], lambdas[-1]))

def plot_avg_obj(G, lambdas, betas, n_iter, nb_instances, verbose=False):
    E = [avg_obj_S(G, betas, lambda_, n_iter, nb_instances, verbose) for lambda_ in lambdas]

    plt.plot(lambdas, E, 'o')
    plt.title("Expectation of the objective function over multiple instances of G1 for different lambdas")
    plt.xlabel("Lambda")
    plt.ylabel("Expectation")
    plt.savefig('avg_obj_{}to{}.pdf'.format(lambdas[0], lambdas[-1]))
