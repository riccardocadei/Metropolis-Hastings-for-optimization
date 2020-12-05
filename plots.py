import numpy as np
import matplotlib.pyplot as plt

from helpers import *
from markov_algos import *
from DatasetGenerator import *


def sample_S_approx(G, betas, lambda_, n_iter, nb_instances, verbose=False):
    """
    Compute nb_instances approximation of S* for different instances of G.
    The approximation follows the simulated annealing algorithm. The starting state
    is chosen randomly.

    Returns
    -------
    S_approxs: list of S approximated by solving many times the problem over multiple instances of G at lambda_ fixed
    list_params: list of parameters used
    """
    list_S_approx = []
    list_lambdas = []
    list_datas = []

    for k in range(nb_instances):
        data = G()
        starting_state = np.random.randint(0, 2, data.N)

        S_approx = simulated_annealing(starting_state, betas, n_iter, lambda_, data, verbose)

        list_S_approx.append(S_approx)
        list_lambdas.append(lambda_)
        list_datas.append(data)

        if verbose:
            print("[lambda={} : {}/{}]".format(lambda_, k + 1, nb_instances))

    return list_S_approx, list_lambdas, list_datas

# def avg_size_S(G, betas, lambda_, n_iter, nb_instances, verbose=False):
#     """
#     Compute the average size of selected cities for the (approximated) best set S.
#     """
#     list_S_approx, _, _ = sample_S_approx(G, betas, lambda_, n_iter, nb_instances, verbose)
#     return np.sum(list(map(len, list_S_approx))) / nb_instances

# def avg_obj_S(G, betas, lambda_, n_iter, nb_instances, verbose=False):
#     """
#     Compute the average value of the objective function for the (approximated) best set S.
#     """
#     list_S_approx, list_lambdas, list_datas = sample_S_approx(G, betas, lambda_, n_iter, nb_instances, verbose)
#     return np.sum(list(map(f, list_S_approx, list_lambdas, list_datas))) / nb_instances

# def plot_avg_size(G, lambdas, betas, n_iter, nb_instances, verbose=False):
#     E = [avg_size_S(G, betas, lambda_, n_iter, nb_instances, verbose) for lambda_ in lambdas]

#     plt.plot(lambdas, E, 'o')
#     plt.title("Expectation of the number of cities over multiple instances of G1 for different lambdas")
#     plt.xlabel("Lambda")
#     plt.ylabel("Expectation")
#     plt.savefig('plots/avg_size_{}to{}.pdf'.format(lambdas[0], lambdas[-1]))

# def plot_avg_obj(G, lambdas, betas, n_iter, nb_instances, verbose=False):
#     E = [avg_obj_S(G, betas, lambda_, n_iter, nb_instances, verbose) for lambda_ in lambdas]

#     plt.plot(lambdas, E, 'o')
#     plt.title("Expectation of the objective function over multiple instances of G1 for different lambdas")
#     plt.xlabel("Lambda")
#     plt.ylabel("Expectation")
#     plt.savefig('plots/avg_obj_{}to{}.pdf'.format(lambdas[0], lambdas[-1]))

def avg(G, betas, lambda_, n_iter, nb_instances, verbose=False):
    list_S_approx, list_lambdas, list_datas = sample_S_approx(G, betas, lambda_, n_iter, nb_instances, verbose)
    avg_obj = np.sum(list(map(f, list_S_approx, list_lambdas, list_datas))) / nb_instances
    avg_size = np.sum(list(map(len, list_S_approx))) / nb_instances

    return [avg_obj, avg_size]

def plot_avg_lambda(G, lambdas, betas, n_iter, nb_instances, verbose=False):
    E = [avg(G, betas, lambda_, n_iter, nb_instances, verbose) for lambda_ in lambdas]
    E = np.array(E)

    fig_obj, ax_obj = plt.subplots(figsize=(1 + len(lambdas), 4))
    ax_obj.plot(lambdas, E[:, 0], '+', ':')
    ax_obj.set_xlabel("Lambda")
    ax_obj.set_ylabel("Average max obj")

    fig_size, ax_size = plt.subplots(figsize=(1 + len(lambdas), 4))
    ax_size.plot(lambdas, E[:, 1], '+', ':')
    ax_size.set_xlabel("Lambda")
    ax_size.set_ylabel("Average maxi size")

    fig_obj.savefig('plots/avg_obj_{}to{}.pdf'.format(lambdas[0], lambdas[-1]))
    fig_size.savefig('plots/avg_size_{}to{}.pdf'.format(lambdas[0], lambdas[-1]))
    fig_obj.show()
    fig_size.show()
