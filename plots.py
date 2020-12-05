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

    starting_state = np.random.randint(0, 2, data.N)
    for k in range(nb_instances):
        data = G()

        S_approx = simulated_annealing(starting_state, betas, n_iter, lambda_, data, verbose)

        list_S_approx.append(S_approx)
        list_lambdas.append(lambda_)
        list_datas.append(data)

        if verbose:
            print("[lambda={} : {}/{}]".format(lambda_, k + 1, nb_instances))

    return list_S_approx, list_lambdas, list_datas

def avg(G, betas, lambda_, n_iter, nb_instances, verbose=False):
    list_S_approx, list_lambdas, list_datas = sample_S_approx(G, betas, lambda_, n_iter, nb_instances, verbose)
    avg_obj = np.sum(list(map(f, list_S_approx, list_lambdas, list_datas))) / nb_instances
    avg_size = np.sum(list(map(len, list_S_approx))) / (nb_instances * 100)

    return [avg_obj, avg_size]

def plot_avg_lambda(G, lambdas, betas, n_iter, nb_instances, verbose=False):
    E = [avg(G, betas, lambda_, n_iter, nb_instances, verbose) for lambda_ in lambdas]
    E = np.array(E)

    fig_obj, ax_obj = plt.subplots(figsize=(1 + len(lambdas), 4))
    ax_obj.plot(lambdas, E[:, 0], '+', ls=':')
    ax_obj.set_xlabel("Lambda")
    ax_obj.set_ylabel("Average max obj")

    fig_size, ax_size = plt.subplots(figsize=(1 + len(lambdas), 4))
    ax_size.plot(lambdas, E[:, 1], '+', ls=':')
    ax_size.set_xlabel("Lambda")
    ax_size.set_ylabel("Average maxi size")

    fig_obj.savefig('plots/avg_obj_{}to{}.pdf'.format(lambdas[0], lambdas[-1]))
    fig_size.savefig('plots/avg_size_{}to{}.pdf'.format(lambdas[0], lambdas[-1]))
    fig_obj.show()
    fig_size.show()
