import numpy as np
import matplotlib.pyplot as plt

from helpers import *
from markov_algos import *
from DatasetGenerator import *


def sample_S_approx(datas, betas, lambda_, n_iter, verbose=False):
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


    starting_state = np.zeros(datas[0].N)

    for k in range(len(datas)):

        starting_state[ datas[k].v == np.max(datas[k].v)] = 1
        S_approx = simulated_annealing(starting_state, betas, n_iter, lambda_, datas[k], verbose)

        list_S_approx.append(S_approx)
        list_lambdas.append(lambda_)

        if verbose:
            print("[lambda={} : {}/{}]".format(lambda_, k + 1, len(datas)))
        starting_state[ datas[k].v == np.max(datas[k].v)] = 0

    return list_S_approx, list_lambdas

def avg(datas, betas, lambda_, n_iter, verbose=False):
    list_S_approx, list_lambdas = sample_S_approx(datas, betas, lambda_, n_iter, verbose)
    avg_obj = np.sum(list(map(f, list_S_approx, list_lambdas, datas))) / len(datas)
    avg_size = np.sum(list(map(len, list_S_approx))) / len(datas)

    return [avg_obj, avg_size]

def plot_avg_lambda(G, lambdas, betas, n_iter, nb_instances, verbose=False):
    datas = [G() for i in range(nb_instances)]
    E = [avg(datas, betas, lambda_, n_iter, verbose) for lambda_ in lambdas]
    E = np.array(E)

    fig, ax_obj = plt.subplots(figsize=(1 + len(lambdas), 6), constrained_layout=True)
    ax_obj.plot(lambdas, E[:, 0], 'b+', ls=':')
    ax_obj.set_xlabel("Lambda", fontsize=25)
    ax_obj.set_ylabel("Average maximum of f", color='blue', fontsize=25)
    ax_obj.tick_params(labelsize=17)

    ax_size = ax_obj.twinx()
    ax_size.plot(lambdas, E[:, 1], 'r+', ls=':')
    ax_size.set_xlabel("Lambda", fontsize=25)
    ax_size.set_ylabel("Average size of S maximizing", color='red', fontsize=25)
    ax_size.tick_params(labelsize=17)


    data_name = "G1" if isinstance(datas[0], G1) else "G2"

    plt.tight_layout()
    fig.savefig('plots\\avg_{}.pdf'.format(data_name))
