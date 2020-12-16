import numpy as np

from helpers import preprocessing_data, submission, f
from markov_algos import simulated_annealing

"""
The idea will be to spam run.py as it will create "Aquarium_submission_N_{}_score_{}.csv" files with the number of cities 
(first or second data set) and the score of the run.

When the time is over, we'll just submit the best one we got for the two N they'll give us
"""

### Given in the assignment
lambda_ = 100

### TO TUNE FOR A DATASET OF SIZE N = tens of thousands of cities AND N = thousands of cities. So N = 50 000 and N = 5 000 ???
"""
If you have time, you can even do N = 1000,2000,3000...
Another idea would be to create a function 'def tuning(N)' that for a given N, returns the best n_iter and betas
"""

n_iter = 10000 #TO TUNE
betas = np.logspace(0, 3, 7) #TO TUNE

def compute_Sapprox(csv_file):
    starting_state, data = preprocessing_data(csv_file)
    N = len(starting_state)
    Sapprox = simulated_annealing(starting_state, betas, n_iter, lambda_, data, verbose=False, plot=False)

    #Computing score of this run with the Sapprox found and save it in a file called Aquarium_submission_SCORE with SCORE being the score of the run.
    score = f(np.array(Sapprox), lambda_, data)
    submission(csv_file, Sapprox, "Aquarium_submission_N_{}_score_{}.csv".format(N,score))


compute_Sapprox("input_excel.csv")