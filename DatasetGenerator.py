"""
Class to generate a dataset of cities.
"""
import scipy.stats as st
import numpy as np


class DatasetGenerator(object):
    def __init__(self, N=100):
        self.N = N
        self.x = None
        self.v = None
        self.d = None
        self.refresh()

    def refresh(self):
        raise Exception("undefined")

class G1(DatasetGenerator):
    def refresh(self):
        self.x = st.uniform().rvs((self.N,2))
        self.v = st.uniform().rvs((self.N,))
        self.d = sp.spatial.distance.cdist(self.x, self.x, 'euclidean')

class G2(DatasetGenerator):
    def refresh(self):
        self.x = st.uniform().rvs((self.N,2))
        self.v = np.exp(st.norm(-0.85, 1.3).rvs((self.N,)))
        self.d = sp.spatial.distance.cdist(self.x, self.x, 'euclidean')
