import unittest
from data import HyperbolicPairsDataset
from itertools import product
from utils import reset_rngs, hyperbolic_dist_np
import numpy as np


class TestDataset(unittest.TestCase):
    def test_n2_various_cs(self):
        reset_rngs(777)

        settings = {
            #'seed': 777,
            'inverse_transform': 'hyperbolic',
            'min_r': 0.1,
            'max_r': 5.3,
            'n_samples': 1000
        }

        dims = range(2, 16)
        curvs = [0.9, 0.99, 1, 1.01, 1.1]

        for dim, curv in product(dims, curvs):
            print(f'Checking {dim}, {curv}')
            dataset = HyperbolicPairsDataset(dim=dim, curv=curv, **settings)
            assert not np.isnan(dataset.pairs.numpy()).any()
            assert not np.isnan(dataset.distances.numpy()).any()
