import unittest
from utils import hyperbolic_volume, e, hyperbolic_volume_inverse
import numpy as np
import matplotlib.pyplot as plt
from itertools import product


class TestHyperbolicVolume(unittest.TestCase):
    def setUp(self):
        self.r = [0.00001, 0.0001, 0.001, 0.01, 0.1, 0.5, 1, 2, 5, 10, 20, 40, 80]

    def test_n2_various_cs(self):
        print()
        print('dim 2, various curv tests')
        n = 2
        cs = [0.01, 0.1, 0.5, 1, 2, 5, 10]

        for c in cs:
            vol = lambda r: hyperbolic_volume(n, c, r)

            # check for nans
            y = np.array([vol(r) for r in self.r])
            print(f'computed values: {y}')
            self.assertFalse(np.isnan(y).any())

            # check if values match
            print('True values formula: 2*pi/c * (cosh(sqrt(c)r) - 1)')
            y_true = np.array([2 * np.pi/c * (np.cosh(np.sqrt(c) * r) - 1) for r in self.r])
            print(f'true values: {y_true}')
            cond = np.allclose(y, y_true)
            self.assertTrue(cond)

    def test_n3_c1(self):
        print()
        print('dim 3, curv 1 tests')
        n = 3
        c = 1
        vol = lambda r: hyperbolic_volume(n, c, r)

        # check for nans
        y = np.array([vol(r) for r in self.r])
        print(f'computed values: {y}')
        self.assertFalse(np.isnan(y).any())

        # check if values match
        print('True values formula: 2*pi*(sinh(2r)/2 - r)')
        y_true = np.array([2 * np.pi * (np.sinh(2*r)/2 - r) for r in self.r])
        print(f'true values: {y_true}')
        cond = np.allclose(y, y_true)
        self.assertTrue(cond)

    def test_n4_c1(self):
        print()
        print('dim 4, curv 1 tests')
        n = 4
        c = 1
        vol = lambda r: hyperbolic_volume(n, c, r)

        # check for nans
        y = np.array([vol(r) for r in self.r])
        print(f'computed values: {y}')
        self.assertFalse(np.isnan(y).any())

        # check if values match
        print('True values formula: e(3) * 4/3 sinh(r/2)^4 (cosh(r) + 2)')
        y_true = np.array([e(3) * 4 / 3 * (np.sinh(r/2) ** 4) * (np.cosh(r) + 2) for r in self.r])
        print(f'true values: {y_true}')
        cond = np.allclose(y, y_true)
        self.assertTrue(cond)

    def test_n5_c1(self):
        print()
        print('dim 5, curv 1 tests')
        n = 5
        c = 1
        vol = lambda r: hyperbolic_volume(n, c, r)

        # check for nans
        y = np.array([vol(r) for r in self.r])
        print(f'computed values: {y}')
        self.assertFalse(np.isnan(y).any())

        # check if values match
        print('True values formula: e(4) * 1/32 (12r - 8sinh(2r) + sinh(4r))')
        y_true = np.array([e(4) * 1 / 32 * (12*r - 8*np.sinh(2*r) + np.sinh(4*r)) for r in self.r])
        print(f'true values: {y_true}')
        cond = np.allclose(y, y_true)
        self.assertTrue(cond)


class TestHyperbolicVolumeInverse(unittest.TestCase):
    def test_maxr10_dims_leq16(self):
        print()
        print('various dims and curvs tests')
        ns = range(2, 17)
        cs = [0.7, 0.9, 0.95, 1, 2, 5, 10]
        min_r_sample = 0.09
        max_r_sample = 5.31
        n_samples = 100000

        min_r_test = 0.1
        max_r_test = 5.3
        r_true = np.linspace(min_r_test, max_r_test, 1000)

        for n, c in product(ns, cs):
            print(f'n: {n}. c: {c}')
            vol = lambda r: hyperbolic_volume(n, c, r)
            inv = hyperbolic_volume_inverse(n, c, min_r_sample, max_r_sample, n_samples)

            #print(f'true r: {r_true}')

            max_vol = vol(max_r_sample)
            min_vol = vol(min_r_sample)
            v_true = np.array([(vol(r) - min_vol)/(max_vol - min_vol) for r in r_true])
            #print(f'true normalized vols: {v_true}')

            r_approx = [inv(v) for v in v_true]
            #print(f'approx r: {r_approx}')
            cond = np.allclose(r_true, r_approx, atol=1e-2, rtol=0)
            self.assertTrue(cond)
