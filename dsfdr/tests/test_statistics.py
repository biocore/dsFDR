from unittest import TestCase, main

import numpy as np
import numpy.testing as npt
from dsfdr import statistics


class StatisticsTests(TestCase):

    def setUp(self):s
        self.labels = np.array([1, 1, 1, 1, 0, 0, 0, 0])
        self.data = np.array([[0, 1, 3, 5, 10, 30, 40, 50],
                              [1, 2, 3, 4, 4, 3, 2, 1],
                              [0, 0, 0, 1, 0, 3, 0, 1]])

        self.labels2 = np.array([2, 2, 2, 2, 1, 1, 1, 1, 0, 0, 0, 0])
        self.data2 = np.array([[-1, 15, 2, 4, 0, 1, 3, 5, 10, 30, 40, 50],
                               [2, 1, 3, 4, 1, 2, 3, 4, 4, 3, 2, 1],
                               [1, 3, 3, 0, 0, 0, 0, 1, 0, 3, 0, 1]])

    def test_meandiff(self):
        res = statistics.meandiff(self.data, self.labels)
        self.assertEqual(len(res), self.data.shape[0])
        self.assertEqual(res[0], -30.25)
        self.assertEqual(res[1], 0)
        self.assertEqual(res[2], -0.75)

    def test_stdmeandiff(self):
        res = statistics.stdmeandiff(self.data, self.labels)
        self.assertEqual(len(res), self.data.shape[0])
        npt.assert_almost_equal(res[0], -1.57, decimal=2)
        self.assertEqual(res[1], 0)
        npt.assert_almost_equal(res[2], -0.39, decimal=2)

    def test_mannwhitney(self):
        res = statistics.mannwhitney(self.data, self.labels)
        self.assertEqual(len(res), self.data.shape[0])
        self.assertEqual(res[0], 0)
        self.assertEqual(res[1], 8)
        self.assertEqual(res[2], 5.5)

    def test_kruwallis(self):
        res = statistics.kruwallis(self.data2, self.labels2)
        self.assertEqual(len(res), self.data.shape[0])
        npt.assert_almost_equal(res[0], 6.58, decimal=2)
        self.assertEqual(res[1], 0)
        npt.assert_almost_equal(res[2], 2.55, decimal=2)

    def pearson(self):
        res = statistics.pearson(self.data, self.labels)
        self.assertEqual(len(res), self.data.shape[0])
        npt.assert_almost_equal(res[0], -0.82, decimal=2)
        self.assertEqual(res[1], 0)
        npt.assert_almost_equal(res[2], -0.38, decimal=2)

    def spearman(self):
        res = statistics.spearman(self.data, self.labels)
        self.assertEqual(len(res), self.data.shape[0])
        npt.assert_almost_equal(res[0], -0.87, decimal=2)
        self.assertEqual(res[1], 0)
        npt.assert_almost_equal(res[2], -0.31, decimal=2)


if __name__ == '__main__':
    main()
