from unittest import TestCase, main

import numpy as np
from dsfdr import dsfdr
from dsfdr import simulation


class fdr_methodsTests(TestCase):

    def setUp(self):
        np.random.seed(31)
        self.labels = np.array([1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0])
        self.data = np.array([[0, 1, 3, 5, 0, 1, 100, 300, 400, 500, 600, 700],
                              [1, 2, 3, 4, 5, 6, 6, 5, 4, 3, 2, 1],
                              [0, 0, 0, 0, 0, 1, 0, 0, 0, 3, 0, 1]])

        np.random.seed(31)
        self.data_sim, self.labels_sim = simulation.simulatedat(
            numsamples=10, numdiff=100, numc=100,
            numd=800, sigma=0.1, normalize=False, numreads=10000)

    def test_dsfdr(self):
        # test on dummy self.data
        res_ds = dsfdr.dsfdr(self.data, self.labels, method='meandiff',
                             transform_type=None, alpha=0.1, numperm=1000,
                             fdr_method='dsfdr')
        self.assertEqual(np.shape(res_ds)[0], self.data.shape[0])
        np.testing.assert_array_equal(res_ds[0], [True, False, False])

        res_bh = dsfdr.dsfdr(self.data, self.labels, method='meandiff',
                             transform_type=None, alpha=0.1, numperm=1000,
                             fdr_method='bhfdr')
        self.assertEqual(np.shape(res_bh)[0], self.data.shape[0])
        np.testing.assert_array_equal(res_bh[0], [True, False, False])

        res_by = dsfdr.dsfdr(self.data, self.labels, method='meandiff',
                             transform_type=None, alpha=0.1, numperm=1000,
                             fdr_method='byfdr')
        self.assertEqual(np.shape(res_by)[0], self.data.shape[0])
        np.testing.assert_array_equal(res_by[0], [True, False, False])

        # test on simulated self.data_sim
        np.random.seed(31)
        res_ds2 = dsfdr.dsfdr(self.data_sim, self.labels_sim,
                              method='meandiff', transform_type=None,
                              alpha=0.1, numperm=1000, fdr_method='dsfdr')[0]
        fdr_ds2 = (np.sum(np.where(res_ds2)[0] >= 100)) / np.sum(res_ds2)
        np.testing.assert_equal(fdr_ds2 <= 0.1, True)

        np.random.seed(31)
        res_bh2 = dsfdr.dsfdr(self.data_sim, self.labels_sim,
                              method='meandiff', transform_type=None,
                              alpha=0.1, numperm=1000, fdr_method='bhfdr')[0]
        self.assertEqual(np.shape(res_bh2)[0], self.data_sim.shape[0])
        fdr_bh2 = (np.sum(np.where(res_bh2)[0] >= 100)) / np.sum(res_bh2)
        np.testing.assert_equal(fdr_bh2 <= 0.1, True)

        np.random.seed(31)
        res_by2 = dsfdr.dsfdr(self.data_sim, self.labels_sim,
                              method='meandiff', transform_type=None,
                              alpha=0.1, numperm=1000, fdr_method='byfdr')[0]
        self.assertEqual(np.shape(res_by2)[0], self.data_sim.shape[0])
        fdr_by2 = (np.sum(np.where(res_by2)[0] >= 100)) / np.sum(res_by2)
        np.testing.assert_equal(fdr_by2 <= 0.1, True)


if __name__ == '__main__':
    main()
