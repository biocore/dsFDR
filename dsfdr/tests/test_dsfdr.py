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
                              [0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 1, 0, 0, 0, 3, 0, 1],
                              [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]])

        np.random.seed(31)
        self.data_sim, self.labels_sim = simulation.simulatedat(
            numsamples=10, numdiff=100, numc=100,
            numd=800, sigma=0.1, normalize=False, numreads=10000)

    def test_dsfdr(self):
        # test on dummy self.data
        res_ds = dsfdr.dsfdr(self.data, self.labels, method='meandiff',
                             transform_type=None, alpha=0.1, numperm=1000,
                             fdr_method='dsfdr')
        np.testing.assert_array_equal(res_ds[0], [True, False, False, False, True])

        res_bh = dsfdr.dsfdr(self.data, self.labels, method='meandiff',
                             transform_type=None, alpha=0.1, numperm=1000,
                             fdr_method='bhfdr')
        np.testing.assert_array_equal(res_bh[0], [True, False, False, False, True])

        res_by = dsfdr.dsfdr(self.data, self.labels, method='meandiff',
                             transform_type=None, alpha=0.1, numperm=1000,
                             fdr_method='byfdr')
        np.testing.assert_array_equal(res_by[0], [True, False, False, False, True])

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

    def test_dsfdr_filterBH_simple(self):
        reject, tstat, pval = dsfdr.dsfdr(self.data, self.labels, method='meandiff',
                                          transform_type=None, alpha=0.1, numperm=1000,
                                          fdr_method='filterBH')
        # test rejects are correct
        np.testing.assert_array_equal(reject, [True, False, False, False, True])

        # test features were pre-filtered ok by looking at the tstats
        # where a feature was pre-filtered, we should have a nan there
        np.testing.assert_array_equal(np.isnan(tstat), [False, False, True, True, False])

        # and check p-vals are as expected
        np.testing.assert_array_equal(pval < 0.1, [True, False, False, False, True])

    def test_dsfdr_filterBH_simulation(self):
        # mix the simulated data so the rejects should be position 100:199
        new_data = self.data_sim[np.hstack([np.arange(900, 1000), np.arange(900)]), :]
        reject, tstat, pval = dsfdr.dsfdr(new_data, self.labels_sim,
                                          method='meandiff', transform_type=None,
                                          alpha=0.1, numperm=1000,
                                          fdr_method='filterBH')
        # test we reject enough
        self.assertTrue(np.sum(reject) > 100)

        # test we control the FDR
        okpos = np.where(reject)[0]
        self.assertLessEqual((np.sum((okpos < 100) | (okpos > 199)) / np.sum(reject)), 0.1)

        # test we pre-filtered a lot of features
        self.assertGreaterEqual(np.sum(np.isnan(tstat)), 400)


if __name__ == '__main__':
    main()
