from unittest import TestCase, main

import numpy as np
from dsfdr import transform


class TransformTests(TestCase):
    def setUp(self):
        self.data = np.array([[0, 1, 3, 5, 10, 30, 40, 50],
                              [1, 2, 3, 4, 4, 3, 2, 1],
                              [0, 0, 0, 1, 0, 3, 0, 1]])

    def test_rankdata(self):
        tdata = transform.rankdata(self.data)
        self.assertEqual(np.shape(tdata), self.data.shape)
        np.testing.assert_array_equal(tdata[0, :], [1, 2, 3, 4, 5, 6, 7, 8])
        np.testing.assert_array_equal(tdata[1, :],
                                      [1.5, 3.5, 5.5, 7.5, 7.5, 5.5, 3.5, 1.5])
        np.testing.assert_array_equal(tdata[2, :],
                                      [3, 3, 3, 6.5, 3, 8, 3, 6.5])

    def test_log2data(self):
        tdata = transform.log2data(self.data)
        self.assertEqual(np.shape(tdata), self.data.shape)
        np.testing.assert_array_almost_equal(tdata[0, :], [1, 1, 1.58,
                                             2.32, 3.32, 4.9, 5.32, 5.64],
                                             decimal=2)
        np.testing.assert_array_almost_equal(tdata[1, :], [1, 1, 1.58,
                                             2, 2, 1.58, 1, 1], decimal=2)
        np.testing.assert_array_almost_equal(tdata[2, :], [1, 1, 1, 1, 1,
                                             1.58, 1, 1], decimal=2)

    def test_binarydata(self):
        tdata = transform.binarydata(self.data)
        self.assertEqual(np.shape(tdata), self.data.shape)
        np.testing.assert_array_equal(tdata[0, :], [0, 1, 1, 1, 1, 1, 1, 1])
        np.testing.assert_array_equal(tdata[1, :], [1, 1, 1, 1, 1, 1, 1, 1])
        np.testing.assert_array_equal(tdata[2, :], [0, 0, 0, 1, 0, 1, 0, 1])

    def test_normdata(self):
        tdata = transform.normdata(self.data)
        self.assertEqual(np.shape(tdata), self.data.shape)
        np.testing.assert_array_almost_equal(tdata[0, :], [0, 0.33,
                                             0.5, 0.5, 0.71, 0.83, 0.95, 0.96],
                                             decimal=2)
        np.testing.assert_array_almost_equal(tdata[1, :], [1, 0.67,
                                             0.5, 0.4, 0.29, 0.08, 0.05, 0.02],
                                             decimal=2)
        np.testing.assert_array_almost_equal(tdata[2, :], [0, 0, 0, 0.1, 0,
                                             0.08, 0, 0.02], decimal=2)


if __name__ == '__main__':
    main()