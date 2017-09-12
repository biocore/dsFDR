import unittest
import numpy as np
from dsfdr.simulation import simulatedatbalance


class TestBalanceSimulation(unittest.TestCase):

    def test_balances_and_tree(self):
        np.random.seed(0)
        data, labels, balances, tree = simulatedatbalance(
            numsamples=5, numdiff=4, numc=2, numd=3,
            sigma=0.1, numreads=100)


if __name__ == "__main__":
    unittest.main()
