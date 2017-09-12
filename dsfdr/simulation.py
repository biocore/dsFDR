import numpy as np
from gneiss.cluster import random_linkage
from gneiss.balances import balance_basis
from skbio.stats.composition import ilr
import pandas as pd


def simulatedatbalance(numsamples=5, numdiff=100, numc=100, numd=800,
                       sigma=0.1, numreads=10000,
                       pseudo=0.0001):
    """ Generates simulation with tree and balances

    input:
    numsamples : int
        number of samples in each group
    numdiff : int
        number of different bacteria between the groups
    numc : int
        number of high freq. bacteria similar between the groups
    numd : int
        number of low freq. bacteria similar between the groups
    sigma : float
        the standard deviation

    output:
    data
        the underlying simulated data
    labels
        metadata category labels
    balances,
        the ground truth balances
    tree
        the ground truth tree
    """

    data, labels = simulatedat(numsamples=numsamples,
                               numdiff=numdiff,
                               numc=numc,
                               numd=numd,
                               sigma=sigma,
                               numreads=numreads)
    n = data.shape[0]
    tree = random_linkage(n)
    group1 = np.arange(0, numdiff // 2)
    group2 = np.arange(numdiff // 2, numdiff)
    group3 = np.arange(numdiff, numdiff+numc+numd)
    balances = []
    for i in group1:
        for j in group2:
            if j > i:
                ti, tj = tree.find(str(i)), tree.find(str(j))
                lca = tree.lowest_common_ancestor([ti, tj])
                balances.append(lca.name)

    for i in group1:
        for j in group3:
            if j > i:
                ti, tj = tree.find(str(i)), tree.find(str(j))
                lca = tree.lowest_common_ancestor([ti, tj])
                balances.append(lca.name)


    for i in group2:
        for j in group3:
            if j > i:
                ti, tj = tree.find(str(i)), tree.find(str(j))
                lca = tree.lowest_common_ancestor([ti, tj])
                balances.append(lca.name)

    balances = list(set(balances))
    basis, _ = balance_basis(tree)
    data = ilr(data.T+pseudo, basis).T
    data = pd.DataFrame(data,
                        index=[n.name for n in tree.levelorder()
                               if not n.is_tip()])
    return data, labels, balances, tree


def simulatedat(numsamples=5, numdiff=100, numc=100, numd=800,
                sigma=0.1, numreads=10000):
    '''
    new simulation code

    input:
    numsamples : int
        number of samples in each group
    numdiff : int
        number of different bacteria between the groups
    numc : int
        number of high freq. bacteria similar between the groups
    numd : int
        number of low freq. bacteria similar between the groups
    sigma : float
        the standard deviation
    '''

    A = np.zeros([int(numdiff), 2 * numsamples])
    for i in range(int(numdiff)):
        mu_H = np.random.uniform(0.1, 1)
        mu_S = np.random.uniform(1.1, 2)
        h = np.random.normal(mu_H, sigma, numsamples)
        s = np.random.normal(mu_S, sigma, numsamples)
        # zero inflation
        h[h < 0] = 0
        s[s < 0] = 0
        # randomize the difference in S or H groups
        coin = np.random.randint(2)
        if coin == 0:
            A[i, :] = np.hstack((h, s))
        else:
            A[i, :] = np.hstack((s, h))

    C = np.zeros([numc, 2 * numsamples])
    for j in range(numc):
        mu = np.random.uniform(10, 11)
        C[j, :] = np.random.normal(mu, sigma, 2 * numsamples)

    numnoise = np.random.randint(1, 7, numd)
    D = np.zeros([numd, 2 * numsamples])
    for k in range(numd):
        for cnoise in range(numnoise[k]):
            cpos = np.random.randint(2 * numsamples)
            D[k, cpos] = np.random.uniform(0.1, 1)

    data = np.vstack((A, C, D))


    data = data / np.sum(data, axis=0)


    _data = []
    for i in range(data.shape[1]):
        _data.append(
            np.random.multinomial(numreads, data[:, i])
        )
    data = np.vstack(_data).T

    # labels
    x = np.array([0, 1])
    labels = np.repeat(x, numsamples)
    labels = (labels == 1)

    return (data, labels)
