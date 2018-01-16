import numpy as np


def simulatedat(numsamples=5, numdiff=100, numc=100, numd=800,
                sigma=0.01, normalize=False, numreads=10000):
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

    if normalize is True:
        data = data / np.sum(data, axis=0)
        # normalize by column

    # labels
    x = np.array([0, 1])
    labels = np.repeat(x, numsamples)
    labels = (labels == 1)

    return (data, labels)


def simulatedat2(numsamples=5, numdiff=100, numc=100, numd=800,
                 sigma=2, normalize=False, numreads=10000):
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
        #mu_H = np.random.uniform(5, 6)
        #mu_S = np.random.uniform(3, 4)
        mu_H = np.random.uniform(5, 5.5)
        mu_S = np.random.uniform(5.5, 5.7)
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
        mu = np.random.uniform(20, 21)
        C[j, :] = np.random.normal(mu, sigma, 2 * numsamples)

    numnoise = np.random.randint(1, 7, numd)
    D = np.zeros([numd, 2 * numsamples])
    for k in range(numd):
        for cnoise in range(numnoise[k]):
            cpos = np.random.randint(2 * numsamples)
            D[k, cpos] = np.random.uniform(0.1, 1)

    data = np.vstack((A, C, D))

    if normalize is True:
        data = data / np.sum(data, axis=0)
        # normalize by column

    # labels
    x = np.array([0, 1])
    labels = np.repeat(x, numsamples)
    labels = (labels == 1)

    return (data, labels)


def simulatedat3(numsamples=5, numdiff=100, numc=100, numd=800,
                 sigma=2, normalize=False, numreads=10000):
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
        mu_H = np.random.uniform(1, 5)
        mu_S = np.random.uniform(25, 30)
        sigma_H = 0.1
        sigma_S = 5
        h = np.random.normal(mu_H, sigma_H, numsamples)
        s = np.random.normal(mu_S, sigma_S, numsamples)
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
        mu = np.random.uniform(20, 21)
        C[j, :] = np.random.normal(mu, sigma, 2 * numsamples)

    numnoise = np.random.randint(1, 7, numd)
    D = np.zeros([numd, 2 * numsamples])
    for k in range(numd):
        for cnoise in range(numnoise[k]):
            cpos = np.random.randint(2 * numsamples)
            D[k, cpos] = np.random.uniform(0.1, 1)

    data = np.vstack((A, C, D))

    if normalize is True:
        data = data / np.sum(data, axis=0)
        # normalize by column

    # labels
    x = np.array([0, 1])
    labels = np.repeat(x, numsamples)
    labels = (labels == 1)

    return (data, labels)


# def simulatecompdat(numsamples=5, numdiff=100, numc=100, numd=800,
#                     sigma=0.01, normalize=True):
#     '''
#     new simulation code for compositional data

#     input:
#     numsamples : int
#         number of samples in each group
#     numdiff : int
#         number of different bacteria between the groups
#     numc : int
#         number of high freq. bacteria similar between the groups
#     numd : int
#         number of low freq. bacteria similar between the groups
#     sigma : float
#         the standard deviation
#     '''

#     A = np.zeros([int(numdiff), 2 * numsamples])
#     for i in range(int(numdiff)):
#         mu_H = np.random.uniform(0.1, 1)
#         mu_S = np.random.uniform(1.1, 2)
#         h = np.random.normal(mu_H, sigma, numsamples)
#         s = np.random.normal(mu_S, sigma, numsamples)
#         # zero inflation
#         h[h < 0] = 0
#         s[s < 0] = 0
#         # randomize the difference in S or H groups
#         coin = np.random.randint(2)
#         if coin == 0:
#             A[i, :] = np.hstack((h, s))
#         else:
#             A[i, :] = np.hstack((s, h))
#     # the first 10% bacteria in A have high frequency
#     A[0:int(0.1 * numdiff)] = 100 * A[0:int(0.1 * numdiff)]

#     C = np.zeros([numc, 2 * numsamples])
#     for j in range(numc):
#         mu = np.random.uniform(0.1, 1)
#         C[j, :] = np.random.normal(mu, sigma, 2 * numsamples)
#     # first 10% bacteria in C has high frequency
#     C[0:int(0.1 * numc)] = 100 * C[0:int(0.1 * numc)]

#     numnoise = np.random.randint(1, 7, numd)
#     D = np.zeros([numd, 2 * numsamples])
#     for k in range(numd):
#         for cnoise in range(numnoise[k]):
#             cpos = np.random.randint(2 * numsamples)
#             D[k, cpos] = np.random.uniform(0.1, 1)

#     data = np.vstack((A, C, D))

#     if normalize is True:
#         data = data / np.sum(data, axis=0)
#         # normalize by column

#     # labels
#     x = np.array([0, 1])
#     labels = np.repeat(x, numsamples)
#     labels = (labels == 1)

#     return (data, labels)
