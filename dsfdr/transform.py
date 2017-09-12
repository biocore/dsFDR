import numpy as np
import scipy.stats
from skbio.stats.composition import clr


# data transformation
def rankdata(data):
    rdata = np.zeros(np.shape(data))
    for crow in range(np.shape(data)[0]):
        rdata[crow, :] = scipy.stats.rankdata(data[crow, :])
    return rdata


def log2data(data):
    data[data < 2] = 2
    data = np.log2(data)
    return data


def binarydata(data):
    data[data != 0] = 1
    return data


def normdata(data):
    data = data / np.sum(data, axis=0)
    return data


def clrdata(data):
    data[data == 0] = 1
    clrdata = np.zeros(np.shape(data))
    for ncol in range(np.shape(data)[1]):
        clrdata[:, ncol] = clr(data[:, ncol])
    return clrdata
