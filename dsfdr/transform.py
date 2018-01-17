import numpy as np
import scipy.stats
from skbio.stats.composition import clr
from logging import getLogger

logger = getLogger(__name__)


# data transformation
def rankdata(data):
    logger.debug('Rank transforming data')
    rdata = np.zeros(np.shape(data))
    for crow in range(np.shape(data)[0]):
        rdata[crow, :] = scipy.stats.rankdata(data[crow, :])
    return rdata


def log2data(data):
    logger.debug('log2 transforming data')
    data[data < 2] = 2
    data = np.log2(data)
    return data


def binarydata(data):
    logger.debug('Binary transforming data')
    data[data != 0] = 1
    return data


def normdata(data):
    logger.debug('Normalizing data to constant sum per sample')
    data = data / np.sum(data, axis=0)
    return data


def clrdata(data):
    logger.debug('clr transforming data')
    data[data == 0] = 1
    clrdata = np.zeros(np.shape(data))
    for ncol in range(np.shape(data)[1]):
        clrdata[:, ncol] = clr(data[:, ncol])
    return clrdata
