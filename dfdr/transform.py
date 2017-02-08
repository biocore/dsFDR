import numpy as np
import pandas as pd
import scipy as sp
import scipy.stats

import matplotlib as mpl
import matplotlib.pyplot as plt

# data transformation
def rankdata(data):
	rdata=np.zeros(np.shape(data))
	for crow in range(np.shape(data)[0]):
		rdata[crow,:]=sp.stats.rankdata(data[crow,:])
	return rdata

def log2data(data):
	data[data < 2] = 2
	data = np.log2(data)
	return data

def binarydata(data):
	data[data != 0] = 1
	return data

def normdata(data):
	data = data / np.sum(data, axis = 0)
	return data	
	
