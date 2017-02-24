import numpy as np
import pandas as pd
import scipy as sp
import scipy.stats

import matplotlib as mpl
import matplotlib.pyplot as plt


# different methods to calculate test statistic
def meandiff(data, labels):
	mean0 = np.mean(data[:, labels==0], axis = 1)
	mean1 = np.mean(data[:, labels==1], axis = 1)
	tstat = mean1 - mean0
	return tstat	


def stdmeandiff(data, labels):	
	mean0 = np.mean(data[:, labels==0], axis = 1)
	mean1 = np.mean(data[:, labels==1], axis = 1)
	sd0 = np.std(data[:, labels==0], axis = 1, ddof = 1)
	sd1 = np.std(data[:, labels==1], axis = 1, ddof = 1)
	tstat = (mean1 - mean0)/(sd1 + sd0) 
	return tstat


def mannwhitney(data, labels):
	group0 = data[:, labels == 0]
	group1 = data[:, labels == 1]
	tstat = np.array([scipy.stats.mannwhitneyu(group0[i, :], 
		group1[i, :]).statistic for i in range(np.shape(data)[0])])
	return tstat    

def kruwallis(data, labels):  
	n = len(np.unique(labels))
	allt=[]
	for cbact in range(np.shape(data)[0]):
		group = []
		for j in range(n):
			group.append(data[cbact, labels == j])
		tstat = scipy.stats.kruskal(*group).statistic
		allt.append(tstat)
	return allt

def pearson(data, labels):
	tstat = np.array([scipy.stats.pearsonr(data[i, :], 
		labels)[0] for i in range(np.shape(data)[0])])
	return tstat  


def spearman(data, labels):
	tstat = np.array([scipy.stats.spearmanr(data[i, :], 
		labels).correlation for i in range(np.shape(data)[0])])
	return tstat  				





