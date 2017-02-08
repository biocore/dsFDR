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


# different methods to calculate test statistic
def meandiff(data, labels):
	mean0 = np.mean(data[:, labels==0], axis = 1)
	mean1 = np.mean(data[:, labels==1], axis = 1)
	tstat = abs(mean1 - mean0)
	return tstat	


def mannwhitney(data, labels):
	group0 = data[:, labels == 0]
	group1 = data[:, labels == 1]
	tstat = np.array([scipy.stats.mannwhitneyu(group0[i, :], 
		group1[i, :]).statistic for i in range(np.shape(data)[0])])
	return tstat    


# kruwallis give a column vector while others give row vector
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

def stdmeandiff(data, labels):	
	mean0 = np.mean(data[:, labels==0], axis = 1)
	mean1 = np.mean(data[:, labels==1], axis = 1)
	sd0 = np.std(data[:, labels==0], axis = 1, ddof = 1)
	sd1 = np.std(data[:, labels==1], axis = 1, ddof = 1)
	tstat = abs(mean1 - mean0)/(sd1 + sd0) 
	return tstat
		


# new fdr method
def pfdr(data,labels, method, transform=None, alpha=0.1,numperm=1000, fdrbefast=False):
	data=data.copy()

	if transform == 'rankdata':
		data = rankdata(data)
	elif transform == 'log2data':
		data = log2data(data)
	elif transform == 'binarydata':
		data = binarydata(data) 
	elif transform == 'normdata':
		data = normdata(data) 	 

	#print('permuting')
	numbact=np.shape(data)[0]

	if method == "meandiff":    
		method = meandiff   
		t=method(data,labels)
		numsamples=np.shape(data)[1]
		p=np.zeros([numsamples,numperm])
		k1=1/np.sum(labels == 0)
		k2=1/np.sum(labels == 1)
		for cperm in range(numperm):
			np.random.shuffle(labels)
			p[labels==0, cperm] = k1
		p2 = np.ones(p.shape)*k2
		p2[p>0] = 0
		mean1 = np.dot(data, p)
		mean2 = np.dot(data, p2)
		u = np.abs(mean1 - mean2)
		
	elif method == 'mannwhitney':
		method = mannwhitney
		t=method(data,labels)
		u=np.zeros([numbact,numperm])
		for cperm in range(numperm):
			rlabels=np.random.permutation(labels)
			rt=method(data,rlabels)
			u[:,cperm]=rt

	elif method == 'kruwallis':
		method = kruwallis
		t=method(data,labels)
		u=np.zeros([numbact,numperm])
		for cperm in range(numperm):
			rlabels=np.random.permutation(labels)
			rt=method(data,rlabels)
			u[:,cperm]=rt	

	elif method == 'stdmeandiff':
		method = stdmeandiff
		t=method(data,labels)
		u=np.zeros([numbact,numperm])
		for cperm in range(numperm):
			rlabels=np.random.permutation(labels)
			rt=method(data,rlabels)
			u[:,cperm]=rt


	elif method == 'spearman' or method == 'pearson':
		if method == 'spearman':
			data = rankdata(data)
			labels = sp.stats.rankdata(labels)
		meanval=np.mean(data,axis=1).reshape([data.shape[0],1])
		data=data-np.repeat(meanval,data.shape[1],axis=1)
		labels=labels-np.mean(labels)
		t=np.abs(np.dot(data, labels))
		permlabels = np.zeros([len(labels), numperm])
		for cperm in range(numperm):
			rlabels=np.random.permutation(labels)
			permlabels[:,cperm] = rlabels
		u=np.abs(np.dot(data,permlabels))	

	elif method == 'nonzerospearman' or method == 'nonzeropearson':
		t = np.zeros([numbact])
		u = np.zeros([numbact, numperm])
		for i in range(numbact):
			index = np.nonzero(data[i, :])
			label_nonzero = labels[index]
			sample_nonzero = data[i, :][index]
			if method == 'nonzerospearman':
				sample_nonzero = sp.stats.rankdata(sample_nonzero)
				label_nonzero = sp.stats.rankdata(label_nonzero)
			sample_nonzero = sample_nonzero - np.mean(sample_nonzero)
			label_nonzero = label_nonzero - np.mean(label_nonzero)
			t[i] = np.abs(np.dot(sample_nonzero, label_nonzero))

			permlabels = np.zeros([len(label_nonzero), numperm])
			for cperm in range(numperm):
				rlabels=np.random.permutation(label_nonzero)
				permlabels[:,cperm] = rlabels
			u[i, :] = np.abs(np.dot(sample_nonzero, permlabels))

	else:
		# method = userfunction 
		#t=method(data,labels)         
		u=np.zeros([numbact,numperm])
		for cperm in range(numperm):
			rlabels=np.random.permutation(labels)
			rt=method(data,rlabels)
			u[:,cperm]=rt

	#print('calculating fdr')
	#print(np.shape(u))

	#print('fixing floating point errors')
	for crow in range(numbact):
		closepos=np.isclose(t[crow],u[crow,:])
		u[crow,closepos]=t[crow]

	# calculate permutation p-vals
	for crow in range(numbact):
		cvec=np.hstack([t[crow],u[crow,:]])
		cvec=1-(sp.stats.rankdata(cvec,method='min')/len(cvec))
		t[crow]=cvec[0]
		u[crow,:]=cvec[1:]

	# calculate FDR

	# sort the p-values from big to small
	sortt=list(set(t))
	sortt=np.sort(sortt)
	sortt=sortt[::-1]

	foundit=False
	allfdr=[]
	allt=[]
	#print(sortt)
	for cp in sortt:
		realnum=np.sum(t<=cp)
		fdr=(realnum+np.count_nonzero(u<=cp)) / (realnum*(numperm+1))
		allfdr.append(fdr)
		allt.append(cp)
		# print(fdr)
		if fdr<=alpha:
			realcp=cp
			foundit=True
			break

	if not foundit:
		#print('not low enough. number of rejects : 0')
		reject=np.zeros(numbact,dtype=int)
		reject= (reject>10) # just want to output "FALSE"
		return reject

	# and fill the reject null hypothesis
	reject=np.zeros(numbact,dtype=int)
	reject= (t<=realcp)
	#print('rejected %d ' % (np.sum(t<=realcp)))
	return reject
