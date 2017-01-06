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

def logdata(data):
	data[data < 2] = 2
	data = np.log2(data)
	return data

def apdata(data):
	data[data != 0] = 1
	return data

def normdata(data):
	data = data / np.sum(data, axis = 0)
	return data	


# different methods to calculate test statistic
def meandiff(data, labels):
	# normalize the data first
	
	# calculate mean difference
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
	# normalize the data first
	# calculate mean difference
	mean0 = np.mean(data[:, labels==0], axis = 1)
	mean1 = np.mean(data[:, labels==1], axis = 1)
	sd0 = np.std(data[:, labels==0], axis = 1, ddof = 1)
	sd1 = np.std(data[:, labels==1], axis = 1, ddof = 1)
	tstat = abs(mean1 - mean0)/(sd1 + sd0) 
	return tstat


def pearson(data, labels):
	tstat = np.array([scipy.stats.pearsonr(data[i, :], 
		labels)[0] for i in range(np.shape(data)[0])])
	return tstat  


def spearman(data, labels):
	tstat = np.array([scipy.stats.spearmanr(data[i, :], 
		labels).correlation for i in range(np.shape(data)[0])])
	return tstat  			


def BHfdr(data,labels,method, transform=None, alpha=0.1,numperm=1000):
	import statsmodels.sandbox.stats.multicomp

	if transform == 'rankdata':
		data = rankdata(data)
	elif transform == 'logdata':
		data = logdata(data)
	elif transform == 'apdata':
		data = apdata(data)  
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
            
	elif method == 'pearson':
		method = pearson
		t=method(data,labels)         
		u=np.zeros([numbact,numperm])
		for cperm in range(numperm):
			rlabels=np.random.permutation(labels)
			rt=method(data,rlabels)
			u[:,cperm]=rt             
 
	elif method == 'spearman':
		method = spearman
		t=method(data,labels)         
		u=np.zeros([numbact,numperm])
		for cperm in range(numperm):
			rlabels=np.random.permutation(labels)
			rt=method(data,rlabels)
			u[:,cperm]=rt 
            
	else:
		# method = userfunction 
		#t=method(data,labels)         
		u=np.zeros([numbact,numperm])
		for cperm in range(numperm):
			rlabels=np.random.permutation(labels)
			rt=method(data,rlabels)
			u[:,cperm]=rt


	#print('fixing floating point errors')
	for crow in range(numbact):
		closepos=np.isclose(t[crow],u[crow,:])
		u[crow,closepos]=t[crow]
		
	#print('calculating BH fdr')
	trep=np.tile(t[np.newaxis].transpose(),(1,numperm))
	pvals=(np.sum(u>=trep,axis=1)+1)/(numperm+1)
	# plt.figure()
	# plt.hist(pvals,50)

	reject,pvc,als,alb=statsmodels.sandbox.stats.multicomp.multipletests(pvals,alpha=alpha,method='fdr_bh')
	#print('number of rejects : %d' % np.sum(reject))
	return reject

def BHfdr2(data,labels,method, transform=None, alpha=0.1,numperm=1000):
	import statsmodels.sandbox.stats.multicomp

	if transform == 'rankdata':
		data = rankdata(data)
	elif transform == 'logdata':
		data = logdata(data)
	elif transform == 'apdata':
		data = apdata(data)  
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
            
	elif method == 'pearson':
		method = pearson
		t=method(data,labels)         
		u=np.zeros([numbact,numperm])
		for cperm in range(numperm):
			rlabels=np.random.permutation(labels)
			rt=method(data,rlabels)
			u[:,cperm]=rt             
 
	elif method == 'spearman':
		method = spearman
		t=method(data,labels)         
		u=np.zeros([numbact,numperm])
		for cperm in range(numperm):
			rlabels=np.random.permutation(labels)
			rt=method(data,rlabels)
			u[:,cperm]=rt 
            
	else:
		# method = userfunction 
		#t=method(data,labels)         
		u=np.zeros([numbact,numperm])
		for cperm in range(numperm):
			rlabels=np.random.permutation(labels)
			rt=method(data,rlabels)
			u[:,cperm]=rt


	#print('fixing floating point errors')
	for crow in range(numbact):
		closepos=np.isclose(t[crow],u[crow,:])
		u[crow,closepos]=t[crow]

	# calculate permutation p-vals
	pval = []
	for crow in range(numbact):
		cvec=1-(sp.stats.rankdata(u[crow,:],method='min')/numperm)
		pval.append(cvec)		
		
	#print('calculating BH fdr')
	trep=np.tile(t[np.newaxis].transpose(),(1,numperm))
	pvals=(np.sum(u>=trep,axis=1)+1)/(numperm+1)
	# plt.figure()
	# plt.hist(pvals,50)

	reject,pvc,als,alb=statsmodels.sandbox.stats.multicomp.multipletests(pvals,alpha=alpha,method='fdr_bh')
	#print('number of rejects : %d' % np.sum(reject))
	return reject


def BYfdr(data,labels,method, transform=None, alpha=0.1,numperm=1000):
	import statsmodels.sandbox.stats.multicomp

	if transform == 'rankdata':
		data = rankdata(data)
	elif transform == 'logdata':
		data = logdata(data)
	elif transform == 'apdata':
		data = apdata(data)  
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
            
	elif method == 'pearson':
		method = pearson
		t=method(data,labels)         
		u=np.zeros([numbact,numperm])
		for cperm in range(numperm):
			rlabels=np.random.permutation(labels)
			rt=method(data,rlabels)
			u[:,cperm]=rt             
 
	elif method == 'spearman':
		method = spearman
		t=method(data,labels)         
		u=np.zeros([numbact,numperm])
		for cperm in range(numperm):
			rlabels=np.random.permutation(labels)
			rt=method(data,rlabels)
			u[:,cperm]=rt 
            
	else:
		# method = userfunction 
		#t=method(data,labels)         
		u=np.zeros([numbact,numperm])
		for cperm in range(numperm):
			rlabels=np.random.permutation(labels)
			rt=method(data,rlabels)
			u[:,cperm]=rt

	#print('fixing floating point errors')
	for crow in range(numbact):
		closepos=np.isclose(t[crow],u[crow,:])
		u[crow,closepos]=t[crow]

	#print('calculating BY fdr')
	trep=np.tile(t[np.newaxis].transpose(),(1,numperm))
	pvals=(np.sum(u>=trep,axis=1)+1)/(numperm+1)
	# plt.figure()
	# plt.hist(pvals,50)

	reject,pvc,als,alb=statsmodels.sandbox.stats.multicomp.multipletests(pvals,alpha=alpha,method='fdr_by')
	#print('number of rejects : %d' % np.sum(reject))
	return reject

def BRfwer(data,labels,method, transform=None, alpha=0.1,numperm=1000):
	import statsmodels.sandbox.stats.multicomp

	if transform == 'rankdata':
		data = rankdata(data)
	elif transform == 'logdata':
		data = logdata(data)
	elif transform == 'apdata':
		data = apdata(data)  
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
            
	elif method == 'pearson':
		method = pearson
		t=method(data,labels)         
		u=np.zeros([numbact,numperm])
		for cperm in range(numperm):
			rlabels=np.random.permutation(labels)
			rt=method(data,rlabels)
			u[:,cperm]=rt             
 
	elif method == 'spearman':
		method = spearman
		t=method(data,labels)         
		u=np.zeros([numbact,numperm])
		for cperm in range(numperm):
			rlabels=np.random.permutation(labels)
			rt=method(data,rlabels)
			u[:,cperm]=rt 
            
	else:
		# method = userfunction 
		#t=method(data,labels)         
		u=np.zeros([numbact,numperm])
		for cperm in range(numperm):
			rlabels=np.random.permutation(labels)
			rt=method(data,rlabels)
			u[:,cperm]=rt

	#print('fixing floating point errors')
	for crow in range(numbact):
		closepos=np.isclose(t[crow],u[crow,:])
		u[crow,closepos]=t[crow]

	#print('calculating BY fdr')
	trep=np.tile(t[np.newaxis].transpose(),(1,numperm))
	pvals=(np.sum(u>=trep,axis=1)+1)/(numperm+1)
	# plt.figure()
	# plt.hist(pvals,50)

	reject,pvc,als,alb=statsmodels.sandbox.stats.multicomp.multipletests(pvals,alpha=alpha,method='bonferroni')
	#print('number of rejects : %d' % np.sum(reject))
	return reject	