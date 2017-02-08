import numpy as np
import pandas as pd
import scipy as sp
import scipy.stats

import matplotlib as mpl
import matplotlib.pyplot as plt


# simulate data with two groups
def simulatemix(numsamples=5,numdiff=100,numc=100,numd=800,noise=0,numreads=10000):
	"""
	create a simulation of 2 multinomial distributions

	input:
	numsamples : int
		number of samples in each group
	numdiff : int
		number of different bacteria between the groups
	numc : int
		number of high freq. bacteria similar between the groups
	numd : int
		number of low freq. bacteria similar between the groups
	noise : float
		the noise
	"""

	# init otus different between A and B
	# (same sum so no compositionallity effect)
	Aa = np.random.uniform(0.1, 1, int(numdiff/2))
	Ba = np.random.uniform(0.1, 1, int(numdiff/2))
	#Ab = Ba
	#Bb = Aa
	# high freq. similar
	Ac = Bc = np.random.uniform(0.1, 1, numc)
	# low freq. similar
	#Ad = Bd = np.random.uniform(0, 0.001, numd)
	numnoise = np.random.randint(1, 7, numd)

	# normalize the probabilities
	#pA = np.hstack((Aa, Ab, Ac, Ad))
	#pA = np.hstack((Aa, Ab, Ac))
	pA = np.hstack((Aa, Ac))
	pA_normed = pA / np.sum(pA)
	#pB = np.hstack((Ba, Bb, Bc, Bd))
	#pB = np.hstack((Ba, Bb, Bc))
	pB = np.hstack((Ba, Bc))
	pB_normed = pB / np.sum(pB)
	#print(pA_normed)
	# simulate data
	numbact = np.shape(pA)[0]
	A = np.zeros([numbact, numsamples])
	B = np.zeros([numbact, numsamples])
	for i in range(numsamples):
		cpA=pA_normed+(np.random.randn(numbact)*noise*np.mean(pA_normed))
		cpA[cpA<0]=0
		cpA=cpA/np.sum(cpA)
		cpB=pB_normed+(np.random.randn(numbact)*noise*np.mean(pA_normed))
		cpB[cpB<0]=0
		cpB=cpB/np.sum(cpB)
		# rA = np.random.multinomial(numreads, pA_normed, size = 1)
		# rB = np.random.multinomial(numreads, pB_normed, size = 1)
		rA = np.random.multinomial(numreads, cpA, size = 1)
		rB = np.random.multinomial(numreads, cpB, size = 1)
		A[:, i] = rA
		B[:, i] = rB
	data = np.hstack((A, B))

	D = np.zeros([numd, 2*numsamples])
	for j in range(numd):
		for cnoise in range(numnoise[j]):
			cpos=np.random.randint(2*numsamples)
			D[j,cpos]=1

	data = np.vstack((data, D))

	# labels
	x = np.array([0, 1])
	labels = np.repeat(x, numsamples)
	labels = (labels==1)

	return (data,labels)

def simulatemix2(numsamples=5,numdiff=100,numc=100,numd=800,sigma=0.1,numreads=10000):
	"""
	revised simulation code

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
	"""

	A = np.zeros([int(numdiff), 2*numsamples])
	#B = np.zeros([int(numdiff/2), 2*numsamples])
	for i in range(int(numdiff)):
		mu_H = np.random.uniform(0.1,1)
		mu_S = np.random.uniform(0.1,1) # change the value to make two groups really different
		h = np.random.normal(mu_H, sigma, numsamples)
		s = np.random.normal(mu_S, sigma, numsamples)
		# zero inflation
		h[h<0]=0
		s[s<0]=0
		A[i,:] = np.hstack((h,s))
		#B[i,:] = np.hstack((s,h))

	C = np.zeros([numc, 2*numsamples])
	for j in range(numc):	
		mu = np.random.uniform(0.1,1)
		C[j,:] = np.random.normal(mu, sigma, 2*numsamples)

	numnoise = np.random.randint(1, 7, numd)
	D = np.zeros([numd, 2*numsamples])
	for k in range(numd):
		for cnoise in range(numnoise[k]):
			cpos=np.random.randint(2*numsamples)
			D[k,cpos]=np.random.uniform(0.1,1)

	#data = np.vstack((A,B,C,D))
	data = np.vstack((A,C,D))
	data = data/np.sum(data, axis = 0) # normalize by column

	# labels
	x = np.array([0, 1])
	labels = np.repeat(x, numsamples)
	labels = (labels==1)

	return (data,labels)



# simulate data with more than 2 groups
def multisimulatemix(numsamples=5,numgroup=3, numdiff=100,numc=100,numd=800,noise=0,numreads=10000):
	"""
	create a simulation of 2 multinomial distributions

	input:
	numsamples : int
		number of samples in each group
	numdiff : int
		number of different bacteria between the groups
	numc : int
		number of high freq. bacteria similar between the groups
	numd : int
		number of low freq. bacteria similar between the groups
	noise : float
		the noise
	"""

	# init otus different between A and B
	# (same sum so no compositionallity effect)
	Aa = np.random.uniform(0.1, 1, int(numdiff/2))
	Ba = np.random.uniform(0.1, 1, int(numdiff/2))
	Ab = Ba
	Bb = Aa
	# high freq. similar
	Ac = Bc = np.random.uniform(0.1, 1, numc)
	# low freq. similar
	#Ad = Bd = np.random.uniform(0, 0.001, numd)
	numnoise = np.random.randint(1, 7, numd)

	# normalize the probabilities
	#pA = np.hstack((Aa, Ab, Ac, Ad))
	pA = np.hstack((Aa, Ab, Ac))
	pA_normed = pA / np.sum(pA)
	#pB = np.hstack((Ba, Bb, Bc, Bd))
	pB = np.hstack((Ba, Bb, Bc))
	pB_normed = pB / np.sum(pB)

	# simulate data
	numbact = np.shape(pA)[0]
	A = np.zeros([numbact, numsamples])
	B = np.zeros([numbact, numsamples])
	for i in range(numsamples):
		cpA=pA_normed+(np.random.randn(numbact)*noise)
		cpA[cpA<0]=0
		cpA=cpA/np.sum(cpA)
		cpB=pB_normed+(np.random.randn(numbact)*noise)
		cpB[cpB<0]=0
		cpB=cpB/np.sum(cpB)
		# rA = np.random.multinomial(numreads, pA_normed, size = 1)
		# rB = np.random.multinomial(numreads, pB_normed, size = 1)
		rA = np.random.multinomial(numreads, cpA, size = 1)
		rB = np.random.multinomial(numreads, cpB, size = 1)
		A[:, i] = rA
		B[:, i] = rB
	data = np.hstack((A, B))

	D = np.zeros([numd, 2*numsamples])
	for j in range(numd):
		for cnoise in range(numnoise[j]):
			cpos=np.random.randint(2*numsamples)
			D[j,cpos]=1
	data = np.vstack((data, D))

	for i in range(numgroup-2):
		data = np.hstack((data, data[:, 0:numsamples]))

	# labels 
	x = np.arange(numgroup)
	labels = np.repeat(x, numsamples)
	for j in range(numgroup):
		labels == j
	return (data,labels)

# simulate data for correlation test
def simulatecor(numa=100,numc=100,numd=800, numsamples = 10):
	"""
	create a simulation of 2 multinomial distributions

	input:
	numa : int
		number of bacterials correlated
	numb : int
		balance the compositionaly with section A
	numc : int
		number of bacterial uncorrelated
	numd : int
		number of low freq. bacteria similar between the groups
	numsamples : int
		number of samples for each bacteria
	"""

	labels = np.arange(1, int(numsamples + 1))
	A = np.zeros([numa, numsamples])
	for i in range(int(numa)):
		u = np.random.uniform(-1,1)
#		u=-1
		s = np.random.uniform(0, 100)
		#r = np.random.uniform(-2, 2) Problem: negative power produces inf
		#r = np.random.uniform(0, 2)
		r = np.random.uniform(1, 2)
#		r=1
#		s=100
		coin=np.random.randint(2)
		if coin == 0:
			a = s + u*((labels)**r)
		else:
#			a = s + u*((labels)**r)
			a = s + u*((numsamples+1 - labels)**r)
		A[i,:] = a

	T = np.sum(A, axis = 0) # column sum
	B = max(T) - T
	#print(np.shape(B))


	numc = 100 # how to choose C
	C = np.zeros([numc, numsamples])
	for i in range(numc):
		c = np.random.uniform(3000, 3600, numsamples)
		C[i, :] = c
	#print(np.shape(C))  

	numd = 100
	numnoise = np.random.randint(1, 7, numd)
	D = np.zeros([numd, numsamples])
	for j in range(numd):
		for cnoise in range(numnoise[j]):
			cpos=np.random.randint(numsamples)
			D[j,cpos]=1
	#print(np.shape(D)) 

	data = np.vstack((A, B, C, D))
	#print(np.shape(data))
	#print(data[0,:])
	return (data,labels)