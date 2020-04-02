""" This script uses the Analyze_KTN class to benchmark mean first passage time
calculations on the unbranched nearest neighbor model first considered in Weiss
1967.

Deepti Kannan 2020 """

from code_wrapper import ParsedPathsample
from code_wrapper import ScanPathsample
from ktn_analysis import Analyze_KTN
import numpy as np
from numpy.linalg import inv
import scipy 
import scipy.linalg as spla 
from scipy.sparse.linalg import eigs
from scipy.sparse import csr_matrix
from scipy.linalg import eig
from scipy.linalg import expm
from pathlib import Path
import pandas as pd
import os
import subprocess


PATHSAMPLE = "/home/dk588/svn/PATHSAMPLE/build/gfortran/PATHSAMPLE"
path = Path('/scratch/dk588/databases/chain/metastable')

"""Define variables needed to calculate rate as a function of
temperature."""

mindata = np.loadtxt(path/'min.data')
tsdata = np.loadtxt(path/'ts.data')

nmin = mindata.shape[0]
emin = mindata[:,0]
fvibmin = mindata[:, 1]
hordermin = mindata[:, 2]

ets = np.zeros((nmin, nmin))
fvibts = np.zeros((nmin, nmin))
horderts = np.ones((nmin, nmin))
exist = np.zeros((nmin, nmin))

for i in range(tsdata.shape[0]):
    j1 = int(tsdata[i, 3]) - 1
    j2 = int(tsdata[i, 4]) - 1
    exist[j1, j2] = 1
    exist[j2, j1] = 1
    ets[j1, j2] = tsdata[i, 0]
    ets[j2, j1] = tsdata[i, 0]
    fvibts[j1, j2] = tsdata[i, 1]
    fvibts[j2, j1] = tsdata[i, 1]
    horderts[j1, j2] = tsdata[i, 2]
    horderts[j2, j1] = tsdata[i, 2]

def Kmat(temp):
    """Return a rate matrix, nmin x nmin for specified temperature."""
    K = np.zeros((nmin, nmin))
    for j in range(nmin):
        vib = np.exp((fvibmin - fvibts[:,j])/2)
        order = hordermin/(2*np.pi*horderts[:,j])
        nrg = np.exp(-(ets[:,j] - emin)/temp)
        K[:, j] = exist[:, j]*vib*order*nrg

    K = K.T
    for i in range(nmin):
        K[i, i] = -np.sum(K[:,i])
    #return transpose since ts.data assumes i->j and we want i<-j
    return K

def peq(temp):
    """Return equilibrium probabilities for specified temperature."""
    zvec = np.exp(-fvibmin/2)*np.exp(-emin/temp)/hordermin
    return zvec/np.sum(zvec)

def weiss(temp):
    """Return the matrix of mean first passage times using the recursive
    formulae in Weiss (1967) Adv. Chem. Phys. 13, 1-18."""
    K = Kmat(temp)
    def eta(j):
        if j == 0:
            return 0
        else:
            return (K[j, j-1]*eta(j-1) + 1)/K[j-1, j]

    def theta(j):
        if j==0:
            return 1
        else:
            return theta(j-1)*K[j, j-1]/K[j-1, j]

    etavec = [eta(j) for j in range(0, nmin-1)]
    thetavec = [theta(j) for j in range(0, nmin-1)]
    tmean_oneton = lambda n: (eta(n)/theta(n))*np.sum(thetavec[0:n]) - np.sum(etavec[0:n]) 

    def xeta(j):
        if j == nmin-1:
            return 0
        else:
            return (K[j, j+1]*xeta(j+1) + 1)/K[j+1, j]

    def xtheta(j):
        if j == nmin-1:
            return 1
        else:
            return xtheta(j+1)*K[j, j+1]/K[j+1, j]

    xetavec = [xeta(j) for j in range(1, nmin)]
    xthetavec = [xtheta(j) for j in range(1, nmin)]
    tmean_nmintoone = lambda n: (xeta(n)/xtheta(n))*np.sum(xthetavec[n:nmin-1]) - np.sum(xetavec[n:nmin-1]) 
    print(tmean_nmintoone(10))
    print(tmean_nmintoone(9))
    mfpt = np.zeros((nmin, nmin))
    for i in range(0, nmin):
        for j in range(0, i):
            mfpt[i][j] = tmean_oneton(i) - tmean_oneton(j)
        for j in range(i+1, nmin):
            mfpt[i][j] = tmean_nmintoone(i) - tmean_nmintoone(j)
    return mfpt

def mfpt_from_correlation(temp):
    """Calculate the matrix of mean first passage times using Eq. (49) of KRA
    JCP paper. """
    pi = peq(temp)
    K = Kmat(temp)
    pioneK = spla.inv(pi.reshape((nmin,1))@np.ones((1,nmin)) + K)
    zvec = np.diag(pioneK)
    mfpt = np.diag(1./pi)@(pioneK - zvec.reshape((nmin,1))@np.ones((1,nmin)))
    
    negpioneK = spla.inv(pi.reshape((nmin,1))@np.ones((1,nmin)) - K)
    zvec = np.diag(negpioneK)
    mfpt2 = np.diag(1./pi)@(zvec.reshape((nmin,1))@np.ones((1,nmin)) - negpioneK)
    return mfpt2

def compare_weiss_pt(min1, min2):
    """Plot PTAB vs. weiss[min1, min2] to compare, where min1 is a minimum in A
    and min2 is a minimum in B."""

    ktn = Analyze_KTN('/scratch/dk588/databases/chain/metastable', communities
                      = {1:[1,2,3], 2:[4,5,6,7,8], 3:[9,10,11]})
    invT = np.linspace(0.001, 5, 100)
    weissAB = np.zeros_like(invT)
    weissBA = np.zeros_like(invT)
    PTAB = np.zeros_like(invT)
    PTBA= np.zeros_like(invT)
    for j, invtemp in enumerate(invT):
        mfpt_weiss = weiss(1./invtemp)
        weissAB[j] = mfpt_weiss[min1, min2]
        weissBA[j] = mfpt_weiss[min2, min1]
        pi = peq(1./invtemp)
        commpi = ktn.get_comm_stat_probs(np.log(pi), log=False)
        pt = ktn.get_kells_cluster_passage_times(pi, commpi, mfpt_weiss)
        PTAB[j] = pt[0,2]
        PTBA[j] = pt[2,0]

    fig, ax = plt.subplots()
    ax.plot(1./invT, weissAB/PTAB, label='AB')
    ax.plot(1./invT, weissBA/PTBA, label='BA')
    plt.xlabel('1/T')
    plt.title('weiss {min1+1} in A, {min2+1} in B')    
