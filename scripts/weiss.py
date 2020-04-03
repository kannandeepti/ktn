""" This script uses the Analyze_KTN class to benchmark mean first passage time
calculations on the unbranched nearest neighbor model first considered in Weiss
1967.

Deepti Kannan 2020 """

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
from matplotlib import pyplot as plt
from scipy.interpolate import CubicSpline


textwidth_inches = 6.47699
plot_params = {'axes.edgecolor': 'black', 
                  'axes.facecolor':'white', 
                  'axes.grid': False, 
                  'axes.linewidth': 0.5, 
                  'backend': 'ps',
                  'savefig.format': 'ps',
                  'axes.titlesize': 11,
                  'axes.labelsize': 11,
                  'legend.fontsize': 9,
                  'xtick.labelsize': 9,
                  'ytick.labelsize': 9,
                  'text.usetex': True,
                  'figure.figsize': [7, 5],
                  'font.family': 'serif', 
                  'font.serif': ['Computer Modern Roman'],
                  #'mathtext.fontset': 'cm', 
                  'xtick.bottom':True,
                  'xtick.top': False,
                  'xtick.direction': 'in',
                  'xtick.major.pad': 2, 
                  'xtick.major.size': 3,
                  'xtick.major.width': 0.25,

                  'ytick.left':True, 
                  'ytick.right':False, 
                  'ytick.direction':'in',
                  'ytick.major.pad': 2,
                  'ytick.major.size': 3, 
                  'ytick.major.width': 0.25,
                  'ytick.minor.right':False, 
                  'lines.linewidth':1}
plt.rcParams.update(plot_params)
path = Path('/Users/deepti/Documents/Wales/databases/chain/metastable')

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
    K = np.zeros((nmin, nmin), dtype=np.longdouble)
    for j in range(nmin):
        vib = np.exp((fvibmin - fvibts[:,j])/2).astype(np.longdouble)
        order = hordermin/(2*np.pi*horderts[:,j])
        nrg = np.exp(-(ets[:,j] - emin)/temp).astype(np.longdouble)
        K[:, j] = exist[:, j]*vib*order*nrg

    K = K.T
    for i in range(nmin):
        K[i, i] = -np.sum(K[:,i])
    #return transpose since ts.data assumes i->j and we want i<-j
    return K

def peq(temp):
    """Return equilibrium probabilities for specified temperature."""
    zvec = np.exp(-fvibmin/2)*np.exp(-emin/temp)/hordermin
    zvec = zvec.astype(np.longdouble)
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
    mfpt = np.zeros((nmin, nmin), dtype=np.longdouble)
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

def compare_weiss_pt(i, j, I, J):
    """Plot PTAB vs. weiss[min1, min2] to compare, where min1 is a minimum in A
    and min2 is a minimum in B."""
    comm = {0:'A', 1:'I', 2:'J'}
    ktn = Analyze_KTN('/scratch/dk588/databases/chain/metastable', communities
                      = {1:[1,2,3], 2:[4,5,6,7,8], 3:[9,10,11]})
    invT = np.linspace(0.001, 5, 100)
    weissAB = np.zeros_like(invT)
    weissBA = np.zeros_like(invT)
    PTAB = np.zeros_like(invT)
    PTBA= np.zeros_like(invT)
    for k, invtemp in enumerate(invT):
        mfpt_weiss = weiss(1./invtemp)
        weissAB[k] = mfpt_weiss[i, j]
        weissBA[k] = mfpt_weiss[j, i]
        pi = peq(1./invtemp)
        commpi = ktn.get_comm_stat_probs(np.log(pi), log=False)
        pt = ktn.get_kells_cluster_passage_times(pi, commpi, mfpt_weiss)
        PTAB[k] = pt[I,J]
        PTBA[k] = pt[J,I]

    fig, ax = plt.subplots(figsize=[textwidth_inches/3, textwidth_inches/3])
    ax.plot(invT, weissAB/PTAB, label=f't({i}<-{j})/PT({comm[I]}<-{comm[J]})')
    ax.plot(invT, weissBA/PTBA, label=f't({j}<-{i})/PT({comm[J]}<-{comm[I]})')
    plt.xlabel('1/T')
    plt.legend()
    fig.tight_layout()

def plot_weiss_landscape_mfpt_benchmark():
    """Plot energy of minima and transition states as a function of the 11
    states."""
    
    stat_pts = np.zeros((2*len(emin) - 1, )) 
    stat_pts[::2] = emin
    stat_pts[1:-1:2] = tsdata[:,0]
    states = np.arange(1, 11.5, 0.5)
    xrange = np.linspace(1, 11, 1000)
    cs = CubicSpline(states, stat_pts, bc_type='clamped')
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=[textwidth_inches,
                                                 textwidth_inches/2])
    ax.plot(xrange, cs(xrange), 'k')
    ax.plot(states[0:5:2], stat_pts[0:5:2], 'ro')
    ax.plot(states[6:16:2], stat_pts[6:16:2], 'ko')
    ax.plot(states[16::2], stat_pts[16::2], 'bo')
    ax.set_xticks(np.arange(1, 12, 1))
    ax.set_xlabel('States')
    ax.set_ylabel('Energy')

    """ Plot 4 different calculations of mfpt: weiss analytical answer,
    eigendecomposition, GT, Kells inversion. """

    ktn = Analyze_KTN('/scratch/dk588/databases/chain/metastable', communities
                      = {1:[1,2,3], 2:[4,5,6,7,8], 3:[9,10,11]})
    invT = np.linspace(0.001, 10/3, 10000)
    temps = np.array([100.0, 90.0, 80.0, 70.0, 60.0, 50.0, 40.0, 30.0, 
           20.0, 10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 
           1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3])
    weissAB = np.zeros_like(invT)
    weissBA = np.zeros_like(invT)
    gtAB = np.zeros_like(temps)
    gtBA = np.zeros_like(temps)
    eigenAB = np.zeros_like(temps)
    eigenBA = np.zeros_like(temps)
    corrAB = np.zeros_like(temps)
    corrBA = np.zeros_like(temps)
    n=11
    i=1
    j=9
    for k, invtemp in enumerate(invT):
        mfpt_weiss = weiss(1./invtemp)
        weissAB[k] = mfpt_weiss[i, j]
        weissBA[k] = mfpt_weiss[j, i]

    for k, T in enumerate(temps):
        mfpt_corr = mfpt_from_correlation(T)
        corrAB[k] = mfpt_corr[i, j]
        corrBA[k] = mfpt_corr[j, i]
        K = Kmat(T)
        eigenAB[k] = -spla.solve(K[np.arange(n)!=i, :][:, np.arange(n)!=i],
                                (np.arange(n)==j)[np.arange(n)!=i]).sum()
        eigenBA[k] = -spla.solve(K[np.arange(n)!=j, :][:, np.arange(n)!=j],
                                (np.arange(n)==i)[np.arange(n)!=j]).sum()

    ax2.plot(invT, weissAB, 'k', label='Weiss')
    ax2.plot(1./temps, eigenAB, 'o', markersize=11, alpha=0.3,
             markeredgewidth=0, label='Eigendecomposition')
    ax2.plot(1./temps, corrAB, 'ro', markersize=5, markeredgewidth=0.25,
             markeredgecolor='k', label='Correlation function')
    ax2.plot(1./temps, eigenAB, 'kx', markersize=7, 
            label='GT')
    ax2.set_xlabel('1/T')
    ax2.set_ylabel('$t_{2 \leftarrow 10}$')
    ax2.set_yscale('log')
    ax2.legend()
    plt.subplots_adjust(left=0.05, right=0.99, top=0.99, bottom=0.11)

def plot_mfpt_benchmark():

    """ Plot 4 different calculations of mfpt: weiss analytical answer,
    eigendecomposition, GT, Kells inversion. """

    ktn = Analyze_KTN('/scratch/dk588/databases/chain/metastable', communities
                      = {1:[1,2,3], 2:[4,5,6,7,8], 3:[9,10,11]})
    invT = np.linspace(5, 43, 1000)
    temps = 1./np.linspace(5, 43, 30)
    weissAB = np.zeros_like(invT)
    weissBA = np.zeros_like(invT)
    #gtAB = np.zeros_like(temps)
    #gtBA = np.zeros_like(temps)
    eigenAB = np.zeros_like(temps)
    eigenBA = np.zeros_like(temps)
    corrAB = np.zeros_like(temps)
    corrBA = np.zeros_like(temps)
    n=11
    i=1
    j=9
    for k, invtemp in enumerate(invT):
        mfpt_weiss = weiss(1./invtemp)
        weissAB[k] = mfpt_weiss[i, j]
        weissBA[k] = mfpt_weiss[j, i]

    for k, T in enumerate(temps):
        try:
            mfpt_corr = mfpt_from_correlation(T)
            corrAB[k] = mfpt_corr[i, j]
            corrBA[k] = mfpt_corr[j, i]
        except:
            corrAB[k] = np.nan
            corrBA[k] = np.nan
        try:
            K = Kmat(T)
            eigenAB[k] = -spla.solve(K[np.arange(n)!=i, :][:, np.arange(n)!=i],
                                    (np.arange(n)==j)[np.arange(n)!=i]).sum()
            eigenBA[k] = -spla.solve(K[np.arange(n)!=j, :][:, np.arange(n)!=j],
                                    (np.arange(n)==i)[np.arange(n)!=j]).sum()
        except:
            eigenAB[k] = np.nan
            eigenBA[k] = np.nan

    fig, ax = plt.subplots(figsize=[textwidth_inches/2, textwidth_inches/3])
    ax.plot(invT, weissAB, 'k', label='Weiss')
    ax.plot(1./temps, eigenAB, 'o', markersize=11, alpha=0.3,
            markeredgewidth=0, label='Eigendecomposition')
    ax.plot(1./temps, corrAB, 'bo', markersize=5, markeredgewidth=0.25,
            markeredgecolor='k', label='Correlation function')
    ax.set_xlabel('1/T')
    ax.set_ylabel('$t_{2 \leftarrow 10}$')
    ax.set_yscale('log')
    ax.legend()
    plt.subplots_adjust(left=0.05, right=0.99, top=0.99, bottom=0.11)