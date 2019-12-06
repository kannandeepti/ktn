""" Script to visualize rate constants as a function of
temperature and product/reactant states. """

from pathlib import Path
import seaborn as sns
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.legend import Legend
import seaborn as sns
sns.set()
import numpy as np

params = {'axes.edgecolor': 'black', 'axes.grid': True, 'axes.titlesize': 20.0,
          'axes.linewidth': 0.75, 'backend': 'pdf','axes.labelsize':
          18,'legend.fontsize': 18,
          'xtick.labelsize': 18,'ytick.labelsize': 18,'text.usetex':
          False,'figure.figsize': [7, 5],
          'mathtext.fontset': 'stixsans', 'savefig.format': 'pdf',
          'xtick.bottom':True, 'xtick.major.pad': 5, 'xtick.major.size': 5,
          'xtick.major.width': 0.5,
          'ytick.left':True, 'ytick.right':False, 'ytick.major.pad': 5,
          'ytick.major.size': 5, 'ytick.major.width': 0.5,
          'ytick.minor.right':False, 'lines.linewidth':2}

plt.rcParams.update(params)

def plot_kAB_Gthresh(direction='BA'):
    """plot kSSAB, kNSSAB as a function of Gthresh, and the exact kNSS
    for two different definitions of products and reactants."""
    df = pd.read_csv('csvs/rates_ABBA_1inA_1inB.csv')
    df2 = df.set_index('T')
    df3 = pd.read_csv('csvs/rates_ABBA_5inA_395inB.csv')
    df3 = df3.set_index('T')
    for temp, vals in df2.groupby('T'):
        vals2 = df3.loc[temp]
        nrgthreshs = np.array(vals['Gthresh']).astype(float)
        kSSs = np.array(vals[f'kSS{direction}']).astype(float)
        kNSSs = np.array(vals[f'kNSS{direction}']).astype(float)
        kNSSexact = np.array(vals[f'kNSSexact{direction}']).astype(float)
        fig, ax = plt.subplots()
        colors = sns.color_palette()
        plt.plot(nrgthreshs, kSSs, label=f'kSS{direction}', color=colors[0], markersize=2)
        plt.plot(nrgthreshs, vals2[f'kSS{direction}'],'--',
                 label='kSS_5inA',
                 color=colors[0])
        plt.plot(nrgthreshs, kNSSs, label=f'kNSS{direction}',
                    color=colors[1], markersize=2)
        plt.plot(nrgthreshs, vals2[f'kNSS{direction}'], '--',
                 label='kNSS_5inA',
                 color=colors[1])
        plt.plot(nrgthreshs, kNSSexact,
                    label=f'kNSSexact{direction}', color=colors[2])
        plt.plot(nrgthreshs,
                 vals2[f'kNSSexact{direction}'], '--',
                 label='kNSSexact_5inA',
                 color=colors[2])
        plt.xlabel(r'$G_{thresh}$')
        plt.ylabel(f'$k_{{direction}}$')
        plt.title(f'T={temp}')
        plt.legend()
        fig.tight_layout()
        plt.savefig(f'k{direction}_Gthresh_T{temp}.png')

def plot_kNSS_temp(direction='BA'):
    """Plot the exact kNSS as a function of temperature for different
    definitions of products and reactants."""

    df = pd.read_csv('csvs/rates_kNSS.csv')
    temps = np.unique(df['T'])
    df = df.set_index(['T','numInA','numInB'])
    kNSS1 = []
    kNSS2 = []
    for temp in temps:
        kNSS1.append(df.loc[temp,1,1][f'kNSS{direction}'])
        kNSS2.append(df.loc[temp,5,395][f'kNSS{direction}'])
    fig, ax = plt.subplots()
    plt.plot(temps, kNSS1, 'o-', label='1inA, 1inB')
    plt.plot(temps, kNSS2, 'o-', label='5inA, 395inB')
    plt.xlabel('Temperature')
    plt.ylabel(f'kNSS{direction}')
    plt.yscale('log')
    plt.legend()
    fig.tight_layout()
    plt.savefig(f'plots/kNSS{direction}_temp_diffAB.png')
    return np.array(kNSS1), np.array(kNSS2)

def plot_NGTrates(direction='AB', arrhenius=True):
    """ Plot kSS, kNSS, and the true kNGT rate constant as a function of
    temperature, varying the number of states in the B set. """
    df = pd.read_csv('csvs/rates_kNGT_ABscan.csv')
    num_ABstates = len(np.unique(df['numInB']))
    colors = sns.color_palette("GnBu_d", num_ABstates)
    j = 0
    fig, ax = plt.subplots()
    linestyles = []
    linecolors = []
    Blabels = []
    for id, vals in df.groupby(['numInA','numInB']):
        temps = 1./vals['T']
        if arrhenius is False:
            temps = vals['T']
        if j==0:
            linestyles += ax.plot(temps, vals[f'kSS{direction}'], ':',
                                  color='k')
                    #label='kSS')
            linestyles += ax.plot(temps, vals[f'kNSS{direction}'], '--',
                                  color='k')
                    #label='kNSS')
            linestyles += ax.plot(temps, vals[f'k{direction}'], '-',
                                  color='k')
                    #label=f'kNGT')
        ax.plot(temps, vals[f'kSS{direction}'], ':',color=colors[j])
                #label=None)
        ax.plot(temps, vals[f'kNSS{direction}'], '--', color=colors[j])
                #label=None)
        linecolors += ax.plot(temps, vals[f'k{direction}'], '-',
                              color=colors[j])
                #label=f'{id[1]} in B')
        Blabels.append(f'{id[1]} in B')
        j += 1
    if arrhenius is True:
        plt.xlabel(r'$1/k_BT$')
    if arrhenius is False:
        plt.xlabel(r'$k_BT$')
    plt.ylabel(f'log(k{direction})')
    plt.yscale('log')
    ax.legend(linestyles, ['kSS', 'kNSS', 'kNGT'], loc='upper right')
    leg = Legend(ax, linecolors, Blabels, loc='lower left')
    ax.add_artist(leg)
    fig.tight_layout()
    if arrhenius is True:
        plt.savefig(f'plots/kNGT{direction}_scanB_temp_arrhenius.png')
    else:
        plt.savefig(f'plots/kNGT{direction}_scanB_temp.png')

def plot_kNSS_difference():

    kNSSBA_1inA, kNSSBA_5inA = plot_kNSS_temp('BA')
    kNSSAB_1inA, kNSSAB_5inA = plot_kNSS_temp('AB')
    colors = sns.color_palette()
    fig, ax = plt.subplots()
    df = pd.read_csv('csvs/rates_kNSS.csv')
    temps = np.unique(df['T'])
    ax.plot(temps, kNSSBA_5inA - kNSSBA_1inA, 'o-', color=colors[2],
            label='BA')
    ax.plot(temps, kNSSAB_5inA - kNSSAB_1inA, 'o-', color=colors[3],
            label='AB')
    plt.legend()
    plt.xlabel('Temperature')
    plt.ylabel('kNSS_5inA - kNSS_1inA')
    #plt.yscale('log')
    fig.tight_layout()
