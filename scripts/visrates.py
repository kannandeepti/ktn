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
import scipy

params = {'axes.edgecolor': 'black', 'axes.facecolor':'white', 'axes.grid': False, 'axes.titlesize': 20.0,
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

def plot_kAB_Gthresh(temp, direction='AB'):
    """plot kSSAB, kNGTAB as a function of Gthresh, and the exact kNGT
    for two different definitions of products and reactants.
    
    Parameters
    ----------
    temp : float .2f
        temperature that REGROUPFREE was run at

    """
    df = pd.read_csv('csvs/rates_regroupfree_ABsize.csv')
    temps = [float(f'{temp:.2f}') for temp in df['T']]
    df['T'] = temps
    df = df.set_index(['T', 'numInA', 'numInB'])
    df = df.xs(temp)
    #plot kSSAB, kNGTAB for 5inA and 395inB vs. 1inA, 1inB
    fig, ax = plt.subplots()
    colors = sns.color_palette()
    nrgthreshs = df.loc[1,1]['Gthresh']
    plt.plot(nrgthreshs, df.loc[5,395][f'kSS{direction}'], label=f'kSS{direction}_395inB', color=colors[0])
    plt.plot(nrgthreshs, df.loc[1,1][f'kSS{direction}'], '--',color=colors[0],
              label=f'kSS{direction}_1inB')
    plt.plot(nrgthreshs, df.loc[5,395][f'k{direction}'], label=f'k{direction}_395inB',
                color=colors[1])
    plt.plot(nrgthreshs, df.loc[1,1][f'k{direction}'], '--', color=colors[1],
             label=f'k{direction}_1inB')
    plt.plot(nrgthreshs, df.loc[1,1][f'kNGTexact{direction}'],
                label=f'kNGTexact{direction}', color=colors[2])
    #plt.plot(nrgthreshs, df.loc[5,395][f'kNGTexact{direction}'],
    #         color=colors[2])
    plt.xlabel(r'$G_{thresh}$')
    plt.ylabel(f'k{direction}')
    plt.title(f'T={temp:.2f}')
    plt.legend()
    fig.tight_layout()
    plt.savefig(f'plots/kNGT{direction}_Gthresh_T{temp:.2f}_diffAB.png')

    #also plot number of products vs. Gthresh
    fig, ax = plt.subplots()
    ax.plot(nrgthreshs, df.loc[1,1]['regroupedA'], 'r--', label='1 in A')
    ax.plot(nrgthreshs, df.loc[5,395]['regroupedA'], 'r', label='5 in A')
    plt.legend()
    plt.title(f'Regrouped A, T={temp:.2f}')
    fig.tight_layout()
    plt.savefig(f'plots/regroupedA_vs_Gthresh_T{temp:.2f}.png')
    #and for reactants
    fig, ax = plt.subplots()
    ax.plot(nrgthreshs, df.loc[1,1]['regroupedB'], 'b--', label='1 in B')
    ax.plot(nrgthreshs, df.loc[5,395]['regroupedB'], 'b', label='395 in B')
    plt.legend()
    plt.title(f'Regrouped B, T={temp:.2f}')
    fig.tight_layout()
    plt.savefig(f'plots/regroupedB_vs_Gthresh_T{temp:.2f}.png')

def plot_kRG_kNGT_ratio(temps_to_plot):
    temps_to_plot = 1./temps_to_plot
    df = pd.read_csv('csvs/rates_regroupfree_ABsize.csv')
    temps = [float(f'{temp:.2f}') for temp in df['T']]
    df['T'] = temps
    df = df.set_index(['T', 'numInA', 'numInB'])
    fig, ax = plt.subplots()
    colors = sns.color_palette()
    for i, temp in enumerate(temps_to_plot):
        df2 = df.xs(temp)
        #plot kSSAB, kNGTAB for 5inA and 395inB vs. 1inA, 1inB
        nrgthreshs = df2.loc[5,395]['Gthresh']
        #ax.plot(nrgthreshs,
        #         df2.loc[5,395]['kAB']/df2.loc[5,395][f'kNGTexactAB'], 
        #         label=f'T{temp}',
        #         color=colors[i])
        ax.plot(nrgthreshs, df2.loc[5,395][f'kBA']/df2.loc[5,395][f'kNGTexactBA'],  
                 color=colors[i],
                 label=f'T{temp}')
    plt.xlabel(r'$\Delta G^{RG}$')
    plt.ylabel(r'$k^{F RG}_{BA} / k^{F}_{BA}$')
    plt.legend()
    fig.tight_layout()
    #plt.savefig(f'plots/kRG_kNGT_ratio_Gthresh.png')


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

def plot_NGT_arrhenius(thresh):
    """Compare rates from LEA vs. HS vs. NGT at a regrouping threshold."""
    df = pd.read_csv('csvs/rates_Gthresh_temp_scan_LEA_HS_NGT_Negs.csv')
    df = df.set_index('Gthresh')
    #first plot A<-B direction
    fig, ax = plt.subplots()
    colors = sns.color_palette("Set2", 4)
    df2 = df.xs(thresh)
    #first plot LEA at all temperatures (since it works at all)
    temps = 1./df2[df2['T']>=0.70]['T']
    ax.plot(temps, np.log(df2[df2['T']>=0.70][f'kAB_LEA']), '-s', label='LEA',
                            color=colors[0], linewidth=1) 
    #then only plot NGT for temperatures that are not NaN
    df2NGT = df2[-df2['kAB_NGT_Neg13'].isna()]
    print(f"NGT non-NA temps: {df2NGT['T']}")
    ax.plot(1./df2NGT['T'], np.log(df2NGT[f'kAB_NGT_Neg13']), '-o', label='MFPT',
                            color=colors[1], linewidth=1) 
    #then only plot HS for temperatures that are not NaN
    df2HS = df2[-df2['kAB_HS_Neg13'].isna()]
    print(f"HS non-NA temps: {df2HS['T']}")
    ax.plot(1./df2HS['T'], np.log(df2HS[f'kAB_HS_Neg13']), '-o', label='H-S',
                            color=colors[2], linewidth=1)
    #then only plot HSK for temperatures that are not NaN
    df2HS = df2[-df2['kAB_HSK_Neg13'].isna()]
    ax.plot(1./df2HS['T'], np.log(df2HS[f'kAB_HSK_Neg13']), '-o', label='H-S-K',
                            color=colors[3], linewidth=1)
    print(f"HSK non-NA temps: {df2HS['T']}")
    #plot exact NGT answer
    ax.plot(temps, np.log(df2[df2['T']>=0.70][f'kNGTexactAB']), '-^', color='k', linewidth=1, 
                                label=r'$k^F_{A\leftarrow B}$') 
    plt.xlabel('1/T')
    #plt.xlim([temps.min(), (1./df2HS['T']).max()])
    plt.ylabel(r'ln($k^{F\hspace{1} CG}_{A\leftarrow B}$)')
    plt.title(r'$\Delta G^{RG}$ = ' + f'{thresh}')
    plt.legend()
    fig.tight_layout()
    plt.savefig(f'plots/modelA_kLEA_kHS_kNGT_Neg13_arrhenius_Gthresh{thresh}_AB.pdf')

    #next plot B<-A direction
    fig, ax = plt.subplots()
    ax.plot(temps, np.log(df2[df2['T']>=0.70][f'kBA_LEA']), '-s', label='LEA',
                            color=colors[0], linewidth=1) 
    #then only plot NGT for temperatures that are not NaN
    df2NGT = df2[-df2['kBA_NGT_Neg13'].isna()]
    ax.plot(1./df2NGT['T'], np.log(df2NGT[f'kBA_NGT_Neg13']), '-o', label='MFPT',
                            color=colors[1], linewidth=1) 
    #then only plot HS for temperatures that are not NaN
    df2HS = df2[-df2['kBA_HS_Neg13'].isna()]
    ax.plot(1./df2HS['T'], np.log(df2HS[f'kBA_HS_Neg13']), '-o', label='H-S',
                            color=colors[2], linewidth=1)
    #then only plot HSK for temperatures that are not NaN
    df2HS = df2[-df2['kBA_HSK_Neg13'].isna()]
    ax.plot(1./df2HS['T'], np.log(df2HS[f'kBA_HSK_Neg13']), '-o', label='H-S-K',
                            color=colors[3], linewidth=1)
    #plot exact NGT answer
    ax.plot(temps, np.log(df2[df2['T']>=0.70][f'kNGTexactBA']), '-^', color='k', linewidth=1, 
                                label=r'$k^F_{B\leftarrow A}$') 
    plt.xlabel('1/T')
    #plt.xlim([temps.min(), (1./df2HS['T']).max()])
    plt.ylabel(r'ln($k^{F\hspace{1} CG}_{B\leftarrow A}$)')
    plt.title(r'$\Delta G^{RG}$ = ' + f'{thresh}')
    plt.legend()
    fig.tight_layout()
    plt.savefig(f'plots/modelA_kLEA_kHS_kNGT_Neg13_arrhenius_Gthresh{thresh}_BA.pdf')

def plot_NGT_arrhenius_Gthresh(direction='AB'):
    df = pd.read_csv('csvs/rates_Gthresh_temp_scan_LEA_HS_NGT.csv')
    nrgthreshs = np.unique(df['Gthresh'])
    df = df.set_index('Gthresh')
    fig, ax = plt.subplots()
    threshlegend = []
    threshlabels = []
    colors = sns.color_palette("Set2", len(nrgthreshs))
    for i, thresh in enumerate(nrgthreshs):
        df2 = df.xs(thresh)
        temps = 1./dfHS2['T']
        threshlabels.append(r'$\Delta G^{RG}$ = ' + f'{thresh}')
        threshlegend += ax.plot(temps, np.log(df2[f'k{direction}_HS']), '-o',
                                color=colors[i], linewidth=1) 
        ax.plot(temps, np.log(df2[f'k{direction}_HS']), '-o',
                                color=colors[i], linewidth=1) 
        if (i == len(nrgthreshs)-1):
            threshlegend += ax.plot(1./df2['T'], np.log(df2[f'kNGTexact{direction}']), '-^', color='k', linewidth=1)
            threshlabels.append(r'$k^F_{A\leftarrow B}$') 
    plt.xlabel('1/T')
    plt.ylabel(r'ln($k^{F\hspace{1} CG}_{A\leftarrow B}$)')
    ax.legend(threshlegend, threshlabels)
    fig.tight_layout()
    plt.savefig(f'plots/modelA_kHS_kNGT_arrhenius_{direction}.pdf')

def plot_NGTrates_minInB(direction='AB', arrhenius=True):
    """ Plot kSS, kNSS, and the true kNGT rate constant as a function of
    temperature, varying the number of states in the B set. """
    df = pd.read_csv('csvsk/rates_kNGT_ABscan3_postregroup.csv')
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
            linestyles += ax.plot(temps, vals[f'kNSS{direction}'], '--',
                                  color='k')
            linestyles += ax.plot(temps, vals[f'k{direction}'], '-',
                                  color='k')
        ax.plot(temps, vals[f'kSS{direction}'], ':',color=colors[j])
        ax.plot(temps, vals[f'kNSS{direction}'], '--', color=colors[j])
        legendline = ax.plot(temps, vals[f'k{direction}'], '-',
                              color=colors[j])
        if j in [0, 9, 99, 299, 394]:
            linecolors += legendline
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
        plt.savefig(f'plots/kNGT{direction}_scanB3_temp_arrhenius.png')
    else:
        plt.savefig(f'plots/kNGT{direction}_scanB3_temp.png')

def plot_NGTrates_slopes(dir='AB'):
    """Plot the slope of the arrhenius plot of rate vs. 1/kBT vs the number
    of minima in the B state."""
    df = pd.read_csv('csvs/rates_kNGT_ABscan3_postregroup.csv')
    numInB = np.unique(df['numInB'])
    print(numInB.shape)
    slopeskSS = []
    slopeskNSS = []
    slopeskNGT = []
    colors = sns.color_palette("Set2")
    for id, vals in df.groupby(['numInA','numInB']):
        temps = 1./vals['T']
        slopeskSS.append(scipy.stats.linregress(temps, vals[f'kSS{dir}'])[0])
        slopeskNSS.append(scipy.stats.linregress(temps, vals[f'kNSS{dir}'])[0])
        slopeskNGT.append(scipy.stats.linregress(temps, vals[f'k{dir}'])[0])
    fig, ax = plt.subplots()
    print(np.array(slopeskSS).shape)
    ax.plot(numInB, slopeskSS, ':', color=colors[0], label=f'kSS{dir}')
    ax.plot(numInB, slopeskNGT, '-', color=colors[2], label=f'kNGT{dir}')
    ax.plot(numInB, slopeskNSS, '--', color=colors[1], label=f'kNSS{dir}')
    plt.xlabel('Minima in B')
    plt.ylabel('slope of arrhenius plot')
    if dir=='BA':
        plt.ylim([-8.0*10**-9, -7.6*10**-9])   
    if dir=='AB':
        plt.ylim([-2.5*10**-9, 0])
    plt.legend()
    fig.tight_layout()
    plt.savefig(f'plots/kNGT{dir}_slopes_numInB2.pdf')

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
