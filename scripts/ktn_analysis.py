""" This module analyzes properties of coarse-grained kinetic transition networks constructed
from a pre-specified set of communities, including graph transformation, the local equilibrium
approximation, Hummer-Szabo relation, and other expressions. The module is
designed to interface with files outputted by the ParsedPathsample and
ScanPathsample classes.

Deepti Kannan 2020 """

from code_wrapper import ParsedPathsample

import numpy as np
from numpy.linalg import inv
import scipy 
import scipy.linalg as spla 
from scipy.sparse.linalg import eigs
from scipy.sparse import csr_matrix
from scipy.linalg import eig
from scipy.linalg import expm
from pathlib import Path
from matplotlib import pyplot as plt
import seaborn as sns
sns.set()
import pandas as pd
import os

PATHSAMPLE = "/home/dk588/svn/PATHSAMPLE/build/gfortran/PATHSAMPLE"
params = {'axes.edgecolor': 'black', 'axes.facecolor':'white', 
          'axes.grid': False, 'axes.titlesize': 20.0,
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

class Analyze_KTN(object):

    def __init__(self, path, communities=None,
                 commdata=None, pathsample=None):
        self.path = Path(path) #path to directory with all relevant files
        if communities is not None:
            self.communities = communities
        elif commdata is not None:
            self.communities = self.read_communities(self.path/commdata)
        else:
            raise AttributeError('Either communities or commdata must' \
                                 'be specified.')
        #for analyzing rate matrices generated from PATHSAMPLE
        if pathsample is not None:
            self.parse = pathsample

    def calc_inter_community_rates_NGT(self, C1, C2):
        """Calculate k_{C1<-C2} using NGT. Here, C1 and C2 are community IDs
        (i.e. groups identified in DUMPGROUPS file from REGROUPFREE). This
        function isolates the minima in C1 union C2 and the transition states
        that connect them and feeds this subnetwork into PATHSAMPLE, using the
        NGT keyword to calculate inter-community rates."""

        #minima to isolate
        mintoisolate = self.communities[C1] + self.communities[C2]
        #parse min.data and write a new min.data file with isolated minima
        #also keep track of the new minIDs based on line numbers in new file
        newmin = {}
        j = 1
        with open(self.path/f'min.data.{C1}.{C2}', 'w') as newmindata:
            with open(self.path/'min.data','r') as ogmindata:
                #read min.data and check if line number is in C1 U C2
                for i, line in enumerate(ogmindata, 1):
                    if i in mintoisolate:
                        #save mapping from old minIDs to new minIDs
                        newmin[i] = j
                        #NOTE: these min have new line numbers now
                        #so will have to re-number min.A,min.B,ts.data
                        newmindata.write(line)
                        j += 1
                    
        #exclude transition states in ts.data that connect minima not in C1/2
        ogtsdata = pd.read_csv(self.path/'ts.data', sep='\s+', header=None,
                               names=['nrg','fvibts','pointgroup','min1','min2','itx','ity','itz'])
        newtsdata = []
        noconnections = True #flag for whether C1 and C2 are disconnected
        for ind, row in ogtsdata.iterrows():
            min1 = int(row['min1'])
            min2 = int(row['min2'])
            if min1 in mintoisolate and min2 in mintoisolate:
                # turn off noconnections flag as soon as one TS between C1 and
                # C2 is found
                if ((min1 in self.communities[C1] and min2 in self.communities[C2]) or
                (min1 in self.communities[C2] and min2 in self.communities[C1])):
                    noconnections = False
                #copy line to new ts.data file, renumber min
                modifiedrow = pd.DataFrame(row).transpose()
                modifiedrow['min1'] = newmin[min1]
                modifiedrow['min2'] = newmin[min2]
                modifiedrow['pointgroup'] = int(modifiedrow['pointgroup'])
                newtsdata.append(modifiedrow)
        if noconnections or len(newtsdata)==0:
            #no transition states between these minima, return 0
            print(f"No transition states exist between communities {C1} and {C2}")
            return 0.0, 0.0
        newtsdata = pd.concat(newtsdata)
        #write new ts.data file
        newtsdata.to_csv(self.path/f'ts.data.{C1}.{C2}',header=False, index=False, sep=' ')
        #write new min.A/min.B files with nodes in C1 and C2 (using new
        #minIDs of course)
        numInC1 = len(self.communities[C1])
        minInC1 = []
        for min in self.communities[C1]:
            minInC1.append(newmin[min] - 1)
        numInC2 = len(self.communities[C2])
        minInC2 = []
        for j in self.communities[C2]:
            minInC2.append(newmin[j] - 1)
        self.minA = minInC1
        self.minB = minInC2
        self.numInA = numInC1
        self.numInB = numInC2
        self.write_minA_minB(self.path/f'min.A.{C1}', self.path/f'min.B.{C2}')
        #run PATHSAMPLE
        files_to_modify = [self.path/'min.A', self.path/'min.B',
                           self.path/'min.data', self.path/'ts.data']
        for f in files_to_modify:
            os.system(f'mv {f} {f}.old')
        os.system(f"cp {self.path}/min.A.{C1} {self.path}/min.A")
        os.system(f"cp {self.path}/min.B.{C2} {self.path}/min.B")
        os.system(f"cp {self.path}/min.data.{C1}.{C2} {self.path}/min.data")
        os.system(f"cp {self.path}/ts.data.{C1}.{C2} {self.path}/ts.data")
        outfile = self.path/f'out.{C1}.{C2}.T{temp}'
        os.system(f"{PATHSAMPLE} > {outfile}")
        #parse output
        self.parse_output(outfile=outfile)
        for f in files_to_modify:
            os.system(f'mv {f}.old {f}')
        #return rates k(C1<-C2), k(C2<-C1)
        return self.output['kAB'], self.output['kBA']


    def construct_coarse_rate_matrix_NGT(self):
        """ Calculate inter-community rate constants using communities defined
        by minima_groups file at specified temperature. Returns a NxN rate
        matrimatrix where N is the number of communities."""

        N = len(self.communities.keys())
        print(N)
        R = np.zeros((N,N))
        for i in range(N):
            for j in range(N):
                if i < j:
                    try:
                        Rij, Rji = self.calc_inter_community_rates_NGT(i+1, j+1)
                    except:
                        print(f'PATHSAMPLE errored out for communities {i} and {j}')
                        continue
                    R[i, j] = Rij
                    R[j, i] = Rji
        for i in range(N):
            R[i, i] = -np.sum(R[:, i])
        return R

    def construct_coarse_rate_matrix_LEA(self, temp):
        """Calculate the coarse-grained rate matrix obtained using the local
        equilibrium approximation (LEA)."""

        N = len(self.communities)
        Rlea = np.zeros((N,N))
        logpi, Kmat = self.read_ktn_info(f'T{temp:.2f}', log=True)
        pi = np.exp(logpi)
        commpi = self.get_comm_stat_probs(logpi, log=False)

        for i in range(N):
            for j in range(N):
                if i < j:
                    ci = np.array(self.communities[i+1]) - 1
                    cj = np.array(self.communities[j+1]) - 1
                    Rlea[i, j] = np.sum(Kmat[np.ix_(ci, cj)]@pi[cj]) / commpi[j]
                    Rlea[j, i] = np.sum(Kmat[np.ix_(cj, ci)]@pi[ci]) / commpi[i]
        
        for i in range(N):
            Rlea[i, i] = -np.sum(Rlea[:, i])
        return Rlea

    def construct_coarse_matrix_Hummer_Szabo(self, temp):
        """ Calculate the coarse-grained rate matrix using the Hummer-Szabo
        relation, aka eqn. (12) in Hummer & Szabo (2015) J.Phys.Chem.B."""

        N = len(self.communities)
        logpi, Kmat = self.read_ktn_info(f'T{temp:.2f}', log=True)
        pi = np.exp(logpi)
        pi = pi/np.sum(pi)
        V = len(pi)
        commpi = self.get_comm_stat_probs(logpi, log=False)
        commpi = commpi / np.sum(commpi)
        D_N = np.diag(commpi)
        D_V = np.diag(pi)

        #construct clustering matrix M from community assignments
        M = np.zeros((V, N))
        for ci in self.communities:
            col = np.zeros((V,))
            comm_idxs = np.array(self.communities[ci]) - 1
            col[comm_idxs] = 1.0
            M[:, ci-1] = col

        Pi_col = commpi.reshape((N, 1))
        pi_col = pi.reshape((V, 1))
        mat_to_invert = pi_col@np.ones((1,V)) - Kmat
        first_inverse = spla.inv(mat_to_invert)
        #check that Pi = M^T pi
        Pi_calc = M.T@pi
        for entry in np.abs(commpi - Pi_calc):
            assert(entry < 1.0E-10)

        #H-S relation
        second_inversion = spla.inv(M.T@first_inverse@D_V@M)
        R_HS = Pi_col@np.ones((1,N)) - D_N@second_inversion
        self.check_detailed_balance(np.log(commpi), R_HS)
        return R_HS

    def hummer_szabo_from_mfpt(self, temp):
        """Calculate Hummer-Szabo coarse-grained rate matrix using Eqn. (72)
        of Kells et al. (2019) paper on correlation functions and the Kemeny
        constant."""
        N = len(self.communities)
        logpi, Kmat = self.read_ktn_info(f'T{temp:.2f}', log=True)
        pi = np.exp(logpi)
        pi = pi/np.sum(pi)
        n = len(pi)
        commpi = self.get_comm_stat_probs(logpi, log=False)
        commpi = commpi / np.sum(commpi)
        D_N = np.diag(commpi)
        D_n = np.diag(pi)

        #construct clustering matrix M from community assignments
        M = np.zeros((n, N))
        for ci in self.communities:
            col = np.zeros((n,))
            comm_idxs = np.array(self.communities[ci]) - 1
            col[comm_idxs] = 1.0
            M[:, ci-1] = col

        Pi_col = commpi.reshape((N, 1))
        pi_col = pi.reshape((n, 1))
        mfpt = self.get_MFPT_from_Kmat(Kmat)
        R = Pi_col@np.ones((1,N)) - spla.inv(Pi_col@np.ones((1,N)) +
                                 M.T@D_n@mfpt@pi_col@np.ones((1,N)) -
                                 M.T@D_n@mfpt@D_n@M@np.diag(1./commpi))
        self.check_detailed_balance(np.log(commpi), R)
        return R

    def get_MFPT_from_Kmat(self, K):
        """Use Tom's linear solver method to extract MFPTs from a rate matrix
        K. Equivalent to Eqn. (14) in Swinburne & Wales (2020)."""

        n = K.shape[0]
        mfpt = np.zeros((n,n))
        for i in range(n):
            for j in range(n):
                if i==0 and j==0:
                    mfpt[i][j] = 0.
                else:
                    mfpt[i][j] = -spla.solve(K[np.arange(n)!=i, :][:, np.arange(n)!=i],
                                            (np.arange(n)==j)[np.arange(n)!=i]).sum()

        return mfpt

    def get_MFPT_between_communities(self, K, pi):
        """Use Tom's linear solver method to extract MFPTs between communities
        using a rate matrix K. Equivalent to Eqn. (14) in Swinbourne & Wales
        2020."""
       
        n = K.shape[0]
        N = len(self.communities)
        mfpt = np.zeros((N,N))
        for i in range(N):
            for j in range(N):
                if i==j:
                    mfpt[i][j] = 0.
                else:
                    ci = np.array(self.communities[i+1]) - 1
                    cj = np.array(self.communities[j+1]) - 1
                    cond = np.ones((n,), dtype=bool)
                    cond[ci] = False
                    initial_cond = np.zeros((n,))
                    #initialize probability density to local boltzman in cj
                    initial_cond[cj] = pi[cj]/pi[cj].sum()
                    mfpt[i][j] = -spla.solve(K[cond, :][:, cond],
                                            initial_cond[cond]).sum()


        return mfpt

    def check_kells_mfpt_condition(self, pi, commpi, mfpt):
        """Comppute the t_JI as defined in Eqn. 66 in Kells, Rosta,
        Annibale (2019)."""

        N = len(self.communities)
        tJI = np.zeros((N,N))
        for i in range(N):
            for j in range(N):
                ci = np.array(self.communities[i+1]) - 1
                cj = np.array(self.communities[j+1]) - 1
                tJI[j][i] = pi[cj]@ mfpt[cj,:][:,ci] @ pi[ci] / (commpi[i]*commpi[j])
                tJI[j][i] -= pi[cj] @ mfpt[cj,:][:,cj] @ pi[cj] / (commpi[j])**2

        return tJI

    def construct_coarse_rate_matrix_from_MFPTs(self, temp):
        """ Calculate a rate matrix using  MFPTs between communities."""
        """ 
        parse = ParsedPathsample(self.path/'pathdata')
        files_to_modify = [self.path/'min.A', self.path/'min.B']
        for f in files_to_modify:
            if not f.exists():
                print(f'File {f} does not exists')
                raise FileNotFoundError 
            os.system(f'mv {f} {f}.original')
        communities = self.communities
        N = len(communities)
        MFPT = np.zeros((N,N))
        for ci in range(N):
            for cj in range(N):
                if ci < cj:
                    parse.minA = np.array(communities[ci+1]) - 1
                    parse.numInA = len(communities[ci+1])
                    parse.minB = np.array(communities[cj+1]) - 1
                    parse.numInB = len(communities[cj+1])
                    parse.write_minA_minB(self.path/'min.A', self.path/'min.B')
                    parse.append_input('NGT', '0 T')
                    parse.append_input('TEMPERATURE', f'{temp}')
                    parse.write_input(self.path/'pathdata')
                    #run PATHSAMPLE
                    outfile = self.path/f'out.{ci+1}.{cj+1}.T{temp}'
                    os.system(f"{PATHSAMPLE} > {outfile}")
                    #parse output
                    output = {}
                    with open(outfile) as f:
                        for line in f:
                            if not line:
                                continue
                            words = line.split()
                            if len(words) > 1:
                                if words[0] == 'NGT>' and words[1]=='MFPT(A<-B)=':
                                    output['MFPTAB'] = float(words[2])
                                    output['MFPTBA'] = float(words[4])
                                    #this is the last line we're interested in
                                    break
                    MFPT[ci, cj] = output['MFPTAB']
                    MFPT[cj, ci] = output['MFPTBA']

        print(MFPT)
        #restore original min.A and min.B files
        for f in files_to_modify:
            os.system(f'mv {f}.original {f}')
        """
        # get R from matrix of MFPTs
        N = len(self.communities)
        logpi, Kmat = self.read_ktn_info(f'T{temp:.2f}', log=True)
        pi = np.exp(logpi)
        pi = pi/pi.sum()
        commpi = self.get_comm_stat_probs(logpi, log=False)
        D_N = np.diag(commpi)
        mfpt = self.get_MFPT_between_communities(Kmat, pi)
        
        matrix_of_ones = np.ones((N,1))@np.ones((1,N))
        #R_MFPT = inv(MFPT)@(inv(D_N) - matrix_of_ones)
        R_MFPT = spla.solve(mfpt,np.diag(1.0/commpi) - matrix_of_ones)
        self.check_detailed_balance(np.log(commpi), R_MFPT)
        return R_MFPT

    def get_free_energies_from_rates(self, R, thresh, temp, kB=1.0, planck=1.0):
        """ Estimate free energies of all super states and transition states
        between super states from an arbitrary rate matrix R. """

        N = len(self.communities)
        #subtract diagonal from R to recover only postive/zero entries
        R_nodiag = R - np.diag(np.diag(R))
        #replace any negative, near-zero entries with zero
        R_nodiag = np.where(np.abs(R_nodiag) < 1.E-13, np.zeros_like(R_nodiag), R_nodiag)
        #print(R_nodiag)
        if np.any(R_nodiag < 0.0):
            raise ValueError('The rate matrix R has negative entries.')
        if np.all(R_nodiag == 0.0):
            raise ValueError('The rate matrix R has all zero entries.')
        #TODO: first get connected set of R_nodiag (not the same as nonzero)
        #TODO: make sure A is connected to B
        #set a reference minima's p_eq to be 1
        idx, idy = np.nonzero(R_nodiag)
        commpi_renorm = {}
        commpi_renorm[idx[0]] = 1.0
        for i in range(len(idx)):
            if idy[i] not in commpi_renorm:
                dbalance = R_nodiag[idy[i], idx[i]] / R_nodiag[idx[i], idy[i]]
                if idx[i] not in commpi_renorm:
                    raise ValueError('The rate matrix R has unconnected \
                                     components.')
                commpi_renorm[idy[i]] = commpi_renorm[idx[i]]*dbalance 
        #calculate free energies for this connected set
        min_nrgs = {}
        for key in commpi_renorm:
            min_nrgs[key] = -kB*temp*np.log(commpi_renorm[key])
        if len(min_nrgs) != N:
            raise ValueError('The rate matrix R has unconnected components.')
        #create min.data file
        minfree = np.loadtxt(f'G{thresh:.1f}/min.data.regrouped.{temp:.10f}.G{thresh:.1f}')
        correction_factor = minfree[0, 1]
        with open(self.path/f'min.data.T{temp:.2f}','w') as f:
            for i in range(N):
                f.write(f'{min_nrgs[i]} {correction_factor} 1 1.0 1.0 1.0\n') 

        ts_nrgs_LJ = []
        Ls = []
        Js = []
        df = pd.DataFrame(columns=['nrg','fvibts','pointgroup','min1','min2','itx','ity','itz'])
        #loop over connected minima
        for i in range(len(idx)):
            L = idx[i]
            J = idy[i]
            if L < J:
                #for nonzeros rates, calculate free energy
                free = min_nrgs[J] - kB*temp*np.log(planck*R[L,J]/(kB*temp))
                ts_nrgs_LJ.append(free)
                Ls.append(L)
                Js.append(J)
        df = pd.DataFrame()
        df['nrg'] = ts_nrgs_LJ
        df['fvibts'] = 0.0
        df['pointgroup'] = 1
        df['min1'] = np.array(Ls)+1 #community ID is 1-indexed
        df['min2'] = np.array(Js)+1
        df['itx'] = 1.0
        df['ity'] = 1.0
        df['itz'] = 1.0
        df.to_csv(self.path/f'ts.data.T{temp:.2f}',header=False, index=False, sep=' ')

    def calc_eigenvectors(self, K, k, which_eig='SM', norm=False):
        # calculate k dominant eigenvectors and eigenvalues of sparse matrix
        # using the implictly restarted Arnoldi method
        evals, evecs = eigs(K, k, which=which_eig)
        evecs = np.transpose(evecs)
        evecs = np.array([evec for _,evec in sorted(zip(list(evals),list(evecs)),
                                 key=lambda pair: pair[0], reverse=True)],dtype=float)
        evals = np.array(sorted(list(evals),reverse=True),dtype=float)
        if norm:
            row_sums = evecs.sum(axis=1)
            evecs = evecs / row_sums[:, np.newaxis] 
        return evals, evecs
    
    def construct_transition_matrix(self, K, tau_lag):
        """ Return column-stochastic transition matrix T = expm(K*tau).
        Columns sum to 1. """
        T = expm(tau_lag*K)
        for x in np.sum(T, axis=0):
            #assert( abs(x - 1.0) < 1.0E-10) 
            print(f'Transition matrix is not column-stochastic at' \
                  f'tau={tau_lag}')
        return T

    def get_timescales(self, K, m, tau_lag):
        """ Return characteristic timescales obtained from the m dominant 
        eigenvalues of the transition matrix constructed from K at lag time
        tau_lag."""
        T = self.construct_transition_matrix(K, tau_lag)
        evals, evecs = self.calc_eigenvectors(T, m, which_eig='LM')
        char_times = np.zeros((np.shape(evals)[0]),dtype=float)
        # note that we ignore the zero eigenvalue, associated with
        # infinite time (stationary distribution)
        for i, eigval in enumerate(evals[1:]):
            char_times[i+1] = -tau_lag/np.log(eigval)
        return char_times

    def get_timescale_error(self, m, K, R):
        """ Calculate the ith timescale error for i in {1,2,...m} of a
        coarse-grained rate matrix R compared to the full matrix K.
        
        Parameters
        ----------
        m : int
            Number of dominant eigenvalues (m < N)
        K : np.ndarray[float] (V, V)
            Rate matrix for full network
        R : np.ndarray[float] (N, N)
            Coarse-grained rate matrix

        Returns
        -------
        timescale_errors : np.ndarray[float] (m-1,)
            Errors for m-1 slowest timescales
        
        """
        
        ncomms = len(self.communities)
        if m >= ncomms:
            raise ValueError('The number of dominant eigenvectors must be' \
                             'less than the number of communities.')
        Kevals, Kevecs = self.calc_eigenvectors(K, m, which_eig='SM')
        Revals, Revecs = self.calc_eigenvectors(R, m, which_eig='SM')
        #the 0th eigenvalue corresponds to infinite time
        Ktimescales = -1./Kevals[1:]
        Rtimescales = -1./Revals[1:]
        timescale_errors = np.abs(Rtimescales - Ktimescales)
        return timescale_errors

    def calculate_spectral_error(self, m, Rs, labels):
        """ Calculate spectral error, where m is the number of dominant
        eigenvalues in both the reduced and original transition networks, as a
        function of lag time. Plots the decay. """

        tau_lags = np.logspace(-4, 4, 1000)
        colors = sns.color_palette("BrBG", 5)
        colors = [colors[0], colors[-1]]
        fig, ax = plt.subplots()

        for j,R in enumerate(Rs):
            spectral_errors = np.zeros(tau_lags.shape)
            for i, tau in enumerate(tau_lags):
                T = self.construct_transition_matrix(R, tau)
                Tevals, Tevecs = self.calc_eigenvectors(T, m+1, which_eig='LM')
                # compare m+1 th eigenvalue (first fast eigenvalue) to slowest
                # eigenmodde
                spectral_errors[i] = Tevals[m]/Tevals[1]
            ax.plot(tau_lags, spectral_errors, label=labels[j], color=colors[j])

        plt.xlabel(r'$\tau$')
        plt.ylabel(r'$\eta(\tau)$')
        plt.yscale('log')
        plt.legend()
        plt.title('3h_pot, G=1.5, T=0.60')
        fig.tight_layout()


    def calculate_eigenfunction_error(self, m, K, R):
        """ Calculate the ith eigenvector approximation error for i in {1, 2,
        ... m} of a coarse-grained rate matrix R by comparing its eigenvector 
        to the correspcorresponding eigenvector of the full matrix.
        TODO: test on small system for which we are confident in eigenvectors.
        """
        
        ncomms = len(self.communities)
        if m >= ncomms:
            raise ValueError('The number of dominant eigenvectors must be' \
                             'less than the number of communities.')
        Kevals, Kevecs = self.calc_eigenvectors(K, m, which_eig='SM')
        Revals, Revecs = self.calc_eigenvectors(R, m, which_eig='SM')
        errors = np.zeros((m,))
        for i in range(0, m):
            print(i)
            for ck in self.communities:
                #minima in community ck
                minima = np.array(self.communities[ck]) - 1
                coarse_evec = np.tile(Revecs[i, ck-1], len(minima)) #scalar
                errors[i] += np.linalg.norm(coarse_evec - Kevecs[i, minima])
        return errors


    def read_ktn_info(self, suffix, log=False):
        #read in Daniel's files stat_prob.dat and ts_weights.dat

        logpi = np.loadtxt(self.path/f'stat_prob_{suffix}.dat')
        pi = np.exp(logpi)
        nnodes = len(pi)
        assert(abs(1.0 - np.sum(pi)) < 1.E-10)
        logk = np.loadtxt(self.path/f'ts_weights_{suffix}.dat', 'float')
        k = np.exp(logk)
        tsconns = np.loadtxt(self.path/f'ts_conns_{suffix}.dat', 'int')
        Kmat = np.zeros((nnodes, nnodes))
        for i in range(tsconns.shape[0]):
            Kmat[tsconns[i,1]-1, tsconns[i,0]-1] = k[2*i]
            Kmat[tsconns[i,0]-1, tsconns[i,1]-1] = k[2*i+1]
        #set diagonals
        for i in range(nnodes):
            Kmat[i, i] = 0.0
            Kmat[i, i] = -np.sum(Kmat[:, i])
        #identify isolated minima (where row of Kmat = 0)
        #TODO: identify unconnected minima
        Kmat_nonzero_rows = np.where(~np.all(Kmat==0, axis=1))
        Kmat_connected = Kmat[np.ix_(Kmat_nonzero_rows[0],
                                     Kmat_nonzero_rows[0])]
        pi = pi[Kmat_nonzero_rows]/np.sum(pi[Kmat_nonzero_rows])
        logpi = np.log(pi)
        assert(len(pi) == Kmat_connected.shape[0])
        assert(Kmat_connected.shape[0] == Kmat_connected.shape[1])

        if log:
            return logpi, Kmat_connected 
        else:
            return pi, Kmat_connected

    def get_comm_stat_probs(self, logpi, log=True):
        """ Calculate the community stationary probabilities by summing over
        thethe stationary probabilities of the nodes in each community.
        
        Parameters
        ----------
        pi : list (nnodes,)
            stationary probabilities of all minima in full network

        Returns
        -------
        commpi : list (ncomms,)
            stationary probabilities of communities in coarse coarse_network

        TODO: account for unconnected minima (set their occupation probs to 0)
        """

        pi = np.exp(logpi)
        if (np.sum(pi) - 1.0) > 1.E-10:
            pi = pi/np.sum(pi)
            logpi = np.log(pi)
        ncomms = len(self.communities)
        logcommpi = np.zeros((ncomms,))
        for ci in self.communities:
            #zero-indexed list of minima in community ci
            nodelist = np.array(self.communities[ci]) - 1 
            logcommpi[ci-1] = -np.inf
            for node in nodelist:
                logcommpi[ci-1] = np.log(np.exp(logcommpi[ci-1]) + np.exp(logpi[node]))
        commpi = np.exp(logcommpi)
        assert abs(np.sum(commpi) - 1.0) < 1.E-10
        if log:
            return logcommpi
        else:
            return commpi

    def check_detailed_balance(self, logpi, K):
        """ Check if network satisfies detailed balance condition, which is
        thatthat :math:`k_{ij} \pi_j = k_{ji} \pi_i` for all :math:`i,j`.

        Parameters
        ----------
        logpi : list (nnodes,) or (ncomms,)
            log stationary probabilities
        K : np.ndarray (nnodes, nnodes) or (ncomms, ncomms)
            inter-minima rate constants in matrix form

        """
        for i in range(K.shape[0]):
            for j in range(K.shape[1]):
                if i < j and K[i,j] > 0.0:
                    left = np.log(K[i,j]) + logpi[j]
                    right = np.log(K[j,i]) + logpi[i]
                    diff = abs(left - right)
                    if (diff > 1.E-10):
                        print(f'Detailed balance not satisfied for i={i}, j={j}')

    @staticmethod
    def read_communities(commdat):
        """Read in a single column file called communities.dat where each line
        is the community ID (zero-indexed) of the minima given by the line
        numbenumber.
        
        Parameters
        ----------
        commdat : .dat file
            single-column file containing community IDs of each minimum

        Returns
        -------
        communities : dict
            mapping from community ID (1-indexed) to minima ID (1-indexed)
        """

        communities = {}
        with open(commdat, 'r') as f:
            for minID, line in enumerate(f, 1):
                groupID =  int(line) + 1
                if groupID in communities:
                    communities[groupID].append(minID)
                else:
                    communities[groupID] = [minID]
        return communities

"""Functions that use the Analyze_KTN class to perform useful tasks."""

def compare_HS_LEA():
    """ Calculate coarse-grained rate matrices using the Hummer-Szabo, NGT, and LEA
    methods and compute kAB/kBA using NGT to be compared to the rates on the
    fulfull network. """

    temps = [0.7, 0.8, 0.9, 1.0, 10.,
             20., 30., 40., 50., 60., 70., 80., 90., 100.]
    nrgthreshs = [1, 5, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
    dfs = []
    for temp in temps:
        for thresh in nrgthreshs:
            df = pd.DataFrame()
            df['T'] = [temp]
            df['Gthresh'] = [thresh]
            ktn = Analyze_KTN('/scratch/dk588/databases/modelA',
                            commdata=f'communities_G{thresh:.2f}_T{temp:.2f}.dat')
            labels = []
            matrices = []
            try:
                Rhs_mfpt = ktn.hummer_szabo_from_mfpt(temp)
                matrices.append(Rhs_mfpt)
                labels.append('HSK')
            except Exception as e:
                print(f'Hummer Szabo Kells had the following error: {e}')

            try:
                Rhs = ktn.construct_coarse_matrix_Hummer_Szabo(temp)
                matrices.append(Rhs)
                labels.append('HS')
            except Exception as e:
                print(f'Hummer Szabo had the following error: {e}')


            try:
                Rngt = ktn.construct_coarse_rate_matrix_from_MFPTs(temp)
                matrices.append(Rngt)
                labels.append('NGT')
            except Exception as e:
                print(f'NGT had the following error: {e}')
            
            if len(matrices)==0:
                continue

            for i, R in enumerate(matrices):
                try:
                    ktn.get_free_energies_from_rates(R, thresh, temp)
                except ValueError as e:
                    print(f'{labels[i]}: {e} at T={temp}, G={thresh}')
                    continue
                #run NGT on regrouped minima using HS rates instead of LEA
                og_input_files = [ktn.path/'min.A', ktn.path/'min.B',
                            ktn.path/'min.data', ktn.path/'ts.data']
                new_input_files = [ktn.path/f'G{thresh:.1f}/min.A.regrouped.{temp:.10f}.G{thresh:.1f}',
                                ktn.path/f'G{thresh:.1f}/min.B.regrouped.{temp:.10f}.G{thresh:.1f}',
                                ktn.path/f'min.data.T{temp:.2f}',
                                ktn.path/f'ts.data.T{temp:.2f}']
                for j, f in enumerate(og_input_files):
                    os.system(f'mv {f} {f}.original')
                    os.system(f'mv {new_input_files[j]} {f}')
                #run PATHSAMPLE
                parse = ParsedPathsample(ktn.path/'pathdata')
                parse.append_input('TEMPERATURE', temp)
                parse.comment_input('REGROUPFREE')
                parse.comment_input('DUMPGROUPS')
                parse.append_input('NGT', '0 T')
                parse.write_input(ktn.path/'pathdata')
                outfile = parse.path/f'out.{thresh:.1f}.T{temp:.1f}'
                os.system(f"{PATHSAMPLE} > {outfile}")
                #parse output
                parse.parse_output(outfile=outfile)
                for j, f in enumerate(og_input_files):
                    os.system(f'mv {f} {new_input_files[j]}')
                    os.system(f'mv {f}.original {f}')
                #return rates k(C1<-C2), k(C2<-C1)
                df[f'kAB_{labels[i]}'] = [parse.output['kAB']]
                df[f'kBA_{labels[i]}'] = [parse.output['kBA']]
            dfs.append(df)
    bigdf = pd.concat(dfs, ignore_index=True)
    bigdf.to_csv('csvs/rates_HS_NGT_modelA_replaceNeg13.csv')

def compute_HS_matrices(temp):

    ktn = Analyze_KTN('/scratch/dk588/databases/modelA',
                      commdata='communities_G1.00_T{temp:.2f}.dat')

    logpi, Kmat = ktn.read_ktn_info(f'T{temp:.2f}')
    Rhs = ktn.construct_coarse_matrix_Hummer_Szabo(temp)
    Rhs_mfpt = ktn.hummer_szab_from_mfpt(temp)
    return
