r"""Script to analyze the dynamical patterns of flow in KTNs.

Code adapted from Harush & Barzel, Nat. Comm. (2017)."""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.linalg import expm
from scipy.linalg import norm
from scipy.sparse.linalg import eigs
from scipy.linalg import eig
from scipy.linalg import expm
import seaborn as sns

def linear_master_eqn(t, y, K):
    """Linear master equation based on rate matrix K_ij.
    Here, y is the vector of probabilities, i.e. y[i] = p_i(t)."""
    if np.sum(K, axis=0) != np.zeros(K.shape[0]):
        raise ValueError("The column sums of K must all be zero")
    return K@y

def linear_master_eqn_ssnodes(t, y, K, ssnodes):
    """For ssnodes, set rows in K matrix to 0 in order
    to set steady state conditions dy[n]/dt = 0."""
    for i in ssnodes:
        K[i,:] = 0
    return K@y

def get_adjacency_from_rate_matrix(K):
    """A_ij is just K_ij with the diagonals set to 0."""
    A = K
    for i in range(A.shape[0]):
        A[i, i] = 0.0
    return A

def construct_toy_network(case=0):
    K = np.zeros((4,4))

    if case==0 or case==1:
        K[1,0] = 0.10
        K[3,1] = 0.10
        #higher probability of taking the 1->3->4 path
        K[2,0] = 0.20
        K[3,2] = 0.20 
        #reverse pathways a bit slower
        K[0,1] = 0.05
        K[1,3] = 0.05
        K[0,2] = 0.10
        K[2,3] = 0.10

        if case==1:
            #same as above but add edge 2->3
            K[2,1] = 0.05
            K[1,2] = 0.025

    #set diagonals
    for i in range(4):
        K[i,i] = -np.sum(K[:,i])

    #print(K)
    return K

def calc_Gmn_runge_kutta(pi, K, n=0, alpha=-0.10):
    """ Calculate linear response matrix G_mn in response to
    perturubing source node n.
    For now, assume there's only a single source node. This means
    G_mn is a column vector. In future, G_mn will be a matrix 
    where we have multiple source nodes (n=0.....N-1). (which means
    we solve the system of ODE's N times to obtain each of the N
    column vectors of G_mn)
    """

    num_nodes = K.shape[0] #total number of nodes in system
    #initial conditions: start with system in steady state
    y0 = pi
    #perturb source node n by alpha
    y0[n] = pi[n]*(1+alpha)

    #solve ODE's
    def ode_to_solve(t,y): 
        linear_master_eqn_ssnodes(t, y, K, [n])

    T = 0 #t0 = 0
    L = 3 #amount of time in between checking termination condition
    epsilon = 1.0*10**-11 #how close we need to be to SS to terminate
    error=5.0*10**-18 #error for one step
    not_converged = True

    while not_converged:
        #solve ODE
        sol = solve_ivp(ode_to_solve, t_span=(T, T+L), y0=y0, method='RK45',
                        vectorized=True, rtol=error, atol=error*np.ones(num_nodes))

        #termination condition in (129)
        if max(abs((sol.y[:,-1] - sol.y[:,-2])/(sol.y[:,-1]*(sol.t[-1] - sol.t[-2])))) < epsilon:
            not_converged = False

        y0 = sol.y[:,-1]
        T = sol.t[-1]

    y = y0 #the final solution post convergence, i.e. new steady state

    Gmn = np.zeros(num_nodes,)
    for m in range(num_nodes):
        Gmn[m] = np.abs(((pi[m] - y[m])/pi[m])/((pi[n] - y[n])/pi[n]))

    return Gmn

def calc_Gmn_matrix_exponential(pi, K, ssnodes=[0], inodes=[], alpha=-0.10, 
                    timestep=0.10, epsilon=1.0*10**-6, **kwargs):    
    """Calculate a column of the linear response matrix Gmn by perturbing source
    node away from steady state and solving for the resultant new
    steady state."""

    num_nodes = K.shape[0] #total number of nodes in system
    y0 = np.zeros((num_nodes,))

    #initial condition: start at steady state 
    for i in range(pi.size):
       y0[i] = pi[i]
    # perturb source node n by alpha and set steady state conditions
    for n in ssnodes:
       y0[n] = pi[n]*(1+alpha)
       K[n,:] = 0 
    #hold intermediate node i at steady state
    for i in inodes:
        K[i, :] = 0

    #print(y0)
    #print(K)
    #re-normalize initial occupation probabilities
    #y0 = y0/(np.sum(y0))
    #assert(np.sum(y0)-1. < 1.0E-8)
    #simulate markov trajectory until termination condition is reached

    t=0.0
    ts = []
    ts.append(t)
    sol = []
    sol.append(y0)
    not_converged = True
    #epsilon = 1.0*10**-11 #how close we need to be to SS to terminate
    while not_converged:
        t = t+timestep
        T = expm(K*t)
        #Note: columns of T should sum to 1 b/c transition probabilities out of i (Tji) should sum to unity
        #however, they won't sum to unity in this case b/c one node held at SS 
        yt = T@y0
        #for i in range(T.shape[1]):
        #    assert abs(np.sum(T[:,i])-1.) < 1.0E-10
        #assert(np.sum(yt)-1. < 1.0E-8)
        sol.append(yt)
        ts.append(t)
        criteria = max(abs((sol[-1] - sol[-2])/(sol[-1]*(ts[-1] - ts[-2]))))
        if criteria < epsilon:
            not_converged = False 

    ts = np.array(ts)
    sol = np.array(sol).T

    y = sol[:,-1]
    #print(np.sum(y))
    Gmn = np.zeros(num_nodes,)
    for m in range(num_nodes):
        Gmn[m] = np.abs(((pi[m] - y[m])/pi[m])/((pi[n] - y[n])/pi[n]))

    return ts, sol, Gmn

def calc_Gmni_silence_i(pi, K, i, mode=0, plot=False, **kwargs):
    """Calculate the nth column of G_mn^{i}, i.e. the response matrix
    under the silencing of intermediate node i."""

    num_nodes = K.shape[0]
    Gmni_matrix = np.zeros_like(K)
    for n in range(num_nodes):
        K = construct_toy_network(mode)
        ts, sol, Gmni = calc_Gmn_matrix_exponential(pi, K, ssnodes=[n], 
                                                    inodes=[i], **kwargs)
        if plot:
            plot_time_course(ts, sol, pi)
        Gmni_matrix[:,n] = Gmni

    return Gmni_matrix

def calc_linear_response_matrix(pi, K, mode=0, plot=False, **kwargs):
    """ Solve the linear master equation with each source node n held
    at steady state, one at a time, in order to obtain the full
    linear response matrix Gmn."""

    num_nodes = K.shape[0]
    Gmn_matrix = np.zeros_like(K)
    for n in range(num_nodes):
        K = construct_toy_network(mode) #which toy matrix to perform calc on
        ts, sol, Gmn = calc_Gmn_matrix_exponential(pi, K, ssnodes=[n], **kwargs)
        if plot:
            plot_time_course(ts, sol, pi)
        Gmn_matrix[:,n] = Gmn

    return Gmn_matrix

def calc_Gmn_silence_ij(pi, K, i, j, alpha=-0.10, mode=0,
                    timestep=0.10, epsilon=1.0*10**-6, **kwargs):    
    """Calculate a column of the linear response matrix Gmn by perturbing source
    node away from steady state and solving for the resultant new
    steady state."""

    num_nodes = K.shape[0] #total number of nodes in system
    y0 = np.zeros((num_nodes,))

    #initial condition: start at steady state 
    for i in range(pi.size):
       y0[i] = pi[i]

    #calculate each column of G_mn^{ij}
    for n in range(num_nodes):
        K = construct_toy_network(mode)
        # perturb source node n by alpha and set steady state conditions
        y0[n] = pi[n]*(1+alpha)
        K[n,:] = 0 
        #freeze edge from j to i, i.e. set Kij=0 in matrix, 
        # add term to ODE that's all 0's except the jth term, which is K_ij*pi_j
        b = np.zeros((num_nodes,))
        b[j] = K[i,j]*pi[j]
        K[i,j] = 0 #j->i edge won't affect dynamics

        #print(y0)
        print(K)

        #solve inhomogenous system dy/dt = Ky + b
        #solution is homogenous + particular

        #simulate markov trajectory until termination condition is reached
        t=0.0
        ts = []
        ts.append(t)
        sol = []
        sol.append(y0)
        not_converged = True
        #epsilon = 1.0*10**-11 #how close we need to be to SS to terminate
        while not_converged:
            t = t+timestep
            T = expm(K*t) + 
            #Note: columns of T should sum to 1 b/c transition probabilities out of i (Tji) should sum to unity
            #however, they won't sum to unity in this case b/c one nold helt at SS 
            yt = T@y0
            #for i in range(T.shape[1]):
            #    assert abs(np.sum(T[:,i])-1.) < 1.0E-10
            #assert(np.sum(yt)-1. < 1.0E-8)
            sol.append(yt)
            ts.append(t)
            criteria = max(abs((sol[-1] - sol[-2])/(sol[-1]*(ts[-1] - ts[-2]))))
            if criteria < epsilon:
                not_converged = False 

        ts = np.array(ts)
        sol = np.array(sol).T

        y = sol[:,-1]
        #print(np.sum(y))
        Gmn = np.zeros(num_nodes,)
        for m in range(num_nodes):
            Gmn[m] = np.abs(((pi[m] - y[m])/pi[m])/((pi[n] - y[n])/pi[n]))


def calc_flow_through_node_i(pi, K, i, Gmn=None, Gmni=None, mode=1, **kwargs):
    """TODO: vectorize"""
    num_nodes = K.shape[0]
    if Gmn is None:
        Gmn = calc_linear_response_matrix(pi, K, mode=mode)
    if Gmni is None:
        Gmni = calc_Gmni_silence_i(pi, K, i, mode=mode)
    Gmn_sums = np.sum(Gmn, axis=0) #sum over columns, one value per n

    Fmni = np.zeros_like(Gmn)
    Fmni_sums = np.zeros((num_nodes,)) #sum over columns to get F_n^{i}
    for n in range(num_nodes):
        Fmni[:,n] = (Gmn[:,n] - Gmni[:,n])/Gmn_sums[n]
        Fmni_sums[n] = np.sum(Fmni[:,n]) - Fmni[i, n] #don't includ m=i

    #average over N-1 source nodes
    average_Fi = (np.sum(Fmni_sums) - Fmni_sums[i])/(num_nodes - 1)
    return Fmni_sums, average_Fi

def calc_flow_through_edge_ij(pi, K, i, j, Gmn=None, Gmnij=None, mode=0, **kwargs):



def simulate_markov_trajectory(K, tspan=None, y0=None, seed=None):
    """Given an initial condition y0, and rate matrix K, simulate
    trajectories for each of the nodes in the network over a time
    specified by tspan."""
    num_nodes = K.shape[0] #total number of nodes in system

    #initial conditions: start with random occupation probabilities
    if y0 is None:
        if seed is not None:
            seed = 19
            np.random.seed(seed) 
        y0 = np.random.rand(num_nodes,)
        y0 = y0/(np.sum(y0)) #normalize
    assert (np.sum(y0)-1. < 1.0E-8)

    print(y0)
    print(K)

    if tspan is None:
        tspan = np.linspace(0.0, 10, 1000)

    sol = np.zeros((num_nodes, tspan.size))
    #calculate transition matrix via matrix exponential
    for i, t in enumerate(tspan):
        T = expm(K*t)
        sol[:,i] = T@y0
        assert (np.sum(sol[:,i])-1. < 1.0E-8)

    return tspan, sol

def plot_time_course(tspan, sol, pi):
    """ Plot occupation probabilities for each node over time. """

    fig, ax = plt.subplots()
    colors = sns.color_palette("bright", 4)
    for i in range(sol.shape[0]):
        ax.plot(tspan, sol[i, :], '-o', color=colors[i], markersize=1, label=f'p_{i}(t)')
        stationary_distribution = np.tile(pi[i], tspan.size)
        ax.plot(tspan, stationary_distribution, '--', color=colors[i])

    plt.xlabel('Time')
    plt.ylabel('Occupation Probability')
    plt.legend()













