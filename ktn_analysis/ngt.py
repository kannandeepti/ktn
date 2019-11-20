"""Script to perform NGT calculations on toy networks

Deepti Kannan 2019"""

import numpy as np
import scipy

### DATA STRUCTURES ###
#for comparison, will use similar data structures to David's
#mathematica notebook, where a matrix `exists` is used to
#determine if edges exist between states.
#also have a matrix of branching probabilities and
#vector of waiting times

def remove_x(x, edges, pbranch, twait, precise=False):
    """Remove a minimum x from the network.
    TODO: test against Mathematica version for toy networks
    """

    num_nodes = len(twait)
    onepxx = 1 - pbranch[x,x]
    #if we want numerical precision for 1 - P[x,x]
    if precise:
        #sum over rows, pick out xth column
        onepxx = np.sum(pbranch, axis=0)[x]

    for beta in range(num_nodes):
        #David's mathematica code has this as edges[beta, x]... why flipped?
        #I guess doesnt matter since all edges are bidirectional
        if edges[x, beta]==1 and beta!=x:
            #update waiting time -- must we do this before or after?? shouldn't matter
            twait[beta] = twait[beta] + (pbranch[x, beta]*twait[x])/onepxx
            for gamma in range(num_nodes):
                #loop through all gamma in the set adjacent to x and beta, excluding x
                if gamma!=x and (edges[gamma, beta]==1 or edges[gamma, x]==1):
                    #update branching probability
                    pbranch[gamma, beta] = pbranch[gamma, beta] + (pbranch[gamma, x]*pbranch[x, beta])/onepxx
                    #add edge if gamma connected to x 
                    #(gamma might already be connected to beta, but add an edge in case not)
                    if edges[gamma, x] == 1:
                        edges[gamma, beta] = 1
                        edges[beta, gamma] = 1
                    #remove all edges from x to gamma, and set P_{gamma, x} = 0
                    edges[gamma, x] = 0
                    edges[x, gamma] = 0
                    #mathematica doesnt do following line... is it necessary?
                    pbranch[gamma, x] = 0
        #disconnect x and beta
        edges[x, beta] = 0
        edges[beta, x] = 0
        pbranch[x, beta] = 0

