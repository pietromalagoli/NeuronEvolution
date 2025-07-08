# here I implement the model with a fixed expression for the transfer function (as a Boltzmann function, see Obsidian)

import numpy as np
from numpy import random as nrd
import graph_tool as gt
from graph_tool.all import *
import copy
from typing import NamedTuple
import matplotlib.pyplot as plt
import random

##### NETWORK DEFINITION #####

class NetworkParams(NamedTuple):
    """NamedTuple class for the parameters of the network."""
    J: np.ndarray          # weight matrix
    C: np.ndarray          # connectivity matrix
    B: np.ndarray          # bias matrix
    G: Graph                # Graph of the network (Graph_tool)
    T: np.ndarray           # array of parameters g of the transfer function of each cell
    N: int = 0              # side lenght of the squared lattice (number of cells = N)
    loss: float = -1.     # loss value of the network
    
# ORA DEFINISCO LE VARIE FUNZIONI PER OPERARE SUI NETWORK
# - GENERATE(PAR,NETPAR): CREA (IN REALTà NON CREA L'OGGETTO) UN NETWORK RANDOM DATI I PARAMETRI 
#   PAR E RITORNA I PARAMETRI DEL NETWORK (NETPAR) AGGIORNATI
# - COMPUTE_loss(PAR,NETPAR): CALCOLA LA loss DEL NETWORK ATTRAVERSO IL METODO FF() E
#   RITORNA TALE loss.
# - FF(INPUT(S),NETPAR): FA IL 'FEED-FORWARD' DEL NETWORK DATO UN SET DI INPUT E RITORNA 
#   L'OUTPUT 
    
def generate(par:dict,verb:bool=False) -> NetworkParams:  # JAXXED!
    """Generate a random network given the parameters par by returning a 
        NetPar object.

    Args:
        par (dict): simulation/generation parameters.

    Returns:
        NetworkParams: returns the new network's parameters set. 
    """
    N = par['L']**2                                     # fully-populated squared lattice, number of cells = side**2
    C = nrd.choice(np.arange(2),(N,N),p=(1-par['C_density'],par['C_density'])) * (1-np.eye(N))      # initialize with a given link density and no self links                                  
    J = nrd.normal(0,0.25,(N,N)) * C                 
    J = np.clip(J, min(par['J_range']), max(par['J_range']))   # Check the boundary conditions for J       
    B = nrd.normal(0,0.25,size=(N))                 # extract from a normal distribution centered in 0 with st. dv. = 0.25 the biases 
    G = Graph(np.column_stack(np.nonzero(C)))         # Generate the Graph given the connectivity matrix
    T = np.zeros(N)                                 # the transfer function is initialized as a constant 
    NetPar = NetworkParams(J,C,B,G,T,N)              # I don't specify the loss argument so it stays as default (-1)
    loss,_,_ = compute_loss(NetPar,par,verb=verb)                      # Compute the loss value of the network 
    return NetworkParams(J,C,B,G,T,N,loss)           # return the set of updated parameters
    
def compute_loss(NetPar:NetworkParams,par:dict,verb:bool=False) -> tuple[float,float,float]:    # JAXXED!
    """Function for computing the loss of a given network.

    Args:
        NetPar (NetworkParams): parameter of the network in question.
        par (dict): simulation parameters.
        verb (bool, optional): verbosity. Defaults to False.

    Returns:
        tuple[float,float,float]: overall loss, target distance loss and link cost. 
    """
    loss = 0.                # I need this check, bc otherwise I risk adding loss over loss 
    target_dists = 0.
    weights = 0.
    for i,input in enumerate(par['input_set']):
        output = ff(input,NetPar,par)
        if verb:
            print(f'Input: {input} -> Output: {output}')
        norm_factor = NetPar.G.num_vertices()**2-NetPar.G.num_vertices()            # normalize by N(N-1)
        target_dist = (par['target_set'][i] - output)**2                            # square distance between network output and target (theoretical) output
        idx = np.arange(NetPar.N)
        row_idx = idx // par['L']
        col_idx = idx % par['L']
        row_diff = row_idx[:, None] - row_idx[None, :]
        col_diff = col_idx[:, None] - col_idx[None, :]
        dist_matrix = np.sqrt(row_diff**2 + col_diff**2)   
        weight_cost = np.sum((np.abs(NetPar.J) * dist_matrix))/norm_factor                # average weights cost (combination of strenght of link and length of link)    
        if verb:
            print(f'Target distance:{target_dist}')
            print(f'Weight cost:{weight_cost}')
        target_dists += target_dist
        weights += weight_cost 
    target_dists /= len(par['input_set'])    
    weights /= len(par['input_set'])    
    loss = target_dist + weights
    return loss, target_dists, weights
    
def ff(input:list,NetPar:NetworkParams,par:dict,verb:int=0) -> float:     # JAXXED!
    """Function to execute the 'feed-forward' computation on a given network, given inputs.

    Args:
        input (list): inputs to the network.
        NetPar (NetworkParams): network in question.
        par (dict): simulation parameters.
        verb (int, optional): verbosity. Defaults to 0.

    Returns:
        float: output of the 'feed-forward'.
    """
    state = np.zeros(NetPar.N)        # Initialize each cell's state to zero.
    '''
    state[0] = input[0]                        # cell in the upper left corner of the 2D lattice
    state[par['L']] = input[1]    # cell in the upper right corner of the 2D lattice
    # Precompute the contributions for each cell to optimize the loop (this is incredibly faster)
    contributions = np.dot(NetPar.C * NetPar.J, state) + np.sum(NetPar.B, axis=1)      # weight x value + bias - contributions così ha shape = state.shape
    # Retrive the T function of each cell by doing ifft and then fit/interpolation
    f_n = np.fft.irfft(NetPar.T,axis=1)      # inverse FT for moving back to coordinates space
    for cell in range(1,f_n.shape[0]):                  # from 1 (included) so to skip the update of the 1st input cell
        if cell == (NetPar.N - 1) * NetPar.N-1:
            continue                                    # skip the update of the 2nd input cell
        coeff = np.polyfit(par['x_n'],f_n[cell,:],3)    # fit the series of function value with a polynomial of degree up to 2
        new_fun = np.poly1d(coeff)                      # generate the new functional from the fit results
        state[cell] = new_fun(contributions[cell])    # apply the new functional as the T function for the associated cell
    # Retrive the T function of each cell by doing ifft and then fit/interpolation
    for t in range(par['ff_iter']):
        for cell in range(1,NetPar.N):                  # from 1 (included) so to skip the update of the 1st input cell
            contributions = np.dot(NetPar.C[cell,:] * NetPar.J[cell,:],state) + NetPar.B[cell]
            if cell == par['L']:
                continue                                    # skip the update of the 2nd input cell
            transfer = lambda x: par['gamma'] / (1+np.exp(-NetPar.T[cell]*x)) - par['gamma']/2      # define the transfer function given the gain parameter
            state = transfer(contributions)                             # apply the transfer function
            # set boundary conditions on the state (done like this is wrong, I should do it on the transfer function's construction)
            if state < par['state_bound'][0]:
                state = par['state_bound'][0]
            if state > par['state_bound'][1]:
                state = par['state_bound'][1]
            state[cell] = state     # apply the new functional as the T function for the associated cell
    '''
    transfer = lambda x: par['gamma'] / (1+np.exp(-NetPar.T*x)) - par['gamma']/2    # transfer function as a sigmoid with trainable gain parameter (slope)
    for t in range(par['ff_iter']):
        state[par['L']//2] = input[0]                        # cell in the upper left corner of the 2D lattice
        state[2*par['L']] = input[1]                   # cell in the upper right corner of the 2D lattice
        state = transfer(np.matmul(NetPar.J,state) + NetPar.B)
    if par['majority_ratio'] is not None:
        output = np.mean(state[int(par['majority_ratio']*par['L']):-1])                          # output as the mean of the value of a group of neurons (majority rule)
    else:
        output = state[-1]
    if verb > 1: 
        print(f'State of each cell: {state}')
    if verb > 0: 
        return output, state
    return output

############# GENETIC ALGORITHM #################
def crossover(par1:NetworkParams,par2:NetworkParams) -> tuple[NetworkParams,NetworkParams]:
    """Function to generate 2 offsprings_list from 2 parents by crossover ricombination.

    Args:
        par1 (NetworkParams): parent network #1.
        par2 (NetworkParams): parent network #2.

    Returns:
        tuple[NetworkParams,NetworkParams]: pair of offsprings_list.
    """   
    # Implements the crossover ricombination given 2 parent NetworkParams and returns 2 offspring NetworksParams.
    if par1.C.shape != par2.C.shape:
        print(f'Error: length mismatch between the connectivity matrices of the two parents! Got parent1={par1.C.shape} and parent2={par2.C.shape}')
        raise ValueError
    if par1.T.shape != par2.T.shape:
        print(f'Error: length mismatch between the T matrices of the two parents! Got parent1={par1.T.shape} and parent2={par2.T.shape}')
        raise ValueError
    '''
    # First, I do C crossover
    cut_idx = nrd.choice(np.arange(len(par1.C)))  # randomly pick where to cut the chromosomes
    C1 = np.append(par1.C.reshape(-1)[:cut_idx],par2.C.reshape(-1)[cut_idx:]).reshape(par1.C.shape)      # create the new C matrices by crossover
    C1 = C1 - np.eye(par1.N)                                                                             # eliminate possible self links
    C2 = np.append(par2.C.reshape(-1)[:cut_idx],par1.C.reshape(-1)[cut_idx:]).reshape(par1.C.shape)
    C2 = C2 - np.eye(par2.N)                                                                             # eliminate possible self links
    J1 = np.append(par1.J.reshape(-1)[:cut_idx],par2.J.reshape(-1)[cut_idx:]).reshape(par1.J.shape)      # create the new J matrices by crossover
    J2 = np.append(par2.J.reshape(-1)[:cut_idx],par1.J.reshape(-1)[cut_idx:]).reshape(par1.J.shape)
    J1 *= C1        # eliminate the weights for non existing links
    J2 *= C2
    B1 = np.append(par1.B[:cut_idx],par2.B[cut_idx:])      # create the new B matrices by crossover
    B2 = np.append(par2.B[:cut_idx],par1.B[cut_idx:])
    # Now I set the other elements of NetPar that are not affected by the crossover
    G1 = Graph(np.column_stack(np.nonzero(C1)))                       # Generate the Graph given the connectivity matrix
    G2 = Graph(np.column_stack(np.nonzero(C2)))      
    '''
    # Now T crossover
    cut_idx = nrd.choice(np.arange(par1.T.shape[0]))            # randomly pick where to cut on the gain parameters string      
    T1 = np.append(par1.T[:cut_idx],par2.T[cut_idx:])  
    T2 = np.append(par2.T[:cut_idx],par1.T[cut_idx:])  
    return (NetworkParams(par1.J,par1.C,par1.B,par1.G,T1,par1.N), NetworkParams(par2.J,par2.C,par2.B,par2.G,T2,par1.N))
    
def mutation(n:NetworkParams,par:dict,gen_verb:bool=False) -> NetworkParams:
    """Function to mutate a given network.

    Args:
        n (NetworkParams): parameters of the network to mutate.
        par (dict): simulation parameters.
        gen_verb (bool,optional): verbosity on the compute_loss function.

    Returns:
        NetworkParams: mutated parameters of the network
    """
    # Function to mutate a given network
    ## Mutate C (probability of link inversly proportional to the length of the link)
    idx = np.arange(n.N)
    row_idx = idx // par['L']
    col_idx = idx % par['L']
    row_diff = row_idx[:, None] - row_idx[None, :]
    col_diff = col_idx[:, None] - col_idx[None, :]
    #dist_matrix = np.sqrt(row_diff**2 + col_diff**2)      
    #prob_matrix = np.where(np.eye(n.N, dtype=bool),  # Create the probability matrix for connectivity
    #   par['p_self_link'],np.exp(-0.188*dist_matrix))       # link probability decays exponentially with the distance (from literature, 
    dist_matrix = np.sqrt(row_diff**2 + col_diff**2) + np.eye(n.N)     
    prob_matrix = np.where(np.eye(n.N, dtype=bool),  # Create the probability matrix for connectivity
    par['p_self_link'],1/dist_matrix - np.eye(n.N))       # link probability decays exponentially with the distance (from literature, 
                                                                                            # https://www.sciencedirect.com/science/article/pii/S0896627313006600#sec2 - pag.3)

    mutationC = np.where(nrd.binomial(n=1,p=np.array(prob_matrix)),1,0)    # Generate C (binomial distribution with n=1 is the Bernoulli distribution) 
    C_m = n.C * mutationC   # Apply the mutation
    ## Mutate J
    mutationJ = par['J_mutation_radius']*nrd.normal(size=n.J.shape)   # mutation as gaussian noise (I multiply for the sd 
                                                                            # bc JAX only gives the unit normal)                                                                      
    J_m = C_m * (n.J + mutationJ)        # Apply the mutation and eliminate weights for non existing links
    J_m = np.clip(J_m, min(par['J_range']), max(par['J_range']))   # Check the boundary conditions for J
    ## Mutate B
    mutationB = par['B_mutation_radius']*nrd.normal(size=n.B.shape)   # mutation as gaussian noise (I multiply for the sd 
                                                                            # bc JAX only gives the unit normal)
    B_m = n.B + mutationB        # Apply the mutation and eliminate biases for non existing links
    # Mutate the T function
    mutationT = par['T_mutation_radius']*par['L']*nrd.normal(size=n.T.shape)   # mutation as gaussian noise (I multiply for the sd 
                                                                            # bc JAX only gives the unit normal)
    T_m = n.T + mutationT        # Apply the mutation 
    G_m = Graph(np.column_stack(np.nonzero(C_m)))     # create a new graph given the mutations
    NetPar = NetworkParams(J_m,C_m,B_m,G_m,T_m,n.N)
    loss_m,_,_ = compute_loss(NetPar,par,verb=gen_verb)
    return NetworkParams(J_m,C_m,B_m,G_m,T_m,n.N,loss_m)
    
def evolution(par:dict,verb:int=1,gen_verb:bool=False,stat:bool=False,early_stop:bool=True) -> tuple[list,list,list,list,list,list,list]:
    """Genetic algolrithm for evolving a network population (both the networks and the single nodes).

    Args:
        par (dict): simulation parameters.
        verb (int, optional): verbosity. Defaults to 1.
        early_stop (bool, optional): early stopping condition. Defaults to True.
        gen_verb (bool,optional): verbosity on the generate function. Default to False.

    Raises:
        ValueError: returns an error if par['N_sol'] is not even.
        ValueError: returns an error if the population is not conserved from one generation to the next.

    Returns:
        tuple[list,list]: offsprings_list list and mean loss values list.
    """
    rndm_flag = False           # some flags for avoiding printing warnings multiple times
    weights_flag = False
    if par['N_sol'] % 2 != 0:     # N_sol must be even
        print(f'Error:The number of solutions N_sol must be an even positive number.')
        raise ValueError
    solutions = []              # Generate the initial batch of solutions
    if stat:
        t_dists = np.zeros(par['N_sol'])                # Container for target distance for each solution
        weights = np.zeros(par['N_sol'])                # Container for link costs for each solution
    for s in range(par['N_sol']):
        # Generate a solution
        sol = generate(par,verb=gen_verb)     # already computes also the loss value
        solutions.append(sol)
        if stat:
            _,t_dist, weight = compute_loss(sol,par)
            t_dists[s] = t_dist
            weights[s] = weight
    if stat:
        Lmean_values = np.zeros(par['n_iter'])               # Initiate some container for statistic
        mean_t_dist_values = np.zeros(par['n_iter'])         # Container for mean target distance
        mean_weight_values = np.zeros(par['n_iter'])         # Container for mean link cost
        mean_asp_values = np.zeros(par['n_iter'])                   # Mean average shortest path length over the solutions
        T_values = np.zeros((par['N_sol'],par['L']**2,par['n_iter']))            # Values of the gain parameter for each network's each cell at each iteration
        mean_t_dist_values[0] = np.mean(t_dists)
        mean_weight_values[0] = np.mean(weights)
        Lmean_values[0] = np.sum(np.array([sol.loss for sol in solutions])) / par['N_sol']   # statistics
        loss_ev = np.zeros((par['N_sol'],par['n_iter']))        #store losses
    
    ##  EVOLUTION
    for iter in range(par['n_iter']):
        '''
        # Save the old solution set for possible early stopping (QUESTO DOVREBBE TRIGGERARE SOLO SE SIAMO IN EARLY STOPPING IN QUESTA ITERAZIONE)
        if early_stop:
            solutions_old = []
            for sol in solutions:
                solutions_old.append(copy.deepcopy(sol))
        '''
        # SELECTION
        solutions = sorted(solutions, key=lambda sol: sol.loss)    # sort in ascending order based on loss
        n_parents = int(np.floor(par['N_sol'] * par['reproduction_ratio'])-np.floor(par['N_sol'] * par['reproduction_ratio'])%2)  # the 2nd floor assures that n_parents is even
        parents_idx = np.zeros((int(n_parents/2),2),dtype=int)
        loss = np.array([sol.loss for sol in solutions]) 
        prob = np.exp(-par['tau']*(loss)) / np.sum(np.exp(-par['tau']*(loss)))        # define the extraction probability for being a parent as the softmax of the negative loss
        for i in range(int(n_parents/2)):
            pair = nrd.choice(np.arange(len(solutions)), size=2, replace=False, p=prob)    # Select unique parent indices for each pair (no repeats in a pair, but pairs can overlap)
            parents_idx[i,:] = pair
        # REPRODUCTION
        offsprings_list = []
        for pair in range(len(parents_idx)):
            if nrd.uniform() <= par['crossover_ratio']:
                children = crossover(solutions[parents_idx[pair,0]],solutions[parents_idx[pair,1]])    # generate 2 offspring by crossover
            else:
                children = [solutions[parents_idx[pair,0]],solutions[parents_idx[pair,1]]]                   # no crossover, only mutation
            offspring0 = mutation(children[0],par,gen_verb=gen_verb)     # mutate them (here also the loss is computed, see mutation function definition)
            offsprings_list.append(offspring0)          # add them to the new population
            offspring1 = mutation(children[1],par,gen_verb=gen_verb)
            offsprings_list.append(offspring1)          # add them to the new population
        best_loss = np.min([sol.loss for sol in solutions])
        if best_loss < 0.1:         # stop random generation if a good solutions has already been found
            if not rndm_flag: 
                print(f'Random generation stopped at iteration #{iter}')
                rndm_flag = True
            n_elite = par['N_sol']-n_parents
        else:
            n_elite = int(par['N_sol'] * par['elitist_ratio'])
        # ELITIST CONSERVATION
        offsprings_list.extend(solutions[:n_elite])         # save the n_elite best individuals to the next generation
        if not best_loss < 0.1:
            # RANDOM GENERATION
            for _ in range(par['N_sol']-n_parents-n_elite):
                sol = generate(par)
                offsprings_list.append(sol)
        random.shuffle(offsprings_list)     # shuffle the new population for good measure
        solutions = offsprings_list         # set the offsprings as the new population
        if len(offsprings_list) != par['N_sol']:      # check population conservation
            print(f'Error: population not conserved; mismatching number of individuals between old and new populations. Got {len(offsprings_list)}, expected {len(t_dists)}')
            raise ValueError
        if stat:
            # Compute the loss
            t_dists = np.zeros(len(solutions))                # Container for target distance for each solution
            links = np.zeros(len(solutions))                # Container for link costs for each solution
            asps = np.zeros(len(solutions))
            for s,sol in enumerate(solutions): 
                _,t_dist, weight = compute_loss(sol,par)
                dist = shortest_distance(sol.G, directed=True).get_2d_array()            # calculate the shortest path length for each pair of vertecies
                dist = np.where(dist >= 2147483647, 0, dist)                               # set each value that overflows (bc no path exists) to 0          # average wiring length cost
                path_cost = np.sum(dist,axis=(0,1))/(sol.G.num_vertices()**2-sol.G.num_vertices())  
                asps[s] = path_cost
                t_dists[s] = t_dist
                weights[s] = weight
            mean_t_dist_values[iter] = np.mean(t_dists)
            mean_weight_values[iter] = np.mean(weights)
            mean_asp_values[iter] = np.mean(asps)
            loss_ev[:,iter] = np.array([sol.loss for sol in solutions])
            Lmean_values[iter] = np.mean(loss_ev[:,iter])  # statistics
            T_values[:,:,iter] = np.array([sol.T for sol in solutions])
        if iter%100 == 0 and verb == 1:
            print(f'Iteration #{iter}...')
            print(f'Mean loss: {Lmean_values[iter]}')
            print(f'Best loss: {np.min(loss_ev[:,iter], axis=0)}')
            if iter == 0:
                print(f'Cumulative Best loss: {np.min(loss_ev[:,iter], axis=0)}')
            else:
                print(f'Cumulative Best loss: {np.min(loss_ev[:,:iter])}')
            #print(f'Mean target distance: {mean_t_dist}')
            #print(f'Mean weight cost: {mean_weight}')
            #print(f'Parents indexes: {parents_idx}')
            #print(f'Selection Probabilities: {prob}')
            #print(f'Inverse losses: {loss}')
        elif iter%10 == 0 and verb == 2:
            print(f'Iteration #{iter}...')
            print(f'Mean loss: {Lmean_values[iter]}')
            print(f'Best loss: {np.min(loss_ev[:,iter], axis=0)}')
            if iter == 0:
                print(f'Cumulative Best loss: {np.min(loss_ev[:,iter], axis=0)}')
            else:
                print(f'Cumulative Best loss: {np.min(loss_ev[:,:iter])}')
            print(f'Mean target distance: {mean_t_dist_values[iter]}')
            #print(f'Mean volume: {np.mean([np.sum(sol.C)/sol.N**2 for sol in solutions])}')
            print(f'Mean weight cost: {mean_weight_values[iter]}')
            #print(f'Parents indexes: {parents_idx}')
            #print(f'Selection Probabilities: {prob}')
            #print(f'Inverse losses: {loss}')
        elif verb == 3: 
            print(f'Iteration #{iter}...')
            print(f'Mean loss: {Lmean_values[iter]}')
            print(f'Best loss: {np.min(loss_ev[:,iter], axis=0)}')
            if iter == 0:
                print(f'Cumulative Best loss: {np.min(loss_ev[:,iter], axis=0)}')
            else:
                print(f'Cumulative Best loss: {np.min(loss_ev[:,:iter])}')
            #print(f'Cumulative Best loss: {np.min(loss_ev[:,:iter])}')
            #print(f'Mean target distance: {mean_t_dist}')
            #print(f'Mean weight cost: {mean_weight}')
            #print(f'Parents indexes: {parents_idx}')
            #print(f'Selection Probabilities: {prob}')
            #print(f'Inverse losses: {loss}')
        #if np.sum(loss_ev[:, iter] < 0.1) >= 0.8*par['N_sol'] and stat:        # check if at least the par['early_stop_ratio'] solutions have loss lower than 0.01
        if best_loss < 0.05:
            if not weights_flag: 
                print(f"Task solved after {iter} iterations. Now weights cost is introduced.")
                weights_flag = True
            solutions = [NetworkParams(s.J,s.C,s.B,s.G,s.T,s.N,s.loss + weights[n]) for n,s in enumerate(solutions)]    # add the weights cost to the loss
        if early_stop:      # very bare-bones version of early stop
            if np.sum(loss_ev[:, iter] < 0.01) >= par['early_stop_ratio']*par['N_sol']:        # check if at least the par['early_stop_ratio'] solutions have loss lower than 0.01
                print(f"Early stopping at iteration #{iter}: 75% of solutions have loss < 0.01")
                # Cut statistics arrays at the current iteration
                Lmean_values = Lmean_values[:iter+1]
                mean_t_dist_values = mean_t_dist_values[:iter+1]
                mean_weight_values = mean_weight_values[:iter+1]
                loss_ev = loss_ev[:, :iter+1]
                mean_asp_values = mean_asp_values[:iter+1]
                T_values = T_values[:, :, :iter+1]
                break
    if stat:
        return solutions, Lmean_values, mean_t_dist_values, mean_weight_values, loss_ev, mean_asp_values, T_values
    else:
        return solutions
        
########## UTILITY FUNCTIONS #################
def plot_T(g:float,par:dict) -> None:
    """Function to plot the shape of the transfer function given the Fourier coefficients.

    Args:
        T (np.ndarray): array of the Fourier coefficients of A SINGLE cell.
        par (dict): simulation parameters.
    """
    new_fun = lambda x: par['gamma'] / (1+np.exp(-g*x)) - par['gamma']/2
    x = np.linspace(-100,100,1000)
    plt.plot(x,new_fun(x))
    plt.grid()
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.title('Transfer function')
    plt.show()
        
        
        