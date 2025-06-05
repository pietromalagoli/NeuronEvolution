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
    state: np.ndarray      # values of the cells in the network
    T: np.ndarray   # array of the T functions of the single cells in the network
    N: int = 0              # side lenght of the squared lattice (number of cells = N**2)
    fitness: float = -1.     # fitness value of the network
    
# ORA DEFINISCO LE VARIE FUNZIONI PER OPERARE SUI NETWORK
# - GENERATE(PAR,NETPAR): CREA (IN REALTà NON CREA L'OGGETTO) UN NETWORK RANDOM DATI I PARAMETRI 
#   PAR E RITORNA I PARAMETRI DEL NETWORK (NETPAR) AGGIORNATI
# - COMPUTE_FITNESS(PAR,NETPAR): CALCOLA LA FITNESS DEL NETWORK ATTRAVERSO IL METODO FF() E
#   RITORNA TALE FITNESS.
# - FF(INPUT(S),NETPAR): FA IL 'FEED-FORWARD' DEL NETWORK DATO UN SET DI INPUT E RITORNA 
#   LO STATE (OSSIA L'ARRAY DEGLI STATI 0 O 1 DELLE SINGOLE CELL)
    
def generate(par:dict) -> NetworkParams:  # JAXXED!
    """Generate a random network given the parameters par by updating the 
        NetPar object.

    Args:
        par (dict): simulation/generation parameters.

    Returns:
        NetworkParams: returns the new network's parameters set. 
    """
    N = par['N']  
    C = np.ones((N**2,N**2))                        # as first, initialize C as fully connected (all ones)
    C = C - np.eye(N**2)                                # no self links                                    
    J = nrd.uniform(size=(N**2,N**2)) * C                # uniformly populate the weights in the weight matrix J                                       
    B = nrd.normal(size=(N**2,N**2)) * C                 # extract from a normal distribution centered in 0 with st. dv. = 1 the biases (spero vada bene fatto così)
    G = Graph(np.column_stack(np.nonzero(C)))         # Generate the Graph given the connectivity matrix
    state = np.abs(nrd.normal(size=(N**2,)))        # Randomly assign a state value to each cell distributed as the absolute value of a normal around 0
    f_n = par['T_initial'](par['x_n'])                  # evaluate the initial transfer function on x_n
    T = np.fft.fft(f_n)                                # compute the Fourier coefficients of f_n to move to the frequency space
    T = np.repeat(T,N**2).reshape(N**2,len(par['x_n']))        # each cell of the network at initialization has the same transfer function
    NetPar = NetworkParams(J,C,B,G,state,T,N)              # I don't specify the fitness argument so it stays as default (-1)
    fitness, state = compute_fitness(NetPar,par,verb=True)                      # Compute the fitness value of the network (also updates the state)
    return NetworkParams(J,C,B,G,state,T,N,fitness)           # return the set of updated parameters

def compute_fitness(NetPar:NetworkParams,par:dict,verb:bool=False) -> tuple[float,np.ndarray]:    # JAXXED!
    """Function for computing the fitness of a given network.

    Args:
        NetPar (NetworkParams): parameter of the network in question.
        par (dict): simulation parameters.
        verb (bool, optional): verbosity. Defaults to False.

    Returns:
        tuple[float,np.ndarray]: computed fitness and updated states of the nodes of the network. 
    """
    fitness = 0.                # I need this check, bc otherwise I risk adding fitness over fitness  
    for i,input in enumerate(par['input_set']):
        output,state = ff(input,NetPar,par)
        if verb:
            print(f'Input: {input} -> {output}')
        norm_factor = NetPar.G.num_vertices()**2-NetPar.G.num_vertices()            # normalize by N(N-1)
        target_dist = (par['target_set'][i] - output)**2                            # square distance between network output and target (theoretical) output
        volume_cost = (np.sum(NetPar.C.reshape(-1))/norm_factor)**2                # average wiring volume cost (i.e. # of links)
        dist = shortest_distance(NetPar.G, directed=True).get_2d_array()            # calculate the shortest path length for each pair of vertecies
        dist = np.where(dist >= 2147483647, 0, dist)                               # set each value that overflows (bc no path exists) to 0
        length_cost = (np.sum((NetPar.C.reshape(-1) * dist.reshape(-1))
                                **par['length_powerlaw'])/norm_factor)**2            # average wiring length cost
        path_cost = (np.sum(dist,axis=(0,1))/norm_factor)**2                       # average shortest path length cost
        if verb:
            print(f'Target distance:{target_dist}')
            print(f'Volume cost:{volume_cost}')
            print(f'Length cost:{length_cost}')
            print(f'Path cost:{path_cost}')
        fitness += np.exp(target_dist + volume_cost + length_cost + path_cost) # take the exponential of the sum of the costs (weighted if needed)
    fitness /= len(par['input_set'])    
    return fitness, state

def ff(input:list,NetPar:NetworkParams,par:dict,verb:int=0) -> tuple[float,np.ndarray]:     # JAXXED!
    """Function to execute the 'feed-forward' computation on a given network, given inputs.

    Args:
        input (list): inputs to the network.
        NetPar (NetworkParams): network in question.
        par (dict): simulation parameters.
        verb (int, optional): verbosity. Defaults to 0.

    Returns:
        tuple[float,np.ndarray]: output of the 'feed-forward' and updated states of each node of the network.
    """
    state = NetPar.state                                # set the value of the two inputs cells through the input value
    state[0] = input[0]                        # cell in the upper left corner of the 2D lattice
    state[(NetPar.N - 1) * NetPar.N-1] = input[1]    # cell in the lower left corner of the 2D lattice
    # Precompute the contributions for each cell to optimize the loop (this is incredibly faster)
    contributions = np.dot(NetPar.C * NetPar.J, state) + np.sum(NetPar.B, axis=1)      # weight x value + bias - contributions così ha shape = state.shape
    # Retrive the T function of each cell by doing ifft and then fit/interpolation
    f_n = np.fft.ifft(NetPar.T,axis=1)      # inverse FT for moving back to coordinates space
    for cell in range(f_n.shape[0]):
        coeff = np.polyfit(par['x_n'],f_n[cell,:],2)    # fit the series of function value with a polynomial of degree up to 2
        new_fun = np.poly1d(coeff)                      # generate the new functional from the fit results
        state[cell] = new_fun(contributions[cell])    # apply the new functional as the T function for the associated cell
    if verb > 0: 
        print(f'State of each cell: {state}')
    output = state[-1]                          # the cell in the right lower corner is the output
    return output, state

############# GENETIC ALGORITHM #################
def crossover(par1:NetworkParams,par2:NetworkParams) -> tuple[NetworkParams,NetworkParams]:
    """Function to generate 2 offsprings from 2 parents by crossover ricombination.

    Args:
        par1 (NetworkParams): parent network #1.
        par2 (NetworkParams): parent network #2.

    Returns:
        tuple[NetworkParams,NetworkParams]: pair of offsprings.
    """   
    # Implements the crossover ricombination given 2 parent NetworkParams
    # and returns 2 offspring NetworksParams.
    if par1.C.shape != par2.C.shape:
        print(f'Error: length mismatch between the connectivity matrices of the two parents! Got parent1={par1.C.shape} and parent2={par2.C.shape}')
        raise ValueError
    if par1.T.shape != par2.T.shape:
        print(f'Error: length mismatch between the T matrices of the two parents! Got parent1={par1.T.shape} and parent2={par2.T.shape}')
        raise ValueError
    # First, I do C crossover
    cut_idx = nrd.choice(np.arange(len(par1.C)))  # randomly pick where to cut the chromosomes
    C1 = np.append(par1.C.reshape(-1)[:cut_idx],par2.C.reshape(-1)[cut_idx:]).reshape(par1.C.shape)      # create the new C matrices by crossover
    C1 = C1 - np.eye(par1.N**2)                                                                             # eliminate possible self links
    C2 = np.append(par2.C.reshape(-1)[:cut_idx],par1.C.reshape(-1)[cut_idx:]).reshape(par1.C.shape)
    C2 = C2 - np.eye(par2.N**2)                                                                             # eliminate possible self links
    J1 = np.append(par1.J.reshape(-1)[:cut_idx],par2.J.reshape(-1)[cut_idx:]).reshape(par1.J.shape)      # create the new J matrices by crossover
    J2 = np.append(par2.J.reshape(-1)[:cut_idx],par1.J.reshape(-1)[cut_idx:]).reshape(par1.J.shape)
    J1 *= C1        # eliminate the weights for non existing links
    J2 *= C2
    B1 = np.append(par1.B.reshape(-1)[:cut_idx],par2.B.reshape(-1)[cut_idx:]).reshape(par1.B.shape)      # create the new B matrices by crossover
    B2 = np.append(par2.B.reshape(-1)[:cut_idx],par1.B.reshape(-1)[cut_idx:]).reshape(par1.B.shape)
    B1 *= C1        # eliminate the biases for non existing links
    B2 *= C2
    # Now T crossover
    cut_idx = nrd.choice(np.arange(par1.T.shape[1]),par1.T.shape[0])  # randomly pick where to cut each cell's T coefficients string
    T1 = np.zeros((par1.T.shape))
    T2 = np.zeros((par1.T.shape))
    for cell in range(par1.T.shape[0]):      # I have to use this loop because append only works with one dimensional arrays
        T1[cell,:] = np.append(par1.T[cell,:cut_idx[cell]],par2.T[cell,cut_idx[cell]:])
        T2[cell,:] = np.append(par2.T[cell,:cut_idx[cell]],par1.T[cell,cut_idx[cell]:])
    # Now I set the other elements of NetPar that are not affected by the crossover
    state1 = np.abs(nrd.normal(size=par1.state.shape))     # State distributed as in the initialization because, biologically,
    state2 = np.abs(nrd.normal(size=par1.state.shape))     # it makes sense that the children do not inherit the value of the neurons of the parents
    G1 = Graph(np.column_stack(np.nonzero(C1)))                       # Generate the Graph given the connectivity matrix
    G2 = Graph(np.column_stack(np.nonzero(C2)))      
    # LE PROSSIME RIGHE SONO COMMENTATE PERCHé PER ORA NON MI SERVE CHE QUESTA FUNZIONE CALCOLI ANCHE LA FITNESS, PERCHé
    # LO FA LA FUNZIONE MUTATION, CHE PER ORA VIENE SEMPRE ESEGUITA UNA VOLTA CHE VIENE ESEGUITA QUESTA                 
    #NetPar1 = NetworkParams(J1,C1,B1,G1,state1,T1,par1.N)
    #NetPar2 = NetworkParams(J2,C2,B2,G2,state2,T2,par1.N)
    #fitness1, state1 = compute_fitness(par,NetPar1)
    #fitness2, state2 = compute_fitness(par,NetPar2)
    #return [NetworkParams(J1,C1,B1,G1,state1,T1,par1.N,fitness1), NetworkParams(J2,C2,B2,G2,state2,T2,par1.N,fitness2)]
    return (NetworkParams(J1,C1,B1,G1,state1,T1,par1.N), NetworkParams(J2,C2,B2,G2,state2,T2,par1.N))
    
def mutation(n:NetworkParams,par:dict) -> NetworkParams:
    """Function to mutate a given network.

    Args:
        n (NetworkParams): parameters of the network to mutate.
        par (dict): simulation parameters.

    Returns:
        NetworkParams: mutated parameters of the network
    """
    # Function to mutate a given network
    ## Mutate C (probability of link inversly proportional to the length of the link)
    idx = np.arange(par['N']**2)
    row_idx = idx // par['N']
    col_idx = idx % par['N']
    row_diff = row_idx[:, None] - row_idx[None, :]
    col_diff = col_idx[:, None] - col_idx[None, :]
    dist_matrix = np.sqrt(row_diff**2 + col_diff**2) + np.eye(par['N']**2)      # I add the np.eye so to avoid a division by zero in the next line
    prob_matrix = np.where(np.eye(par['N']**2, dtype=bool),  # Create the probability matrix for connectivity
        par['p_self_link'],1/dist_matrix-np.eye(par['N']**2))       # 1/dist_matrix - np.eye to erase the previous addition (it does not make an actual difference 
                                                                    # because the elements on the diagonal are dictated by par['p_self_link])
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
    B_m = C_m * (n.B + mutationB)        # Apply the mutation and eliminate biases for non existing links
    # Mutate the T function
    mutationT = par['T_mutation_radius']*nrd.normal(size=n.T.shape)   # mutation as gaussian noise (I multiply for the sd 
                                                                            # bc JAX only gives the unit normal)
    T_m = n.T + mutationT        # Apply the mutation 
    G_m = Graph(np.column_stack(np.nonzero(C_m)))     # create a new graph given the mutations
    NetPar = NetworkParams(J_m,C_m,B_m,G_m,n.state,T_m,n.N)
    fitness_m, state_m = compute_fitness(NetPar,par)
    return NetworkParams(J_m,C_m,B_m,G_m,state_m,T_m,n.N,fitness_m)

def evolution(par:dict,verb:int=1,early_stop:bool=True) -> tuple[list,list]:
    """Genetic algolrithm for evolving a network population (both the networks and the single nodes).

    Args:
        par (dict): simulation parameters.
        verb (int, optional): verbosity. Defaults to 1.
        early_stop (bool, optional): early stopping condition. Defaults to True.

    Raises:
        ValueError: returns an error if par['N_sol'] is not even.
        ValueError: returns an error if the population is not conserved from one generation to the next.

    Returns:
        tuple[list,list]: offsprings list and mean fitness values list.
    """
    if par['N_sol'] % 2 != 0:     # N_sol must be even
        print(f'Error:The number of solutions N_sol must be an even positive number.')
        raise ValueError
    solutions = []              # Generate the initial batch of solutions
    for n in range(par['N_sol']):
        # Generate a solution
        sol = generate(par)     # already computes also the fitness value
        solutions.append(sol)
    Fmean_values = []               # Initiate some container for statistic
    
    ##  EVOLUTION
    for iter in range(par['n_iter']):
        # Compute the mean fitness for statistics
        mean_fit = np.sum(np.array([sol.fitness for sol in solutions])) / par['N_sol']    # Here I don't have to divide also by 4, because I've already done it in the compute_fitness method
        # Save the old solution set for possible early stopping (QUESTO DOVREBBE TRIGGERARE SOLO SE SIAMO IN EARLY STOPPING IN QUESTA ITERAZIONE)
        '''
        if early_stop:
            solutions_old = []
            for sol in solutions:
                solutions_old.append(copy.deepcopy(sol))
        '''
        # SELECTION
        solutions = sorted(solutions, key=lambda sol: sol.fitness)    # sort in ascending order based on fitness
        n_parents = int(np.floor(par['N_sol'] * par['reproduction_ratio'])-np.floor(par['N_sol'] * par['reproduction_ratio'])%2)  # the 2nd floor assures that n_parents is even
        parents_idx = nrd.choice(np.arange(par['N_sol']),
                                 size=(int(n_parents/2),2),replace=True,
                                 p=np.array([1/sol.fitness for sol in solutions])
                                 /np.sum(np.array([1/sol.fitness for sol in solutions])))
        # REPRODUCTION
        offsprings = []
        for pair in range(len(parents_idx)):
            offspring = crossover(solutions[parents_idx[pair,0]],solutions[parents_idx[pair,1]])    # generate 2 offspring by crossover
            offspring0 = mutation(offspring[0],par)     # mutate them
            offsprings.append(offspring0)          # add them to the new population
            offspring1 = mutation(offspring[1],par)
            offsprings.append(offspring1)          # add them to the new population
        if iter%100 == 0 and verb == 1:
            print(f'Iteration #{iter}...')
            print(f'# of childs: {len(offspring)}')
        elif iter%10 == 0 and verb == 2:
            print(f'Iteration #{iter}...')
            print(f'# of childs: {len(offspring)}')
        elif verb == 3: 
            print(f'Iteration #{iter}...')
            print(f'# of childs: {len(offspring)}')
        # RANDOM GENERATION
        for _ in range(par['N_sol']-n_parents):
            sol = generate(par)
            offsprings.append(sol)
        if len(offsprings) != par['N_sol']:      # check population conservation
            print(f'Error: population not conserved; mismatching number of individuals between old and new populations. Got {len(offsprings)}, expected {par['N_sol']}')
            raise ValueError
        random.shuffle(offsprings)     # shuffle the new population for good measure
        Fmean_values.append(mean_fit)   # statistics
    return offsprings, Fmean_values
        
########## UTILITY FUNCTIONS #################
def plot_T(T:np.ndarray,par:dict) -> None:
    f_n = np.fft.ifft(T)
    coeff = np.polyfit(par['x_n'],f_n,2)    # fit the series of function value with a polynomial of degree up to 5
    new_fun = np.poly1d(coeff)                      # generate the new functional from the fit results
    x = np.arange(min(par['x_n']),max(par['x_n']),1000)
    plt.plot(x,new_fun(x))
    plt.show()
        
        
        