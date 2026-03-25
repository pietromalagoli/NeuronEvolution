import numpy as np
import matplotlib.pyplot as plt
import copy

import aux  # some auxiliary functions I often use

######## NETWORK ###########
class Network:
    """
    Class for a generic neural network, defined by 
    - 2 input neurons, 1 output neuron
    - S: # of intra neurons.
    - Theta: threshold value.
    - J: weight matrix.
    - C: connectivity matrix.
    - loss: loss value of the network.
    """
    def __init__(self,par:dict):
        self.S = 2                                                  # Initialize with two intra neurons
        self.Theta = np.zeros(2+self.S+1)                           # I instantiate a threshold value for each neuron, even the input, just for consistency of the indexes     
        self.J = self.initJ(par)                                    # weight matrix J
        self.C = self.initJ(par)                                    # connectivity matrix
        self.activation = lambda x, theta: 0 if x < theta else 1    # activation function (if x = theta -> returns 1)
        self.neurons = np.zeros(2+self.S+1)                         # container for storing the values of the neurons
        self.loss = None                                         # Initialize it to None for avoiding recomputation 
        
    def initJ(self,par:dict):
        """This method is used to initialize the weight matrix J and the connectivity matrix C to the desired form.

        Returns:
            J0 (np.ndarray): mask used to initialize the matrix to the desired form.
            par(dict): Needed for the bool par['longLinks']. If True, the input layer neurons can link directly to the output neuron. If false, they can link only to the hidden layer.
        """
        # Define the weight matrix J0_ij
        J0 = np.ones((2+self.S+1,2+self.S+1)) # Initialized all to one, so all connnected
        J0 = np.tril(J0,-1)                 # Set the diagonal and the elements above the diagonal to zero
        J0[0,1] = 0.            # Impose on the weight matrix the fact that neurons on the same layer cannot be connected 
        J0[1,0] = 0.
        for i in range(2, 2 + self.S):      # Impose on the weight matrix the fact that neurons on the same layer cannot be connected 
            for j in range(2, 2 + self.S):
                if i != j:
                    J0[i, j] = 0. 
        if not par['longLinks']:       # Prohibit link from input to output directly
            J0[-1,0] = 0.
            J0[-1,1] = 0.
        return J0
    
    def generate(self,par:dict):
        """Generate a random network.
        Args:
            par (dict, optional): parameters of the generation.
        """
        self.S = np.random.choice(par['Sk_range'])      # Generate S
        self.Theta = np.random.uniform(par['Theta_range'][0],par['Theta_range'][1],2+self.S+1) # Generate the set of thresholds, Theta. 
        self.J = self.initJ(par) * np.random.uniform(par['J_range'][0],par['J_range'][1],(2+self.S+1)**2).reshape((2+self.S+1,2+self.S+1))   # Generate the weight matrix, J
        self.neurons = np.zeros(2+self.S+1)     # Initialize the values of the neurons (the lenght of this array depends on S)
        self.C = self.initJ(par) * np.random.choice([0,1],(2+self.S+1)**2).reshape((2+self.S+1,2+self.S+1))    # I SHOULD MAKE IT SYMMETRIC    
        self.compute_loss(par)               # Already compute the loss value of the network
        
    def ff(self,input:list,verb:int=0): # feed-forward
        """Compute the output of the network by feed-forward, given an input.
        Args:
            input (list): input to the network. Supported inputs are [0,0];[0,1];[1,0];[1,1].
            verb (int,optional): verbosity. If > 0, returns the value of each neuron. Default to 0.
        Returns:
            output (int): output of the network.
        """
        self.neurons[0:2] = input   # set the value of the first two neurons to the input value
        for neuron in range(2,2+self.S+1):    # compute the value of the intra neurons by feed forward
            self.neurons[neuron] = self.C[neuron,0] * self.J[neuron,0] * self.neurons[0] + self.C[neuron,1] * self.J[neuron,1] * self.neurons[1]   # I refer to the lower triangle of the matrix J
            self.neurons[neuron] = self.activation(self.neurons[neuron],self.Theta[neuron])     # Activation
        out = 0.        # Initialize a variable for containing the value of the output neuron 
        for neuron in range(0,2+self.S+1):  # compute the value of the ouput
            out += self.C[-1,neuron] * self.J[-1,neuron] * self.neurons[neuron]     # Add each neuron's weighted contribution to the output
        self.neurons[-1] = self.activation(out,self.Theta[-1])     # activation
        output = self.neurons[-1]
        if verb > 0:
            return self.neurons
        return output
        
    def compute_loss(self,par:dict):
        """Compute the output of the network and from that the loss value 
            and with that updates the network's loss value.
        Args:
            par (dict): parameters of the generation.
        """
        if self.loss == None:        # I need this check, bc otherwise I risk adding loss over loss
            self.loss = 0.           # This is to avoid type conflict and to make sure that I'm not computing the loss of a network that already has it
            for i,input in enumerate(par['input_set']):
                output = self.ff(input)
                squared_dist = (par['target_set'][i] - output)**2     # square distance between network output and target (theoretical) output
                cost = (np.sum(self.C.flatten())) * self.S      # here the cost is computed only on the presence or absence of links
                self.loss += par['alpha']*squared_dist + cost # add to the loss value of the network
            self.loss /= len(par['input_set'])    # normalize over the inputs
    
def compute_score(n,par):
    """Compute the output of a network and from that the score value 
        and with that updates the network's loss value.
    Args:
        par (dict): parameters of the generation.
    Returns:
        score (float): score of the network on the given task.
    """
    
    score = 0.           # This is to avoid type conflict and to make sure that I'm not computing the loss of a network that already has it
    for i,input in enumerate(par['input_set']):
        output = n.ff(input)
        score += (par['target_set'][i] - output)**2     # square distance between network output and target (theoretical) output
    return score / len(par['input_set'])    # normalize over the inputs  
        
####### EVOLUTION ###########
def evolution(par:dict,verb:int=1,early_stop:bool=True):
    """This function implements the evolutionary algorithm utilizied to evolve the networks population.

    Args:
        par (dict): parameters of the simulation.
        verb (int, optional): verbosity. If == 0, nothing is printed on standard output, if == 1, number of children is printed each 100 iterations,
                            if == 2, each 10 iterations. Defaults to 1.
        early_stop (bool, optional): early stopping option. Defaults to True.

    Returns:
        solutions, Fmean_values (np.ndarray): solutions = array of the final solutions of the algorithm. Fmean_values = array of mean loss values for each iteration.
    """
    # Generate the solutions
    solutions = []
    for _ in range(par['N_sol']):
        # Generate a solution
        n = Network(par)
        n.generate(par)     # already computes also the loss value
        solutions.append(n)
    solutions = np.array(solutions)
    # Initiate some container for statistic
    Fmean_values = []
    S_values = []
    bestLs = []
    meanT = np.zeros((2,par['n_iter'])) # container for the mean value of theta for the mean of the hidden layer neurons (row 0) and thh output neuron (row 1)
    Ndiscards = np.zeros(par['n_iter'])     # container for the number of discarded elements at each iteration
    
    ##  EVOLUTION
    for iter in range(par['n_iter']):
        # Compute the mean loss
        mean_loss = np.sum([sol.loss for sol in solutions]) / par['N_sol']    # Here I don't have to divide also by 4, because I've already done it in the compute_loss method
        # Save the old solution set for possible early stopping
        solutions_old = []
        for sol in solutions:
            solutions_old.append(copy.deepcopy(sol))
        # Discard elements in sol whose loss value is above average 
        solutions = np.array([sol for sol in solutions if sol.loss <= mean_loss])  
        # Compute the number of discarded elements, m
        m = par['N_sol'] - len(solutions)
        # Extract the parents between the survivors (here we can either take them random or take the fittest survivors)
        n_parents = int(np.floor(par['mutation_ratio'] * m))
        parents_idx = np.random.randint(low=0,high=int(len(solutions)),size=n_parents,dtype=int)
        parents = solutions[parents_idx]
        # Firstly, instantiate the offsprings as copies of the parents
        offspring = []
        for parent in parents:
            offspring.append(copy.deepcopy(parent))
        offspring = np.array(offspring)
        if iter%100 == 0 and verb == 1:
            print(f'Iteration #{iter}...')
            print(f'# of childs: {len(offspring)}')
        elif iter%10 == 0 and verb == 2:
            print(f'Iteration #{iter}...')
            print(f'# of childs: {len(offspring)}')
                 
        ## MUTATION
        for i,child in enumerate(offspring):
            ## Now, mutate S first
            sign_array = [-1,0,+1]                # this is used to apply the sign
            sign = np.random.choice(sign_array,p=[0.2,0.6,0.2])     # sample -1,0 or +1 with probabilities of 0.2,0.6 or 0.2, respectively
            child.S += sign * par['S_mutation_radius']
            # Check the boundary conditions for S
            if child.S < min(par['Sk_range']):
                child.S = min(par['Sk_range'])
            if child.S > max(par['Sk_range']):
                child.S = max(par['Sk_range'])
            ## Now mutate the thresholds - if S decreased, you mutate only the remaining neurons, else, you randomly generate the new theta(s)
            # Compute the difference between the orignal S and the mutated one
            deltaS = child.S - parents[i].S
            if deltaS > 0:  # i.e. S increased
                new_thetas = np.random.uniform(par['Theta_range'][0],par['Theta_range'][1],deltaS)
                child.Theta = np.insert(child.Theta,obj=-1,values=new_thetas)   # add the newly generated thetas before the output neuron
            elif deltaS < 0: # i.e. S decrease
                for _ in range(np.abs(deltaS)):
                    child.Theta = np.delete(child.Theta,obj=-1) # delete the theta values of the deleted neurons
            # mutate theta (I mutate also the newly generated thetas, but not the input neurons' thetas, since they have (by definition of NN) no activation function)
            mutationTheta = np.random.normal(0.,par['Theta_mutation_radius'],child.S+1)  # mutation as gaussian noise (not mutate inputs' thetas)
            child.Theta[2:] += mutationTheta    # (not mutate inputs' thetas)
            # Check the boundary conditions for Theta
            for i,theta in enumerate(child.Theta[2:]):      # check on all but inputs' thetas
                if theta < min(par['Theta_range']):
                    child.Theta[i] = min(par['Theta_range'])
                if theta > max(par['Theta_range']):
                    child.Theta[i] = max(par['Theta_range'])  
            ## Mutate C
            # first check for S
            C0 = child.initJ(par)     # I utilize the method initJ() to create a new C given the new S
            if deltaS > 0:
                for _ in range(deltaS):
                    c = np.random.choice([0,1])  # generate a random value to be given to the new links
                    child.C = np.insert(child.C,obj=-1,values=c,axis=0)    # insert a row of 1s (axis 0)
                    child.C = np.insert(child.C,obj=-1,values=c,axis=1)    # insert a column of 1s (axis 1)
            elif deltaS < 0:
                for _ in range(np.abs(deltaS)):
                    child.C = np.delete(child.C,obj=-1,axis=0) # delete the C values of the deleted neurons on axis 0
                    child.C = np.delete(child.C,obj=-1,axis=1) # delete the C values of the deleted neurons on axis 1                    
            # then mutate C 
            mutationC = np.random.choice([0,1], len(child.C)**2,p=[0.2,0.8]).reshape(child.C.shape)   # generate the mutation radius 
            # Apply the mutation
            child.C = C0 * mutationC
            ## Mutate J
            J0 = child.initJ(par)     # I utilize the method initJ() to create a new J given the new S
            if deltaS > 0:
                for _ in range(deltaS):
                    j = np.random.uniform(par['J_range'][0],par['J_range'][1])  # generate a random value to be given to the new links
                    child.J = np.insert(child.J,obj=-1,values=j,axis=0)    # insert a row of 1s (axis 0)
                    child.J = np.insert(child.J,obj=-1,values=j,axis=1)    # insert a column of 1s (axis 1)
            elif deltaS < 0:
                for _ in range(np.abs(deltaS)):
                    child.J = np.delete(child.J,obj=-1,axis=0) # delete the J values of the deleted neurons on axis 0
                    child.J = np.delete(child.J,obj=-1,axis=1) # delete the J values of the deleted neurons on axis 1                    
            # mutate (I mutate also the newly generated thetas)
            mutationJ = np.random.normal(0.,par['J_mutation_radius'],len(child.J)**2).reshape(child.J.shape)    # mutation as gaussian noise 
            # Apply the mutation
            child.J += mutationJ
            # multiplicate with J0 to set to 0 where necessary by the conditions         
            child.J *= J0
            # Check the boundary conditions for J
            for r,row in enumerate(child.J):
                for c,j in enumerate(row):
                    if j < min(par['J_range']):
                        child.J[r,c] = min(par['J_range'])
                    if j > max(par['J_range']):
                        child.J[r,c] = max(par['J_range'])        
            # Fix also the other attributes
            if deltaS > 0:
                value = np.random.choice(np.array([0.0,1.0]))
                child.neurons = np.insert(child.neurons,obj=-1,values=value)      # i set the new neuron to either 0 or 1, uniformly
            elif deltaS < 0:
                np.delete(child.neurons,2+child.S,deltaS)   # delete the eliminated intra neurons
            # Already compute the loss value of the network
            child.loss = None            # I have to first set it to None beacuse child inherits self.loss from the parent
            child.compute_loss(par)      # Here I have to do it explicitly because I'm not generating the Network object through Network.generate()       
        # Add the offsprings generated by mutation to the survived solutions
        solutions = np.concatenate([solutions,offspring])
        # Randomly generate the remaining individuals (population must be constant)
        n_generation = m - n_parents
        generated = []
        for _ in range(n_generation):
            # Generate a solution
            n = Network(par)
            n.generate(par)             # already with computed loss
            generated.append(n)
        generated = np.array(generated)     # transform it to a np.ndarray
        # Add the randomly generated solutions to the other solutions
        solutions = np.concatenate([solutions,generated])
        # Add a check for conservation of population
        if len(solutions) != par['N_sol']:
            msg = f'The solutions population was not conserved. Expected {par['N_sol']}, but instead got {len(solutions)}.'
            aux.checkpoint(True,msg=msg,stop=True)    
        # Shuffle the order of the solutions for good measure
        np.random.shuffle(solutions)
        # Statistic 
        bestLs.append(np.min([sol.loss for sol in solutions])) # best loss in the population
        meanS = np.mean([sol.S for sol in solutions])
        Fmean_values.append(mean_loss)
        S_values.append(meanS)
        meanT[0,iter] = np.mean(([np.mean(sol.Theta[2:-1]) for sol in solutions]))       # mean of the thetas of the mean thetas on the hidden layer
        meanT[1,iter] = np.mean([sol.Theta[-1] for sol in solutions],axis=0)             # mean of the thetas on the output neuron
        Ndiscards[iter] = m/par['N_sol']         # number of discarded individuals at each iteration (it's kinda of a measure of variance on the population)
        # Possible early stopping
        if early_stop:
            mean_score = np.mean([compute_score(sol,par) for sol in solutions])
            if len(offspring) <= par['n_early_stop'] and mean_score < 0.01:
                print(f'Convergence reached after {iter} iterations with mean score {mean_score} .')
                return solutions_old, Fmean_values, S_values, bestLs, meanT[:,:iter], Ndiscards[:iter]
        
    return solutions, Fmean_values, S_values, bestLs, meanT, Ndiscards
