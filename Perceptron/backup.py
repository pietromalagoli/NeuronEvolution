def evolution(par:dict):
    # Generate the solutions
    solutions = []
    for _ in range(par['N_sol']):
        # Generate a solution
        n = Network()
        n.generate(par)     # already computes also the fitness value
        solutions.append(n)
    solutions = np.array(solutions)
    # Initiate some container for statistic
    mean_values = []
    S_values = []
    
    ##  EVOLUTION
    for iter in range(par['n_iter']):
        print(f'Iteration #{iter}...')
        # Compute the mean fitness
        mean_fit = np.sum([sol.fitness for sol in solutions]) / par['N_sol']    # Here I don't have to divide also by 4, because I've already done it in the compute_fitness method
        # Discard elements in sol whose fitness value is below average 
        solutions = np.array([sol for sol in solutions if sol.fitness >= mean_fit])
        # Compute the number of discarded elements, m
        m = par['N_sol'] - len(solutions)
        # Extract the parents between the survivors (here we can either take them random or take the fittest survivors)
        n_parents = int(np.floor(par['mutation_ratio'] * m))
        parents_idx = np.random.randint(low=0,high=int(len(solutions)),size=n_parents,dtype=int)
        parents = solutions[parents_idx]
        # Firstly, instantiate the offsprings as copies of the parents
        offspring = copy.deepcopy(parents)      # I must use copy.deepcopy because otherwise I modify both objects when I modify one
        
        ## MUTATION
        # Now, mutate S first
        for i,child in enumerate(offspring):
            sign_idx = np.random.randint(2)     # extract uniformly 0 or 1 for choosing the sign of the mutation
            sign_array = [-1,+1]                # this is used to apply the sign
            child.S += sign_array[sign_idx] * par['S_mutation_radius']
            # Check the boundary conditions for S
            if child.S < min(par['Sk_range']):
                child.S = min(par['Sk_range'])
            if child.S > max(par['Sk_range']):
                child.S = max(par['Sk_range'])
            # Now mutate the thresholds - if S decreased, you mutate only the remaining neurons, else, you randomly generate the new theta(s)
            # compute the difference between the orignal S and the mutated one
            deltaS = child.S - parents[i].S
            print(f'DeltaS:{deltaS}')
            if deltaS > 0:
                new_thetas = np.random.uniform(par['Theta_range'][0],par['Theta_range'][1],deltaS)
                child.Theta = np.insert(child.Theta,obj=-1,values=new_thetas)   # add the newly generated thetas before the output neuron
            else:
                for _ in range(np.abs(deltaS)):
                    np.delete(child.Theta,obj=-1) # delete the theta values of the deleted neurons
            # mutate theta (I mutate also the newly generated thetas)
            mutationTheta = np.random.uniform(par['Theta_range'][0]*par['Theta_mutation_radius'], 
                                        par['Theta_range'][1]*par['Theta_mutation_radius'], len(child.Theta))   # generate the mutation radius 
            child.Theta += mutationTheta
            # Check the boundary conditions for Theta
            for theta in child.Theta:
                if theta < min(par['Theta_range']):
                    theta = min(par['Theta_range'])
                if theta > max(par['Theta_range']):
                    theta = max(par['Theta_range'])     
            # Mutate J
            J0 = child.initJ()     # I utilize the method initJ() to create a new J given the new S
            if deltaS > 0:
                for _ in range(deltaS):
                    j = np.random.uniform(par['J_range'][0],par['J_range'][1])  # generate a random value to be given to the new links
                    child.J = np.insert(child.J,obj=-1,values=j,axis=0)    # insert a row of 1s (axis 0)
                    child.J = np.insert(child.J,obj=-1,values=j,axis=1)    # insert a column of 1s (axis 1)
            else:
                for _ in range(deltaS):
                    np.delete(child.J,np.linspace(2+parents[i].S,2+child.S,deltaS,axis=0)) # delete the J values of the deleted neurons on axis 0
                    np.delete(child.J,np.linspace(2+parents[i].S,2+child.S,deltaS,axis=1)) # delete the J values of the deleted neurons on axis 1        
            # mutate (I mutate also the newly generated thetas)
            mutationJ = np.random.uniform(par['J_range'][0]*par['J_mutation_radius'], 
                                        par['J_range'][1]*par['J_mutation_radius'], len(child.J)**2).reshape(child.J.shape)   # generate the mutation radius 
            # Apply the mutation
            child.J += mutationJ
            # multiplicate with J0 to set to 0 where necessary by the conditions
            print(f'J:{child.J.shape}')
            print(f'J0:{J0.shape}')           
            child.J *= J0
            # Check the boundary conditions for J
            for row in child.J:
                for j in row:
                    if j < min(par['J_range']):
                        j = min(par['J_range'])
                    if j > max(par['J_range']):
                        j = max(par['J_range'])        
            # Fix also the other attributes
            if deltaS > 0:
                child.neurons = np.insert(child.neurons,obj=-1,values=0.5)      # i set the new intra neurons to 0.5
            else:
                np.delete(child.neurons,2+child.S,deltaS)   # delete the eliminated intra neurons
                
            # Already compute the fitness value of the network
            child.fitness = None            # I have to first set it to None beacuse child inherits self.fitness from the parent
            child.compute_fitness(par)      # Here I have to do it explicitly because I'm not generating the Network object through Network.generate()       
        # Add the offsprings generated by mutation to the survived solutions
        solutions = np.concatenate([solutions,offspring])
        # Randomly generate the remaining individuals (population must be constant)
        n_generation = m - n_parents
        generated = []
        for _ in range(n_generation):
            # Generate a solution
            n = Network()
            n = n.generate(par)             # already with computed fitness
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
        meanS = np.mean(solutions.S)
        mean_values.append(mean_fit)
        S_values.append(meanS)
    return solutions, mean_values, meanS
