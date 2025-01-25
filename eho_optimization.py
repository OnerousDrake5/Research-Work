import numpy as np

def eho_feature_selection(features, fitness_function, num_iterations=100, population_size=50):
    """
    Perform feature selection using Elephant Herding Optimization (EHO).
    :param features: Input feature matrix.
    :param fitness_function: Function to evaluate fitness of a feature subset.
    :param num_iterations: Number of iterations for EHO.
    :param population_size: Size of the population.
    :return: Best feature subset.
    """
    # Initialize population (binary vectors representing feature subsets)
    num_features = features.shape[1]
    population = np.random.randint(2, size=(population_size, num_features))
    
    # Iterate through generations
    for iteration in range(num_iterations):
        print(f"Iteration {iteration + 1}/{num_iterations}")
        
        # Evaluate fitness of each individual
        fitness_scores = []
        for individual in population:
            selected_features = features[:, individual == 1]  # Select features
            if selected_features.shape[1] == 0:  # Skip if no features are selected
                fitness_scores.append(0)
                continue
            fitness_scores.append(fitness_function(selected_features))
        fitness_scores = np.array(fitness_scores)
        
        # Select the best individuals (simplified EHO logic)
        best_indices = np.argsort(fitness_scores)[-population_size//2:]
        population = population[best_indices]
        
        # Generate new individuals (crossover and mutation)
        new_population = []
        for i in range(population_size):
            # Select two parents randomly from the best individuals
            parent1, parent2 = np.random.choice(len(best_indices), 2, replace=False)
            parent1 = population[parent1]
            parent2 = population[parent2]
            
            # Perform crossover (bitwise OR operation)
            child = parent1 | parent2
            
            # Perform mutation (flip a random bit with 10% probability)
            if np.random.rand() < 0.1:
                mutation_index = np.random.randint(num_features)
                child[mutation_index] = 1 - child[mutation_index]
            
            new_population.append(child)
        
        # Update the population
        population = np.array(new_population)
    
    # Return the best feature subset
    best_individual = population[np.argmax(fitness_scores)]
    return best_individual == 1  # Return a boolean mask for selected features