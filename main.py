import random
import openpyxl
import numpy as np  # Add this import statement
from deap import base, creator, tools, algorithms
import pandas as pd
import matplotlib.pyplot as plt

# Loading the dataset
df = pd.read_excel('./DSBA.xlsx', sheet_name='DSBA')

# Data and constraints
intakes = df['INTAKE CODE'].unique()
modules = df['MODULE NAME'].unique()
max_modules_per_intake = 5

# Penalties for constraints
penalty_per_extra_module = 0.5
research_methodology_penalty = 2

# DEAP setup
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))  # We are minimizing the cost
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
toolbox.register("attr_module", random.choice, modules)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_module, n=len(intakes))
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# Fitness function
def evaluate(individual):
    intake_schedule = dict(zip(modules, individual))
    # Constraint: Maximum number of modules per intake
    over_limit = sum(max(0, len(mods) - max_modules_per_intake * (penalty_per_extra_module + research_methodology_penalty)) for mods in intake_schedule.values())
    return over_limit,  

# Genetic operators
toolbox.register("evaluate", evaluate)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.05)
toolbox.register("select", tools.selTournament, tournsize=3)

# Algorithm parameters
population_size = 300
crossover_probability = 0.7
mutation_probability = 0.2
number_of_generations = 40

# Initialize population
population = toolbox.population(n=population_size)

# Statistics
stats = tools.Statistics(key=lambda ind: ind.fitness.values)
stats.register("avg", np.mean)  # Updated to use np.mean
stats.register("min", np.min)  # Updated to use np.min
stats.register("max", np.max)  # Updated to use np.max

# Evolutionary algorithm
random.seed(64)
population, logbook = algorithms.eaSimple(population, toolbox, cxpb=crossover_probability, mutpb=mutation_probability, ngen=number_of_generations, stats=stats, verbose=True)

# Printing out the final population
best_ind = tools.selBest(population, 1)[0]
print("Best Individual: ", best_ind)
print("Best Fitness: ", best_ind.fitness.values)

# Convert best individual into a more readable schedule format
best_schedule = {intake: module for intake, module in zip(intakes, best_ind)}
print("Best Schedule: ", best_schedule)

# Extracting the best fitness values from the logbook
gen = logbook.select("gen")
best_fits = logbook.select("min")  # Assuming we are minimizing the fitness, hence 'min'

# Plotting the best fitness over generations
plt.figure(figsize=(10, 5))
plt.plot(gen, best_fits, 'b-', label='Best Fitness')

# Adding plot labels and title
plt.xlabel('Generation')
plt.ylabel('Best Fitness')
plt.title('Best Fitness Over Generations')
plt.legend(loc='best')

# Show plot
plt.tight_layout()
plt.show()
