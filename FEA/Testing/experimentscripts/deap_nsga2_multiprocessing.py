import array
import random
import json
import time

import numpy as np
import matplotlib.pyplot as plt

from math import sqrt

from deap import algorithms
from deap import base
from deap import benchmarks
from deap.benchmarks.tools import diversity, convergence, hypervolume
from deap import creator
from deap import tools
from scoop import futures


if __name__ == '__main__':
    #Creating a goodness-of-fit class that is optimized by minimizing the goodness of fit
    creator.create("FitnessMin", base.Fitness, weights=(-1.0, -1.0, -1.0))
    #Create individual class Individual
    creator.create("Individual", array.array, typecode='d', fitness=creator.FitnessMin)
    #Creating a Toolbox
    toolbox = base.Toolbox()
    #Specify the range of values ​​that a gene can take
    BOUND_LOW, BOUND_UP = 0.0, 1.0
    #Specify the number of genes in one individual
    NDIM = 30
    NOBJ = 3

    #Gene generation function
    def uniform(low, up, size=None):
        try:
            return [random.uniform(a, b) for a, b in zip(low, up)]
        except TypeError:
            return [random.uniform(a, b) for a, b in zip([low] * size, [up] * size)]

    #Functions that generate genes"attr_gene"
    toolbox.register("attr_float", uniform, BOUND_LOW, BOUND_UP, NDIM)
    #Function to generate an individual "individual""
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.attr_float)
    #Functions that generate populations"population"
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    #Evaluation function"evaluate"
    toolbox.register("evaluate", benchmarks.dtlz1, obj=NOBJ)
    #Function to cross"mate"
    toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=BOUND_LOW, up=BOUND_UP, eta=20.0)
    #Mutant function"mutate"
    toolbox.register("mutate", tools.mutPolynomialBounded, low=BOUND_LOW, up=BOUND_UP, eta=20.0, indpb=1.0/NDIM)
    #Individual selection method"select"
    toolbox.register("select", tools.selNSGA2)
    toolbox.register("map", futures.map)

    NGEN = 1000 #Number of repeating generations
    MU = 500 #Population in the population
    CXPB = 0.9 #Crossover rate

    #Setting what to output to the log during the generation loop
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("min", np.min, axis=0)
    stats.register("max", np.max, axis=0)

    logbook = tools.Logbook()
    logbook.header = "gen", "evals", "std", "min", "avg", "max"

    #First generation generation
    pop = toolbox.population(n=MU)
    pop_init = pop[:]
    invalid_ind = [ind for ind in pop if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit
    pop = toolbox.select(pop, len(pop))

    record = stats.compile(pop)
    logbook.record(gen=0, evals=len(invalid_ind), **record)

    start = time.time()

    #Performing optimal calculations
    for gen in range(1, NGEN):
        #Population generation
        offspring = tools.selTournamentDCD(pop, len(pop))
        offspring = [toolbox.clone(ind) for ind in offspring]
        
        #Crossover and mutation
        for ind1, ind2 in zip(offspring[::2], offspring[1::2]):
            #Select individuals to cross
            if random.random() <= CXPB:
                #Crossing
                toolbox.mate(ind1, ind2)
            
            #mutation
            toolbox.mutate(ind1)
            toolbox.mutate(ind2)
            
            #Crossed and mutated individuals remove fitness
            del ind1.fitness.values, ind2.fitness.values
        
        #Re-evaluate fitness for individuals with fitness removed
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        #Select the next generation
        pop = toolbox.select(pop + offspring, MU)
        record = stats.compile(pop)
        logbook.record(gen=gen, evals=len(invalid_ind), **record)
        print(logbook.stream)

    #Output the last generation hyper volume
    print("Final population hypervolume is %f" % hypervolume(pop, [1000,1000,1000]))

    end = time.time()
    elapsed = end - start
    print(elapsed)