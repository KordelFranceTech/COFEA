from re import I
from deap import benchmarks
from datetime import timedelta
import time
from functools import partial

from optimizationproblems.knapsack import *
from MOO.MOEA import NSGA2
from basealgorithms.ga import GA
from utilities.util import *


# benchmarks.dtlz1  # individual, number of objectives 'obj'

BOUND_LOW, BOUND_UP = 0.0, 1.0
#Specify the number of genes in one individual
NDIM = 30
NOBJ = 3
NGEN = 1000 #Number of repeating generations
MU = 500 #Population in the population
CXPB = 0.9 #Crossover rate
MRPB = 1.0/NDIM #mutation rate

ga = GA

@add_method(NSGA2)
def calc_fitness(variables, gs=None, factor=None):
    fitnesses = benchmarks.dtlz1(variables, NOBJ)
    return tuple(fitnesses)

start = time.time()
nsga = NSGA2(dimensions=NDIM, population_size=MU, ea_runs=NGEN, evolutionary_algorithm=partial(ga, mutation_rate = MRPB, crossover_rate = CXPB, crossover_type = "multi", mutation_type = ""),
                ref_point=[1000,1000,1000])
nsga.run()
print(nsga.iteration_stats[-1])
end = time.time()
elapsed = end - start
print(elapsed)