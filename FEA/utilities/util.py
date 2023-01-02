from collections import namedtuple
from functools import wraps
import math
import string
from functools import partial
import numpy as np
from collections import namedtuple
import numba

PopulationMember = namedtuple('PopulationMember', ['variables', 'fitness'])


def maxmin_indeces(idx1, idx2):
    if idx1 > idx2:
        max_index = idx1
        min_index = idx2
    else:
        max_index = idx2
        min_index = idx1
    return min_index, max_index


@numba.jit(nopython=True)
def euclidean_distance(a, b):
    return math.sqrt(np.sum((a - b) ** 2))


def compare_solutions(solution1, solution2):
    dominate1 = False
    dominate2 = False
    if isinstance(solution1, PopulationMember):
        n_objs = len(solution1.fitness)
        fitness1 = solution1.fitness
        fitness2 = solution2.fitness
    else:
        n_objs = len(solution1)
        fitness1 = solution1
        fitness2 = solution2

    for i in range(n_objs):
        o1 = fitness1[i]
        o2 = fitness2[i]

        if o1 < o2:
            dominate1 = True
                
            if dominate2:
                return 0
        elif o1 > o2:
            dominate2 = True
            
            if dominate1:
                return 0
        
    if dominate1 == dominate2:
        return 0
    elif dominate1:
        return -1
    else:
        return 1


def add_method(cls):
    def decorator(func):
        @wraps(func) 
        def wrapper(self, *args, **kwargs): 
            return func(*args, **kwargs)
        setattr(cls, func.__name__, wrapper)
        # Note we are not binding func, but wrapper which accepts self but does exactly the same as func
        return func # returning func means func can still be used normally
    return decorator


def memo(f):
    cache = {}

    @wraps(f)
    def wrap(*arg):
        if arg not in cache:
            cache['arg'] = f(*arg)
            return cache['arg']
    return wrap

# def project():
#     return partial(
#         pyproj.transform,
#         pyproj.Proj('+init=EPSG:26912', preserve_units=True),  # 26912 , 32612
#         pyproj.Proj('+init=EPSG:4326'))  # 4326


# def project_to_meters(x, y):
#     inProj = pyproj.Proj(init='epsg:4326')
#     outProj = pyproj.Proj(init='epsg:3857')
#     xp, yp = pyproj.transform(inProj, outProj, x, y)
#     return xp, yp


def is_hex(s):
    hex_digits = set(string.hexdigits)
    return all(c in hex_digits for c in s)


def calculate_statistics(run, curr_population):
    """
    Calculate statistics across all prescription solutions for the current run.
    Param:
        run -- generation index
    Returns:
        Dictionary containing statistics:
            overall fitness, jump score, stratification score, variance, worst and best score.
    """
    keys = ["run", "overall", "jumps", "strat", "fertilizer", "variance", "min_score", "max_score"]
    stat_values = []
    scores = [_solution.overall_fitness for _solution in curr_population]
    stat_values.append(run)
    stat_values.append(np.mean(scores))
    stat_values.append(np.mean([_solution.jumps for _solution in curr_population]))
    stat_values.append(np.mean([_solution.strat for _solution in curr_population]))
    stat_values.append(np.mean([_solution.fertilizer_rate for _solution in curr_population]))
    stat_values.append(np.var(scores))
    stat_values.append(min(scores))
    stat_values.append(max(scores))
    stats_dictionary = dict(zip(keys, stat_values))
    return stats_dictionary