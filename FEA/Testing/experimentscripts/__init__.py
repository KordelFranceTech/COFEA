# from pymoo.algorithms.moead import *
# from pymoo.factory import get_problem, get_reference_directions
# from pymoo.optimize import minimize
#
#
# problem = get_problem("dtlz2")
#
# ref_dirs = get_reference_directions("das-dennis", 3, n_partitions=12)
#
# algorithm = MOEAD(
#     ref_dirs,
#     n_neighbors=15,
#     prob_neighbor_mating=0.7,
#     seed=1,
#     verbose=False
# )
#
# res = minimize(problem, algorithm, termination=('n_gen', 10), verbose=False)
#
#
# print(algorithm.neighbors[2])
#
# print(res.F)