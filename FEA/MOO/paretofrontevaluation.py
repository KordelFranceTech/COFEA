import itertools

from pymoo.factory import get_performance_indicator
from pymoo.algorithms.nsga2 import calc_crowding_distance
import numpy as np
import math
from operator import attrgetter, itemgetter


class ParetoOptimization:

    def __init__(self, obj_size=3):
        """
        @param obj_size: The number of objectives in the problem definition.
        """
        self.n_obj = obj_size
        self.approximate_pareto_front = []

    def evaluate_solution(self, approx_pareto_set, reference_point):
        """
        @param approx_pareto_set: The discovered pareto set to calculate different evaluations for.
                                -> format: List of PopulationMembers.
        @param reference_point: Which reference point (representing the worst possible solution) to use to calculate HV.
        """
        # reference_points = self.calculate_ref_points(h, self.obj_size)

        diversity = self.calculate_diversity(approx_pareto_set)

        updated_pareto_set = [np.array(sol.fitness) for sol in approx_pareto_set]
        hv = get_performance_indicator("hv",
                                       ref_point=np.array(reference_point))  # hypervolume(np.array(updated_pareto_set))
        return {'hypervolume': hv.calc(np.array(updated_pareto_set)), 'diversity': diversity}

    def calculate_diversity(self, approx_pareto_set, minmax=None):
        """
        Spread/delta indicator:
        estimate the extent of the spread of the obtained Pareto front,
        supposedly created by Deb? But I cant find a copy of his 2001 book

        Definition:
        Square root of the sum of the square of the difference between max and min solutions
        (determined based on overall fitness) for each objective.
        Alternate wording: the sum of the width for each objective.
        Adra and Fleming, 2000. Coello Coello, Dhaenens, and Jordan, 2010. Ischubishi and shibata, 2004, GECCO

        @param approx_pareto_set: generated approximate pareto front to be evaluated.
                                   -> format: List of PopulationMembers.
        @param minmax: Dictionary of the minimum and maximum value of each objective to perform normalization.
                       -> format: {i: [min, max]} where i is the objective index
        @return float: spread indicator
        """

        spread_indicator = 0
        for i in range(self.n_obj):
            try:
                to_sort = [x.objective_values[i] for x in approx_pareto_set]
            except AttributeError:
                to_sort = [x.fitness[i] for x in approx_pareto_set]
            if minmax:
                min_ = minmax[i][0]
                max_ = minmax[i][1]
            else:
                min_ = 0
                max_ = max(to_sort)
            sorted_set = [(x - min_) / (max_ - min_) for x in sorted(to_sort)]
            last = np.array(sorted_set[-1])
            first = np.array(sorted_set[0])
            spread_indicator += np.square(np.linalg.norm(last - first))
        return np.sqrt(spread_indicator)

    def calculate_ref_points(self, h: int):
        """A slightly worse point than the nadir point is usually used for hypervolume calculation in the EMO
        community.
         ...
         The basic idea is to specify the reference point so that a set of well-distributed solutions over the entire
         linear Pareto front has a large hypervolume and all solutions in such a solution set have similar
         hypervolume contributions.
         ...
         Our discussions and experimental results clearly show that a slightly worse point than the nadir point
         is not always appropriate for performance comparison of EMO algorithms.
         (Hisao Ishibuchi, Ryo Imada, Yu Setoguchi, Yusuke Nojima;
         How to Specify a Reference Point in Hypervolume Calculation for Fair Performance Comparison.
         Evol Comput 2018; 26 (3): 411–440.)

         OR

         use the solution with the worst found fitness values for each of the objectives: keep track of this while optimizing
         """
        ref_value = 1 + 1 / h
        ref_points = np.full(self.n_obj, ref_value)
        return ref_points
