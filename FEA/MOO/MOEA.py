"""
Contains different algorithm implementations of single population multi-objective optimization algorithms.
MOOEA class -- Superclass for all Multi-objective Evolutionary Algorithms.
NSGA2 class -- Inherits from MOOEA. Non-Dominated Sorting Genetic Algorithm II (Deb, et al.)
SPEA2 class -- Strength Pareto Evolutionary Algorithm 2
MOEAD class -- Multi-Objective Evolutionary Algorithm with Decomposition
"""
import math

from pymoo.util.dominator import Dominator
from scipy.spatial.distance import cdist

from MOO.paretofrontevaluation import *
from basealgorithms.ga import GA
from utilities.util import PopulationMember, compare_solutions, euclidean_distance

from pymoo.algorithms.nsga2 import calc_crowding_distance
from pymoo.util.nds.non_dominated_sorting import find_non_dominated
from pymoo.util.nds.fast_non_dominated_sort import fast_non_dominated_sort

import numpy as np
import random


class MOEA:
    def __init__(self, evolutionary_algorithm=GA, ea_runs=100, population_size=100, dimensions=10,
                 combinatorial_options=None, value_range=None, reference_point=None,
                 factor=None, global_solution=None,
                 crossover_rate=.9, crossover_type='',
                 mutation_rate=.1, mutation_type=''):
        """
        Superclass for all MOEAs.
        @param evolutionary_algorithm: base algorithm to use: genetic algorithm (GA) (default), PSO
        @param ea_runs: how many runs:generations/iterations of the algorithm
        @param population_size: Integer. Number of individuals to form the population in a single generation.
        @param dimensions:  Integer. Number of variables in a single individual. When no combinatorial_options are
                defined, these will be floating point numbers between 0.0 and 1.0]
        @param combinatorial_options: List of values, E.g. [0,1]. Set of values to create solutions from when dealing
                with combinatorial optimization.
        @param value_range: List of two values: [min, max] for continuous optimization, variables must fall within this range.
        @param reference_point: worst point in the objective space to calculate hypervolume.

        FEA specific, i.e., when MOEA is used within FEA context.
        @param factor: variables belonging to the factor represented by MOEA instance
        @param global_solution: complete solution representing all variables which the factor variables are plugged into to evaluate the full solution.

        TODO: pass through as extra params using kwargs
        NEXT ARE GA SPECIFIC, since all MOEAS have been using GA's, but this should be changed to it can accept any keyword args for any base alg
        @param crossover_rate: probability to perform crossover
        crossover_type: what kind of crossover to perform
        mutation_rate: probability to perform mutation
        mutation_type: what kind of mutation to perform
        Contains methods initialize_population(gs, factor) and calc_fitness(variables, gs, factor).
        Where 'gs'(=global solution) and 'factor' are only used when using the MOEA's as a base-algorithm for FEA.
        """
        self.combinatorial_values = combinatorial_options
        if value_range is not None:
            self.value_range = value_range
        else:
            self.value_range = [0.0, 1.0]
        self.population_size = population_size
        self.dimensions = dimensions
        self.factor = factor
        self.global_solution = global_solution
        self.ea_runs = ea_runs
        self.iteration_stats = []
        self.nondom_archive = []
        if reference_point is not None:
            self.worst_fitness_ref = reference_point
        else:
            self.worst_fitness_ref = [1, 1, 1]
        if self.combinatorial_values is not None:
            self.crossover_type = 'uniform'
            self.mutation_type = 'multi bitflip'
        else:
            self.crossover_type = 'multi'
            self.mutation_type = 'polynomial'
        if crossover_type:
            self.crossover_type = crossover_type
        if mutation_type:
            self.mutation_type = mutation_type

        #TODO: send through kwargs instead of GA specific params
        if self.combinatorial_values is not None:
            self.ea = evolutionary_algorithm(dimensions=dimensions, population_size=population_size, tournament_size=2,
                                             offspring_size=population_size,
                                             combinatorial_options=self.combinatorial_values,
                                             crossover_type=self.crossover_type, crossover_rate=crossover_rate,
                                             mutation_type=self.mutation_type, mutation_rate=mutation_rate)
        else:
            self.ea = evolutionary_algorithm(dimensions=dimensions, population_size=population_size, tournament_size=2,
                                             offspring_size=population_size, continuous_var_space=True,
                                             value_range=self.value_range,
                                             mutation_type=self.mutation_type, mutation_rate=mutation_rate,
                                             crossover_type=self.crossover_type, crossover_rate=crossover_rate)

    def initialize_population(self, gs=None, factor=None):
        """
        @param gs: Global Solution. Only used for FEA based-algorithms. When implementing any MOOEA as the base-algorithm
                for FEA, each algorithm instance will only be optimizing a subpopulation consisting of a subset of the
                variable space. But to evaluate the fitness of the individuals, a global solution is necessary to plug
                in the subset of variables.
        @param factor: Only used for FEA base-algorithms. The factor defines which subset of variables a specific
                subpopulation will be looking at.
        @return: The initial population: A list of PopulationMember(variables=[dimensions], fitness=())) instances.
        """
        i = 0
        curr_population = []
        initial_solution = []
        while i < self.population_size:
            if self.combinatorial_values:
                new_solution = random.choices(self.combinatorial_values,
                                              k=self.dimensions)  # Prescription(field=field, index=i, gs=self.global_solution, factor=self.factor)
            else:
                new_solution = [random.uniform(self.value_range[0], self.value_range[1]) for x in
                                range(self.dimensions)]
                # new_solution = [random.random() for x in range(self.dimensions)]
            curr_population.append(
                PopulationMember(new_solution, self.calc_fitness(variables=new_solution, gs=gs, factor=factor)))
            if i == 0:
                initial_solution = new_solution
            i = i + 1
        return curr_population, initial_solution

    def set_iteration_stats(self, iteration_idx, nd_archive=None):
        '''
        Calculate generation statistics for the found non-dominated solutions
        '''
        if nd_archive is None:
            nd_archive = self.nondom_archive
        po = ParetoOptimization()
        eval_dict = po.evaluate_solution(nd_archive, self.worst_fitness_ref)
        eval_dict['GA_run'] = iteration_idx
        eval_dict['ND_size'] = len(nd_archive)
        self.iteration_stats.append(eval_dict)

    def calc_fitness(self, variables, gs=None, factor=None):
        """
        An empty function to fill in dynamically based on the problem.
        @param variables: List of variables to evaluate.
        @param gs: Global Solution. Only used for FEA base-algorithms. When implementing any MOOEA as the base-algorithm
                for FEA, each algorithm instance will only be optimizing a subpopulation consisting of a subset of the
                variable space. But to evaluate the fitness of the individuals, a global solution is necessary to plug
                in the subset of variables.
        @param factor: Only used for FEA base-algorithms. The factor defines which subset of variables a specific
                subpopulation will be looking at.
        @return: Tuple of fitness values. The number of fitness values depends on the number of objectives.
        """
        pass


class NSGA2(MOEA):
    def __init__(self, evolutionary_algorithm=GA, dimensions=100, population_size=500, ea_runs=100,
                 combinatorial_values=None, value_range=None, reference_point=None,
                 factor=None, global_solution=None):
        """

        @param evolutionary_algorithm: The base algorithm to use. Currently only Genetic Algorithm (GA) is implemented.
        @param dimensions: Integer. Number of variables in a single individual.
        @param population_size: Integer. Number of individuals to form the population in a single generation.
        @param ea_runs: Integer. How many generations to run NSGA2 for.
        @param combinatorial_values: List of values, E.g. [0,1]. Set of values to create solutions from when dealing
                with combinatorial optimization. If empty list [], floating point numbers between 0.0 and 1.0 are used.
        @param factor:
        @param global_solution: Global Solution. List of variables at full dimensionality.
                Only used for FEA base-algorithms when implementing any MOOEA as the base-algorithm for FEA.
                To evaluate the fitness of the individuals, a global solution is necessary to plug
                in the subset of variables.
        @param ref_point: List of worst potential fitness values. Has as many fitness values as there are objectives.
                Used to calculate the Hypervolume indicator.
        """
        super().__init__(evolutionary_algorithm=evolutionary_algorithm, ea_runs=ea_runs,
                         population_size=population_size, reference_point=reference_point,
                         dimensions=dimensions, combinatorial_options=combinatorial_values, value_range=value_range,
                         factor=factor, global_solution=global_solution)
        self.curr_population = []
        self.initial_solution = []
        self.worst_index = None
        self.random_nondom_solutions = []
        self.nondom_pop = []

    def run(self, fea_run=0):
        """
        Run the full algorithm for the set number of 'ea_runs' or until a convergence criterion is met.
        @param fea_run: Which generation the FEA is on, if NSGA2 is being used as the base-algorithm.
        """
        if fea_run == 0:
            self.curr_population, self.initial_solution = self.initialize_population(gs=self.global_solution,
                                                                                     factor=self.factor)
        i = 1
        change_in_nondom_size = []
        old_archive_length = 0
        '''
        Convergence criterion based on change in non-dominated solution set size
        '''
        while i != self.ea_runs:  # and len(change_in_nondom_size) < 10:
            children = self.ea.create_offspring(self.curr_population)
            self.curr_population.extend(
                [PopulationMember(c, self.calc_fitness(c, self.global_solution, self.factor)) for c in children])
            self.select_new_generation(i)
            '''
            Only run this part of the algorithm in the single-population case.
            I.e. when NSGA2 is NOT used as the base-algorithm for FEA.
            '''
            if self.factor is None:
                # self.nondom_archive = self.update_archive()
                self.set_iteration_stats(i, self.nondom_pop)
                # if len(self.nondom_archive) == old_archive_length and len(self.nondom_archive) >= 10:
                #     change_in_nondom_size.append(True)
                # else:
                #     change_in_nondom_size = []
                # old_archive_length = len(self.nondom_archive)

            i += 1

    def replace_worst_solution(self, gs):
        """

        @param gs: List of Global Solution(s).
                Used to replace the worst solution(s) with.
        Finds the worst solution based on fitness values and replaces with global solution.
        There can be multiple "worst" solutions, in which case as many are replaced as there are global solutions.
        """
        if self.worst_index is None or self.worst_index == []:
            diverse_sort = self.sorting_mechanism(self.curr_population)
            self.worst_index = [i for i, x in enumerate(self.curr_population) if x.fitness == diverse_sort[-1].fitness]
        if len(gs) > len(self.worst_index):
            for i, idx in enumerate(self.worst_index):
                self.curr_population[idx] = PopulationMember(variables=[gs[i].variables[j] for j in self.factor],
                                                             fitness=gs[i].fitness)
        elif len(self.worst_index) > len(gs):
            for i, sol in enumerate(gs):
                self.curr_population[self.worst_index[i]] = PopulationMember(
                    variables=[sol.variables[j] for j in self.factor], fitness=sol.fitness)
        elif len(self.worst_index) == len(gs):
            for idx, sol in zip(self.worst_index, gs):
                self.curr_population[idx] = PopulationMember(variables=[sol.variables[j] for j in self.factor],
                                                             fitness=sol.fitness)

    def update_archive(self, nd_archive=None):
        """
        Function to update the existing archive object, or a chosen archive
        @param nd_archive: The archive to be updated, if None, this is the algorithm's current generation archive
        @return: updated non-dominated archive as list of PopulationMembers
        """
        if nd_archive is None:
            nd_archive = self.nondom_archive
        nondom_indeces = find_non_dominated(np.array([np.array(x.fitness) for x in nd_archive]))
        old_nondom_archive = [nd_archive[i] for i in nondom_indeces]
        seen = set()
        nondom_archive = []
        for s in old_nondom_archive:
            if s.fitness not in seen:
                seen.add(s.fitness)
                nondom_archive.append(s)
        return nondom_archive

    def sorting_mechanism(self, population):
        """
        NSGA2 specific operation.
        Sorts solutions based on the crowding distance to maximize diversity.
        @param population: List of individuals to be sorted.
        @return: List of sorted individuals.
        """
        fitnesses = np.array([np.array(x.fitness) for x in population])
        distances = calc_crowding_distance(fitnesses)
        return [x for y, x in sorted(zip(distances, population))]

    def select_new_generation(self, generation_idx=0):
        """
        After the base-algorithm (e.g. GA) has created the offspring:
        1. check for non-domination
        2. sort based on crowding distance
        3. select next generation based on sorted population
        @param generation_idx: Which generation is the algorithm on.
                If it is on the last generation, calculate which individual has the worst fitness.
        @return: shuffled new population
        """
        total_population = [x for x in self.curr_population]
        fitnesses = np.array([np.array(x.fitness) for x in total_population])
        nondom_indeces = find_non_dominated(fitnesses)
        self.nondom_pop = [total_population[i] for i in nondom_indeces]
        # self.nondom_archive.extend(nondom_pop)
        # self.update_archive(self.nondom_archive)
        # Less non-dominated solutions than population size
        if len(self.nondom_pop) < self.population_size:
            new_population = []
            fronts = fast_non_dominated_sort(fitnesses)
            last_front = []
            n_ranked = 0
            for front in fronts:
                # increment the n_ranked solution counter
                n_ranked += len(front)
                # stop if more solutions are n_ranked than the population size
                if n_ranked >= self.population_size:
                    last_front = front
                    break
                new_population.extend([total_population[i] for i in front])

            # add however many solutions necessary to fill out the population based on crowding distance
            sorted_population = self.sorting_mechanism([total_population[i] for i in last_front])
            length_to_add = self.population_size - len(new_population)
            new_population.extend(sorted_population[:length_to_add])
            self.curr_population = new_population
            worst_fitness = tuple([x for x in self.curr_population[-1].fitness])
        # More non-dominated solutions than the population size
        # Select the first n based on crowding distance
        else:
            sorted_population = self.sorting_mechanism(self.nondom_pop)
            self.curr_population = sorted_population[:self.population_size]
            worst_fitness = tuple([x for x in self.curr_population[-1].fitness])
        random.shuffle(self.curr_population)

        # Last generation if used by FEA or CCEA
        if generation_idx == self.ea_runs - 1 and self.factor is None:
            self.nondom_archive = self.nondom_pop
        if generation_idx == self.ea_runs - 1 and self.factor is not None:
            self.nondom_archive = self.nondom_pop
            self.update_archive(self.nondom_archive)
            # randomly select a non-dom solution to add to FEA
            choice = random.choice(self.nondom_archive)
            full_solution = [x for x in self.global_solution.variables]
            for i, x in zip(self.factor, choice.variables):
                full_solution[i] = x
            self.random_nondom_solutions.append(full_solution)
            self.worst_index = [i for i, x in enumerate(self.curr_population) if
                                x.fitness == worst_fitness]  # np.where(np.array(self.curr_population, dtype=object) == worst)


class SPEA2(MOEA):
    def __init__(self, evolutionary_algorithm=GA, dimensions=100, population_size=200, ea_runs=100,
                 combinatorial_values=None, value_range=None, reference_point=None,
                 factor=None, global_solution=None,
                 archive_size=200):
        super().__init__(evolutionary_algorithm=evolutionary_algorithm, ea_runs=ea_runs,
                         population_size=population_size, reference_point=reference_point,
                         dimensions=dimensions, combinatorial_options=combinatorial_values, value_range=value_range,
                         factor=factor, global_solution=global_solution)
        self.curr_population = []
        self.initial_solution = []
        self.archive_size = archive_size
        self.strength_pop = None
        self.distance_matrix = None
        self.final_strengths = None
        self.worst_index = None
        self.random_nondom_solutions = []
        self.k_to_find = round(math.sqrt(self.population_size + self.archive_size))

    def run(self, fea_run=0):
        """
        Run the full algorithm for the set number of 'ea_runs' or until a convergence criterion is met.
        @param fea_run: Which generation the FEA is on.
        """
        # initialize population
        if fea_run == 0:
            self.curr_population, self.initial_solution = self.initialize_population(gs=self.global_solution,
                                                                                     factor=self.factor)
        i = 1

        change_in_nondom_size = []
        old_archive_length = 0
        '''
        Stop based on criterion
        '''
        while i <= self.ea_runs:  # and len(change_in_nondom_size) < 10:
            # calculate strength value fitness for entire population
            self.strength_pop = self.sorting_mechanism(self.curr_population, self.nondom_archive)
            # set non-dominated archive keeping the original FITNESS score
            self.nondom_archive = [y for x, y in zip(self.strength_pop, self.curr_population) if x.fitness < 1.0]
            # update archive to have exact length set by algorithm parameter
            self.nondom_archive = self.update_archive()
            # find worst index
            if self.factor is not None:
                self.worst_index = self.final_strengths.argmax()

            # create new generation
            children = self.ea.create_offspring(self.strength_pop)
            self.curr_population = [PopulationMember(c, self.calc_fitness(c, self.global_solution, self.factor)) for c
                                    in children]
            if self.factor is None:
                self.set_iteration_stats(i)
            i += 1

        if self.factor is not None:
            choice = random.choice(self.nondom_archive)
            full_solution = [x for x in self.global_solution.variables]
            for j, x in zip(self.factor, choice.variables):
                full_solution[j] = x
            self.random_nondom_solutions.append(full_solution)

    def replace_worst_solution(self, gs):
        """

        @param gs: List of Global Solution(s).
                Used to replace the worst solution(s) with.
        Finds the worst solution based on fitness values and replaces with global solution.
        There can be multiple "worst" solutions, in which case as many are replaces as there are global solutions.
        """
        if self.worst_index is None or self.worst_index == []:
            self.worst_index = self.final_strengths.index(max(self.final_strengths))
        else:
            sol = random.choice(gs)
            self.curr_population[self.worst_index] = PopulationMember(
                variables=[sol.variables[j] for j in self.factor], fitness=sol.fitness)

    def update_archive(self, nd_archive=None, strength_pop=None):
        '''
        Archive update function
        If length is shorter, archive is filled.
        If it is longer, archive is truncated.
        '''
        if nd_archive is None:
            # No archive being sent though, so update own archive
            nd_archive = self.nondom_archive
        if strength_pop is None:
            strength_pop = self.strength_pop
        if len(nd_archive) < self.archive_size:
            remaining = [y for x, y in sorted(zip(strength_pop, self.curr_population), key=lambda pair: pair[0].fitness)
                         if x.fitness >= 1.0]
            nd_archive.extend(remaining[:(self.archive_size - len(nd_archive))])
        elif len(nd_archive) > self.archive_size:
            distance_matrix = self.calculate_distance_matrix(np.array([np.array(x.fitness) for x in nd_archive]))
            while len(nd_archive) > self.archive_size:
                most_crowded_idx = self.find_most_crowded(distance_matrix)
                self.remove_point_from_dist_mtx(distance_matrix, most_crowded_idx)
                del nd_archive[most_crowded_idx]
        return nd_archive

    def sorting_mechanism(self, population, nondom_archive=None):
        """
        Code based on Platypus implementation:
        https://platypus.readthedocs.io/en/latest/_modules/platypus/algorithms.html
        calculates pareto strength value and returns new list of population members with variables and strengths.
        K = sqrt(pop size + archive size)
        Original population is not altered.
        """
        if nondom_archive:
            total_population = [x for x in population]
            total_population.extend(nondom_archive)
            fitnesses = np.array([np.array(x.fitness) for x in total_population])
        else:
            fitnesses = np.array([np.array(x.fitness) for x in population])

        if len(population) == self.population_size:
            self.distance_matrix = self.calculate_distance_matrix(fitnesses)
            temp_distance_matrix = self.distance_matrix
        else:
            temp_distance_matrix = self.calculate_distance_matrix(fitnesses)
        domination_matrix = Dominator().calc_domination_matrix(fitnesses)
        # the number of solutions each individual dominates
        S = (domination_matrix == 1).sum(axis=1)

        # the raw fitness of each solution = sum of the dominance counts in S of all solutions it is dominated by
        raw_strength = ((domination_matrix == -1) * S).sum(axis=1)

        # add density to fitness
        final_strengths = []
        for i in range(len(population)):
            final_strengths.append(raw_strength[i] + (1.0 / (self.kth_distance(i, self.k_to_find,
                                                                               temp_distance_matrix) + 2.0)))  # 1 stands for k=1, to find kth nearest neighbor
        if len(population) == self.population_size:
            self.final_strengths = np.array(final_strengths)
        # key is used to force sort only on strengths when there are still duplicate strength values
        return [PopulationMember(x.variables, fs) for fs, x in
                sorted(zip(final_strengths, population), key=lambda pair: pair[0])]

    def calculate_distance_matrix(self, fitnesses, distance_fun=euclidean_distance):
        """
        Code from Platypus library!
        Maintains the pairwise distances between solutions.  
        It also provides convenient routines to lookup the distance between any two solutions, 
        find the most crowded solution, and remove a solution.
        """
        distances = []
        for i in range(len(fitnesses)):
            distances_i = [(j, distance_fun(fitnesses[i], fitnesses[j])) for j in range(len(fitnesses)) if i != j]
            distances.append(sorted(distances_i, key=lambda x: x[1]))
        return distances

    def find_most_crowded(self, distance_matrix):
        """
        Code from Platypus library!
        Finds the most crowded solution.
        
        Returns the index of the most crowded solution, which is the solution
        with the smallest distance to the nearest neighbor.  Any ties are
        broken by looking at the next distant neighbor.
        """
        minimum_distance = np.inf
        minimum_index = -1

        for i in range(len(distance_matrix)):
            distances_i = distance_matrix[i]

            if distances_i[0][1] < minimum_distance:
                minimum_distance = distances_i[0][1]
                minimum_index = i
            elif distances_i[0][1] == minimum_distance:
                for j in range(len(distances_i)):
                    dist1 = distances_i[j][1]
                    dist2 = distance_matrix[minimum_index][j][1]

                    if dist1 < dist2:
                        minimum_index = i
                        break
                    if dist2 < dist1:
                        break

        return minimum_index

    def remove_point_from_dist_mtx(self, distance_matrix, index):
        """
        Code based on Platypus library!
        Removes the distance entries for the given solution.
        
        Parameters
        ----------
        index : int
            The index of the solution
        """
        del distance_matrix[index]
        for i in range(len(distance_matrix)):
            distance_matrix[i] = [(x if x < index else x - 1, y) for (x, y) in distance_matrix[i] if x != index]

    def kth_distance(self, i, k, distance_matrix):
        """
        Code based on  Platypus library!
        Returns the distance to the k-th nearest neighbor.
        
        Parameters
        ----------
        i : int
            The index of the solution
        k : int
            Finds the k-th nearest neightbor distance
            K = sqrt(pop size + archive size)
        """
        return distance_matrix[i][k][1]


class MOEAD(MOEA):
    """
    Qingfu Zhang and Hui Li. 
    A multi-objective evolutionary algorithm based on decomposition. 
    IEEE Transactions on Evolutionary Computation, 2007.

    Based on PYMOO implementation, and uses their decomposition methods: PBI, Tchebicheff, weighted_sum
    https://github.com/anyoptimization/pymoo/blob/db63995689e446c343ca0ccb05cac8c682fcb98d/pymoo/algorithms/moo/moead.py
    """

    def __init__(self, evolutionary_algorithm=GA, dimensions=100, ea_runs=100,
                 combinatorial_values=None, value_range=None, reference_point=None,
                 factor=None, global_solution=None,
                 problem_decomposition=None, n_neighbors=10, weight_vector=None, prob_neighbor_mating=0.9):
        super().__init__(evolutionary_algorithm=evolutionary_algorithm, ea_runs=ea_runs,
                         population_size=len(weight_vector), reference_point=reference_point,
                         crossover_rate=1.0, crossover_type='simulated binary',
                         dimensions=dimensions, combinatorial_options=combinatorial_values, value_range=value_range,
                         factor=factor, global_solution=global_solution)
        self.curr_population = []
        self.initial_solution = []
        self.problem_decomposition = problem_decomposition
        self.prob_neighbor_mating = prob_neighbor_mating
        self.n_obj = 3
        self.decomposed_archive_fitnesses = []
        self.archive_size = 250
        self.decomposed_fitnesses = np.zeros(len(weight_vector))
        self.weight_vector = weight_vector
        self.ideal_point_fitness = None
        self.neighbors = np.argsort(cdist(self.weight_vector, self.weight_vector), axis=1, kind='quicksort')[:,
                         :n_neighbors]
        self.random_nondom_solutions = []

    def run(self, fea_run=0):
        """
        for each population member:
            Get parents from neighborhood selection
            Perform offspring generation
            Recalculate fitness and set temp "ideal point"
            Perform replacement of neighboring solutions if decomposed fitness is better
            Decomposition here is using scalar weights, the weights are used to calculate the decomposed fitness;
            the weights are selected based on the chosen neighbors
            (neighborhood includes itself so this replacement will also replace the "self" if its better)

            NOTE: the weight vectors serve as reference directions, guiding the solutions through the objective space
            by assigning different weights to different parts of the objective space
        """
        # initialize population and "ideal point" approximation
        # check fea-run, since we do not want to start over if we are running fea iterations
        if fea_run == 0:
            self.curr_population, self.initial_solution = self.initialize_population(gs=self.global_solution,
                                                                                     factor=self.factor)
            self.ideal_point_fitness = np.min([x.fitness for x in self.curr_population], axis=0)

        for i in range(self.ea_runs):
            for N in np.random.permutation(len(self.curr_population)):
                # create offspring
                offspring = self.generate_offspring_from_neighborhood(N)
                if any(offspring.fitness) < 0:
                    print(offspring.fitness)

                # update ideal point
                self.ideal_point_fitness = np.min(np.vstack([self.ideal_point_fitness, offspring.fitness]), axis=0)

                # update neighborhood
                neighborhood_to_check = self.neighbors[N]
                neighborhood_fitnesses = np.array(
                    [np.array(self.curr_population[idx].fitness) for idx in neighborhood_to_check])
                decomposed_og_fitness = self.problem_decomposition.do(neighborhood_fitnesses,
                                                                      weights=self.weight_vector[neighborhood_to_check,
                                                                              :],
                                                                      ideal_point=self.ideal_point_fitness)
                decomposed_offspring_fitness = self.problem_decomposition.do(np.array(offspring.fitness),
                                                                             weights=self.weight_vector[
                                                                                     neighborhood_to_check, :],
                                                                             ideal_point=self.ideal_point_fitness)
                improved_indeces = np.where(decomposed_offspring_fitness < decomposed_og_fitness)[0]
                for idx in improved_indeces:
                    self.curr_population[neighborhood_to_check[idx]] = offspring

                # Save decomposed fitness for ordering
                if N in improved_indeces:
                    self.decomposed_fitnesses[N] = decomposed_offspring_fitness[0]
                else:
                    self.decomposed_fitnesses[N] = decomposed_og_fitness[0]

                # update archive
                self.update_archive(offspring=offspring, decomposed_fitness=decomposed_offspring_fitness[0])

            if self.factor is None:
                self.set_iteration_stats(i)

        if self.factor is not None:
            choice = random.choice(self.nondom_archive)
            full_solution = [x for x in self.global_solution.variables]
            for j, x in zip(self.factor, choice.variables):
                full_solution[j] = x
            self.random_nondom_solutions.append(full_solution)

    def replace_worst_solution(self, gs):
        """
        @param gs: global solution to replace "worst" solution with.
        When algorithm is part of FEA, we want to keep track of the "worst" solution according to algorithmic method.
        In this case, we check the decomposed fitness for each population member (and its own corresponding weight vector).
        """
        sol = random.choice(gs)
        worst_idx = np.argmax(self.decomposed_fitnesses)
        self.curr_population[worst_idx] = PopulationMember(
            variables=[sol.variables[j] for j in self.factor], fitness=sol.fitness)

    def update_archive(self, nd_archive=None, offspring=None, decomposed_fitness=None):
        """
        Function to update the existing archive object, or a chosen archive
        @param nd_archive: The archive to be updated, if None, this is the algorithm's current generation archive
        @return: updated non-dominated archive as list of PopulationMembers
        """
        if nd_archive is None:
            nd_archive = self.nondom_archive
        nd_archive.append(offspring)
        self.decomposed_archive_fitnesses.append(decomposed_fitness)
        if len(nd_archive) > 1:
            if self.archive_size > len(nd_archive):
                nondom_indeces = find_non_dominated(np.array([np.array(x.fitness) for x in nd_archive]))
                old_nondom_archive = [nd_archive[i] for i in nondom_indeces]
                old_decomposed_archive_fitnesses = [self.decomposed_archive_fitnesses[i] for i in nondom_indeces]
            else:
                sorted_ = self.sorting_mechanism(nd_archive)
                self.decomposed_archive_fitnesses.sort()
                nondom_indeces = find_non_dominated(
                    np.array([np.array(x.fitness) for x in sorted_[:self.archive_size]]))
                old_nondom_archive = [sorted_[i] for i in nondom_indeces]
                old_decomposed_archive_fitnesses = [self.decomposed_archive_fitnesses[i] for i in nondom_indeces]
            seen = set()
            nondom_archive = []
            decomposed_fitnesses = []
            for i, s in enumerate(old_nondom_archive):
                if s.fitness not in seen:
                    seen.add(s.fitness)
                    nondom_archive.append(s)
                    decomposed_fitnesses.append(old_decomposed_archive_fitnesses[i])
            self.nondom_archive = nondom_archive
            self.decomposed_archive_fitnesses = decomposed_fitnesses

    def sorting_mechanism(self, population, decomposed_fitnesses=None):
        """
        Sorts solutions based on fitness.
        For MOEA/D this is only used by FEA to sort the archive and to replace the worst solution.
        @param population: List of individuals to be sorted.
        @param decomposed_fitnesses: decompoed values to sort on
        @return: List of sorted individuals based on their decomposed fitness values.
        """
        if decomposed_fitnesses is None:
            decomposed_fitnesses = self.decomposed_archive_fitnesses
        return [x for y, x in sorted(zip(decomposed_fitnesses, population), key=lambda pair: pair[0])]

    def generate_offspring_from_neighborhood(self, N):
        """
        @param N: current individual being processed by algorithm
        Select two parents from current population member N's neighborhood and perform:
        1. simulated binary crossover
        2. Polynomial mutation
        Return resulting offspring
        """
        # select parents randomly from neighborhood
        parents_idx = np.random.choice(self.neighbors[N], 2, replace=False)
        parents = [[x for x in self.curr_population[parents_idx[0]].variables],
                   [x for x in self.curr_population[parents_idx[1]].variables]]
        # create offspring using regular EA methods
        if self.dimensions == 1:
            child = self.ea.mutate(parents[0], lbound=self.value_range[0],
                                   ubound=self.value_range[1])
        else:
            child = self.ea.mutate(self.ea.crossover(parents[0], parents[1])[0], lbound=self.value_range[0],
                                   ubound=self.value_range[1])
        child = [min(max(var, self.value_range[0]), self.value_range[1]) for var in child]
        return PopulationMember(child, self.calc_fitness(child, gs=self.global_solution, factor=self.factor))


class HYPE(MOEA):
    def __init__(self, evolutionary_algorithm, dimensions=100, population_size=200, ea_runs=100,
                 # data_distribution=False,
                 combinatorial_values=[], factor=None, global_solution=None):
        super().__init__()
        self.dimensions = dimensions
        self.population_size = population_size
        self.ea = evolutionary_algorithm(dimensions, population_size)
        self.curr_population, self.initial_solution = self.initialize_population(gs=global_solution, factor=factor)
        self.factor = factor
        self.global_solution = global_solution
        self.ea_runs = ea_runs
