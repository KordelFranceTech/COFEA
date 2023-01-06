import gc
from utilities.util import PopulationMember

from MOO.paretofrontevaluation import ParetoOptimization
from FEA.factorarchitecture import FactorArchitecture
from pymoo.util.nds.non_dominated_sorting import find_non_dominated
from pymoo.algorithms.nsga2 import calc_crowding_distance

import numpy as np
import random, pickle, re, os


class MOFEA:
    """
    Multi-Objective Factored Evolutionary Algorithm.
    Adjusts compete and share steps to create a set of non-dominated solutions, called the archive, which createsan approximate Pareto front.
    """
    def __init__(self, fea_iterations, dimensions, factor_architecture=None, base_alg=None,
                 combinatorial_options=None, value_range=[0, 1], ref_point=[1, 1, 1]):
        self.combinatorial_options = combinatorial_options
        self.value_range = value_range
        self.dim = dimensions
        self.nondom_archive = []
        self.population = []
        self.factor_architecture = factor_architecture
        self.base_algorithm = base_alg
        self.fea_runs = fea_iterations
        self.current_iteration = 0
        self.global_solutions = []
        self.worst_fitness_ref = ref_point
        self.po = ParetoOptimization(obj_size=len(ref_point))
        self.subpopulations = None
        self.iteration_stats = []

        current_working_dir = os.getcwd()
        path = re.search(r'^(.*?[\\/]FEA)', current_working_dir)
        self.path = path.group()

    def initialize_moo_subpopulations(self, factors=None):
        """
        Initialize subpopulations based on factor architecture.
        @param factors: Ability to send through factors if they were not available at initialization time.
        """
        if factors:
            self.factor_architecture = FactorArchitecture(factors=factors)
            self.factor_architecture.get_factor_topology_elements()
        if self.combinatorial_options:
            random_global_variables = random.choices(self.combinatorial_options, k=self.dim)
        else:
            random_global_variables = [random.randrange(self.value_range[0], self.value_range[1]) for x in
                                       range(self.dim)]
        objs = self.base_algorithm(dimensions=self.dim).calc_fitness(random_global_variables)
        random_global_solution = PopulationMember(random_global_variables, objs)
        self.global_solutions.append(random_global_solution)
        return [self.base_algorithm(dimensions=len(factor),
                                    combinatorial_values=self.combinatorial_options, value_range=self.value_range,
                                    factor=factor, global_solution=random_global_solution) for factor in
                self.factor_architecture.factors]

    def run(self):
        '''
        For each subpopulation:
            Optimize along all objectives
            Apply non-dominated sorting strategy
            Include diversity measure individuals
        Compete (based on non-domination of solution, i.e. overall fitness)
        -> or ignore compete and create set of solutions based on overlapping variables: improves diversity? Check this with diversity measure
        -> spread different non-domination solutions across different subpopulations, i.e., different subpopulations have different global solutions: this should also improve diversity along the PF?
        @return: eval_dict {HV, diversity, ND_size, FEA_run}
        '''
        self.subpopulations = self.initialize_moo_subpopulations()
        change_in_nondom_size = []
        old_archive_length = 0
        fea_run = 0
        while fea_run != self.fea_runs:  # len(change_in_nondom_size) < 4 and
            for s, alg in enumerate(self.subpopulations):
                #print('Subpopulation: ', s, alg.dimensions, type(alg))
                alg.run(fea_run=fea_run)
            self.compete()
            self.share_solution()
            if len(self.nondom_archive) == old_archive_length:
                change_in_nondom_size.append(True)
            else:
                change_in_nondom_size = []
            # print("last nondom solution: ", self.nondom_archive[-1].fitness)
            old_archive_length = len(self.nondom_archive)
            eval_dict = self.po.evaluate_solution(self.nondom_archive, self.worst_fitness_ref)
            eval_dict['FEA_run'] = fea_run
            eval_dict['ND_size'] = len(self.nondom_archive)
            self.iteration_stats.append(eval_dict)
            print('fitnesses: ', self.nondom_archive[0].fitness)
            print("eval dict", eval_dict)
            # [print(s.objective_values) for s in self.nondom_archive]
            # [print(i, ': ', s.objective_values) for i,s in enumerate(self.iteration_stats[fea_run+1]['global solutions'])]
            fea_run = fea_run+1
            #file = open(self.path+"/results/checkpoints/latest_run.p", "wb")
            #pickle.dump(self, file)
            #file.close()

    def compete(self):
        """
        For each variable:
            - gather subpopulations with said variable
            - randomly select non-dominated solution from subpopulation
            - add randomly selected solution to temporary archive
            - replace variable value in global solution with corresponding subpop value and add to temporary archive
        Once this has been done for all variables and subpopulations: re-evaluate temporary archive for non-dominance to get new archive.
        """
        new_solutions = []
        seen_populations = set()
        for var_idx in range(self.dim):
            # randomly pick one of the global solutions to perform competition for this variable
            chosen_global_solution = random.choice(self.global_solutions)
            vars = [x for x in chosen_global_solution.variables]
            if not new_solutions:
                new_solutions.append(vars)
            # for each population with said variable perform competition on this single randomly chosen global solution
            if len(self.factor_architecture.optimizers[var_idx]) > 1:
                for pop_idx in self.factor_architecture.optimizers[var_idx]:
                    curr_pop = self.subpopulations[pop_idx]
                    if pop_idx not in seen_populations:
                        seen_populations.add(pop_idx)
                        new_solutions.append([x for x in curr_pop.random_nondom_solutions[-1]])
                    pop_var_idx = np.where(np.array(curr_pop.factor) == var_idx)
                    # pick one of the nondominated solutions from this population based on sorting criterium or randomly if no nondom solutions
                    if len(curr_pop.nondom_archive) == 0:
                        random_sol = random.choice(curr_pop.curr_population)
                    elif len(curr_pop.nondom_archive) < 3:
                        random_sol = random.choice(curr_pop.nondom_archive)
                    else:
                        sorted = curr_pop.sorting_mechanism(curr_pop.nondom_archive)
                        random_sol = sorted[0]
                    var_candidate_value = random_sol.variables[pop_var_idx[0][0]]
                    vars[var_idx] = var_candidate_value
                    if vars not in new_solutions:
                        new_solutions.append(vars)
            elif len(self.factor_architecture.optimizers[var_idx]) == 1:
                curr_pop = self.subpopulations[self.factor_architecture.optimizers[var_idx][0]]
                pop_var_idx = np.where(np.array(curr_pop.factor) == var_idx)
                if len(curr_pop.nondom_archive) != 0:
                    for solution in curr_pop.nondom_archive:
                        var_candidate_value = solution.variables[pop_var_idx[0][0]]
                        vars[var_idx] = var_candidate_value
                        if vars not in new_solutions:
                            new_solutions.append(vars)
        # Recalculate fitnesses for new solutions
        new_solutions = [PopulationMember(vars, self.base_algorithm(dimensions=self.dim).calc_fitness(vars)) for vars in
                         new_solutions]
        # Reassert non-dominance
        nondom_indeces = find_non_dominated(np.array([np.array(x.fitness) for x in new_solutions]))
        # Assign current iteration of global solutions based on non-dominance
        self.global_solutions = [new_solutions[i] for i in nondom_indeces]
        # Extend non-dom archive with found non-dom solutions
        self.nondom_archive.extend(self.global_solutions)
        self.nondom_archive = self.update_archive()

    def share_solution(self):
        """
        Share non-dominated solutions from the archive created in the compete step.
        Each subpopulation is randomly assigned a non-dominated solution 
        (without replacement until all solutions have been assigned to at least one subpop).
        """
        if len(self.global_solutions) != 0:
            to_pick = [s for s in self.global_solutions]
        else:
            to_pick = [s for s in self.nondom_archive]
        for i, alg in enumerate(self.subpopulations):
            if len(to_pick) > 1:
                gs = random.choice(to_pick)
                to_pick.remove(gs)
            elif len(to_pick) == 1:
                gs = to_pick[0]
                # repopulate the list to restart if not at the last subpopulation
                if i < len(self.subpopulations) - 1:
                    if len(self.global_solutions) != 0:
                        to_pick = [s for s in self.global_solutions]
                    else:
                        to_pick = [s for s in self.nondom_archive]
            else:
                print('there are no elements in the global solutions list.')
                raise IndexError
            # update fitnesses
            alg.global_solution = gs
            alg.curr_population = [PopulationMember(p.variables,
                                                    self.base_algorithm().calc_fitness(p.variables, alg.global_solution,
                                                                                       alg.factor)) for p in
                                   alg.curr_population]
            # set best solution and replace worst solution with global solution across FEA
            alg.replace_worst_solution(self.global_solutions)

    # TODO: include a diversity measure to reduce archive size when it goes above a certain size (similar to SPEA2 but more general)
    def update_archive(self, nd_archive=None, reduce_archive=False):
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
        #ADD-ON to reduce non-dom archive size
        if reduce_archive and len(nondom_archive) > 1000:
            fitnesses = np.array([np.array(x.fitness) for x in population])
            distances = calc_crowding_distance(fitnesses)
            sorted_arch = [x for y, x in sorted(zip(distances, population))]
            nondom_archive = sorted_arch[:self.population_size]
        return nondom_archive
