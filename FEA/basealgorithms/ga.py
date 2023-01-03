"""
Contains the GA class with the three basic methods: tournament_selection(population), mutation(original_individual)
and crossover(parent1, parent2).
These three methods combined form the generate_offspring() method.
Can be used a standalone algorithm using run(), or as the base-algorithm for MOO EA's or FEA's.
"""

import numpy as np
import random
from FEA.utilities.util import *

#TODO: seperate out mutation and crossover functions and create switch statement to improve code readability.


class GA:
    def __init__(self, dimensions=100, population_size=200, tournament_size=5, mutation_rate=0.1, crossover_rate=0.95,
                 ga_runs=100, mutation_type="scramble", crossover_type="multi", parent_selection = "tournament", offspring_size=100,
                 continuous_var_space=False, value_range=[0, 1], combinatorial_options=[], eta=20):
        """
        @param dimensions: Integer. Number of variables in a single individual.
        @param population_size: Integer. Number of individuals to form the population in a single generation.
        @param tournament_size: Integer. How many candidates to select a single parent from.
        @param mutation_rate: Float between 0.0 and 1.0. Chance of performing mutation on each individual.
        @param crossover_rate: Float between 0.0 and 1.0. Chance of performing crossover of the selected parent pair.
        @param ga_runs: Integer. How many generations to run the GA for.
        @param mutation_type: String 'swap', 'scramble', 'multi bitflip' or 'single bitflip'. Which type of mutation to perform.
        @param crossover_type: String 'single', 'uniform', or 'multi'. Which type of crossover to perform.
        @param offspring_size: Integer. How many children do you wish to create each generation.
        """
        self.name = 'genetic algorithm'
        self.population_size = population_size
        self.offspring_size = offspring_size
        self.dimensions = dimensions
        self.tournament_size = tournament_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.ga_runs = ga_runs
        self.mutation_type = mutation_type
        self.crossover_type = crossover_type
        self.parent_selection = parent_selection
        self.continuous_bool = continuous_var_space
        self.value_range = value_range
        self.combinatorial_values = combinatorial_options
        self.population = []
        self.eta = eta

    def initialize_population(self):
        for i in range(self.population_size):
            if self.continuous_bool:
                variables = [random.uniform(self.value_range[0], self.value_range[1]) for x in range(self.dimensions)]
            else:
                variables = random.choices(self.combinatorial_values, k=self.dimensions)
            self.population.append(PopulationMember(variables, self.calc_fitness(variables)))

    def calc_fitness(self, variables):
        """
        An empty function to fill in dynamically based on the problem.
        @param variables: List of variables to evaluate.
        @return: Tuple of fitness values. The number of fitness values depends on the number of objectives.
        """
        pass

    def tournament_selection(self, population):
        """
        Entities in population is of namedTuple format: PopulationMember('variables', 'fitness')
        Picks a number of solutions and finds and returns the solution dict with the best score.
        """
        i = 0
        chosen_solution = None
        curr_fitness = None
        idx = 0
        while i < self.tournament_size:
            rand = random.randint(0, len(population) - 1)
            temp = population[rand]
            if curr_fitness is not None:
                if temp.fitness <= curr_fitness:
                    chosen_solution = temp
                    curr_fitness = temp.fitness
                    idx = rand
            else:
                curr_fitness = temp.fitness
                chosen_solution = temp
                idx = rand
            i += 1
        return chosen_solution, idx
    
    def dominance_based_tournament(self, individuals, k):
        def tourn(ind1, ind2):
            if ind1.fitness.dominates(ind2.fitness):
                return ind1
            elif ind2.fitness.dominates(ind1.fitness):
                return ind2

            if ind1.fitness.crowding_dist < ind2.fitness.crowding_dist:
                return ind2
            elif ind1.fitness.crowding_dist > ind2.fitness.crowding_dist:
                return ind1

            if random.random() <= 0.5:
                return ind1
            return ind2

        individuals_1 = random.sample(individuals, len(individuals))
        individuals_2 = random.sample(individuals, len(individuals))

        chosen = []
        for i in range(0, k, 4):
            chosen.append(tourn(individuals_1[i],   individuals_1[i+1]))
            chosen.append(tourn(individuals_1[i+2], individuals_1[i+3]))
            chosen.append(tourn(individuals_2[i],   individuals_2[i+1]))
            chosen.append(tourn(individuals_2[i+2], individuals_2[i+3]))

    def mutate(self, original_solution, mutation_rate=None, eta=20, lbound=None, ubound=None):
        """
        @param original_solution: The solution to be mutated.
        @param mutation_rate: Ability to manually set mutation rate separate from GA object.
        Type = Swap, scramble or bitflip.
        Swap = Choose two indices and swap the values.
        Scramble = Choose two indices and randomly scramble all the values between these indices.
        Single Bitflip = Randomly choose a single index and flip the value.
        Multi Bitflip = For each variable decide whether the bit will be flipped using the mutation rate.
        @return: The mutated child or a copy of the original solution if mutation was not performed.
        """
        if not lbound:
            lbound = self.value_range[0]
        if not ubound:
            ubound = self.value_range[1]
        if not mutation_rate:
            mutation_rate = self.mutation_rate

        _solution = [x for x in original_solution]
        if self.continuous_bool:
            if self.mutation_type == "polynomial":
                for i, x in enumerate(_solution):
                    if random.random() < mutation_rate:
                        delta_1 = (x - lbound) / (ubound - lbound)
                        delta_2 = (ubound - x) / (ubound - lbound)
                        rand = random.random()
                        mut_pow = 1.0 / (eta + 1.)

                        if rand < 0.5:
                            xy = 1.0 - delta_1
                            val = 2.0 * rand + (1.0 - 2.0 * rand) * xy ** (eta + 1)
                            delta_q = val ** mut_pow - 1.0
                        else:
                            xy = 1.0 - delta_2
                            val = 2.0 * (1.0 - rand) + 2.0 * (rand - 0.5) * xy ** (eta + 1)
                            delta_q = 1.0 - val ** mut_pow
                        if isinstance(delta_q, complex):
                            delta_q = delta_q.real
                        x = x + delta_q * (ubound - lbound)
                        _solution[i] = min(max(x, lbound), ubound)
            else:
                for i in range(len(_solution)):
                    if random.random() < mutation_rate:
                        _solution[i] = random.uniform(self.value_range[0], self.value_range[1])
        else:
            if random.random() < mutation_rate:
                if self.mutation_type == "swap":
                    numbers = list(range(0, self.dimensions))
                    index_1 = random.choice(numbers)
                    numbers.remove(index_1)
                    index_2 = random.choice(numbers)
                    while original_solution[index_2] == _solution[index_1]:
                        index_2 = random.choice(numbers)
                    _solution[index_1] = original_solution[index_2]
                    _solution[index_2] = original_solution[index_1]

                elif self.mutation_type == "scramble":
                    index_1 = random.choice(list(range(0, self.dimensions-1)))
                    numbers = list(range(0, self.dimensions))  # numbers to choose next index from
                    numbers.remove(index_1)  # remove already chosen index from these numbers
                    while True:
                        index_2 = random.choice(numbers)
                        if abs(index_2-index_1) <= 2:
                            index_2 = random.choice(numbers)
                        else:
                            min_index, max_index = maxmin_indeces(index_1, index_2)
                            temp_list = original_solution[min_index:max_index]
                            if len(set(temp_list)) == 1:
                                continue
                            else:
                                break
                    while True:
                        np.random.shuffle(temp_list)
                        if temp_list == _solution[min_index:max_index]:
                            continue
                        else:
                            break
                    _solution[min_index:max_index] = temp_list

                elif self.mutation_type == "single bitflip":
                    numbers = list(range(0, self.dimensions))
                    index_1 = random.choice(numbers)
                    temp = _solution[index_1]
                    if temp == 0:
                        _solution[index_1] = 1
                    elif temp ==1:
                        _solution[index_1] = 0

                elif self.mutation_type == "multi bitflip":
                    for i, x in enumerate(_solution):
                        if random.random() < mutation_rate:
                            if x == 0:
                                _solution[i] = 1
                            elif x == 1:
                                _solution[i] = 0
                
                elif self.mutation_type == "grouping":
                    print(_solution)
                    # randomly select a gene to mutate
                    numbers = list(range(0, self.dimensions))
                    index_1 = random.choice(numbers)
                    # get variables in this gene's group
                    selected_group = [x for x in _solution[index_1]]
                    # remove gene from chromosome
                    del(_solution[index_1])
                    # create new groups
                    new_nr_groups = random.randint(1,len(selected_group))
                    random.shuffle(selected_group)
                    new_groups = [[selected_group[grp_idx]] for grp_idx in range(new_nr_groups)]
                    del(selected_group[:len(new_nr_groups)])
                    while len(selected_group) != 0:
                        for grp in new_groups:
                            if len(selected_group) != 0:
                                chosen_var_idx = random.randrange(0,len(selected_group))
                                grp.append(selected_group[chosen_var_idx])
                                del(selected_group[chosen_var_idx])
                            else:
                                break
                    for grp in new_groups:
                        _solution.append(grp)
                    print(_solution)

        return _solution

    def crossover(self, first_solution, second_solution, crossover_rate=None, eta=20, number_of_genes=0):
        """
        @param first_solution: The first parent selected.
        @param second_solution: The second parent selected.
        @param crossover_rate: Ability to manually set crossover rate separate from GA object.
        Crossover is performed between these parents to create offspring.
        Type = single, multi or uniform.
        Single = Single point crossover, a random index is selected and the parents are crossed based on this index.
        Multi = Multi-point crossover, two indices are selected and the values between these indices are swapped between
                the parents.
        Uniform = For each variable, swap variable value between parents if crossover rate is met.
        Blend boolean = Crossover for continuous variable space: take the average of the two parents' variable values.
        This can be applied to any of the three above methods.
        @return: The crossed over children, or an empty array if crossover was not performed.
        """
        if self.continuous_bool:
            lbound = self.value_range[0]
            ubound = self.value_range[1]
        if not crossover_rate:
            crossover_rate = self.crossover_rate

        if random.random() < crossover_rate:
            _first_solution = [x for x in first_solution]
            _second_solution = [x for x in second_solution]
            if self.crossover_type == 'multi':
                index_1 = random.randint(1, self.dimensions - 1)
                index_2 = random.randint(1, self.dimensions - 1)
                while abs(index_2 - index_1) <= 2:
                    index_1 = random.randint(1, self.dimensions - 1)
                    index_2 = random.randint(1, self.dimensions - 1)
                min_index, max_index = maxmin_indeces(index_1, index_2)
                if self.continuous_bool:
                    third_solution = [min(max((x + y)/2, lbound), ubound) for (x, y) in zip(first_solution, second_solution)]
                    _first_solution[min_index:max_index] = third_solution[min_index:max_index]
                    _second_solution[0:min_index] = third_solution[0:min_index]
                    _second_solution[max_index:] = third_solution[max_index:]
                else:
                    _first_solution[min_index:max_index] = second_solution[min_index:max_index]
                    _second_solution[min_index:max_index] = first_solution[min_index:max_index]
            elif self.crossover_type == 'single':
                index_1 = random.randint(1, self.dimensions - 1)
                if self.continuous_bool:
                    third_solution = [min(max((x + y)/2, lbound), ubound) for (x, y) in zip(first_solution, second_solution)]
                    _first_solution[0:index_1] = third_solution[0:index_1]
                    _second_solution[index_1:] = third_solution[index_1:]
                else:
                    _first_solution[0:index_1] = second_solution[0:index_1]
                    _second_solution[index_1:] = first_solution[index_1:]
            elif self.crossover_type == 'uniform':
                for i in range(len(first_solution)):
                    if random.random() < crossover_rate:
                        if self.continuous_bool:
                            third_solution = [min(max((x + y)/2, lbound), ubound) for (x, y) in zip(first_solution, second_solution)]
                            _first_solution[i] = third_solution[i]
                            _second_solution[i] = third_solution[i]
                        else:
                            _first_solution[i] = second_solution[i]
                            _second_solution[i] = first_solution[i]
            elif self.crossover_type == 'simulated binary':
                for i, (x1, x2) in enumerate(zip(first_solution, second_solution)):
                    rand = random.random()
                    if rand <= 0.5:
                        beta = 2. * rand
                    else:
                        beta = 1. / (2. * (1. - rand))
                    beta **= 1. / (eta + 1.)
                    _first_solution[i] = 0.5 * (((1 + beta) * x1) + ((1 - beta) * x2))
                    _second_solution[i] = 0.5 * (((1 - beta) * x1) + ((1 + beta) * x2))
            elif self.crossover_type == 'grouping':
                index_1 = random.randint(1, number_of_genes - 1)
                index_2 = random.randint(1, number_of_genes - 1)
                while abs(index_2 - index_1) <= 2:
                    index_1 = random.randint(1, number_of_genes - 1)
                    index_2 = random.randint(1, number_of_genes - 1)
                min_index, max_index = maxmin_indeces(index_1, index_2)
                _first_solution[min_index:max_index] = second_solution[min_index:max_index]
                _second_solution[min_index:max_index] = first_solution[min_index:max_index]
                idx_range = np.arange(min_index, max_index)
                first_sol_vars = []
                second_sol_vars = []
                for idx in idx_range:
                    first_sol_vars.extend(_first_solution[idx])
                    second_sol_vars.extend(_second_solution[idx])
                duplicate_vars_first = []
                duplicate_vars_second = []
                for idx, gene in enumerate(_first_solution[:min_index]):
                    print(gene)
                    print(first_sol_vars)
                    if any(gene) in first_sol_vars:
                        print('duplicate')
                        duplicate_vars_first.extend(gene)
                        del(_first_solution[idx])
                        print(duplicate_vars_first)
                    if any(_second_solution[idx]) in second_sol_vars:
                        duplicate_vars_second.extend(_second_solution[idx])
                        del(_second_solution[idx])
                for idx, gene in enumerate(_first_solution[max_index:]):
                    if any(gene) in first_sol_vars:
                        print('duplicate')
                        duplicate_vars_first.extend(gene)
                        del(_first_solution[idx])
                        print(duplicate_vars_first)
                for idx, gene in enumerate(_second_solution[max_index:]):
                    if any(gene) in second_sol_vars:
                        print('duplicate')
                        duplicate_vars_second.extend(gene)
                        del(_second_solution[idx])
                        print(duplicate_vars_second)
                _first_solution.append([x for x in duplicate_vars_first if x not in first_sol_vars])
                _second_solution.append([x for x in duplicate_vars_second if x not in second_sol_vars])
                _first_solution.append([x for x in range(0, self.dimensions) if x not in _first_solution.flatten()])
                _second_solution.append([x for x in range(0, self.dimensions) if x not in _second_solution.flatten()])

            return [_first_solution, _second_solution]
        else:
            return []

    def create_offspring(self, curr_population=None):
        """
        @param curr_population: If population is being generated by MOEA, adjust this population
        @return: The children created from the current population.
        Performs the necessary steps to create the set number of offspring.
        1. Tournament selection to select parents
        2. Crossover of the selected parents
        3. Mutation of the resulting offspring
        These steps are repeated until the length of the children set is equal to offspring_size.
        """
        children = []
        if curr_population:
            self.population = [x for x in curr_population]
        if len(self.population[0].variables) <= 6 and self.crossover_type == 'multi':
            self.crossover_type = 'single'
        offspring_iter = 0
        if self.parent_selection == "grouping":
            random.shuffle(self.population)
        while len(children) < self.offspring_size:
            if len(self.population[0].variables) == 1:
                first_solution, idx1 = self.tournament_selection(self.population)
                child1 = self.mutate([x for x in first_solution.variables], mutation_rate = 0.99)
                children.append(child1)
            else:
                if self.parent_selection == "grouping":
                    first_solution = self.population[offspring_iter]
                    offspring_iter += 1
                    second_solution = self.population[offspring_iter]
                    offspring_iter += 1
                else:
                    first_solution, idx1 = self.tournament_selection(self.population)
                    self.population.pop(idx1)
                    second_solution, idx2 = self.tournament_selection(self.population)
                    self.population.insert(idx1, first_solution)
                first_solution = [x for x in first_solution.variables]
                second_solution = [x for x in second_solution.variables]
                crossed_over = self.crossover(first_solution, second_solution)
                if crossed_over:
                    child1 = crossed_over[0]
                    child2 = crossed_over[1]
                else:
                    child1 = [x for x in first_solution]
                    child2 = [x for x in second_solution]
                child1 = self.mutate(child1)
                child2 = self.mutate(child2)
                children.append(child1)
                children.append(child2)         
        return children

    def selection(self, total_population):
        """
        Select the new population from the total population
        """
        # if self.crossover == "grouping":
        #     pass
        # else:
        fitnesses = []
        for sol in total_population:
            fitnesses.append(sol.fitness)
        new_population = [x for i, y, x in enumerate(sorted(zip(fitnesses, total_population))) if i < self.population_size]
        return new_population

    def run(self, generations):
        """
        run a basic GA for i generations
        """
        self.initialize_population()
        i=0
        while i < generations:
            children = self.create_offspring()
            offspring = []
            for child in children:
                offspring.append(PopulationMember(child, self.calc_fitness(child)))
            total_population = self.population + offspring
            self.population = self.selection(total_population)
